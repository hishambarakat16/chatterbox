import time
from uuid import uuid4

import torch

from ..models.s3tokenizer import drop_invalid_tokens
from ..models.t3.inference.alignment_stream_analyzer import AlignmentStreamAnalyzer
from ..models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
from ..runtime.session import apply_exaggeration, clone_conditionals
from ..mtl_tts import SUPPORTED_LANGUAGES
from .worker import ChatterboxMultilingualStreamingWorker
from ..vllm_t3_bridge import (
    build_vllm_prompt,
    build_vllm_uncond_prompt,
    make_sampling_params,
    make_cfg_pair_sampling_params,
    prepare_vllm_text_tokens,
)


def _ensure_alignment_eager_attention(t3) -> None:
    cfg = getattr(getattr(t3, "tfmr", None), "config", None)
    if cfg is None:
        return
    try:
        setattr(cfg, "_attn_implementation", "eager")
    except Exception:
        pass
    try:
        setattr(cfg, "attn_implementation", "eager")
    except Exception:
        pass


def _is_forced_eos(logits: torch.Tensor, eos_idx: int) -> bool:
    if logits.ndim != 2 or logits.shape[0] == 0:
        return False
    row = logits[0]
    if eos_idx < 0 or eos_idx >= row.shape[0]:
        return False
    eos_logit = float(row[eos_idx].item())
    other = row.clone()
    other[eos_idx] = float("-inf")
    max_other = float(other.max().item())
    return eos_logit >= float((2**15) - 1) and max_other <= float(-(2**15) + 1)


def _sample_from_logits(logits: torch.Tensor, options) -> int:
    """Sample one token from logits using options (temperature, top_p, min_p)."""
    temperature = float(getattr(options, "temperature", 0.0)) if options is not None else 0.0
    top_p = float(getattr(options, "top_p", 1.0)) if options is not None else 1.0
    min_p = float(getattr(options, "min_p", 0.0)) if options is not None else 0.0

    lp = (logits[0] if logits.ndim > 1 else logits).float()

    if temperature <= 1e-6:
        return int(lp.argmax().item())

    lp = lp / temperature

    if min_p > 0.0:
        probs = lp.softmax(-1)
        lp = lp.masked_fill(probs < min_p * float(probs.max().item()), float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(lp, descending=True)
        cumulative_probs = sorted_logits.softmax(-1).cumsum(-1)
        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        lp[sorted_indices[remove]] = float("-inf")

    return int(torch.multinomial(lp.softmax(-1), num_samples=1).item())


def _replay_original_alignment_stop(
    *,
    t3,
    t3_cond,
    text_tokens: torch.Tensor,
    token_ids: list[int],
    cfg_weight: float,
    options=None,
) -> tuple[list[int], dict[str, float | str]]:
    diagnostics: dict[str, float | str] = {
        "replay_enabled": 0.0,
        "replay_error": "",
        "replay_forced_eos": 0.0,
        "replay_trimmed": 0.0,
        "replay_trim_index": -1.0,
        "generated_tokens_before": float(len(token_ids)),
        "generated_tokens_after": float(len(token_ids)),
    }

    if not token_ids:
        return token_ids, diagnostics

    stop_token_id = int(t3.hp.stop_speech_token)
    analyzer = None

    try:
        _ensure_alignment_eager_attention(t3)

        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=t3.device)
        if text_tokens.shape[0] == 1:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        if int((text_tokens == int(t3.hp.start_text_token)).sum().item()) < int(text_tokens.shape[0]):
            raise ValueError("Missing start_text_token in replay text_tokens.")
        if int((text_tokens == int(t3.hp.stop_text_token)).sum().item()) < int(text_tokens.shape[0]):
            raise ValueError("Missing stop_text_token in replay text_tokens.")

        initial_speech_tokens = int(t3.hp.start_speech_token) * torch.ones_like(text_tokens[:, :1])
        embeds, len_cond = t3.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
            cfg_weight=float(cfg_weight),
        )

        analyzer = AlignmentStreamAnalyzer(
            t3.tfmr,
            None,
            text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
            alignment_layer_idx=9,
            eos_idx=stop_token_id,
        )

        patched_model = T3HuggingfaceBackend(
            config=t3.cfg,
            llama=t3.tfmr,
            speech_enc=t3.speech_emb,
            speech_head=t3.speech_head,
            alignment_stream_analyzer=analyzer,
        )

        bos_token = torch.tensor([[int(t3.hp.start_speech_token)]], dtype=torch.long, device=t3.device)
        bos_embed = t3.speech_emb(bos_token)
        bos_embed = bos_embed + t3.speech_pos_emb.get_fixed_embedding(0)
        bos_embed = bos_embed.expand(embeds.shape[0], -1, -1)

        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)
        output = patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = output.past_key_values

        generated_ids = [int(t3.hp.start_speech_token)]
        trim_index = None

        # When cfg_weight > 0 we have a cond+uncond batch and can apply the CFG
        # formula to get proper guidance.  In that mode we generate our OWN token
        # sequence from the CFG-adjusted logits (the vLLM token_ids are a fallback
        # if something goes wrong).  When cfg_weight == 0 we simply replay the
        # vLLM tokens through the analyzer to find the trim point.
        use_cfg_generation = cfg_weight > 0.0 and options is not None
        max_new_tokens = int(getattr(options, "max_new_tokens", len(token_ids))) if use_cfg_generation else len(token_ids)
        diagnostics["cfg_generation"] = 1.0 if use_cfg_generation else 0.0

        for idx in range(max_new_tokens):
            logits_step = output.logits[:, -1, :]
            cond_logits = logits_step[0:1, :]
            if logits_step.shape[0] > 1:
                uncond_logits = logits_step[1:2, :]
                cfg = torch.as_tensor(float(cfg_weight), device=cond_logits.device, dtype=cond_logits.dtype)
                logits = cond_logits + cfg * (cond_logits - uncond_logits)
            else:
                logits = cond_logits

            last_token = generated_ids[-1] if generated_ids else int(t3.hp.start_speech_token)
            logits = analyzer.step(logits, next_token=last_token)
            if _is_forced_eos(logits, stop_token_id):
                diagnostics["replay_forced_eos"] = 1.0
                diagnostics["replay_trimmed"] = 1.0
                diagnostics["replay_trim_index"] = float(idx)
                trim_index = idx
                break

            if use_cfg_generation:
                token_id = _sample_from_logits(logits, options)
            else:
                if idx >= len(token_ids):
                    break
                token_id = int(token_ids[idx])

            generated_ids.append(token_id)
            if token_id == stop_token_id:
                diagnostics["replay_trim_index"] = float(idx + 1)
                trim_index = idx + 1
                break

            next_token = torch.tensor([[token_id]], dtype=torch.long, device=t3.device)
            next_token_embed = t3.speech_emb(next_token)
            next_token_embed = next_token_embed + t3.speech_pos_emb.get_fixed_embedding(idx + 1)
            next_token_embed = next_token_embed.expand(embeds.shape[0], -1, -1)

            output = patched_model(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = output.past_key_values

        diagnostics["replay_enabled"] = 1.0

        # Build final token list
        if use_cfg_generation:
            result_ids = generated_ids[1:]  # strip BOS
            if trim_index is not None:
                result_ids = result_ids[:trim_index]
        else:
            result_ids = list(token_ids)
            if trim_index is not None and trim_index < len(result_ids):
                result_ids = result_ids[:trim_index]

        diagnostics["generated_tokens_after"] = float(len(result_ids))
        return result_ids, diagnostics

    except Exception as exc:  # noqa: BLE001
        diagnostics["replay_error"] = repr(exc)
        diagnostics["generated_tokens_after"] = float(len(token_ids))
        return token_ids, diagnostics

    finally:
        if analyzer is not None and hasattr(analyzer, "close"):
            analyzer.close()


def _replay_many_with_cfg(
    *,
    t3,
    prepared_list: list[dict],
) -> list[tuple[list[int], dict]]:
    """
    Batched CFG replay for N concurrent requests.

    Runs a single joint HF T3 generation loop over all N requests (each with
    its own cond+uncond CFG pair), so the per-step cost is O(1) GPU calls
    instead of O(N).  Falls back to sequential if initial embed shapes differ
    (e.g. mixed-length texts) to avoid padding artefacts.

    Returns a list of (token_ids, diagnostics) tuples, one per request.
    """
    N = len(prepared_list)
    if N == 0:
        return []
    if N == 1:
        req = prepared_list[0]
        result = _replay_original_alignment_stop(
            t3=t3,
            t3_cond=req["active_conds"].t3,
            text_tokens=req["text_tokens"],
            token_ids=list(req["_vllm_token_ids"]),
            cfg_weight=float(req["active_options"].cfg_weight),
            options=req["active_options"],
        )
        return [result]

    stop_token_id = int(t3.hp.stop_speech_token)
    _ensure_alignment_eager_attention(t3)

    # ------------------------------------------------------------------ #
    # Build per-request initial embeds                                     #
    # ------------------------------------------------------------------ #
    all_embeds: list[torch.Tensor] = []
    all_text_tokens: list[torch.Tensor] = []
    all_len_cond: list[int] = []

    for req in prepared_list:
        text_tokens = torch.atleast_2d(req["text_tokens"]).to(dtype=torch.long, device=t3.device)
        if text_tokens.shape[0] == 1:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
        initial_speech = int(t3.hp.start_speech_token) * torch.ones_like(text_tokens[:, :1])
        embeds, len_cond = t3.prepare_input_embeds(
            t3_cond=req["active_conds"].t3,
            text_tokens=text_tokens,
            speech_tokens=initial_speech,
            cfg_weight=float(req["active_options"].cfg_weight),
        )
        all_embeds.append(embeds)       # (2, L_i, dim)
        all_text_tokens.append(text_tokens)
        all_len_cond.append(len_cond)

    # Fall back to sequential if prompt lengths differ
    shapes = [e.shape for e in all_embeds]
    if len(set(shapes)) > 1:
        results = []
        for i, req in enumerate(prepared_list):
            r = _replay_original_alignment_stop(
                t3=t3,
                t3_cond=req["active_conds"].t3,
                text_tokens=req["text_tokens"],
                token_ids=list(req["_vllm_token_ids"]),
                cfg_weight=float(req["active_options"].cfg_weight),
                options=req["active_options"],
            )
            results.append(r)
        return results

    # ------------------------------------------------------------------ #
    # Stack into (2N, L, dim) and set up N analyzers                      #
    # ------------------------------------------------------------------ #
    combined_embeds = torch.cat(all_embeds, dim=0)          # (2N, L, dim)
    bos_token = torch.tensor([[int(t3.hp.start_speech_token)]], dtype=torch.long, device=t3.device)
    bos_embed = t3.speech_emb(bos_token)
    bos_embed = bos_embed + t3.speech_pos_emb.get_fixed_embedding(0)
    bos_embed = bos_embed.expand(2 * N, -1, -1)             # (2N, 1, dim)
    inputs_embeds = torch.cat([combined_embeds, bos_embed], dim=1)  # (2N, L+1, dim)

    analyzers: list[AlignmentStreamAnalyzer] = []
    try:
        for i, (req, text_tokens) in enumerate(zip(prepared_list, all_text_tokens)):
            lc = all_len_cond[i]
            analyzer = AlignmentStreamAnalyzer(
                t3.tfmr,
                None,
                text_tokens_slice=(lc, lc + text_tokens.size(-1)),
                alignment_layer_idx=9,
                eos_idx=stop_token_id,
                batch_row=2 * i,
            )
            analyzers.append(analyzer)

        patched_model = T3HuggingfaceBackend(
            config=t3.cfg,
            llama=t3.tfmr,
            speech_enc=t3.speech_emb,
            speech_head=t3.speech_head,
        )

        # ------------------------------------------------------------------ #
        # Initial forward (full prompt + BOS)                                 #
        # ------------------------------------------------------------------ #
        output = patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = output.past_key_values

        # Per-request state
        generated_ids: list[list[int]] = [[int(t3.hp.start_speech_token)] for _ in range(N)]
        done = [False] * N
        trim_indices: list[int | None] = [None] * N
        forced_eos_flags = [False] * N
        max_tokens_per_req = [
            int(getattr(req["active_options"], "max_new_tokens", len(req["_vllm_token_ids"])))
            for req in prepared_list
        ]

        # ------------------------------------------------------------------ #
        # Autoregressive loop                                                 #
        # ------------------------------------------------------------------ #
        for step in range(max(max_tokens_per_req)):
            if all(done):
                break

            logits_all = output.logits[:, -1, :]  # (2N, vocab)

            next_embeds_list: list[torch.Tensor] = []
            for i, req in enumerate(prepared_list):
                if done[i]:
                    # Pad with a dummy embed (won't affect other requests' KV slices)
                    dummy = torch.zeros(1, 1, combined_embeds.shape[-1], device=t3.device, dtype=combined_embeds.dtype)
                    next_embeds_list.append(dummy)
                    continue

                cond_logits = logits_all[2 * i: 2 * i + 1, :]
                uncond_logits = logits_all[2 * i + 1: 2 * i + 2, :]
                cfg_weight = float(req["active_options"].cfg_weight)
                if cfg_weight > 0.0:
                    cfg = torch.as_tensor(cfg_weight, device=cond_logits.device, dtype=cond_logits.dtype)
                    logits_i = cond_logits + cfg * (cond_logits - uncond_logits)
                else:
                    logits_i = cond_logits

                last_token = generated_ids[i][-1] if generated_ids[i] else int(t3.hp.start_speech_token)
                logits_i = analyzers[i].step(logits_i, next_token=last_token)

                if _is_forced_eos(logits_i, stop_token_id):
                    forced_eos_flags[i] = True
                    trim_indices[i] = step
                    done[i] = True
                    dummy = torch.zeros(1, 1, combined_embeds.shape[-1], device=t3.device, dtype=combined_embeds.dtype)
                    next_embeds_list.append(dummy)
                    continue

                if step >= max_tokens_per_req[i]:
                    done[i] = True
                    dummy = torch.zeros(1, 1, combined_embeds.shape[-1], device=t3.device, dtype=combined_embeds.dtype)
                    next_embeds_list.append(dummy)
                    continue

                token_id = _sample_from_logits(logits_i, req["active_options"])
                generated_ids[i].append(token_id)

                if token_id == stop_token_id:
                    trim_indices[i] = len(generated_ids[i]) - 1
                    done[i] = True
                    dummy = torch.zeros(1, 1, combined_embeds.shape[-1], device=t3.device, dtype=combined_embeds.dtype)
                    next_embeds_list.append(dummy)
                    continue

                tok_t = torch.tensor([[token_id]], dtype=torch.long, device=t3.device)
                tok_emb = t3.speech_emb(tok_t)
                tok_emb = tok_emb + t3.speech_pos_emb.get_fixed_embedding(step + 1)
                next_embeds_list.append(tok_emb)  # (1, 1, dim)

            # (2N, 1, dim) — each request contributes 2 rows (cond + uncond share same token)
            next_step_embeds_per_req = [
                e.expand(2, -1, -1) for e in next_embeds_list
            ]  # list of N tensors, each (2, 1, dim)
            next_step_embeds = torch.cat(next_step_embeds_per_req, dim=0)  # (2N, 1, dim)

            output = patched_model(
                inputs_embeds=next_step_embeds,
                past_key_values=past,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = output.past_key_values

        # ------------------------------------------------------------------ #
        # Build results                                                        #
        # ------------------------------------------------------------------ #
        results = []
        for i, req in enumerate(prepared_list):
            raw_token_ids = list(req["_vllm_token_ids"])
            ids = generated_ids[i][1:]  # strip BOS
            trim_idx = trim_indices[i]
            if trim_idx is not None:
                ids = ids[:trim_idx]
            diag: dict[str, float | str] = {
                "replay_enabled": 1.0,
                "replay_error": "",
                "replay_forced_eos": 1.0 if forced_eos_flags[i] else 0.0,
                "replay_trimmed": 1.0 if trim_idx is not None else 0.0,
                "replay_trim_index": float(trim_idx) if trim_idx is not None else -1.0,
                "generated_tokens_before": float(len(raw_token_ids)),
                "generated_tokens_after": float(len(ids)),
                "cfg_generation": 1.0,
                "batched_replay": 1.0,
                "batch_size": float(N),
            }
            results.append((ids, diag))
        return results

    except Exception as exc:  # noqa: BLE001
        # Fall back to sequential on any error
        results = []
        for req in prepared_list:
            try:
                r = _replay_original_alignment_stop(
                    t3=t3,
                    t3_cond=req["active_conds"].t3,
                    text_tokens=req["text_tokens"],
                    token_ids=list(req["_vllm_token_ids"]),
                    cfg_weight=float(req["active_options"].cfg_weight),
                    options=req["active_options"],
                )
            except Exception:  # noqa: BLE001
                r = (list(req["_vllm_token_ids"]), {"replay_error": repr(exc), "replay_enabled": 0.0, "generated_tokens_before": float(len(req["_vllm_token_ids"])), "generated_tokens_after": float(len(req["_vllm_token_ids"]))})
            results.append(r)
        return results

    finally:
        for a in analyzers:
            if hasattr(a, "close"):
                a.close()


class ChatterboxMultilingualVllmWorker(ChatterboxMultilingualStreamingWorker):
    """
    Experimental vLLM T3 worker.

    Design choices for the current spike:
    - keep session creation and turbo S3 local
    - build token ids and conditioning payloads locally
    - let the served vLLM model reconstruct the T3 prompt internally
    - Hydra is intentionally disabled
    - CFG is implemented via the alignment replay pass (HF T3 with cond+uncond batch)
    """

    def __init__(
        self,
        *args,
        vllm_engine,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vllm_engine = vllm_engine

    def _prepare_request(self, *, session, text: str, options=None, request_start: float | None = None) -> dict:
        if request_start is None:
            request_start = time.perf_counter()
        profile = dict(getattr(session, "profile", {}) or {})
        profile.update(self._default_s3_profile())
        profile.update(getattr(session, "profile", {}) or {})
        profile.update(
            {
                "text_prep_s": 0.0,
                "t3_first_token_s": 0.0,
                "t3_wait_s": 0.0,
                "t3_active_s": 0.0,
                "t3_s": 0.0,
                "s3_s": 0.0,
                "audio_ready_s": 0.0,
                "watermark_s": 0.0,
                "t3_alignment_replay_s": 0.0,
                "t3_alignment_replay_enabled": 0.0,
                "t3_alignment_replay_forced_eos": 0.0,
                "t3_alignment_trimmed": 0.0,
                "t3_alignment_trim_index": -1.0,
                "t3_alignment_empty_fallback": 0.0,
            }
        )

        active_options = session.options if options is None else session.options.merged(**options.__dict__)
        language_id = active_options.language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        prep_start = time.perf_counter()
        active_conds = clone_conditionals(session.conditionals)
        active_conds = apply_exaggeration(active_conds, active_options.exaggeration, self.device)

        text_tokens = prepare_vllm_text_tokens(
            tokenizer=self.tokenizer,
            text=text,
            language_id=language_id,
            device="cpu",
        )
        prompt, prompt_meta = build_vllm_prompt(
            t3_cond=active_conds.t3,
            text_tokens=text_tokens,
            return_metadata=True,
        )
        profile["text_prep_s"] = time.perf_counter() - prep_start
        profile["t3_text_token_len"] = float(prompt_meta["text_token_len"])
        profile["t3_prompt_speech_token_len"] = float(prompt_meta["prompt_speech_token_len"])
        profile["t3_initial_speech_len"] = float(prompt_meta["initial_speech_len"])
        profile["t3_cond_seq_len"] = float(prompt_meta["cond_seq_len"])
        profile["t3_prompt_seq_len"] = float(prompt_meta["prompt_seq_len"])
        profile["t3_prompt_hidden_size"] = float(prompt_meta["prompt_hidden_size"])
        profile["t3_prompt_token_len_before_mm"] = float(prompt_meta["prompt_token_len_before_mm"])

        if float(active_options.cfg_weight) != 0.0:
            profile["t3_cfg_requested"] = float(active_options.cfg_weight)
        profile["t3_cfg_supported"] = 0.0
        profile["t3_hydra_supported"] = 0.0
        profile["t3_engine_vllm"] = 1.0

        sampling_params = make_sampling_params(
            options=active_options,
            hp=self.t3.hp,
        )
        return {
            "request_start": request_start,
            "profile": profile,
            "active_conds": active_conds,
            "active_options": active_options,
            "text_tokens": text_tokens,
            "prompt": prompt,
            "sampling_params": sampling_params,
        }

    def inspect_prompt_embed(self, *, session, text: str, options=None) -> dict:
        prepared = self._prepare_request(session=session, text=text, options=options)
        profile = prepared["profile"]
        sampling_params = prepared["sampling_params"]
        return {
            "session_id": session.session_id,
            "text": text,
            "text_chars": int(len(text)),
            "text_words": int(len(text.split())),
            "t3_text_token_len": int(profile.get("t3_text_token_len", 0.0)),
            "t3_prompt_speech_token_len": int(profile.get("t3_prompt_speech_token_len", 0.0)),
            "t3_initial_speech_len": int(profile.get("t3_initial_speech_len", 0.0)),
            "t3_cond_seq_len": int(profile.get("t3_cond_seq_len", 0.0)),
            "t3_prompt_seq_len": int(profile.get("t3_prompt_seq_len", 0.0)),
            "t3_prompt_hidden_size": int(profile.get("t3_prompt_hidden_size", 0.0)),
            "t3_prompt_token_len_before_mm": int(profile.get("t3_prompt_token_len_before_mm", 0.0)),
            "sampling_max_tokens": int(getattr(sampling_params, "max_tokens", 0) or 0),
            "sampling_stop_token_ids": [
                int(token_id) for token_id in (getattr(sampling_params, "stop_token_ids", None) or [])
            ],
        }

    def _finalize_request(
        self,
        *,
        prepared: dict,
        output,
        t3_duration_s: float,
    ) -> tuple[torch.Tensor, dict]:
        profile = prepared["profile"]
        profile["t3_s"] = t3_duration_s
        profile["t3_active_s"] = t3_duration_s
        profile["t3_batch_size"] = float(prepared.get("batch_size", 1))

        raw_token_ids = list(output.token_ids) if output is not None else []
        token_ids = list(raw_token_ids)
        finish_reason = getattr(output, "finish_reason", None)
        stop_reason = getattr(output, "stop_reason", None)
        stop_token_id = int(self.t3.hp.stop_speech_token)

        replay_start = time.perf_counter()
        token_ids, replay_diag = _replay_original_alignment_stop(
            t3=self.t3,
            t3_cond=prepared["active_conds"].t3,
            text_tokens=prepared["text_tokens"],
            token_ids=token_ids,
            cfg_weight=float(prepared["active_options"].cfg_weight),
            options=prepared["active_options"],
        )
        profile["t3_alignment_replay_s"] = time.perf_counter() - replay_start
        profile["t3_alignment_replay_enabled"] = float(replay_diag.get("replay_enabled", 0.0) or 0.0)
        profile["t3_alignment_replay_forced_eos"] = float(replay_diag.get("replay_forced_eos", 0.0) or 0.0)
        profile["t3_alignment_trimmed"] = float(replay_diag.get("replay_trimmed", 0.0) or 0.0)
        profile["t3_alignment_trim_index"] = float(replay_diag.get("replay_trim_index", -1.0) or -1.0)
        replay_error = str(replay_diag.get("replay_error", "") or "")
        if replay_error:
            profile["t3_alignment_replay_error"] = replay_error

        if len(token_ids) == 0 and len(raw_token_ids) > 0:
            token_ids = raw_token_ids
            profile["t3_alignment_empty_fallback"] = 1.0

        profile["t3_finish_reason_stop"] = 1.0 if finish_reason == "stop" else 0.0
        profile["t3_finish_reason_length"] = 1.0 if finish_reason == "length" else 0.0
        profile["t3_output_has_stop_token"] = 1.0 if stop_token_id in token_ids else 0.0
        profile["t3_generated_tokens"] = float(len(token_ids))
        profile["t3_generated_tokens_before_alignment"] = float(len(raw_token_ids))
        profile["t3_tail_trimmed"] = profile["t3_alignment_trimmed"]
        profile["t3_tail_trim_tokens"] = float(max(0, len(raw_token_ids) - len(token_ids)))
        profile["t3_tail_trim_pattern_size"] = 0.0
        profile["t3_tail_trim_repeats"] = 0.0
        profile["t3_stop_reason_is_stop_token"] = 1.0 if stop_reason == stop_token_id else 0.0

        speech_tokens = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        if speech_tokens.ndim == 1:
            speech_tokens = speech_tokens.unsqueeze(0)
        speech_tokens = drop_invalid_tokens(speech_tokens[0]).to(self.device)

        s3_start = time.perf_counter()
        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=prepared["active_conds"].gen,
        )
        profile.update(self.s3gen.get_last_profile() or {})
        profile["s3_s"] = time.perf_counter() - s3_start
        profile["audio_ready_s"] = time.perf_counter() - prepared["request_start"]
        wav = wav.squeeze(0).detach().cpu().numpy()
        watermark_start = time.perf_counter()
        watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        profile["watermark_s"] = time.perf_counter() - watermark_start
        return torch.from_numpy(watermarked_wav).unsqueeze(0), profile

    def _build_vllm_inputs(self, prepared_list: list[dict]) -> tuple[list, list, list[int]]:
        """
        Build (prompts, sampling_params, cond_output_indices) for a batch of requests.

        For each request with cfg_weight > 0 a cond + uncond pair is emitted.
        cond_output_indices[i] is the index inside the returned lists (and
        therefore inside vllm_engine.generate() outputs) that holds the
        conditioned result for prepared_list[i].

        Requests with cfg_weight == 0 emit only a single cond prompt.
        The T3CFGLogitsProcessor is a no-op for those (no extra_args set).
        """
        prompts: list = []
        sampling_params_list: list = []
        cond_output_indices: list[int] = []

        for item in prepared_list:
            cfg_weight = float(item["active_options"].cfg_weight)
            text_token_len = int(item["profile"].get("t3_text_token_len", 0))
            # Minimum speech tokens before EOS is allowed: rough lower bound
            # based on text length.  Speech runs at ~50 tokens/second; we give
            # at least 0.5 s (25 tokens) and scale with text length.
            min_speech_tokens = max(25, text_token_len * 3)

            cond_idx = len(prompts)
            cond_output_indices.append(cond_idx)

            if cfg_weight > 0.0:
                pair_id = str(uuid4())
                cond_sp, uncond_sp = make_cfg_pair_sampling_params(
                    options=item["active_options"],
                    hp=self.t3.hp,
                    pair_id=pair_id,
                    min_speech_tokens=min_speech_tokens,
                )
                prompts.append(item["prompt"])
                sampling_params_list.append(cond_sp)
                prompts.append(build_vllm_uncond_prompt(t3_cond=item["active_conds"].t3))
                sampling_params_list.append(uncond_sp)
            else:
                prompts.append(item["prompt"])
                sampling_params_list.append(item["sampling_params"])

        return prompts, sampling_params_list, cond_output_indices

    def generate(self, *, session, text: str, options=None) -> torch.Tensor:
        prepared = self._prepare_request(session=session, text=text, options=options)
        prepared["batch_size"] = 1

        with torch.inference_mode():
            prompts, sampling_params_list, cond_output_indices = self._build_vllm_inputs([prepared])

            t3_start = time.perf_counter()
            outputs = self.vllm_engine.generate(
                prompts,
                sampling_params=sampling_params_list,
                use_tqdm=False,
            )
            t3_duration_s = time.perf_counter() - t3_start

            cond_out = outputs[cond_output_indices[0]].outputs[0] if outputs else None
            token_ids = list(cond_out.token_ids) if cond_out is not None else []
            finish_reason = getattr(cond_out, "finish_reason", None)
            stop_reason = getattr(cond_out, "stop_reason", None)

            replay_diag = {
                "replay_enabled": 0.0,
                "replay_error": "",
                "replay_forced_eos": 0.0,
                "replay_trimmed": 0.0,
                "replay_trim_index": -1.0,
                "generated_tokens_before": float(len(token_ids)),
                "generated_tokens_after": float(len(token_ids)),
                "cfg_in_vllm": 1.0,
            }
            output_wav, profile = self._finalize_from_tokens(
                prepared=prepared,
                raw_token_ids=token_ids,
                token_ids=token_ids,
                replay_diag=replay_diag,
                t3_duration_s=t3_duration_s,
                finish_reason=finish_reason,
                stop_reason=stop_reason,
            )

        self._set_last_profile(profile)
        return output_wav

    def _finalize_from_tokens(
        self,
        *,
        prepared: dict,
        raw_token_ids: list[int],
        token_ids: list[int],
        replay_diag: dict,
        t3_duration_s: float,
        finish_reason=None,
        stop_reason=None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Complete finalization (S3 + watermark + profile) given pre-computed token_ids
        from the alignment replay.  Avoids re-running the replay pass.
        """
        profile = prepared["profile"]
        profile["t3_s"] = t3_duration_s
        profile["t3_active_s"] = t3_duration_s
        profile["t3_batch_size"] = float(prepared.get("batch_size", 1))

        stop_token_id = int(self.t3.hp.stop_speech_token)

        profile["t3_alignment_replay_enabled"] = float(replay_diag.get("replay_enabled", 0.0) or 0.0)
        profile["t3_alignment_replay_forced_eos"] = float(replay_diag.get("replay_forced_eos", 0.0) or 0.0)
        profile["t3_alignment_trimmed"] = float(replay_diag.get("replay_trimmed", 0.0) or 0.0)
        profile["t3_alignment_trim_index"] = float(replay_diag.get("replay_trim_index", -1.0) or -1.0)
        replay_error = str(replay_diag.get("replay_error", "") or "")
        if replay_error:
            profile["t3_alignment_replay_error"] = replay_error

        if len(token_ids) == 0 and len(raw_token_ids) > 0:
            token_ids = raw_token_ids
            profile["t3_alignment_empty_fallback"] = 1.0

        profile["t3_finish_reason_stop"] = 1.0 if finish_reason == "stop" else 0.0
        profile["t3_finish_reason_length"] = 1.0 if finish_reason == "length" else 0.0
        profile["t3_output_has_stop_token"] = 1.0 if stop_token_id in token_ids else 0.0
        profile["t3_generated_tokens"] = float(len(token_ids))
        profile["t3_generated_tokens_before_alignment"] = float(len(raw_token_ids))
        profile["t3_tail_trimmed"] = profile["t3_alignment_trimmed"]
        profile["t3_tail_trim_tokens"] = float(max(0, len(raw_token_ids) - len(token_ids)))
        profile["t3_tail_trim_pattern_size"] = 0.0
        profile["t3_tail_trim_repeats"] = 0.0
        profile["t3_stop_reason_is_stop_token"] = 1.0 if stop_reason == stop_token_id else 0.0

        speech_tokens = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        if speech_tokens.ndim == 1:
            speech_tokens = speech_tokens.unsqueeze(0)
        speech_tokens = drop_invalid_tokens(speech_tokens[0]).to(self.device)

        s3_start = time.perf_counter()
        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=prepared["active_conds"].gen,
        )
        profile.update(self.s3gen.get_last_profile() or {})
        profile["s3_s"] = time.perf_counter() - s3_start
        profile["audio_ready_s"] = time.perf_counter() - prepared["request_start"]
        wav = wav.squeeze(0).detach().cpu().numpy()
        watermark_start = time.perf_counter()
        watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        profile["watermark_s"] = time.perf_counter() - watermark_start
        return torch.from_numpy(watermarked_wav).unsqueeze(0), profile

    def generate_many(self, *, sessions, texts: list[str], options_list=None) -> list[dict]:
        if len(sessions) != len(texts):
            raise ValueError("sessions and texts must have the same length")

        if options_list is None:
            options_list = [None] * len(sessions)
        if len(options_list) != len(sessions):
            raise ValueError("options_list and sessions must have the same length")

        request_start = time.perf_counter()
        prepared = [
            self._prepare_request(
                session=session,
                text=text,
                options=options,
                request_start=request_start,
            )
            for session, text, options in zip(sessions, texts, options_list)
        ]

        batch_size = len(prepared)
        for item in prepared:
            item["batch_size"] = batch_size

        with torch.inference_mode():
            # Build prompts list: each request may emit 1 (no CFG) or 2 (CFG pair) entries
            prompts, sampling_params_list, cond_output_indices = self._build_vllm_inputs(prepared)

            t3_start = time.perf_counter()
            outputs = self.vllm_engine.generate(
                prompts,
                sampling_params=sampling_params_list,
                use_tqdm=False,
            )
            t3_duration_s = time.perf_counter() - t3_start

            results = []
            for item, cond_idx in zip(prepared, cond_output_indices):
                cond_out = outputs[cond_idx].outputs[0] if outputs[cond_idx].outputs else None
                token_ids = list(cond_out.token_ids) if cond_out is not None else []
                finish_reason = getattr(cond_out, "finish_reason", None)
                stop_reason = getattr(cond_out, "stop_reason", None)

                replay_diag = {
                    "replay_enabled": 0.0,
                    "replay_error": "",
                    "replay_forced_eos": 0.0,
                    "replay_trimmed": 0.0,
                    "replay_trim_index": -1.0,
                    "generated_tokens_before": float(len(token_ids)),
                    "generated_tokens_after": float(len(token_ids)),
                    "cfg_in_vllm": 1.0,
                    "batch_size": float(batch_size),
                }
                wav, profile = self._finalize_from_tokens(
                    prepared=item,
                    raw_token_ids=token_ids,
                    token_ids=token_ids,
                    replay_diag=replay_diag,
                    t3_duration_s=t3_duration_s,
                    finish_reason=finish_reason,
                    stop_reason=stop_reason,
                )
                results.append({"wav": wav, "profile": profile})

        return results
