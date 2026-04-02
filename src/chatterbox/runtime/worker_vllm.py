import time
from concurrent.futures import ThreadPoolExecutor

import torch

from ..models.s3tokenizer import drop_invalid_tokens
from ..runtime.session import apply_exaggeration, clone_conditionals
from ..mtl_tts import SUPPORTED_LANGUAGES
from .worker import ChatterboxMultilingualStreamingWorker
from ..vllm_t3_bridge import build_prompt_embeds, make_sampling_params, prepare_vllm_text_tokens


def _find_repeated_suffix(
    token_ids: list[int],
    *,
    min_repeats: int = 3,
    max_pattern_size: int = 4,
):
    if len(token_ids) < min_repeats:
        return None

    best = None
    limit = min(max_pattern_size, len(token_ids) // min_repeats)
    for pattern_size in range(1, limit + 1):
        suffix = token_ids[-pattern_size:]
        repeats = 1
        pos = len(token_ids) - pattern_size
        while pos - pattern_size >= 0 and token_ids[pos - pattern_size : pos] == suffix:
            repeats += 1
            pos -= pattern_size
        if repeats < min_repeats:
            continue
        trim_index = pos + pattern_size
        trim_tokens = len(token_ids) - trim_index
        candidate = {
            "trim_index": trim_index,
            "trim_tokens": trim_tokens,
            "pattern_size": pattern_size,
            "repeats": repeats,
        }
        if best is None or candidate["trim_tokens"] > best["trim_tokens"]:
            best = candidate
    return best


def _trim_length_capped_tail(
    token_ids: list[int],
    *,
    finish_reason: str | None,
    stop_token_id: int,
):
    diagnostics = {
        "generated_tokens": float(len(token_ids)),
        "finish_reason_stop": 1.0 if finish_reason == "stop" else 0.0,
        "finish_reason_length": 1.0 if finish_reason == "length" else 0.0,
        "output_has_stop_token": 1.0 if stop_token_id in token_ids else 0.0,
        "tail_trimmed": 0.0,
        "tail_trim_tokens": 0.0,
        "tail_trim_pattern_size": 0.0,
        "tail_trim_repeats": 0.0,
    }

    if finish_reason != "length" or stop_token_id in token_ids:
        return token_ids, diagnostics

    repeated_suffix = _find_repeated_suffix(token_ids)
    if repeated_suffix is None:
        return token_ids, diagnostics

    trim_index = repeated_suffix["trim_index"]
    trimmed = token_ids[:trim_index]
    diagnostics["tail_trimmed"] = 1.0
    diagnostics["tail_trim_tokens"] = float(repeated_suffix["trim_tokens"])
    diagnostics["tail_trim_pattern_size"] = float(repeated_suffix["pattern_size"])
    diagnostics["tail_trim_repeats"] = float(repeated_suffix["repeats"])
    diagnostics["generated_tokens"] = float(len(trimmed))
    return trimmed, diagnostics


def _resolve_effective_max_new_tokens(
    *,
    requested_max_new_tokens: int,
    text_token_len: int,
    auto_enabled: bool,
    auto_cap: int,
) -> int:
    requested = max(1, int(requested_max_new_tokens))
    if not auto_enabled:
        return requested

    cap = max(1, int(auto_cap))
    # Small slack avoids clipping borderline endings near tier thresholds.
    text_token_slack = 7
    content_tokens = max(1, int(text_token_len) - 2 + text_token_slack)

    if content_tokens <= 8:
        dynamic = 32
    elif content_tokens <= 16:
        dynamic = 48
    elif content_tokens <= 32:
        dynamic = 64
    elif content_tokens <= 64:
        dynamic = 96
    else:
        dynamic = cap

    return max(1, min(requested, cap, dynamic))


class ChatterboxMultilingualVllmWorker(ChatterboxMultilingualStreamingWorker):
    """
    Experimental vLLM T3 worker.

    Design choices for the first spike:
    - keep session creation and turbo S3 local
    - build prompt embeddings locally
    - let vLLM handle speech-token generation
    - Hydra and CFG are intentionally disabled
    """

    def __init__(
        self,
        *args,
        prompt_builder_t3,
        prompt_builder_device: str,
        vllm_engine,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prompt_builder_t3 = prompt_builder_t3
        self.prompt_builder_device = prompt_builder_device
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
                "request_conditionals_clone_s": 0.0,
                "request_prompt_conditionals_s": 0.0,
                "t3_text_normalize_s": 0.0,
                "t3_text_tokenize_s": 0.0,
                "t3_text_pad_s": 0.0,
                "t3_text_tokens_total_s": 0.0,
                "t3_prompt_embed_cond_to_device_s": 0.0,
                "t3_prompt_embed_prepare_s": 0.0,
                "t3_prompt_embed_cpu_s": 0.0,
                "t3_prompt_embed_s": 0.0,
                "t3_sampling_params_s": 0.0,
                "t3_first_token_s": 0.0,
                "t3_wait_s": 0.0,
                "t3_active_s": 0.0,
                "t3_s": 0.0,
                "t3_vllm_generate_s": 0.0,
                "t3_alignment_analyzer_supported": 0.0,
                "t3_alignment_analyzer_active": 0.0,
                "t3_alignment_analyzer_s": 0.0,
                "t3_output_extract_s": 0.0,
                "t3_tail_trim_s": 0.0,
                "t3_to_s3_tokens_s": 0.0,
                "s3_finalize_queue_delay_s": 0.0,
                "s3_finalize_order": 0.0,
                "s3_finalize_batch_size": 0.0,
                "batch_s3_finalize_loop_s": 0.0,
                "s3_s": 0.0,
                "audio_ready_s": 0.0,
                "watermark_s": 0.0,
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
        active_conds_start = time.perf_counter()
        active_conds = clone_conditionals(session.conditionals)
        active_conds = apply_exaggeration(active_conds, active_options.exaggeration, self.device)
        profile["request_conditionals_clone_s"] = time.perf_counter() - active_conds_start

        prompt_conds_start = time.perf_counter()
        prompt_conds = clone_conditionals(active_conds)
        prompt_conds = apply_exaggeration(
            prompt_conds,
            active_options.exaggeration,
            self.prompt_builder_device,
        )
        profile["request_prompt_conditionals_s"] = time.perf_counter() - prompt_conds_start

        text_tokens, text_token_meta = prepare_vllm_text_tokens(
            tokenizer=self.tokenizer,
            text=text,
            language_id=language_id,
            device=self.prompt_builder_device,
            return_metadata=True,
        )
        profile["t3_text_normalize_s"] = float(text_token_meta["text_normalize_s"])
        profile["t3_text_tokenize_s"] = float(text_token_meta["text_tokenize_s"])
        profile["t3_text_pad_s"] = float(text_token_meta["text_pad_s"])
        profile["t3_text_tokens_total_s"] = float(text_token_meta["text_tokens_total_s"])

        prompt_embeds, prompt_embed_meta = build_prompt_embeds(
            prompt_builder_t3=self.prompt_builder_t3,
            t3_cond=prompt_conds.t3,
            text_tokens=text_tokens,
            return_metadata=True,
        )
        profile["t3_prompt_embed_cond_to_device_s"] = float(prompt_embed_meta["prompt_embed_cond_to_device_s"])
        profile["t3_prompt_embed_prepare_s"] = float(prompt_embed_meta["prompt_embed_prepare_s"])
        profile["t3_prompt_embed_cpu_s"] = float(prompt_embed_meta["prompt_embed_cpu_s"])
        profile["t3_prompt_embed_s"] = float(prompt_embed_meta["prompt_embed_total_s"])
        profile["text_prep_s"] = time.perf_counter() - prep_start
        profile["t3_text_token_len"] = float(prompt_embed_meta["text_token_len"])
        profile["t3_prompt_speech_token_len"] = float(prompt_embed_meta["prompt_speech_token_len"])
        profile["t3_initial_speech_len"] = float(prompt_embed_meta["initial_speech_len"])
        profile["t3_cond_seq_len"] = float(prompt_embed_meta["cond_seq_len"])
        profile["t3_prompt_embed_seq_len"] = float(prompt_embed_meta["prompt_embed_seq_len"])
        profile["t3_prompt_embed_hidden_size"] = float(prompt_embed_meta["prompt_embed_hidden_size"])

        effective_max_new_tokens = _resolve_effective_max_new_tokens(
            requested_max_new_tokens=active_options.max_new_tokens,
            text_token_len=prompt_embed_meta["text_token_len"],
            auto_enabled=bool(getattr(active_options, "auto_max_new_tokens", False)),
            auto_cap=int(getattr(active_options, "auto_max_new_tokens_cap", 128)),
        )
        profile["t3_max_new_tokens_requested"] = float(active_options.max_new_tokens)
        profile["t3_auto_max_new_tokens_enabled"] = 1.0 if bool(getattr(active_options, "auto_max_new_tokens", False)) else 0.0
        profile["t3_auto_max_new_tokens_cap"] = float(int(getattr(active_options, "auto_max_new_tokens_cap", 128)))
        profile["t3_max_new_tokens_effective"] = float(effective_max_new_tokens)

        if float(active_options.cfg_weight) != 0.0:
            profile["t3_cfg_requested"] = float(active_options.cfg_weight)
        profile["t3_cfg_supported"] = 0.0
        profile["t3_hydra_supported"] = 0.0
        profile["t3_engine_vllm"] = 1.0

        sampling_options = active_options.merged(max_new_tokens=effective_max_new_tokens)
        sampling_start = time.perf_counter()
        sampling_params = make_sampling_params(
            options=sampling_options,
            hp=self.prompt_builder_t3.hp,
        )
        profile["t3_sampling_params_s"] = time.perf_counter() - sampling_start
        return {
            "request_start": request_start,
            "profile": profile,
            "active_conds": active_conds,
            "prompt": {"prompt_embeds": prompt_embeds},
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
            "t3_prompt_embed_seq_len": int(profile.get("t3_prompt_embed_seq_len", 0.0)),
            "t3_prompt_embed_hidden_size": int(profile.get("t3_prompt_embed_hidden_size", 0.0)),
            "sampling_max_tokens": int(getattr(sampling_params, "max_tokens", 0) or 0),
            "t3_max_new_tokens_requested": int(profile.get("t3_max_new_tokens_requested", 0.0)),
            "t3_max_new_tokens_effective": int(profile.get("t3_max_new_tokens_effective", 0.0)),
            "t3_auto_max_new_tokens_enabled": bool(profile.get("t3_auto_max_new_tokens_enabled", 0.0)),
            "t3_auto_max_new_tokens_cap": int(profile.get("t3_auto_max_new_tokens_cap", 0.0)),
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
        profile["t3_vllm_generate_s"] = t3_duration_s
        profile["t3_active_s"] = t3_duration_s
        profile["t3_batch_size"] = float(prepared.get("batch_size", 1))
        profile["s3_finalize_order"] = float(prepared.get("s3_finalize_order", 0))
        profile["s3_finalize_batch_size"] = float(prepared.get("batch_size", 1))
        profile["s3_finalize_queue_delay_s"] = float(prepared.get("s3_finalize_queue_delay_s", 0.0))

        extract_start = time.perf_counter()
        token_ids = list(output.token_ids) if output is not None else []
        finish_reason = getattr(output, "finish_reason", None)
        stop_reason = getattr(output, "stop_reason", None)
        profile["t3_output_extract_s"] = time.perf_counter() - extract_start

        stop_token_id = int(self.prompt_builder_t3.hp.stop_speech_token)
        trim_start = time.perf_counter()
        token_ids, trim_diag = _trim_length_capped_tail(
            token_ids,
            finish_reason=finish_reason,
            stop_token_id=stop_token_id,
        )
        profile["t3_tail_trim_s"] = time.perf_counter() - trim_start
        profile["t3_finish_reason_stop"] = trim_diag["finish_reason_stop"]
        profile["t3_finish_reason_length"] = trim_diag["finish_reason_length"]
        profile["t3_output_has_stop_token"] = trim_diag["output_has_stop_token"]
        profile["t3_generated_tokens"] = trim_diag["generated_tokens"]
        profile["t3_tail_trimmed"] = trim_diag["tail_trimmed"]
        profile["t3_tail_trim_tokens"] = trim_diag["tail_trim_tokens"]
        profile["t3_tail_trim_pattern_size"] = trim_diag["tail_trim_pattern_size"]
        profile["t3_tail_trim_repeats"] = trim_diag["tail_trim_repeats"]
        profile["t3_stop_reason_is_stop_token"] = 1.0 if stop_reason == stop_token_id else 0.0

        token_prep_start = time.perf_counter()
        speech_tokens = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        if speech_tokens.ndim == 1:
            speech_tokens = speech_tokens.unsqueeze(0)
        speech_tokens = drop_invalid_tokens(speech_tokens[0]).to(self.device)
        profile["t3_to_s3_tokens_s"] = time.perf_counter() - token_prep_start

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

    def generate(self, *, session, text: str, options=None) -> torch.Tensor:
        prepared = self._prepare_request(session=session, text=text, options=options)
        prepared["batch_size"] = 1
        prepared["s3_finalize_order"] = 0
        prepared["s3_finalize_queue_delay_s"] = 0.0

        with torch.inference_mode():
            t3_start = time.perf_counter()
            outputs = self.vllm_engine.generate(
                [prepared["prompt"]],
                sampling_params=prepared["sampling_params"],
                use_tqdm=False,
            )
            t3_duration_s = time.perf_counter() - t3_start
            output_wav, profile = self._finalize_request(
                prepared=prepared,
                output=(outputs[0].outputs[0] if outputs and outputs[0].outputs else None),
                t3_duration_s=t3_duration_s,
            )
        self._set_last_profile(profile)
        return output_wav

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

        # T3 decode: batched vLLM generate (must stay in inference_mode)
        with torch.inference_mode():
            t3_start = time.perf_counter()
            outputs = self.vllm_engine.generate(
                [item["prompt"] for item in prepared],
                sampling_params=[item["sampling_params"] for item in prepared],
                use_tqdm=False,
            )
            t3_duration_s = time.perf_counter() - t3_start

        t3_batch_end = time.perf_counter()
        n = len(prepared)
        for i, item in enumerate(prepared):
            item["batch_size"] = n
            item["s3_finalize_order"] = i

        # S3 finalize: one CUDA stream per request, all run in parallel.
        # S3 methods are decorated @torch.inference_mode() internally so no
        # outer context is needed here.
        use_streams = torch.cuda.is_available() and n > 1
        streams = [torch.cuda.Stream() for _ in range(n)] if use_streams else [None] * n

        def _finalize_one(args):
            item, raw_output, stream = args
            item["s3_finalize_queue_delay_s"] = time.perf_counter() - t3_batch_end
            vllm_output = raw_output.outputs[0] if raw_output.outputs else None
            if stream is not None:
                with torch.cuda.stream(stream):
                    wav, profile = self._finalize_request(
                        prepared=item,
                        output=vllm_output,
                        t3_duration_s=t3_duration_s,
                    )
                stream.synchronize()
            else:
                wav, profile = self._finalize_request(
                    prepared=item,
                    output=vllm_output,
                    t3_duration_s=t3_duration_s,
                )
            return wav, profile

        s3_finalize_loop_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=min(n, 4)) as executor:
            finalize_results = list(executor.map(
                _finalize_one,
                [(item, output, stream) for item, output, stream in zip(prepared, outputs, streams)],
            ))
        s3_finalize_loop_s = time.perf_counter() - s3_finalize_loop_start

        results = [{"wav": wav, "profile": profile} for wav, profile in finalize_results]
        for result in results:
            result["profile"]["batch_s3_finalize_loop_s"] = s3_finalize_loop_s
            result["profile"]["s3_parallel_workers"] = float(n)
        return results
