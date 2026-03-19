import time

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
                "t3_first_token_s": 0.0,
                "t3_wait_s": 0.0,
                "t3_active_s": 0.0,
                "t3_s": 0.0,
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
        active_conds = clone_conditionals(session.conditionals)
        active_conds = apply_exaggeration(active_conds, active_options.exaggeration, self.device)

        prompt_conds = clone_conditionals(active_conds)
        prompt_conds = apply_exaggeration(
            prompt_conds,
            active_options.exaggeration,
            self.prompt_builder_device,
        )

        text_tokens = prepare_vllm_text_tokens(
            tokenizer=self.tokenizer,
            text=text,
            language_id=language_id,
            device=self.prompt_builder_device,
        )
        prompt_embeds = build_prompt_embeds(
            prompt_builder_t3=self.prompt_builder_t3,
            t3_cond=prompt_conds.t3,
            text_tokens=text_tokens,
        )
        profile["text_prep_s"] = time.perf_counter() - prep_start

        if float(active_options.cfg_weight) != 0.0:
            profile["t3_cfg_requested"] = float(active_options.cfg_weight)
        profile["t3_cfg_supported"] = 0.0
        profile["t3_hydra_supported"] = 0.0
        profile["t3_engine_vllm"] = 1.0

        sampling_params = make_sampling_params(
            options=active_options,
            hp=self.prompt_builder_t3.hp,
        )
        return {
            "request_start": request_start,
            "profile": profile,
            "active_conds": active_conds,
            "prompt": {"prompt_embeds": prompt_embeds},
            "sampling_params": sampling_params,
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
        token_ids = list(output.token_ids) if output is not None else []
        finish_reason = getattr(output, "finish_reason", None)
        stop_reason = getattr(output, "stop_reason", None)
        stop_token_id = int(self.prompt_builder_t3.hp.stop_speech_token)
        token_ids, trim_diag = _trim_length_capped_tail(
            token_ids,
            finish_reason=finish_reason,
            stop_token_id=stop_token_id,
        )
        profile["t3_finish_reason_stop"] = trim_diag["finish_reason_stop"]
        profile["t3_finish_reason_length"] = trim_diag["finish_reason_length"]
        profile["t3_output_has_stop_token"] = trim_diag["output_has_stop_token"]
        profile["t3_generated_tokens"] = trim_diag["generated_tokens"]
        profile["t3_tail_trimmed"] = trim_diag["tail_trimmed"]
        profile["t3_tail_trim_tokens"] = trim_diag["tail_trim_tokens"]
        profile["t3_tail_trim_pattern_size"] = trim_diag["tail_trim_pattern_size"]
        profile["t3_tail_trim_repeats"] = trim_diag["tail_trim_repeats"]
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

    def generate(self, *, session, text: str, options=None) -> torch.Tensor:
        prepared = self._prepare_request(session=session, text=text, options=options)

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

        with torch.inference_mode():
            t3_start = time.perf_counter()
            outputs = self.vllm_engine.generate(
                [item["prompt"] for item in prepared],
                sampling_params=[item["sampling_params"] for item in prepared],
                use_tqdm=False,
            )
            t3_duration_s = time.perf_counter() - t3_start

            results = []
            for item in prepared:
                item["batch_size"] = len(prepared)
            for item, output in zip(prepared, outputs):
                wav, profile = self._finalize_request(
                    prepared=item,
                    output=(output.outputs[0] if output.outputs else None),
                    t3_duration_s=t3_duration_s,
                )
                results.append(
                    {
                        "wav": wav,
                        "profile": profile,
                    }
                )
        return results
