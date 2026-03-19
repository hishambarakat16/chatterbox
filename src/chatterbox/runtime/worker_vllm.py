import time

import torch

from ..models.s3tokenizer import drop_invalid_tokens
from ..runtime.session import apply_exaggeration, clone_conditionals
from ..mtl_tts import SUPPORTED_LANGUAGES
from .worker import ChatterboxMultilingualStreamingWorker
from ..vllm_t3_bridge import build_prompt_embeds, make_sampling_params, prepare_vllm_text_tokens


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

    def _finalize_request(self, *, prepared: dict, token_ids: list[int], t3_duration_s: float) -> tuple[torch.Tensor, dict]:
        profile = prepared["profile"]
        profile["t3_s"] = t3_duration_s
        profile["t3_active_s"] = t3_duration_s
        profile["t3_batch_size"] = float(prepared.get("batch_size", 1))

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
            token_ids = outputs[0].outputs[0].token_ids if outputs and outputs[0].outputs else []
            output_wav, profile = self._finalize_request(
                prepared=prepared,
                token_ids=token_ids,
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
                token_ids = output.outputs[0].token_ids if output.outputs else []
                wav, profile = self._finalize_request(
                    prepared=item,
                    token_ids=token_ids,
                    t3_duration_s=t3_duration_s,
                )
                results.append(
                    {
                        "wav": wav,
                        "profile": profile,
                    }
                )
        return results
