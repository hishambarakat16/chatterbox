import logging
import os
import time

import torch
import torch.nn.functional as F

from ..models.s3tokenizer import drop_invalid_tokens
from ..models.t3.inference.scheduled_decode import ScheduledDecodeRequest
from ..mtl_tts import SUPPORTED_LANGUAGES, punc_norm
from .t3_scheduler import T3DecodeScheduler
from .worker import ChatterboxMultilingualStreamingWorker


shape_logger = logging.getLogger("chatterbox.shape")


class ChatterboxMultilingualScheduledWorker(ChatterboxMultilingualStreamingWorker):
    """
    Scheduler-driven T3 worker.

    Compared with the `concurrent` path:
    - one background scheduler thread owns batched T3 decode
    - requests submit request-local T3 state and wait for speech tokens
    - T3 weights remain shared
    - S3 remains unchanged for now
    """

    def __init__(
        self,
        *args,
        batching_window_ms: float = 5.0,
        enable_alignment_controller: bool = False,
        hydra_model=None,
        hydra_speculate_k: int = 3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.t3_scheduler = T3DecodeScheduler(
            self.t3,
            batching_window_ms=batching_window_ms,
            enable_alignment_controller=enable_alignment_controller,
            hydra_model=hydra_model,
            hydra_speculate_k=hydra_speculate_k,
        )

    def generate(self, *, session, text: str, options=None) -> torch.Tensor:
        request_start = time.perf_counter()
        profile = dict(getattr(session, "profile", {}) or {})
        profile.update(self._default_s3_profile())
        profile.update(getattr(session, "profile", {}) or {})
        profile.update({
            "text_prep_s": 0.0,
            "t3_first_token_s": 0.0,
            "t3_wait_s": 0.0,
            "t3_active_s": 0.0,
            "t3_s": 0.0,
            "s3_s": 0.0,
            "audio_ready_s": 0.0,
            "watermark_s": 0.0,
        })
        active_options = session.options if options is None else session.options.merged(**options.__dict__)
        language_id = active_options.language_id
        t3_temperature = 0.0 if self.t3_scheduler.hydra_model is not None else active_options.temperature
        if os.getenv("CHATTERBOX_TRACE_SHAPES"):
            shape_logger.info("[runtime/worker_scheduled.py] generate.input")
            shape_logger.info("  session_id %s", session.session_id)
            shape_logger.info("  text %r", text)
            shape_logger.info("  options %s", active_options.__dict__)

        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        prep_start = time.perf_counter()
        active_conds = session.clone_conditionals()
        active_conds = active_conds.to(device=self.device)
        active_conds = self._apply_exaggeration_copy(active_conds, active_options.exaggeration)

        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(
            text, language_id=language_id.lower() if language_id else None
        ).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)
        profile["text_prep_s"] = time.perf_counter() - prep_start
        if os.getenv("CHATTERBOX_TRACE_SHAPES"):
            shape_logger.info("[runtime/worker_scheduled.py] generate.text_tokens")
            shape_logger.info("  session_id %s", session.session_id)
            shape_logger.info("  text_tokens %s %s %s", tuple(text_tokens.shape), text_tokens.dtype, text_tokens.device)

        decode_request = ScheduledDecodeRequest(
            session_id=session.session_id,
            t3_cond=active_conds.t3,
            text_tokens=text_tokens,
            max_new_tokens=active_options.max_new_tokens,
            temperature=t3_temperature,
            top_p=active_options.top_p,
            min_p=active_options.min_p,
            repetition_penalty=active_options.repetition_penalty,
            cfg_weight=active_options.cfg_weight,
        )

        with torch.inference_mode():
            speech_tokens, scheduler_metrics = self.t3_scheduler.submit(decode_request)
            profile.update(scheduler_metrics)
            speech_tokens = speech_tokens[0]
            filter_started = time.perf_counter()
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)
            profile["t3_filter_s"] = time.perf_counter() - filter_started

            s3_start = time.perf_counter()
            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=active_conds.gen,
            )
            profile.update(self.s3gen.get_last_profile() or {})
            profile["s3_s"] = time.perf_counter() - s3_start
            profile["audio_ready_s"] = time.perf_counter() - request_start
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermark_start = time.perf_counter()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
            profile["watermark_s"] = time.perf_counter() - watermark_start
        self._set_last_profile(profile)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    def _apply_exaggeration_copy(self, conds, exaggeration: float):
        current = conds.t3.emotion_adv
        if current is not None and float(exaggeration) == float(current.view(-1)[0].item()):
            return conds

        updated = conds.t3.__class__(**conds.t3.__dict__)
        updated.emotion_adv = exaggeration * torch.ones(1, 1, 1)
        conds.t3 = updated.to(device=self.device)
        return conds
