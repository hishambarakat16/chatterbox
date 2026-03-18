import logging
import os
import threading
import time

import torch
import torch.nn.functional as F

from ..models.s3tokenizer import drop_invalid_tokens
from ..mtl_tts import SUPPORTED_LANGUAGES, punc_norm
from .worker import ChatterboxMultilingualStreamingWorker
from ..models.t3.inference.concurrent_decode import run_concurrent_t3_inference


shape_logger = logging.getLogger("chatterbox.shape")


class ChatterboxMultilingualConcurrentWorker(ChatterboxMultilingualStreamingWorker):
    """
    A/B worker for the first concurrency-correctness step.

    Differences from the current streaming worker:
    - coarse lock around full T3 decode
    - request-local T3 backend/analyzer state
    - no changes to S3 concurrency yet
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t3_decode_lock = threading.Lock()

    def generate(self, *, session, text: str, options=None) -> torch.Tensor:
        request_start = time.perf_counter()
        profile = dict(getattr(session, "profile", {}) or {})
        profile.update({
            "text_prep_s": 0.0,
            "t3_first_token_s": 0.0,
            "t3_wait_s": 0.0,
            "t3_decode_s": 0.0,
            "t3_s": 0.0,
            "s3_s": 0.0,
            "audio_ready_s": 0.0,
            "watermark_s": 0.0,
        })
        active_options = session.options if options is None else session.options.merged(**options.__dict__)
        language_id = active_options.language_id
        if os.getenv("CHATTERBOX_TRACE_SHAPES"):
            shape_logger.info("[runtime/worker_concurrent.py] generate.input")
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
            shape_logger.info("[runtime/worker_concurrent.py] generate.text_tokens")
            shape_logger.info("  session_id %s", session.session_id)
            shape_logger.info("  text_tokens %s %s %s", tuple(text_tokens.shape), text_tokens.dtype, text_tokens.device)

        with torch.inference_mode():
            wait_start = time.perf_counter()
            with self.t3_decode_lock:
                acquired_at = time.perf_counter()
                if os.getenv("CHATTERBOX_TRACE_SHAPES"):
                    shape_logger.info("[runtime/worker_concurrent.py] acquire_t3_decode_lock")
                    shape_logger.info("  session_id %s", session.session_id)
                speech_tokens, t3_metrics = run_concurrent_t3_inference(
                    self.t3,
                    t3_cond=active_conds.t3,
                    text_tokens=text_tokens,
                    max_new_tokens=active_options.max_new_tokens,
                    temperature=active_options.temperature,
                    cfg_weight=active_options.cfg_weight,
                    repetition_penalty=active_options.repetition_penalty,
                    min_p=active_options.min_p,
                    top_p=active_options.top_p,
                )
                t3_end = time.perf_counter()
            profile["t3_wait_s"] = acquired_at - wait_start
            profile["t3_decode_s"] = t3_end - acquired_at
            profile["t3_s"] = t3_end - wait_start
            profile["t3_first_token_s"] = profile["t3_wait_s"] + float(t3_metrics.get("first_token_decode_s", 0.0))
            speech_tokens = speech_tokens[0]
            if os.getenv("CHATTERBOX_TRACE_SHAPES"):
                shape_logger.info("[runtime/worker_concurrent.py] generate.speech_tokens.raw")
                shape_logger.info("  session_id %s", session.session_id)
                shape_logger.info("  speech_tokens %s %s %s", tuple(speech_tokens.shape), speech_tokens.dtype, speech_tokens.device)
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)
            if os.getenv("CHATTERBOX_TRACE_SHAPES"):
                shape_logger.info("[runtime/worker_concurrent.py] generate.speech_tokens.filtered")
                shape_logger.info("  session_id %s", session.session_id)
                shape_logger.info("  speech_tokens %s %s %s", tuple(speech_tokens.shape), speech_tokens.dtype, speech_tokens.device)

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
        if os.getenv("CHATTERBOX_TRACE_SHAPES"):
            shape_logger.info("[runtime/worker_concurrent.py] generate.output")
            shape_logger.info("  session_id %s", session.session_id)
            shape_logger.info("  wav %s %s", watermarked_wav.shape, watermarked_wav.dtype)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    def _apply_exaggeration_copy(self, conds, exaggeration: float):
        current = conds.t3.emotion_adv
        if current is not None and float(exaggeration) == float(current.view(-1)[0].item()):
            return conds

        updated = conds.t3.__class__(**conds.t3.__dict__)
        updated.emotion_adv = exaggeration * torch.ones(1, 1, 1)
        conds.t3 = updated.to(device=self.device)
        return conds
