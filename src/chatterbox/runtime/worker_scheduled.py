import logging
import os
import threading
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

    def __init__(self, *args, batching_window_ms: float = 5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.s3_lock = threading.Lock()
        self.first_audio_chunk_token_count = 16
        self.t3_scheduler = T3DecodeScheduler(
            self.t3,
            batching_window_ms=batching_window_ms,
            partial_audio_callback=self._synthesize_first_audio_chunk,
        )

    def generate(self, *, session, text: str, options=None) -> torch.Tensor:
        request_start = time.perf_counter()
        profile = {
            "text_prep_s": 0.0,
            "t3_first_token_s": 0.0,
            "first_audio_chunk_s": 0.0,
            "first_audio_chunk_num_samples": 0.0,
            "first_audio_chunk_token_count": 0.0,
            "first_audio_chunk_s3_s": 0.0,
            "t3_wait_s": 0.0,
            "t3_active_s": 0.0,
            "t3_s": 0.0,
            "s3_s": 0.0,
            "audio_ready_s": 0.0,
            "watermark_s": 0.0,
        }
        active_options = session.options if options is None else session.options.merged(**options.__dict__)
        language_id = active_options.language_id
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
            s3_ref_dict=active_conds.gen,
            text_tokens=text_tokens,
            max_new_tokens=active_options.max_new_tokens,
            temperature=active_options.temperature,
            top_p=active_options.top_p,
            min_p=active_options.min_p,
            repetition_penalty=active_options.repetition_penalty,
            cfg_weight=active_options.cfg_weight,
            first_audio_chunk_token_count=self.first_audio_chunk_token_count,
        )

        with torch.inference_mode():
            speech_tokens, scheduler_metrics = self.t3_scheduler.submit(decode_request)
            profile.update(scheduler_metrics)
            speech_tokens = speech_tokens[0]
            if os.getenv("CHATTERBOX_TRACE_SHAPES"):
                shape_logger.info("[runtime/worker_scheduled.py] generate.speech_tokens.raw")
                shape_logger.info("  session_id %s", session.session_id)
                shape_logger.info("  speech_tokens %s %s %s", tuple(speech_tokens.shape), speech_tokens.dtype, speech_tokens.device)
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)
            if os.getenv("CHATTERBOX_TRACE_SHAPES"):
                shape_logger.info("[runtime/worker_scheduled.py] generate.speech_tokens.filtered")
                shape_logger.info("  session_id %s", session.session_id)
                shape_logger.info("  speech_tokens %s %s %s", tuple(speech_tokens.shape), speech_tokens.dtype, speech_tokens.device)

            s3_start = time.perf_counter()
            with self.s3_lock:
                wav, _ = self.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=active_conds.gen,
                )
            profile["s3_s"] = time.perf_counter() - s3_start
            profile["audio_ready_s"] = time.perf_counter() - request_start
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermark_start = time.perf_counter()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
            profile["watermark_s"] = time.perf_counter() - watermark_start
        self._set_last_profile(profile)
        if os.getenv("CHATTERBOX_TRACE_SHAPES"):
            shape_logger.info("[runtime/worker_scheduled.py] generate.output")
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

    def _synthesize_first_audio_chunk(self, request: ScheduledDecodeRequest, speech_tokens: torch.Tensor):
        speech_tokens = speech_tokens[0]
        speech_tokens = drop_invalid_tokens(speech_tokens)
        if speech_tokens.numel() <= 3:
            return None

        speech_tokens = speech_tokens.to(self.device).unsqueeze(0)
        speech_token_lens = torch.tensor([speech_tokens.shape[-1]], device=self.device, dtype=torch.long)

        with self.s3_lock:
            output_mels = self.s3gen.flow_inference(
                speech_tokens=speech_tokens,
                ref_dict=request.s3_ref_dict,
                finalize=False,
                speech_token_lens=speech_token_lens,
            )
            output_mels = output_mels.to(dtype=self.s3gen.dtype)
            output_wavs, _ = self.s3gen.hift_inference(output_mels, None)
            if not self.s3gen.training:
                output_wavs[:, :len(self.s3gen.trim_fade)] *= self.s3gen.trim_fade
        return int(output_wavs.shape[-1])
