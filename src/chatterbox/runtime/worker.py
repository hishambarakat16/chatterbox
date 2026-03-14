import os
import logging
import threading
import time

import librosa
import torch
import torch.nn.functional as F

from ..models.s3gen import S3GEN_SR, S3Gen
from ..models.s3tokenizer import S3_SR, drop_invalid_tokens
from ..models.t3 import T3
from ..models.t3.modules.cond_enc import T3Cond
from ..models.tokenizers import MTLTokenizer
from ..models.voice_encoder import VoiceEncoder
from ..mtl_tts import Conditionals, SUPPORTED_LANGUAGES, punc_norm
from ..watermarking import create_watermarker
from .session import StreamingSession, apply_exaggeration, clone_conditionals
from .types import GenerationOptions

shape_logger = logging.getLogger("chatterbox.shape")


class ChatterboxMultilingualStreamingWorker:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
        device: str,
        default_conds: Conditionals | None = None,
    ):
        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.default_conds = default_conds
        self.watermarker = create_watermarker()
        self._profile_local = threading.local()

    def _set_last_profile(self, profile: dict):
        self._profile_local.last_profile = profile

    def get_last_profile(self) -> dict:
        return getattr(self._profile_local, "last_profile", {})

    def build_conditionals_from_wav(self, wav_fpath: str, exaggeration: float) -> Conditionals:
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        conds = Conditionals(t3_cond, s3gen_ref_dict)
        if os.getenv("CHATTERBOX_TRACE_SHAPES"):
            shape_logger.info("[runtime/worker.py] build_conditionals_from_wav")
            shape_logger.info("  wav_fpath %s", wav_fpath)
            shape_logger.info("  speaker_emb %s %s %s", tuple(t3_cond.speaker_emb.shape), t3_cond.speaker_emb.dtype, t3_cond.speaker_emb.device)
            if t3_cond.cond_prompt_speech_tokens is not None:
                shape_logger.info(
                    "  cond_prompt_speech_tokens %s %s %s",
                    tuple(t3_cond.cond_prompt_speech_tokens.shape),
                    t3_cond.cond_prompt_speech_tokens.dtype,
                    t3_cond.cond_prompt_speech_tokens.device,
                )
            shape_logger.info("  emotion_adv %s %s %s", tuple(t3_cond.emotion_adv.shape), t3_cond.emotion_adv.dtype, t3_cond.emotion_adv.device)
            shape_logger.info("  s3_ref.prompt_token %s %s %s", tuple(s3gen_ref_dict["prompt_token"].shape), s3gen_ref_dict["prompt_token"].dtype, s3gen_ref_dict["prompt_token"].device)
            shape_logger.info("  s3_ref.prompt_feat %s %s %s", tuple(s3gen_ref_dict["prompt_feat"].shape), s3gen_ref_dict["prompt_feat"].dtype, s3gen_ref_dict["prompt_feat"].device)
            shape_logger.info("  s3_ref.embedding %s %s %s", tuple(s3gen_ref_dict["embedding"].shape), s3gen_ref_dict["embedding"].dtype, s3gen_ref_dict["embedding"].device)
        return conds

    def create_session(
        self,
        *,
        audio_prompt_path: str | None = None,
        options: GenerationOptions | None = None,
        session_id: str | None = None,
    ) -> StreamingSession:
        options = options or GenerationOptions()

        if audio_prompt_path:
            conds = self.build_conditionals_from_wav(audio_prompt_path, options.exaggeration)
        else:
            if self.default_conds is None:
                raise AssertionError("Please provide `audio_prompt_path` or load a checkpoint with builtin conds.")
            conds = clone_conditionals(self.default_conds)
            conds = apply_exaggeration(conds, options.exaggeration, self.device)

        session = StreamingSession(conditionals=conds, options=options)
        if session_id is not None:
            session.session_id = session_id
        if os.getenv("CHATTERBOX_TRACE_SHAPES"):
            shape_logger.info("[runtime/worker.py] create_session")
            shape_logger.info("  session_id %s", session.session_id)
            shape_logger.info("  options %s", options.__dict__)
        return session

    def generate(self, *, session: StreamingSession, text: str, options: GenerationOptions | None = None) -> torch.Tensor:
        request_start = time.perf_counter()
        profile = {
            "text_prep_s": 0.0,
            "t3_s": 0.0,
            "s3_s": 0.0,
            "audio_ready_s": 0.0,
            "watermark_s": 0.0,
        }
        active_options = session.options if options is None else session.options.merged(**options.__dict__)
        language_id = active_options.language_id
        if os.getenv("CHATTERBOX_TRACE_SHAPES"):
            shape_logger.info("[runtime/worker.py] generate.input")
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
        active_conds = clone_conditionals(session.conditionals)
        active_conds = apply_exaggeration(active_conds, active_options.exaggeration, self.device)

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
            shape_logger.info("[runtime/worker.py] generate.text_tokens")
            shape_logger.info("  session_id %s", session.session_id)
            shape_logger.info("  text_tokens %s %s %s", tuple(text_tokens.shape), text_tokens.dtype, text_tokens.device)

        with torch.inference_mode():
            t3_start = time.perf_counter()
            speech_tokens = self.t3.inference(
                t3_cond=active_conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=active_options.max_new_tokens,
                temperature=active_options.temperature,
                cfg_weight=active_options.cfg_weight,
                repetition_penalty=active_options.repetition_penalty,
                min_p=active_options.min_p,
                top_p=active_options.top_p,
            )
            profile["t3_s"] = time.perf_counter() - t3_start
            speech_tokens = speech_tokens[0]
            if os.getenv("CHATTERBOX_TRACE_SHAPES"):
                shape_logger.info("[runtime/worker.py] generate.speech_tokens.raw")
                shape_logger.info("  session_id %s", session.session_id)
                shape_logger.info("  speech_tokens %s %s %s", tuple(speech_tokens.shape), speech_tokens.dtype, speech_tokens.device)
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)
            if os.getenv("CHATTERBOX_TRACE_SHAPES"):
                shape_logger.info("[runtime/worker.py] generate.speech_tokens.filtered")
                shape_logger.info("  session_id %s", session.session_id)
                shape_logger.info("  speech_tokens %s %s %s", tuple(speech_tokens.shape), speech_tokens.dtype, speech_tokens.device)

            s3_start = time.perf_counter()
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
            shape_logger.info("[runtime/worker.py] generate.output")
            shape_logger.info("  session_id %s", session.session_id)
            shape_logger.info("  wav %s %s", watermarked_wav.shape, watermarked_wav.dtype)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)
