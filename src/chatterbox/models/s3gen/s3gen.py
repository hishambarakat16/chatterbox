# Modified from CosyVoice https://github.com/FunAudioLLM/CosyVoice
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import threading
import time

import numpy as np
import torch
import torchaudio as ta
from functools import lru_cache
from typing import Optional

from ..s3tokenizer import S3_SR, SPEECH_VOCAB_SIZE, S3Tokenizer
from .const import S3GEN_SR
from .flow import CausalMaskedDiffWithXvec
from .xvector import CAMPPlus
from .utils.mel import mel_spectrogram
from .f0_predictor import ConvRNNF0Predictor
from .hifigan import HiFTGenerator
from .transformer.upsample_encoder import UpsampleConformerEncoder
from .flow_matching import CausalConditionalCFM
from .decoder import ConditionalDecoder
from .configs import CFM_PARAMS

shape_logger = logging.getLogger("chatterbox.shape")
_TRACE_COUNTS: dict[str, int] = {}


def _trace_s3_enabled() -> bool:
    return bool(os.getenv("CHATTERBOX_TRACE_SHAPES") or os.getenv("CHATTERBOX_TRACE_S3_SHAPES"))


def _should_trace_event(name: str) -> bool:
    occurrence = _TRACE_COUNTS.get(name, 0) + 1
    _TRACE_COUNTS[name] = occurrence
    return occurrence == 1


def _maybe_sync(device):
    device_type = getattr(device, "type", None)
    if device_type is None and isinstance(device, str):
        device_type = "cuda" if device.startswith("cuda") else None
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _tensor_max_length(value, *, fallback_shape_dim: int | None = None) -> int:
    if value is None:
        return 0
    if torch.is_tensor(value):
        if value.numel() == 0:
            return 0
        if value.ndim == 0:
            return int(value.item())
        return int(value.max().item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0
        return int(value.max())
    if fallback_shape_dim is not None and hasattr(value, "shape"):
        return int(value.shape[fallback_shape_dim])
    return int(value)


def drop_invalid_tokens(x):
    assert len(x.shape) <= 2 and x.shape[0] == 1, "only batch size of one allowed for now"
    return x[x < SPEECH_VOCAB_SIZE]


# TODO: global resampler cache
@lru_cache(100)
def get_resampler(src_sr, dst_sr, device):
    return ta.transforms.Resample(src_sr, dst_sr).to(device)


class S3Token2Mel(torch.nn.Module):
    """
    S3Gen's CFM decoder maps S3 speech tokens to mel-spectrograms.

    TODO: make these modules configurable?
    """
    def __init__(self, meanflow=False):
        super().__init__()
        self.tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
        self.mel_extractor = mel_spectrogram # TODO: make it a torch module?
        self.speaker_encoder = CAMPPlus(
            # NOTE: This doesn't affect inference. It turns off activation checkpointing
            # (a training optimization), which causes a crazy DDP error with accelerate
            memory_efficient=False,
        )
        self.meanflow = meanflow

        encoder = UpsampleConformerEncoder(
            output_size=512,
            attention_heads=8,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
            input_layer='linear',
            pos_enc_layer_type='rel_pos_espnet',
            selfattention_layer_type='rel_selfattn',
            input_size=512,
            use_cnn_module=False,
            macaron_style=False,
        )

        estimator = ConditionalDecoder(
            in_channels=320,
            out_channels=80,
            causal=True,
            channels=[256],
            dropout=0.0,
            attention_head_dim=64,
            n_blocks=4,
            num_mid_blocks=12,
            num_heads=8,
            act_fn='gelu',
            meanflow=self.meanflow,
        )
        cfm_params = CFM_PARAMS
        decoder = CausalConditionalCFM(
            spk_emb_dim=80,
            cfm_params=cfm_params,
            estimator=estimator,
        )

        self.flow = CausalMaskedDiffWithXvec(
            encoder=encoder,
            decoder=decoder
        )

        self.resamplers = {}
        self._profile_local = threading.local()

    def _set_last_profile(self, profile: dict):
        self._profile_local.last_profile = profile

    def get_last_profile(self) -> dict:
        return getattr(self._profile_local, "last_profile", {})

    @property
    def device(self):
        params = self.tokenizer.parameters()
        return next(params).device

    @property
    def dtype(self):
        params = self.flow.parameters()
        return next(params).dtype

    def embed_ref(
        self,
        ref_wav: torch.Tensor,
        ref_sr: int,
        device="auto",
        ref_fade_out=True,
    ):
        device = self.device if device == "auto" else device
        profile = {}
        embed_start = time.perf_counter()
        if isinstance(ref_wav, np.ndarray):
            ref_wav = torch.from_numpy(ref_wav).float()

        if ref_wav.device != device:
            ref_wav = ref_wav.to(device)

        if len(ref_wav.shape) == 1:
            ref_wav = ref_wav.unsqueeze(0)  # (B, L)

        if ref_wav.size(1) > 10 * ref_sr:
            print("WARNING: s3gen received ref longer than 10s")

        mel_start = time.perf_counter()
        ref_wav_24 = ref_wav
        if ref_sr != S3GEN_SR:
            ref_wav_24 = get_resampler(ref_sr, S3GEN_SR, device)(ref_wav)
        ref_wav_24 = ref_wav_24.to(device=device, dtype=self.dtype)

        ref_mels_24 = self.mel_extractor(ref_wav_24).transpose(1, 2).to(dtype=self.dtype)
        _maybe_sync(device)
        profile["s3_ref_mel_s"] = time.perf_counter() - mel_start
        ref_mels_24_len = None

        # Resample to 16kHz
        speaker_start = time.perf_counter()
        ref_wav_16 = ref_wav
        if ref_sr != S3_SR:
            ref_wav_16 = get_resampler(ref_sr, S3_SR, device)(ref_wav)

        # Speaker embedding
        ref_x_vector = self.speaker_encoder.inference(ref_wav_16.to(dtype=self.dtype))
        _maybe_sync(device)
        profile["s3_ref_speaker_s"] = time.perf_counter() - speaker_start

        # Tokenize 16khz reference
        tokenize_start = time.perf_counter()
        ref_speech_tokens, ref_speech_token_lens = self.tokenizer(ref_wav_16.float())
        _maybe_sync(device)
        profile["s3_ref_tokenize_s"] = time.perf_counter() - tokenize_start

        # Make sure mel_len = 2 * stoken_len (happens when the input is not padded to multiple of 40ms)
        align_start = time.perf_counter()
        if ref_mels_24.shape[1] != 2 * ref_speech_tokens.shape[1]:
            logging.warning(
                "Reference mel length is not equal to 2 * reference token length.\n"
            )
            ref_speech_tokens = ref_speech_tokens[:, :ref_mels_24.shape[1] // 2]
            ref_speech_token_lens[0] = ref_speech_tokens.shape[1]
        profile["s3_ref_align_s"] = time.perf_counter() - align_start

        ref_dict = dict(
            prompt_token=ref_speech_tokens.to(device),
            prompt_token_len=ref_speech_token_lens,
            prompt_feat=ref_mels_24,
            prompt_feat_len=ref_mels_24_len,
            embedding=ref_x_vector,
        )
        _maybe_sync(device)
        profile["s3_ref_embed_s"] = time.perf_counter() - embed_start
        self._set_last_profile(profile)
        if _trace_s3_enabled():
            shape_logger.info("[models/s3gen/s3gen.py] embed_ref")
            shape_logger.info("  prompt_token %s %s %s", tuple(ref_dict["prompt_token"].shape), ref_dict["prompt_token"].dtype, ref_dict["prompt_token"].device)
            shape_logger.info("  prompt_token_len %s %s %s", tuple(ref_dict["prompt_token_len"].shape), ref_dict["prompt_token_len"].dtype, ref_dict["prompt_token_len"].device)
            shape_logger.info("  prompt_feat %s %s %s", tuple(ref_dict["prompt_feat"].shape), ref_dict["prompt_feat"].dtype, ref_dict["prompt_feat"].device)
            if ref_dict["prompt_feat_len"] is not None:
                shape_logger.info("  prompt_feat_len %s %s %s", tuple(ref_dict["prompt_feat_len"].shape), ref_dict["prompt_feat_len"].dtype, ref_dict["prompt_feat_len"].device)
            shape_logger.info("  embedding %s %s %s", tuple(ref_dict["embedding"].shape), ref_dict["embedding"].dtype, ref_dict["embedding"].device)
        return ref_dict

    def forward(
        self,
        speech_tokens: torch.LongTensor,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor],
        ref_sr: Optional[int],
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        n_cfm_timesteps = None,
        finalize: bool = False,
        speech_token_lens=None,
        noised_mels=None,
    ):
        """
        Generate waveforms from S3 speech tokens and a reference waveform, which the speaker timbre is inferred from.

        NOTE:
        - The speaker encoder accepts 16 kHz waveform.
        - S3TokenizerV2 accepts 16 kHz waveform.
        - The mel-spectrogram for the reference assumes 24 kHz input signal.
        - This function is designed for batch_size=1 only.

        Args
        ----
        - `speech_tokens`: S3 speech tokens [B=1, T]
        - `ref_wav`: reference waveform (`torch.Tensor` with shape=[B=1, T])
        - `ref_sr`: reference sample rate
        - `finalize`: whether streaming is finished or not. Note that if False, the last 3 tokens will be ignored.
        """
        assert (ref_wav is None) ^ (ref_dict is None), f"Must provide exactly one of ref_wav or ref_dict (got {ref_wav} and {ref_dict})"

        if ref_dict is None:
            ref_prepare_start = time.perf_counter()
            ref_dict = self.embed_ref(ref_wav, ref_sr)
            profile = dict(self.get_last_profile() or {})
            _maybe_sync(self.device)
            profile["s3_ref_prepare_s"] = time.perf_counter() - ref_prepare_start
        else:
            profile = {}
            # type/device casting (all values will be numpy if it's from a prod API call)
            ref_prepare_start = time.perf_counter()
            for rk in list(ref_dict):
                if isinstance(ref_dict[rk], np.ndarray):
                    ref_dict[rk] = torch.from_numpy(ref_dict[rk])
                if torch.is_tensor(ref_dict[rk]):
                    ref_dict[rk] = ref_dict[rk].to(device=self.device, dtype=self.dtype)
            _maybe_sync(self.device)
            profile["s3_ref_prepare_s"] = time.perf_counter() - ref_prepare_start

        speech_tokens = torch.atleast_2d(speech_tokens)

        # backcompat
        if speech_token_lens is None:
            speech_token_lens = torch.LongTensor([st.size(-1) for st in speech_tokens]).to(self.device)

        speech_token_len = _tensor_max_length(speech_token_lens, fallback_shape_dim=-1)
        prompt_token_len = _tensor_max_length(
            ref_dict.get("prompt_token_len"),
            fallback_shape_dim=-1,
        )
        prompt_feat_frames = int(ref_dict["prompt_feat"].shape[1]) if ref_dict.get("prompt_feat") is not None else 0
        embedding_dim = int(ref_dict["embedding"].shape[-1]) if ref_dict.get("embedding") is not None else 0

        if _trace_s3_enabled():
            shape_logger.info("[models/s3gen/s3gen.py] token2mel.input")
            shape_logger.info("  speech_tokens %s %s %s", tuple(speech_tokens.shape), speech_tokens.dtype, speech_tokens.device)
            shape_logger.info("  speech_token_lens %s %s %s", tuple(speech_token_lens.shape), speech_token_lens.dtype, speech_token_lens.device)
            shape_logger.info("  finalize %s", finalize)
            shape_logger.info("  n_cfm_timesteps %s", n_cfm_timesteps)
            shape_logger.info("  ref.prompt_token %s %s %s", tuple(ref_dict["prompt_token"].shape), ref_dict["prompt_token"].dtype, ref_dict["prompt_token"].device)
            shape_logger.info("  ref.prompt_feat %s %s %s", tuple(ref_dict["prompt_feat"].shape), ref_dict["prompt_feat"].dtype, ref_dict["prompt_feat"].device)
            shape_logger.info("  ref.embedding %s %s %s", tuple(ref_dict["embedding"].shape), ref_dict["embedding"].dtype, ref_dict["embedding"].device)
        token2mel_start = time.perf_counter()
        output_mels, _ = self.flow.inference(
            token=speech_tokens,
            token_len=speech_token_lens,
            finalize=finalize,
            noised_mels=noised_mels,
            n_timesteps=n_cfm_timesteps,
            meanflow=self.meanflow,
            **ref_dict,
        )
        _maybe_sync(self.device)
        profile["s3_token2mel_s"] = time.perf_counter() - token2mel_start
        profile["s3_token2mel_batch_size"] = float(int(speech_tokens.shape[0]))
        profile["s3_token2mel_speech_token_len"] = float(speech_token_len)
        profile["s3_token2mel_prompt_token_len"] = float(prompt_token_len)
        profile["s3_token2mel_total_token_len"] = float(prompt_token_len + speech_token_len)
        profile["s3_token2mel_prompt_mel_frames"] = float(prompt_feat_frames)
        profile["s3_token2mel_generated_mel_frames"] = float(int(output_mels.shape[-1]))
        profile["s3_token2mel_total_mel_frames"] = float(prompt_feat_frames + int(output_mels.shape[-1]))
        profile["s3_token2mel_mel_channels"] = float(int(output_mels.shape[1]) if output_mels.ndim >= 2 else 0)
        profile["s3_token2mel_embedding_dim"] = float(embedding_dim)
        profile["s3_token2mel_ratio"] = float(getattr(self.flow, "token_mel_ratio", 0))
        profile["s3_token2mel_finalize"] = 1.0 if finalize else 0.0
        self._set_last_profile(profile)
        if _trace_s3_enabled() and _should_trace_event("token2mel.output"):
            shape_logger.info("[models/s3gen/s3gen.py] token2mel.output")
            shape_logger.info("  output_mels %s %s %s", tuple(output_mels.shape), output_mels.dtype, output_mels.device)
        return output_mels


class S3Token2Wav(S3Token2Mel):
    """
    The decoder of S3Gen is a concat of token-to-mel (CFM) and a mel-to-waveform (HiFiGAN) modules.

    TODO: make these modules configurable?
    """

    ignore_state_dict_missing = ("tokenizer._mel_filters", "tokenizer.window")

    def __init__(self, meanflow=False):
        super().__init__(meanflow)

        f0_predictor = ConvRNNF0Predictor()
        self.mel2wav = HiFTGenerator(
            sampling_rate=S3GEN_SR,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            f0_predictor=f0_predictor,
        )

        # silence out a few ms and fade audio in to reduce artifacts
        n_trim = S3GEN_SR // 50  # 20ms = half of a frame
        trim_fade = torch.zeros(2 * n_trim)
        trim_fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim)) + 1) / 2
        self.register_buffer("trim_fade", trim_fade, persistent=False) # (buffers get automatic device casting)
        self.estimator_dtype = "fp32"

    def forward(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor],
        ref_sr: Optional[int],
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        finalize: bool = False,
        speech_token_lens=None,
        skip_vocoder=False,
        n_cfm_timesteps=None,
        noised_mels=None,

    ):
        """
        Generate waveforms from S3 speech tokens and a reference waveform, which the speaker timbre is inferred from.
        NOTE: used for sync synthesis only. Please use `S3GenStreamer` for streaming synthesis.
        """
        output_mels = super().forward(
            speech_tokens, speech_token_lens=speech_token_lens, ref_wav=ref_wav,
            ref_sr=ref_sr, ref_dict=ref_dict, finalize=finalize,
            n_cfm_timesteps=n_cfm_timesteps, noised_mels=noised_mels,
        )

        if skip_vocoder:
            return output_mels

        # TODO jrm: ignoring the speed control (mel interpolation) and the HiFTGAN caching mechanisms for now.
        hift_cache_source = torch.zeros(1, 1, 0).to(self.device)

        output_wavs, *_ = self.mel2wav.inference(speech_feat=output_mels, cache_source=hift_cache_source)

        if not self.training:
            # NOTE: ad-hoc method to reduce "spillover" from the reference clip.
            output_wavs[:, :len(self.trim_fade)] *= self.trim_fade

        return output_wavs

    @torch.inference_mode()
    def flow_inference(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        n_cfm_timesteps = None,
        finalize: bool = False,
        speech_token_lens=None,
    ):
        n_cfm_timesteps = n_cfm_timesteps or (2 if self.meanflow else 10)
        noise = None
        if self.meanflow:
            noise = torch.randn(1, 80, speech_tokens.size(-1) * 2, dtype=self.dtype, device=self.device)
        output_mels = super().forward(
            speech_tokens, speech_token_lens=speech_token_lens, ref_wav=ref_wav, ref_sr=ref_sr, ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps, finalize=finalize, noised_mels=noise,
        )
        return output_mels

    @torch.inference_mode()
    def hift_inference(self, speech_feat, cache_source: torch.Tensor = None):
        profile = {}
        if cache_source is None:
            cache_source = torch.zeros(1, 1, 0).to(device=self.device, dtype=self.dtype)
        if _trace_s3_enabled() and _should_trace_event("hift.input"):
            shape_logger.info("[models/s3gen/s3gen.py] hift.input")
            shape_logger.info("  speech_feat %s %s %s", tuple(speech_feat.shape), speech_feat.dtype, speech_feat.device)
            shape_logger.info("  cache_source %s %s %s", tuple(cache_source.shape), cache_source.dtype, cache_source.device)
        hift_start = time.perf_counter()
        output_wavs, output_sources = self.mel2wav.inference(speech_feat=speech_feat, cache_source=cache_source)
        _maybe_sync(self.device)
        profile["s3_hift_s"] = time.perf_counter() - hift_start
        profile["s3_hift_input_batch_size"] = float(int(speech_feat.shape[0]) if speech_feat.ndim >= 1 else 0)
        profile["s3_hift_input_mel_channels"] = float(int(speech_feat.shape[1]) if speech_feat.ndim >= 2 else 0)
        profile["s3_hift_input_mel_frames"] = float(int(speech_feat.shape[-1]) if speech_feat.ndim >= 1 else 0)
        profile["s3_hift_output_samples"] = float(int(output_wavs.shape[-1]) if output_wavs.ndim >= 1 else 0)
        self._set_last_profile(profile)
        if _trace_s3_enabled() and _should_trace_event("hift.output"):
            shape_logger.info("[models/s3gen/s3gen.py] hift.output")
            shape_logger.info("  output_wavs %s %s %s", tuple(output_wavs.shape), output_wavs.dtype, output_wavs.device)
            if output_sources is not None:
                shape_logger.info("  output_sources %s %s %s", tuple(output_sources.shape), output_sources.dtype, output_sources.device)
        return output_wavs, output_sources

    @torch.inference_mode()
    def inference(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        # left as a kwarg because this can change input/output size ratio
        drop_invalid_tokens=True,
        n_cfm_timesteps=None,
        speech_token_lens=None,
    ):
        # hallucination prevention, drop special tokens
        # if drop_invalid_tokens:
        #     speech_tokens, speech_token_lens = drop_invalid(speech_tokens, pad=S3_QUIET_PAD)

        inference_start = time.perf_counter()
        output_mels = self.flow_inference(
            speech_tokens,
            speech_token_lens=speech_token_lens,
            ref_wav=ref_wav,
            ref_sr=ref_sr,
            ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps,
            finalize=True,
        )
        profile = dict(self.get_last_profile() or {})
        output_mels = output_mels.to(dtype=self.dtype) # FIXME (fp16 mode) is this still needed?
        output_wavs, output_sources = self.hift_inference(output_mels, None)
        profile.update(self.get_last_profile() or {})

        # NOTE: ad-hoc method to reduce "spillover" from the reference clip.
        trim_start = time.perf_counter()
        output_wavs[:, :len(self.trim_fade)] *= self.trim_fade
        _maybe_sync(self.device)
        profile["s3_trim_s"] = time.perf_counter() - trim_start
        profile["s3_inference_internal_s"] = time.perf_counter() - inference_start
        self._set_last_profile(profile)

        if _trace_s3_enabled() and _should_trace_event("inference.output"):
            shape_logger.info("[models/s3gen/s3gen.py] inference.output")
            shape_logger.info("  output_mels %s %s %s", tuple(output_mels.shape), output_mels.dtype, output_mels.device)
            shape_logger.info("  output_wavs %s %s %s", tuple(output_wavs.shape), output_wavs.dtype, output_wavs.device)

        return output_wavs, output_sources
