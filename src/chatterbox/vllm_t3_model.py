from __future__ import annotations

import logging
import os
import torch
from torch import nn

from .models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings

try:
    from vllm.config import VllmConfig
    from vllm.model_executor.models.llama import LlamaForCausalLM
    from vllm.model_executor.models.utils import AutoWeightsLoader
    from vllm.sequence import IntermediateTensors
except ImportError as exc:  # pragma: no cover - this file is imported only in a vLLM env.
    raise ImportError(
        "chatterbox.vllm_t3_model requires vLLM to be installed."
    ) from exc


shape_logger = logging.getLogger("chatterbox.shape")


def _trace_shapes() -> bool:
    return bool(os.getenv("CHATTERBOX_TRACE_SHAPES"))


class ChatterboxT3ForCausalLM(LlamaForCausalLM):
    """
    vLLM T3 speech decoder with correct speech-relative position semantics.

    Native T3 inference assigns speech_pos=0 to the BOS token at the end of
    the prompt, and speech_pos=1,2,3,... to each subsequently generated token.
    These are *request-relative* positions, not absolute positions in the
    combined prompt+decode sequence.

    We replicate this by recording the last prompt position during the prefill
    pass (_prefill_max_pos) and subtracting it from vLLM's absolute positions
    during the decode pass so the generated tokens see speech_pos 1, 2, 3, ...

    Hydra and CFG are intentionally not supported here.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        num_pos = int(getattr(config, "chatterbox_speech_pos_embeddings", 4096))
        self.speech_pos_emb = LearnedPositionEmbeddings(num_pos, config.hidden_size)
        # Absolute position of the last prompt token (the extra BOS, speech_pos=0).
        # Set during the prefill forward pass; used to compute speech-relative
        # positions during decode.  Falls back to 0 if prefill is never seen
        # (e.g. a standalone decode-only call, which should not happen in practice).
        self._prefill_max_pos: int = 0

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        if inputs_embeds is not None:
            # Prefill pass: the full prompt embedding is supplied directly.
            # Record the maximum absolute position seen so we can offset decode
            # positions to speech-relative ones.
            self._prefill_max_pos = int(positions.max().item())
            if _trace_shapes():
                shape_logger.info("[vllm_t3_model] prefill pass")
                shape_logger.info("  inputs_embeds %s", tuple(inputs_embeds.shape))
                shape_logger.info("  positions min=%s max=%s (= _prefill_max_pos)", int(positions.min()), self._prefill_max_pos)

        if inputs_embeds is None and input_ids is not None:
            # Decode pass: map absolute vLLM positions → speech-relative positions.
            # The extra BOS token sits at absolute position _prefill_max_pos with
            # speech_pos=0.  The first generated token is at _prefill_max_pos+1,
            # which should therefore get speech_pos=1, and so on.
            num_embeddings = self.speech_pos_emb.emb.num_embeddings
            speech_pos = (positions - self._prefill_max_pos).clamp(1, num_embeddings - 1)
            inputs_embeds = self.model.embed_input_ids(input_ids)
            inputs_embeds = inputs_embeds + self.speech_pos_emb.get_fixed_embedding(speech_pos)
            if _trace_shapes() and int(positions.min()) <= self._prefill_max_pos + 2:
                # Only log the very first decode step to avoid flooding the trace.
                shape_logger.info("[vllm_t3_model] decode step (first)")
                shape_logger.info("  _prefill_max_pos %s", self._prefill_max_pos)
                shape_logger.info("  positions (abs) %s", positions.tolist())
                shape_logger.info("  speech_pos (rel) %s", speech_pos.tolist())
            input_ids = None

        return super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def load_weights(self, weights):
        mapped = []
        for name, tensor in weights:
            if name.startswith(
                (
                    "cond_enc.",
                    "text_emb.",
                    "text_head.",
                    "text_pos_emb.",
                    "tfmr.embed_tokens.",
                )
            ):
                continue
            if name.startswith("tfmr."):
                name = "model." + name[len("tfmr.") :]
            elif name == "speech_emb.weight":
                name = "model.embed_tokens.weight"
            elif name == "speech_head.weight":
                name = "lm_head.weight"
            elif name == "speech_pos_emb.emb.weight":
                name = "speech_pos_emb.emb.weight"
            else:
                continue
            mapped.append((name, tensor))

        loader = AutoWeightsLoader(self)
        return loader.load_weights(mapped)
