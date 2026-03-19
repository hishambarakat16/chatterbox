from __future__ import annotations

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


class ChatterboxT3ForCausalLM(LlamaForCausalLM):
    """
    First-pass vLLM T3 speech decoder.

    Important limitation:
    - prompt embeddings are computed outside the model
    - generated speech-token positions use an approximate absolute-position
      mapping in this spike, not the original request-relative schedule path
    - Hydra and CFG are intentionally not supported here
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        num_pos = int(getattr(config, "chatterbox_speech_pos_embeddings", 4096))
        self.speech_pos_emb = LearnedPositionEmbeddings(num_pos, config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        if inputs_embeds is None and input_ids is not None:
            speech_pos = positions.clamp_min(0).clamp_max(
                self.speech_pos_emb.emb.num_embeddings - 1
            )
            inputs_embeds = self.model.embed_input_ids(input_ids)
            inputs_embeds = inputs_embeds + self.speech_pos_emb.get_fixed_embedding(speech_pos)
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
