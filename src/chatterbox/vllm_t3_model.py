from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import nn

from .models.t3.modules.cond_enc import T3CondEnc
from .models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings
from .models.t3.modules.t3_config import T3Config
from .vllm_t3_bridge import get_conditioning_seq_len, get_vllm_prompt_layout

try:
    from vllm.config import VllmConfig
    from vllm.config.multimodal import BaseDummyOptions
    from vllm.model_executor.models.llama import LlamaForCausalLM
    from vllm.model_executor.models.utils import AutoWeightsLoader, _merge_multimodal_embeddings
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.inputs import MultiModalDataDict, MultiModalFieldConfig, MultiModalEmbeddings
    from vllm.multimodal.parse import DictEmbeddingItems, MultiModalDataItems, MultiModalDataParser
    from vllm.multimodal.processing import (
        BaseDummyInputsBuilder,
        BaseMultiModalProcessor,
        BaseProcessingInfo,
        PromptReplacement,
        PromptUpdate,
    )
    from vllm.sequence import IntermediateTensors
except ImportError as exc:  # pragma: no cover - this file is imported only in a vLLM env.
    raise ImportError(
        "chatterbox.vllm_t3_model requires vLLM to be installed."
    ) from exc


HP = T3Config.multilingual()
PROMPT_LAYOUT = get_vllm_prompt_layout(HP)
TEXT_TOKEN_OFFSET = int(PROMPT_LAYOUT["text_token_offset"])
CONDITIONING_TOKEN_ID = int(PROMPT_LAYOUT["conditioning_token_id"])
CONDITIONING_SEQ_LEN = int(PROMPT_LAYOUT["conditioning_seq_len"])


def _conditioning_field_config(_: Mapping[str, torch.Tensor]) -> Mapping[str, MultiModalFieldConfig]:
    return {
        "speaker_emb": MultiModalFieldConfig.batched("conditioning"),
        "cond_prompt_speech_tokens": MultiModalFieldConfig.batched("conditioning"),
        "emotion_adv": MultiModalFieldConfig.batched("conditioning"),
    }


class ChatterboxT3ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"conditioning": 1}

    def get_data_parser(self) -> MultiModalDataParser:
        return ChatterboxT3MultiModalDataParser(
            expected_hidden_size=self._get_expected_hidden_size(),
        )


class ChatterboxT3MultiModalDataParser(MultiModalDataParser):
    def _parse_conditioning_data(
        self,
        data,
    ):
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValueError(
                "Chatterbox conditioning inputs must be provided as a dictionary of tensors."
            )
        return DictEmbeddingItems(
            data,
            modality="conditioning",
            required_fields={"speaker_emb", "cond_prompt_speech_tokens", "emotion_adv"},
            fields_factory=_conditioning_field_config,
        )

    def _get_subparsers(self):
        return {
            **super()._get_subparsers(),
            "conditioning": self._parse_conditioning_data,
        }


class ChatterboxT3DummyInputsBuilder(BaseDummyInputsBuilder[ChatterboxT3ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        del mm_counts
        cond_tok = f"tok_{CONDITIONING_TOKEN_ID}"
        text_sot = f"tok_{TEXT_TOKEN_OFFSET + int(HP.start_text_token)}"
        text_eot = f"tok_{TEXT_TOKEN_OFFSET + int(HP.stop_text_token)}"
        speech_bos = f"tok_{int(HP.start_speech_token)}"
        return " ".join([cond_tok, text_sot, text_eot, speech_bos])

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        del seq_len, mm_options
        num_items = max(1, int(mm_counts.get("conditioning", 0) or 0))
        return {
            "conditioning": {
                "speaker_emb": torch.zeros((num_items, 1, HP.speaker_embed_size), dtype=torch.float32),
                "cond_prompt_speech_tokens": torch.zeros(
                    (num_items, int(HP.speech_cond_prompt_len)),
                    dtype=torch.long,
                ),
                "emotion_adv": 0.5 * torch.ones((num_items, 1, 1), dtype=torch.float32),
            }
        }


class ChatterboxT3MultiModalProcessor(
    BaseMultiModalProcessor[ChatterboxT3ProcessingInfo]
):
    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, torch.Tensor],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        del hf_processor_mm_kwargs
        return _conditioning_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs,
    ) -> list[PromptUpdate]:
        del mm_items, hf_processor_mm_kwargs, out_mm_kwargs
        return [
            PromptReplacement(
                modality="conditioning",
                target=[CONDITIONING_TOKEN_ID],
                replacement=[CONDITIONING_TOKEN_ID] * CONDITIONING_SEQ_LEN,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    ChatterboxT3MultiModalProcessor,
    info=ChatterboxT3ProcessingInfo,
    dummy_inputs=ChatterboxT3DummyInputsBuilder,
)
class ChatterboxT3ForCausalLM(LlamaForCausalLM):
    """
    vLLM T3 speech decoder that reconstructs the prompt inside the served model.

    Design choices in this iteration:
    - the worker passes token ids plus conditioning tensors
    - the served model rebuilds cond/text/speech prompt embeddings internally
    - decode-side speech positions still use the existing approximate absolute
      mapping for generated speech tokens
    - Hydra and CFG are intentionally not supported here
    """

    supports_multimodal = True
    requires_raw_input_tokens = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        del i
        if modality != "conditioning":
            raise ValueError(f"Unsupported modality: {modality}")
        return f"tok_{CONDITIONING_TOKEN_ID}"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise NotImplementedError(
                "ChatterboxT3ForCausalLM currently supports tensor_parallel_size=1 only."
            )

        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config

        self.hp = T3Config.multilingual()
        self.speech_vocab_size = int(getattr(config, "chatterbox_speech_vocab_size", self.hp.speech_tokens_dict_size))
        self.text_vocab_size = int(getattr(config, "chatterbox_text_vocab_size", self.hp.text_tokens_dict_size))
        self.text_token_offset = int(getattr(config, "chatterbox_text_token_offset", TEXT_TOKEN_OFFSET))
        self.text_token_end = int(self.text_token_offset + self.text_vocab_size)
        self.conditioning_token_id = int(
            getattr(config, "chatterbox_conditioning_token_id", CONDITIONING_TOKEN_ID)
        )
        self.conditioning_seq_len = int(
            getattr(config, "chatterbox_conditioning_seq_len", get_conditioning_seq_len(self.hp))
        )

        text_pos = int(getattr(config, "chatterbox_text_pos_embeddings", self.hp.max_text_tokens + 4))
        speech_pos = int(getattr(config, "chatterbox_speech_pos_embeddings", self.hp.max_speech_tokens + 4))

        self.text_emb = nn.Embedding(self.text_vocab_size, config.hidden_size)
        self.text_pos_emb = LearnedPositionEmbeddings(text_pos, config.hidden_size)
        self.speech_pos_emb = LearnedPositionEmbeddings(speech_pos, config.hidden_size)
        self.cond_enc = T3CondEnc(self.hp)

        with torch.no_grad():
            self.model.embed_tokens.weight.zero_()
            self.lm_head.weight.zero_()

    def _is_text_token(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (input_ids >= self.text_token_offset) & (input_ids < self.text_token_end)

    def _is_conditioning_token(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids == self.conditioning_token_id

    def _is_speech_token(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (input_ids >= 0) & (input_ids < self.speech_vocab_size)

    def _embed_text_tokens(self, text_ids: torch.Tensor) -> torch.Tensor:
        text_emb = self.text_emb(text_ids)
        text_pos = torch.arange(text_ids.shape[0], device=text_ids.device, dtype=torch.long)
        text_pos_emb = self.text_pos_emb.get_fixed_embedding(text_pos).reshape(-1, text_emb.shape[-1])
        return text_emb + text_pos_emb.to(dtype=text_emb.dtype)

    def _embed_speech_tokens(self, speech_ids: torch.Tensor, speech_pos: torch.Tensor) -> torch.Tensor:
        speech_emb = self.model.embed_input_ids(speech_ids)
        speech_pos_emb = self.speech_pos_emb.get_fixed_embedding(speech_pos).reshape(
            -1, speech_emb.shape[-1]
        )
        return speech_emb + speech_pos_emb.to(dtype=speech_emb.dtype)

    def _segment_ranges(self, positions: torch.Tensor) -> list[tuple[int, int]]:
        if positions.numel() == 0:
            return []
        starts = [0]
        for idx in range(1, positions.shape[0]):
            if int(positions[idx].item()) <= int(positions[idx - 1].item()):
                starts.append(idx)
        starts.append(int(positions.shape[0]))
        return list(zip(starts[:-1], starts[1:]))

    def _build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        full_embeds = inputs_embeds.clone()

        for start, end in self._segment_ranges(positions):
            seg_ids = input_ids[start:end]
            seg_pos = positions[start:end]
            seg_embeds = full_embeds[start:end]

            cond_mask = self._is_conditioning_token(seg_ids)
            text_mask = self._is_text_token(seg_ids)
            speech_mask = self._is_speech_token(seg_ids) & ~cond_mask

            # Decode-only segment: keep the current approximate absolute
            # speech-position mapping used by the original vLLM spike.
            if not bool(cond_mask.any() or text_mask.any()):
                if bool(speech_mask.any()):
                    speech_ids = seg_ids[speech_mask]
                    speech_pos = seg_pos[speech_mask].clamp_min(0).clamp_max(
                        self.speech_pos_emb.emb.num_embeddings - 1
                    )
                    seg_embeds[speech_mask] = self._embed_speech_tokens(speech_ids, speech_pos)
                full_embeds[start:end] = seg_embeds
                continue

            if bool(text_mask.any()):
                text_ids = (seg_ids[text_mask] - self.text_token_offset).to(dtype=torch.long)
                seg_embeds[text_mask] = self._embed_text_tokens(text_ids)

            if bool(speech_mask.any()):
                speech_ids = seg_ids[speech_mask].to(dtype=torch.long)
                rel_speech_pos = torch.arange(
                    speech_ids.shape[0],
                    device=speech_ids.device,
                    dtype=torch.long,
                )
                seg_embeds[speech_mask] = self._embed_speech_tokens(speech_ids, rel_speech_pos)

            full_embeds[start:end] = seg_embeds

        return full_embeds

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        speaker_emb = kwargs.get("speaker_emb")
        if speaker_emb is None:
            return []

        speaker_emb = torch.as_tensor(speaker_emb, device=self.text_emb.weight.device)
        speaker_emb = speaker_emb.view(-1, self.hp.speaker_embed_size).to(
            dtype=self.text_emb.weight.dtype
        )
        batch_size = int(speaker_emb.shape[0])

        cond_prompt_speech_tokens = kwargs.get("cond_prompt_speech_tokens")
        if cond_prompt_speech_tokens is None:
            cond_prompt_speech_tokens = torch.empty(
                (batch_size, 0),
                dtype=torch.long,
                device=self.text_emb.weight.device,
            )
        else:
            cond_prompt_speech_tokens = torch.as_tensor(
                cond_prompt_speech_tokens,
                device=self.text_emb.weight.device,
                dtype=torch.long,
            ).view(batch_size, -1)

        emotion_adv = kwargs.get("emotion_adv")
        if emotion_adv is None:
            emotion_adv = torch.zeros(
                (batch_size, 1, 1),
                dtype=self.text_emb.weight.dtype,
                device=self.text_emb.weight.device,
            )
        else:
            emotion_adv = torch.as_tensor(
                emotion_adv,
                device=self.text_emb.weight.device,
                dtype=self.text_emb.weight.dtype,
            ).view(batch_size, 1, 1)

        cond_spkr = self.cond_enc.spkr_enc(speaker_emb)[:, None]
        empty = torch.zeros_like(cond_spkr[:, :0])

        if cond_prompt_speech_tokens.shape[1] == 0:
            cond_prompt_speech_emb = empty
        else:
            cond_prompt_speech_emb = self.model.embed_tokens(cond_prompt_speech_tokens)
            cond_prompt_pos = torch.arange(
                cond_prompt_speech_tokens.shape[1],
                device=cond_prompt_speech_tokens.device,
                dtype=torch.long,
            )
            cond_prompt_pos_emb = self.speech_pos_emb.get_fixed_embedding(cond_prompt_pos).reshape(
                1,
                cond_prompt_speech_tokens.shape[1],
                -1,
            )
            cond_prompt_speech_emb = cond_prompt_speech_emb + cond_prompt_pos_emb.to(
                dtype=cond_prompt_speech_emb.dtype
            )
            if self.cond_enc.perceiver is not None:
                cond_prompt_speech_emb = self.cond_enc.perceiver(cond_prompt_speech_emb)

        cond_emotion_adv = empty
        if self.hp.emotion_adv:
            cond_emotion_adv = self.cond_enc.emotion_adv_fc(emotion_adv)

        cond_embeds = torch.cat(
            (
                cond_spkr,
                empty,
                cond_prompt_speech_emb,
                cond_emotion_adv,
            ),
            dim=1,
        )
        return list(cond_embeds.unbind(0))

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        safe_input_ids = torch.where(
            self._is_speech_token(input_ids),
            input_ids,
            torch.zeros_like(input_ids),
        )
        inputs_embeds = self.model.embed_input_ids(safe_input_ids)

        if multimodal_embeddings is None or is_multimodal is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        if input_ids is None:
            return super().forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)

        rebuilt_inputs_embeds = self._build_inputs_embeds(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )
        return super().forward(
            input_ids=None,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=rebuilt_inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            return None
        return logits[..., : self.speech_vocab_size]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapped = []
        embed_weight = self.model.embed_tokens.weight
        lm_head_weight = self.lm_head.weight

        with torch.no_grad():
            for name, tensor in weights:
                if name.startswith("tfmr.embed_tokens."):
                    continue
                if name.startswith("tfmr."):
                    name = "model." + name[len("tfmr.") :]
                    mapped.append((name, tensor))
                    continue
                if name == "speech_emb.weight":
                    embed_weight[: tensor.shape[0]].copy_(tensor)
                    continue
                if name == "speech_head.weight":
                    lm_head_weight[: tensor.shape[0]].copy_(tensor)
                    continue
                if name in (
                    "text_emb.weight",
                    "text_pos_emb.emb.weight",
                    "speech_pos_emb.emb.weight",
                ) or name.startswith("cond_enc."):
                    mapped.append((name, tensor))

            if embed_weight.shape[0] > self.speech_vocab_size:
                embed_weight[self.speech_vocab_size :].zero_()
            if lm_head_weight.shape[0] > self.speech_vocab_size:
                lm_head_weight[self.speech_vocab_size :].zero_()

        loader = AutoWeightsLoader(self)
        return loader.load_weights(mapped)
