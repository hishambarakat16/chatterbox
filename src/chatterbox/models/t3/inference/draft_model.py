from copy import deepcopy
from dataclasses import dataclass

from torch import nn

from ..t3 import T3


@dataclass(frozen=True)
class DraftModelMetadata:
    mode: str
    num_layers: int
    total_layers: int
    layer_selection: str
    layer_indices: tuple[int, ...]


def select_layer_indices(total_layers: int, num_layers: int, selection: str) -> list[int]:
    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if num_layers > total_layers:
        raise ValueError(f"num_layers={num_layers} exceeds total_layers={total_layers}")

    if selection == "first":
        return list(range(num_layers))
    if selection == "last":
        return list(range(total_layers - num_layers, total_layers))
    if selection == "even":
        if num_layers == 1:
            return [total_layers - 1]
        indices = {
            round(index * (total_layers - 1) / (num_layers - 1))
            for index in range(num_layers)
        }
        if len(indices) != num_layers:
            raise ValueError(
                f"even selection produced duplicates for total_layers={total_layers}, num_layers={num_layers}"
            )
        return sorted(indices)
    raise ValueError(f"unsupported layer selection: {selection}")


def build_layer_subset_draft_model(
    base_t3: T3,
    *,
    num_layers: int,
    layer_selection: str = "even",
) -> tuple[T3, DraftModelMetadata]:
    if base_t3.is_gpt:
        raise NotImplementedError("layer-subset draft builder currently supports the multilingual llama-based T3 only")

    total_layers = len(base_t3.tfmr.layers)
    layer_indices = select_layer_indices(total_layers, num_layers, layer_selection)

    draft_hp = deepcopy(base_t3.hp)
    draft = T3(draft_hp)

    # Reuse the exact token/conditioning interface weights to stay compatible.
    draft.cond_enc = base_t3.cond_enc
    draft.text_emb = base_t3.text_emb
    draft.speech_emb = base_t3.speech_emb
    draft.text_pos_emb = base_t3.text_pos_emb
    draft.speech_pos_emb = base_t3.speech_pos_emb
    draft.text_head = base_t3.text_head
    draft.speech_head = base_t3.speech_head

    if hasattr(base_t3.tfmr, "embed_tokens") and hasattr(draft.tfmr, "embed_tokens"):
        draft.tfmr.embed_tokens = base_t3.tfmr.embed_tokens
    if hasattr(base_t3.tfmr, "norm") and hasattr(draft.tfmr, "norm"):
        draft.tfmr.norm = base_t3.tfmr.norm

    draft.tfmr.layers = nn.ModuleList([base_t3.tfmr.layers[index] for index in layer_indices])
    draft.tfmr.config.num_hidden_layers = len(layer_indices)
    draft.cfg = draft.tfmr.config
    draft.dim = draft.cfg.hidden_size
    draft.eval()

    metadata = DraftModelMetadata(
        mode="layer_subset",
        num_layers=len(layer_indices),
        total_layers=total_layers,
        layer_selection=layer_selection,
        layer_indices=tuple(layer_indices),
    )
    return draft, metadata
