from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.t3.t3 import T3

from .medusa_distill import load_multilingual_t3


IGNORE_INDEX = -100
EXPECTED_HIDDEN_SIZE = 1024
EXPECTED_SPEECH_VOCAB = 8194
_DANGLING_QUOTE_RE = re.compile(r'["“”]$')


def _flatten_text_tokens(text_tokens: list[int] | list[list[int]]) -> list[int]:
    if text_tokens and isinstance(text_tokens[0], list):
        assert len(text_tokens) == 1, f"expected single text token row, got {len(text_tokens)}"
        return text_tokens[0]
    return text_tokens  # type: ignore[return-value]


def _resolve_dataset_path(raw_path: str, dataset_dir: Path, *, subdir: str | None = None) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    if subdir is not None:
        fallback = dataset_dir / subdir / path.name
        if fallback.exists():
            return fallback
    fallback = dataset_dir / path.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing dataset sidecar: {raw_path}")


def _describe_value(value):
    if torch.is_tensor(value):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {key: _describe_value(inner) for key, inner in value.items()}
    if value is None:
        return None
    return type(value).__name__


@lru_cache(maxsize=128)
def _load_base_hidden_states_cached(path: str) -> Tensor:
    path_obj = Path(path)
    if path_obj.suffix == ".safetensors":
        payload = load_safetensors(path_obj)
    else:
        payload = torch.load(path_obj, map_location="cpu", weights_only=True)
    if torch.is_tensor(payload):
        hidden_states = payload
    elif isinstance(payload, dict):
        for key in ("base_hidden_states", "speech_hidden", "hidden_states"):
            if key in payload and torch.is_tensor(payload[key]):
                hidden_states = payload[key]
                break
        else:
            raise ValueError(f"Unsupported Hydra hidden-state payload keys: {sorted(payload.keys())}")
    else:
        raise TypeError(f"Unsupported hidden-state payload type: {type(payload).__name__}")

    if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
        hidden_states = hidden_states.squeeze(0)
    assert hidden_states.dim() == 2, tuple(hidden_states.shape)
    return hidden_states


def describe_hydra_hidden_file(path: str | Path) -> dict:
    path_obj = Path(path)
    if path_obj.suffix == ".safetensors":
        payload = load_safetensors(path_obj)
    else:
        payload = torch.load(path_obj, map_location="cpu", weights_only=True)
    summary = {
        "path": str(path),
        "payload_type": type(payload).__name__,
    }
    if isinstance(payload, dict):
        summary["top_level_keys"] = sorted(payload.keys())
        for key in ("base_hidden_states", "speech_hidden", "hidden_states"):
            if key in payload:
                summary[key] = _describe_value(payload[key])
    elif torch.is_tensor(payload):
        summary["tensor"] = _describe_value(payload)
    return summary


@dataclass
class HydraRecord:
    sample_id: str
    text: str
    text_tokens: list[int]
    speech_tokens: list[int]
    num_text_tokens: int
    num_speech_tokens: int
    conditionals_path: Path
    hydra_base_hidden_states_path: Path
    teacher_max_new_tokens: int
    hydra_supervision_len: int
    num_decode_positions: int


class T3HydraDistillDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        limit: int = 0,
        drop_capped: bool = True,
        drop_dangling_quotes: bool = True,
        seed: int = 1337,
        shuffle: bool = True,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        jsonl_paths = sorted(self.dataset_dir.glob("*.jsonl"))
        if not jsonl_paths:
            raise FileNotFoundError(f"No Hydra dataset jsonl files found in {self.dataset_dir}")

        records: list[HydraRecord] = []
        stats = {
            "total_rows": 0,
            "dropped_capped": 0,
            "dropped_dangling_quote": 0,
        }

        for path in jsonl_paths:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    stats["total_rows"] += 1
                    row = json.loads(line)
                    speech_tokens = row["speech_tokens"]
                    teacher_max_new_tokens = int(row["teacher_decode"]["max_new_tokens"])
                    if drop_capped and len(speech_tokens) >= teacher_max_new_tokens:
                        stats["dropped_capped"] += 1
                        continue
                    if drop_dangling_quotes and _DANGLING_QUOTE_RE.search(row["text"]):
                        stats["dropped_dangling_quote"] += 1
                        continue

                    hidden_path = row.get("hydra_base_hidden_states_path") or row.get("base_hidden_states_path")
                    if not hidden_path:
                        raise KeyError("Hydra dataset row is missing hydra_base_hidden_states_path")

                    hydra_supervision_len = int(
                        row.get("hydra_supervision_len", row.get("num_decode_positions", len(speech_tokens)))
                    )
                    num_decode_positions = int(row.get("num_decode_positions", hydra_supervision_len))

                    records.append(
                        HydraRecord(
                            sample_id=row["sample_id"],
                            text=row["text"],
                            text_tokens=_flatten_text_tokens(row["text_tokens"]),
                            speech_tokens=speech_tokens,
                            num_text_tokens=int(row["num_text_tokens"]),
                            num_speech_tokens=int(row["num_speech_tokens"]),
                            conditionals_path=_resolve_dataset_path(
                                row["conditionals_path"],
                                self.dataset_dir,
                                subdir="conditionals",
                            ),
                            hydra_base_hidden_states_path=_resolve_dataset_path(
                                hidden_path,
                                self.dataset_dir,
                                subdir="hydra_base_hidden_states",
                            ),
                            teacher_max_new_tokens=teacher_max_new_tokens,
                            hydra_supervision_len=hydra_supervision_len,
                            num_decode_positions=num_decode_positions,
                        )
                    )

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(records)

        if limit > 0:
            records = records[:limit]

        self.records = records
        self.stats = stats

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> HydraRecord:
        return self.records[index]


def describe_hydra_record(record: HydraRecord) -> dict:
    return {
        "sample_id": record.sample_id,
        "text_preview": record.text[:120],
        "num_text_tokens": record.num_text_tokens,
        "num_speech_tokens": record.num_speech_tokens,
        "hydra_supervision_len": record.hydra_supervision_len,
        "num_decode_positions": record.num_decode_positions,
        "conditionals_path": str(record.conditionals_path),
        "hydra_base_hidden_states_path": str(record.hydra_base_hidden_states_path),
    }


def collate_t3_hydra_batch(records: list[HydraRecord], hp: T3Config):
    text_tensors = [torch.tensor(record.text_tokens, dtype=torch.long) for record in records]
    speech_tensors = [torch.tensor(record.speech_tokens, dtype=torch.long) for record in records]
    hidden_state_tensors = []

    for record in records:
        if record.num_speech_tokens <= 0:
            raise ValueError(f"Sample {record.sample_id} has no speech tokens")
        hidden_states = _load_base_hidden_states_cached(str(record.hydra_base_hidden_states_path))
        if hidden_states.size(-1) != EXPECTED_HIDDEN_SIZE:
            raise ValueError(
                f"Sample {record.sample_id} hidden size mismatch: {tuple(hidden_states.shape)}"
            )
        if hidden_states.size(0) < record.hydra_supervision_len:
            raise ValueError(
                f"Sample {record.sample_id} supervision length exceeds hidden states: "
                f"{record.hydra_supervision_len} > {hidden_states.size(0)}"
            )
        if hidden_states.size(0) != len(record.speech_tokens):
            raise ValueError(
                f"Sample {record.sample_id} hidden/token length mismatch: "
                f"{hidden_states.size(0)} != {len(record.speech_tokens)}"
            )
        hidden_state_tensors.append(hidden_states.to(dtype=torch.float32))

    batch = {
        "sample_ids": [record.sample_id for record in records],
        "texts": [record.text for record in records],
        "text_tokens": pad_sequence(text_tensors, batch_first=True, padding_value=hp.stop_text_token),
        "text_token_lens": torch.tensor([tensor.numel() for tensor in text_tensors], dtype=torch.long),
        "speech_tokens": pad_sequence(speech_tensors, batch_first=True, padding_value=hp.stop_speech_token),
        "speech_token_lens": torch.tensor([tensor.numel() for tensor in speech_tensors], dtype=torch.long),
        "base_hidden_states": pad_sequence(hidden_state_tensors, batch_first=True, padding_value=0.0),
        "hydra_supervision_lens": torch.tensor(
            [record.hydra_supervision_len for record in records],
            dtype=torch.long,
        ),
    }
    return batch


class HydraResBlock(nn.Module):
    def __init__(self, hidden_size: int, num_condition: int = 0):
        super().__init__()
        input_size = hidden_size * (num_condition + 1)
        self.linear = nn.Linear(input_size, hidden_size)
        self.residual = nn.Linear(input_size, hidden_size) if num_condition > 0 else nn.Identity()
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + self.act(self.linear(x))


class T3HydraHeadModel(nn.Module):
    def __init__(
        self,
        t3: T3,
        *,
        hydra_num_heads: int = 2,
        hydra_num_layers: int = 1,
        freeze_base: bool = True,
        grounded_heads: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.t3 = t3
        self.hydra_num_heads = hydra_num_heads
        self.hydra_num_layers = hydra_num_layers
        self.freeze_base = freeze_base
        self.grounded_heads = grounded_heads
        self.dropout_rate = dropout_rate

        hidden_size = t3.cfg.hidden_size
        vocab_size = t3.hp.speech_tokens_dict_size
        if hidden_size != EXPECTED_HIDDEN_SIZE:
            raise ValueError(f"Unexpected T3 hidden size: {hidden_size}")
        if vocab_size != EXPECTED_SPEECH_VOCAB:
            raise ValueError(f"Unexpected T3 speech vocab size: {vocab_size}")

        self.hydra_heads = nn.ModuleList()
        self.hydra_lm_heads = nn.ModuleList()
        for head_index in range(hydra_num_heads):
            layers: list[nn.Module] = []
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(HydraResBlock(hidden_size, num_condition=head_index + 1))
            for _ in range(max(0, hydra_num_layers - 1)):
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                layers.append(HydraResBlock(hidden_size))
            self.hydra_heads.append(nn.Sequential(*layers))

            lm_head = nn.Linear(hidden_size, vocab_size, bias=self.t3.speech_head.bias is not None)
            lm_head.weight.data.copy_(self.t3.speech_head.weight.data)
            if lm_head.bias is not None and self.t3.speech_head.bias is not None:
                lm_head.bias.data.copy_(self.t3.speech_head.bias.data)
            self.hydra_lm_heads.append(lm_head)

        if freeze_base:
            for param in self.t3.parameters():
                param.requires_grad_(False)
            self.t3.eval()

    @property
    def device(self):
        return self.t3.device

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return list(self.hydra_heads.parameters()) + list(self.hydra_lm_heads.parameters())

    def hydra_state_dict(self) -> dict[str, Tensor]:
        model_state = self.state_dict()
        return {
            key: value
            for key, value in model_state.items()
            if key.startswith("hydra_heads.") or key.startswith("hydra_lm_heads.")
        }

    def forward_hydra_heads(
        self,
        *,
        base_hidden_states: Tensor,
        speech_tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        context_manager = torch.no_grad() if self.freeze_base else torch.enable_grad()
        with context_manager:
            speech_embeds = self.t3.speech_emb(speech_tokens)

        hydra_hidden_states = []
        hydra_logits = []
        for head_index, head in enumerate(self.hydra_heads):
            head_inputs = [base_hidden_states]
            for token_offset in range(head_index + 1):
                head_inputs.append(torch.roll(speech_embeds, shifts=-token_offset, dims=1))
            head_input = torch.cat(head_inputs, dim=-1)
            hidden = head(head_input)
            hydra_hidden_states.append(hidden)
            hydra_logits.append(self.hydra_lm_heads[head_index](hidden))
        return torch.stack(hydra_hidden_states, dim=0), torch.stack(hydra_logits, dim=0)

    def forward(
        self,
        *,
        base_hidden_states: Tensor,
        speech_tokens: Tensor,
        speech_token_lens: Tensor,
        trace_shapes: bool = False,
    ):
        assert base_hidden_states.dim() == 3, tuple(base_hidden_states.shape)
        assert speech_tokens.dim() == 2, tuple(speech_tokens.shape)
        assert speech_tokens.size(0) == base_hidden_states.size(0), (
            tuple(base_hidden_states.shape),
            tuple(speech_tokens.shape),
        )
        assert speech_tokens.size(1) == base_hidden_states.size(1), (
            tuple(base_hidden_states.shape),
            tuple(speech_tokens.shape),
        )
        assert base_hidden_states.size(-1) == EXPECTED_HIDDEN_SIZE, tuple(base_hidden_states.shape)

        context_manager = torch.no_grad() if self.freeze_base else torch.enable_grad()
        with context_manager:
            base_logits = self.t3.speech_head(base_hidden_states)

        hydra_hidden_states, hydra_logits = self.forward_hydra_heads(
            base_hidden_states=base_hidden_states,
            speech_tokens=speech_tokens,
        )

        if trace_shapes:
            print("[t3_hydra.train] batch")
            print("  base_hidden_states", tuple(base_hidden_states.shape), base_hidden_states.dtype, base_hidden_states.device)
            print("  speech_tokens", tuple(speech_tokens.shape), speech_tokens.dtype, speech_tokens.device)
            print("  base_logits", tuple(base_logits.shape), base_logits.dtype, base_logits.device)
            print("  hydra_hidden_states", tuple(hydra_hidden_states.shape), hydra_hidden_states.dtype, hydra_hidden_states.device)
            print("  hydra_logits", tuple(hydra_logits.shape), hydra_logits.dtype, hydra_logits.device)

        return {
            "base_logits": base_logits,
            "hydra_hidden_states": hydra_hidden_states,
            "hydra_logits": hydra_logits,
        }

    def compute_loss(
        self,
        *,
        hydra_logits: Tensor,
        hydra_hidden_states: Tensor,
        base_logits: Tensor,
        base_hidden_states: Tensor,
        speech_tokens: Tensor,
        speech_token_lens: Tensor,
        lm_loss_weight: float = 1.0,
        teacher_loss_weight: float = 0.0,
        reconstruction_loss_weight: float = 0.0,
    ):
        assert hydra_logits.dim() == 4, tuple(hydra_logits.shape)
        assert hydra_hidden_states.dim() == 4, tuple(hydra_hidden_states.shape)
        _, batch_size, seq_len, vocab_size = hydra_logits.shape
        assert speech_tokens.shape == (batch_size, seq_len), (
            tuple(hydra_logits.shape),
            tuple(speech_tokens.shape),
        )
        assert base_hidden_states.shape == (batch_size, seq_len, EXPECTED_HIDDEN_SIZE), (
            tuple(base_hidden_states.shape),
            tuple(speech_tokens.shape),
        )

        positions = torch.arange(seq_len, device=speech_tokens.device)[None, :]
        masked_targets = speech_tokens.masked_fill(positions >= speech_token_lens[:, None], IGNORE_INDEX)

        loss_terms = []
        metrics = {}
        with torch.no_grad():
            base_top1 = base_logits.argmax(dim=-1)
            base_mask = masked_targets.ne(IGNORE_INDEX)
            if base_mask.any():
                metrics["base_top1"] = (
                    base_top1[base_mask].eq(masked_targets[base_mask]).float().mean().item()
                )
            else:
                metrics["base_top1"] = 0.0

        for head_index in range(self.hydra_num_heads):
            shift = head_index + 1
            if shift >= seq_len:
                break

            logits = hydra_logits[head_index, :, :-shift, :].contiguous()
            hidden_pred = hydra_hidden_states[head_index, :, :-shift, :].contiguous()
            labels = masked_targets[:, shift:].contiguous()
            teacher_logits = base_logits[:, shift:, :].detach().contiguous()
            hidden_labels = base_hidden_states[:, shift:, :].detach().contiguous()

            lm_loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )

            mask = labels.ne(IGNORE_INDEX)
            if mask.any():
                pred = logits.argmax(dim=-1)
                acc = pred[mask].eq(labels[mask]).float().mean().item()
            else:
                acc = 0.0

            teacher_loss = torch.zeros((), device=logits.device)
            if teacher_loss_weight > 0.0 and mask.any():
                teacher_loss = F.kl_div(
                    F.log_softmax(logits[mask], dim=-1),
                    F.softmax(teacher_logits[mask], dim=-1),
                    reduction="batchmean",
                )

            reconstruct_loss = torch.zeros((), device=logits.device)
            if reconstruction_loss_weight > 0.0 and mask.any():
                reconstruct_loss = F.smooth_l1_loss(hidden_pred[mask], hidden_labels[mask])

            total_head_loss = (
                lm_loss_weight * lm_loss
                + teacher_loss_weight * teacher_loss
                + reconstruction_loss_weight * reconstruct_loss
            )
            loss_terms.append(total_head_loss)

            metrics[f"hydra_head_{head_index}_top1"] = acc
            metrics[f"hydra_head_{head_index}_lm_loss"] = lm_loss.item()
            metrics[f"hydra_head_{head_index}_teacher_loss"] = teacher_loss.item()
            metrics[f"hydra_head_{head_index}_reconstruct_loss"] = reconstruct_loss.item()
            metrics[f"hydra_head_{head_index}_loss"] = total_head_loss.item()

        if not loss_terms:
            raise ValueError("No valid Hydra loss terms were produced for this batch")

        total_loss = torch.stack(loss_terms).mean()
        metrics["loss"] = total_loss.item()
        return total_loss, metrics


def create_t3_hydra_model(
    *,
    device: str,
    checkpoint_dir: str | None = None,
    hydra_num_heads: int = 2,
    hydra_num_layers: int = 1,
    freeze_base: bool = True,
    grounded_heads: bool = True,
    dropout_rate: float = 0.0,
) -> T3HydraHeadModel:
    t3 = load_multilingual_t3(device=device, checkpoint_dir=checkpoint_dir)
    return T3HydraHeadModel(
        t3,
        hydra_num_heads=hydra_num_heads,
        hydra_num_layers=hydra_num_layers,
        freeze_base=freeze_base,
        grounded_heads=grounded_heads,
        dropout_rate=dropout_rate,
    ).to(device)


def save_hydra_checkpoint(
    model: T3HydraHeadModel,
    output_dir: str | Path,
    *,
    step: int,
    extra_state: dict | None = None,
) -> Path:
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / f"checkpoint_step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    save_safetensors(model.hydra_state_dict(), ckpt_dir / "t3_hydra_heads.safetensors")
    config = {
        "hydra_num_heads": model.hydra_num_heads,
        "hydra_num_layers": model.hydra_num_layers,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "speech_vocab_size": EXPECTED_SPEECH_VOCAB,
        "grounded_heads": model.grounded_heads,
        "dropout_rate": model.dropout_rate,
    }
    if extra_state:
        config["training"] = extra_state
    with (ckpt_dir / "t3_hydra_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    return ckpt_dir


def load_hydra_heads_from_checkpoint(
    *,
    base_t3: T3,
    checkpoint_dir: str | Path,
    freeze_base: bool = True,
) -> T3HydraHeadModel:
    checkpoint_dir = Path(checkpoint_dir)
    with (checkpoint_dir / "t3_hydra_config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    model = T3HydraHeadModel(
        base_t3,
        hydra_num_heads=int(config["hydra_num_heads"]),
        hydra_num_layers=int(config["hydra_num_layers"]),
        freeze_base=freeze_base,
        grounded_heads=bool(config.get("grounded_heads", True)),
        dropout_rate=float(config.get("dropout_rate", 0.0)),
    )
    state_dict = load_safetensors(checkpoint_dir / "t3_hydra_heads.safetensors")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected Hydra checkpoint keys: {unexpected}")
    allowed_missing = {key for key in model.state_dict().keys() if not key.startswith("hydra_")}
    remaining_missing = [key for key in missing if key not in allowed_missing]
    if remaining_missing:
        raise ValueError(f"Missing Hydra checkpoint keys: {remaining_missing}")
    return model
