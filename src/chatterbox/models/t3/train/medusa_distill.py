from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from chatterbox.mtl_tts import Conditionals, REPO_ID
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.t3.t3 import T3


IGNORE_INDEX = -100
_DANGLING_QUOTE_RE = re.compile(r'["“”]$')


def _flatten_text_tokens(text_tokens: list[int] | list[list[int]]) -> list[int]:
    if text_tokens and isinstance(text_tokens[0], list):
        assert len(text_tokens) == 1, f"expected single text token row, got {len(text_tokens)}"
        return text_tokens[0]
    return text_tokens  # type: ignore[return-value]


def _resolve_conditionals_path(raw_path: str, dataset_dir: Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    fallback = dataset_dir / "conditionals" / path.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing conditionals file: {raw_path} (fallback {fallback})")


@lru_cache(maxsize=64)
def _load_t3_cond_cached(path: str) -> T3Cond:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "t3" in payload and "gen" in payload:
        return Conditionals.load(path, map_location="cpu").t3
    return T3Cond(**payload)


def _normalize_cond_tensor(tensor: Tensor) -> Tensor:
    if tensor.dim() > 0 and tensor.size(0) == 1:
        return tensor.squeeze(0)
    return tensor


def _stack_cond_field(values: list[Tensor | None], field_name: str) -> Tensor | None:
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"Mixed None/non-None values for T3Cond.{field_name}")

    normalized = [_normalize_cond_tensor(value) for value in values if value is not None]
    first_shape = normalized[0].shape
    if any(tensor.shape != first_shape for tensor in normalized):
        raise ValueError(
            f"Mismatched shapes for T3Cond.{field_name}: {[tuple(t.shape) for t in normalized]}"
        )
    return torch.stack(normalized, dim=0)


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


def stack_t3_conds(conds: list[T3Cond]) -> T3Cond:
    return T3Cond(
        speaker_emb=_stack_cond_field([cond.speaker_emb for cond in conds], "speaker_emb"),
        clap_emb=_stack_cond_field([cond.clap_emb for cond in conds], "clap_emb"),
        cond_prompt_speech_tokens=_stack_cond_field(
            [cond.cond_prompt_speech_tokens for cond in conds], "cond_prompt_speech_tokens"
        ),
        cond_prompt_speech_emb=_stack_cond_field(
            [cond.cond_prompt_speech_emb for cond in conds], "cond_prompt_speech_emb"
        ),
        emotion_adv=_stack_cond_field([cond.emotion_adv for cond in conds], "emotion_adv"),
    )


@dataclass
class DistillRecord:
    sample_id: str
    text: str
    text_tokens: list[int]
    speech_tokens: list[int]
    num_text_tokens: int
    num_speech_tokens: int
    conditionals_path: Path
    teacher_max_new_tokens: int


class T3MedusaDistillDataset(Dataset):
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
            raise FileNotFoundError(f"No shard jsonl files found in {self.dataset_dir}")

        records: list[DistillRecord] = []
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

                    records.append(
                        DistillRecord(
                            sample_id=row["sample_id"],
                            text=row["text"],
                            text_tokens=_flatten_text_tokens(row["text_tokens"]),
                            speech_tokens=speech_tokens,
                            num_text_tokens=int(row["num_text_tokens"]),
                            num_speech_tokens=int(row["num_speech_tokens"]),
                            conditionals_path=_resolve_conditionals_path(row["conditionals_path"], self.dataset_dir),
                            teacher_max_new_tokens=teacher_max_new_tokens,
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

    def __getitem__(self, index: int) -> DistillRecord:
        return self.records[index]


def describe_conditionals_file(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    summary = {
        "path": str(path),
        "payload_type": type(payload).__name__,
    }
    if isinstance(payload, dict):
        summary["top_level_keys"] = sorted(payload.keys())
        if "t3" in payload:
            summary["t3"] = _describe_value(payload["t3"])
        if "gen" in payload:
            summary["gen"] = _describe_value(payload["gen"])
    return summary


def describe_distill_record(record: DistillRecord) -> dict:
    return {
        "sample_id": record.sample_id,
        "text_preview": record.text[:120],
        "num_text_tokens": record.num_text_tokens,
        "num_speech_tokens": record.num_speech_tokens,
        "teacher_max_new_tokens": record.teacher_max_new_tokens,
        "conditionals_path": str(record.conditionals_path),
    }


def collate_t3_medusa_batch(records: list[DistillRecord], hp: T3Config):
    text_tensors = [torch.tensor(record.text_tokens, dtype=torch.long) for record in records]
    speech_tensors = []
    for record in records:
        tokens = record.speech_tokens
        ended_naturally = len(tokens) < record.teacher_max_new_tokens
        if ended_naturally and (not tokens or tokens[-1] != hp.stop_speech_token):
            tokens = [*tokens, hp.stop_speech_token]
        speech_tensors.append(torch.tensor(tokens, dtype=torch.long))
    conds = [_load_t3_cond_cached(str(record.conditionals_path)) for record in records]

    batch = {
        "sample_ids": [record.sample_id for record in records],
        "texts": [record.text for record in records],
        "text_tokens": pad_sequence(text_tensors, batch_first=True, padding_value=hp.stop_text_token),
        "text_token_lens": torch.tensor([tensor.numel() for tensor in text_tensors], dtype=torch.long),
        "speech_tokens": pad_sequence(speech_tensors, batch_first=True, padding_value=hp.stop_speech_token),
        "speech_token_lens": torch.tensor([tensor.numel() for tensor in speech_tensors], dtype=torch.long),
        "t3_cond": stack_t3_conds(conds),
    }
    return batch


class ResBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.act(self.linear(x))


class T3MedusaHeadModel(nn.Module):
    def __init__(
        self,
        t3: T3,
        *,
        medusa_num_heads: int = 4,
        medusa_num_layers: int = 1,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.t3 = t3
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.freeze_base = freeze_base

        hidden_size = t3.cfg.hidden_size
        self.medusa_heads = nn.ModuleList(
            [
                nn.Sequential(*(ResBlock(hidden_size) for _ in range(medusa_num_layers)))
                for _ in range(medusa_num_heads)
            ]
        )

        if freeze_base:
            for param in self.t3.parameters():
                param.requires_grad_(False)
            self.t3.eval()

    @property
    def device(self):
        return self.t3.device

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return self.medusa_heads.parameters()

    def forward_medusa_heads(self, hidden_states: Tensor) -> Tensor:
        logits = []
        for head in self.medusa_heads:
            medusa_hidden = head(hidden_states)
            logits.append(self.t3.speech_head(medusa_hidden))
        return torch.stack(logits, dim=0)

    def forward(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: Tensor,
        speech_tokens: Tensor,
        speech_token_lens: Tensor,
        trace_shapes: bool = False,
    ):
        assert text_tokens.dim() == 2, tuple(text_tokens.shape)
        assert speech_tokens.dim() == 2, tuple(speech_tokens.shape)
        assert speech_tokens.size(0) == text_tokens.size(0), (
            tuple(text_tokens.shape),
            tuple(speech_tokens.shape),
        )

        bos = torch.full(
            (speech_tokens.size(0), 1),
            fill_value=self.t3.hp.start_speech_token,
            dtype=torch.long,
            device=speech_tokens.device,
        )
        speech_inputs = torch.cat([bos, speech_tokens[:, :-1]], dim=1)
        assert speech_inputs.shape == speech_tokens.shape

        context_manager = torch.no_grad() if self.freeze_base else torch.enable_grad()
        with context_manager:
            embeds, len_cond = self.t3.prepare_input_embeds(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                speech_tokens=speech_inputs,
            )
            tfmr_out = self.t3.tfmr(
                input_ids=None,
                inputs_embeds=embeds,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
            hidden_states = tfmr_out.last_hidden_state
            speech_start = len_cond + text_tokens.size(1)
            speech_hidden = hidden_states[:, speech_start : speech_start + speech_tokens.size(1)]
            base_logits = self.t3.speech_head(speech_hidden)

        medusa_logits = self.forward_medusa_heads(speech_hidden)

        if trace_shapes:
            print("[t3_medusa.train] batch")
            print("  text_tokens", tuple(text_tokens.shape), text_tokens.dtype, text_tokens.device)
            print("  speech_tokens", tuple(speech_tokens.shape), speech_tokens.dtype, speech_tokens.device)
            print("  speech_inputs", tuple(speech_inputs.shape), speech_inputs.dtype, speech_inputs.device)
            print("  hidden_states", tuple(hidden_states.shape), hidden_states.dtype, hidden_states.device)
            print("  speech_hidden", tuple(speech_hidden.shape), speech_hidden.dtype, speech_hidden.device)
            print("  base_logits", tuple(base_logits.shape), base_logits.dtype, base_logits.device)
            print("  medusa_logits", tuple(medusa_logits.shape), medusa_logits.dtype, medusa_logits.device)

        return {
            "base_logits": base_logits,
            "medusa_logits": medusa_logits,
        }

    def compute_loss(
        self,
        *,
        medusa_logits: Tensor,
        base_logits: Tensor,
        speech_tokens: Tensor,
        speech_token_lens: Tensor,
    ):
        assert medusa_logits.dim() == 4, tuple(medusa_logits.shape)
        _, batch_size, seq_len, vocab_size = medusa_logits.shape
        assert speech_tokens.shape == (batch_size, seq_len), (
            tuple(medusa_logits.shape),
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

        for head_index in range(self.medusa_num_heads):
            shift = head_index + 1
            if shift >= seq_len:
                break
            logits = medusa_logits[head_index, :, :-shift, :].contiguous()
            labels = masked_targets[:, shift:].contiguous()
            loss_i = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
            loss_terms.append(loss_i)

            with torch.no_grad():
                mask = labels.ne(IGNORE_INDEX)
                if mask.any():
                    pred = logits.argmax(dim=-1)
                    acc = pred[mask].eq(labels[mask]).float().mean().item()
                else:
                    acc = 0.0
            metrics[f"medusa_head_{head_index}_top1"] = acc
            metrics[f"medusa_head_{head_index}_loss"] = loss_i.item()

        if not loss_terms:
            raise ValueError("No valid Medusa loss terms were produced for this batch")

        total_loss = torch.stack(loss_terms).mean()
        metrics["loss"] = total_loss.item()
        return total_loss, metrics


def load_multilingual_t3(device: str, checkpoint_dir: str | None = None) -> T3:
    if checkpoint_dir:
        ckpt_dir = Path(checkpoint_dir)
    else:
        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main",
                allow_patterns=["t3_mtl23ls_v2.safetensors"],
            )
        )

    t3 = T3(T3Config.multilingual())
    state = load_safetensors(ckpt_dir / "t3_mtl23ls_v2.safetensors")
    if "model" in state:
        state = state["model"][0]
    t3.load_state_dict(state)
    t3.to(device).eval()
    return t3


def create_t3_medusa_model(
    *,
    device: str,
    checkpoint_dir: str | None,
    medusa_num_heads: int,
    medusa_num_layers: int,
    freeze_base: bool = True,
) -> T3MedusaHeadModel:
    t3 = load_multilingual_t3(device=device, checkpoint_dir=checkpoint_dir)
    model = T3MedusaHeadModel(
        t3,
        medusa_num_heads=medusa_num_heads,
        medusa_num_layers=medusa_num_layers,
        freeze_base=freeze_base,
    )
    model.to(device)
    return model


def load_medusa_heads_from_checkpoint(
    *,
    base_t3: T3,
    checkpoint_dir: str | Path,
    freeze_base: bool = True,
) -> T3MedusaHeadModel:
    checkpoint_dir = Path(checkpoint_dir)
    config = json.loads((checkpoint_dir / "t3_medusa_config.json").read_text(encoding="utf-8"))
    model = T3MedusaHeadModel(
        base_t3,
        medusa_num_heads=int(config["medusa_num_heads"]),
        medusa_num_layers=int(config["medusa_num_layers"]),
        freeze_base=freeze_base,
    )
    state = load_safetensors(checkpoint_dir / "t3_medusa_heads.safetensors")
    model.medusa_heads.load_state_dict(state, strict=True)
    model.to(base_t3.device).eval()
    return model


def save_medusa_checkpoint(
    model: T3MedusaHeadModel,
    output_dir: str | Path,
    *,
    step: int,
    extra_state: dict | None = None,
) -> Path:
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / f"checkpoint_step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    medusa_state = {k: v.detach().cpu() for k, v in model.medusa_heads.state_dict().items()}
    save_safetensors(medusa_state, ckpt_dir / "t3_medusa_heads.safetensors")

    config = {
        "medusa_num_heads": model.medusa_num_heads,
        "medusa_num_layers": model.medusa_num_layers,
        "hidden_size": model.t3.cfg.hidden_size,
        "speech_vocab_size": model.t3.hp.speech_tokens_dict_size,
        "freeze_base": model.freeze_base,
        "step": step,
    }
    if extra_state:
        config.update(extra_state)
    (ckpt_dir / "t3_medusa_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ckpt_dir
