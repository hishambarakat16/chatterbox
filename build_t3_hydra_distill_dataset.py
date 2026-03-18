import argparse
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from safetensors.torch import save_file as save_safetensors
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal envs
    tqdm = None

from chatterbox.mtl_tts import Conditionals
from chatterbox.mtl_tts_scheduled import ChatterboxMultilingualScheduledTTS
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.runtime.session import clone_t3_cond


_DANGLING_QUOTE_RE = re.compile(r'["“”]$')


def load_model(
    device: str,
    checkpoint_dir: str | None,
    *,
    enable_alignment_controller: bool = False,
):
    if checkpoint_dir:
        return ChatterboxMultilingualScheduledTTS.from_local(
            checkpoint_dir,
            device,
            enable_alignment_controller=enable_alignment_controller,
        )
    return ChatterboxMultilingualScheduledTTS.from_pretrained(
        device,
        enable_alignment_controller=enable_alignment_controller,
    )


def _flatten_text_tokens(text_tokens: list[int] | list[list[int]]) -> list[int]:
    if text_tokens and isinstance(text_tokens[0], list):
        assert len(text_tokens) == 1, f"expected single text token row, got {len(text_tokens)}"
        return text_tokens[0]
    return text_tokens  # type: ignore[return-value]


def _flatten_speech_tokens(speech_tokens: list[int] | list[list[int]]) -> list[int]:
    if speech_tokens and isinstance(speech_tokens[0], list):
        assert len(speech_tokens) == 1, f"expected single speech token row, got {len(speech_tokens)}"
        return speech_tokens[0]
    return speech_tokens  # type: ignore[return-value]


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


def _load_all_source_rows(source_dataset_dir: Path) -> list[dict]:
    jsonl_paths = sorted(source_dataset_dir.glob("*.jsonl"))
    if not jsonl_paths:
        raise FileNotFoundError(f"No shard jsonl files found in {source_dataset_dir}")

    rows: list[dict] = []
    for path in jsonl_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def _validate_text_tokens(text_tokens: list[int], hp) -> None:
    if not text_tokens:
        raise ValueError("text_tokens must be non-empty")
    if text_tokens[0] != hp.start_text_token:
        raise ValueError(f"missing start_text_token: got {text_tokens[0]}, expected {hp.start_text_token}")
    if text_tokens[-1] != hp.stop_text_token:
        raise ValueError(f"missing stop_text_token: got {text_tokens[-1]}, expected {hp.stop_text_token}")


def _validate_speech_tokens(speech_tokens: list[int], hp) -> None:
    if not speech_tokens:
        raise ValueError("speech_tokens must be non-empty")
    bad = [token for token in speech_tokens if token < 0 or token >= hp.speech_tokens_dict_size]
    if bad:
        raise ValueError(f"speech_tokens contain invalid ids: {bad[:8]}")


def _count_existing_rows(jsonl_path: Path) -> int:
    if not jsonl_path.exists():
        return 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a Chatterbox-native Hydra distillation dataset from an existing Medusa-style T3 corpus."
    )
    parser.add_argument("--source-dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jsonl-stem", default="samples")
    parser.add_argument(
        "--sidecar-subdir",
        default="hydra_base_hidden_states",
        help="Directory inside output-dir used for per-sample hidden-state sidecars.",
    )
    parser.add_argument(
        "--sidecar-dtype",
        choices=("fp16", "bf16", "fp32"),
        default="fp16",
        help="On-disk dtype for saved hidden-state sidecars.",
    )
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--enable-alignment-controller", action="store_true")
    parser.add_argument("--trace-shapes", action="store_true")
    return parser


@dataclass
class HydraSourceRecord:
    sample_index: int
    sample_id: str
    text: str
    normalized_text: str
    text_tokens: list[int]
    speech_tokens: list[int]
    conditionals_path: Path
    teacher_max_new_tokens: int
    source_wav_path: str
    source_duration: str
    raw_row: dict


def _load_records(args: argparse.Namespace, hp) -> list[HydraSourceRecord]:
    source_dataset_dir = Path(args.source_dataset_dir)
    rows = _load_all_source_rows(source_dataset_dir)

    if args.offset > 0:
        rows = rows[args.offset :]
    if args.limit > 0:
        rows = rows[: args.limit]

    records: list[HydraSourceRecord] = []
    dropped_capped = 0
    dropped_dangling_quote = 0

    for index, row in enumerate(rows):
        text_tokens = _flatten_text_tokens(row["text_tokens"])
        speech_tokens = _flatten_speech_tokens(row["speech_tokens"])
        teacher_decode = row.get("teacher_decode") or {}
        teacher_max_new_tokens = int(teacher_decode.get("max_new_tokens", len(speech_tokens)))

        _validate_text_tokens(text_tokens, hp)
        _validate_speech_tokens(speech_tokens, hp)

        if len(speech_tokens) >= teacher_max_new_tokens:
            dropped_capped += 1
            continue
        if _DANGLING_QUOTE_RE.search(row["text"]):
            dropped_dangling_quote += 1
            continue

        conditionals_path = _resolve_conditionals_path(row["conditionals_path"], source_dataset_dir)
        records.append(
            HydraSourceRecord(
                sample_index=index,
                sample_id=row.get("sample_id") or f"sample_{index:06d}",
                text=row["text"],
                normalized_text=row.get("normalized_text", row["text"]),
                text_tokens=text_tokens,
                speech_tokens=speech_tokens,
                conditionals_path=conditionals_path,
                teacher_max_new_tokens=teacher_max_new_tokens,
                source_wav_path=row.get("source_wav_path", ""),
                source_duration=row.get("source_duration", row.get("duration", "")),
                raw_row=row,
            )
        )

    print(
        "source_stats="
        + json.dumps(
            {
                "total_rows": len(rows),
                "kept_rows": len(records),
                "dropped_capped": dropped_capped,
                "dropped_dangling_quote": dropped_dangling_quote,
            }
        )
    )
    return records


def _sidecar_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def _save_sidecar(path: Path, hidden_states: torch.Tensor, dtype: torch.dtype) -> None:
    payload = {"base_hidden_states": hidden_states.to(dtype=dtype).contiguous().cpu()}
    save_safetensors(payload, str(path))


def _build_hydra_record(
    record: HydraSourceRecord,
    *,
    sidecar_path: Path,
    hidden_states: torch.Tensor,
    hidden_dtype: str,
) -> dict:
    return {
        "sample_index": record.sample_index,
        "sample_id": record.sample_id,
        "text": record.text,
        "normalized_text": record.normalized_text,
        "language_id": "ar",
        "audio_prompt_path": str(Path(record.raw_row.get("audio_prompt_path", "")).resolve())
        if record.raw_row.get("audio_prompt_path")
        else "",
        "conditionals_path": str(record.conditionals_path.resolve()),
        "text_tokens": record.text_tokens,
        "speech_tokens": record.speech_tokens,
        "num_text_tokens": len(record.text_tokens),
        "num_speech_tokens": len(record.speech_tokens),
        "hydra_base_hidden_states_path": str(sidecar_path.resolve()),
        "hydra_supervision_len": int(hidden_states.shape[0]),
        "num_decode_positions": int(hidden_states.shape[0]),
        "hydra_hidden_size": int(hidden_states.shape[-1]),
        "hydra_hidden_state_dtype": hidden_dtype,
        "teacher_max_new_tokens": record.teacher_max_new_tokens,
        "teacher_decode": record.raw_row.get("teacher_decode", {}),
        "source_wav_path": record.source_wav_path,
        "source_duration": record.source_duration,
    }


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir = output_dir / args.sidecar_subdir
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{args.jsonl_stem}.jsonl"

    model = load_model(
        args.device,
        args.checkpoint_dir,
        enable_alignment_controller=args.enable_alignment_controller,
    )
    t3 = model.worker.t3
    if int(t3.cfg.hidden_size) != 1024:
        raise ValueError(f"Expected T3 hidden size 1024, got {t3.cfg.hidden_size}")

    records = _load_records(args, t3.hp)
    start_index = 0
    if args.resume_existing:
        start_index = _count_existing_rows(jsonl_path)
        if start_index > len(records):
            raise ValueError(
                f"resume_existing requested but existing JSONL has {start_index} rows and source only has {len(records)}"
            )

    dtype = _sidecar_dtype(args.sidecar_dtype)
    hidden_dtype_name = str(dtype).replace("torch.", "")
    written = 0
    failures = 0
    existing_lines = start_index
    mode = "a" if args.resume_existing and jsonl_path.exists() else "w"

    progress_iter = records[start_index:]
    if tqdm is not None:
        progress = tqdm(progress_iter, total=len(progress_iter), desc="hydra-distill", dynamic_ncols=True, leave=True)
    else:
        progress = progress_iter

    with jsonl_path.open(mode, encoding="utf-8") as sink:
        for record in progress:
            try:
                text_tokens = torch.tensor(record.text_tokens, dtype=torch.long, device=model.worker.device).unsqueeze(0)
                speech_tokens = torch.tensor(record.speech_tokens, dtype=torch.long, device=model.worker.device).unsqueeze(0)
                t3_cond = clone_t3_cond(_load_t3_cond_cached(str(record.conditionals_path))).to(device=model.worker.device)
                bos = torch.full(
                    (1, 1),
                    fill_value=t3.hp.start_speech_token,
                    dtype=torch.long,
                    device=model.worker.device,
                )
                speech_inputs = torch.cat([bos, speech_tokens[:, :-1]], dim=1)

                with torch.inference_mode():
                    embeds, len_cond = t3.prepare_input_embeds(
                        t3_cond=t3_cond,
                        text_tokens=text_tokens,
                        speech_tokens=speech_inputs,
                    )
                    tfmr_out = t3.tfmr(
                        input_ids=None,
                        inputs_embeds=embeds,
                        output_hidden_states=False,
                        return_dict=True,
                        use_cache=False,
                    )
                    hidden_states = tfmr_out.last_hidden_state
                    speech_start = len_cond + text_tokens.size(1)
                    hidden_states = hidden_states[:, speech_start : speech_start + speech_tokens.size(1)]

                hidden_states = hidden_states.squeeze(0)
                if hidden_states.dim() != 2:
                    raise ValueError(f"Expected (decode_len, hidden_size), got {tuple(hidden_states.shape)}")
                if hidden_states.shape[-1] != t3.cfg.hidden_size:
                    raise ValueError(
                        f"Expected hidden size {t3.cfg.hidden_size}, got {hidden_states.shape[-1]}"
                    )
                if hidden_states.shape[0] != len(record.speech_tokens):
                    raise ValueError(
                        "Hidden-state length mismatch: "
                        f"{hidden_states.shape[0]} vs {len(record.speech_tokens)}"
                    )

                sidecar_path = sidecar_dir / f"sample_{record.sample_index:06d}.safetensors"
                _save_sidecar(sidecar_path, hidden_states, dtype)
                hydra_record = _build_hydra_record(
                    record,
                    sidecar_path=sidecar_path,
                    hidden_states=hidden_states,
                    hidden_dtype=hidden_dtype_name,
                )
                sink.write(json.dumps(hydra_record, ensure_ascii=False) + "\n")
                written += 1
                if tqdm is None and written % max(args.save_every, 1) == 0:
                    print(f"written={written} failures={failures}")
            except Exception as exc:  # pragma: no cover - builder should continue on bad rows
                failures += 1
                print(f"failed sample_index={record.sample_index}: {exc}")

    if tqdm is not None:
        progress.close()

    print(f"output_dir={output_dir.resolve()}")
    print(f"jsonl_path={jsonl_path.resolve()}")
    print(f"sidecar_dir={sidecar_dir.resolve()}")
    print(f"written={written}")
    print(f"failures={failures}")
    print(f"resume_existing={args.resume_existing}")
    print(f"start_index={start_index}")
    print(f"total_input_rows={len(records)}")
    if existing_lines:
        print(f"existing_rows={existing_lines}")
    return failures


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
