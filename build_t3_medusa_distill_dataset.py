import argparse
import csv
import json
import multiprocessing as mp
from pathlib import Path

import torch
import torch.nn.functional as F

from chatterbox.mtl_tts import SUPPORTED_LANGUAGES, punc_norm
from chatterbox.mtl_tts_scheduled import ChatterboxMultilingualScheduledTTS
from chatterbox.models.s3tokenizer import drop_invalid_tokens
from chatterbox.models.t3.inference.scheduled_decode import ScheduledDecodeRequest
from chatterbox.models.t3.inference.speculative_decode import run_baseline_greedy_decode
from chatterbox.runtime.session import clone_conditionals


def load_model(device: str, checkpoint_dir: str | None):
    if checkpoint_dir:
        return ChatterboxMultilingualScheduledTTS.from_local(checkpoint_dir, device)
    return ChatterboxMultilingualScheduledTTS.from_pretrained(device)


def build_single_request(
    model: ChatterboxMultilingualScheduledTTS,
    *,
    text: str,
    language_id: str,
    audio_prompt_path: str,
    max_new_tokens: int,
    cfg_weight: float,
    temperature: float,
    repetition_penalty: float,
    min_p: float,
    top_p: float,
):
    worker = model.worker
    if language_id.lower() not in SUPPORTED_LANGUAGES:
        supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
        raise ValueError(f"Unsupported language_id '{language_id}'. Supported languages: {supported_langs}")

    session = model.create_session(
        audio_prompt_path=audio_prompt_path,
        language_id=language_id,
        cfg_weight=cfg_weight,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        min_p=min_p,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    options = session.options

    normalized_text = punc_norm(text)
    text_tokens_single = worker.tokenizer.text_to_tokens(
        normalized_text,
        language_id=language_id.lower(),
    ).to(worker.device)
    text_tokens_single = F.pad(text_tokens_single, (1, 0), value=worker.t3.hp.start_text_token)
    text_tokens_single = F.pad(text_tokens_single, (0, 1), value=worker.t3.hp.stop_text_token)

    text_tokens_cfg = torch.cat([text_tokens_single, text_tokens_single], dim=0)
    conds = clone_conditionals(session.conditionals).to(worker.device)
    request = ScheduledDecodeRequest(
        session_id="medusa_distill",
        t3_cond=conds.t3,
        text_tokens=text_tokens_cfg,
        max_new_tokens=max_new_tokens,
        temperature=options.temperature,
        top_p=options.top_p,
        min_p=options.min_p,
        repetition_penalty=options.repetition_penalty,
        cfg_weight=options.cfg_weight,
    )
    return request, session, normalized_text, text_tokens_single


def shard_jsonl_path(output_dir: Path, num_shards: int, shard_index: int) -> Path:
    if num_shards == 1:
        return output_dir / "samples.jsonl"
    return output_dir / f"samples.shard_{shard_index:02d}_of_{num_shards:02d}.jsonl"


def save_conditionals_once(session, output_dir: Path, *, num_shards: int, shard_index: int) -> Path:
    conds_dir = output_dir / "conditionals"
    conds_dir.mkdir(parents=True, exist_ok=True)
    if num_shards == 1:
        conds_path = conds_dir / "prompt_000.pt"
    else:
        conds_path = conds_dir / f"prompt_shard_{shard_index:02d}_of_{num_shards:02d}.pt"
    if not conds_path.exists():
        clone_conditionals(session.conditionals).save(conds_path)
    return conds_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a T3 Medusa distillation dataset from Arabic text prompts."
    )
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--audio-prompt-path", required=True)
    parser.add_argument("--language-id", default="ar")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--mp-workers",
        type=int,
        default=1,
        help="Spawn this many worker processes and split the manifest evenly across them.",
    )
    return parser


def load_records(manifest_csv: str, limit: int, num_shards: int, shard_index: int):
    records = []
    with Path(manifest_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(row)

    if limit > 0:
        records = records[:limit]

    if num_shards > 1:
        records = records[shard_index::num_shards]

    return records


def validate_args(args: argparse.Namespace) -> None:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    if args.mp_workers < 1:
        raise ValueError("--mp-workers must be >= 1")
    if args.mp_workers > 1 and (args.num_shards != 1 or args.shard_index != 0):
        raise ValueError("Use either --mp-workers or manual --num-shards/--shard-index, not both together")


def run_shard(args: argparse.Namespace) -> int:
    validate_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = shard_jsonl_path(output_dir, args.num_shards, args.shard_index)

    model = load_model(args.device, args.checkpoint_dir)
    records = load_records(args.manifest_csv, args.limit, args.num_shards, args.shard_index)

    written = 0
    failures = 0
    conds_path_written = None
    log_prefix = f"[shard {args.shard_index}/{args.num_shards}] " if args.num_shards > 1 else ""
    with jsonl_path.open("w", encoding="utf-8") as sink:
        for index, row in enumerate(records):
            text = row["text"]
            sample_id = row.get("sample_id") or f"sample_{index:06d}"
            try:
                request, session, normalized_text, text_tokens_single = build_single_request(
                    model,
                    text=text,
                    language_id=args.language_id,
                    audio_prompt_path=args.audio_prompt_path,
                    max_new_tokens=args.max_new_tokens,
                    cfg_weight=args.cfg_weight,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    min_p=args.min_p,
                    top_p=args.top_p,
                )
                if conds_path_written is None:
                    conds_path_written = save_conditionals_once(
                        session,
                        output_dir,
                        num_shards=args.num_shards,
                        shard_index=args.shard_index,
                    )

                teacher_tokens = run_baseline_greedy_decode(model.worker.t3, request)
                teacher_tokens = drop_invalid_tokens(teacher_tokens[0]).to("cpu")
                text_tokens_single = text_tokens_single.to("cpu")

                record = {
                    "sample_id": sample_id,
                    "text": text,
                    "normalized_text": normalized_text,
                    "language_id": args.language_id,
                    "audio_prompt_path": str(Path(args.audio_prompt_path).resolve()),
                    "conditionals_path": str(conds_path_written.resolve()),
                    "text_tokens": text_tokens_single.tolist(),
                    "speech_tokens": teacher_tokens.tolist(),
                    "num_text_tokens": int(text_tokens_single.numel()),
                    "num_speech_tokens": int(teacher_tokens.numel()),
                    "teacher_decode": {
                        "cfg_weight": args.cfg_weight,
                        "temperature": args.temperature,
                        "repetition_penalty": args.repetition_penalty,
                        "min_p": args.min_p,
                        "top_p": args.top_p,
                        "max_new_tokens": args.max_new_tokens,
                    },
                    "source_wav_path": row.get("source_wav_path", ""),
                    "source_duration": row.get("duration", ""),
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if written % max(args.save_every, 1) == 0:
                    print(f"{log_prefix}written={written} failures={failures}")
            except Exception as exc:  # pragma: no cover - dataset generation should continue on bad rows
                failures += 1
                print(f"{log_prefix}failed sample_id={sample_id}: {exc}")

    print(f"{log_prefix}output_dir={output_dir}")
    print(f"{log_prefix}jsonl_path={jsonl_path}")
    print(f"{log_prefix}num_shards={args.num_shards}")
    print(f"{log_prefix}shard_index={args.shard_index}")
    print(f"{log_prefix}records_assigned={len(records)}")
    print(f"{log_prefix}written={written}")
    print(f"{log_prefix}failures={failures}")
    if conds_path_written is not None:
        print(f"{log_prefix}conditionals_path={conds_path_written}")
    return failures


def launch_mp_workers(args: argparse.Namespace) -> None:
    validate_args(args)
    ctx = mp.get_context("spawn")
    procs = []
    for shard_index in range(args.mp_workers):
        worker_args = argparse.Namespace(**vars(args))
        worker_args.mp_workers = 1
        worker_args.num_shards = args.mp_workers
        worker_args.shard_index = shard_index
        proc = ctx.Process(target=run_shard, args=(worker_args,), name=f"medusa_distill_{shard_index}")
        proc.start()
        procs.append(proc)

    exit_codes = []
    for proc in procs:
        proc.join()
        exit_codes.append(proc.exitcode)

    if any(code != 0 for code in exit_codes):
        raise SystemExit(f"mp worker exit codes={exit_codes}")

    print(f"mp_workers={args.mp_workers}")
    print(f"completed_shards={args.mp_workers}")
    print(f"output_dir={args.output_dir}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.mp_workers > 1:
        launch_mp_workers(args)
        return
    run_shard(args)


if __name__ == "__main__":
    main()
