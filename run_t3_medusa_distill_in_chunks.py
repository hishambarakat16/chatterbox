import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the T3 Medusa distillation builder in restartable manifest chunks."
    )
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--audio-prompt-path", required=True)
    parser.add_argument("--language-id", default="ar")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--total-limit", type=int, default=0)
    parser.add_argument("--base-jsonl-stem", default="samples")
    parser.add_argument("--decode-impl", choices=("scheduled", "greedy"), default="scheduled")
    parser.add_argument("--mp-workers", type=int, default=1)
    parser.add_argument("--scheduler-inflight", type=int, default=4)
    parser.add_argument("--scheduler-batching-window-ms", type=float, default=10.0)
    parser.add_argument("--enable-alignment-controller", action="store_true")
    parser.add_argument("--disable-batch-key-sort", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    return parser


def count_manifest_rows(manifest_csv: str) -> int:
    with Path(manifest_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def expected_jsonl_paths(output_dir: Path, *, jsonl_stem: str, mp_workers: int) -> list[Path]:
    if mp_workers <= 1:
        return [output_dir / f"{jsonl_stem}.jsonl"]
    return [
        output_dir / f"{jsonl_stem}.shard_{shard_index:02d}_of_{mp_workers:02d}.jsonl"
        for shard_index in range(mp_workers)
    ]


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = count_manifest_rows(args.manifest_csv)
    final_stop = total_rows
    if args.total_limit > 0:
        final_stop = min(final_stop, args.start_offset + args.total_limit)

    if args.start_offset >= final_stop:
        raise ValueError(
            f"start offset {args.start_offset} is beyond the selected manifest range ending at {final_stop}"
        )

    builder_path = Path(__file__).with_name("build_t3_medusa_distill_dataset.py").resolve()
    env = os.environ.copy()

    chunk_index = 0
    for chunk_start in range(args.start_offset, final_stop, args.chunk_size):
        chunk_limit = min(args.chunk_size, final_stop - chunk_start)
        chunk_end = chunk_start + chunk_limit
        jsonl_stem = f"{args.base_jsonl_stem}.chunk_{chunk_index:03d}_{chunk_start:06d}_{chunk_end - 1:06d}"
        expected = expected_jsonl_paths(output_dir, jsonl_stem=jsonl_stem, mp_workers=args.mp_workers)

        if args.resume_existing and all(path.exists() for path in expected):
            print(f"skip_existing chunk_index={chunk_index} offset={chunk_start} limit={chunk_limit} jsonl_stem={jsonl_stem}")
            chunk_index += 1
            continue

        cmd = [
            sys.executable,
            str(builder_path),
            "--manifest-csv",
            args.manifest_csv,
            "--audio-prompt-path",
            args.audio_prompt_path,
            "--language-id",
            args.language_id,
            "--device",
            args.device,
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--cfg-weight",
            str(args.cfg_weight),
            "--temperature",
            str(args.temperature),
            "--repetition-penalty",
            str(args.repetition_penalty),
            "--min-p",
            str(args.min_p),
            "--top-p",
            str(args.top_p),
            "--offset",
            str(chunk_start),
            "--limit",
            str(chunk_limit),
            "--jsonl-stem",
            jsonl_stem,
            "--decode-impl",
            args.decode_impl,
            "--mp-workers",
            str(args.mp_workers),
            "--scheduler-inflight",
            str(args.scheduler_inflight),
            "--scheduler-batching-window-ms",
            str(args.scheduler_batching_window_ms),
        ]
        if args.checkpoint_dir:
            cmd.extend(["--checkpoint-dir", args.checkpoint_dir])
        if args.enable_alignment_controller:
            cmd.append("--enable-alignment-controller")
        if args.disable_batch_key_sort:
            cmd.append("--disable-batch-key-sort")

        print(
            f"run_chunk chunk_index={chunk_index} offset={chunk_start} limit={chunk_limit} "
            f"jsonl_stem={jsonl_stem}"
        )
        subprocess.run(cmd, check=True, env=env)

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)
        chunk_index += 1

    print(f"completed_chunks={chunk_index}")
    print(f"output_dir={output_dir}")
    print(f"decode_impl={args.decode_impl}")
    print(f"alignment_controller_enabled={args.enable_alignment_controller}")


if __name__ == "__main__":
    main()
