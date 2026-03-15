import argparse
import csv
import json
import multiprocessing as mp
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal envs
    tqdm = None

from chatterbox.mtl_tts import SUPPORTED_LANGUAGES, punc_norm
from chatterbox.mtl_tts_scheduled import ChatterboxMultilingualScheduledTTS
from chatterbox.models.s3tokenizer import drop_invalid_tokens
from chatterbox.models.t3.inference.scheduled_decode import ScheduledDecodeRequest
from chatterbox.models.t3.inference.speculative_decode import run_baseline_greedy_decode
from chatterbox.runtime.session import clone_conditionals, clone_t3_cond


def load_model(device: str, checkpoint_dir: str | None):
    if checkpoint_dir:
        return ChatterboxMultilingualScheduledTTS.from_local(checkpoint_dir, device)
    return ChatterboxMultilingualScheduledTTS.from_pretrained(device)


@dataclass
class PreparedRecord:
    ordinal: int
    sample_id: str
    text: str
    normalized_text: str
    text_tokens_single: torch.Tensor
    request: ScheduledDecodeRequest
    source_wav_path: str
    source_duration: str


def build_base_session(model: ChatterboxMultilingualScheduledTTS, args: argparse.Namespace):
    return model.create_session(
        audio_prompt_path=args.audio_prompt_path,
        language_id=args.language_id,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        min_p=args.min_p,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        session_id="medusa_distill_prompt",
    )


def build_single_request(
    model: ChatterboxMultilingualScheduledTTS,
    *,
    base_session,
    text: str,
    language_id: str,
    session_id: str,
):
    worker = model.worker
    if language_id.lower() not in SUPPORTED_LANGUAGES:
        supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
        raise ValueError(f"Unsupported language_id '{language_id}'. Supported languages: {supported_langs}")

    options = base_session.options

    normalized_text = punc_norm(text)
    text_tokens_single = worker.tokenizer.text_to_tokens(
        normalized_text,
        language_id=language_id.lower(),
    ).to(worker.device)
    text_tokens_single = F.pad(text_tokens_single, (1, 0), value=worker.t3.hp.start_text_token)
    text_tokens_single = F.pad(text_tokens_single, (0, 1), value=worker.t3.hp.stop_text_token)

    text_tokens_cfg = torch.cat([text_tokens_single, text_tokens_single], dim=0)
    t3_cond = clone_t3_cond(base_session.conditionals.t3).to(device=worker.device)
    request = ScheduledDecodeRequest(
        session_id=session_id,
        t3_cond=t3_cond,
        text_tokens=text_tokens_cfg,
        max_new_tokens=options.max_new_tokens,
        temperature=options.temperature,
        top_p=options.top_p,
        min_p=options.min_p,
        repetition_penalty=options.repetition_penalty,
        cfg_weight=options.cfg_weight,
    )
    return request, normalized_text, text_tokens_single


def shard_jsonl_path(
    output_dir: Path,
    *,
    jsonl_stem: str,
    num_shards: int,
    shard_index: int,
) -> Path:
    if num_shards == 1:
        return output_dir / f"{jsonl_stem}.jsonl"
    return output_dir / f"{jsonl_stem}.shard_{shard_index:02d}_of_{num_shards:02d}.jsonl"


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
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument(
        "--jsonl-stem",
        default="samples",
        help="Output file stem. Useful for chunked runs that restart the builder process.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--decode-impl",
        choices=("scheduled", "greedy"),
        default="scheduled",
        help="Use the shared scheduled T3 path or the legacy one-request greedy loop.",
    )
    parser.add_argument(
        "--scheduler-inflight",
        type=int,
        default=4,
        help="Per-process in-flight T3 requests when --decode-impl=scheduled.",
    )
    parser.add_argument(
        "--scheduler-batching-window-ms",
        type=float,
        default=10.0,
        help="Scheduler batching window for scheduled dataset generation.",
    )
    parser.add_argument(
        "--disable-batch-key-sort",
        action="store_true",
        help="Disable sorting manifest rows by scheduled batch key before decode.",
    )
    parser.add_argument(
        "--mp-workers",
        type=int,
        default=1,
        help="Spawn this many worker processes and split the manifest evenly across them.",
    )
    return parser


def load_records(manifest_csv: str, offset: int, limit: int, num_shards: int, shard_index: int):
    records = []
    with Path(manifest_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(row)

    if offset > 0:
        records = records[offset:]

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
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")
    if args.mp_workers < 1:
        raise ValueError("--mp-workers must be >= 1")
    if args.scheduler_inflight < 1:
        raise ValueError("--scheduler-inflight must be >= 1")
    if args.mp_workers > 1 and (args.num_shards != 1 or args.shard_index != 0):
        raise ValueError("Use either --mp-workers or manual --num-shards/--shard-index, not both together")


def _prompt_len_from_session(base_session) -> int:
    prompt = getattr(base_session.conditionals.t3, "cond_prompt_speech_tokens", None)
    return 0 if prompt is None else int(prompt.shape[-1])


def _preview_batch_key(
    model: ChatterboxMultilingualScheduledTTS,
    *,
    text: str,
    language_id: str,
    prompt_len: int,
    cache: dict[tuple[str, str], tuple[int, int]],
) -> tuple[int, int]:
    normalized_text = punc_norm(text)
    cache_key = (language_id.lower(), normalized_text)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    text_tokens = model.worker.tokenizer.text_to_tokens(
        normalized_text,
        language_id=language_id.lower(),
    )
    batch_key = (int(text_tokens.numel()) + 2, prompt_len)
    cache[cache_key] = batch_key
    return batch_key


def sort_records_for_scheduled(
    model: ChatterboxMultilingualScheduledTTS,
    *,
    base_session,
    records: list[tuple[int, dict]],
    language_id: str,
) -> list[tuple[int, dict]]:
    prompt_len = _prompt_len_from_session(base_session)
    preview_cache: dict[tuple[str, str], tuple[int, int]] = {}
    return sorted(
        records,
        key=lambda item: _preview_batch_key(
            model,
            text=item[1]["text"],
            language_id=language_id,
            prompt_len=prompt_len,
            cache=preview_cache,
        ),
    )


def prepare_record(
    model: ChatterboxMultilingualScheduledTTS,
    *,
    base_session,
    row: dict,
    ordinal: int,
    shard_index: int,
    num_shards: int,
    language_id: str,
) -> PreparedRecord:
    text = row["text"]
    sample_id = row.get("sample_id") or f"sample_{ordinal:06d}"
    session_id = f"medusa_distill_s{shard_index:02d}_of_{num_shards:02d}_{ordinal:07d}_{sample_id}"
    request, normalized_text, text_tokens_single = build_single_request(
        model,
        base_session=base_session,
        text=text,
        language_id=language_id,
        session_id=session_id,
    )
    return PreparedRecord(
        ordinal=ordinal,
        sample_id=sample_id,
        text=text,
        normalized_text=normalized_text,
        text_tokens_single=text_tokens_single,
        request=request,
        source_wav_path=row.get("source_wav_path", ""),
        source_duration=row.get("duration", ""),
    )


def build_output_record(
    prepared: PreparedRecord,
    *,
    teacher_tokens: torch.Tensor,
    args: argparse.Namespace,
    conds_path_written: Path,
) -> dict:
    text_tokens_single = prepared.text_tokens_single.to("cpu")
    return {
        "sample_id": prepared.sample_id,
        "text": prepared.text,
        "normalized_text": prepared.normalized_text,
        "language_id": args.language_id,
        "audio_prompt_path": str(Path(args.audio_prompt_path).resolve()),
        "conditionals_path": str(conds_path_written.resolve()),
        "text_tokens": text_tokens_single.tolist(),
        "speech_tokens": teacher_tokens.tolist(),
        "num_text_tokens": int(text_tokens_single.numel()),
        "num_speech_tokens": int(teacher_tokens.numel()),
        "teacher_decode": {
            "impl": args.decode_impl,
            "cfg_weight": args.cfg_weight,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty,
            "min_p": args.min_p,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "scheduler_inflight": args.scheduler_inflight,
            "scheduler_batching_window_ms": args.scheduler_batching_window_ms,
        },
        "source_wav_path": prepared.source_wav_path,
        "source_duration": prepared.source_duration,
    }


def _maybe_set_progress(progress_bar, *, written: int, failures: int):
    if tqdm is not None:
        progress_bar.set_postfix(written=written, failures=failures)


def _decode_scheduled_tokens(model: ChatterboxMultilingualScheduledTTS, request: ScheduledDecodeRequest) -> torch.Tensor:
    teacher_tokens, _ = model.worker.t3_scheduler.submit(request)
    return drop_invalid_tokens(teacher_tokens[0]).to("cpu")


def run_records_greedy(
    *,
    model: ChatterboxMultilingualScheduledTTS,
    base_session,
    records: list[tuple[int, dict]],
    args: argparse.Namespace,
    sink,
    progress_bar,
    output_dir: Path,
    conds_path_written: Path,
) -> tuple[int, int]:
    written = 0
    failures = 0
    log_prefix = f"[shard {args.shard_index}/{args.num_shards}] " if args.num_shards > 1 else ""
    for ordinal, row in records:
        sample_id = row.get("sample_id") or f"sample_{ordinal:06d}"
        try:
            prepared = prepare_record(
                model,
                base_session=base_session,
                row=row,
                ordinal=ordinal,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                language_id=args.language_id,
            )
            teacher_tokens = run_baseline_greedy_decode(model.worker.t3, prepared.request)
            teacher_tokens = drop_invalid_tokens(teacher_tokens[0]).to("cpu")
            record = build_output_record(
                prepared,
                teacher_tokens=teacher_tokens,
                args=args,
                conds_path_written=conds_path_written,
            )
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
        except Exception as exc:  # pragma: no cover - dataset generation should continue on bad rows
            failures += 1
            print(f"{log_prefix}failed sample_id={sample_id}: {exc}")
        if tqdm is not None:
            progress_bar.update(1)
            _maybe_set_progress(progress_bar, written=written, failures=failures)
        elif written % max(args.save_every, 1) == 0:
            print(f"{log_prefix}written={written} failures={failures}")
    return written, failures


def run_records_scheduled(
    *,
    model: ChatterboxMultilingualScheduledTTS,
    base_session,
    records: list[tuple[int, dict]],
    args: argparse.Namespace,
    sink,
    progress_bar,
    output_dir: Path,
    conds_path_written: Path,
) -> tuple[int, int]:
    written = 0
    failures = 0
    log_prefix = f"[shard {args.shard_index}/{args.num_shards}] " if args.num_shards > 1 else ""
    records_iter = iter(records)
    pending: dict[object, PreparedRecord] = {}

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        nonlocal failures
        while len(pending) < args.scheduler_inflight:
            try:
                ordinal, row = next(records_iter)
            except StopIteration:
                return False

            sample_id = row.get("sample_id") or f"sample_{ordinal:06d}"
            try:
                prepared = prepare_record(
                    model,
                    base_session=base_session,
                    row=row,
                    ordinal=ordinal,
                    shard_index=args.shard_index,
                    num_shards=args.num_shards,
                    language_id=args.language_id,
                )
                future = executor.submit(_decode_scheduled_tokens, model, prepared.request)
                pending[future] = prepared
            except Exception as exc:
                failures += 1
                print(f"{log_prefix}failed sample_id={sample_id}: {exc}")
                if tqdm is not None:
                    progress_bar.update(1)
                    _maybe_set_progress(progress_bar, written=written, failures=failures)
        return True

    with ThreadPoolExecutor(max_workers=args.scheduler_inflight, thread_name_prefix="medusa_distill_sched") as executor:
        submit_next(executor)
        while pending:
            done, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                prepared = pending.pop(future)
                try:
                    teacher_tokens = future.result()
                    record = build_output_record(
                        prepared,
                        teacher_tokens=teacher_tokens,
                        args=args,
                        conds_path_written=conds_path_written,
                    )
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                except Exception as exc:
                    failures += 1
                    print(f"{log_prefix}failed sample_id={prepared.sample_id}: {exc}")

                if tqdm is not None:
                    progress_bar.update(1)
                    _maybe_set_progress(progress_bar, written=written, failures=failures)
                elif written % max(args.save_every, 1) == 0:
                    print(f"{log_prefix}written={written} failures={failures}")

            submit_next(executor)

    return written, failures


def run_shard(args: argparse.Namespace) -> int:
    validate_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = shard_jsonl_path(
        output_dir,
        jsonl_stem=args.jsonl_stem,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )

    model = load_model(args.device, args.checkpoint_dir)
    if hasattr(model.worker, "t3_scheduler"):
        model.worker.t3_scheduler.batching_window_ms = float(args.scheduler_batching_window_ms)

    base_session = build_base_session(model, args)
    records = load_records(args.manifest_csv, args.offset, args.limit, args.num_shards, args.shard_index)
    indexed_records = list(enumerate(records))
    if args.decode_impl == "scheduled" and not args.disable_batch_key_sort:
        indexed_records = sort_records_for_scheduled(
            model,
            base_session=base_session,
            records=indexed_records,
            language_id=args.language_id,
        )

    conds_path_written = save_conditionals_once(
        base_session,
        output_dir,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    log_prefix = f"[shard {args.shard_index}/{args.num_shards}] " if args.num_shards > 1 else ""
    progress_desc = (
        f"shard {args.shard_index + 1}/{args.num_shards}"
        if args.num_shards > 1
        else "medusa-distill"
    )
    progress_position = args.shard_index if args.num_shards > 1 else 0
    progress_bar = (
        tqdm(
            records,
            total=len(records),
            desc=progress_desc,
            position=progress_position,
            dynamic_ncols=True,
            leave=True,
        )
        if tqdm is not None
        else indexed_records
    )
    with jsonl_path.open("w", encoding="utf-8") as sink:
        if args.decode_impl == "scheduled":
            written, failures = run_records_scheduled(
                model=model,
                base_session=base_session,
                records=indexed_records,
                args=args,
                sink=sink,
                progress_bar=progress_bar,
                output_dir=output_dir,
                conds_path_written=conds_path_written,
            )
        else:
            written, failures = run_records_greedy(
                model=model,
                base_session=base_session,
                records=indexed_records,
                args=args,
                sink=sink,
                progress_bar=progress_bar,
                output_dir=output_dir,
                conds_path_written=conds_path_written,
            )

    if tqdm is not None:
        progress_bar.close()

    print(f"{log_prefix}output_dir={output_dir}")
    print(f"{log_prefix}jsonl_path={jsonl_path}")
    print(f"{log_prefix}decode_impl={args.decode_impl}")
    print(f"{log_prefix}offset={args.offset}")
    print(f"{log_prefix}limit={args.limit}")
    print(f"{log_prefix}jsonl_stem={args.jsonl_stem}")
    print(f"{log_prefix}scheduler_inflight={args.scheduler_inflight}")
    print(f"{log_prefix}scheduler_batching_window_ms={args.scheduler_batching_window_ms}")
    print(f"{log_prefix}num_shards={args.num_shards}")
    print(f"{log_prefix}shard_index={args.shard_index}")
    print(f"{log_prefix}records_assigned={len(indexed_records)}")
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
