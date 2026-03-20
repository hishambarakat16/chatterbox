import argparse
import json
import time
from pathlib import Path


DEFAULT_SENTENCES_FILE = Path(__file__).with_name("arabic_streaming_sentences.txt")
_RUNTIME_HELPERS = None


def runtime_helpers():
    global _RUNTIME_HELPERS
    if _RUNTIME_HELPERS is None:
        from benchmark_multilingual_concurrency import (  # noqa: PLC0415
            describe_vllm_hydra_mode,
            load_model,
            maybe_sync,
        )

        _RUNTIME_HELPERS = {
            "describe_vllm_hydra_mode": describe_vllm_hydra_mode,
            "load_model": load_model,
            "maybe_sync": maybe_sync,
        }
    return _RUNTIME_HELPERS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose the custom Chatterbox vLLM input path."
    )
    parser.add_argument(
        "--mode",
        choices=[
            "inspect",
            "sequential_singletons",
            "batched",
        ],
        default="inspect",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--base-checkpoint-dir")
    parser.add_argument("--turbo-s3-checkpoint-dir")
    parser.add_argument("--vllm-model-dir")
    parser.add_argument("--vllm-export-dir")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--vllm-dtype", default="auto")
    parser.add_argument("--vllm-max-model-len", type=int, default=2048)
    prefix_group = parser.add_mutually_exclusive_group()
    prefix_group.add_argument("--vllm-enable-prefix-caching", action="store_true")
    prefix_group.add_argument("--no-vllm-prefix-caching", action="store_true")
    chunked_prefill_group = parser.add_mutually_exclusive_group()
    chunked_prefill_group.add_argument("--vllm-enable-chunked-prefill", action="store_true")
    chunked_prefill_group.add_argument("--no-vllm-chunked-prefill", action="store_true")
    parser.add_argument("--language-id", default="ar")
    parser.add_argument("--audio-prompt-path")
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--cfg-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=2.0)
    parser.add_argument("--min-p", type=float, default=0.05)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--text", action="append")
    parser.add_argument("--texts-file")
    parser.add_argument("--fixed-text")
    parser.add_argument("--repeat-fixed-text", type=int, default=1)
    parser.add_argument("--text-limit", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-json")
    return parser.parse_args()


def load_texts(args) -> list[str]:
    if args.fixed_text is not None:
        repeat = max(1, int(args.repeat_fixed_text))
        return [args.fixed_text] * repeat

    if args.text:
        return list(args.text)

    text_path = DEFAULT_SENTENCES_FILE if args.texts_file is None else Path(args.texts_file)
    texts = []
    for raw_line in text_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        texts.append(line)
    if args.text_limit is not None:
        texts = texts[: max(0, int(args.text_limit))]
    if not texts:
        raise ValueError("No texts supplied for diagnosis.")
    return texts


def chunked(values: list, size: int):
    step = max(1, int(size))
    for start in range(0, len(values), step):
        yield values[start : start + step]


def shape_key(metadata: dict) -> str:
    return (
        f"text_tok={metadata['t3_text_token_len']}"
        f"/prompt_speech={metadata['t3_prompt_speech_token_len']}"
        f"/cond={metadata['t3_cond_seq_len']}"
        f"/prompt_seq={metadata['t3_prompt_seq_len']}"
    )


def histogram(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def inspect_requests(model, sessions, texts, args) -> list[dict]:
    records = []
    for index, (session, text) in enumerate(zip(sessions, texts)):
        metadata = model.inspect_prompt_embed_with_session(
            session,
            text,
            language_id=args.language_id,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            min_p=args.min_p,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        metadata["index"] = index
        metadata["shape_key"] = shape_key(metadata)
        records.append(metadata)
    return records


def build_request_options(session, args):
    return session.options.merged(
        language_id=args.language_id,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        min_p=args.min_p,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )


def run_sequential_singletons(model, sessions, texts, inspect_rows, args) -> list[dict]:
    maybe_sync = runtime_helpers()["maybe_sync"]
    results = []
    for row, session, text in zip(inspect_rows, sessions, texts):
        started = time.perf_counter()
        error = None
        profile = {}
        try:
            batch_results = model.generate_many_with_sessions(
                [session],
                [text],
                language_id=args.language_id,
                exaggeration=args.exaggeration,
                cfg_weight=args.cfg_weight,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                min_p=args.min_p,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )
            maybe_sync(args.device)
            if batch_results:
                profile = dict(batch_results[0]["profile"])
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
            maybe_sync(args.device)
        results.append(
            {
                "mode": "sequential_singletons",
                "index": row["index"],
                "shape_key": row["shape_key"],
                "text": text,
                "error": error,
                "wall_s": round(time.perf_counter() - started, 4),
                "audio_ready_s": round(float(profile.get("audio_ready_s", 0.0)), 4) if profile else 0.0,
                "t3_s": round(float(profile.get("t3_s", 0.0)), 4) if profile else 0.0,
                "prompt_seq_len": int(profile.get("t3_prompt_seq_len", row["t3_prompt_seq_len"])),
                "prompt_token_len_before_mm": int(
                    profile.get(
                        "t3_prompt_token_len_before_mm",
                        row["t3_prompt_token_len_before_mm"],
                    )
                ),
                "cond_seq_len": int(profile.get("t3_cond_seq_len", row["t3_cond_seq_len"])),
            }
        )
    return results


def run_batched(model, sessions, texts, inspect_rows, args) -> list[dict]:
    maybe_sync = runtime_helpers()["maybe_sync"]
    results = []
    for batch_index, index_chunk in enumerate(chunked(list(range(len(texts))), args.batch_size)):
        session_chunk = [sessions[idx] for idx in index_chunk]
        text_chunk = [texts[idx] for idx in index_chunk]
        inspect_chunk = [inspect_rows[idx] for idx in index_chunk]
        started = time.perf_counter()
        error = None
        try:
            batch_results = model.generate_many_with_sessions(
                session_chunk,
                text_chunk,
                language_id=args.language_id,
                exaggeration=args.exaggeration,
                cfg_weight=args.cfg_weight,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                min_p=args.min_p,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )
            maybe_sync(args.device)
        except Exception as exc:  # noqa: BLE001
            batch_results = []
            error = repr(exc)
            maybe_sync(args.device)

        shape_keys = [row["shape_key"] for row in inspect_chunk]
        batch_wall_s = round(time.perf_counter() - started, 4)
        results.append(
            {
                "mode": "batched",
                "batch_index": batch_index,
                "batch_size": len(index_chunk),
                "request_indexes": index_chunk,
                "shape_keys": shape_keys,
                "mixed_shape_keys": len(set(shape_keys)) > 1,
                "error": error,
                "wall_s": batch_wall_s,
                "num_outputs": len(batch_results),
            }
        )
    return results


def build_summary(*, args, texts, inspect_rows, run_rows) -> dict:
    shape_hist = histogram([row["shape_key"] for row in inspect_rows])
    success_rows = [row for row in run_rows if row.get("error") is None]
    error_rows = [row for row in run_rows if row.get("error") is not None]
    summary = {
        "mode": args.mode,
        "request_count": len(texts),
        "shape_hist": shape_hist,
        "inspect_rows": inspect_rows,
        "run_rows": run_rows,
        "num_success": len(success_rows),
        "num_errors": len(error_rows),
        "errors": [row["error"] for row in error_rows],
    }
    if args.mode == "sequential_singletons":
        transitions = []
        for prev, curr in zip(inspect_rows, inspect_rows[1:]):
            transitions.append(f"{prev['shape_key']} -> {curr['shape_key']}")
        summary["shape_transition_hist"] = histogram(transitions)
    if args.mode == "batched":
        summary["mixed_batch_count"] = sum(1 for row in run_rows if row["mixed_shape_keys"])
    return summary


def main():
    args = parse_args()
    texts = load_texts(args)
    helpers = runtime_helpers()
    load_model = helpers["load_model"]
    describe_vllm_hydra_mode = helpers["describe_vllm_hydra_mode"]
    model = load_model(
        "vllm_turbo_s3",
        args.device,
        args.checkpoint_dir,
        base_checkpoint_dir=args.base_checkpoint_dir,
        turbo_s3_checkpoint_dir=args.turbo_s3_checkpoint_dir,
        vllm_model_dir=args.vllm_model_dir,
        vllm_export_dir=args.vllm_export_dir,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enforce_eager=args.vllm_enforce_eager,
        vllm_dtype=args.vllm_dtype,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_enable_prefix_caching=args.vllm_enable_prefix_caching and not args.no_vllm_prefix_caching,
        vllm_enable_chunked_prefill=args.vllm_enable_chunked_prefill or not args.no_vllm_chunked_prefill,
    )

    sessions = [
        model.create_session(
            audio_prompt_path=args.audio_prompt_path,
            language_id=args.language_id,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            min_p=args.min_p,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        for _ in texts
    ]

    print("impl=vllm_turbo_s3")
    print(f"mode={args.mode}")
    print(f"device={args.device}")
    print(f"text_count={len(texts)}")
    print(f"request_text_mode={'fixed' if args.fixed_text is not None else 'explicit_or_file'}")
    print(f"vllm_enforce_eager={args.vllm_enforce_eager}")
    print(
        f"vllm_enable_prefix_caching={args.vllm_enable_prefix_caching and not args.no_vllm_prefix_caching}"
    )
    print(
        f"vllm_enable_chunked_prefill={args.vllm_enable_chunked_prefill or not args.no_vllm_chunked_prefill}"
    )
    for note in describe_vllm_hydra_mode(
        impl="vllm_turbo_s3",
        hydra_checkpoint_dir=None,
        hydra_speculate_k=3,
    ):
        print(note)

    inspect_rows = inspect_requests(model, sessions, texts, args)
    print(f"shape_hist={histogram([row['shape_key'] for row in inspect_rows])}")
    for row in inspect_rows:
        print(
            "inspect_row="
            + json.dumps(
                {
                    "index": row["index"],
                    "session_id": row["session_id"],
                    "shape_key": row["shape_key"],
                    "text_chars": row["text_chars"],
                    "text_words": row["text_words"],
                    "t3_text_token_len": row["t3_text_token_len"],
                    "t3_prompt_speech_token_len": row["t3_prompt_speech_token_len"],
                    "t3_cond_seq_len": row["t3_cond_seq_len"],
                    "t3_prompt_seq_len": row["t3_prompt_seq_len"],
                    "t3_prompt_hidden_size": row["t3_prompt_hidden_size"],
                    "t3_prompt_token_len_before_mm": row["t3_prompt_token_len_before_mm"],
                    "text": row["text"],
                },
                ensure_ascii=False,
            )
        )

    if args.mode == "inspect":
        run_rows = []
    elif args.mode == "sequential_singletons":
        run_rows = run_sequential_singletons(model, sessions, texts, inspect_rows, args)
    else:
        run_rows = run_batched(model, sessions, texts, inspect_rows, args)

    summary = build_summary(args=args, texts=texts, inspect_rows=inspect_rows, run_rows=run_rows)
    if run_rows:
        print(f"num_success={summary['num_success']}")
        print(f"num_errors={summary['num_errors']}")
        print(f"errors={summary['errors']}")
        if args.mode == "sequential_singletons":
            print(f"shape_transition_hist={summary['shape_transition_hist']}")
        if args.mode == "batched":
            print(f"mixed_batch_count={summary['mixed_batch_count']}")
        for row in run_rows:
            print("run_row=" + json.dumps(row, ensure_ascii=False))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"output_json={output_path}")

    close = getattr(model, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()
