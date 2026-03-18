import argparse
import json
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torchaudio as ta

from benchmark_multilingual_concurrency import (
    begin_vram_measurement,
    finish_vram_measurement,
    get_last_profile,
    load_model,
    maybe_sync,
    percentile,
)


DEFAULT_SENTENCES_FILE = Path(__file__).with_name("arabic_streaming_sentences.txt")
SESSION_IMPLS = {"streaming", "concurrent", "scheduled", "scheduled_turbo_s3"}


def load_sentences(path: str | None) -> list[str]:
    sentence_path = DEFAULT_SENTENCES_FILE if path is None else Path(path)
    lines = sentence_path.read_text(encoding="utf-8").splitlines()
    sentences = []
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        sentences.append(text)
    if not sentences:
        raise ValueError(f"No usable sentences found in {sentence_path}")
    return sentences


def mean_or_zero(values: list[float]) -> float:
    return 0.0 if not values else float(statistics.mean(values))


def pstdev_or_zero(values: list[float]) -> float:
    return 0.0 if len(values) < 2 else float(statistics.pstdev(values))


def build_session(
    *,
    model,
    audio_prompt_path: str | None,
    language_id: str,
    cfg_weight: float,
    temperature: float,
    repetition_penalty: float,
    min_p: float,
    top_p: float,
    max_new_tokens: int,
):
    return model.create_session(
        audio_prompt_path=audio_prompt_path,
        language_id=language_id,
        cfg_weight=cfg_weight,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        min_p=min_p,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )


def run_warmup(
    *,
    model,
    impl: str,
    audio_prompt_path: str | None,
    language_id: str,
    warmup_text: str,
    warmup_runs: int,
    cfg_weight: float,
    temperature: float,
    repetition_penalty: float,
    min_p: float,
    top_p: float,
    max_new_tokens: int,
):
    if warmup_runs <= 0:
        return

    for _ in range(warmup_runs):
        session = build_session(
            model=model,
            audio_prompt_path=audio_prompt_path,
            language_id=language_id,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        if impl not in SESSION_IMPLS:
            raise ValueError(f"Unsupported impl for service simulator: {impl}")
        _ = model.generate_with_session(session, warmup_text)
        maybe_sync(model.device)


def request_metric(item: dict, key: str) -> float:
    value = item["profile"].get(key, 0.0)
    return 0.0 if value is None else float(value)


def summarize_requests(level: int, requests: list[dict], wall_s: float, vram_summary: dict) -> dict:
    ok = [item for item in requests if item["error"] is None]
    errors = [item["error"] for item in requests if item["error"] is not None]

    latencies = [float(item["latency_s"]) for item in ok]
    first_tokens = [request_metric(item, "t3_first_token_s") for item in ok]
    audio_ready = [request_metric(item, "audio_ready_s") for item in ok]
    t3_total = [request_metric(item, "t3_s") for item in ok]
    s3_total = [request_metric(item, "s3_s") for item in ok]
    s3_token2mel = [request_metric(item, "s3_token2mel_s") for item in ok]
    s3_hift = [request_metric(item, "s3_hift_s") for item in ok]
    num_samples = [int(item["num_samples"]) for item in ok if item["num_samples"] is not None]
    total_audio_s = sum(num_samples) / 24000.0 if num_samples else 0.0

    return {
        "concurrency": level,
        "num_requests": len(requests),
        "num_success": len(ok),
        "num_errors": len(errors),
        "errors": errors,
        "wall_s": round(float(wall_s), 4),
        "audio_seconds_total": round(float(total_audio_s), 4),
        "audio_seconds_per_second": round(float(total_audio_s / wall_s), 4) if wall_s > 0 else 0.0,
        "mean_latency_s": round(mean_or_zero(latencies), 4),
        "p95_latency_s": round(percentile(latencies, 0.95), 4) if latencies else 0.0,
        "latency_std_s": round(pstdev_or_zero(latencies), 4),
        "mean_first_token_s": round(mean_or_zero(first_tokens), 4),
        "p95_first_token_s": round(percentile(first_tokens, 0.95), 4) if first_tokens else 0.0,
        "first_token_std_s": round(pstdev_or_zero(first_tokens), 4),
        "mean_audio_ready_s": round(mean_or_zero(audio_ready), 4),
        "p95_audio_ready_s": round(percentile(audio_ready, 0.95), 4) if audio_ready else 0.0,
        "audio_ready_std_s": round(pstdev_or_zero(audio_ready), 4),
        "mean_t3_s": round(mean_or_zero(t3_total), 4),
        "mean_s3_s": round(mean_or_zero(s3_total), 4),
        "mean_s3_token2mel_s": round(mean_or_zero(s3_token2mel), 4),
        "mean_s3_hift_s": round(mean_or_zero(s3_hift), 4),
        "request_texts": [item["text"] for item in requests],
        "requests": requests,
        **vram_summary,
    }


def save_representative_wavs(
    *,
    output_dir: Path,
    save_mode: str,
    level_summaries: list[dict],
    sample_rate: int,
) -> list[dict]:
    saved = []
    if save_mode == "none":
        return saved

    for summary in level_summaries:
        level = summary["concurrency"]
        ok = [item for item in summary["requests"] if item["error"] is None and item["wav"] is not None]
        if not ok:
            continue

        if save_mode == "all":
            level_saved = []
            for item in ok:
                wav_path = output_dir / "wav_outputs" / f"c{level}_round{item['round_index']}_req{item['request_index']}.wav"
                wav_path.parent.mkdir(parents=True, exist_ok=True)
                ta.save(str(wav_path), item["wav"], sample_rate)
                item["wav_path"] = str(wav_path)
                level_saved.append(str(wav_path))
            saved.append({
                "concurrency": level,
                "mode": "all",
                "wav_paths": level_saved,
            })
            continue

        median_target = statistics.median([request_metric(item, "audio_ready_s") for item in ok])
        representative = min(ok, key=lambda item: abs(request_metric(item, "audio_ready_s") - median_target))
        wav_path = output_dir / "wav_outputs" / f"representative_c{level}.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        ta.save(str(wav_path), representative["wav"], sample_rate)
        representative["wav_path"] = str(wav_path)
        saved.append({
            "concurrency": level,
            "mode": "representative",
            "wav_path": str(wav_path),
            "text": representative["text"],
            "round_index": representative["round_index"],
            "request_index": representative["request_index"],
            "audio_ready_s": round(request_metric(representative, "audio_ready_s"), 4),
            "latency_s": round(float(representative["latency_s"]), 4),
        })

    return saved


def write_markdown_report(path: Path, report: dict):
    lines = []
    lines.append("# Streaming Service Simulation")
    lines.append("")
    lines.append(f"- `impl`: `{report['impl']}`")
    lines.append(f"- `device`: `{report['device']}`")
    lines.append(f"- `language_id`: `{report['language_id']}`")
    lines.append(f"- `stagger_ms`: `{report['stagger_ms']}`")
    lines.append(f"- `rounds_per_level`: `{report['rounds_per_level']}`")
    lines.append(f"- `warmup_runs`: `{report['warmup_runs']}`")
    lines.append("")
    lines.append("Important note:")
    lines.append("")
    lines.append("- current runtime is session-based but still returns a full WAV at the end")
    lines.append("- this simulator therefore uses `t3_first_token_s` as the earliest internal readiness signal")
    lines.append("- `audio_ready_s` remains the first full-audio-ready metric")
    lines.append("")
    lines.append("## Stability Summary")
    lines.append("")
    lines.append("| Concurrency | Requests | First token mean | First token p95 | Audio ready mean | Audio ready p95 | Mean latency | Throughput |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for summary in report["levels"]:
        lines.append(
            f"| `c{summary['concurrency']}` | `{summary['num_success']}` | "
            f"`{summary['mean_first_token_s']:.4f}s` | `{summary['p95_first_token_s']:.4f}s` | "
            f"`{summary['mean_audio_ready_s']:.4f}s` | `{summary['p95_audio_ready_s']:.4f}s` | "
            f"`{summary['mean_latency_s']:.4f}s` | `{summary['audio_seconds_per_second']:.4f}` |"
        )
    lines.append("")
    lines.append("## Stage Means")
    lines.append("")
    lines.append("| Concurrency | Mean `T3` | Mean `S3` | Mean `S3 token2mel` | Mean `S3 HiFT` |")
    lines.append("|---|---:|---:|---:|---:|")
    for summary in report["levels"]:
        lines.append(
            f"| `c{summary['concurrency']}` | `{summary['mean_t3_s']:.4f}s` | `{summary['mean_s3_s']:.4f}s` | "
            f"`{summary['mean_s3_token2mel_s']:.4f}s` | `{summary['mean_s3_hift_s']:.4f}s` |"
        )
    lines.append("")
    if report["saved_audio"]:
        lines.append("## Saved Audio")
        lines.append("")
        for item in report["saved_audio"]:
            if item["mode"] == "representative":
                lines.append(
                    f"- `c{item['concurrency']}` representative: `{item['wav_path']}`"
                )
                lines.append(f"  text: `{item['text']}`")
            else:
                lines.append(f"- `c{item['concurrency']}` saved all WAVs under `wav_outputs/`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def play_saved_audio(play_command: str, saved_audio: list[dict]):
    for item in saved_audio:
        wav_path = item.get("wav_path")
        if not wav_path:
            continue
        command = play_command.format(path=wav_path, text=item.get("text", ""))
        subprocess.run(command, shell=True, check=False)


def sanitize_level_summaries(level_summaries: list[dict]) -> list[dict]:
    cleaned = []
    for summary in level_summaries:
        summary_copy = dict(summary)
        cleaned_requests = []
        for item in summary["requests"]:
            cleaned_item = dict(item)
            cleaned_item.pop("wav", None)
            cleaned_requests.append(cleaned_item)
        summary_copy["requests"] = cleaned_requests
        cleaned.append(summary_copy)
    return cleaned


def main():
    parser = argparse.ArgumentParser(
        description="Simulate a warmed multilingual TTS service under staggered multi-request load."
    )
    parser.add_argument("--impl", choices=sorted(SESSION_IMPLS), default="scheduled_turbo_s3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--hydra-checkpoint-dir")
    parser.add_argument("--hydra-speculate-k", type=int, default=3)
    parser.add_argument("--turbo-s3-checkpoint-dir")
    parser.add_argument("--enable-alignment-controller", action="store_true")
    parser.add_argument("--audio-prompt-path", required=True)
    parser.add_argument("--language-id", required=True)
    parser.add_argument("--sentences-file")
    parser.add_argument("--concurrency-levels", nargs="+", type=int, default=[1, 2, 4, 6, 8])
    parser.add_argument("--rounds-per-level", type=int, default=2)
    parser.add_argument("--stagger-ms", type=float, default=250.0)
    parser.add_argument("--output-dir", default="streaming_service_sim")
    parser.add_argument("--save-mode", choices=["representative", "all", "none"], default="representative")
    parser.add_argument("--play-command")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--warmup-text")
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=2.0)
    parser.add_argument("--min-p", type=float, default=0.05)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    sentences = load_sentences(args.sentences_file)
    warmup_text = args.warmup_text or sentences[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_start = time.perf_counter()
    model = load_model(
        args.impl,
        args.device,
        args.checkpoint_dir,
        enable_alignment_controller=args.enable_alignment_controller,
        hydra_checkpoint_dir=args.hydra_checkpoint_dir,
        hydra_speculate_k=args.hydra_speculate_k,
        turbo_s3_checkpoint_dir=args.turbo_s3_checkpoint_dir,
    )
    maybe_sync(args.device)
    load_s = time.perf_counter() - load_start

    print(f"impl={args.impl}")
    print(f"device={args.device}")
    print(f"load_s={load_s:.4f}")
    print(f"warmup_runs={args.warmup_runs}")
    print(f"stagger_ms={args.stagger_ms}")
    print(f"rounds_per_level={args.rounds_per_level}")
    print(f"save_mode={args.save_mode}")

    run_warmup(
        model=model,
        impl=args.impl,
        audio_prompt_path=args.audio_prompt_path,
        language_id=args.language_id,
        warmup_text=warmup_text,
        warmup_runs=args.warmup_runs,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        min_p=args.min_p,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )

    sentence_cursor = 0
    level_summaries = []

    for level in args.concurrency_levels:
        requests = []
        vram_state = begin_vram_measurement(args.device)
        level_wall_start = time.perf_counter()

        for round_index in range(args.rounds_per_level):
            sessions = [
                build_session(
                    model=model,
                    audio_prompt_path=args.audio_prompt_path,
                    language_id=args.language_id,
                    cfg_weight=args.cfg_weight,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    min_p=args.min_p,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                )
                for _ in range(level)
            ]

            texts = []
            for _ in range(level):
                texts.append(sentences[sentence_cursor % len(sentences)])
                sentence_cursor += 1

            round_results = [None] * level
            round_start = time.perf_counter()

            def worker(index: int):
                scheduled_at = round_start + (index * args.stagger_ms / 1000.0)
                sleep_s = scheduled_at - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)

                session = sessions[index]
                text = texts[index]
                started = time.perf_counter()
                error = None
                wav = None
                try:
                    wav = model.generate_with_session(session, text)
                    maybe_sync(args.device)
                except Exception as exc:  # noqa: BLE001
                    error = repr(exc)
                    maybe_sync(args.device)
                ended = time.perf_counter()
                profile = get_last_profile(model) if error is None else {}
                round_results[index] = {
                    "round_index": round_index,
                    "request_index": index,
                    "session_id": session.session_id,
                    "text": text,
                    "arrival_offset_s": round(index * args.stagger_ms / 1000.0, 4),
                    "started_at_s": round(started - round_start, 4),
                    "latency_s": round(ended - started, 4),
                    "num_samples": None if wav is None else int(wav.shape[-1]),
                    "error": error,
                    "wav": wav,
                    "profile": profile,
                }

            with ThreadPoolExecutor(max_workers=level) as executor:
                futures = [executor.submit(worker, idx) for idx in range(level)]
                for future in futures:
                    future.result()

            requests.extend(round_results)

        maybe_sync(args.device)
        level_wall_s = time.perf_counter() - level_wall_start
        vram_summary = finish_vram_measurement(args.device, vram_state)
        summary = summarize_requests(level, requests, level_wall_s, vram_summary)
        level_summaries.append(summary)

        print(f"concurrency={summary['concurrency']}")
        print(f"num_success={summary['num_success']}")
        print(f"mean_first_token_s={summary['mean_first_token_s']}")
        print(f"p95_first_token_s={summary['p95_first_token_s']}")
        print(f"mean_audio_ready_s={summary['mean_audio_ready_s']}")
        print(f"p95_audio_ready_s={summary['p95_audio_ready_s']}")
        print(f"mean_latency_s={summary['mean_latency_s']}")
        print(f"audio_seconds_per_second={summary['audio_seconds_per_second']}")
        print(f"mean_t3_s={summary['mean_t3_s']}")
        print(f"mean_s3_s={summary['mean_s3_s']}")
        print(f"mean_s3_token2mel_s={summary['mean_s3_token2mel_s']}")
        print(f"errors={summary['errors']}")

    saved_audio = save_representative_wavs(
        output_dir=output_dir,
        save_mode=args.save_mode,
        level_summaries=level_summaries,
        sample_rate=model.sr,
    )

    report = {
        "impl": args.impl,
        "device": args.device,
        "load_s": round(load_s, 4),
        "language_id": args.language_id,
        "audio_prompt_path": args.audio_prompt_path,
        "sentences_file": str(DEFAULT_SENTENCES_FILE if args.sentences_file is None else Path(args.sentences_file)),
        "warmup_runs": args.warmup_runs,
        "warmup_text": warmup_text,
        "rounds_per_level": args.rounds_per_level,
        "stagger_ms": args.stagger_ms,
        "save_mode": args.save_mode,
        "levels": sanitize_level_summaries(level_summaries),
        "saved_audio": saved_audio,
    }

    summary_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    serializable = json.loads(json.dumps(report, default=str))
    summary_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown_report(markdown_path, serializable)

    if args.play_command:
        play_saved_audio(args.play_command, saved_audio)

    print(f"summary_json={summary_path}")
    print(f"summary_md={markdown_path}")
    print(f"saved_audio={saved_audio}")


if __name__ == "__main__":
    main()
