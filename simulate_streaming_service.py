import argparse
import json
import shlex
import shutil
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from chatterbox.audio_utils import save_wav
from benchmark_multilingual_concurrency import (
    begin_vram_measurement,
    describe_vllm_hydra_mode,
    finish_vram_measurement,
    get_last_profile,
    load_model,
    maybe_sync,
    percentile,
)


DEFAULT_SENTENCES_FILE = Path(__file__).with_name("arabic_streaming_sentences.txt")
SESSION_IMPLS = {"streaming", "concurrent", "scheduled", "scheduled_turbo_s3", "vllm_turbo_s3"}


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


def histogram(values: list[int | str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def estimate_vllm_text_len(model, text: str, language_id: str) -> int:
    worker = getattr(model, "worker", None)
    tokenizer = getattr(worker, "tokenizer", None)
    if tokenizer is not None:
        try:
            tokens = tokenizer.text_to_tokens(
                text,
                language_id=language_id.lower() if language_id else None,
            )
            return int(tokens.numel())
        except Exception:
            pass
    return max(1, len(text.split()))


def estimate_vllm_prompt_len(session) -> int:
    t3_cond = getattr(getattr(session, "conditionals", None), "t3", None)
    prompt_tokens = getattr(t3_cond, "cond_prompt_speech_tokens", None)
    if prompt_tokens is None:
        return 0
    return int(prompt_tokens.shape[-1])


def group_vllm_pending_requests(
    pending_requests: list[dict],
    *,
    text_bucket_width: int,
) -> list[tuple[tuple[int, int], list[dict]]]:
    if not pending_requests:
        return []

    grouped_by_prompt: dict[int, list[dict]] = {}
    for item in pending_requests:
        grouped_by_prompt.setdefault(item["prompt_len"], []).append(item)

    grouped: list[tuple[tuple[int, int], list[dict]]] = []
    width = int(text_bucket_width)
    for prompt_len, items in sorted(grouped_by_prompt.items(), key=lambda item: item[0]):
        if width <= 0:
            max_text_len = max(item["text_len"] for item in items)
            grouped.append(((max_text_len, prompt_len), sorted(items, key=lambda item: item["arrival_offset_s"])))
            continue

        ordered = sorted(items, key=lambda item: item["text_len"])
        current_items: list[dict] = []
        current_min = 0
        current_max = 0
        for item in ordered:
            text_len = item["text_len"]
            if not current_items:
                current_items = [item]
                current_min = text_len
                current_max = text_len
                continue

            next_min = min(current_min, text_len)
            next_max = max(current_max, text_len)
            if (next_max - next_min) >= width:
                grouped.append(((current_max, prompt_len), current_items))
                current_items = [item]
                current_min = text_len
                current_max = text_len
                continue

            current_items.append(item)
            current_min = next_min
            current_max = next_max

        if current_items:
            grouped.append(((current_max, prompt_len), current_items))

    return grouped


def run_vllm_service_round(
    *,
    model,
    sessions: list,
    texts: list[str],
    language_id: str,
    device: str,
    round_index: int,
    stagger_ms: float,
    batching_window_ms: float,
    text_bucket_width: int,
    cfg_weight: float,
    temperature: float,
    repetition_penalty: float,
    min_p: float,
    top_p: float,
    max_new_tokens: int,
) -> tuple[list[dict], float]:
    scheduled_requests = []
    for index, (session, text) in enumerate(zip(sessions, texts)):
        scheduled_requests.append(
            {
                "round_index": round_index,
                "request_index": index,
                "session": session,
                "text": text,
                "arrival_offset_s": index * stagger_ms / 1000.0,
                "text_len": estimate_vllm_text_len(model, text, language_id),
                "prompt_len": estimate_vllm_prompt_len(session),
            }
        )

    pending_requests: list[dict] = []
    next_request_idx = 0
    service_time_s = 0.0
    batching_window_s = batching_window_ms / 1000.0
    round_results: list[dict | None] = [None] * len(scheduled_requests)

    while next_request_idx < len(scheduled_requests) or pending_requests:
        if not pending_requests and next_request_idx < len(scheduled_requests):
            service_time_s = max(
                service_time_s,
                scheduled_requests[next_request_idx]["arrival_offset_s"],
            )

        while (
            next_request_idx < len(scheduled_requests)
            and scheduled_requests[next_request_idx]["arrival_offset_s"] <= service_time_s
        ):
            pending_requests.append(scheduled_requests[next_request_idx])
            next_request_idx += 1

        batching_deadline_s = service_time_s + batching_window_s
        while (
            next_request_idx < len(scheduled_requests)
            and scheduled_requests[next_request_idx]["arrival_offset_s"] <= batching_deadline_s
        ):
            pending_requests.append(scheduled_requests[next_request_idx])
            next_request_idx += 1

        grouped_pending = group_vllm_pending_requests(
            pending_requests,
            text_bucket_width=text_bucket_width,
        )
        if not grouped_pending:
            continue

        group_key, cohort = max(
            grouped_pending,
            key=lambda item: (
                len(item[1]),
                -min(entry["arrival_offset_s"] for entry in item[1]),
            ),
        )
        cohort_ids = {item["request_index"] for item in cohort}
        pending_requests = [
            item for item in pending_requests if item["request_index"] not in cohort_ids
        ]
        active_cohorts_at_admit = max(0, len(grouped_pending) - 1)
        cohort_size = len(cohort)

        batch_started = time.perf_counter()
        batch_error = None
        batch_results = []
        try:
            batch_results = model.generate_many_with_sessions(
                [item["session"] for item in cohort],
                [item["text"] for item in cohort],
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            maybe_sync(device)
        except Exception as exc:  # noqa: BLE001
            batch_error = repr(exc)
            maybe_sync(device)
        batch_wall_s = time.perf_counter() - batch_started
        batch_started_at_s = service_time_s
        service_time_s = batch_started_at_s + batch_wall_s

        if batch_error is not None:
            for item in cohort:
                queue_wait_s = max(0.0, batch_started_at_s - item["arrival_offset_s"])
                round_results[item["request_index"]] = {
                    "round_index": item["round_index"],
                    "request_index": item["request_index"],
                    "session_id": item["session"].session_id,
                    "text": item["text"],
                    "arrival_offset_s": round(item["arrival_offset_s"], 4),
                    "started_at_s": round(batch_started_at_s, 4),
                    "latency_s": round(queue_wait_s + batch_wall_s, 4),
                    "num_samples": None,
                    "error": batch_error,
                    "wav": None,
                    "profile": {},
                }
            continue

        for item, batch_item in zip(cohort, batch_results):
            wav = batch_item["wav"]
            profile = dict(batch_item["profile"])
            queue_wait_s = max(0.0, batch_started_at_s - item["arrival_offset_s"])
            profile["t3_wait_s"] = float(profile.get("t3_wait_s", 0.0)) + queue_wait_s
            profile["audio_ready_s"] = queue_wait_s + float(profile.get("audio_ready_s", batch_wall_s))
            profile["t3_batch_text_len"] = float(item["text_len"])
            profile["t3_batch_prompt_len"] = float(item["prompt_len"])
            profile["t3_group_text_len"] = float(group_key[0])
            profile["t3_group_prompt_len"] = float(group_key[1])
            profile["t3_admission_cohort_size"] = float(cohort_size)
            profile["t3_active_cohorts_at_admit"] = float(active_cohorts_at_admit)
            profile["t3_admission_singleton"] = 1.0 if cohort_size == 1 else 0.0

            round_results[item["request_index"]] = {
                "round_index": item["round_index"],
                "request_index": item["request_index"],
                "session_id": item["session"].session_id,
                "text": item["text"],
                "arrival_offset_s": round(item["arrival_offset_s"], 4),
                "started_at_s": round(batch_started_at_s, 4),
                "latency_s": round(float(profile["audio_ready_s"]), 4),
                "num_samples": None if wav is None else int(wav.shape[-1]),
                "error": None,
                "wav": wav,
                "profile": profile,
            }

    finalized_results = [item for item in round_results if item is not None]
    round_wall_s = 0.0
    if finalized_results:
        round_wall_s = max(
            float(item["arrival_offset_s"]) + float(item["latency_s"])
            for item in finalized_results
        )
    return finalized_results, round_wall_s


def summarize_requests(level: int, requests: list[dict], wall_s: float, vram_summary: dict) -> dict:
    ok = [item for item in requests if item["error"] is None]
    errors = [item["error"] for item in requests if item["error"] is not None]

    latencies = [float(item["latency_s"]) for item in ok]
    first_tokens = [request_metric(item, "t3_first_token_s") for item in ok]
    audio_ready = [request_metric(item, "audio_ready_s") for item in ok]
    t3_wait = [request_metric(item, "t3_wait_s") for item in ok]
    t3_active = [request_metric(item, "t3_active_s") for item in ok]
    t3_total = [request_metric(item, "t3_s") for item in ok]
    t3_acceptance = [request_metric(item, "t3_acceptance_rate") for item in ok if "t3_acceptance_rate" in item["profile"]]
    t3_rounds = [request_metric(item, "t3_rounds") for item in ok if "t3_rounds" in item["profile"]]
    t3_cohort_sizes = [int(request_metric(item, "t3_admission_cohort_size")) for item in ok if "t3_admission_cohort_size" in item["profile"]]
    t3_active_cohorts_at_admit = [request_metric(item, "t3_active_cohorts_at_admit") for item in ok if "t3_active_cohorts_at_admit" in item["profile"]]
    t3_text_lens = [int(request_metric(item, "t3_batch_text_len")) for item in ok if "t3_batch_text_len" in item["profile"]]
    t3_prompt_lens = [int(request_metric(item, "t3_batch_prompt_len")) for item in ok if "t3_batch_prompt_len" in item["profile"]]
    t3_group_text_lens = [int(request_metric(item, "t3_group_text_len")) for item in ok if "t3_group_text_len" in item["profile"]]
    t3_group_prompt_lens = [int(request_metric(item, "t3_group_prompt_len")) for item in ok if "t3_group_prompt_len" in item["profile"]]
    s3_total = [request_metric(item, "s3_s") for item in ok]
    s3_token2mel = [request_metric(item, "s3_token2mel_s") for item in ok]
    s3_hift = [request_metric(item, "s3_hift_s") for item in ok]
    num_samples = [int(item["num_samples"]) for item in ok if item["num_samples"] is not None]
    total_audio_s = sum(num_samples) / 24000.0 if num_samples else 0.0
    batch_key_hist = histogram([f"{text_len}/{prompt_len}" for text_len, prompt_len in zip(t3_text_lens, t3_prompt_lens)])
    group_key_hist = histogram([f"{text_len}/{prompt_len}" for text_len, prompt_len in zip(t3_group_text_lens, t3_group_prompt_lens)])
    admission_cohort_hist = histogram(t3_cohort_sizes)
    singleton_fraction = 0.0 if not t3_cohort_sizes else (sum(1 for size in t3_cohort_sizes if size == 1) / len(t3_cohort_sizes))

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
        "mean_t3_wait_s": round(mean_or_zero(t3_wait), 4),
        "mean_t3_active_s": round(mean_or_zero(t3_active), 4),
        "mean_t3_s": round(mean_or_zero(t3_total), 4),
        "mean_t3_acceptance_rate": round(mean_or_zero(t3_acceptance), 4),
        "mean_t3_rounds": round(mean_or_zero(t3_rounds), 4),
        "mean_t3_active_cohorts_at_admit": round(mean_or_zero(t3_active_cohorts_at_admit), 4),
        "admission_cohort_size_hist": admission_cohort_hist,
        "batch_key_hist": batch_key_hist,
        "group_key_hist": group_key_hist,
        "singleton_request_fraction": round(float(singleton_fraction), 4),
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
                save_wav(wav_path, item["wav"], sample_rate)
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
        save_wav(wav_path, representative["wav"], sample_rate)
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
    lines.append(f"- `batching_window_ms`: `{report['batching_window_ms']}`")
    lines.append(f"- `text_bucket_width`: `{report['text_bucket_width']}`")
    if report["impl"] == "vllm_turbo_s3":
        lines.append(f"- `vllm_enforce_eager`: `{report.get('vllm_enforce_eager', False)}`")
        lines.append(
            f"- `vllm_prompt_len_only_grouping`: `{report.get('vllm_prompt_len_only_grouping', False)}`"
        )
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
    lines.append("| Concurrency | Mean `T3 wait` | Mean `T3 active` | Mean `T3 total` | Mean Hydra acceptance | Mean `S3` | Mean `S3 token2mel` | Mean `S3 HiFT` |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for summary in report["levels"]:
        lines.append(
            f"| `c{summary['concurrency']}` | `{summary['mean_t3_wait_s']:.4f}s` | `{summary['mean_t3_active_s']:.4f}s` | "
            f"`{summary['mean_t3_s']:.4f}s` | `{summary['mean_t3_acceptance_rate']:.4f}` | `{summary['mean_s3_s']:.4f}s` | "
            f"`{summary['mean_s3_token2mel_s']:.4f}s` | `{summary['mean_s3_hift_s']:.4f}s` |"
        )
    lines.append("")
    lines.append("## Admission Forensics")
    lines.append("")
    lines.append("| Concurrency | Mean active cohorts at admit | Singleton request fraction | Admission cohort size hist | Request key hist | Group key hist |")
    lines.append("|---|---:|---:|---|---|---|")
    for summary in report["levels"]:
        lines.append(
            f"| `c{summary['concurrency']}` | `{summary['mean_t3_active_cohorts_at_admit']:.4f}` | "
            f"`{summary['singleton_request_fraction']:.4f}` | `{summary['admission_cohort_size_hist']}` | "
            f"`{summary['batch_key_hist']}` | `{summary['group_key_hist']}` |"
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


def resolve_play_command(play_command: str | None) -> tuple[str | None, str | None]:
    if not play_command:
        return None, None
    if play_command == "auto":
        candidates = [
            "ffplay -nodisp -autoexit -loglevel error {path}",
            "mpv --no-terminal --really-quiet {path}",
            "paplay {path}",
            "aplay {path}",
        ]
        for candidate in candidates:
            executable = shlex.split(candidate)[0]
            if shutil.which(executable):
                return candidate, None
        return None, "No supported audio player found in PATH (`ffplay`, `mpv`, `paplay`, `aplay`)."

    executable = shlex.split(play_command)[0]
    if shutil.which(executable) is None:
        return None, f"Playback command not found in PATH: `{executable}`."
    return play_command, None


def play_saved_audio(play_command: str, saved_audio: list[dict]) -> list[str]:
    errors = []
    for item in saved_audio:
        wav_path = item.get("wav_path")
        if not wav_path:
            continue
        command = play_command.format(path=wav_path, text=item.get("text", ""))
        result = subprocess.run(shlex.split(command), check=False)
        if result.returncode != 0:
            errors.append(f"`{command}` exited with code {result.returncode}")
    return errors


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
    parser.add_argument("--base-checkpoint-dir")
    parser.add_argument("--hydra-checkpoint-dir")
    parser.add_argument("--hydra-speculate-k", type=int, default=3)
    parser.add_argument("--turbo-s3-checkpoint-dir")
    parser.add_argument("--vllm-model-dir")
    parser.add_argument("--vllm-export-dir")
    parser.add_argument("--vllm-prompt-builder-device", default="cpu")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--allow-vllm-compiled-service-sim", action="store_true")
    parser.add_argument("--vllm-dtype", default="auto")
    parser.add_argument("--vllm-max-model-len", type=int, default=2048)
    parser.add_argument("--vllm-enable-prefix-caching", action="store_true")
    parser.add_argument("--no-vllm-prefix-caching", action="store_true")
    parser.add_argument("--vllm-export-copy", action="store_true")
    parser.add_argument("--enable-alignment-controller", action="store_true")
    parser.add_argument("--batching-window-ms", type=float, default=5.0)
    parser.add_argument("--text-bucket-width", type=int, default=1)
    parser.add_argument("--allow-vllm-text-bucketing", action="store_true")
    parser.add_argument("--audio-prompt-path", required=True)
    parser.add_argument("--language-id", required=True)
    parser.add_argument("--sentences-file")
    parser.add_argument("--concurrency-levels", nargs="+", type=int, default=[1, 2, 4, 6, 8])
    parser.add_argument("--rounds-per-level", type=int, default=2)
    parser.add_argument("--stagger-ms", type=float, default=250.0)
    parser.add_argument("--output-dir", default="streaming_service_sim")
    parser.add_argument("--save-mode", choices=["representative", "all", "none"], default="representative")
    parser.add_argument("--play-command")
    parser.add_argument("--print-forensics", action="store_true")
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
    effective_vllm_enforce_eager = args.vllm_enforce_eager
    if args.impl == "vllm_turbo_s3" and not args.allow_vllm_compiled_service_sim:
        effective_vllm_enforce_eager = True
    effective_text_bucket_width = args.text_bucket_width
    if args.impl == "vllm_turbo_s3" and not args.allow_vllm_text_bucketing:
        effective_text_bucket_width = 0

    model = None
    try:
        load_start = time.perf_counter()
        model = load_model(
            args.impl,
            args.device,
            args.checkpoint_dir,
            base_checkpoint_dir=args.base_checkpoint_dir,
            batching_window_ms=args.batching_window_ms,
            text_bucket_width=effective_text_bucket_width,
            enable_alignment_controller=args.enable_alignment_controller,
            hydra_checkpoint_dir=args.hydra_checkpoint_dir,
            hydra_speculate_k=args.hydra_speculate_k,
            turbo_s3_checkpoint_dir=args.turbo_s3_checkpoint_dir,
            vllm_model_dir=args.vllm_model_dir,
            vllm_export_dir=args.vllm_export_dir,
            vllm_prompt_builder_device=args.vllm_prompt_builder_device,
            vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_enforce_eager=effective_vllm_enforce_eager,
            vllm_dtype=args.vllm_dtype,
            vllm_max_model_len=args.vllm_max_model_len,
            vllm_enable_prefix_caching=(
                args.vllm_enable_prefix_caching and not args.no_vllm_prefix_caching
            ),
            vllm_export_copy=args.vllm_export_copy,
        )
        maybe_sync(args.device)
        load_s = time.perf_counter() - load_start

        print(f"impl={args.impl}")
        print(f"device={args.device}")
        print(f"load_s={load_s:.4f}")
        if args.impl == "vllm_turbo_s3":
            print(f"base_checkpoint_dir={args.base_checkpoint_dir}")
            print(f"vllm_gpu_memory_utilization={args.vllm_gpu_memory_utilization}")
            print(f"vllm_max_model_len={args.vllm_max_model_len}")
            print(f"vllm_enforce_eager={effective_vllm_enforce_eager}")
            if effective_vllm_enforce_eager and not args.vllm_enforce_eager:
                print("vllm_enforce_eager_reason=mixed_shape_service_sim_default")
            print(
                "vllm_enable_prefix_caching="
                f"{args.vllm_enable_prefix_caching and not args.no_vllm_prefix_caching}"
            )
        print(f"warmup_runs={args.warmup_runs}")
        print(f"stagger_ms={args.stagger_ms}")
        print(f"batching_window_ms={args.batching_window_ms}")
        print(f"text_bucket_width={effective_text_bucket_width}")
        if args.impl == "vllm_turbo_s3":
            print(f"vllm_prompt_len_only_grouping={not args.allow_vllm_text_bucketing}")
        print(f"rounds_per_level={args.rounds_per_level}")
        print(f"save_mode={args.save_mode}")
        for note in describe_vllm_hydra_mode(
            impl=args.impl,
            hydra_checkpoint_dir=args.hydra_checkpoint_dir,
            hydra_speculate_k=args.hydra_speculate_k,
        ):
            print(note)

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
            simulated_level_wall_s = 0.0

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

                if args.impl == "vllm_turbo_s3" and hasattr(model, "generate_many_with_sessions"):
                    round_results, round_wall_s = run_vllm_service_round(
                        model=model,
                        sessions=sessions,
                        texts=texts,
                        language_id=args.language_id,
                        device=args.device,
                        round_index=round_index,
                        stagger_ms=args.stagger_ms,
                        batching_window_ms=args.batching_window_ms,
                        text_bucket_width=effective_text_bucket_width,
                        cfg_weight=args.cfg_weight,
                        temperature=args.temperature,
                        repetition_penalty=args.repetition_penalty,
                        min_p=args.min_p,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                    )
                    simulated_level_wall_s += round_wall_s
                    requests.extend(round_results)
                else:
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
            if args.impl == "vllm_turbo_s3" and hasattr(model, "generate_many_with_sessions"):
                level_wall_s = simulated_level_wall_s
            else:
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
            print(f"mean_t3_wait_s={summary['mean_t3_wait_s']}")
            print(f"mean_t3_active_s={summary['mean_t3_active_s']}")
            print(f"mean_t3_s={summary['mean_t3_s']}")
            print(f"mean_t3_acceptance_rate={summary['mean_t3_acceptance_rate']}")
            print(f"mean_t3_rounds={summary['mean_t3_rounds']}")
            print(f"mean_t3_active_cohorts_at_admit={summary['mean_t3_active_cohorts_at_admit']}")
            print(f"singleton_request_fraction={summary['singleton_request_fraction']}")
            print(f"admission_cohort_size_hist={summary['admission_cohort_size_hist']}")
            print(f"batch_key_hist={summary['batch_key_hist']}")
            print(f"group_key_hist={summary['group_key_hist']}")
            print(f"mean_s3_s={summary['mean_s3_s']}")
            print(f"mean_s3_token2mel_s={summary['mean_s3_token2mel_s']}")
            print(f"errors={summary['errors']}")
            if args.print_forensics:
                print("forensics_requests=[")
                for item in summary["requests"]:
                    if item["error"] is not None:
                        continue
                    profile = item["profile"]
                    print(
                        "  {"
                        f"'round': {item['round_index']}, "
                        f"'request': {item['request_index']}, "
                        f"'arrival_offset_s': {item['arrival_offset_s']}, "
                        f"'text': {item['text']!r}, "
                        f"'batch_key': ({int(profile.get('t3_batch_text_len', 0))}, {int(profile.get('t3_batch_prompt_len', 0))}), "
                        f"'group_key': ({int(profile.get('t3_group_text_len', 0))}, {int(profile.get('t3_group_prompt_len', 0))}), "
                        f"'admission_cohort_size': {int(profile.get('t3_admission_cohort_size', 0))}, "
                        f"'active_cohorts_at_admit': {float(profile.get('t3_active_cohorts_at_admit', 0.0)):.4f}, "
                        f"'t3_wait_s': {float(profile.get('t3_wait_s', 0.0)):.4f}, "
                        f"'t3_active_s': {float(profile.get('t3_active_s', 0.0)):.4f}, "
                        f"'t3_s': {float(profile.get('t3_s', 0.0)):.4f}"
                        "},"
                    )
                print("]")

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
            "batching_window_ms": args.batching_window_ms,
            "text_bucket_width": effective_text_bucket_width,
            "vllm_enforce_eager": effective_vllm_enforce_eager,
            "vllm_prompt_len_only_grouping": (
                args.impl == "vllm_turbo_s3" and not args.allow_vllm_text_bucketing
            ),
            "save_mode": args.save_mode,
            "levels": sanitize_level_summaries(level_summaries),
            "saved_audio": saved_audio,
        }

        summary_path = output_dir / "summary.json"
        markdown_path = output_dir / "summary.md"
        serializable = json.loads(json.dumps(report, default=str))
        summary_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_markdown_report(markdown_path, serializable)

        playback_errors = []
        resolved_play_command = None
        if args.play_command:
            resolved_play_command, playback_error = resolve_play_command(args.play_command)
            if playback_error is not None:
                playback_errors.append(playback_error)
            elif resolved_play_command is not None:
                playback_errors.extend(play_saved_audio(resolved_play_command, saved_audio))

        print(f"summary_json={summary_path}")
        print(f"summary_md={markdown_path}")
        print(f"saved_audio={saved_audio}")
        if resolved_play_command is not None:
            print(f"play_command={resolved_play_command}")
        if playback_errors:
            print(f"playback_errors={playback_errors}")
    finally:
        if model is not None and hasattr(model, "close"):
            model.close()


if __name__ == "__main__":
    main()
