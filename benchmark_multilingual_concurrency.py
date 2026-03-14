import argparse
import logging
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torchaudio as ta

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.mtl_tts_concurrent import ChatterboxMultilingualConcurrentTTS
from chatterbox.mtl_tts_streaming import ChatterboxMultilingualStreamingTTS


def maybe_sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model(impl: str, device: str, checkpoint_dir: str | None):
    if impl == "baseline":
        model_cls = ChatterboxMultilingualTTS
    elif impl == "streaming":
        model_cls = ChatterboxMultilingualStreamingTTS
    else:
        model_cls = ChatterboxMultilingualConcurrentTTS
    if checkpoint_dir:
        return model_cls.from_local(checkpoint_dir, device)
    return model_cls.from_pretrained(device)


def configure_shape_logging(enabled: bool):
    if not enabled:
        os.environ.pop("CHATTERBOX_TRACE_SHAPES", None)
        return

    os.environ["CHATTERBOX_TRACE_SHAPES"] = "1"
    logger = logging.getLogger("chatterbox.shape")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_request(session_or_none, model, impl: str, text: str, language_id: str, audio_prompt_path: str | None):
    if impl in {"streaming", "concurrent"}:
        return lambda: model.generate_with_session(session_or_none, text)
    return lambda: model.generate(
        text=text,
        language_id=language_id,
        audio_prompt_path=audio_prompt_path,
    )


def run_concurrency_level(
    *,
    model,
    impl: str,
    concurrency: int,
    text: str,
    language_id: str,
    audio_prompt_path: str | None,
    device: str,
    output_dir: str | None,
):
    sessions = []
    if impl in {"streaming", "concurrent"}:
        for _ in range(concurrency):
            sessions.append(
                model.create_session(
                    audio_prompt_path=audio_prompt_path,
                    language_id=language_id,
                )
            )
    else:
        sessions = [None] * concurrency

    barrier = threading.Barrier(concurrency)
    results = [None] * concurrency

    def worker(index: int):
        fn = build_request(sessions[index], model, impl, text, language_id, audio_prompt_path)
        barrier.wait()
        started = time.perf_counter()
        error = None
        wav = None
        try:
            wav = fn()
            maybe_sync(device)
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
            maybe_sync(device)
        ended = time.perf_counter()
        results[index] = {
            "latency_s": ended - started,
            "num_samples": None if wav is None else int(wav.shape[-1]),
            "error": error,
            "wav": wav,
        }

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, idx) for idx in range(concurrency)]
        for future in futures:
            future.result()
    maybe_sync(device)
    wall_s = time.perf_counter() - wall_start

    saved_wavs = []
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(results):
            wav = item["wav"]
            if wav is None or item["error"] is not None:
                continue
            wav_path = output_path / f"{impl}_c{concurrency}_r{index}.wav"
            ta.save(str(wav_path), wav, model.sr)
            saved_wavs.append(str(wav_path))

    latencies = [item["latency_s"] for item in results if item["error"] is None]
    sample_counts = [item["num_samples"] for item in results if item["num_samples"] is not None]
    total_audio_s = sum(sample_counts) / 24000.0 if sample_counts else 0.0

    return {
        "concurrency": concurrency,
        "wall_s": wall_s,
        "request_latencies_s": [round(item["latency_s"], 4) for item in results],
        "mean_latency_s": round(statistics.mean(latencies), 4) if latencies else None,
        "p95_latency_s": round(percentile(latencies, 0.95), 4) if latencies else None,
        "num_samples": sample_counts,
        "audio_seconds_total": round(total_audio_s, 4),
        "audio_seconds_per_second": round(total_audio_s / wall_s, 4) if wall_s > 0 else None,
        "errors": [item["error"] for item in results if item["error"] is not None],
        "saved_wavs": saved_wavs,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark baseline vs streaming-safe multilingual Chatterbox under simultaneous requests.")
    parser.add_argument("--impl", choices=["baseline", "streaming", "concurrent"], required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--language-id", required=True)
    parser.add_argument("--audio-prompt-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--concurrency-levels", type=int, nargs="+", required=True)
    parser.add_argument("--trace-shapes", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    configure_shape_logging(args.trace_shapes)

    load_start = time.perf_counter()
    model = load_model(args.impl, args.device, args.checkpoint_dir)
    maybe_sync(args.device)
    load_s = time.perf_counter() - load_start

    print(f"impl={args.impl}")
    print(f"device={args.device}")
    print(f"load_s={load_s:.4f}")

    for concurrency in args.concurrency_levels:
        summary = run_concurrency_level(
            model=model,
            impl=args.impl,
            concurrency=concurrency,
            text=args.text,
            language_id=args.language_id,
            audio_prompt_path=args.audio_prompt_path,
            device=args.device,
            output_dir=args.output_dir,
        )
        print(f"concurrency={summary['concurrency']}")
        print(f"wall_s={summary['wall_s']:.4f}")
        print(f"request_latencies_s={summary['request_latencies_s']}")
        print(f"mean_latency_s={summary['mean_latency_s']}")
        print(f"p95_latency_s={summary['p95_latency_s']}")
        print(f"num_samples={summary['num_samples']}")
        print(f"audio_seconds_total={summary['audio_seconds_total']}")
        print(f"audio_seconds_per_second={summary['audio_seconds_per_second']}")
        print(f"saved_wavs={summary['saved_wavs']}")
        print(f"errors={summary['errors']}")


if __name__ == "__main__":
    main()
