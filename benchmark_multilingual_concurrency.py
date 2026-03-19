import argparse
import inspect
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
from chatterbox.mtl_tts_scheduled import ChatterboxMultilingualScheduledTTS
from chatterbox.mtl_tts_scheduled_turbo_s3 import ChatterboxMultilingualScheduledTurboS3TTS
from chatterbox.mtl_tts_streaming import ChatterboxMultilingualStreamingTTS
from chatterbox.mtl_tts_vllm_turbo_s3 import ChatterboxMultilingualVllmTurboS3TTS


def maybe_sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def get_cuda_device(device: str):
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    return torch.device(device if ":" in device else "cuda:0")


def load_model(
    impl: str,
    device: str,
    checkpoint_dir: str | None,
    *,
    base_checkpoint_dir: str | None = None,
    batching_window_ms: float = 5.0,
    text_bucket_width: int = 1,
    enable_alignment_controller: bool = False,
    hydra_checkpoint_dir: str | None = None,
    hydra_speculate_k: int = 3,
    turbo_s3_checkpoint_dir: str | None = None,
    vllm_model_dir: str | None = None,
    vllm_export_dir: str | None = None,
    vllm_prompt_builder_device: str = "cpu",
    vllm_tensor_parallel_size: int = 1,
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_enforce_eager: bool = False,
    vllm_dtype: str = "auto",
    vllm_export_copy: bool = False,
):
    if impl == "baseline":
        model_cls = ChatterboxMultilingualTTS
    elif impl == "streaming":
        model_cls = ChatterboxMultilingualStreamingTTS
    elif impl == "concurrent":
        model_cls = ChatterboxMultilingualConcurrentTTS
    elif impl == "scheduled_turbo_s3":
        model_cls = ChatterboxMultilingualScheduledTurboS3TTS
    elif impl == "vllm_turbo_s3":
        model_cls = ChatterboxMultilingualVllmTurboS3TTS
    else:
        model_cls = ChatterboxMultilingualScheduledTTS
    if checkpoint_dir:
        if impl == "scheduled":
            return model_cls.from_local(
                checkpoint_dir,
                device,
                batching_window_ms=batching_window_ms,
                text_bucket_width=text_bucket_width,
                enable_alignment_controller=enable_alignment_controller,
                hydra_checkpoint_dir=hydra_checkpoint_dir,
                hydra_speculate_k=hydra_speculate_k,
            )
        if impl == "scheduled_turbo_s3":
            return model_cls.from_local(
                checkpoint_dir,
                device,
                turbo_s3_checkpoint_dir=turbo_s3_checkpoint_dir,
                batching_window_ms=batching_window_ms,
                text_bucket_width=text_bucket_width,
                enable_alignment_controller=enable_alignment_controller,
                hydra_checkpoint_dir=hydra_checkpoint_dir,
                hydra_speculate_k=hydra_speculate_k,
            )
        if impl == "vllm_turbo_s3":
            return model_cls.from_local(
                checkpoint_dir,
                device,
                base_checkpoint_dir=base_checkpoint_dir,
                turbo_s3_checkpoint_dir=turbo_s3_checkpoint_dir,
                vllm_model_dir=vllm_model_dir,
                vllm_export_dir=vllm_export_dir,
                vllm_prompt_builder_device=vllm_prompt_builder_device,
                vllm_tensor_parallel_size=vllm_tensor_parallel_size,
                vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
                vllm_enforce_eager=vllm_enforce_eager,
                vllm_dtype=vllm_dtype,
                vllm_export_copy=vllm_export_copy,
            )
        return model_cls.from_local(checkpoint_dir, device)
    if impl == "scheduled":
        return model_cls.from_pretrained(
            device,
            batching_window_ms=batching_window_ms,
            text_bucket_width=text_bucket_width,
            enable_alignment_controller=enable_alignment_controller,
            hydra_checkpoint_dir=hydra_checkpoint_dir,
            hydra_speculate_k=hydra_speculate_k,
        )
    if impl == "scheduled_turbo_s3":
        return model_cls.from_pretrained(
            device,
            turbo_s3_checkpoint_dir=turbo_s3_checkpoint_dir,
            batching_window_ms=batching_window_ms,
            text_bucket_width=text_bucket_width,
            enable_alignment_controller=enable_alignment_controller,
            hydra_checkpoint_dir=hydra_checkpoint_dir,
            hydra_speculate_k=hydra_speculate_k,
        )
    if impl == "vllm_turbo_s3":
        return model_cls.from_pretrained(
            device,
            turbo_s3_checkpoint_dir=turbo_s3_checkpoint_dir,
            vllm_model_dir=vllm_model_dir,
            vllm_export_dir=vllm_export_dir,
            vllm_prompt_builder_device=vllm_prompt_builder_device,
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            vllm_enforce_eager=vllm_enforce_eager,
            vllm_dtype=vllm_dtype,
            vllm_export_copy=vllm_export_copy,
        )
    return model_cls.from_pretrained(device)


def describe_vllm_hydra_mode(
    *,
    impl: str,
    hydra_checkpoint_dir: str | None,
    hydra_speculate_k: int,
) -> list[str]:
    if impl != "vllm_turbo_s3":
        return []

    notes = ["hydra_mode=disabled"]
    if hydra_checkpoint_dir:
        notes.append(f"hydra_checkpoint_dir_ignored={hydra_checkpoint_dir}")
    if hydra_speculate_k != 3:
        notes.append(f"hydra_speculate_k_ignored={hydra_speculate_k}")
    return notes


def configure_shape_logging(enabled: bool, *, trace_s3_shapes: bool = False):
    if not enabled and not trace_s3_shapes:
        os.environ.pop("CHATTERBOX_TRACE_SHAPES", None)
        os.environ.pop("CHATTERBOX_TRACE_S3_SHAPES", None)
        return

    if enabled:
        os.environ["CHATTERBOX_TRACE_SHAPES"] = "1"
    else:
        os.environ.pop("CHATTERBOX_TRACE_SHAPES", None)

    if trace_s3_shapes:
        os.environ["CHATTERBOX_TRACE_S3_SHAPES"] = "1"
    else:
        os.environ.pop("CHATTERBOX_TRACE_S3_SHAPES", None)

    logger = logging.getLogger("chatterbox.shape")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)


def begin_vram_measurement(device: str):
    cuda_device = get_cuda_device(device)
    if cuda_device is None:
        return None

    maybe_sync(device)
    allocated_start = torch.cuda.memory_allocated(cuda_device)
    reserved_start = torch.cuda.memory_reserved(cuda_device)
    torch.cuda.reset_peak_memory_stats(cuda_device)
    return {
        "device": cuda_device,
        "allocated_start": allocated_start,
        "reserved_start": reserved_start,
    }


def finish_vram_measurement(device: str, state):
    if state is None:
        return {}

    cuda_device = state["device"]
    maybe_sync(device)
    peak_allocated = torch.cuda.max_memory_allocated(cuda_device)
    peak_reserved = torch.cuda.max_memory_reserved(cuda_device)
    allocated_end = torch.cuda.memory_allocated(cuda_device)
    reserved_end = torch.cuda.memory_reserved(cuda_device)

    mib = 1024 * 1024
    return {
        "vram_allocated_start_mb": round(state["allocated_start"] / mib, 1),
        "vram_reserved_start_mb": round(state["reserved_start"] / mib, 1),
        "vram_allocated_end_mb": round(allocated_end / mib, 1),
        "vram_reserved_end_mb": round(reserved_end / mib, 1),
        "vram_peak_allocated_mb": round(peak_allocated / mib, 1),
        "vram_peak_reserved_mb": round(peak_reserved / mib, 1),
        "vram_peak_allocated_delta_mb": round(max(0, peak_allocated - state["allocated_start"]) / mib, 1),
        "vram_peak_reserved_delta_mb": round(max(0, peak_reserved - state["reserved_start"]) / mib, 1),
    }


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


def _call_with_supported_kwargs(fn, **kwargs):
    signature = inspect.signature(fn)
    accepted = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return fn(**accepted)


def build_request(
    session_or_none,
    model,
    impl: str,
    text: str,
    language_id: str,
    audio_prompt_path: str | None,
    *,
    cfg_weight: float,
    temperature: float,
    repetition_penalty: float,
    min_p: float,
    top_p: float,
    max_new_tokens: int,
):
    if impl in {"streaming", "concurrent", "scheduled", "scheduled_turbo_s3", "vllm_turbo_s3"}:
        return lambda: _call_with_supported_kwargs(
            model.generate_with_session,
            session=session_or_none,
            text=text,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
    return lambda: _call_with_supported_kwargs(
        model.generate,
        text=text,
        language_id=language_id,
        audio_prompt_path=audio_prompt_path,
        cfg_weight=cfg_weight,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        min_p=min_p,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )


def get_last_profile(model) -> dict:
    if hasattr(model, "get_last_profile"):
        return model.get_last_profile() or {}

    worker = getattr(model, "worker", None)
    if worker is not None and hasattr(worker, "get_last_profile"):
        return worker.get_last_profile() or {}

    return {}


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
    cfg_weight: float,
    temperature: float,
    repetition_penalty: float,
    min_p: float,
    top_p: float,
    max_new_tokens: int,
):
    vram_state = begin_vram_measurement(device)

    sessions = []
    if impl in {"streaming", "concurrent", "scheduled", "scheduled_turbo_s3", "vllm_turbo_s3"}:
        for _ in range(concurrency):
            sessions.append(
                model.create_session(
                    audio_prompt_path=audio_prompt_path,
                    language_id=language_id,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                )
            )
    else:
        sessions = [None] * concurrency

    barrier = threading.Barrier(concurrency)
    results = [None] * concurrency

    def worker(index: int):
        fn = build_request(
            sessions[index],
            model,
            impl,
            text,
            language_id,
            audio_prompt_path,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
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
        profile = get_last_profile(model) if error is None else {}
        results[index] = {
            "latency_s": ended - started,
            "num_samples": None if wav is None else int(wav.shape[-1]),
            "error": error,
            "wav": wav,
            "profile": profile,
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
    vram_summary = finish_vram_measurement(device, vram_state)
    profile_keys = sorted(
        {
            key
            for item in results
            for key in item.get("profile", {}).keys()
        }
    )
    profile_summary = {}
    for key in profile_keys:
        values = [item["profile"].get(key) for item in results if item["error"] is None and key in item.get("profile", {})]
        if not values:
            continue
        rounded_values = [round(float(value), 4) for value in values]
        profile_summary[f"stage_{key}"] = rounded_values
        profile_summary[f"stage_{key}_mean"] = round(statistics.mean(values), 4)

    summary = {
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
    summary.update(vram_summary)
    summary.update(profile_summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark multilingual Chatterbox runtime variants under simultaneous requests.")
    parser.add_argument("--impl", choices=["baseline", "streaming", "concurrent", "scheduled", "scheduled_turbo_s3", "vllm_turbo_s3"], required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--language-id", required=True)
    parser.add_argument("--audio-prompt-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--base-checkpoint-dir")
    parser.add_argument("--concurrency-levels", type=int, nargs="+", required=True)
    parser.add_argument("--enable-alignment-controller", action="store_true")
    parser.add_argument("--batching-window-ms", type=float, default=5.0)
    parser.add_argument("--text-bucket-width", type=int, default=1)
    parser.add_argument("--hydra-checkpoint-dir")
    parser.add_argument("--hydra-speculate-k", type=int, default=3)
    parser.add_argument("--turbo-s3-checkpoint-dir")
    parser.add_argument("--vllm-model-dir")
    parser.add_argument("--vllm-export-dir")
    parser.add_argument("--vllm-prompt-builder-device", default="cpu")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--vllm-dtype", default="auto")
    parser.add_argument("--vllm-export-copy", action="store_true")
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=2.0)
    parser.add_argument("--min-p", type=float, default=0.05)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--trace-shapes", action="store_true")
    parser.add_argument("--trace-s3-shapes", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    configure_shape_logging(args.trace_shapes, trace_s3_shapes=args.trace_s3_shapes)

    load_start = time.perf_counter()
    model = load_model(
        args.impl,
        args.device,
        args.checkpoint_dir,
        base_checkpoint_dir=args.base_checkpoint_dir,
        batching_window_ms=args.batching_window_ms,
        text_bucket_width=args.text_bucket_width,
        enable_alignment_controller=args.enable_alignment_controller,
        hydra_checkpoint_dir=args.hydra_checkpoint_dir,
        hydra_speculate_k=args.hydra_speculate_k,
        turbo_s3_checkpoint_dir=args.turbo_s3_checkpoint_dir,
        vllm_model_dir=args.vllm_model_dir,
        vllm_export_dir=args.vllm_export_dir,
        vllm_prompt_builder_device=args.vllm_prompt_builder_device,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enforce_eager=args.vllm_enforce_eager,
        vllm_dtype=args.vllm_dtype,
        vllm_export_copy=args.vllm_export_copy,
    )
    maybe_sync(args.device)
    load_s = time.perf_counter() - load_start

    print(f"impl={args.impl}")
    print(f"device={args.device}")
    print(f"load_s={load_s:.4f}")
    if args.impl in {"scheduled", "scheduled_turbo_s3"}:
        print(f"hydra_checkpoint_dir={args.hydra_checkpoint_dir}")
        print(f"hydra_speculate_k={args.hydra_speculate_k}")
        print(f"batching_window_ms={args.batching_window_ms}")
        print(f"text_bucket_width={args.text_bucket_width}")
    if args.impl == "scheduled_turbo_s3":
        print(f"turbo_s3_checkpoint_dir={args.turbo_s3_checkpoint_dir}")
    if args.impl == "vllm_turbo_s3":
        print(f"base_checkpoint_dir={args.base_checkpoint_dir}")
        print(f"turbo_s3_checkpoint_dir={args.turbo_s3_checkpoint_dir}")
        print(f"vllm_model_dir={args.vllm_model_dir}")
        print(f"vllm_export_dir={args.vllm_export_dir}")
        print(f"vllm_prompt_builder_device={args.vllm_prompt_builder_device}")
        print(f"vllm_tensor_parallel_size={args.vllm_tensor_parallel_size}")
        print(f"vllm_gpu_memory_utilization={args.vllm_gpu_memory_utilization}")
        print(f"vllm_enforce_eager={args.vllm_enforce_eager}")
        print(f"vllm_dtype={args.vllm_dtype}")
        for note in describe_vllm_hydra_mode(
            impl=args.impl,
            hydra_checkpoint_dir=args.hydra_checkpoint_dir,
            hydra_speculate_k=args.hydra_speculate_k,
        ):
            print(note)
    print(f"cfg_weight={args.cfg_weight}")
    print(f"temperature={args.temperature}")
    print(f"repetition_penalty={args.repetition_penalty}")
    print(f"min_p={args.min_p}")
    print(f"top_p={args.top_p}")
    print(f"max_new_tokens={args.max_new_tokens}")

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
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            min_p=args.min_p,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"concurrency={summary['concurrency']}")
        print(f"wall_s={summary['wall_s']:.4f}")
        print(f"request_latencies_s={summary['request_latencies_s']}")
        print(f"mean_latency_s={summary['mean_latency_s']}")
        print(f"p95_latency_s={summary['p95_latency_s']}")
        print(f"num_samples={summary['num_samples']}")
        print(f"audio_seconds_total={summary['audio_seconds_total']}")
        print(f"audio_seconds_per_second={summary['audio_seconds_per_second']}")
        for key in [
            "vram_allocated_start_mb",
            "vram_reserved_start_mb",
            "vram_allocated_end_mb",
            "vram_reserved_end_mb",
            "vram_peak_allocated_mb",
            "vram_peak_reserved_mb",
            "vram_peak_allocated_delta_mb",
            "vram_peak_reserved_delta_mb",
        ]:
            if key in summary:
                print(f"{key}={summary[key]}")
        for key in sorted(summary.keys()):
            if key.startswith("stage_"):
                print(f"{key}={summary[key]}")
        print(f"saved_wavs={summary['saved_wavs']}")
        print(f"errors={summary['errors']}")


if __name__ == "__main__":
    main()
