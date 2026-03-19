import argparse
import inspect
import logging
import os
import time

import torch

from chatterbox.audio_utils import save_wav

def maybe_sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def resolve_model_cls(impl: str):
    if impl == "baseline":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        return ChatterboxMultilingualTTS
    if impl == "streaming":
        from chatterbox.mtl_tts_streaming import ChatterboxMultilingualStreamingTTS

        return ChatterboxMultilingualStreamingTTS
    if impl == "concurrent":
        from chatterbox.mtl_tts_concurrent import ChatterboxMultilingualConcurrentTTS

        return ChatterboxMultilingualConcurrentTTS
    if impl == "scheduled_turbo_s3":
        from chatterbox.mtl_tts_scheduled_turbo_s3 import ChatterboxMultilingualScheduledTurboS3TTS

        return ChatterboxMultilingualScheduledTurboS3TTS
    if impl == "vllm_turbo_s3":
        from chatterbox.mtl_tts_vllm_turbo_s3 import ChatterboxMultilingualVllmTurboS3TTS

        return ChatterboxMultilingualVllmTurboS3TTS

    from chatterbox.mtl_tts_scheduled import ChatterboxMultilingualScheduledTTS

    return ChatterboxMultilingualScheduledTTS


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
    vllm_gpu_memory_utilization: float = 0.5,
    vllm_enforce_eager: bool = False,
    vllm_dtype: str = "auto",
    vllm_max_model_len: int = 2048,
    vllm_enable_prefix_caching: bool = False,
    vllm_export_copy: bool = False,
):
    model_cls = resolve_model_cls(impl)
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
                vllm_max_model_len=vllm_max_model_len,
                vllm_enable_prefix_caching=vllm_enable_prefix_caching,
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
            vllm_max_model_len=vllm_max_model_len,
            vllm_enable_prefix_caching=vllm_enable_prefix_caching,
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


def _call_with_supported_kwargs(fn, **kwargs):
    signature = inspect.signature(fn)
    accepted = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return fn(**accepted)


def main():
    parser = argparse.ArgumentParser(description="Compare multilingual Chatterbox runtime variants.")
    parser.add_argument("--impl", choices=["baseline", "streaming", "concurrent", "scheduled", "scheduled_turbo_s3", "vllm_turbo_s3"], required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--language-id", required=True)
    parser.add_argument("--audio-prompt-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--base-checkpoint-dir")
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
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--vllm-dtype", default="auto")
    parser.add_argument("--vllm-max-model-len", type=int, default=2048)
    parser.add_argument("--vllm-enable-prefix-caching", action="store_true")
    parser.add_argument("--no-vllm-prefix-caching", action="store_true")
    parser.add_argument("--vllm-export-copy", action="store_true")
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=2.0)
    parser.add_argument("--min-p", type=float, default=0.05)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output-wav")
    parser.add_argument("--trace-shapes", action="store_true")
    args = parser.parse_args()

    configure_shape_logging(args.trace_shapes)

    model = None
    try:
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
            vllm_max_model_len=args.vllm_max_model_len,
            vllm_enable_prefix_caching=(
                args.vllm_enable_prefix_caching and not args.no_vllm_prefix_caching
            ),
            vllm_export_copy=args.vllm_export_copy,
        )
        maybe_sync(args.device)
        load_s = time.perf_counter() - load_start

        if args.impl in {"streaming", "concurrent", "scheduled", "scheduled_turbo_s3", "vllm_turbo_s3"}:
            session = model.create_session(
                audio_prompt_path=args.audio_prompt_path,
                language_id=args.language_id,
            )
            generate_fn = lambda: _call_with_supported_kwargs(
                model.generate_with_session,
                session=session,
                text=args.text,
                cfg_weight=args.cfg_weight,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                min_p=args.min_p,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            generate_fn = lambda: _call_with_supported_kwargs(
                model.generate,
                text=args.text,
                language_id=args.language_id,
                audio_prompt_path=args.audio_prompt_path,
                cfg_weight=args.cfg_weight,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                min_p=args.min_p,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )

        for _ in range(args.warmup_runs):
            _ = generate_fn()
            maybe_sync(args.device)

        latencies = []
        wav = None
        for _ in range(args.runs):
            start = time.perf_counter()
            wav = generate_fn()
            maybe_sync(args.device)
            latencies.append(time.perf_counter() - start)

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
            print(f"vllm_max_model_len={args.vllm_max_model_len}")
            print(
                "vllm_enable_prefix_caching="
                f"{args.vllm_enable_prefix_caching and not args.no_vllm_prefix_caching}"
            )
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
        print(f"runs={args.runs}")
        print(f"latency_s={[round(x, 4) for x in latencies]}")
        if wav is not None:
            print(f"num_samples={wav.shape[-1]}")
            if args.output_wav:
                save_wav(args.output_wav, wav, model.sr)
                print(f"saved_wav={args.output_wav}")
    finally:
        if model is not None and hasattr(model, "close"):
            model.close()


if __name__ == "__main__":
    main()
