import argparse
import logging
import os
import time

import torch
import torchaudio as ta

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.mtl_tts_concurrent import ChatterboxMultilingualConcurrentTTS
from chatterbox.mtl_tts_scheduled import ChatterboxMultilingualScheduledTTS
from chatterbox.mtl_tts_streaming import ChatterboxMultilingualStreamingTTS


def maybe_sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model(
    impl: str,
    device: str,
    checkpoint_dir: str | None,
    *,
    enable_alignment_controller: bool = False,
):
    if impl == "baseline":
        model_cls = ChatterboxMultilingualTTS
    elif impl == "streaming":
        model_cls = ChatterboxMultilingualStreamingTTS
    elif impl == "concurrent":
        model_cls = ChatterboxMultilingualConcurrentTTS
    else:
        model_cls = ChatterboxMultilingualScheduledTTS
    if checkpoint_dir:
        if impl == "scheduled":
            return model_cls.from_local(
                checkpoint_dir,
                device,
                enable_alignment_controller=enable_alignment_controller,
            )
        return model_cls.from_local(checkpoint_dir, device)
    if impl == "scheduled":
        return model_cls.from_pretrained(
            device,
            enable_alignment_controller=enable_alignment_controller,
        )
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


def main():
    parser = argparse.ArgumentParser(description="Compare multilingual Chatterbox runtime variants.")
    parser.add_argument("--impl", choices=["baseline", "streaming", "concurrent", "scheduled"], required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--language-id", required=True)
    parser.add_argument("--audio-prompt-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--enable-alignment-controller", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output-wav")
    parser.add_argument("--trace-shapes", action="store_true")
    args = parser.parse_args()

    configure_shape_logging(args.trace_shapes)

    load_start = time.perf_counter()
    model = load_model(
        args.impl,
        args.device,
        args.checkpoint_dir,
        enable_alignment_controller=args.enable_alignment_controller,
    )
    maybe_sync(args.device)
    load_s = time.perf_counter() - load_start

    if args.impl in {"streaming", "concurrent", "scheduled"}:
        session = model.create_session(
            audio_prompt_path=args.audio_prompt_path,
            language_id=args.language_id,
        )
        generate_fn = lambda: model.generate_with_session(session, args.text)
    else:
        generate_fn = lambda: model.generate(
            text=args.text,
            language_id=args.language_id,
            audio_prompt_path=args.audio_prompt_path,
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
    print(f"runs={args.runs}")
    print(f"latency_s={[round(x, 4) for x in latencies]}")
    if wav is not None:
        print(f"num_samples={wav.shape[-1]}")
        if args.output_wav:
            ta.save(args.output_wav, wav, model.sr)
            print(f"saved_wav={args.output_wav}")


if __name__ == "__main__":
    main()
