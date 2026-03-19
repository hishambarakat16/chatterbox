import argparse
import json
from pathlib import Path

from chatterbox.vllm_t3_bridge import create_vllm_engine, register_vllm_t3_model


def main():
    parser = argparse.ArgumentParser(description="Preflight a vLLM-exported T3 speech-decoder package.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--skip-engine-init", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config_path = model_dir / "config.json"
    export_meta_path = model_dir / "chatterbox_vllm_export.json"
    weights_path = model_dir / "model.safetensors"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    if not export_meta_path.exists():
        raise FileNotFoundError(f"Missing chatterbox_vllm_export.json: {export_meta_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing model.safetensors: {weights_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    meta = json.loads(export_meta_path.read_text(encoding="utf-8"))
    print(f"model_dir={model_dir}")
    print(f"architecture={config.get('architectures', [''])[0]}")
    print(f"hydra_supported={meta.get('hydra_supported')}")
    print(f"cfg_supported={meta.get('cfg_supported')}")
    print(f"pos_strategy={config.get('chatterbox_pos_strategy')}")

    register_vllm_t3_model()
    print("model_registry=ok")

    if args.skip_engine_init:
        print("engine_init=skipped")
        return

    _ = create_vllm_engine(
        model_dir=model_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        dtype=args.dtype,
    )
    print("engine_init=ok")


if __name__ == "__main__":
    main()
