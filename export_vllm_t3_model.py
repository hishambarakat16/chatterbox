import argparse

from chatterbox.vllm_t3_bridge import export_vllm_t3_model


def main():
    parser = argparse.ArgumentParser(description="Export a vLLM-friendly T3 model package.")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--base-checkpoint-dir")
    parser.add_argument("--from-pretrained", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--copy", action="store_true", help="Copy weights instead of symlinking them.")
    args = parser.parse_args()

    if not args.from_pretrained and not args.checkpoint_dir and not args.base_checkpoint_dir:
        parser.error(
            "Provide `--checkpoint-dir`, `--base-checkpoint-dir`, or use `--from-pretrained`."
        )

    output_dir = export_vllm_t3_model(
        args.checkpoint_dir,
        output_dir=args.output_dir,
        base_checkpoint_dir=args.base_checkpoint_dir,
        allow_pretrained_fallback=args.from_pretrained,
        use_symlink=(not args.copy),
    )
    print(f"export_dir={output_dir}")


if __name__ == "__main__":
    main()
