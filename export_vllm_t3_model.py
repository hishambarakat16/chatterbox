import argparse

from chatterbox.vllm_t3_bridge import export_vllm_t3_model


def main():
    parser = argparse.ArgumentParser(description="Export a vLLM-friendly T3 model package.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--copy", action="store_true", help="Copy weights instead of symlinking them.")
    args = parser.parse_args()

    output_dir = export_vllm_t3_model(
        args.checkpoint_dir,
        output_dir=args.output_dir,
        use_symlink=(not args.copy),
    )
    print(f"export_dir={output_dir}")


if __name__ == "__main__":
    main()
