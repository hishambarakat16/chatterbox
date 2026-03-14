import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from chatterbox.mtl_tts import SUPPORTED_LANGUAGES, punc_norm
from chatterbox.mtl_tts_scheduled import ChatterboxMultilingualScheduledTTS
from chatterbox.models.s3tokenizer import drop_invalid_tokens
from chatterbox.models.t3.inference.scheduled_decode import ScheduledDecodeRequest
from chatterbox.models.t3.inference.speculative_decode import run_baseline_greedy_decode
from chatterbox.runtime.session import clone_conditionals


def load_model(device: str, checkpoint_dir: str | None):
    if checkpoint_dir:
        return ChatterboxMultilingualScheduledTTS.from_local(checkpoint_dir, device)
    return ChatterboxMultilingualScheduledTTS.from_pretrained(device)


def build_single_request(
    model: ChatterboxMultilingualScheduledTTS,
    *,
    text: str,
    language_id: str,
    audio_prompt_path: str,
    max_new_tokens: int,
    cfg_weight: float,
    temperature: float,
    repetition_penalty: float,
    min_p: float,
    top_p: float,
):
    worker = model.worker
    if language_id.lower() not in SUPPORTED_LANGUAGES:
        supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
        raise ValueError(f"Unsupported language_id '{language_id}'. Supported languages: {supported_langs}")

    session = model.create_session(
        audio_prompt_path=audio_prompt_path,
        language_id=language_id,
        cfg_weight=cfg_weight,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        min_p=min_p,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    options = session.options

    normalized_text = punc_norm(text)
    text_tokens_single = worker.tokenizer.text_to_tokens(
        normalized_text,
        language_id=language_id.lower(),
    ).to(worker.device)
    text_tokens_single = F.pad(text_tokens_single, (1, 0), value=worker.t3.hp.start_text_token)
    text_tokens_single = F.pad(text_tokens_single, (0, 1), value=worker.t3.hp.stop_text_token)

    text_tokens_cfg = torch.cat([text_tokens_single, text_tokens_single], dim=0)
    conds = clone_conditionals(session.conditionals).to(worker.device)
    request = ScheduledDecodeRequest(
        session_id="medusa_distill",
        t3_cond=conds.t3,
        text_tokens=text_tokens_cfg,
        max_new_tokens=max_new_tokens,
        temperature=options.temperature,
        top_p=options.top_p,
        min_p=options.min_p,
        repetition_penalty=options.repetition_penalty,
        cfg_weight=options.cfg_weight,
    )
    return request, session, normalized_text, text_tokens_single


def save_conditionals_once(session, output_dir: Path) -> Path:
    conds_dir = output_dir / "conditionals"
    conds_dir.mkdir(parents=True, exist_ok=True)
    conds_path = conds_dir / "prompt_000.pt"
    if not conds_path.exists():
        clone_conditionals(session.conditionals).save(conds_path)
    return conds_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a T3 Medusa distillation dataset from Arabic text prompts."
    )
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--audio-prompt-path", required=True)
    parser.add_argument("--language-id", default="ar")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "samples.jsonl"

    model = load_model(args.device, args.checkpoint_dir)

    records = []
    with Path(args.manifest_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(row)

    if args.limit > 0:
        records = records[: args.limit]

    written = 0
    failures = 0
    conds_path_written = None
    with jsonl_path.open("w", encoding="utf-8") as sink:
        for index, row in enumerate(records):
            text = row["text"]
            sample_id = row.get("sample_id") or f"sample_{index:06d}"
            try:
                request, session, normalized_text, text_tokens_single = build_single_request(
                    model,
                    text=text,
                    language_id=args.language_id,
                    audio_prompt_path=args.audio_prompt_path,
                    max_new_tokens=args.max_new_tokens,
                    cfg_weight=args.cfg_weight,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    min_p=args.min_p,
                    top_p=args.top_p,
                )
                if conds_path_written is None:
                    conds_path_written = save_conditionals_once(session, output_dir)

                teacher_tokens = run_baseline_greedy_decode(model.worker.t3, request)
                teacher_tokens = drop_invalid_tokens(teacher_tokens[0]).to("cpu")
                text_tokens_single = text_tokens_single.to("cpu")

                record = {
                    "sample_id": sample_id,
                    "text": text,
                    "normalized_text": normalized_text,
                    "language_id": args.language_id,
                    "audio_prompt_path": str(Path(args.audio_prompt_path).resolve()),
                    "conditionals_path": str(conds_path_written.resolve()),
                    "text_tokens": text_tokens_single.tolist(),
                    "speech_tokens": teacher_tokens.tolist(),
                    "num_text_tokens": int(text_tokens_single.numel()),
                    "num_speech_tokens": int(teacher_tokens.numel()),
                    "teacher_decode": {
                        "cfg_weight": args.cfg_weight,
                        "temperature": args.temperature,
                        "repetition_penalty": args.repetition_penalty,
                        "min_p": args.min_p,
                        "top_p": args.top_p,
                        "max_new_tokens": args.max_new_tokens,
                    },
                    "source_wav_path": row.get("source_wav_path", ""),
                    "source_duration": row.get("duration", ""),
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if written % max(args.save_every, 1) == 0:
                    print(f"written={written} failures={failures}")
            except Exception as exc:  # pragma: no cover - dataset generation should continue on bad rows
                failures += 1
                print(f"failed sample_id={sample_id}: {exc}")

    print(f"output_dir={output_dir}")
    print(f"jsonl_path={jsonl_path}")
    print(f"written={written}")
    print(f"failures={failures}")
    if conds_path_written is not None:
        print(f"conditionals_path={conds_path_written}")


if __name__ == "__main__":
    main()
