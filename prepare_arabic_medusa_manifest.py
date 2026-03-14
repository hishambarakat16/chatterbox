import argparse
import csv
import json
import random
import re
from pathlib import Path


ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "").strip()
    text = " ".join(text.split())
    return text


def looks_usable(text: str, min_chars: int, max_chars: int, min_arabic_chars: int) -> bool:
    if not text:
        return False
    if "$$$" in text:
        return False
    if len(text) < min_chars or len(text) > max_chars:
        return False
    arabic_chars = sum(1 for ch in text if ARABIC_RE.search(ch))
    if arabic_chars < min_arabic_chars:
        return False
    if text.count("http") or text.count("www"):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a clean Arabic text manifest for T3 Medusa distillation."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=180)
    parser.add_argument("--min-arabic-chars", type=int, default=4)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_dir = input_path.parent

    rows = []
    seen = set()

    def maybe_add_row(sample_id: str, text: str, source_wav_path: str = "", duration: str = "") -> None:
        normalized = normalize_text(text)
        if not looks_usable(normalized, args.min_chars, args.max_chars, args.min_arabic_chars):
            return
        if normalized in seen:
            return
        seen.add(normalized)
        rows.append(
            {
                "sample_id": sample_id,
                "text": normalized,
                "source_wav_path": source_wav_path,
                "duration": duration,
            }
        )

    if input_path.suffix.lower() == ".jsonl":
        with input_path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "messages" in record:
                    for turn_index, message in enumerate(record["messages"]):
                        if message.get("role") != "assistant":
                            continue
                        maybe_add_row(
                            sample_id=f"jsonl_{line_index:06d}_assistant_{turn_index:02d}",
                            text=message.get("content", ""),
                        )
                elif "response" in record:
                    maybe_add_row(
                        sample_id=f"jsonl_{line_index:06d}_response",
                        text=record.get("response", ""),
                    )
                else:
                    maybe_add_row(
                        sample_id=f"jsonl_{line_index:06d}",
                        text=record.get("text", ""),
                    )
    else:
        with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                relative_wav = row.get("relative_wav_path", "")
                source_wav_path = str((metadata_dir / relative_wav).resolve()) if relative_wav else ""
                maybe_add_row(
                    sample_id=row.get("id", ""),
                    text=row.get("text", ""),
                    source_wav_path=source_wav_path,
                    duration=row.get("duration", ""),
                )

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)

    if args.limit > 0:
        rows = rows[: args.limit]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "text", "source_wav_path", "duration"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"input_csv={input_path}")
    print(f"output_csv={output_path}")
    print(f"num_rows={len(rows)}")


if __name__ == "__main__":
    main()
