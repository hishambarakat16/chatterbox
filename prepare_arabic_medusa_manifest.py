import argparse
import csv
import json
import random
import re
from pathlib import Path


ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[\.\!\?؟؛…])\s+|[\r\n]+")
CLAUSE_BOUNDARY_RE = re.compile(r"(?<=[،,:;؛])\s+")
PLACEHOLDER_RE = re.compile(r"\(…\)|\(\.\.\.\)|\[…\]|\[\.\.\.\]|…|\.\.\.")


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "").strip()
    text = " ".join(text.split())
    return text


def looks_usable(text: str, min_chars: int, max_chars: int, min_arabic_chars: int) -> bool:
    if not text:
        return False
    if "$$$" in text:
        return False
    if PLACEHOLDER_RE.search(text):
        return False
    if len(text) < min_chars or len(text) > max_chars:
        return False
    arabic_chars = sum(1 for ch in text if ARABIC_RE.search(ch))
    if arabic_chars < min_arabic_chars:
        return False
    if text.count("http") or text.count("www"):
        return False
    return True


def split_by_max_chars(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        chunks.append(current)
        current = word
    chunks.append(current)
    return chunks


def iter_text_segments(text: str, *, max_chars: int, split_sentences: bool) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    if not split_sentences:
        return [normalized]

    sentence_candidates = []
    for part in SENTENCE_BOUNDARY_RE.split(normalized):
        part = normalize_text(part)
        if not part:
            continue
        if len(part) <= max_chars:
            sentence_candidates.append(part)
            continue

        clause_parts = []
        for clause in CLAUSE_BOUNDARY_RE.split(part):
            clause = normalize_text(clause)
            if not clause:
                continue
            if len(clause) <= max_chars:
                clause_parts.append(clause)
            else:
                clause_parts.extend(split_by_max_chars(clause, max_chars))
        sentence_candidates.extend(clause_parts)

    return sentence_candidates or [normalized]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a clean Arabic text manifest for T3 Medusa distillation."
    )
    parser.add_argument("--input-csv", help="Single input path kept for backward compatibility.")
    parser.add_argument(
        "--input-path",
        action="append",
        default=[],
        help="Input path to include. Can be passed multiple times for JSONL and/or CSV sources.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=180)
    parser.add_argument("--min-arabic-chars", type=int, default=4)
    parser.add_argument(
        "--split-sentences",
        action="store_true",
        help="Split long multi-sentence rows into shorter sentence/clause-sized samples.",
    )
    args = parser.parse_args()

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_paths = [Path(path) for path in args.input_path]
    if args.input_csv:
        input_paths.append(Path(args.input_csv))
    if not input_paths:
        raise ValueError("Provide at least one input via --input-path or --input-csv")

    rows = []
    seen = set()

    def maybe_add_row(sample_id: str, text: str, source_wav_path: str = "", duration: str = "") -> None:
        segments = iter_text_segments(
            text,
            max_chars=args.max_chars,
            split_sentences=args.split_sentences,
        )
        for segment_index, normalized in enumerate(segments):
            if not looks_usable(normalized, args.min_chars, args.max_chars, args.min_arabic_chars):
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            segment_sample_id = sample_id if len(segments) == 1 else f"{sample_id}_seg_{segment_index:02d}"
            rows.append(
                {
                    "sample_id": segment_sample_id,
                    "text": normalized,
                    "source_wav_path": source_wav_path,
                    "duration": duration,
                }
            )

    for input_path in input_paths:
        metadata_dir = input_path.parent
        source_prefix = input_path.stem
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
                                sample_id=f"{source_prefix}_{line_index:06d}_assistant_{turn_index:02d}",
                                text=message.get("content", ""),
                            )
                    elif "response" in record:
                        maybe_add_row(
                            sample_id=f"{source_prefix}_{line_index:06d}_response",
                            text=record.get("response", ""),
                        )
                    else:
                        maybe_add_row(
                            sample_id=f"{source_prefix}_{line_index:06d}",
                            text=record.get("text", ""),
                        )
        else:
            with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    relative_wav = row.get("relative_wav_path", "")
                    source_wav_path = str((metadata_dir / relative_wav).resolve()) if relative_wav else ""
                    raw_id = row.get("id", "")
                    maybe_add_row(
                        sample_id=f"{source_prefix}_{raw_id}" if raw_id else source_prefix,
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

    print(f"input_paths={[str(path) for path in input_paths]}")
    print(f"output_csv={output_path}")
    print(f"num_rows={len(rows)}")


if __name__ == "__main__":
    main()
