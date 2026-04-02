#!/usr/bin/env python3
"""
Client for the /v1/tts/stream_chunks NDJSON streaming endpoint.

Measures:
  - time-to-headers (connection established + HTTP 200 received)
  - time-to-first-chunk (first "chunk" event decoded)
  - per-chunk timings (queue_wait_s, t3_s, s3_s, chunk_total_s)
  - total request wall time

Saves per-chunk WAV files when --save-dir is given, named
  <save-dir>/req_<N>_chunk_<K>.wav

Usage examples:
  # Single request, print timings
  python stream_chunks_client.py

  # Concurrency=4, 8 total requests, save chunks
  python stream_chunks_client.py --concurrency 4 --num-requests 8 --save-dir /tmp/chunks

  # Custom text
  python stream_chunks_client.py --text "Hello world. How are you today?"
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import http.client
import io
import json
import os
import statistics
import time
import urllib.parse
import wave
from typing import Any


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_TEXT = (
    "أهلاً وسهلاً، كيف حالك اليوم؟ "
    "نحن سعداء بوجودك هنا. "
    "هل تريد أن تسمع المزيد من الكلام؟ "
    "يمكنني التحدث عن أي موضوع تريده."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--url",
        default="http://127.0.0.1:8000/v1/tts/stream_chunks",
        help="Full URL of the stream_chunks endpoint.",
    )
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument(
        "--num-requests", type=int, default=None,
        help="Total requests. Defaults to --concurrency.",
    )
    p.add_argument(
        "--text", action="append", default=[],
        help="Request text. Can be repeated for round-robin assignment.",
    )
    p.add_argument("--language-id", default="ar")
    p.add_argument("--audio-prompt-path", default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--auto-max-new-tokens", action="store_true", default=True)
    p.add_argument(
        "--no-auto-max-new-tokens",
        dest="auto_max_new_tokens", action="store_false",
    )
    p.add_argument("--auto-max-new-tokens-cap", type=int, default=128)
    p.add_argument("--chunk-target-words", type=int, default=5)
    p.add_argument("--chunk-max-words", type=int, default=8)
    p.add_argument("--chunk-auto-max-new-tokens-cap", type=int, default=64)
    p.add_argument("--timeout-s", type=float, default=180.0)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--summary-json", default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-request streaming call
# ---------------------------------------------------------------------------

def run_stream_request(
    *,
    url: str,
    request_index: int,
    payload: dict[str, Any],
    timeout_s: float,
    save_dir: str | None,
) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}

    wall_start = time.perf_counter()

    conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    conn.request("POST", path, body=body, headers=headers)
    resp = conn.getresponse()

    headers_s = time.perf_counter() - wall_start

    if resp.status != 200:
        body_text = resp.read().decode(errors="replace")
        return {
            "request_index": request_index,
            "status": resp.status,
            "error": body_text,
            "headers_s": headers_s,
        }

    # Read NDJSON stream line by line.
    first_chunk_s: float | None = None
    chunks: list[dict[str, Any]] = []
    buf = b""

    while True:
        data = resp.read(4096)
        if not data:
            break
        buf += data
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            evt = event.get("event")
            if evt == "chunk":
                if first_chunk_s is None:
                    first_chunk_s = time.perf_counter() - wall_start
                chunks.append(event)
                # Save chunk WAV if requested.
                if save_dir and event.get("audio_wav_b64"):
                    os.makedirs(save_dir, exist_ok=True)
                    wav_bytes = base64.b64decode(event["audio_wav_b64"])
                    path_wav = os.path.join(
                        save_dir,
                        f"req_{request_index:03d}_chunk_{event['chunk_index']:03d}.wav",
                    )
                    with open(path_wav, "wb") as fh:
                        fh.write(wav_bytes)
            elif evt in ("done", "error"):
                if evt == "error":
                    chunks.append(event)
                break

    conn.close()
    total_s = time.perf_counter() - wall_start

    # Aggregate audio duration.
    audio_duration_s = 0.0
    for c in chunks:
        if c.get("audio_wav_b64"):
            try:
                wav_bytes = base64.b64decode(c["audio_wav_b64"])
                with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                    frames = wf.getnframes()
                    sr = wf.getframerate()
                    if sr > 0:
                        audio_duration_s += frames / float(sr)
            except Exception:
                pass

    return {
        "request_index": request_index,
        "status": resp.status,
        "headers_s": headers_s,
        "first_chunk_s": first_chunk_s,
        "total_s": total_s,
        "chunk_count": len([c for c in chunks if c.get("event") == "chunk"]),
        "audio_duration_s": audio_duration_s,
        "rtf": (audio_duration_s / total_s) if total_s > 0 else 0.0,
        "per_chunk": [
            {
                "chunk_index": c.get("chunk_index"),
                "text": c.get("text", ""),
                "queue_wait_s": c.get("queue_wait_s", 0.0),
                "t3_s": c.get("t3_s", 0.0),
                "s3_s": c.get("s3_s", 0.0),
                "chunk_total_s": c.get("chunk_total_s", 0.0),
                "is_final": c.get("is_final", False),
            }
            for c in chunks
            if c.get("event") == "chunk"
        ],
    }


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


def _p50(vals: list[float]) -> float:
    return statistics.median(vals) if vals else 0.0


def _fmt(label: str, val: float) -> str:
    return f"  {label:<32s} {val:.4f}s"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    texts = args.text or [DEFAULT_TEXT]
    num_requests = args.num_requests or args.concurrency

    payloads = []
    for i in range(num_requests):
        text = texts[i % len(texts)]
        p: dict[str, Any] = {
            "text": text,
            "language_id": args.language_id,
            "max_new_tokens": args.max_new_tokens,
            "auto_max_new_tokens": args.auto_max_new_tokens,
            "auto_max_new_tokens_cap": args.auto_max_new_tokens_cap,
            "chunk_target_words": args.chunk_target_words,
            "chunk_max_words": args.chunk_max_words,
            "chunk_auto_max_new_tokens_cap": args.chunk_auto_max_new_tokens_cap,
        }
        if args.audio_prompt_path:
            p["audio_prompt_path"] = args.audio_prompt_path
        payloads.append(p)

    if not args.quiet:
        print(f"Sending {num_requests} request(s) at concurrency={args.concurrency}")
        print(f"  endpoint : {args.url}")
        print(f"  text[0]  : {texts[0][:80]!r}")
        print()

    wall_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                run_stream_request,
                url=args.url,
                request_index=i,
                payload=payloads[i],
                timeout_s=args.timeout_s,
                save_dir=args.save_dir,
            )
            for i in range(num_requests)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    wall_total_s = time.perf_counter() - wall_start

    # Sort by request_index for deterministic output.
    results.sort(key=lambda r: r.get("request_index", 0))

    ok = [r for r in results if r.get("status") == 200]
    err = [r for r in results if r.get("status") != 200]

    if not args.quiet:
        for r in results:
            idx = r["request_index"]
            status = r.get("status", "?")
            if status != 200:
                print(f"[req {idx}] ERROR {status}: {r.get('error', '')[:120]}")
                continue
            first = r.get("first_chunk_s")
            first_str = f"{first:.3f}s" if first is not None else "N/A"
            chunks_n = r.get("chunk_count", 0)
            rtf = r.get("rtf", 0.0)
            total = r.get("total_s", 0.0)
            print(
                f"[req {idx}] OK  headers={r['headers_s']:.3f}s  "
                f"first_chunk={first_str}  chunks={chunks_n}  "
                f"total={total:.3f}s  RTF={rtf:.2f}x"
            )
            if not args.quiet:
                for c in r.get("per_chunk", []):
                    fin = " [final]" if c["is_final"] else ""
                    print(
                        f"         chunk[{c['chunk_index']}] "
                        f"t3={c['t3_s']:.3f}s s3={c['s3_s']:.3f}s "
                        f"total={c['chunk_total_s']:.3f}s "
                        f"text={c['text'][:40]!r}{fin}"
                    )
        print()

    if ok:
        first_chunks = [r["first_chunk_s"] for r in ok if r.get("first_chunk_s") is not None]
        totals = [r["total_s"] for r in ok]
        rtfs = [r["rtf"] for r in ok]

        print("=== Summary ===")
        print(f"  requests OK / total      : {len(ok)} / {num_requests}")
        if first_chunks:
            print(_fmt("first_chunk_s (mean)", _mean(first_chunks)))
            print(_fmt("first_chunk_s (p50)", _p50(first_chunks)))
        print(_fmt("total_s (mean)", _mean(totals)))
        print(_fmt("total_s (p50)", _p50(totals)))
        print(_fmt("RTF (mean)", _mean(rtfs)))
        print(f"  wall_s                   : {wall_total_s:.4f}s")

    if err:
        print(f"\n  ERRORS: {len(err)} request(s) failed")

    if args.summary_json:
        summary = {
            "num_requests": num_requests,
            "concurrency": args.concurrency,
            "ok_count": len(ok),
            "error_count": len(err),
            "wall_s": wall_total_s,
            "results": results,
        }
        with open(args.summary_json, "w") as fh:
            json.dump(summary, fh, indent=2)
        if not args.quiet:
            print(f"\nSummary written to {args.summary_json}")


if __name__ == "__main__":
    main()
