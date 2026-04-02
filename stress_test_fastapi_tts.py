#!/usr/bin/env python3
"""
Simple concurrent client for the FastAPI TTS service.

This measures end-to-end request timing against `/v1/tts` or `/v1/tts/stream`.
For the current `/v1/tts/stream` endpoint, "streaming" starts only after the
server finishes synthesis, so `first_chunk_s` is still a full-request metric.
"""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/v1/tts/stream",
        help="Full request URL, for example http://127.0.0.1:8000/v1/tts/stream",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help="Total requests to send. Defaults to --concurrency.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Request text. Can be repeated. If omitted, a default Arabic test line is used.",
    )
    parser.add_argument("--language-id", default="ar")
    parser.add_argument("--audio-prompt-path", default=None)
    parser.add_argument("--stream-chunk-bytes", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--auto-max-new-tokens", action="store_true", default=True)
    parser.add_argument("--no-auto-max-new-tokens", dest="auto_max_new_tokens", action="store_false")
    parser.add_argument("--auto-max-new-tokens-cap", type=int, default=128)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--read-size", type=int, default=8192)
    parser.add_argument("--save-dir", default=None, help="Optional directory to save returned WAV files.")
    parser.add_argument("--summary-json", default=None, help="Optional path to write JSON summary.")
    return parser.parse_args()


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lower = int(idx)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    frac = idx - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def audio_duration_s(wav_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            if sample_rate <= 0:
                return 0.0
            return frames / float(sample_rate)
    except wave.Error:
        return 0.0


def make_payload(args: argparse.Namespace, text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": text,
        "language_id": args.language_id,
        "stream_chunk_bytes": args.stream_chunk_bytes,
        "max_new_tokens": args.max_new_tokens,
        "auto_max_new_tokens": args.auto_max_new_tokens,
        "auto_max_new_tokens_cap": args.auto_max_new_tokens_cap,
    }
    if args.audio_prompt_path:
        payload["audio_prompt_path"] = args.audio_prompt_path
    return payload


def run_request(
    *,
    request_index: int,
    url: str,
    timeout_s: float,
    read_size: int,
    payload: dict[str, Any],
    save_dir: str | None,
) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(parsed.hostname, parsed.port, timeout=timeout_s)

    body = json.dumps(payload).encode("utf-8")
    started = time.perf_counter()
    conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    response = conn.getresponse()
    headers_s = time.perf_counter() - started

    chunks: list[bytes] = []
    total_bytes = 0
    first_chunk_s: float | None = None
    while True:
        chunk = response.read(read_size)
        if not chunk:
            break
        if first_chunk_s is None:
            first_chunk_s = time.perf_counter() - started
        chunks.append(chunk)
        total_bytes += len(chunk)

    total_s = time.perf_counter() - started
    wav_bytes = b"".join(chunks)
    wav_duration_s = audio_duration_s(wav_bytes)

    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"request_{request_index:03d}.wav")
        with open(save_path, "wb") as handle:
            handle.write(wav_bytes)

    server_queue_wait_s = float(response.getheader("X-Queue-Wait-S") or 0.0)
    server_total_s = float(response.getheader("X-Total-S") or 0.0)
    server_t3_s = float(response.getheader("X-T3-S") or 0.0)
    server_s3_s = float(response.getheader("X-S3-S") or 0.0)

    conn.close()
    return {
        "request_index": request_index,
        "status": response.status,
        "reason": response.reason,
        "headers_s": headers_s,
        "first_chunk_s": first_chunk_s if first_chunk_s is not None else headers_s,
        "total_s": total_s,
        "bytes": total_bytes,
        "audio_duration_s": wav_duration_s,
        "audio_seconds_per_second": (wav_duration_s / total_s) if total_s > 0 else 0.0,
        "server_queue_wait_s": server_queue_wait_s,
        "server_total_s": server_total_s,
        "server_t3_s": server_t3_s,
        "server_s3_s": server_s3_s,
        "request_id": response.getheader("X-Request-Id"),
        "save_path": save_path,
        "text": payload["text"],
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in results if 200 <= row["status"] < 300]
    headers_s = [row["headers_s"] for row in ok]
    first_chunk_s = [row["first_chunk_s"] for row in ok]
    total_s = [row["total_s"] for row in ok]
    queue_wait_s = [row["server_queue_wait_s"] for row in ok]
    t3_s = [row["server_t3_s"] for row in ok]
    s3_s = [row["server_s3_s"] for row in ok]
    rtfs = [row["audio_seconds_per_second"] for row in ok]
    return {
        "num_requests": len(results),
        "num_success": len(ok),
        "num_error": len(results) - len(ok),
        "mean_headers_s": statistics.fmean(headers_s) if headers_s else 0.0,
        "p95_headers_s": quantile(headers_s, 0.95),
        "mean_first_chunk_s": statistics.fmean(first_chunk_s) if first_chunk_s else 0.0,
        "p95_first_chunk_s": quantile(first_chunk_s, 0.95),
        "mean_total_s": statistics.fmean(total_s) if total_s else 0.0,
        "p95_total_s": quantile(total_s, 0.95),
        "mean_server_queue_wait_s": statistics.fmean(queue_wait_s) if queue_wait_s else 0.0,
        "mean_server_t3_s": statistics.fmean(t3_s) if t3_s else 0.0,
        "mean_server_s3_s": statistics.fmean(s3_s) if s3_s else 0.0,
        "mean_audio_seconds_per_second": statistics.fmean(rtfs) if rtfs else 0.0,
    }


def main() -> None:
    args = parse_args()
    texts = args.text or [
        "مرحبا، هذا اختبار ضغط بسيط لمسار تحويل النص إلى كلام عبر FastAPI."
    ]
    num_requests = args.num_requests or args.concurrency
    payloads = [make_payload(args, texts[i % len(texts)]) for i in range(num_requests)]

    all_started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                run_request,
                request_index=i,
                url=args.url,
                timeout_s=args.timeout_s,
                read_size=args.read_size,
                payload=payloads[i],
                save_dir=args.save_dir,
            )
            for i in range(num_requests)
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda row: row["request_index"])
    wall_s = time.perf_counter() - all_started
    summary = summarize(results)
    summary["wall_s"] = wall_s

    print(f"url={args.url}")
    print(f"concurrency={args.concurrency}")
    print(f"num_requests={num_requests}")
    print(f"wall_s={wall_s:.4f}")
    for key in [
        "num_success",
        "num_error",
        "mean_headers_s",
        "p95_headers_s",
        "mean_first_chunk_s",
        "p95_first_chunk_s",
        "mean_total_s",
        "p95_total_s",
        "mean_server_queue_wait_s",
        "mean_server_t3_s",
        "mean_server_s3_s",
        "mean_audio_seconds_per_second",
    ]:
        value = summary[key]
        if isinstance(value, float):
            print(f"{key}={value:.4f}")
        else:
            print(f"{key}={value}")

    print("per_request=")
    for row in results:
        print(json.dumps(row, ensure_ascii=False))

    if args.summary_json:
        payload = {"summary": summary, "results": results}
        with open(args.summary_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"summary_json={args.summary_json}")


if __name__ == "__main__":
    main()
