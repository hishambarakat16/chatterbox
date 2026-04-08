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
import math
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

LANGUAGE_PATTERNS: dict[str, list[str]] = {
    "ar": [
        "مرحبا، هذا اختبار قصير لقياس سرعة الاستجابة.",
        "كيف يمكننا تحسين جودة الصوت مع الحفاظ على زمن استجابة منخفض؟",
        "نريد التأكد من أن أول جزء صوتي يصل بسرعة حتى عند وجود عدة طلبات.",
        "هذا مثال على جملة متوسطة الطول لاختبار مستويات التزامن المختلفة.",
    ],
    "en": [
        "Hello, this is a short test to measure response speed.",
        "How can we improve voice quality while keeping latency low?",
        "We want to confirm that the first audio chunk arrives quickly, even with several requests at once.",
        "This is a medium-length sentence to test stability across different concurrency levels.",
    ],
    "zh": [
        "你好，这是一个简短的测试，用来测量系统的响应速度。",
        "我们怎样才能在保持较低延迟的同时继续提升声音质量？",
        "我们想确认的是，即使同时来了多个请求，第一段音频也能尽快返回。",
        "这是一句中等长度的话，用来测试不同并发水平下的稳定性。",
    ],
}

NO_SPACE_LANGUAGES = {"zh", "ja", "ko"}

LANGUAGE_TOKEN_HEURISTICS: dict[str, dict[str, float]] = {
    "ar": {
        "full_base": 24.0,
        "full_per_unit": 6.0,
        "full_cap_max": 160.0,
        "chunk_base": 18.0,
        "chunk_per_unit": 8.0,
        "chunk_cap_max": 128.0,
    },
    "en": {
        "full_base": 20.0,
        "full_per_unit": 4.0,
        "full_cap_max": 128.0,
        "chunk_base": 14.0,
        "chunk_per_unit": 5.0,
        "chunk_cap_max": 96.0,
    },
    "zh": {
        "full_base": 18.0,
        "full_per_unit": 3.0,
        "full_cap_max": 112.0,
        "chunk_base": 12.0,
        "chunk_per_unit": 4.0,
        "chunk_cap_max": 80.0,
    },
}

DEFAULT_TOKEN_HEURISTICS = {
    "full_base": 20.0,
    "full_per_unit": 4.5,
    "full_cap_max": 128.0,
    "chunk_base": 14.0,
    "chunk_per_unit": 5.0,
    "chunk_cap_max": 96.0,
}


def _lang_key(language_id: str | None) -> str:
    return (language_id or "ar").strip().lower()


def _text_units(text: str, language_id: str) -> int:
    lang = _lang_key(language_id)
    stripped = text.strip()
    if not stripped:
        return 1
    if lang in NO_SPACE_LANGUAGES:
        punct = " \t\r\n.,!?;:،؛。！？、()[]{}\"'`"
        return max(1, sum(1 for ch in stripped if ch not in punct))
    return max(1, len(stripped.split()))


def _estimate_chunk_units(text: str, language_id: str, chunk_target_words: int, chunk_max_words: int) -> int:
    units = _text_units(text, language_id)
    lang = _lang_key(language_id)
    if lang in NO_SPACE_LANGUAGES:
        target_units = max(8, int(chunk_target_words) * 4)
        max_units = max(target_units, int(chunk_max_words) * 4)
        return min(units, max_units)

    target_words = max(1, int(chunk_target_words))
    max_words = max(target_words, int(chunk_max_words))
    estimated_chunks = max(1, math.ceil(units / target_words))
    estimated_chunk_words = max(1, math.ceil(units / estimated_chunks))
    return min(max_words, estimated_chunk_words)


def _estimate_token_caps(
    *,
    text: str,
    language_id: str,
    chunk_target_words: int,
    chunk_max_words: int,
) -> dict[str, int]:
    heuristics = LANGUAGE_TOKEN_HEURISTICS.get(_lang_key(language_id), DEFAULT_TOKEN_HEURISTICS)
    full_units = _text_units(text, language_id)
    chunk_units = _estimate_chunk_units(text, language_id, chunk_target_words, chunk_max_words)

    full_cap = int(round(heuristics["full_base"] + heuristics["full_per_unit"] * full_units))
    full_cap = max(32, min(full_cap, int(heuristics["full_cap_max"])))

    chunk_cap = int(round(heuristics["chunk_base"] + heuristics["chunk_per_unit"] * chunk_units))
    chunk_cap = max(32, min(chunk_cap, int(heuristics["chunk_cap_max"])))

    return {
        "text_units": full_units,
        "chunk_units": chunk_units,
        "max_new_tokens": full_cap,
        "auto_max_new_tokens_cap": full_cap,
        "chunk_auto_max_new_tokens_cap": chunk_cap,
    }


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
    p.add_argument(
        "--use-language-patterns",
        action="store_true",
        help="Use built-in language-specific sample texts for round-robin requests.",
    )
    p.add_argument("--language-id", default="ar")
    p.add_argument("--audio-prompt-path", default=None)
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override request max_new_tokens. Default: language-aware estimate from the input text.",
    )
    p.add_argument("--auto-max-new-tokens", action="store_true", default=True)
    p.add_argument(
        "--no-auto-max-new-tokens",
        dest="auto_max_new_tokens", action="store_false",
    )
    p.add_argument(
        "--auto-max-new-tokens-cap",
        type=int,
        default=None,
        help="Override whole-request auto token cap. Default: language-aware estimate from the input text.",
    )
    p.add_argument("--chunk-target-words", type=int, default=5)
    p.add_argument("--chunk-max-words", type=int, default=8)
    p.add_argument(
        "--chunk-auto-max-new-tokens-cap",
        type=int,
        default=None,
        help="Override per-chunk token cap. Default: language-aware estimate from the input text.",
    )
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
    done_trace: dict[str, Any] = {}

    while True:
        line = resp.readline()
        if not line:
            break
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
            continue

        if evt == "done":
            done_trace = event.get("trace", {}) or {}
            break

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
        "request_trace": done_trace,
        "per_chunk": [
            {
                "chunk_index": c.get("chunk_index"),
                "text": c.get("text", ""),
                "queue_wait_s": c.get("queue_wait_s", 0.0),
                "t3_s": c.get("t3_s", 0.0),
                "s3_s": c.get("s3_s", 0.0),
                "chunk_total_s": c.get("chunk_total_s", 0.0),
                "is_final": c.get("is_final", False),
                "trace": {
                    "batch_id": (c.get("trace") or {}).get("batch_id"),
                    "batch_size": (c.get("trace") or {}).get("batch_size"),
                    "batch_infer_wall_s": (c.get("trace") or {}).get("infer_wall_s", 0.0),
                    "model_generate_many_s": (c.get("trace") or {}).get("model_generate_many_s", 0.0),
                    "session_create_s": (c.get("trace") or {}).get("session_create_s", 0.0),
                    "text_prep_s": (c.get("trace") or {}).get("text_prep_s", 0.0),
                    "t3_wait_s": (c.get("trace") or {}).get("t3_wait_s", 0.0),
                    "t3_active_s": (c.get("trace") or {}).get("t3_active_s", 0.0),
                    "audio_ready_s": (c.get("trace") or {}).get("audio_ready_s", 0.0),
                    "request_elapsed_s": (c.get("trace") or {}).get("request_elapsed_s", 0.0),
                    "first_chunk_latency_s": (c.get("trace") or {}).get("first_chunk_latency_s", 0.0),
                    "stage_timings": (c.get("trace") or {}).get("stage_timings", {}),
                    "stage_meta": (c.get("trace") or {}).get("stage_meta", {}),
                },
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
    lang = _lang_key(args.language_id)
    pattern_texts = LANGUAGE_PATTERNS.get(lang)
    if args.text:
        texts = args.text
    elif args.use_language_patterns and pattern_texts:
        texts = pattern_texts
    elif pattern_texts:
        texts = pattern_texts[:1]
    else:
        texts = [DEFAULT_TEXT]
    num_requests = args.num_requests or args.concurrency

    payloads = []
    for i in range(num_requests):
        text = texts[i % len(texts)]
        estimated_caps = _estimate_token_caps(
            text=text,
            language_id=args.language_id,
            chunk_target_words=args.chunk_target_words,
            chunk_max_words=args.chunk_max_words,
        )
        max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else estimated_caps["max_new_tokens"]
        auto_cap = (
            args.auto_max_new_tokens_cap
            if args.auto_max_new_tokens_cap is not None
            else estimated_caps["auto_max_new_tokens_cap"]
        )
        chunk_cap = (
            args.chunk_auto_max_new_tokens_cap
            if args.chunk_auto_max_new_tokens_cap is not None
            else estimated_caps["chunk_auto_max_new_tokens_cap"]
        )
        p: dict[str, Any] = {
            "text": text,
            "language_id": args.language_id,
            "max_new_tokens": max_new_tokens,
            "auto_max_new_tokens": args.auto_max_new_tokens,
            "auto_max_new_tokens_cap": auto_cap,
            "chunk_target_words": args.chunk_target_words,
            "chunk_max_words": args.chunk_max_words,
            "chunk_auto_max_new_tokens_cap": chunk_cap,
        }
        if args.audio_prompt_path:
            p["audio_prompt_path"] = args.audio_prompt_path
        payloads.append({
            "request": p,
            "budget": {
            "text_units": estimated_caps["text_units"],
            "chunk_units": estimated_caps["chunk_units"],
            "max_new_tokens": max_new_tokens,
            "auto_max_new_tokens_cap": auto_cap,
            "chunk_auto_max_new_tokens_cap": chunk_cap,
            },
        })

    if not args.quiet:
        print(f"Sending {num_requests} request(s) at concurrency={args.concurrency}")
        print(f"  endpoint : {args.url}")
        print(f"  text[0]  : {texts[0][:80]!r}")
        first_budget = payloads[0].get("budget", {})
        print(
            "  budget[0]: "
            f"text_units={first_budget.get('text_units')} "
            f"chunk_units={first_budget.get('chunk_units')} "
            f"max_new_tokens={first_budget.get('max_new_tokens')} "
            f"auto_cap={first_budget.get('auto_max_new_tokens_cap')} "
            f"chunk_cap={first_budget.get('chunk_auto_max_new_tokens_cap')}"
        )
        print()

    wall_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                run_stream_request,
                url=args.url,
                request_index=i,
                payload=payloads[i]["request"],
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
            req_trace = r.get("request_trace", {}) or {}
            server_first = req_trace.get("first_chunk_latency_s")
            server_first_str = (
                f"{float(server_first):.3f}s" if server_first is not None else "N/A"
            )
            print(
                f"[req {idx}] OK  headers={r['headers_s']:.3f}s  "
                f"first_chunk={first_str}  server_first={server_first_str}  "
                f"chunks={chunks_n}  total={total:.3f}s  RTF={rtf:.2f}x"
            )
            if not args.quiet:
                for c in r.get("per_chunk", []):
                    fin = " [final]" if c["is_final"] else ""
                    trace = c.get("trace", {}) or {}
                    stages = trace.get("stage_timings", {}) or {}
                    stage_meta = trace.get("stage_meta", {}) or {}
                    print(
                        f"         chunk[{c['chunk_index']}] "
                        f"queue={c['queue_wait_s']:.3f}s t3={c['t3_s']:.3f}s "
                        f"s3={c['s3_s']:.3f}s total={c['chunk_total_s']:.3f}s "
                        f"gap={float(stages.get('s3_finalize_queue_delay_s', 0.0)):.3f}s "
                        f"tok2mel={float(stages.get('s3_token2mel_s', 0.0)):.3f}s "
                        f"hift={float(stages.get('s3_hift_s', 0.0)):.3f}s "
                        f"analyzer={int(stage_meta.get('t3_alignment_analyzer_active', 0.0))} "
                        f"batch_id={trace.get('batch_id')} size={trace.get('batch_size')} "
                        f"text={c['text'][:40]!r}{fin}"
                    )
        print()

    if ok:
        first_chunks = [r["first_chunk_s"] for r in ok if r.get("first_chunk_s") is not None]
        totals = [r["total_s"] for r in ok]
        rtfs = [r["rtf"] for r in ok]
        server_firsts = [
            float((r.get("request_trace", {}) or {}).get("first_chunk_latency_s", 0.0))
            for r in ok
            if (r.get("request_trace", {}) or {}).get("first_chunk_latency_s") is not None
        ]
        all_chunks = [c for r in ok for c in r.get("per_chunk", [])]
        first_chunks_only = [c for c in all_chunks if c.get("chunk_index") == 0]
        queue_waits = [float(c.get("queue_wait_s", 0.0)) for c in all_chunks]
        t3_waits = [float((c.get("trace", {}) or {}).get("t3_wait_s", 0.0)) for c in all_chunks]
        t3_actives = [float((c.get("trace", {}) or {}).get("t3_active_s", 0.0)) for c in all_chunks]
        session_conditioning = [float(((c.get("trace", {}) or {}).get("stage_timings", {}) or {}).get("session_conditioning_s", 0.0)) for c in first_chunks_only]
        prompt_embeds = [float(((c.get("trace", {}) or {}).get("stage_timings", {}) or {}).get("t3_prompt_embed_s", 0.0)) for c in all_chunks]
        s3_finalize_waits = [float(((c.get("trace", {}) or {}).get("stage_timings", {}) or {}).get("s3_finalize_queue_delay_s", 0.0)) for c in all_chunks]
        token2mel = [float(((c.get("trace", {}) or {}).get("stage_timings", {}) or {}).get("s3_token2mel_s", 0.0)) for c in all_chunks]
        hift = [float(((c.get("trace", {}) or {}).get("stage_timings", {}) or {}).get("s3_hift_s", 0.0)) for c in all_chunks]

        print("=== Summary ===")
        print(f"  requests OK / total      : {len(ok)} / {num_requests}")
        if first_chunks:
            print(_fmt("first_chunk_s(client mean)", _mean(first_chunks)))
            print(_fmt("first_chunk_s(client p50)", _p50(first_chunks)))
        if server_firsts:
            print(_fmt("first_chunk_s(server mean)", _mean(server_firsts)))
            print(_fmt("first_chunk_s(server p50)", _p50(server_firsts)))
        print(_fmt("total_s (mean)", _mean(totals)))
        print(_fmt("total_s (p50)", _p50(totals)))
        print(_fmt("RTF (mean)", _mean(rtfs)))
        if queue_waits:
            print(_fmt("chunk_queue_wait_s (mean)", _mean(queue_waits)))
        if t3_waits:
            print(_fmt("chunk_t3_wait_s (mean)", _mean(t3_waits)))
        if t3_actives:
            print(_fmt("chunk_t3_active_s (mean)", _mean(t3_actives)))
        if session_conditioning:
            print(_fmt("session_conditioning_s", _mean(session_conditioning)))
        if prompt_embeds:
            print(_fmt("prompt_embed_s (mean)", _mean(prompt_embeds)))
        if s3_finalize_waits:
            print(_fmt("s3_finalize_wait_s (mean)", _mean(s3_finalize_waits)))
        if token2mel:
            print(_fmt("s3_token2mel_s (mean)", _mean(token2mel)))
        if hift:
            print(_fmt("s3_hift_s (mean)", _mean(hift)))
        print(f"  wall_s                   : {wall_total_s:.4f}s")

    if err:
        print(f"\n  ERRORS: {len(err)} request(s) failed")

    if args.summary_json:
        summary = {
            "num_requests": num_requests,
            "concurrency": args.concurrency,
            "language_id": args.language_id,
            "ok_count": len(ok),
            "error_count": len(err),
            "wall_s": wall_total_s,
            "texts": texts,
            "client_token_budgets": [p.get("budget", {}) for p in payloads],
            "results": results,
        }
        with open(args.summary_json, "w") as fh:
            json.dump(summary, fh, indent=2)
        if not args.quiet:
            print(f"\nSummary written to {args.summary_json}")


if __name__ == "__main__":
    main()
