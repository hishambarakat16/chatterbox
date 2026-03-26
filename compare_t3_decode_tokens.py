from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from chatterbox.audio_utils import save_wav
from chatterbox.models.s3tokenizer import drop_invalid_tokens
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.vllm_t3_bridge import (
    build_vllm_prompt,
    create_vllm_engine,
    make_sampling_params,
    prepare_vllm_text_tokens,
)


def _find_repeated_suffix(
    token_ids: list[int],
    *,
    min_repeats: int = 3,
    max_pattern_size: int = 4,
) -> dict[str, int] | None:
    if len(token_ids) < min_repeats:
        return None

    best = None
    limit = min(max_pattern_size, len(token_ids) // min_repeats)
    for pattern_size in range(1, limit + 1):
        suffix = token_ids[-pattern_size:]
        repeats = 1
        pos = len(token_ids) - pattern_size
        while pos - pattern_size >= 0 and token_ids[pos - pattern_size: pos] == suffix:
            repeats += 1
            pos -= pattern_size
        if repeats < min_repeats:
            continue
        trim_index = pos + pattern_size
        trim_tokens = len(token_ids) - trim_index
        candidate = {
            "trim_index": trim_index,
            "trim_tokens": trim_tokens,
            "pattern_size": pattern_size,
            "repeats": repeats,
        }
        if best is None or candidate["trim_tokens"] > best["trim_tokens"]:
            best = candidate
    return best


def _tensor_to_int_list(tokens: torch.Tensor) -> list[int]:
    if tokens.ndim > 1:
        tokens = tokens.reshape(-1)
    return [int(x) for x in tokens.detach().cpu().tolist()]


def _longest_run(values: list[int]) -> int:
    if not values:
        return 0
    best = 1
    current = 1
    for i in range(1, len(values)):
        if values[i] == values[i - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def _entropy(values: list[int]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(len(values))
    h = 0.0
    for c in counts.values():
        p = float(c) / total
        h -= p * math.log2(max(p, 1e-12))
    return h


def _token_stats(*, raw_ids: list[int], valid_ids: list[int], stop_token_id: int) -> dict[str, Any]:
    stop_index = next((i for i, tid in enumerate(raw_ids) if tid == stop_token_id), None)
    valid_counts = Counter(valid_ids)
    top_items = [
        {"token_id": int(token_id), "count": int(count)}
        for token_id, count in valid_counts.most_common(8)
    ]

    return {
        "raw_len": int(len(raw_ids)),
        "valid_len": int(len(valid_ids)),
        "invalid_trimmed": int(len(raw_ids) - len(valid_ids)),
        "has_stop_token": bool(stop_index is not None),
        "stop_index": None if stop_index is None else int(stop_index),
        "valid_unique": int(len(valid_counts)),
        "valid_unique_ratio": 0.0 if not valid_ids else float(len(valid_counts) / len(valid_ids)),
        "valid_entropy_bits": float(_entropy(valid_ids)),
        "valid_longest_run": int(_longest_run(valid_ids)),
        "repeat_tail": _find_repeated_suffix(valid_ids),
        "valid_preview": [int(x) for x in valid_ids[:32]],
        "valid_suffix": [int(x) for x in valid_ids[-32:]] if valid_ids else [],
        "top_counts": top_items,
    }


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _load_texts(args: argparse.Namespace) -> list[str]:
    if args.text:
        texts = [t for t in args.text if t.strip()]
    elif args.texts_file:
        texts = []
        for raw in Path(args.texts_file).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            texts.append(line)
    else:
        texts = ["مرحبا، هذا اختبار تشخيصي لمسار فك التشفير بين baseline و vllm."]

    if args.text_limit is not None:
        texts = texts[: max(0, int(args.text_limit))]
    if not texts:
        raise ValueError("No texts supplied.")
    return texts


def _load_baseline(device: str, checkpoint_dir: str | None) -> ChatterboxMultilingualTTS:
    if checkpoint_dir:
        return ChatterboxMultilingualTTS.from_local(checkpoint_dir, device)
    return ChatterboxMultilingualTTS.from_pretrained(device)


def _tokens_to_audio(baseline: ChatterboxMultilingualTTS, token_ids: list[int]) -> torch.Tensor | None:
    if not token_ids:
        return None
    tokens = torch.tensor(token_ids, dtype=torch.long, device=baseline.device)
    tokens = drop_invalid_tokens(tokens).to(baseline.device)
    if tokens.numel() == 0:
        return None
    with torch.inference_mode():
        wav, _ = baseline.s3gen.inference(
            speech_tokens=tokens,
            ref_dict=baseline.conds.gen,
        )
    wav = wav.squeeze(0).detach().cpu().numpy()
    watermarked = baseline.watermarker.apply_watermark(wav, sample_rate=baseline.sr)
    return torch.from_numpy(watermarked).unsqueeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline T3 decode tokens vs vLLM decode tokens for the same requests, "
            "with optional audio export for each side."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--audio-prompt-path", required=True)
    parser.add_argument("--language-id", default="ar")
    parser.add_argument("--text", action="append")
    parser.add_argument("--texts-file")
    parser.add_argument("--text-limit", type=int)

    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--cfg-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.05)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--baseline-min-temperature",
        type=float,
        default=1e-5,
        help="Baseline T3 path does not support exactly 0 temperature; clamp to this minimum.",
    )

    parser.add_argument("--vllm-model-dir", required=True)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--vllm-dtype", default="auto")
    parser.add_argument("--vllm-max-model-len", type=int, default=2048)
    parser.add_argument("--vllm-enable-prefix-caching", action="store_true")
    parser.add_argument("--no-vllm-chunked-prefill", action="store_true")

    parser.add_argument("--save-audio-dir")
    parser.add_argument("--save-audio-limit", type=int, default=8)
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts = _load_texts(args)

    save_audio_dir: Path | None = None
    if args.save_audio_dir:
        save_audio_dir = Path(args.save_audio_dir)
        save_audio_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    baseline = _load_baseline(args.device, args.checkpoint_dir)
    baseline_load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    baseline.prepare_conditionals(args.audio_prompt_path, exaggeration=args.exaggeration)
    cond_prep_s = time.perf_counter() - t1

    # Baseline decode path enables output_attentions for multilingual alignment checks;
    # force eager attention to avoid SDPA-attention incompatibility in this diagnostic.
    tfmr_cfg = getattr(getattr(baseline, "t3", None), "tfmr", None)
    tfmr_cfg = getattr(tfmr_cfg, "config", None)
    if tfmr_cfg is not None:
        try:
            setattr(tfmr_cfg, "_attn_implementation", "eager")
        except Exception:
            pass
        try:
            setattr(tfmr_cfg, "attn_implementation", "eager")
        except Exception:
            pass

    engine = None
    vllm_init_error = None
    try:
        engine = create_vllm_engine(
            model_dir=args.vllm_model_dir,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            enforce_eager=args.vllm_enforce_eager,
            dtype=args.vllm_dtype,
            max_model_len=args.vllm_max_model_len,
            enable_prefix_caching=args.vllm_enable_prefix_caching,
            enable_chunked_prefill=(not args.no_vllm_chunked_prefill),
        )
    except Exception as exc:  # noqa: BLE001
        vllm_init_error = repr(exc)

    options = SimpleNamespace(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        min_p=float(args.min_p),
        repetition_penalty=float(args.repetition_penalty),
        max_new_tokens=int(args.max_new_tokens),
    )
    sampling_params = None
    if engine is not None:
        sampling_params = make_sampling_params(options=options, hp=baseline.t3.hp)

    stop_token_id = int(baseline.t3.hp.stop_speech_token)
    effective_baseline_temperature = max(float(args.temperature), float(args.baseline_min_temperature))

    rows: list[dict[str, Any]] = []

    try:
        with torch.inference_mode():
            for idx, text in enumerate(texts):
                vllm_text_tokens = prepare_vllm_text_tokens(
                    tokenizer=baseline.tokenizer,
                    text=text,
                    language_id=args.language_id,
                    device=baseline.device,
                )

                baseline_text_tokens = torch.cat([vllm_text_tokens, vllm_text_tokens], dim=0)

                b_start = time.perf_counter()
                baseline_raw = baseline.t3.inference(
                    t3_cond=baseline.conds.t3,
                    text_tokens=baseline_text_tokens,
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(effective_baseline_temperature),
                    cfg_weight=float(args.cfg_weight),
                    repetition_penalty=float(args.repetition_penalty),
                    min_p=float(args.min_p),
                    top_p=float(args.top_p),
                )
                baseline_decode_s = time.perf_counter() - b_start

                baseline_raw_ids = _tensor_to_int_list(baseline_raw[0])
                baseline_valid_ids = _tensor_to_int_list(
                    drop_invalid_tokens(torch.tensor(baseline_raw_ids, dtype=torch.long))
                )
                baseline_stats = _token_stats(
                    raw_ids=baseline_raw_ids,
                    valid_ids=baseline_valid_ids,
                    stop_token_id=stop_token_id,
                )

                baseline_audio_path = None
                if save_audio_dir is not None and idx < int(args.save_audio_limit):
                    baseline_audio = _tokens_to_audio(baseline, baseline_raw_ids)
                    if baseline_audio is not None:
                        baseline_audio_path = save_audio_dir / f"row_{idx:03d}_baseline.wav"
                        save_wav(baseline_audio_path, baseline_audio, baseline.sr)

                vllm_decode_s = None
                vllm_finish_reason = None
                vllm_stop_reason = None
                vllm_raw_ids: list[int] = []
                vllm_valid_ids: list[int] = []
                vllm_stats = None
                vllm_audio_path = None

                if engine is not None and sampling_params is not None:
                    prompt = build_vllm_prompt(
                        t3_cond=baseline.conds.t3,
                        text_tokens=vllm_text_tokens,
                    )
                    v_start = time.perf_counter()
                    vllm_outputs = engine.generate(
                        [prompt],
                        sampling_params=sampling_params,
                        use_tqdm=False,
                    )
                    vllm_decode_s = float(time.perf_counter() - v_start)
                    v_output = vllm_outputs[0].outputs[0] if vllm_outputs and vllm_outputs[0].outputs else None
                    if v_output is not None:
                        vllm_finish_reason = getattr(v_output, "finish_reason", None)
                        vllm_stop_reason = getattr(v_output, "stop_reason", None)
                        vllm_raw_ids = [int(t) for t in v_output.token_ids]
                        vllm_valid_ids = _tensor_to_int_list(
                            drop_invalid_tokens(torch.tensor(vllm_raw_ids, dtype=torch.long))
                        )
                        vllm_stats = _token_stats(
                            raw_ids=vllm_raw_ids,
                            valid_ids=vllm_valid_ids,
                            stop_token_id=stop_token_id,
                        )
                        if save_audio_dir is not None and idx < int(args.save_audio_limit):
                            vllm_audio = _tokens_to_audio(baseline, vllm_raw_ids)
                            if vllm_audio is not None:
                                vllm_audio_path = save_audio_dir / f"row_{idx:03d}_vllm.wav"
                                save_wav(vllm_audio_path, vllm_audio, baseline.sr)

                compare = None
                if vllm_stats is not None:
                    common_prefix = _common_prefix_len(baseline_valid_ids, vllm_valid_ids)
                    shortest = max(1, min(len(baseline_valid_ids), len(vllm_valid_ids)))
                    compare = {
                        "valid_common_prefix_len": int(common_prefix),
                        "valid_prefix_match_ratio": float(common_prefix / shortest),
                        "valid_len_delta": int(len(vllm_valid_ids) - len(baseline_valid_ids)),
                    }

                rows.append(
                    {
                        "index": int(idx),
                        "text": text,
                        "text_chars": int(len(text)),
                        "baseline_decode_s": float(baseline_decode_s),
                        "vllm_decode_s": vllm_decode_s,
                        "vllm_finish_reason": vllm_finish_reason,
                        "vllm_stop_reason": vllm_stop_reason,
                        "baseline_audio_path": None if baseline_audio_path is None else str(baseline_audio_path),
                        "vllm_audio_path": None if vllm_audio_path is None else str(vllm_audio_path),
                        "baseline": baseline_stats,
                        "vllm": vllm_stats,
                        "compare": compare,
                    }
                )
    finally:
        if engine is not None and hasattr(engine, "shutdown"):
            engine.shutdown()

    vllm_rows = [r for r in rows if r["vllm_decode_s"] is not None]
    summary = {
        "num_requests": int(len(rows)),
        "baseline_temperature_requested": float(args.temperature),
        "baseline_temperature_effective": float(effective_baseline_temperature),
        "vllm_available": bool(engine is not None),
        "vllm_init_error": vllm_init_error,
        "mean_baseline_decode_s": _safe_mean([float(r["baseline_decode_s"]) for r in rows]),
        "mean_vllm_decode_s": _safe_mean([float(r["vllm_decode_s"]) for r in vllm_rows]),
        "mean_valid_prefix_match_ratio": _safe_mean([
            float(r["compare"]["valid_prefix_match_ratio"]) for r in rows if r["compare"] is not None
        ]),
        "vllm_finish_reason_counts": dict(Counter(str(r["vllm_finish_reason"]) for r in vllm_rows)),
    }

    report = {
        "request": {
            "device": args.device,
            "checkpoint_dir": args.checkpoint_dir,
            "audio_prompt_path": args.audio_prompt_path,
            "language_id": args.language_id,
            "vllm_model_dir": args.vllm_model_dir,
            "max_new_tokens": int(args.max_new_tokens),
            "cfg_weight": float(args.cfg_weight),
            "temperature": float(args.temperature),
            "repetition_penalty": float(args.repetition_penalty),
            "min_p": float(args.min_p),
            "top_p": float(args.top_p),
            "save_audio_dir": None if save_audio_dir is None else str(save_audio_dir),
        },
        "timing": {
            "baseline_load_s": float(baseline_load_s),
            "conditioning_prep_s": float(cond_prep_s),
        },
        "summary": summary,
        "rows": rows,
    }

    print("=== T3 Decode Runtime Comparison ===")
    print(f"requests={len(rows)}")
    print(f"baseline_load_s={baseline_load_s:.4f}")
    print(f"conditioning_prep_s={cond_prep_s:.4f}")
    print(
        "baseline_temperature_effective="
        f"{effective_baseline_temperature:.6f} (requested={args.temperature:.6f})"
    )
    print(f"vllm_available={summary['vllm_available']}")
    if summary["vllm_init_error"]:
        print("vllm_init_error=" + str(summary["vllm_init_error"]))
    print("mean_decode_s.baseline=" + str(summary["mean_baseline_decode_s"]))
    print("mean_decode_s.vllm=" + str(summary["mean_vllm_decode_s"]))
    print("mean_valid_prefix_match_ratio=" + str(summary["mean_valid_prefix_match_ratio"]))
    print("vllm_finish_reason_counts=" + json.dumps(summary["vllm_finish_reason_counts"]))

    for row in rows[:3]:
        print(
            f"row[{row['index']}].valid_len baseline={row['baseline']['valid_len']} "
            f"vllm={None if row['vllm'] is None else row['vllm']['valid_len']}"
        )
        print(
            f"row[{row['index']}].audio baseline={row['baseline_audio_path']} "
            f"vllm={row['vllm_audio_path']}"
        )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"output_json={out_path}")

    close = getattr(baseline, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()
