from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from chatterbox.mtl_tts import ChatterboxMultilingualTTS, punc_norm
from chatterbox.vllm_t3_bridge import (
    build_vllm_prompt,
    get_conditioning_seq_len,
    get_vllm_prompt_layout,
    prepare_vllm_text_tokens,
)


def _tensor_shape_dtype_device(t: torch.Tensor | None) -> dict[str, Any] | None:
    if t is None:
        return None
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
    }


def _token_preview(tokens: list[int], limit: int) -> list[int]:
    if limit <= 0:
        return tokens
    return tokens[:limit]


def _first_mismatch_index(lhs: list[int], rhs: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(lhs, rhs)):
        if left != right:
            return index
    if len(lhs) != len(rhs):
        return min(len(lhs), len(rhs))
    return None


def _load_model(device: str, checkpoint_dir: str | None) -> ChatterboxMultilingualTTS:
    if checkpoint_dir:
        return ChatterboxMultilingualTTS.from_local(checkpoint_dir, device)
    return ChatterboxMultilingualTTS.from_pretrained(device)


def _build_baseline_contract(
    *,
    model: ChatterboxMultilingualTTS,
    text: str,
    language_id: str | None,
    cfg_weight: float,
    duplicate_cfg_rows: bool,
) -> dict[str, Any]:
    normalized = punc_norm(text)
    raw_text_tokens = model.tokenizer.text_to_tokens(
        normalized,
        language_id=language_id.lower() if language_id else None,
    ).to(model.device)

    hp = model.t3.hp
    single_row_tokens = F.pad(
        F.pad(raw_text_tokens, (1, 0), value=int(hp.start_text_token)),
        (0, 1),
        value=int(hp.stop_text_token),
    )

    if duplicate_cfg_rows:
        baseline_text_tokens = torch.cat([single_row_tokens, single_row_tokens], dim=0)
    else:
        baseline_text_tokens = single_row_tokens

    initial_speech_tokens = int(hp.start_speech_token) * torch.ones_like(
        baseline_text_tokens[:, :1]
    )

    with torch.inference_mode():
        cond_emb = model.t3.prepare_conditioning(model.conds.t3)
        embeds, len_cond = model.t3.prepare_input_embeds(
            t3_cond=model.conds.t3,
            text_tokens=baseline_text_tokens,
            speech_tokens=initial_speech_tokens,
            cfg_weight=cfg_weight,
        )

    return {
        "text_normalized": normalized,
        "cfg_weight": float(cfg_weight),
        "duplicate_cfg_rows": bool(duplicate_cfg_rows),
        "raw_text_tokens": raw_text_tokens.detach().cpu().squeeze(0).to(torch.long).tolist(),
        "single_row_text_tokens": single_row_tokens.detach().cpu().squeeze(0).to(torch.long).tolist(),
        "text_tokens_for_prepare_input_embeds": baseline_text_tokens.detach()
        .cpu()
        .to(torch.long)
        .tolist(),
        "initial_speech_tokens": initial_speech_tokens.detach().cpu().to(torch.long).tolist(),
        "len_cond": int(len_cond),
        "cond_emb": _tensor_shape_dtype_device(cond_emb),
        "embeds": _tensor_shape_dtype_device(embeds),
    }


def _build_vllm_contract(
    *,
    model: ChatterboxMultilingualTTS,
    text: str,
    language_id: str | None,
) -> dict[str, Any]:
    hp = model.t3.hp
    layout = get_vllm_prompt_layout(hp)
    text_tokens = prepare_vllm_text_tokens(
        tokenizer=model.tokenizer,
        text=text,
        language_id=language_id,
        device="cpu",
    )
    prompt, prompt_meta = build_vllm_prompt(
        t3_cond=model.conds.t3,
        text_tokens=text_tokens,
        return_metadata=True,
    )

    prompt_token_ids = [int(token_id) for token_id in prompt["prompt_token_ids"]]
    text_len = int(text_tokens.shape[-1])
    text_segment_with_offset = prompt_token_ids[1 : 1 + text_len]
    text_segment_no_offset = [
        int(token_id - int(layout["text_token_offset"]))
        for token_id in text_segment_with_offset
    ]
    speech_segment = prompt_token_ids[1 + text_len :]
    conditioning = prompt["multi_modal_data"]["conditioning"]

    return {
        "layout": {
            "speech_vocab_size": int(layout["speech_vocab_size"]),
            "text_vocab_size": int(layout["text_vocab_size"]),
            "text_token_offset": int(layout["text_token_offset"]),
            "conditioning_token_id": int(layout["conditioning_token_id"]),
            "conditioning_seq_len": int(layout["conditioning_seq_len"]),
            "input_vocab_size": int(layout["input_vocab_size"]),
        },
        "text_tokens_for_vllm_builder": text_tokens.detach().cpu().squeeze(0).to(torch.long).tolist(),
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_parts": {
            "conditioning_token": int(prompt_token_ids[0]),
            "text_tokens_with_offset": text_segment_with_offset,
            "text_tokens_without_offset": text_segment_no_offset,
            "speech_prefix_tokens": speech_segment,
        },
        "conditioning_payload": {
            "speaker_emb": _tensor_shape_dtype_device(conditioning.get("speaker_emb")),
            "cond_prompt_speech_tokens": _tensor_shape_dtype_device(
                conditioning.get("cond_prompt_speech_tokens")
            ),
            "emotion_adv": _tensor_shape_dtype_device(conditioning.get("emotion_adv")),
        },
        "meta": {
            "prompt_speech_token_len": int(prompt_meta["prompt_speech_token_len"]),
            "text_token_len": int(prompt_meta["text_token_len"]),
            "initial_speech_len": int(prompt_meta["initial_speech_len"]),
            "prompt_seq_len": int(prompt_meta["prompt_seq_len"]),
            "prompt_hidden_size": int(prompt_meta["prompt_hidden_size"]),
            "cond_seq_len": int(prompt_meta["cond_seq_len"]),
            "prompt_token_len_before_mm": int(prompt_meta["prompt_token_len_before_mm"]),
        },
    }


def _build_alignment_report(
    *,
    baseline: dict[str, Any],
    vllm: dict[str, Any],
) -> dict[str, Any]:
    baseline_single = [int(token_id) for token_id in baseline["single_row_text_tokens"]]
    vllm_text = [int(token_id) for token_id in vllm["prompt_token_parts"]["text_tokens_without_offset"]]
    mismatch_index = _first_mismatch_index(baseline_single, vllm_text)
    expected_cond_len = int(get_conditioning_seq_len())

    baseline_cfg_rows = len(baseline["text_tokens_for_prepare_input_embeds"])
    baseline_speech_prefix = [
        int(row[0]) for row in baseline["initial_speech_tokens"]
    ]
    vllm_speech_prefix = [
        int(token_id) for token_id in vllm["prompt_token_parts"]["speech_prefix_tokens"]
    ]

    return {
        "text_tokens_exact_match": mismatch_index is None,
        "text_token_mismatch_index": mismatch_index,
        "text_token_lengths": {
            "baseline_single_row": len(baseline_single),
            "vllm_text_segment": len(vllm_text),
        },
        "cond_seq_len_match": int(baseline["len_cond"]) == int(vllm["meta"]["cond_seq_len"]),
        "cond_seq_len_values": {
            "baseline_len_cond": int(baseline["len_cond"]),
            "vllm_meta_cond_seq_len": int(vllm["meta"]["cond_seq_len"]),
            "expected_from_layout": expected_cond_len,
        },
        "speech_prefix": {
            "baseline_initial_speech_tokens": baseline_speech_prefix,
            "baseline_cfg_rows": baseline_cfg_rows,
            "vllm_speech_prefix_tokens": vllm_speech_prefix,
            "same_speech_prefix_values": baseline_speech_prefix
            and all(token_id == baseline_speech_prefix[0] for token_id in vllm_speech_prefix),
        },
        "vllm_boundary_tokens": {
            "conditioning_token_matches_layout": (
                int(vllm["prompt_token_parts"]["conditioning_token"])
                == int(vllm["layout"]["conditioning_token_id"])
            ),
            "last_two_prompt_tokens_are_bos": (
                len(vllm_speech_prefix) >= 2
                and int(vllm_speech_prefix[-1]) == int(vllm_speech_prefix[-2])
            ),
        },
    }


def _print_summary(report: dict[str, Any], token_preview_limit: int) -> None:
    baseline = report["baseline"]
    vllm = report["vllm"]
    alignment = report["alignment"]

    print("=== T3 Input Contract Comparison (No Decode) ===")
    print(f"text={report['request']['text']!r}")
    print(f"language_id={report['request']['language_id']}")
    print(f"audio_prompt_path={report['request']['audio_prompt_path']}")
    print(f"device={report['request']['device']}")
    print(f"baseline_load_s={report['timing']['baseline_load_s']:.4f}")
    print(f"conditioning_prep_s={report['timing']['conditioning_prep_s']:.4f}")
    print("")
    print("--- Baseline Path ---")
    print(f"text_normalized={baseline['text_normalized']!r}")
    print(f"len_cond={baseline['len_cond']}")
    print(f"text_rows_for_prepare_input_embeds={len(baseline['text_tokens_for_prepare_input_embeds'])}")
    print(f"single_row_text_tokens={_token_preview(baseline['single_row_text_tokens'], token_preview_limit)}")
    print("")
    print("--- vLLM Builder Path ---")
    print(f"conditioning_seq_len={vllm['layout']['conditioning_seq_len']}")
    print(f"prompt_token_len_before_mm={vllm['meta']['prompt_token_len_before_mm']}")
    print(f"prompt_seq_len_after_mm={vllm['meta']['prompt_seq_len']}")
    print(
        "prompt_token_ids_preview="
        f"{_token_preview(vllm['prompt_token_ids'], token_preview_limit)}"
    )
    print("")
    print("--- Alignment Checks ---")
    print(f"text_tokens_exact_match={alignment['text_tokens_exact_match']}")
    print(f"text_token_mismatch_index={alignment['text_token_mismatch_index']}")
    print(f"cond_seq_len_match={alignment['cond_seq_len_match']}")
    print(
        "baseline_initial_speech_tokens="
        f"{alignment['speech_prefix']['baseline_initial_speech_tokens']}"
    )
    print(
        "vllm_speech_prefix_tokens="
        f"{alignment['speech_prefix']['vllm_speech_prefix_tokens']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the baseline T3 input assembly and vLLM prompt assembly "
            "without running decode."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--audio-prompt-path", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--language-id", default="ar")
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--cfg-weight", type=float, default=0.0)
    parser.add_argument(
        "--no-baseline-cfg-row-duplication",
        action="store_true",
        help=(
            "Disable the baseline generate-path duplication that usually creates two "
            "text rows for CFG."
        ),
    )
    parser.add_argument(
        "--token-preview-limit",
        type=int,
        default=24,
        help="How many token ids to print in console previews. Use <=0 for full lists.",
    )
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    t0 = time.perf_counter()
    model = _load_model(args.device, args.checkpoint_dir)
    baseline_load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    model.prepare_conditionals(args.audio_prompt_path, exaggeration=args.exaggeration)
    conditioning_prep_s = time.perf_counter() - t1

    baseline = _build_baseline_contract(
        model=model,
        text=args.text,
        language_id=args.language_id,
        cfg_weight=float(args.cfg_weight),
        duplicate_cfg_rows=(not args.no_baseline_cfg_row_duplication),
    )
    vllm = _build_vllm_contract(
        model=model,
        text=args.text,
        language_id=args.language_id,
    )
    alignment = _build_alignment_report(
        baseline=baseline,
        vllm=vllm,
    )

    report = {
        "request": {
            "text": args.text,
            "language_id": args.language_id,
            "audio_prompt_path": args.audio_prompt_path,
            "device": args.device,
            "cfg_weight": float(args.cfg_weight),
            "baseline_cfg_row_duplication": (not args.no_baseline_cfg_row_duplication),
            "checkpoint_dir": args.checkpoint_dir,
        },
        "timing": {
            "baseline_load_s": float(baseline_load_s),
            "conditioning_prep_s": float(conditioning_prep_s),
        },
        "baseline": baseline,
        "vllm": vllm,
        "alignment": alignment,
    }

    _print_summary(report, args.token_preview_limit)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"output_json={out_path}")

    close = getattr(model, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()
