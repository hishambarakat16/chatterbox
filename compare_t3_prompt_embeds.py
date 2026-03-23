from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.vllm_t3_bridge import prepare_vllm_text_tokens


def _load_model(device: str, checkpoint_dir: str | None) -> ChatterboxMultilingualTTS:
    if checkpoint_dir:
        return ChatterboxMultilingualTTS.from_local(checkpoint_dir, device)
    return ChatterboxMultilingualTTS.from_pretrained(device)


def _stats(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, float]:
    lhs_f = lhs.detach().float().reshape(-1)
    rhs_f = rhs.detach().float().reshape(-1)
    diff = lhs_f - rhs_f
    cosine = torch.nn.functional.cosine_similarity(lhs_f.unsqueeze(0), rhs_f.unsqueeze(0), dim=1)
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt((diff * diff).mean()).item()),
        "cosine": float(cosine.item()),
    }


def _slice_segments(x: torch.Tensor, cond_len: int, text_len: int, speech_len: int) -> dict[str, torch.Tensor]:
    return {
        "cond": x[:, :cond_len, :],
        "text": x[:, cond_len : cond_len + text_len, :],
        "speech": x[:, cond_len + text_len : cond_len + text_len + speech_len, :],
    }


def _build_vllm_style_two_bos_embeds(
    *,
    model: ChatterboxMultilingualTTS,
    cond_emb: torch.Tensor,
    text_tokens: torch.Tensor,
) -> torch.Tensor:
    t3 = model.t3
    hp = t3.hp
    batch = text_tokens.shape[0]

    text_emb = t3.text_emb(text_tokens)
    if hp.input_pos_emb == "learned":
        text_emb = text_emb + t3.text_pos_emb(text_tokens)

    speech_tokens = torch.full(
        (batch, 2),
        int(hp.start_speech_token),
        dtype=torch.long,
        device=text_tokens.device,
    )
    speech_emb = t3.speech_emb(speech_tokens)
    if hp.input_pos_emb == "learned":
        # vLLM path maps both BOS rows to speech position 0.
        speech_pos = torch.zeros((batch, 2), dtype=torch.long, device=text_tokens.device)
        speech_pos_emb = t3.speech_pos_emb.get_fixed_embedding(speech_pos).reshape(batch, 2, -1)
        speech_emb = speech_emb + speech_pos_emb.to(dtype=speech_emb.dtype)

    if cond_emb.shape[0] != batch:
        cond_emb = cond_emb.expand(batch, -1, -1)

    return torch.cat((cond_emb, text_emb, speech_emb), dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline prompt embeddings vs vLLM-style rebuilt prompt embeddings "
            "before decode."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--audio-prompt-path", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--language-id", default="ar")
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    t0 = time.perf_counter()
    model = _load_model(args.device, args.checkpoint_dir)
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    model.prepare_conditionals(args.audio_prompt_path, exaggeration=args.exaggeration)
    cond_prep_s = time.perf_counter() - t1

    with torch.inference_mode():
        text_tokens = prepare_vllm_text_tokens(
            tokenizer=model.tokenizer,
            text=args.text,
            language_id=args.language_id,
            device=model.device,
        )
        cond_emb = model.t3.prepare_conditioning(model.conds.t3)

        hp = model.t3.hp
        one_bos = torch.full(
            (text_tokens.shape[0], 1),
            int(hp.start_speech_token),
            dtype=torch.long,
            device=model.device,
        )
        two_bos = torch.full(
            (text_tokens.shape[0], 2),
            int(hp.start_speech_token),
            dtype=torch.long,
            device=model.device,
        )

        baseline_one_bos, cond_len = model.t3.prepare_input_embeds(
            t3_cond=model.conds.t3,
            text_tokens=text_tokens,
            speech_tokens=one_bos,
            cfg_weight=0.0,
        )
        # Production decode path appends one extra BOS embedding after prefill,
        # with the same speech position (0) as the initial BOS row.
        baseline_prod_style_two_bos = torch.cat(
            [baseline_one_bos, baseline_one_bos[:, -1:, :]],
            dim=1,
        )
        baseline_two_bos, _ = model.t3.prepare_input_embeds(
            t3_cond=model.conds.t3,
            text_tokens=text_tokens,
            speech_tokens=two_bos,
            cfg_weight=0.0,
        )
        vllm_two_bos = _build_vllm_style_two_bos_embeds(
            model=model,
            cond_emb=cond_emb,
            text_tokens=text_tokens,
        )

    text_len = int(text_tokens.shape[-1])
    seg_baseline_two = _slice_segments(baseline_two_bos, int(cond_len), text_len, 2)
    seg_vllm_two = _slice_segments(vllm_two_bos, int(cond_len), text_len, 2)

    report = {
        "request": {
            "text": args.text,
            "language_id": args.language_id,
            "audio_prompt_path": args.audio_prompt_path,
            "device": args.device,
            "checkpoint_dir": args.checkpoint_dir,
        },
        "timing": {
            "model_load_s": float(load_s),
            "conditioning_prep_s": float(cond_prep_s),
        },
        "shape": {
            "cond_len": int(cond_len),
            "text_len": int(text_len),
            "baseline_one_bos": list(baseline_one_bos.shape),
            "baseline_prod_style_two_bos": list(baseline_prod_style_two_bos.shape),
            "baseline_two_bos": list(baseline_two_bos.shape),
            "vllm_two_bos": list(vllm_two_bos.shape),
        },
        "compare_baseline_prod_style_two_vs_vllm_two": {
            "all": _stats(baseline_prod_style_two_bos, vllm_two_bos),
        },
        "compare_baseline_two_vs_vllm_two": {
            "all": _stats(baseline_two_bos, vllm_two_bos),
            "cond": _stats(seg_baseline_two["cond"], seg_vllm_two["cond"]),
            "text": _stats(seg_baseline_two["text"], seg_vllm_two["text"]),
            "speech": _stats(seg_baseline_two["speech"], seg_vllm_two["speech"]),
            "speech_row0": _stats(
                seg_baseline_two["speech"][:, :1, :],
                seg_vllm_two["speech"][:, :1, :],
            ),
            "speech_row1": _stats(
                seg_baseline_two["speech"][:, 1:, :],
                seg_vllm_two["speech"][:, 1:, :],
            ),
        },
        "compare_baseline_one_vs_vllm_two_trimmed": {
            "all": _stats(baseline_one_bos, vllm_two_bos[:, :-1, :]),
        },
    }

    print("=== Prompt Embed Comparison (No Decode) ===")
    print(f"text={args.text!r}")
    print(f"model_load_s={load_s:.4f}")
    print(f"conditioning_prep_s={cond_prep_s:.4f}")
    print(f"cond_len={int(cond_len)} text_len={text_len}")
    print(
        "baseline_prod_style_two_vs_vllm_two.all="
        + json.dumps(report["compare_baseline_prod_style_two_vs_vllm_two"]["all"])
    )
    print(
        "baseline_two_vs_vllm_two.all="
        + json.dumps(report["compare_baseline_two_vs_vllm_two"]["all"])
    )
    print(
        "baseline_two_vs_vllm_two.cond="
        + json.dumps(report["compare_baseline_two_vs_vllm_two"]["cond"])
    )
    print(
        "baseline_two_vs_vllm_two.text="
        + json.dumps(report["compare_baseline_two_vs_vllm_two"]["text"])
    )
    print(
        "baseline_two_vs_vllm_two.speech_row0="
        + json.dumps(report["compare_baseline_two_vs_vllm_two"]["speech_row0"])
    )
    print(
        "baseline_two_vs_vllm_two.speech_row1="
        + json.dumps(report["compare_baseline_two_vs_vllm_two"]["speech_row1"])
    )
    print(
        "baseline_one_vs_vllm_two_trimmed.all="
        + json.dumps(report["compare_baseline_one_vs_vllm_two_trimmed"]["all"])
    )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"output_json={out_path}")

    close = getattr(model, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()
