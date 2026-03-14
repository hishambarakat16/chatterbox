import argparse
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio as ta

from chatterbox.mtl_tts import SUPPORTED_LANGUAGES, punc_norm
from chatterbox.mtl_tts_scheduled import ChatterboxMultilingualScheduledTTS
from chatterbox.models.s3tokenizer import drop_invalid_tokens
from chatterbox.models.t3.inference.scheduled_decode import ScheduledDecodeRequest
from chatterbox.models.t3.inference.speculative_decode import (
    run_baseline_greedy_decode,
    run_self_speculative_decode,
)
from chatterbox.runtime.session import clone_conditionals


def maybe_sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def configure_shape_logging(enabled: bool):
    if not enabled:
        os.environ.pop("CHATTERBOX_TRACE_SHAPES", None)
        return

    os.environ["CHATTERBOX_TRACE_SHAPES"] = "1"
    logger = logging.getLogger("chatterbox.shape")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)


def load_model(device: str, checkpoint_dir: str | None):
    if checkpoint_dir:
        return ChatterboxMultilingualScheduledTTS.from_local(checkpoint_dir, device)
    return ChatterboxMultilingualScheduledTTS.from_pretrained(device)


def build_single_request(
    model: ChatterboxMultilingualScheduledTTS,
    *,
    text: str,
    language_id: str,
    audio_prompt_path: str | None,
    max_new_tokens: int,
):
    worker = model.worker
    if language_id.lower() not in SUPPORTED_LANGUAGES:
        supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
        raise ValueError(f"Unsupported language_id '{language_id}'. Supported languages: {supported_langs}")

    session = model.create_session(
        audio_prompt_path=audio_prompt_path,
        language_id=language_id,
        max_new_tokens=max_new_tokens,
    )
    options = session.options

    normalized_text = punc_norm(text)
    text_tokens = worker.tokenizer.text_to_tokens(
        normalized_text,
        language_id=language_id.lower(),
    ).to(worker.device)
    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
    text_tokens = F.pad(text_tokens, (1, 0), value=worker.t3.hp.start_text_token)
    text_tokens = F.pad(text_tokens, (0, 1), value=worker.t3.hp.stop_text_token)

    conds = clone_conditionals(session.conditionals).to(worker.device)
    request = ScheduledDecodeRequest(
        session_id="speculative_proto",
        t3_cond=conds.t3,
        text_tokens=text_tokens,
        max_new_tokens=max_new_tokens,
        temperature=options.temperature,
        top_p=options.top_p,
        min_p=options.min_p,
        repetition_penalty=options.repetition_penalty,
        cfg_weight=options.cfg_weight,
    )
    return request, session


def first_mismatch_index(left: torch.Tensor, right: torch.Tensor) -> int | None:
    compare_len = min(left.size(1), right.size(1))
    for index in range(compare_len):
        if left[0, index].item() != right[0, index].item():
            return index
    if left.size(1) != right.size(1):
        return compare_len
    return None


def render_tokens(worker, session, tokens: torch.Tensor, output_path: Path):
    filtered = drop_invalid_tokens(tokens[0]).to(worker.device)
    wav, _ = worker.s3gen.inference(
        speech_tokens=filtered,
        ref_dict=clone_conditionals(session.conditionals).gen,
    )
    ta.save(str(output_path), wav.cpu(), worker.sr)
    return int(filtered.numel()), int(wav.shape[-1])


def main():
    parser = argparse.ArgumentParser(
        description="Prototype speculative decoding on the current multilingual T3 using a self-draft greedy path."
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--language-id", required=True)
    parser.add_argument("--audio-prompt-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--speculate-k", type=int, default=4)
    parser.add_argument("--trace-shapes", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    configure_shape_logging(args.trace_shapes)

    load_started = time.perf_counter()
    model = load_model(args.device, args.checkpoint_dir)
    maybe_sync(args.device)
    load_s = time.perf_counter() - load_started

    request, session = build_single_request(
        model,
        text=args.text,
        language_id=args.language_id,
        audio_prompt_path=args.audio_prompt_path,
        max_new_tokens=args.max_new_tokens,
    )

    maybe_sync(args.device)
    baseline_started = time.perf_counter()
    baseline_tokens = run_baseline_greedy_decode(model.worker.t3, request)
    maybe_sync(args.device)
    baseline_s = time.perf_counter() - baseline_started

    maybe_sync(args.device)
    speculative_started = time.perf_counter()
    speculative = run_self_speculative_decode(
        model.worker.t3,
        request,
        speculate_k=args.speculate_k,
    )
    maybe_sync(args.device)
    speculative_s = time.perf_counter() - speculative_started

    speculative_tokens = speculative.speech_tokens
    mismatch_index = first_mismatch_index(baseline_tokens, speculative_tokens)
    exact_match = mismatch_index is None

    saved_wavs = []
    rendered = {}
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        baseline_path = output_dir / "baseline_greedy.wav"
        speculative_path = output_dir / "speculative_self_draft.wav"
        baseline_token_count, baseline_samples = render_tokens(model.worker, session, baseline_tokens, baseline_path)
        speculative_token_count, speculative_samples = render_tokens(model.worker, session, speculative_tokens, speculative_path)
        saved_wavs = [str(baseline_path), str(speculative_path)]
        rendered = {
            "baseline_rendered_token_count": baseline_token_count,
            "baseline_rendered_num_samples": baseline_samples,
            "speculative_rendered_token_count": speculative_token_count,
            "speculative_rendered_num_samples": speculative_samples,
        }

    print("benchmark=t3_speculative_prototype")
    print("mode=self_draft_greedy_cfg_on_alignment_off")
    print(f"device={args.device}")
    print(f"load_s={load_s:.4f}")
    print(f"speculate_k={args.speculate_k}")
    print(f"max_new_tokens={args.max_new_tokens}")
    print(f"text_tokens_shape={tuple(request.text_tokens.shape)}")
    print(f"cfg_weight={request.cfg_weight}")
    print(f"baseline_t3_s={baseline_s:.4f}")
    print(f"speculative_t3_s={speculative_s:.4f}")
    print(f"baseline_num_tokens={baseline_tokens.size(1)}")
    print(f"speculative_num_tokens={speculative_tokens.size(1)}")
    print(f"speculative_rounds={speculative.rounds}")
    print(f"speculative_proposed_tokens_total={speculative.proposed_tokens_total}")
    print(f"speculative_accepted_draft_tokens_total={speculative.accepted_draft_tokens_total}")
    print(f"speculative_correction_tokens_total={speculative.correction_tokens_total}")
    acceptance_rate = (
        0.0 if speculative.proposed_tokens_total == 0
        else speculative.accepted_draft_tokens_total / speculative.proposed_tokens_total
    )
    print(f"speculative_acceptance_rate={acceptance_rate:.4f}")
    print(f"exact_token_match={exact_match}")
    print(f"first_mismatch_index={mismatch_index}")
    if mismatch_index is not None:
        print(f"baseline_mismatch_token={baseline_tokens[0, mismatch_index].item() if mismatch_index < baseline_tokens.size(1) else None}")
        print(f"speculative_mismatch_token={speculative_tokens[0, mismatch_index].item() if mismatch_index < speculative_tokens.size(1) else None}")
    print(f"saved_wavs={saved_wavs}")
    for key, value in rendered.items():
        print(f"{key}={value}")

    if not exact_match:
        raise AssertionError("speculative prototype tokens did not match baseline greedy decode")


if __name__ == "__main__":
    main()
