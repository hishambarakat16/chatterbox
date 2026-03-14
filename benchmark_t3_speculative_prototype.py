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
from chatterbox.models.t3.inference.draft_model import build_layer_subset_draft_model
from chatterbox.models.t3.inference.scheduled_decode import ScheduledDecodeRequest
from chatterbox.models.t3.inference.speculative_decode import (
    run_baseline_greedy_decode,
    run_self_speculative_decode,
    run_speculative_decode_with_draft,
)
from chatterbox.runtime.session import clone_conditionals


def maybe_sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def is_cuda_device(device: str) -> bool:
    return device.startswith("cuda") and torch.cuda.is_available()


def reset_cuda_peak_stats(device: str):
    if is_cuda_device(device):
        torch.cuda.reset_peak_memory_stats(device)


def capture_cuda_memory_stats(device: str) -> dict[str, float]:
    if not is_cuda_device(device):
        return {
            "allocated_start_mb": 0.0,
            "reserved_start_mb": 0.0,
            "allocated_end_mb": 0.0,
            "reserved_end_mb": 0.0,
            "peak_allocated_mb": 0.0,
            "peak_reserved_mb": 0.0,
            "peak_allocated_delta_mb": 0.0,
            "peak_reserved_delta_mb": 0.0,
        }

    to_mb = 1024.0 * 1024.0
    allocated_start = torch.cuda.memory_allocated(device) / to_mb
    reserved_start = torch.cuda.memory_reserved(device) / to_mb
    return {
        "allocated_start_mb": allocated_start,
        "reserved_start_mb": reserved_start,
        "allocated_end_mb": 0.0,
        "reserved_end_mb": 0.0,
        "peak_allocated_mb": 0.0,
        "peak_reserved_mb": 0.0,
        "peak_allocated_delta_mb": 0.0,
        "peak_reserved_delta_mb": 0.0,
    }


def finalize_cuda_memory_stats(device: str, memory_stats: dict[str, float]) -> dict[str, float]:
    if not is_cuda_device(device):
        return memory_stats

    to_mb = 1024.0 * 1024.0
    allocated_end = torch.cuda.memory_allocated(device) / to_mb
    reserved_end = torch.cuda.memory_reserved(device) / to_mb
    peak_allocated = torch.cuda.max_memory_allocated(device) / to_mb
    peak_reserved = torch.cuda.max_memory_reserved(device) / to_mb
    memory_stats["allocated_end_mb"] = allocated_end
    memory_stats["reserved_end_mb"] = reserved_end
    memory_stats["peak_allocated_mb"] = peak_allocated
    memory_stats["peak_reserved_mb"] = peak_reserved
    memory_stats["peak_allocated_delta_mb"] = peak_allocated - memory_stats["allocated_start_mb"]
    memory_stats["peak_reserved_delta_mb"] = peak_reserved - memory_stats["reserved_start_mb"]
    return memory_stats


def mean_or_zero(values: list[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def configure_shape_logging(enabled: bool, trace_spec_every: int):
    if not enabled:
        os.environ.pop("CHATTERBOX_TRACE_SHAPES", None)
        os.environ.pop("CHATTERBOX_TRACE_SPEC_EVERY", None)
        return

    os.environ["CHATTERBOX_TRACE_SHAPES"] = "1"
    os.environ["CHATTERBOX_TRACE_SPEC_EVERY"] = str(max(trace_spec_every, 1))
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


def run_timed_baseline(device: str, t3, request: ScheduledDecodeRequest):
    maybe_sync(device)
    memory_stats = capture_cuda_memory_stats(device)
    reset_cuda_peak_stats(device)
    started = time.perf_counter()
    tokens = run_baseline_greedy_decode(t3, request)
    maybe_sync(device)
    elapsed = time.perf_counter() - started
    memory_stats = finalize_cuda_memory_stats(device, memory_stats)
    return tokens, elapsed, memory_stats


def run_timed_speculative(device: str, target_t3, draft_t3, request: ScheduledDecodeRequest, speculate_k: int):
    maybe_sync(device)
    memory_stats = capture_cuda_memory_stats(device)
    reset_cuda_peak_stats(device)
    started = time.perf_counter()
    if draft_t3 is target_t3:
        result = run_self_speculative_decode(
            target_t3,
            request,
            speculate_k=speculate_k,
        )
    else:
        result = run_speculative_decode_with_draft(
            target_t3,
            draft_t3,
            request,
            speculate_k=speculate_k,
        )
    maybe_sync(device)
    elapsed = time.perf_counter() - started
    memory_stats = finalize_cuda_memory_stats(device, memory_stats)
    return result, elapsed, memory_stats


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
    parser.add_argument("--draft-mode", choices=["self", "layer_subset"], default="self")
    parser.add_argument("--draft-layers", type=int, default=12)
    parser.add_argument("--draft-layer-selection", choices=["even", "first", "last"], default="even")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--trace-shapes", action="store_true")
    parser.add_argument("--trace-spec-every", type=int, default=6)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    configure_shape_logging(args.trace_shapes, args.trace_spec_every)

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
    draft_t3 = model.worker.t3
    draft_metadata = None
    if args.draft_mode == "layer_subset":
        draft_t3, draft_metadata = build_layer_subset_draft_model(
            model.worker.t3,
            num_layers=args.draft_layers,
            layer_selection=args.draft_layer_selection,
        )

    total_runs = args.warmup_runs + args.runs
    orders = ["baseline_first" if run_index % 2 == 0 else "speculative_first" for run_index in range(total_runs)]

    baseline_times: list[float] = []
    speculative_times: list[float] = []
    baseline_tokens_per_s: list[float] = []
    speculative_tokens_per_s: list[float] = []
    baseline_peak_allocated_delta_mb: list[float] = []
    baseline_peak_reserved_delta_mb: list[float] = []
    speculative_peak_allocated_delta_mb: list[float] = []
    speculative_peak_reserved_delta_mb: list[float] = []
    speculative_rounds: list[int] = []
    speculative_proposed_tokens_total: list[int] = []
    speculative_accepted_draft_tokens_total: list[int] = []
    speculative_correction_tokens_total: list[int] = []
    speculative_acceptance_rates: list[float] = []
    speculative_rebuild_count: list[int] = []
    speculative_rebuild_tokens_total: list[int] = []
    speculative_full_accept_rounds: list[int] = []
    speculative_zero_accept_rounds: list[int] = []
    speculative_partial_accept_rounds: list[int] = []
    speculative_match_len_hist_total = [0] * (args.speculate_k + 1)
    exact_match_all_runs = True
    last_mismatch_index = None
    baseline_tokens = None
    speculative_tokens = None

    for run_index, order in enumerate(orders):
        measured = run_index >= args.warmup_runs

        if order == "baseline_first":
            baseline_tokens_run, baseline_elapsed, baseline_memory = run_timed_baseline(
                args.device,
                model.worker.t3,
                request,
            )
            speculative_run, speculative_elapsed, speculative_memory = run_timed_speculative(
                args.device,
                model.worker.t3,
                draft_t3,
                request,
                args.speculate_k,
            )
        else:
            speculative_run, speculative_elapsed, speculative_memory = run_timed_speculative(
                args.device,
                model.worker.t3,
                draft_t3,
                request,
                args.speculate_k,
            )
            baseline_tokens_run, baseline_elapsed, baseline_memory = run_timed_baseline(
                args.device,
                model.worker.t3,
                request,
            )

        speculative_tokens_run = speculative_run.speech_tokens
        mismatch_index = first_mismatch_index(baseline_tokens_run, speculative_tokens_run)
        exact_match = mismatch_index is None
        if not exact_match:
            exact_match_all_runs = False
            last_mismatch_index = mismatch_index
            break

        baseline_tokens = baseline_tokens_run
        speculative_tokens = speculative_tokens_run

        if not measured:
            continue

        baseline_times.append(baseline_elapsed)
        speculative_times.append(speculative_elapsed)
        baseline_tokens_per_s.append(
            0.0 if baseline_elapsed == 0.0 else baseline_tokens_run.size(1) / baseline_elapsed
        )
        speculative_tokens_per_s.append(
            0.0 if speculative_elapsed == 0.0 else speculative_tokens_run.size(1) / speculative_elapsed
        )
        baseline_peak_allocated_delta_mb.append(baseline_memory["peak_allocated_delta_mb"])
        baseline_peak_reserved_delta_mb.append(baseline_memory["peak_reserved_delta_mb"])
        speculative_peak_allocated_delta_mb.append(speculative_memory["peak_allocated_delta_mb"])
        speculative_peak_reserved_delta_mb.append(speculative_memory["peak_reserved_delta_mb"])
        speculative_rounds.append(speculative_run.rounds)
        speculative_proposed_tokens_total.append(speculative_run.proposed_tokens_total)
        speculative_accepted_draft_tokens_total.append(speculative_run.accepted_draft_tokens_total)
        speculative_correction_tokens_total.append(speculative_run.correction_tokens_total)
        speculative_rebuild_count.append(speculative_run.rebuild_count)
        speculative_rebuild_tokens_total.append(speculative_run.rebuild_tokens_total)
        speculative_full_accept_rounds.append(speculative_run.full_accept_rounds)
        speculative_zero_accept_rounds.append(speculative_run.zero_accept_rounds)
        speculative_partial_accept_rounds.append(speculative_run.partial_accept_rounds)
        speculative_acceptance_rates.append(
            0.0 if speculative_run.proposed_tokens_total == 0
            else speculative_run.accepted_draft_tokens_total / speculative_run.proposed_tokens_total
        )
        for index, count in enumerate(speculative_run.match_len_hist):
            speculative_match_len_hist_total[index] += count

    if not exact_match_all_runs:
        raise AssertionError(f"speculative prototype tokens did not match baseline greedy decode at index {last_mismatch_index}")

    saved_wavs = []
    rendered = {}
    if args.output_dir and baseline_tokens is not None and speculative_tokens is not None:
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
    mode = "self_draft_greedy_cfg_on_alignment_off" if args.draft_mode == "self" else "layer_subset_draft_greedy_cfg_on_alignment_off"
    print(f"mode={mode}")
    print(f"device={args.device}")
    print(f"load_s={load_s:.4f}")
    print(f"speculate_k={args.speculate_k}")
    print(f"max_new_tokens={args.max_new_tokens}")
    print(f"draft_mode={args.draft_mode}")
    if draft_metadata is not None:
        print(f"draft_layers={draft_metadata.num_layers}")
        print(f"draft_total_layers={draft_metadata.total_layers}")
        print(f"draft_layer_selection={draft_metadata.layer_selection}")
        print(f"draft_layer_indices={list(draft_metadata.layer_indices)}")
    print(f"warmup_runs={args.warmup_runs}")
    print(f"runs={args.runs}")
    print(f"run_orders={orders}")
    print(f"trace_spec_every={max(args.trace_spec_every, 1)}")
    print(f"text_tokens_shape={tuple(request.text_tokens.shape)}")
    print(f"cfg_weight={request.cfg_weight}")
    print(f"baseline_t3_s={baseline_times}")
    print(f"speculative_t3_s={speculative_times}")
    print(f"baseline_t3_s_mean={mean_or_zero(baseline_times):.4f}")
    print(f"speculative_t3_s_mean={mean_or_zero(speculative_times):.4f}")
    mean_baseline = mean_or_zero(baseline_times)
    mean_speculative = mean_or_zero(speculative_times)
    speedup_pct = 0.0 if mean_baseline == 0.0 else ((mean_baseline - mean_speculative) / mean_baseline) * 100.0
    print(f"speculative_vs_baseline_speedup_pct={speedup_pct:.2f}")
    print(f"baseline_tokens_per_s={baseline_tokens_per_s}")
    print(f"speculative_tokens_per_s={speculative_tokens_per_s}")
    print(f"baseline_tokens_per_s_mean={mean_or_zero(baseline_tokens_per_s):.4f}")
    print(f"speculative_tokens_per_s_mean={mean_or_zero(speculative_tokens_per_s):.4f}")
    print(f"baseline_peak_allocated_delta_mb={baseline_peak_allocated_delta_mb}")
    print(f"baseline_peak_reserved_delta_mb={baseline_peak_reserved_delta_mb}")
    print(f"speculative_peak_allocated_delta_mb={speculative_peak_allocated_delta_mb}")
    print(f"speculative_peak_reserved_delta_mb={speculative_peak_reserved_delta_mb}")
    print(f"baseline_peak_allocated_delta_mb_mean={mean_or_zero(baseline_peak_allocated_delta_mb):.4f}")
    print(f"baseline_peak_reserved_delta_mb_mean={mean_or_zero(baseline_peak_reserved_delta_mb):.4f}")
    print(f"speculative_peak_allocated_delta_mb_mean={mean_or_zero(speculative_peak_allocated_delta_mb):.4f}")
    print(f"speculative_peak_reserved_delta_mb_mean={mean_or_zero(speculative_peak_reserved_delta_mb):.4f}")
    print(f"baseline_num_tokens={baseline_tokens.size(1) if baseline_tokens is not None else 0}")
    print(f"speculative_num_tokens={speculative_tokens.size(1) if speculative_tokens is not None else 0}")
    print(f"speculative_rounds={speculative_rounds}")
    print(f"speculative_rounds_mean={mean_or_zero([float(value) for value in speculative_rounds]):.4f}")
    print(f"speculative_proposed_tokens_total={speculative_proposed_tokens_total}")
    print(f"speculative_accepted_draft_tokens_total={speculative_accepted_draft_tokens_total}")
    print(f"speculative_correction_tokens_total={speculative_correction_tokens_total}")
    print(f"speculative_rebuild_count={speculative_rebuild_count}")
    print(f"speculative_rebuild_count_mean={mean_or_zero([float(value) for value in speculative_rebuild_count]):.4f}")
    print(f"speculative_rebuild_tokens_total={speculative_rebuild_tokens_total}")
    print(f"speculative_rebuild_tokens_total_mean={mean_or_zero([float(value) for value in speculative_rebuild_tokens_total]):.4f}")
    print(f"speculative_full_accept_rounds={speculative_full_accept_rounds}")
    print(f"speculative_zero_accept_rounds={speculative_zero_accept_rounds}")
    print(f"speculative_partial_accept_rounds={speculative_partial_accept_rounds}")
    print(f"speculative_match_len_hist_total={speculative_match_len_hist_total}")
    print(f"speculative_acceptance_rate={speculative_acceptance_rates}")
    print(f"speculative_acceptance_rate_mean={mean_or_zero(speculative_acceptance_rates):.4f}")
    print(f"exact_token_match={exact_match_all_runs}")
    print(f"first_mismatch_index={last_mismatch_index}")
    print(f"saved_wavs={saved_wavs}")
    for key, value in rendered.items():
        print(f"{key}={value}")

if __name__ == "__main__":
    main()
