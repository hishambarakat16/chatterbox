import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from chatterbox.mtl_tts import SUPPORTED_LANGUAGES, punc_norm
from chatterbox.mtl_tts_scheduled import ChatterboxMultilingualScheduledTTS
from chatterbox.models.t3.inference.scheduled_decode import (
    ScheduledDecodeRequest,
    _cat_past_key_values,
    _split_past_key_values,
    prepare_scheduled_cohort,
)
from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
from chatterbox.runtime.session import clone_conditionals


def maybe_sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model(device: str, checkpoint_dir: str | None):
    if checkpoint_dir:
        return ChatterboxMultilingualScheduledTTS.from_local(checkpoint_dir, device)
    return ChatterboxMultilingualScheduledTTS.from_pretrained(device)


def percent_change(old: float, new: float) -> float:
    if old == 0.0:
        return 0.0
    return ((new - old) / old) * 100.0


def build_requests(
    model: ChatterboxMultilingualScheduledTTS,
    *,
    text: str,
    language_id: str,
    audio_prompt_path: str | None,
    concurrency: int,
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

    requests = []
    for index in range(concurrency):
        conds = clone_conditionals(session.conditionals).to(worker.device)
        requests.append(
            ScheduledDecodeRequest(
                session_id=f"micro_{index}",
                t3_cond=conds.t3,
                text_tokens=text_tokens.clone(),
                max_new_tokens=max_new_tokens,
                temperature=options.temperature,
                top_p=options.top_p,
                min_p=options.min_p,
                repetition_penalty=options.repetition_penalty,
                cfg_weight=options.cfg_weight,
            )
        )

    return requests, tuple(text_tokens.shape)


def build_backend(t3):
    return T3HuggingfaceBackend(
        config=t3.cfg,
        llama=t3.tfmr,
        speech_enc=t3.speech_emb,
        speech_head=t3.speech_head,
        alignment_stream_analyzer=None,
    )


def advance_cfg_argmax_step(t3, states, logits_step):
    cond = logits_step[0::2, :]
    uncond = logits_step[1::2, :]
    cfg_weights = torch.tensor(
        [state.request.cfg_weight for state in states],
        device=cond.device,
        dtype=cond.dtype,
    ).unsqueeze(-1)
    logits = cond + cfg_weights * (cond - uncond)
    next_tokens = logits.argmax(dim=-1, keepdim=True)

    for index, state in enumerate(states):
        next_token = next_tokens[index : index + 1]
        state.generated_ids = torch.cat([state.generated_ids, next_token], dim=1)
        next_token_embed = t3.speech_emb(next_token)
        next_token_embed = next_token_embed + t3.speech_pos_emb.get_fixed_embedding(state.decode_step + 1)
        state.next_inputs_embeds = torch.cat([next_token_embed, next_token_embed], dim=0)
        state.decode_step += 1


def run_variant(
    *,
    t3,
    requests,
    output_attentions: bool,
    decode_steps: int,
    warmup_runs: int,
    runs: int,
    device: str,
):
    backend = build_backend(t3)
    prefill_times = []
    decode_times = []
    total_times = []

    total_loops = warmup_runs + runs
    for loop_index in range(total_loops):
        cohort = prepare_scheduled_cohort(t3, requests)
        active_states = cohort.active_states

        inputs_embeds = torch.cat(cohort.prefill_inputs, dim=0)
        maybe_sync(device)
        prefill_start = time.perf_counter()
        output = backend(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        maybe_sync(device)
        prefill_s = time.perf_counter() - prefill_start

        past_splits = _split_past_key_values(output.past_key_values, [2] * len(active_states))
        for state, state_past in zip(active_states, past_splits):
            state.past_key_values = state_past
        advance_cfg_argmax_step(t3, active_states, output.logits[:, -1, :])

        decode_s = 0.0
        for _ in range(max(0, decode_steps - 1)):
            batched_past = _cat_past_key_values([state.past_key_values for state in active_states])
            batched_inputs = torch.cat([state.next_inputs_embeds for state in active_states], dim=0)

            maybe_sync(device)
            step_start = time.perf_counter()
            output = backend(
                inputs_embeds=batched_inputs,
                past_key_values=batched_past,
                use_cache=True,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
            )
            maybe_sync(device)
            decode_s += time.perf_counter() - step_start

            past_splits = _split_past_key_values(output.past_key_values, [2] * len(active_states))
            for state, state_past in zip(active_states, past_splits):
                state.past_key_values = state_past
            advance_cfg_argmax_step(t3, active_states, output.logits[:, -1, :])

        if loop_index >= warmup_runs:
            prefill_times.append(prefill_s)
            decode_times.append(decode_s)
            total_times.append(prefill_s + decode_s)

    return {
        "prefill_s": prefill_times,
        "decode_s": decode_times,
        "total_s": total_times,
        "decode_per_step_ms": [
            (value / max(1, decode_steps - 1)) * 1000.0 if decode_steps > 1 else 0.0
            for value in decode_times
        ],
    }


def summarize(values):
    if not values:
        return 0.0
    return round(statistics.mean(values), 4)


def main():
    parser = argparse.ArgumentParser(
        description="Microbenchmark the isolated T3 output_attentions forward-path cost."
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--language-id", required=True)
    parser.add_argument("--audio-prompt-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    load_start = time.perf_counter()
    model = load_model(args.device, args.checkpoint_dir)
    maybe_sync(args.device)
    load_s = time.perf_counter() - load_start

    requests, text_tokens_shape = build_requests(
        model,
        text=args.text,
        language_id=args.language_id,
        audio_prompt_path=args.audio_prompt_path,
        concurrency=args.concurrency,
        max_new_tokens=max(args.decode_steps + 4, 16),
    )
    batch_key = requests[0].batch_key()
    effective_batch_rows = args.concurrency * 2

    off = run_variant(
        t3=model.worker.t3,
        requests=requests,
        output_attentions=False,
        decode_steps=args.decode_steps,
        warmup_runs=args.warmup_runs,
        runs=args.runs,
        device=args.device,
    )
    on = run_variant(
        t3=model.worker.t3,
        requests=requests,
        output_attentions=True,
        decode_steps=args.decode_steps,
        warmup_runs=args.warmup_runs,
        runs=args.runs,
        device=args.device,
    )

    prefill_off = summarize(off["prefill_s"])
    prefill_on = summarize(on["prefill_s"])
    decode_off = summarize(off["decode_s"])
    decode_on = summarize(on["decode_s"])
    total_off = summarize(off["total_s"])
    total_on = summarize(on["total_s"])
    step_off = summarize(off["decode_per_step_ms"])
    step_on = summarize(on["decode_per_step_ms"])

    print("benchmark=t3_output_attentions")
    print("scope=isolated_t3_backend_forward_only")
    print("note=excludes_alignment_hook_logic_and_measures_output_attentions_flag_cost_on_the_same_t3_shapes")
    print(f"device={args.device}")
    print(f"load_s={load_s:.4f}")
    print(f"concurrency={args.concurrency}")
    print(f"effective_batch_rows={effective_batch_rows}")
    print(f"decode_steps={args.decode_steps}")
    print(f"warmup_runs={args.warmup_runs}")
    print(f"runs={args.runs}")
    print(f"text_tokens_shape={text_tokens_shape}")
    print(f"batch_key={batch_key}")
    print(f"prefill_s_off={off['prefill_s']}")
    print(f"prefill_s_on={on['prefill_s']}")
    print(f"prefill_s_mean_off={prefill_off}")
    print(f"prefill_s_mean_on={prefill_on}")
    print(f"prefill_overhead_pct={round(percent_change(prefill_off, prefill_on), 2)}")
    print(f"decode_s_off={off['decode_s']}")
    print(f"decode_s_on={on['decode_s']}")
    print(f"decode_s_mean_off={decode_off}")
    print(f"decode_s_mean_on={decode_on}")
    print(f"decode_overhead_pct={round(percent_change(decode_off, decode_on), 2)}")
    print(f"decode_per_step_ms_mean_off={step_off}")
    print(f"decode_per_step_ms_mean_on={step_on}")
    print(f"decode_per_step_overhead_pct={round(percent_change(step_off, step_on), 2)}")
    print(f"total_t3_s_mean_off={total_off}")
    print(f"total_t3_s_mean_on={total_on}")
    print(f"total_t3_overhead_pct={round(percent_change(total_off, total_on), 2)}")


if __name__ == "__main__":
    main()
