import logging
import os
from dataclasses import dataclass

import torch
from torch import Tensor

from .scheduled_decode import ScheduledDecodeRequest, prepare_scheduled_cohort


shape_logger = logging.getLogger("chatterbox.shape")


def _trace_enabled() -> bool:
    return bool(os.getenv("CHATTERBOX_TRACE_SHAPES"))


def _cfg_combine(raw_logits: Tensor, cfg_weight: float) -> Tensor:
    assert raw_logits.size(0) == 2, f"expected 2 CFG rows, got {tuple(raw_logits.shape)}"
    cond = raw_logits[0:1]
    uncond = raw_logits[1:2]
    cfg = torch.as_tensor(cfg_weight, device=raw_logits.device, dtype=raw_logits.dtype)
    return cond + cfg * (cond - uncond)


def _kv_seq_len(past_key_values) -> int:
    return int(past_key_values[0][0].shape[2])


def _forward_raw_t3(
    t3,
    *,
    inputs_embeds: Tensor,
    past_key_values=None,
):
    tfmr_out = t3.tfmr(
        inputs_embeds=inputs_embeds,
        past_key_values=past_key_values,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    logits = t3.speech_head(tfmr_out.last_hidden_state)
    return logits, tfmr_out.past_key_values


def _build_pos_embed_block(t3, *, start_pos: int, num_tokens: int, device) -> Tensor:
    pos_embeds = [
        t3.speech_pos_emb.get_fixed_embedding(start_pos + offset).to(device=device)
        for offset in range(num_tokens)
    ]
    pos_block = torch.cat(pos_embeds, dim=1)
    assert pos_block.dim() == 3 and pos_block.size(0) == 1 and pos_block.size(1) == num_tokens
    return pos_block


def _build_cfg_step_inputs(t3, *, tokens: Tensor, start_pos: int) -> Tensor:
    assert tokens.dim() == 2 and tokens.size(0) == 1, f"expected (1, K) tokens, got {tuple(tokens.shape)}"
    base_embed = t3.speech_emb(tokens)
    pos_embed = _build_pos_embed_block(
        t3,
        start_pos=start_pos,
        num_tokens=tokens.size(1),
        device=base_embed.device,
    )
    assert base_embed.shape == pos_embed.shape, (tuple(base_embed.shape), tuple(pos_embed.shape))
    single = base_embed + pos_embed
    duplicated = torch.cat([single, single], dim=0)
    if _trace_enabled():
        shape_logger.info("[models/t3/inference/speculative_decode.py] build_cfg_step_inputs")
        shape_logger.info("  tokens %s %s %s", tuple(tokens.shape), tokens.dtype, tokens.device)
        shape_logger.info("  base_embed %s %s %s", tuple(base_embed.shape), base_embed.dtype, base_embed.device)
        shape_logger.info("  pos_embed %s %s %s", tuple(pos_embed.shape), pos_embed.dtype, pos_embed.device)
        shape_logger.info("  duplicated %s %s %s", tuple(duplicated.shape), duplicated.dtype, duplicated.device)
    return duplicated


@dataclass
class _PrototypeState:
    request: ScheduledDecodeRequest
    generated_ids: Tensor
    past_key_values: object
    next_logits: Tensor
    decode_step: int = 0


@dataclass
class VerifyResult:
    committed_tokens: Tensor
    accepted_draft_tokens: int
    proposed_tokens: int
    correction_token: Tensor | None
    next_logits: Tensor
    next_past_key_values: object


@dataclass
class SpeculativePrototypeResult:
    speech_tokens: Tensor
    rounds: int
    proposed_tokens_total: int
    accepted_draft_tokens_total: int
    correction_tokens_total: int


@torch.inference_mode()
def prefill_prototype_state(t3, request: ScheduledDecodeRequest) -> _PrototypeState:
    cohort = prepare_scheduled_cohort(t3, [request])
    assert len(cohort.active_states) == 1
    state = cohort.active_states[0]
    inputs_embeds = torch.cat(cohort.prefill_inputs, dim=0)
    assert inputs_embeds.size(0) == 2, tuple(inputs_embeds.shape)

    raw_logits, past_key_values = _forward_raw_t3(
        t3,
        inputs_embeds=inputs_embeds,
        past_key_values=None,
    )
    next_logits = _cfg_combine(raw_logits[:, -1, :], request.cfg_weight)
    assert next_logits.shape[0] == 1, tuple(next_logits.shape)

    if _trace_enabled():
        shape_logger.info("[models/t3/inference/speculative_decode.py] prefill")
        shape_logger.info("  inputs_embeds %s %s %s", tuple(inputs_embeds.shape), inputs_embeds.dtype, inputs_embeds.device)
        shape_logger.info("  raw_logits %s %s %s", tuple(raw_logits.shape), raw_logits.dtype, raw_logits.device)
        shape_logger.info("  next_logits %s %s %s", tuple(next_logits.shape), next_logits.dtype, next_logits.device)
        shape_logger.info("  kv_seq_len %s", _kv_seq_len(past_key_values))

    return _PrototypeState(
        request=request,
        generated_ids=state.generated_ids.clone(),
        past_key_values=past_key_values,
        next_logits=next_logits,
        decode_step=0,
    )


@torch.inference_mode()
def run_baseline_greedy_decode(t3, request: ScheduledDecodeRequest) -> Tensor:
    state = prefill_prototype_state(t3, request)
    predicted_tokens: list[Tensor] = []

    for _ in range(request.max_new_tokens):
        next_token = state.next_logits.argmax(dim=-1, keepdim=True)
        predicted_tokens.append(next_token)
        state.generated_ids = torch.cat([state.generated_ids, next_token], dim=1)

        if torch.all(next_token.view(-1) == t3.hp.stop_speech_token):
            break

        step_inputs = _build_cfg_step_inputs(
            t3,
            tokens=next_token,
            start_pos=state.decode_step + 1,
        )
        raw_logits, next_past = _forward_raw_t3(
            t3,
            inputs_embeds=step_inputs,
            past_key_values=state.past_key_values,
        )
        state.next_logits = _cfg_combine(raw_logits[:, -1, :], request.cfg_weight)
        state.past_key_values = next_past
        state.decode_step += 1

    if predicted_tokens:
        return torch.cat(predicted_tokens, dim=1)
    return torch.empty((1, 0), dtype=torch.long, device=t3.device)


@torch.inference_mode()
def draft_propose_block_self(
    t3,
    *,
    state: _PrototypeState,
    speculate_k: int,
) -> Tensor:
    assert speculate_k >= 1
    work_logits = state.next_logits
    work_past = state.past_key_values
    proposed: list[Tensor] = []

    if _trace_enabled():
        shape_logger.info("[models/t3/inference/speculative_decode.py] draft.start")
        shape_logger.info("  decode_step %s", state.decode_step)
        shape_logger.info("  speculate_k %s", speculate_k)
        shape_logger.info("  current_logits %s %s %s", tuple(work_logits.shape), work_logits.dtype, work_logits.device)
        shape_logger.info("  kv_seq_len %s", _kv_seq_len(work_past))

    for block_offset in range(speculate_k):
        next_token = work_logits.argmax(dim=-1, keepdim=True)
        proposed.append(next_token)

        if torch.all(next_token.view(-1) == t3.hp.stop_speech_token):
            break

        step_inputs = _build_cfg_step_inputs(
            t3,
            tokens=next_token,
            start_pos=state.decode_step + block_offset + 1,
        )
        raw_logits, work_past = _forward_raw_t3(
            t3,
            inputs_embeds=step_inputs,
            past_key_values=work_past,
        )
        work_logits = _cfg_combine(raw_logits[:, -1, :], state.request.cfg_weight)

    proposed_tokens = torch.cat(proposed, dim=1)
    if _trace_enabled():
        shape_logger.info("[models/t3/inference/speculative_decode.py] draft.proposed")
        shape_logger.info("  proposed_tokens %s %s %s", tuple(proposed_tokens.shape), proposed_tokens.dtype, proposed_tokens.device)
        shape_logger.info("  proposed_token_ids %s", proposed_tokens.view(-1).tolist())
    return proposed_tokens


@torch.inference_mode()
def verify_block_greedy(
    t3,
    *,
    state: _PrototypeState,
    proposed_tokens: Tensor,
) -> VerifyResult:
    assert proposed_tokens.dim() == 2 and proposed_tokens.size(0) == 1
    proposed_count = int(proposed_tokens.size(1))
    assert proposed_count >= 1

    block_inputs = _build_cfg_step_inputs(
        t3,
        tokens=proposed_tokens,
        start_pos=state.decode_step + 1,
    )
    raw_block_logits, block_past = _forward_raw_t3(
        t3,
        inputs_embeds=block_inputs,
        past_key_values=state.past_key_values,
    )
    cfg_block_logits = _cfg_combine(raw_block_logits, state.request.cfg_weight)

    assert cfg_block_logits.shape == (1, proposed_count, cfg_block_logits.size(-1))
    assert _kv_seq_len(block_past) == _kv_seq_len(state.past_key_values) + proposed_count

    verify_logits = [state.next_logits] + [cfg_block_logits[:, index, :] for index in range(proposed_count - 1)]
    match_len = 0
    for index in range(proposed_count):
        predicted_token = verify_logits[index].argmax(dim=-1, keepdim=True)
        if torch.equal(predicted_token, proposed_tokens[:, index : index + 1]):
            match_len += 1
            continue
        break

    correction_token = None
    if match_len == proposed_count:
        committed_tokens = proposed_tokens
        next_logits = cfg_block_logits[:, -1, :]
        next_past_key_values = block_past
    else:
        correction_token = verify_logits[match_len].argmax(dim=-1, keepdim=True)
        committed_tokens = torch.cat([proposed_tokens[:, :match_len], correction_token], dim=1)
        replay_inputs = _build_cfg_step_inputs(
            t3,
            tokens=committed_tokens,
            start_pos=state.decode_step + 1,
        )
        raw_replay_logits, next_past_key_values = _forward_raw_t3(
            t3,
            inputs_embeds=replay_inputs,
            past_key_values=state.past_key_values,
        )
        replay_cfg_logits = _cfg_combine(raw_replay_logits, state.request.cfg_weight)
        next_logits = replay_cfg_logits[:, -1, :]
        assert _kv_seq_len(next_past_key_values) == _kv_seq_len(state.past_key_values) + committed_tokens.size(1)

    if _trace_enabled():
        shape_logger.info("[models/t3/inference/speculative_decode.py] verify")
        shape_logger.info("  decode_step %s", state.decode_step)
        shape_logger.info("  block_inputs %s %s %s", tuple(block_inputs.shape), block_inputs.dtype, block_inputs.device)
        shape_logger.info("  raw_block_logits %s %s %s", tuple(raw_block_logits.shape), raw_block_logits.dtype, raw_block_logits.device)
        shape_logger.info("  cfg_block_logits %s %s %s", tuple(cfg_block_logits.shape), cfg_block_logits.dtype, cfg_block_logits.device)
        shape_logger.info("  proposed_token_ids %s", proposed_tokens.view(-1).tolist())
        shape_logger.info("  match_len %s", match_len)
        shape_logger.info("  committed_token_ids %s", committed_tokens.view(-1).tolist())
        shape_logger.info("  next_kv_seq_len %s", _kv_seq_len(next_past_key_values))
        if correction_token is not None:
            shape_logger.info("  correction_token_id %s", correction_token.view(-1).tolist())

    return VerifyResult(
        committed_tokens=committed_tokens,
        accepted_draft_tokens=match_len,
        proposed_tokens=proposed_count,
        correction_token=correction_token,
        next_logits=next_logits,
        next_past_key_values=next_past_key_values,
    )


@torch.inference_mode()
def run_self_speculative_decode(
    t3,
    request: ScheduledDecodeRequest,
    *,
    speculate_k: int,
) -> SpeculativePrototypeResult:
    assert speculate_k >= 1
    state = prefill_prototype_state(t3, request)
    predicted_tokens: list[Tensor] = []
    rounds = 0
    proposed_tokens_total = 0
    accepted_draft_tokens_total = 0
    correction_tokens_total = 0

    while sum(token.size(1) for token in predicted_tokens) < request.max_new_tokens:
        remaining = request.max_new_tokens - sum(token.size(1) for token in predicted_tokens)
        proposed_tokens = draft_propose_block_self(
            t3,
            state=state,
            speculate_k=min(speculate_k, remaining),
        )
        verify = verify_block_greedy(
            t3,
            state=state,
            proposed_tokens=proposed_tokens,
        )

        committed_tokens = verify.committed_tokens
        predicted_tokens.append(committed_tokens)
        state.generated_ids = torch.cat([state.generated_ids, committed_tokens], dim=1)
        state.past_key_values = verify.next_past_key_values
        state.next_logits = verify.next_logits
        state.decode_step += committed_tokens.size(1)

        rounds += 1
        proposed_tokens_total += verify.proposed_tokens
        accepted_draft_tokens_total += verify.accepted_draft_tokens
        if verify.correction_token is not None:
            correction_tokens_total += 1

        if torch.any(committed_tokens == t3.hp.stop_speech_token):
            break

    if predicted_tokens:
        speech_tokens = torch.cat(predicted_tokens, dim=1)
    else:
        speech_tokens = torch.empty((1, 0), dtype=torch.long, device=t3.device)

    return SpeculativePrototypeResult(
        speech_tokens=speech_tokens,
        rounds=rounds,
        proposed_tokens_total=proposed_tokens_total,
        accepted_draft_tokens_total=accepted_draft_tokens_total,
        correction_tokens_total=correction_tokens_total,
    )
