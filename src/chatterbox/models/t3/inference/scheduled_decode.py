import logging
import os
from dataclasses import dataclass, field

import torch
from torch import Tensor
from transformers.generation.logits_process import (
    MinPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    TopPLogitsWarper,
)

from .alignment_stream_analyzer_scheduled import (
    ScheduledAlignmentController,
    ScheduledAlignmentState,
)
from .t3_hf_backend import T3HuggingfaceBackend


shape_logger = logging.getLogger("chatterbox.shape")
logger = logging.getLogger(__name__)


@dataclass
class ScheduledDecodeRequest:
    session_id: str
    t3_cond: object
    text_tokens: Tensor
    max_new_tokens: int
    temperature: float
    top_p: float
    min_p: float
    repetition_penalty: float
    cfg_weight: float

    def batch_key(self) -> tuple[int, int]:
        prompt = getattr(self.t3_cond, "cond_prompt_speech_tokens", None)
        prompt_len = 0 if prompt is None else int(prompt.shape[-1])
        return (int(self.text_tokens.shape[-1]), prompt_len)


@dataclass
class _ActiveDecodeState:
    request: ScheduledDecodeRequest
    generated_ids: Tensor
    predicted_tokens: list[Tensor] = field(default_factory=list)
    alignment_state: ScheduledAlignmentState | None = None
    past_key_values: object = None
    decode_step: int = 0
    next_inputs_embeds: Tensor | None = None
    next_logits: Tensor | None = None
    next_hidden: Tensor | None = None
    rounds: int = 0
    proposed_tokens_total: int = 0
    accepted_draft_tokens_total: int = 0
    correction_tokens_total: int = 0
    full_accept_rounds: int = 0
    zero_accept_rounds: int = 0
    partial_accept_rounds: int = 0


@dataclass
class ScheduledDecodeCohort:
    batch_key: tuple[int, int]
    active_states: list[_ActiveDecodeState]
    prefill_inputs: list[Tensor] | None


@dataclass
class ScheduledFinishedResult:
    session_id: str
    speech_tokens: Tensor
    decode_metrics: dict[str, float]


@dataclass
class ScheduledAdvanceResult:
    finished_results: list[ScheduledFinishedResult]
    first_token_session_ids: list[str]
    successor_cohorts: list[ScheduledDecodeCohort]


def _ensure_bot_eot(text_tokens: Tensor, hp):
    batch = text_tokens.size(0)
    assert (text_tokens == hp.start_text_token).int().sum() >= batch, "missing start_text_token"
    assert (text_tokens == hp.stop_text_token).int().sum() >= batch, "missing stop_text_token"


def _cat_past_key_values(past_per_state: list[object]):
    if not past_per_state:
        return None
    num_layers = len(past_per_state[0])
    merged = []
    for layer_index in range(num_layers):
        merged.append(
            tuple(
                torch.cat(
                    [state_past[layer_index][tensor_index] for state_past in past_per_state],
                    dim=0,
                )
                for tensor_index in range(len(past_per_state[0][layer_index]))
            )
        )
    return tuple(merged)


def _split_past_key_values(past_key_values, chunk_sizes: list[int]):
    if past_key_values is None:
        return [None] * len(chunk_sizes)

    split = [list() for _ in chunk_sizes]
    for layer in past_key_values:
        chunks_per_tensor = [tensor.split(chunk_sizes, dim=0) for tensor in layer]
        for index in range(len(chunk_sizes)):
            split[index].append(tuple(chunks[index] for chunks in chunks_per_tensor))
    return [tuple(layer_list) for layer_list in split]


def _log_backend_output_shapes(tag: str, output):
    if not os.getenv("CHATTERBOX_TRACE_SHAPES"):
        return

    shape_logger.info("[models/t3/inference/scheduled_decode.py] %s", tag)
    shape_logger.info(
        "  logits %s %s %s",
        tuple(output.logits.shape),
        output.logits.dtype,
        output.logits.device,
    )
    first_layer = output.past_key_values[0]
    shape_logger.info(
        "  past_key_values[0][0] %s %s %s",
        tuple(first_layer[0].shape),
        first_layer[0].dtype,
        first_layer[0].device,
    )
    shape_logger.info(
        "  past_key_values[0][1] %s %s %s",
        tuple(first_layer[1].shape),
        first_layer[1].dtype,
        first_layer[1].device,
    )


def _cfg_combine_rows(raw_logits: Tensor, cfg_weights: Tensor) -> Tensor:
    cond = raw_logits[0::2]
    uncond = raw_logits[1::2]
    view_shape = [cfg_weights.size(0)] + [1] * (cond.dim() - 1)
    weights = cfg_weights.to(device=cond.device, dtype=cond.dtype).view(*view_shape)
    return cond + weights * (cond - uncond)


def _kv_seq_len(past_key_values) -> int:
    return int(past_key_values[0][0].shape[2])


def _forward_raw_t3_batched(
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
    hidden_states = tfmr_out.last_hidden_state
    logits = t3.speech_head(hidden_states)
    return logits, tfmr_out.past_key_values, hidden_states


def _build_cfg_step_inputs_batch(t3, *, tokens: Tensor, start_pos: int) -> Tensor:
    assert tokens.dim() == 2, f"expected (B, K) tokens, got {tuple(tokens.shape)}"
    base_embed = t3.speech_emb(tokens)
    pos_embeds = [
        t3.speech_pos_emb.get_fixed_embedding(start_pos + offset).to(device=base_embed.device)
        for offset in range(tokens.size(1))
    ]
    pos_block = torch.cat(pos_embeds, dim=1)
    single = base_embed + pos_block
    return torch.cat([single, single], dim=0)


def _build_successor_cohorts(
    cohort: ScheduledDecodeCohort,
    next_round_states: list[_ActiveDecodeState],
) -> list[ScheduledDecodeCohort]:
    if not next_round_states:
        return []

    grouped: dict[int, list[_ActiveDecodeState]] = {}
    for state in next_round_states:
        grouped.setdefault(state.decode_step, []).append(state)

    return [
        ScheduledDecodeCohort(
            batch_key=cohort.batch_key,
            active_states=states,
            prefill_inputs=None,
        )
        for _, states in sorted(grouped.items(), key=lambda item: item[0])
    ]


def _speculative_metrics_from_state(state: _ActiveDecodeState) -> dict[str, float]:
    if state.proposed_tokens_total <= 0:
        return {}
    return {
        "t3_rounds": float(state.rounds),
        "t3_proposed_tokens_total": float(state.proposed_tokens_total),
        "t3_accepted_draft_tokens_total": float(state.accepted_draft_tokens_total),
        "t3_correction_tokens_total": float(state.correction_tokens_total),
        "t3_acceptance_rate": float(state.accepted_draft_tokens_total) / float(state.proposed_tokens_total),
        "t3_full_accept_rounds": float(state.full_accept_rounds),
        "t3_zero_accept_rounds": float(state.zero_accept_rounds),
        "t3_partial_accept_rounds": float(state.partial_accept_rounds),
    }


def _build_initial_state(t3, request: ScheduledDecodeRequest) -> tuple[_ActiveDecodeState, Tensor]:
    _ensure_bot_eot(request.text_tokens, t3.hp)
    text_tokens = torch.atleast_2d(request.text_tokens).to(dtype=torch.long, device=t3.device)
    initial_speech_tokens = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])

    embeds, len_cond = t3.prepare_input_embeds(
        t3_cond=request.t3_cond,
        text_tokens=text_tokens,
        speech_tokens=initial_speech_tokens,
        cfg_weight=request.cfg_weight,
    )
    if os.getenv("CHATTERBOX_TRACE_SHAPES"):
        shape_logger.info("[models/t3/inference/scheduled_decode.py] prefill.input")
        shape_logger.info("  session_id %s", request.session_id)
        shape_logger.info("  text_tokens %s %s %s", tuple(text_tokens.shape), text_tokens.dtype, text_tokens.device)
        shape_logger.info("  embeds %s %s %s", tuple(embeds.shape), embeds.dtype, embeds.device)
        shape_logger.info("  len_cond %s", len_cond)

    device = embeds.device
    bos_token = torch.tensor([[t3.hp.start_speech_token]], dtype=torch.long, device=device)
    bos_pos_embed = t3.speech_pos_emb.get_fixed_embedding(0)
    bos_embed = t3.speech_emb(bos_token)
    bos_embed = bos_embed + bos_pos_embed
    bos_embed = torch.cat([bos_embed, bos_embed], dim=0)
    if os.getenv("CHATTERBOX_TRACE_SHAPES"):
        shape_logger.info("[models/t3/inference/scheduled_decode.py] prefill.bos")
        shape_logger.info("  session_id %s", request.session_id)
        shape_logger.info(
            "  bos_pos_embed %s %s %s",
            tuple(bos_pos_embed.shape),
            bos_pos_embed.dtype,
            bos_pos_embed.device,
        )
        shape_logger.info(
            "  bos_embed %s %s %s",
            tuple(bos_embed.shape),
            bos_embed.dtype,
            bos_embed.device,
        )

    alignment_state = None
    if t3.hp.is_multilingual:
        alignment_state = ScheduledAlignmentState.create(
            text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
            eos_idx=t3.hp.stop_speech_token,
            device=embeds.device,
        )

    return (
        _ActiveDecodeState(
            request=request,
            generated_ids=bos_token.clone(),
            alignment_state=alignment_state,
        ),
        torch.cat([embeds, bos_embed], dim=1),
    )


def build_scheduled_runtime_components(t3, *, enable_alignment_controller: bool = False):
    patched_model = T3HuggingfaceBackend(
        config=t3.cfg,
        llama=t3.tfmr,
        speech_enc=t3.speech_emb,
        speech_head=t3.speech_head,
        alignment_stream_analyzer=None,
    )
    alignment_controller = None
    if enable_alignment_controller and t3.hp.is_multilingual:
        alignment_controller = ScheduledAlignmentController(t3.tfmr)
    return patched_model, alignment_controller


def prepare_scheduled_cohort(t3, requests: list[ScheduledDecodeRequest]) -> ScheduledDecodeCohort:
    if not requests:
        raise ValueError("scheduled cohort requires at least one request")

    batch_keys = {request.batch_key() for request in requests}
    if len(batch_keys) != 1:
        raise ValueError("scheduled cohort currently requires matching text/prompt lengths")

    active_states: list[_ActiveDecodeState] = []
    prefill_inputs = []
    for request in requests:
        state, inputs_embeds = _build_initial_state(t3, request)
        active_states.append(state)
        prefill_inputs.append(inputs_embeds)

    return ScheduledDecodeCohort(
        batch_key=requests[0].batch_key(),
        active_states=active_states,
        prefill_inputs=prefill_inputs,
    )


def _finalize_prediction(t3, state: _ActiveDecodeState) -> Tensor:
    if state.predicted_tokens:
        predicted_tokens = torch.cat(state.predicted_tokens, dim=1)
    else:
        predicted_tokens = torch.empty((1, 0), dtype=torch.long, device=t3.device)

    if os.getenv("CHATTERBOX_TRACE_SHAPES"):
        shape_logger.info("[models/t3/inference/scheduled_decode.py] inference.output")
        shape_logger.info("  session_id %s", state.request.session_id)
        shape_logger.info(
            "  predicted_tokens %s %s %s",
            tuple(predicted_tokens.shape),
            predicted_tokens.dtype,
            predicted_tokens.device,
        )
    return predicted_tokens


@torch.inference_mode()
def _hydrate_prefill_state(
    t3,
    cohort: ScheduledDecodeCohort,
    *,
    patched_model,
    output_attentions: bool,
):
    inputs_embeds = torch.cat(cohort.prefill_inputs, dim=0)
    if os.getenv("CHATTERBOX_TRACE_SHAPES"):
        shape_logger.info("[models/t3/inference/scheduled_decode.py] prefill.batch")
        shape_logger.info("  requests %s", len(cohort.active_states))
        shape_logger.info("  inputs_embeds %s %s %s", tuple(inputs_embeds.shape), inputs_embeds.dtype, inputs_embeds.device)
    output = patched_model(
        inputs_embeds=inputs_embeds,
        past_key_values=None,
        use_cache=True,
        output_attentions=output_attentions,
        output_hidden_states=True,
        return_dict=True,
    )
    _log_backend_output_shapes("prefill.output", output)

    past_splits = _split_past_key_values(output.past_key_values, [2] * len(cohort.active_states))
    hidden_states = output.hidden_states[-1][:, -1:, :]
    cfg_weights = torch.tensor(
        [state.request.cfg_weight for state in cohort.active_states],
        device=output.logits.device,
        dtype=output.logits.dtype,
    )
    next_logits = _cfg_combine_rows(output.logits[:, -1, :], cfg_weights)

    for row_index, (state, state_past) in enumerate(zip(cohort.active_states, past_splits)):
        state.past_key_values = state_past
        state.next_logits = next_logits[row_index : row_index + 1]
        state.next_hidden = hidden_states[row_index * 2 : (row_index + 1) * 2]

    cohort.prefill_inputs = None


@torch.inference_mode()
def _advance_scheduled_cohort_greedy(
    t3,
    cohort: ScheduledDecodeCohort,
    *,
    patched_model,
    alignment_controller,
) -> ScheduledAdvanceResult:
    if not cohort.active_states:
        return ScheduledAdvanceResult([], [], [])

    output_attentions = alignment_controller is not None

    if cohort.prefill_inputs is not None:
        _hydrate_prefill_state(
            t3,
            cohort,
            patched_model=patched_model,
            output_attentions=output_attentions,
        )
        logits = torch.cat([state.next_logits for state in cohort.active_states], dim=0)
    else:
        is_first_cached_step = all(state.decode_step == 1 for state in cohort.active_states)
        next_inputs = [state.next_inputs_embeds for state in cohort.active_states]
        batched_past = _cat_past_key_values([state.past_key_values for state in cohort.active_states])
        batched_inputs = torch.cat(next_inputs, dim=0)
        if os.getenv("CHATTERBOX_TRACE_SHAPES"):
            shape_logger.info("[models/t3/inference/scheduled_decode.py] decode.batch")
            shape_logger.info("  requests %s", len(cohort.active_states))
            shape_logger.info("  inputs_embeds %s %s %s", tuple(batched_inputs.shape), batched_inputs.dtype, batched_inputs.device)
        output = patched_model(
            inputs_embeds=batched_inputs,
            past_key_values=batched_past,
            use_cache=True,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        if is_first_cached_step:
            _log_backend_output_shapes("decode.output.first_cached_step", output)
        past_splits = _split_past_key_values(output.past_key_values, [2] * len(cohort.active_states))
        for state, state_past in zip(cohort.active_states, past_splits):
            state.past_key_values = state_past
        cfg_weights = torch.tensor(
            [state.request.cfg_weight for state in cohort.active_states],
            device=output.logits.device,
            dtype=output.logits.dtype,
        )
        logits = _cfg_combine_rows(output.logits[:, -1, :], cfg_weights)

    if alignment_controller is not None:
        last_tokens = [
            state.generated_ids[0, -1].item() if state.generated_ids.size(1) > 0 else None
            for state in cohort.active_states
        ]
        logits = alignment_controller.step(
            logits,
            active_states=[state.alignment_state for state in cohort.active_states],
            next_tokens=last_tokens,
        )

    finished_results: list[ScheduledFinishedResult] = []
    first_token_session_ids: list[str] = []
    next_round_states: list[_ActiveDecodeState] = []
    for row_index, state in enumerate(cohort.active_states):
        request = state.request
        ids_for_proc = state.generated_ids
        row_logits = logits[row_index : row_index + 1]

        row_logits = RepetitionPenaltyLogitsProcessor(
            penalty=float(request.repetition_penalty)
        )(ids_for_proc, row_logits)

        use_greedy = float(request.temperature) <= 0.0
        if not use_greedy and request.temperature != 1.0:
            row_logits = row_logits / request.temperature

        if use_greedy:
            next_token = row_logits.argmax(dim=-1, keepdim=True)
        else:
            row_logits = MinPLogitsWarper(min_p=request.min_p)(ids_for_proc, row_logits)
            row_logits = TopPLogitsWarper(top_p=request.top_p)(ids_for_proc, row_logits)
            probs = torch.softmax(row_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        state.predicted_tokens.append(next_token)
        state.generated_ids = torch.cat([state.generated_ids, next_token], dim=1)
        if len(state.predicted_tokens) == 1:
            first_token_session_ids.append(request.session_id)

        stop_on_eos = torch.all(next_token.view(-1) == t3.hp.stop_speech_token)
        hit_limit = len(state.predicted_tokens) >= request.max_new_tokens
        if stop_on_eos or hit_limit:
            if stop_on_eos:
                logger.info("✅ EOS token detected for %s", request.session_id)
            finished_results.append(
                ScheduledFinishedResult(
                    session_id=request.session_id,
                    speech_tokens=_finalize_prediction(t3, state),
                    decode_metrics={},
                )
            )
            continue

        next_token_embed = t3.speech_emb(next_token)
        next_token_pos_embed = t3.speech_pos_emb.get_fixed_embedding(state.decode_step + 1)
        if os.getenv("CHATTERBOX_TRACE_SHAPES") and state.decode_step == 0:
            shape_logger.info("[models/t3/inference/scheduled_decode.py] decode.next_token_embed")
            shape_logger.info("  session_id %s", request.session_id)
            shape_logger.info("  decode_step %s", state.decode_step)
            shape_logger.info(
                "  next_token_pos_embed %s %s %s",
                tuple(next_token_pos_embed.shape),
                next_token_pos_embed.dtype,
                next_token_pos_embed.device,
            )
            shape_logger.info(
                "  next_token_embed_base %s %s %s",
                tuple(next_token_embed.shape),
                next_token_embed.dtype,
                next_token_embed.device,
            )
        next_token_embed = next_token_embed + next_token_pos_embed
        state.next_inputs_embeds = torch.cat([next_token_embed, next_token_embed], dim=0)
        state.decode_step += 1
        next_round_states.append(state)

    return ScheduledAdvanceResult(
        finished_results=finished_results,
        first_token_session_ids=first_token_session_ids,
        successor_cohorts=_build_successor_cohorts(cohort, next_round_states),
    )


@dataclass
class _ScheduledVerifyRow:
    state: _ActiveDecodeState
    committed_tokens: Tensor
    accepted_draft_tokens: int
    proposed_tokens: int
    correction_token: Tensor | None
    next_logits: Tensor | None = None
    next_hidden: Tensor | None = None
    next_past_key_values: object = None


@torch.inference_mode()
def _hydra_propose_tokens(
    t3,
    hydra_model,
    *,
    state: _ActiveDecodeState,
    speculate_k: int,
) -> Tensor:
    assert state.next_logits is not None
    assert state.next_hidden is not None

    proposal_cap = min(speculate_k, hydra_model.hydra_num_heads + 1)
    proposed: list[Tensor] = []

    next_token = state.next_logits.argmax(dim=-1, keepdim=True)
    proposed.append(next_token)
    if torch.all(next_token.view(-1) == t3.hp.stop_speech_token) or proposal_cap == 1:
        return torch.cat(proposed, dim=1)

    base_hidden = state.next_hidden[0:1]
    for head_index in range(min(hydra_model.hydra_num_heads, proposal_cap - 1)):
        context_tokens = torch.cat(proposed, dim=1)
        context_embeds = t3.speech_emb(context_tokens)
        head_inputs = [base_hidden]
        for token_offset in range(head_index + 1):
            head_inputs.append(context_embeds[:, token_offset : token_offset + 1, :])
        head_input = torch.cat(head_inputs, dim=-1)
        hidden = hydra_model.hydra_heads[head_index](head_input)
        raw_logits = hydra_model.hydra_lm_heads[head_index](hidden)
        future_token = raw_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        proposed.append(future_token)
        if torch.all(future_token.view(-1) == t3.hp.stop_speech_token):
            break

    return torch.cat(proposed, dim=1)


@torch.inference_mode()
def _verify_block_greedy_batched(
    t3,
    states: list[_ActiveDecodeState],
    proposed_tokens: Tensor,
    *,
    patched_model,
) -> list[_ScheduledVerifyRow]:
    assert states
    assert proposed_tokens.dim() == 2
    assert proposed_tokens.size(0) == len(states)
    proposed_count = int(proposed_tokens.size(1))
    assert proposed_count >= 1

    block_inputs = _build_cfg_step_inputs_batch(
        t3,
        tokens=proposed_tokens,
        start_pos=states[0].decode_step + 1,
    )
    raw_block_logits, block_past, block_hidden = _forward_raw_t3_batched(
        t3,
        inputs_embeds=block_inputs,
        past_key_values=_cat_past_key_values([state.past_key_values for state in states]),
    )
    block_past_splits = _split_past_key_values(block_past, [2] * len(states))
    cfg_weights = torch.tensor(
        [state.request.cfg_weight for state in states],
        device=raw_block_logits.device,
        dtype=raw_block_logits.dtype,
    )
    cfg_block_logits = _cfg_combine_rows(raw_block_logits, cfg_weights)
    if os.getenv("CHATTERBOX_TRACE_SHAPES"):
        shape_logger.info("[models/t3/inference/scheduled_decode.py] hydra.verify.batch")
        shape_logger.info("  requests %s", len(states))
        shape_logger.info("  proposed_tokens %s %s %s", tuple(proposed_tokens.shape), proposed_tokens.dtype, proposed_tokens.device)
        shape_logger.info("  block_inputs %s %s %s", tuple(block_inputs.shape), block_inputs.dtype, block_inputs.device)
        shape_logger.info("  raw_block_logits %s %s %s", tuple(raw_block_logits.shape), raw_block_logits.dtype, raw_block_logits.device)
        shape_logger.info("  cfg_block_logits %s %s %s", tuple(cfg_block_logits.shape), cfg_block_logits.dtype, cfg_block_logits.device)
        shape_logger.info("  block_kv_seq_len %s", _kv_seq_len(block_past))

    expected_block_len = _kv_seq_len(states[0].past_key_values) + proposed_count
    if _kv_seq_len(block_past) != expected_block_len:
        raise RuntimeError(
            f"Hydra scheduled verify KV length mismatch: got {_kv_seq_len(block_past)}, "
            f"expected {expected_block_len}"
        )

    verify_rows: list[_ScheduledVerifyRow] = []
    replay_groups: dict[int, list[int]] = {}
    for row_index, state in enumerate(states):
        verify_logits = [state.next_logits] + [
            cfg_block_logits[row_index : row_index + 1, index, :]
            for index in range(proposed_count - 1)
        ]
        match_len = 0
        for index in range(proposed_count):
            predicted_token = verify_logits[index].argmax(dim=-1, keepdim=True)
            if torch.equal(predicted_token, proposed_tokens[row_index : row_index + 1, index : index + 1]):
                match_len += 1
                continue
            break

        if match_len == proposed_count:
            verify_rows.append(
                _ScheduledVerifyRow(
                    state=state,
                    committed_tokens=proposed_tokens[row_index : row_index + 1],
                    accepted_draft_tokens=match_len,
                    proposed_tokens=proposed_count,
                    correction_token=None,
                    next_logits=cfg_block_logits[row_index : row_index + 1, -1, :],
                    next_hidden=block_hidden[row_index * 2 : (row_index + 1) * 2, -1:, :],
                    next_past_key_values=block_past_splits[row_index],
                )
            )
            continue

        correction_token = verify_logits[match_len].argmax(dim=-1, keepdim=True)
        committed_tokens = torch.cat(
            [proposed_tokens[row_index : row_index + 1, :match_len], correction_token],
            dim=1,
        )
        verify_rows.append(
            _ScheduledVerifyRow(
                state=state,
                committed_tokens=committed_tokens,
                accepted_draft_tokens=match_len,
                proposed_tokens=proposed_count,
                correction_token=correction_token,
            )
        )
        replay_groups.setdefault(committed_tokens.size(1), []).append(len(verify_rows) - 1)

    for committed_len, verify_indices in replay_groups.items():
        group_rows = [verify_rows[index] for index in verify_indices]
        group_states = [row.state for row in group_rows]
        committed_batch = torch.cat([row.committed_tokens for row in group_rows], dim=0)
        replay_inputs = _build_cfg_step_inputs_batch(
            t3,
            tokens=committed_batch,
            start_pos=group_states[0].decode_step + 1,
        )
        raw_replay_logits, replay_past, replay_hidden = _forward_raw_t3_batched(
            t3,
            inputs_embeds=replay_inputs,
            past_key_values=_cat_past_key_values([state.past_key_values for state in group_states]),
        )
        replay_past_splits = _split_past_key_values(replay_past, [2] * len(group_states))
        replay_cfg_weights = torch.tensor(
            [state.request.cfg_weight for state in group_states],
            device=raw_replay_logits.device,
            dtype=raw_replay_logits.dtype,
        )
        replay_cfg_logits = _cfg_combine_rows(raw_replay_logits, replay_cfg_weights)

        expected_replay_len = _kv_seq_len(group_states[0].past_key_values) + committed_len
        if _kv_seq_len(replay_past) != expected_replay_len:
            raise RuntimeError(
                f"Hydra scheduled replay KV length mismatch: got {_kv_seq_len(replay_past)}, "
                f"expected {expected_replay_len}"
            )

        for local_index, verify_row in enumerate(group_rows):
            verify_row.next_logits = replay_cfg_logits[local_index : local_index + 1, -1, :]
            verify_row.next_hidden = replay_hidden[local_index * 2 : (local_index + 1) * 2, -1:, :]
            verify_row.next_past_key_values = replay_past_splits[local_index]

    return verify_rows


@torch.inference_mode()
def _advance_scheduled_cohort_hydra(
    t3,
    cohort: ScheduledDecodeCohort,
    *,
    patched_model,
    hydra_model,
    hydra_speculate_k: int,
) -> ScheduledAdvanceResult:
    if not cohort.active_states:
        return ScheduledAdvanceResult([], [], [])

    if cohort.prefill_inputs is not None:
        _hydrate_prefill_state(
            t3,
            cohort,
            patched_model=patched_model,
            output_attentions=False,
        )

    proposal_groups: dict[int, list[tuple[_ActiveDecodeState, Tensor]]] = {}
    for state in cohort.active_states:
        remaining = state.request.max_new_tokens - sum(token.size(1) for token in state.predicted_tokens)
        if remaining <= 0:
            proposed = torch.empty((1, 0), dtype=torch.long, device=t3.device)
        else:
            proposed = _hydra_propose_tokens(
                t3,
                hydra_model,
                state=state,
                speculate_k=min(hydra_speculate_k, remaining),
            )
        proposal_groups.setdefault(int(proposed.size(1)), []).append((state, proposed))

    finished_results: list[ScheduledFinishedResult] = []
    first_token_session_ids: list[str] = []
    next_round_states: list[_ActiveDecodeState] = []

    for proposed_count, grouped in sorted(proposal_groups.items(), key=lambda item: item[0]):
        if os.getenv("CHATTERBOX_TRACE_SHAPES"):
            shape_logger.info("[models/t3/inference/scheduled_decode.py] hydra.proposal_group")
            shape_logger.info("  proposal_count %s", proposed_count)
            shape_logger.info("  requests %s", len(grouped))
        if proposed_count <= 0:
            for state, _ in grouped:
                finished_results.append(
                    ScheduledFinishedResult(
                        session_id=state.request.session_id,
                        speech_tokens=_finalize_prediction(t3, state),
                        decode_metrics=_speculative_metrics_from_state(state),
                    )
                )
            continue

        group_states = [item[0] for item in grouped]
        proposed_batch = torch.cat([item[1] for item in grouped], dim=0)
        verify_rows = _verify_block_greedy_batched(
            t3,
            group_states,
            proposed_batch,
            patched_model=patched_model,
        )

        for verify_row in verify_rows:
            state = verify_row.state
            had_no_tokens = not state.predicted_tokens
            committed_tokens = verify_row.committed_tokens
            state.predicted_tokens.append(committed_tokens)
            state.generated_ids = torch.cat([state.generated_ids, committed_tokens], dim=1)
            state.past_key_values = verify_row.next_past_key_values
            state.next_logits = verify_row.next_logits
            state.next_hidden = verify_row.next_hidden
            state.decode_step += committed_tokens.size(1)

            state.rounds += 1
            state.proposed_tokens_total += verify_row.proposed_tokens
            state.accepted_draft_tokens_total += verify_row.accepted_draft_tokens
            if verify_row.correction_token is None and verify_row.accepted_draft_tokens == verify_row.proposed_tokens:
                state.full_accept_rounds += 1
            elif verify_row.accepted_draft_tokens == 0:
                state.zero_accept_rounds += 1
            else:
                state.partial_accept_rounds += 1
            if verify_row.correction_token is not None:
                state.correction_tokens_total += 1

            if had_no_tokens and committed_tokens.numel() > 0:
                first_token_session_ids.append(state.request.session_id)

            stop_on_eos = torch.any(committed_tokens == t3.hp.stop_speech_token)
            hit_limit = sum(token.size(1) for token in state.predicted_tokens) >= state.request.max_new_tokens
            if stop_on_eos or hit_limit:
                finished_results.append(
                    ScheduledFinishedResult(
                        session_id=state.request.session_id,
                        speech_tokens=_finalize_prediction(t3, state),
                        decode_metrics=_speculative_metrics_from_state(state),
                    )
                )
                continue

            next_round_states.append(state)

    return ScheduledAdvanceResult(
        finished_results=finished_results,
        first_token_session_ids=first_token_session_ids,
        successor_cohorts=_build_successor_cohorts(cohort, next_round_states),
    )


@torch.inference_mode()
def advance_scheduled_cohort(
    t3,
    cohort: ScheduledDecodeCohort,
    *,
    patched_model,
    alignment_controller,
    hydra_model=None,
    hydra_speculate_k: int = 3,
) -> ScheduledAdvanceResult:
    if hydra_model is not None:
        if alignment_controller is not None:
            raise ValueError("Hydra scheduled decode does not support the alignment controller")
        return _advance_scheduled_cohort_hydra(
            t3,
            cohort,
            patched_model=patched_model,
            hydra_model=hydra_model,
            hydra_speculate_k=hydra_speculate_k,
        )
    return _advance_scheduled_cohort_greedy(
        t3,
        cohort,
        patched_model=patched_model,
        alignment_controller=alignment_controller,
    )
