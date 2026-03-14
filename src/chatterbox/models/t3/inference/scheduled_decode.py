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


@dataclass
class ScheduledDecodeCohort:
    batch_key: tuple[int, int]
    active_states: list[_ActiveDecodeState]
    prefill_inputs: list[Tensor] | None


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


def build_scheduled_runtime_components(t3):
    patched_model = T3HuggingfaceBackend(
        config=t3.cfg,
        llama=t3.tfmr,
        speech_enc=t3.speech_emb,
        speech_head=t3.speech_head,
        alignment_stream_analyzer=None,
    )
    alignment_controller = ScheduledAlignmentController(t3.tfmr) if t3.hp.is_multilingual else None
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
def advance_scheduled_cohort(
    t3,
    cohort: ScheduledDecodeCohort,
    *,
    patched_model,
    alignment_controller,
) -> tuple[list[tuple[str, Tensor]], list[str], bool]:
    if not cohort.active_states:
        return [], [], True

    output_attentions = alignment_controller is not None

    if cohort.prefill_inputs is not None:
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
        for state, state_past in zip(cohort.active_states, past_splits):
            state.past_key_values = state_past
        cohort.prefill_inputs = None
    else:
        # `decode_step` is incremented at the end of the prefill round after we
        # sample the first token and prepare the first cached-step input embed.
        # So the first cached transformer call arrives with `decode_step == 1`.
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

    logits_step = output.logits[:, -1, :]
    cond = logits_step[0::2, :]
    uncond = logits_step[1::2, :]
    cfg_weights = torch.tensor(
        [state.request.cfg_weight for state in cohort.active_states],
        device=cond.device,
        dtype=cond.dtype,
    ).unsqueeze(-1)
    logits = cond + cfg_weights * (cond - uncond)

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

    finished_results: list[tuple[str, Tensor]] = []
    first_token_session_ids: list[str] = []
    next_round_states = []
    for row_index, state in enumerate(cohort.active_states):
        request = state.request
        ids_for_proc = state.generated_ids
        row_logits = logits[row_index : row_index + 1]

        row_logits = RepetitionPenaltyLogitsProcessor(
            penalty=float(request.repetition_penalty)
        )(ids_for_proc, row_logits)

        if request.temperature != 1.0:
            row_logits = row_logits / request.temperature

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
            finished_results.append((request.session_id, _finalize_prediction(t3, state)))
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

    cohort.active_states = next_round_states
    return finished_results, first_token_session_ids, not cohort.active_states
