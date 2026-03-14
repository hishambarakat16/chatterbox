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
    finished: bool = False
    next_inputs_embeds: Tensor | None = None


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
    bos_embed = t3.speech_emb(bos_token)
    bos_embed = bos_embed + t3.speech_pos_emb.get_fixed_embedding(0)
    bos_embed = torch.cat([bos_embed, bos_embed], dim=0)

    alignment_state = None
    if t3.hp.is_multilingual:
        alignment_state = ScheduledAlignmentState.create(
            text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
            eos_idx=t3.hp.stop_speech_token,
        )

    return (
        _ActiveDecodeState(
            request=request,
            generated_ids=bos_token.clone(),
            alignment_state=alignment_state,
        ),
        torch.cat([embeds, bos_embed], dim=1),
    )


@torch.inference_mode()
def run_scheduled_t3_batch(t3, requests: list[ScheduledDecodeRequest]) -> list[Tensor]:
    if not requests:
        return []

    active_states: list[_ActiveDecodeState] = []
    all_states: list[_ActiveDecodeState] = []
    prefill_inputs = []
    batch_keys = {request.batch_key() for request in requests}
    if len(batch_keys) != 1:
        raise ValueError("scheduled batch currently requires matching text/prompt lengths")

    for request in requests:
        state, inputs_embeds = _build_initial_state(t3, request)
        active_states.append(state)
        all_states.append(state)
        prefill_inputs.append(inputs_embeds)

    patched_model = T3HuggingfaceBackend(
        config=t3.cfg,
        llama=t3.tfmr,
        speech_enc=t3.speech_emb,
        speech_head=t3.speech_head,
        alignment_stream_analyzer=None,
    )
    alignment_controller = ScheduledAlignmentController(t3.tfmr) if t3.hp.is_multilingual else None

    try:
        inputs_embeds = torch.cat(prefill_inputs, dim=0)
        if os.getenv("CHATTERBOX_TRACE_SHAPES"):
            shape_logger.info("[models/t3/inference/scheduled_decode.py] prefill.batch")
            shape_logger.info("  requests %s", len(active_states))
            shape_logger.info("  inputs_embeds %s %s %s", tuple(inputs_embeds.shape), inputs_embeds.dtype, inputs_embeds.device)

        output = patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past_splits = _split_past_key_values(output.past_key_values, [2] * len(active_states))
        for state, state_past in zip(active_states, past_splits):
            state.past_key_values = state_past

        global_max_steps = max(request.max_new_tokens for request in requests)

        for _ in range(global_max_steps):
            if not active_states:
                break

            logits_step = output.logits[:, -1, :]
            cond = logits_step[0::2, :]
            uncond = logits_step[1::2, :]
            cfg_weights = torch.tensor(
                [state.request.cfg_weight for state in active_states],
                device=cond.device,
                dtype=cond.dtype,
            ).unsqueeze(-1)
            logits = cond + cfg_weights * (cond - uncond)

            if alignment_controller is not None:
                last_tokens = [state.generated_ids[0, -1].item() if state.generated_ids.size(1) > 0 else None for state in active_states]
                logits = alignment_controller.step(
                    logits,
                    active_states=[state.alignment_state for state in active_states],
                    next_tokens=last_tokens,
                )

            next_round_states = []
            next_inputs = []

            for row_index, state in enumerate(active_states):
                if state.finished:
                    continue

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

                stop_on_eos = torch.all(next_token.view(-1) == t3.hp.stop_speech_token)
                hit_limit = len(state.predicted_tokens) >= request.max_new_tokens
                if stop_on_eos or hit_limit:
                    state.finished = True
                    if stop_on_eos:
                        logger.info("✅ EOS token detected for %s", request.session_id)
                    continue

                next_token_embed = t3.speech_emb(next_token)
                next_token_embed = next_token_embed + t3.speech_pos_emb.get_fixed_embedding(state.decode_step + 1)
                state.next_inputs_embeds = torch.cat([next_token_embed, next_token_embed], dim=0)
                state.decode_step += 1
                next_round_states.append(state)
                next_inputs.append(state.next_inputs_embeds)

            if not next_round_states:
                break

            batched_past = _cat_past_key_values([state.past_key_values for state in next_round_states])
            output = patched_model(
                inputs_embeds=torch.cat(next_inputs, dim=0),
                past_key_values=batched_past,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past_splits = _split_past_key_values(output.past_key_values, [2] * len(next_round_states))
            for state, state_past in zip(next_round_states, past_splits):
                state.past_key_values = state_past
            active_states = next_round_states

        request_to_state = {state.request.session_id: state for state in all_states}
        results: list[Tensor] = []
        for request in requests:
            state = request_to_state[request.session_id]
            if state.predicted_tokens:
                predicted_tokens = torch.cat(state.predicted_tokens, dim=1)
            else:
                predicted_tokens = torch.empty((1, 0), dtype=torch.long, device=t3.device)
            if os.getenv("CHATTERBOX_TRACE_SHAPES"):
                shape_logger.info("[models/t3/inference/scheduled_decode.py] inference.output")
                shape_logger.info("  session_id %s", request.session_id)
                shape_logger.info(
                    "  predicted_tokens %s %s %s",
                    tuple(predicted_tokens.shape),
                    predicted_tokens.dtype,
                    predicted_tokens.device,
                )
            results.append(predicted_tokens)
        return results

    finally:
        if alignment_controller is not None:
            alignment_controller.close()
