# Copyright (c) 2025 Resemble AI
# Author: John Meade, Jeremy Hsu
# MIT License
import logging
from dataclasses import dataclass, field

import torch


logger = logging.getLogger(__name__)


LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]


@dataclass
class ScheduledAlignmentState:
    text_tokens_slice: tuple[int, int]
    eos_idx: int
    alignment: torch.Tensor
    curr_frame_pos: int = 0
    text_position: int = 0
    started: bool = False
    started_at: int | None = None
    complete: bool = False
    completed_at: int | None = None
    generated_tokens: list[int] = field(default_factory=list)
    attention_steps: int = 0
    cheap_steps: int = 0
    eos_blocked_steps: int = 0
    forced_eos_steps: int = 0
    forced_eos_long_tail_steps: int = 0
    forced_eos_alignment_repetition_steps: int = 0
    forced_eos_token_repetition_steps: int = 0
    token_repetition_hits: int = 0

    @classmethod
    def create(cls, *, text_tokens_slice: tuple[int, int], eos_idx: int):
        i, j = text_tokens_slice
        return cls(
            text_tokens_slice=text_tokens_slice,
            eos_idx=eos_idx,
            alignment=torch.zeros(0, j - i),
        )

    def _update_generated_tokens(self, next_token=None) -> bool:
        if next_token is not None:
            token_id = next_token.item() if isinstance(next_token, torch.Tensor) else next_token
            self.generated_tokens.append(token_id)
            if len(self.generated_tokens) > 8:
                self.generated_tokens = self.generated_tokens[-8:]

        token_repetition = len(self.generated_tokens) >= 3 and len(set(self.generated_tokens[-2:])) == 1
        if token_repetition:
            self.token_repetition_hits += 1
            logger.warning("🚨 Detected 2x repetition of token %s", self.generated_tokens[-1])
        return token_repetition

    def step(self, logits: torch.Tensor, aligned_attn: torch.Tensor, next_token=None) -> torch.Tensor:
        self.attention_steps += 1
        i, j = self.text_tokens_slice
        if self.curr_frame_pos == 0:
            attn_chunk = aligned_attn[j:, i:j].clone().cpu()
        else:
            attn_chunk = aligned_attn[:, i:j].clone().cpu()

        attn_chunk[:, self.curr_frame_pos + 1 :] = 0

        self.alignment = torch.cat((self.alignment, attn_chunk), dim=0)

        alignment = self.alignment
        _, text_len = alignment.shape

        cur_text_posn = attn_chunk[-1].argmax()
        discontinuity = not (-4 < cur_text_posn - self.text_position < 7)
        if not discontinuity:
            self.text_position = cur_text_posn

        false_start = (not self.started) and (
            alignment[-2:, -2:].max() > 0.1 or alignment[:, :4].max() < 0.5
        )
        self.started = not false_start
        if self.started and self.started_at is None:
            self.started_at = alignment.size(0)

        self.complete = self.complete or self.text_position >= text_len - 3
        if self.complete and self.completed_at is None:
            self.completed_at = alignment.size(0)

        long_tail = self.complete and (alignment[self.completed_at :, -3:].sum(dim=0).max() >= 5)
        alignment_repetition = self.complete and (
            alignment[self.completed_at :, :-5].max(dim=1).values.sum() > 5
        )

        token_repetition = self._update_generated_tokens(next_token=next_token)

        if cur_text_posn < text_len - 3 and text_len > 5:
            self.eos_blocked_steps += 1
            logits[..., self.eos_idx] = -(2**15)

        if long_tail or alignment_repetition or token_repetition:
            self.forced_eos_steps += 1
            if long_tail:
                self.forced_eos_long_tail_steps += 1
            if alignment_repetition:
                self.forced_eos_alignment_repetition_steps += 1
            if token_repetition:
                self.forced_eos_token_repetition_steps += 1
            logger.warning(
                "forcing EOS token, long_tail=%s, alignment_repetition=%s, token_repetition=%s",
                long_tail,
                alignment_repetition,
                token_repetition,
            )
            logits = -(2**15) * torch.ones_like(logits)
            logits[..., self.eos_idx] = 2**15

        self.curr_frame_pos += 1
        return logits

    def step_without_attention(self, logits: torch.Tensor, next_token=None) -> torch.Tensor:
        self.cheap_steps += 1
        token_repetition = self._update_generated_tokens(next_token=next_token)
        if token_repetition:
            self.forced_eos_steps += 1
            self.forced_eos_token_repetition_steps += 1
            logger.warning("forcing EOS token without attention, token_repetition=%s", token_repetition)
            logits = -(2**15) * torch.ones_like(logits)
            logits[..., self.eos_idx] = 2**15
        self.curr_frame_pos += 1
        return logits

    def export_metrics(self) -> dict[str, float]:
        return {
            "alignment_attention_steps": float(self.attention_steps),
            "alignment_cheap_steps": float(self.cheap_steps),
            "alignment_eos_blocked_steps": float(self.eos_blocked_steps),
            "alignment_forced_eos_steps": float(self.forced_eos_steps),
            "alignment_forced_eos_long_tail_steps": float(self.forced_eos_long_tail_steps),
            "alignment_forced_eos_alignment_repetition_steps": float(self.forced_eos_alignment_repetition_steps),
            "alignment_forced_eos_token_repetition_steps": float(self.forced_eos_token_repetition_steps),
            "alignment_token_repetition_hits": float(self.token_repetition_hits),
        }


class ScheduledAlignmentController:
    """
    Batch-aware attention spy for a scheduler-driven T3 decode loop.

    Each request keeps its own alignment state, while the controller installs a
    single shared set of hooks and extracts only the CFG-conditioned rows
    `(0, 2, 4, ...)` from the batched forward pass.
    """

    def __init__(self, tfmr, *, inspect_every: int = 1):
        self.tfmr = tfmr
        self.inspect_every = max(1, int(inspect_every))
        self.original_output_attentions = None
        self.hook_handles = []
        self.last_aligned_attns = [None] * len(LLAMA_ALIGNED_HEADS)

        for buffer_idx, (layer_idx, head_idx) in enumerate(LLAMA_ALIGNED_HEADS):
            self._add_attention_spy(buffer_idx, layer_idx, head_idx)

    def _add_attention_spy(self, buffer_idx, layer_idx, head_idx):
        def attention_forward_hook(module, inputs, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                step_attention = output[1].cpu()
                self.last_aligned_attns[buffer_idx] = step_attention[0::2, head_idx]

        target_layer = self.tfmr.layers[layer_idx].self_attn
        handle = target_layer.register_forward_hook(attention_forward_hook)
        self.hook_handles.append(handle)

        if hasattr(self.tfmr, "config") and hasattr(self.tfmr.config, "output_attentions"):
            if self.original_output_attentions is None:
                self.original_output_attentions = self.tfmr.config.output_attentions
            self.tfmr.config.output_attentions = True

    def prepare_for_forward(self, request_attentions: bool):
        if request_attentions:
            self.last_aligned_attns = [None] * len(LLAMA_ALIGNED_HEADS)

    def should_inspect_state(self, state: ScheduledAlignmentState) -> bool:
        return state.curr_frame_pos == 0 or state.curr_frame_pos % self.inspect_every == 0

    def should_request_attentions(self, active_states: list[ScheduledAlignmentState]) -> bool:
        return any(self.should_inspect_state(state) for state in active_states)

    def apply(
        self,
        logits: torch.Tensor,
        *,
        active_states: list[ScheduledAlignmentState],
        next_tokens: list[int | None],
        attentions_requested: bool,
    ) -> torch.Tensor:
        if not active_states:
            return logits
        aligned_attn = None
        if attentions_requested and not any(attn is None for attn in self.last_aligned_attns):
            aligned_attn = torch.stack(self.last_aligned_attns).mean(dim=0)
        for index, state in enumerate(active_states):
            next_token = next_tokens[index] if index < len(next_tokens) else None
            if aligned_attn is not None and self.should_inspect_state(state):
                logits[index : index + 1] = state.step(
                    logits[index : index + 1],
                    aligned_attn[index],
                    next_token=next_token,
                )
            else:
                logits[index : index + 1] = state.step_without_attention(
                    logits[index : index + 1],
                    next_token=next_token,
                )
        return logits

    def close(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

        if (
            self.original_output_attentions is not None
            and hasattr(self.tfmr, "config")
            and hasattr(self.tfmr.config, "output_attentions")
        ):
            self.tfmr.config.output_attentions = self.original_output_attentions
