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

    @classmethod
    def create(cls, *, text_tokens_slice: tuple[int, int], eos_idx: int):
        i, j = text_tokens_slice
        return cls(
            text_tokens_slice=text_tokens_slice,
            eos_idx=eos_idx,
            alignment=torch.zeros(0, j - i),
        )

    def step(self, logits: torch.Tensor, aligned_attn: torch.Tensor, next_token=None) -> torch.Tensor:
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

        if next_token is not None:
            token_id = next_token.item() if isinstance(next_token, torch.Tensor) else next_token
            self.generated_tokens.append(token_id)
            if len(self.generated_tokens) > 8:
                self.generated_tokens = self.generated_tokens[-8:]

        token_repetition = len(self.generated_tokens) >= 3 and len(set(self.generated_tokens[-2:])) == 1
        if token_repetition:
            logger.warning("🚨 Detected 2x repetition of token %s", self.generated_tokens[-1])

        if cur_text_posn < text_len - 3 and text_len > 5:
            logits[..., self.eos_idx] = -(2**15)

        if long_tail or alignment_repetition or token_repetition:
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


class ScheduledAlignmentController:
    """
    Batch-aware attention spy for a scheduler-driven T3 decode loop.

    Each request keeps its own alignment state, while the controller installs a
    single shared set of hooks and extracts only the CFG-conditioned rows
    `(0, 2, 4, ...)` from the batched forward pass.
    """

    def __init__(self, tfmr):
        self.tfmr = tfmr
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

    def step(
        self,
        logits: torch.Tensor,
        *,
        active_states: list[ScheduledAlignmentState],
        next_tokens: list[int | None],
    ) -> torch.Tensor:
        if not active_states:
            return logits
        if any(attn is None for attn in self.last_aligned_attns):
            return logits

        aligned_attn = torch.stack(self.last_aligned_attns).mean(dim=0)
        for index, state in enumerate(active_states):
            next_token = next_tokens[index] if index < len(next_tokens) else None
            logits[index : index + 1] = state.step(
                logits[index : index + 1],
                aligned_attn[index],
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
