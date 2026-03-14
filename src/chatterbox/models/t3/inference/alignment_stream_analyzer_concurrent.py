# Copyright (c) 2025 Resemble AI
# Author: John Meade, Jeremy Hsu
# MIT License
import logging
from dataclasses import dataclass

import torch


logger = logging.getLogger(__name__)


LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]


@dataclass
class AlignmentAnalysisResult:
    false_start: bool
    long_tail: bool
    repetition: bool
    discontinuity: bool
    complete: bool
    position: int


class ConcurrentAlignmentStreamAnalyzer:
    """
    Request-local copy of the alignment analyzer.

    The original implementation registers forward hooks on shared transformer layers
    but never removes them. This version stores hook handles and restores shared
    config state on close().
    """

    def __init__(self, tfmr, queue, text_tokens_slice, alignment_layer_idx=9, eos_idx=0):
        self.text_tokens_slice = (i, j) = text_tokens_slice
        self.eos_idx = eos_idx
        self.alignment = torch.zeros(0, j - i)
        self.curr_frame_pos = 0
        self.text_position = 0

        self.started = False
        self.started_at = None

        self.complete = False
        self.completed_at = None
        self.generated_tokens = []

        self.tfmr = tfmr
        self.original_output_attentions = None
        self.hook_handles = []

        self.last_aligned_attns = []
        for buffer_idx, (layer_idx, head_idx) in enumerate(LLAMA_ALIGNED_HEADS):
            self.last_aligned_attns.append(None)
            self._add_attention_spy(tfmr, buffer_idx, layer_idx, head_idx)

    def _add_attention_spy(self, tfmr, buffer_idx, layer_idx, head_idx):
        def attention_forward_hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                step_attention = output[1].cpu()
                self.last_aligned_attns[buffer_idx] = step_attention[0, head_idx]

        target_layer = tfmr.layers[layer_idx].self_attn
        handle = target_layer.register_forward_hook(attention_forward_hook)
        self.hook_handles.append(handle)

        if hasattr(tfmr, "config") and hasattr(tfmr.config, "output_attentions"):
            if self.original_output_attentions is None:
                self.original_output_attentions = tfmr.config.output_attentions
            tfmr.config.output_attentions = True

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

    def step(self, logits, next_token=None):
        aligned_attn = torch.stack(self.last_aligned_attns).mean(dim=0)
        i, j = self.text_tokens_slice
        if self.curr_frame_pos == 0:
            a_chunk = aligned_attn[j:, i:j].clone().cpu()
        else:
            a_chunk = aligned_attn[:, i:j].clone().cpu()

        a_chunk[:, self.curr_frame_pos + 1:] = 0

        self.alignment = torch.cat((self.alignment, a_chunk), dim=0)

        a = self.alignment
        _, s = a.shape

        cur_text_posn = a_chunk[-1].argmax()
        discontinuity = not (-4 < cur_text_posn - self.text_position < 7)
        if not discontinuity:
            self.text_position = cur_text_posn

        false_start = (not self.started) and (a[-2:, -2:].max() > 0.1 or a[:, :4].max() < 0.5)
        self.started = not false_start
        if self.started and self.started_at is None:
            self.started_at = a.size(0)

        self.complete = self.complete or self.text_position >= s - 3
        if self.complete and self.completed_at is None:
            self.completed_at = a.size(0)

        long_tail = self.complete and (a[self.completed_at:, -3:].sum(dim=0).max() >= 5)
        alignment_repetition = self.complete and (a[self.completed_at:, :-5].max(dim=1).values.sum() > 5)

        if next_token is not None:
            if isinstance(next_token, torch.Tensor):
                token_id = next_token.item() if next_token.numel() == 1 else next_token.view(-1)[0].item()
            else:
                token_id = next_token
            self.generated_tokens.append(token_id)
            if len(self.generated_tokens) > 8:
                self.generated_tokens = self.generated_tokens[-8:]

        token_repetition = len(self.generated_tokens) >= 3 and len(set(self.generated_tokens[-2:])) == 1
        if token_repetition:
            logger.warning("🚨 Detected 2x repetition of token %s", self.generated_tokens[-1])

        if cur_text_posn < s - 3 and s > 5:
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
