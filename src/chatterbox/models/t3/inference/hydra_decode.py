from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
import torch
from torch import Tensor, nn

from .scheduled_decode import ScheduledDecodeRequest
from .speculative_decode import (
    _PrototypeState,
    SpeculativePrototypeResult,
    _build_cfg_step_inputs,
    _cfg_combine,
    prefill_prototype_state,
    verify_block_greedy,
)
from ..train import load_hydra_heads_from_checkpoint as _load_hydra_heads_from_checkpoint


shape_logger = logging.getLogger("chatterbox.shape")
_TRACE_COUNTS: dict[str, int] = {}


def _trace_enabled() -> bool:
    return bool(os.getenv("CHATTERBOX_TRACE_SHAPES"))


def _trace_stride() -> int:
    raw_stride = os.getenv("CHATTERBOX_TRACE_SPEC_EVERY", "6")
    try:
        return max(int(raw_stride), 1)
    except ValueError:
        return 6


def _reset_trace_counters() -> None:
    _TRACE_COUNTS.clear()


def _should_trace_event(name: str) -> tuple[bool, int]:
    occurrence = _TRACE_COUNTS.get(name, 0) + 1
    _TRACE_COUNTS[name] = occurrence
    stride = _trace_stride()
    return occurrence == 1 or occurrence % stride == 0, occurrence


class ResBlock(nn.Module):
    def __init__(self, hidden_size: int, input_dim: int | None = None):
        super().__init__()
        if input_dim is None:
            input_dim = hidden_size
        self.linear = nn.Linear(input_dim, hidden_size)
        self.res_connection = nn.Identity() if input_dim == hidden_size else nn.Linear(input_dim, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.res_connection(x) + self.act(self.linear(x))


class HydraGroundedMLPHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        context_tokens: int,
        num_layers: int,
        lm_head_init_weight: Tensor | None = None,
    ):
        super().__init__()
        assert context_tokens >= 1, "Hydra heads need at least one grounded context token"
        assert num_layers >= 1, "Hydra heads need at least one layer"
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.context_tokens = context_tokens

        input_dim = hidden_size * (context_tokens + 1)
        blocks = [ResBlock(hidden_size, input_dim)]
        for _ in range(num_layers - 1):
            blocks.append(ResBlock(hidden_size))
        self.mlp = nn.Sequential(*blocks)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if lm_head_init_weight is not None:
            self.lm_head.weight.data.copy_(lm_head_init_weight)

    def forward(self, base_hidden_states: Tensor, context_embeds: Tensor) -> tuple[Tensor, Tensor]:
        assert base_hidden_states.dim() == 3, tuple(base_hidden_states.shape)
        assert context_embeds.dim() == 3, tuple(context_embeds.shape)
        assert base_hidden_states.size(0) == context_embeds.size(0), (
            tuple(base_hidden_states.shape),
            tuple(context_embeds.shape),
        )
        assert base_hidden_states.size(1) == 1, tuple(base_hidden_states.shape)
        assert context_embeds.size(1) == self.context_tokens, (
            self.context_tokens,
            tuple(context_embeds.shape),
        )

        head_input = torch.cat(
            [base_hidden_states, context_embeds.reshape(context_embeds.size(0), 1, -1)],
            dim=-1,
        )
        hydra_hidden = self.mlp(head_input)
        hydra_logits = self.lm_head(hydra_hidden)
        return hydra_logits, hydra_hidden


class T3HydraHeadModel(nn.Module):
    def __init__(
        self,
        t3,
        *,
        hydra_num_heads: int = 2,
        hydra_num_layers: int = 1,
        grounded_heads: bool = True,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.t3 = t3
        self.hydra_num_heads = hydra_num_heads
        self.hydra_num_layers = hydra_num_layers
        self.grounded_heads = grounded_heads
        self.freeze_base = freeze_base
        self.hidden_size = t3.cfg.hidden_size
        self.vocab_size = t3.hp.speech_tokens_dict_size

        if not grounded_heads:
            raise NotImplementedError("Only grounded Hydra heads are supported in this prototype")

        self.hydra_heads = nn.ModuleList(
            [
                HydraGroundedMLPHead(
                    hidden_size=self.hidden_size,
                    vocab_size=self.vocab_size,
                    context_tokens=head_index + 1,
                    num_layers=hydra_num_layers,
                    lm_head_init_weight=t3.speech_head.weight.data,
                )
                for head_index in range(hydra_num_heads)
            ]
        )

        if freeze_base:
            for param in self.t3.parameters():
                param.requires_grad_(False)
            self.t3.eval()

    @property
    def device(self):
        return self.t3.device

    def trainable_parameters(self):
        return self.hydra_heads.parameters()

    def forward_hydra_head(self, head_index: int, hidden_states: Tensor, context_embeds: Tensor) -> tuple[Tensor, Tensor]:
        return self.hydra_heads[head_index](hidden_states, context_embeds)

    def forward_hydra_heads(self, hidden_states: Tensor, context_embeds: Tensor) -> tuple[Tensor, Tensor]:
        hydra_logits = []
        hydra_hidden_states = []
        for head_index in range(self.hydra_num_heads):
            logits, hidden = self.forward_hydra_head(
                head_index,
                hidden_states,
                context_embeds[:, : head_index + 1, :],
            )
            hydra_logits.append(logits)
            hydra_hidden_states.append(hidden)
        return torch.stack(hydra_logits, dim=0), torch.stack(hydra_hidden_states, dim=0)


def load_hydra_heads_from_checkpoint(*, base_t3, checkpoint_dir, freeze_base: bool = True):
    return _load_hydra_heads_from_checkpoint(
        base_t3=base_t3,
        checkpoint_dir=checkpoint_dir,
        freeze_base=freeze_base,
    )


@torch.inference_mode()
def hydra_propose_block(
    t3,
    hydra_model,
    *,
    state: _PrototypeState,
    speculate_k: int,
) -> Tensor:
    assert speculate_k >= 1
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
        head_logits = raw_logits[:, -1, :]
        future_token = head_logits.argmax(dim=-1, keepdim=True)
        proposed.append(future_token)

        should_trace, occurrence = _should_trace_event("hydra_propose_block")
        if _trace_enabled() and should_trace:
            shape_logger.info("[models/t3/inference/hydra_decode.py] hydra.propose")
            shape_logger.info("  occurrence %s", occurrence)
            shape_logger.info("  head_index %s", head_index)
            shape_logger.info("  context_tokens %s %s %s", tuple(context_tokens.shape), context_tokens.dtype, context_tokens.device)
            shape_logger.info("  base_hidden %s %s %s", tuple(base_hidden.shape), base_hidden.dtype, base_hidden.device)
            shape_logger.info("  context_embeds %s %s %s", tuple(context_embeds.shape), context_embeds.dtype, context_embeds.device)
            shape_logger.info("  raw_logits %s %s %s", tuple(raw_logits.shape), raw_logits.dtype, raw_logits.device)
            shape_logger.info("  future_token_ids %s", future_token.view(-1).tolist())

        if torch.all(future_token.view(-1) == t3.hp.stop_speech_token):
            break

    proposed_tokens = torch.cat(proposed, dim=1)
    if _trace_enabled():
        shape_logger.info("[models/t3/inference/hydra_decode.py] hydra.proposed")
        shape_logger.info("  decode_step %s", state.decode_step)
        shape_logger.info("  speculate_k %s", speculate_k)
        shape_logger.info("  proposal_cap %s", proposal_cap)
        shape_logger.info("  proposed_tokens %s %s %s", tuple(proposed_tokens.shape), proposed_tokens.dtype, proposed_tokens.device)
        shape_logger.info("  proposed_token_ids %s", proposed_tokens.view(-1).tolist())
    return proposed_tokens


@torch.inference_mode()
def run_hydra_speculative_decode(
    target_t3,
    hydra_model: T3HydraHeadModel,
    request: ScheduledDecodeRequest,
    *,
    speculate_k: int,
) -> SpeculativePrototypeResult:
    assert speculate_k >= 1
    _reset_trace_counters()
    target_state = prefill_prototype_state(target_t3, request)
    predicted_tokens: list[Tensor] = []
    rounds = 0
    proposed_tokens_total = 0
    accepted_draft_tokens_total = 0
    correction_tokens_total = 0
    full_accept_rounds = 0
    zero_accept_rounds = 0
    partial_accept_rounds = 0
    match_len_hist = [0] * (speculate_k + 1)

    while sum(token.size(1) for token in predicted_tokens) < request.max_new_tokens:
        remaining = request.max_new_tokens - sum(token.size(1) for token in predicted_tokens)
        proposal = hydra_propose_block(
            target_t3,
            hydra_model,
            state=target_state,
            speculate_k=min(speculate_k, remaining),
        )
        verify = verify_block_greedy(
            target_t3,
            state=target_state,
            proposed_tokens=proposal,
        )

        committed_tokens = verify.committed_tokens
        predicted_tokens.append(committed_tokens)
        target_state.generated_ids = torch.cat([target_state.generated_ids, committed_tokens], dim=1)
        target_state.past_key_values = verify.next_past_key_values
        target_state.next_logits = verify.next_logits
        target_state.next_hidden = verify.next_hidden
        target_state.decode_step += committed_tokens.size(1)

        accepted_tokens = verify.accepted_draft_tokens
        match_len_hist[accepted_tokens] += 1
        if verify.correction_token is None and accepted_tokens == verify.proposed_tokens:
            full_accept_rounds += 1
        elif accepted_tokens == 0:
            zero_accept_rounds += 1
        else:
            partial_accept_rounds += 1

        rounds += 1
        proposed_tokens_total += verify.proposed_tokens
        accepted_draft_tokens_total += verify.accepted_draft_tokens
        if verify.correction_token is not None:
            correction_tokens_total += 1

        if torch.any(committed_tokens == target_t3.hp.stop_speech_token):
            break

    if predicted_tokens:
        speech_tokens = torch.cat(predicted_tokens, dim=1)
    else:
        speech_tokens = torch.empty((1, 0), dtype=torch.long, device=target_t3.device)

    return SpeculativePrototypeResult(
        speech_tokens=speech_tokens,
        rounds=rounds,
        proposed_tokens_total=proposed_tokens_total,
        accepted_draft_tokens_total=accepted_draft_tokens_total,
        correction_tokens_total=correction_tokens_total,
        rebuild_count=0,
        rebuild_tokens_total=0,
        full_accept_rounds=full_accept_rounds,
        zero_accept_rounds=zero_accept_rounds,
        partial_accept_rounds=partial_accept_rounds,
        match_len_hist=tuple(match_len_hist),
    )
