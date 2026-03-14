import logging
import os
from typing import Optional

import torch
from torch import Tensor
from tqdm import tqdm
from transformers.generation.logits_process import (
    MinPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    TopPLogitsWarper,
)

from .alignment_stream_analyzer_concurrent import ConcurrentAlignmentStreamAnalyzer
from .t3_hf_backend import T3HuggingfaceBackend


shape_logger = logging.getLogger("chatterbox.shape")
logger = logging.getLogger(__name__)


def _ensure_bot_eot(text_tokens: Tensor, hp):
    batch = text_tokens.size(0)
    assert (text_tokens == hp.start_text_token).int().sum() >= batch, "missing start_text_token"
    assert (text_tokens == hp.stop_text_token).int().sum() >= batch, "missing stop_text_token"


@torch.inference_mode()
def run_concurrent_t3_inference(
    t3,
    *,
    t3_cond,
    text_tokens: Tensor,
    initial_speech_tokens: Optional[Tensor] = None,
    prepend_prompt_speech_tokens: Optional[Tensor] = None,
    max_new_tokens=None,
    stop_on_eos=True,
    temperature=0.8,
    top_p=0.95,
    min_p=0.05,
    repetition_penalty=1.2,
    cfg_weight=0.5,
    enable_alignment_checks=True,
):
    assert prepend_prompt_speech_tokens is None, "not implemented"
    _ensure_bot_eot(text_tokens, t3.hp)
    text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=t3.device)

    if initial_speech_tokens is None:
        initial_speech_tokens = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])

    max_new_tokens = max_new_tokens or t3.hp.max_speech_tokens

    embeds, len_cond = t3.prepare_input_embeds(
        t3_cond=t3_cond,
        text_tokens=text_tokens,
        speech_tokens=initial_speech_tokens,
        cfg_weight=cfg_weight,
    )
    if os.getenv("CHATTERBOX_TRACE_SHAPES"):
        shape_logger.info("[models/t3/inference/concurrent_decode.py] inference.input")
        shape_logger.info("  text_tokens %s %s %s", tuple(text_tokens.shape), text_tokens.dtype, text_tokens.device)
        shape_logger.info(
            "  initial_speech_tokens %s %s %s",
            tuple(initial_speech_tokens.shape),
            initial_speech_tokens.dtype,
            initial_speech_tokens.device,
        )
        shape_logger.info("  embeds %s %s %s", tuple(embeds.shape), embeds.dtype, embeds.device)
        shape_logger.info("  len_cond %s", len_cond)

    alignment_stream_analyzer = None
    if enable_alignment_checks and t3.hp.is_multilingual:
        alignment_stream_analyzer = ConcurrentAlignmentStreamAnalyzer(
            t3.tfmr,
            None,
            text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
            alignment_layer_idx=9,
            eos_idx=t3.hp.stop_speech_token,
        )

    patched_model = T3HuggingfaceBackend(
        config=t3.cfg,
        llama=t3.tfmr,
        speech_enc=t3.speech_emb,
        speech_head=t3.speech_head,
        alignment_stream_analyzer=alignment_stream_analyzer,
    )
    if os.getenv("CHATTERBOX_TRACE_SHAPES"):
        shape_logger.info("[models/t3/inference/concurrent_decode.py] runtime_state")
        shape_logger.info("  local_backend %s", patched_model.__class__.__name__)
        shape_logger.info("  has_alignment_stream_analyzer %s", alignment_stream_analyzer is not None)

    try:
        device = embeds.device
        bos_token = torch.tensor([[t3.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = t3.speech_emb(bos_token)
        bos_embed = bos_embed + t3.speech_pos_emb.get_fixed_embedding(0)
        bos_embed = torch.cat([bos_embed, bos_embed])

        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)
        generated_ids = bos_token.clone()
        predicted = []

        top_p_warper = TopPLogitsWarper(top_p=top_p)
        min_p_warper = MinPLogitsWarper(min_p=min_p)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

        output = patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = output.past_key_values

        for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):
            logits_step = output.logits[:, -1, :]
            cond = logits_step[0:1, :]
            uncond = logits_step[1:2, :]
            cfg = torch.as_tensor(cfg_weight, device=cond.device, dtype=cond.dtype)
            logits = cond + cfg * (cond - uncond)

            if alignment_stream_analyzer is not None:
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                last_token = generated_ids[0, -1].item() if generated_ids.size(1) > 0 else None
                logits = alignment_stream_analyzer.step(logits, next_token=last_token)

            ids_for_proc = generated_ids[:1, ...]
            logits = repetition_penalty_processor(ids_for_proc, logits)

            if temperature != 1.0:
                logits = logits / temperature

            logits = min_p_warper(ids_for_proc, logits)
            logits = top_p_warper(ids_for_proc, logits)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            predicted.append(next_token)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if stop_on_eos and torch.all(next_token.view(-1) == t3.hp.stop_speech_token):
                logger.info("✅ EOS token detected! Stopping generation at step %s", i + 1)
                break

            next_token_embed = t3.speech_emb(next_token)
            next_token_embed = next_token_embed + t3.speech_pos_emb.get_fixed_embedding(i + 1)
            next_token_embed = torch.cat([next_token_embed, next_token_embed])

            output = patched_model(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = output.past_key_values

        if predicted:
            predicted_tokens = torch.cat(predicted, dim=1)
        else:
            predicted_tokens = torch.empty((1, 0), dtype=torch.long, device=device)

        if os.getenv("CHATTERBOX_TRACE_SHAPES"):
            shape_logger.info("[models/t3/inference/concurrent_decode.py] inference.output")
            shape_logger.info(
                "  predicted_tokens %s %s %s",
                tuple(predicted_tokens.shape),
                predicted_tokens.dtype,
                predicted_tokens.device,
            )
        return predicted_tokens
    finally:
        if alignment_stream_analyzer is not None:
            alignment_stream_analyzer.close()
