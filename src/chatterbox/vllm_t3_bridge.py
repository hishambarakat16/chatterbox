from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from .models.t3.llama_configs import LLAMA_CONFIGS
from .models.t3.modules.t3_config import T3Config

shape_logger = logging.getLogger("chatterbox.shape")

def _trace_shapes() -> bool:
    return bool(os.getenv("CHATTERBOX_TRACE_SHAPES"))


VLLM_T3_ARCHITECTURE = "ChatterboxT3ForCausalLM"
BASE_T3_FILENAME = "t3_mtl23ls_v2.safetensors"
HYDRA_HEADS_FILENAME = "t3_hydra_heads.safetensors"
REPO_ID = "ResembleAI/chatterbox"


def punc_norm(text: str) -> str:
    if len(text) == 0:
        return "You need to add some text for me to talk."

    if text[0].islower():
        text = text[0].upper() + text[1:]

    text = " ".join(text.split())

    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",", "、", "，", "。", "？", "！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


def optional_import_vllm():
    try:
        from vllm import LLM, ModelRegistry, SamplingParams
    except ModuleNotFoundError as exc:
        if exc.name != "vllm":
            raise
        raise ImportError(
            "vLLM is not installed in the active environment. "
            "Activate your dedicated vLLM env (for example `conda activate chatterbox-vllm`) "
            "or install `vllm` into the current env. "
            "See `GPU_MIGRATION_SERVING_PLAN.md`."
        ) from exc
    except ImportError as exc:
        raise ImportError(
            "vLLM is installed but failed to import. "
            f"Original error: {exc}. "
            "This usually means a CUDA / wheel mismatch in the vLLM environment. "
            "Recreate the env, install vLLM with an explicit CUDA backend, and retry. "
            "See `GPU_MIGRATION_SERVING_PLAN.md`."
        ) from exc
    return LLM, ModelRegistry, SamplingParams


def register_vllm_t3_model():
    _, ModelRegistry, _ = optional_import_vllm()
    ModelRegistry.register_model(
        VLLM_T3_ARCHITECTURE,
        "chatterbox.vllm_t3_model:ChatterboxT3ForCausalLM",
    )


def build_vllm_t3_config(hp: T3Config | None = None) -> dict:
    hp = hp or T3Config.multilingual()
    cfg = dict(LLAMA_CONFIGS[hp.llama_config_name])
    cfg.update(
        {
            "architectures": [VLLM_T3_ARCHITECTURE],
            "vocab_size": hp.speech_tokens_dict_size,
            "bos_token_id": hp.start_speech_token,
            "eos_token_id": hp.stop_speech_token,
            "pad_token_id": hp.stop_speech_token,
            "chatterbox_text_vocab_size": hp.text_tokens_dict_size,
            "chatterbox_speech_vocab_size": hp.speech_tokens_dict_size,
            "chatterbox_start_text_token": hp.start_text_token,
            "chatterbox_stop_text_token": hp.stop_text_token,
            "chatterbox_start_speech_token": hp.start_speech_token,
            "chatterbox_stop_speech_token": hp.stop_speech_token,
            "chatterbox_speech_pos_embeddings": hp.max_speech_tokens + 4,
            "chatterbox_prompt_embeds_required": True,
            "chatterbox_cfg_supported": False,
            "chatterbox_hydra_supported": False,
            "chatterbox_pos_strategy": "approx_absolute_speech_positions",
            "tie_word_embeddings": False,
        }
    )
    return cfg


def resolve_base_t3_checkpoint_dir(
    checkpoint_dir: str | Path | None,
    *,
    base_checkpoint_dir: str | Path | None = None,
    allow_pretrained_fallback: bool = False,
) -> Path:
    searched: list[Path] = []
    hydra_like_paths: list[Path] = []
    for raw_path in [base_checkpoint_dir, checkpoint_dir]:
        if raw_path is None:
            continue
        candidate = Path(raw_path)
        searched.append(candidate)
        if (candidate / BASE_T3_FILENAME).exists():
            return candidate
        if (candidate / HYDRA_HEADS_FILENAME).exists():
            hydra_like_paths.append(candidate)

    if allow_pretrained_fallback:
        return Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main",
                allow_patterns=[
                    "ve.pt",
                    BASE_T3_FILENAME,
                    "grapheme_mtl_merged_expanded_v1.json",
                    "conds.pt",
                    "Cangjie5_TC.json",
                ],
                token=os.getenv("HF_TOKEN"),
            )
        )

    if hydra_like_paths:
        hydra_paths = ", ".join(str(path) for path in hydra_like_paths)
        raise FileNotFoundError(
            "Received a Hydra-head checkpoint directory, but the vLLM export/runtime path is Hydra-free "
            f"and needs the base multilingual T3 checkpoint containing `{BASE_T3_FILENAME}`. "
            f"Hydra-only paths seen: {hydra_paths}. "
            "Pass `--base-checkpoint-dir /path/to/base_multilingual_ckpt` or use the exporter with `--from-pretrained`."
        )

    searched_desc = ", ".join(str(path) for path in searched) if searched else "<none>"
    raise FileNotFoundError(
        f"Could not find `{BASE_T3_FILENAME}` in: {searched_desc}. "
        "Pass the base multilingual checkpoint dir, or use the exporter with `--from-pretrained`."
    )


def export_vllm_t3_model(
    checkpoint_dir: str | Path | None,
    output_dir: str | Path | None = None,
    *,
    base_checkpoint_dir: str | Path | None = None,
    allow_pretrained_fallback: bool = False,
    use_symlink: bool = True,
) -> Path:
    requested_checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
    base_checkpoint_dir = resolve_base_t3_checkpoint_dir(
        checkpoint_dir,
        base_checkpoint_dir=base_checkpoint_dir,
        allow_pretrained_fallback=allow_pretrained_fallback,
    )
    output_dir = Path(output_dir) if output_dir is not None else Path(
        tempfile.mkdtemp(prefix="chatterbox_vllm_t3_")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_weights = base_checkpoint_dir / BASE_T3_FILENAME

    config = build_vllm_t3_config(T3Config.multilingual())
    generation_config = {
        "bos_token_id": config["bos_token_id"],
        "eos_token_id": config["eos_token_id"],
        "pad_token_id": config["pad_token_id"],
    }
    export_meta = {
        "requested_checkpoint_dir": None if requested_checkpoint_dir is None else str(requested_checkpoint_dir),
        "source_checkpoint_dir": str(base_checkpoint_dir),
        "source_weights": str(source_weights),
        "architecture": VLLM_T3_ARCHITECTURE,
        "hydra_supported": False,
        "cfg_supported": False,
        "notes": [
            "This is the first vLLM T3 spike package.",
            "Prompt embeddings are constructed outside vLLM.",
            "Hydra and CFG are intentionally deferred.",
            "Decode-side learned speech positions are approximate in this spike.",
        ],
    }

    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "generation_config.json").write_text(
        json.dumps(generation_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "chatterbox_vllm_export.json").write_text(
        json.dumps(export_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    target_weights = output_dir / "model.safetensors"
    if target_weights.exists() or target_weights.is_symlink():
        target_weights.unlink()
    if use_symlink:
        try:
            target_weights.symlink_to(source_weights.resolve())
        except OSError:
            shutil.copy2(source_weights, target_weights)
    else:
        shutil.copy2(source_weights, target_weights)

    return output_dir


def create_vllm_engine(
    *,
    model_dir: str | Path,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.5,
    enforce_eager: bool = False,
    dtype: str = "auto",
    max_model_len: int = 2048,
    enable_prefix_caching: bool = False,
):
    LLM, _, _ = optional_import_vllm()
    register_vllm_t3_model()
    return LLM(
        model=str(model_dir),
        skip_tokenizer_init=True,
        enable_prompt_embeds=True,
        trust_remote_code=False,
        hf_overrides={"architectures": [VLLM_T3_ARCHITECTURE]},
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        dtype=dtype,
        max_model_len=max_model_len,
        enable_prefix_caching=enable_prefix_caching,
    )


def make_sampling_params(*, options, hp: T3Config):
    _, _, SamplingParams = optional_import_vllm()
    return SamplingParams(
        temperature=float(options.temperature),
        top_p=float(options.top_p),
        min_p=float(options.min_p),
        # repetition_penalty is intentionally fixed at 1.0 (disabled).
        # vLLM's penalty path calls _make_prompt_token_ids_tensor(), which reads
        # token_ids_cpu for prompt tokens that were never written for embed-only
        # requests (prompt_token_ids=None). In mixed prefill+decode batches the
        # stale values exceed vocab_size and cause a CUDA device-side assert in
        # scatter_add_. Speech-token repetition is already guarded by the stop
        # token id; further suppression happens upstream in the T3 bridge.
        repetition_penalty=1.0,
        max_tokens=int(options.max_new_tokens),
        stop_token_ids=[int(hp.stop_speech_token)],
        detokenize=False,
        skip_special_tokens=False,
    )


def _maybe_sync(device: str):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def prepare_vllm_text_tokens(
    *,
    tokenizer,
    text: str,
    language_id: str | None,
    device: str,
    return_metadata: bool = False,
) -> torch.Tensor:
    total_start = time.perf_counter()
    normalize_start = time.perf_counter()
    normalized = punc_norm(text)
    normalize_s = time.perf_counter() - normalize_start

    tokenize_start = time.perf_counter()
    text_tokens = tokenizer.text_to_tokens(
        normalized,
        language_id=language_id.lower() if language_id else None,
    ).to(device)
    _maybe_sync(device)
    tokenize_s = time.perf_counter() - tokenize_start

    pad_start = time.perf_counter()
    text_tokens = torch.nn.functional.pad(
        torch.nn.functional.pad(
            text_tokens,
            (1, 0),
            value=int(T3Config.multilingual().start_text_token),
        ),
        (0, 1),
        value=int(T3Config.multilingual().stop_text_token),
    )
    _maybe_sync(device)
    pad_s = time.perf_counter() - pad_start

    if _trace_shapes():
        flat = text_tokens.flatten()
        shape_logger.info("[vllm_t3_bridge] prepare_vllm_text_tokens")
        shape_logger.info("  text (raw) %r", text)
        shape_logger.info("  text (normalized) %r", normalized)
        shape_logger.info("  language_id %s", language_id)
        shape_logger.info("  text_tokens %s %s %s", tuple(text_tokens.shape), text_tokens.dtype, text_tokens.device)
        shape_logger.info("  text_tokens[0] (first token id) %s", int(flat[0]) if flat.numel() > 0 else "empty")
        shape_logger.info("  text_tokens[-1] (last token id) %s", int(flat[-1]) if flat.numel() > 0 else "empty")
        shape_logger.info("  text_tokens values %s", flat.tolist())

    if not return_metadata:
        return text_tokens

    return text_tokens, {
        "text_normalize_s": normalize_s,
        "text_tokenize_s": tokenize_s,
        "text_pad_s": pad_s,
        "text_tokens_total_s": time.perf_counter() - total_start,
    }


def build_prompt_embeds(
    *,
    prompt_builder_t3,
    t3_cond,
    text_tokens: torch.Tensor,
    return_metadata: bool = False,
) -> torch.Tensor:
    total_start = time.perf_counter()
    cond_to_device_start = time.perf_counter()
    t3_cond = t3_cond.to(device=prompt_builder_t3.device)
    text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=prompt_builder_t3.device)
    initial_speech = prompt_builder_t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    _maybe_sync(prompt_builder_t3.device)
    cond_to_device_s = time.perf_counter() - cond_to_device_start

    if _trace_shapes():
        shape_logger.info("[vllm_t3_bridge] build_prompt_embeds.input")
        shape_logger.info("  text_tokens %s %s %s", tuple(text_tokens.shape), text_tokens.dtype, text_tokens.device)
        shape_logger.info("  initial_speech %s %s %s", tuple(initial_speech.shape), initial_speech.dtype, initial_speech.device)
        shape_logger.info("  t3_cond.speaker_emb %s", tuple(t3_cond.speaker_emb.shape) if hasattr(t3_cond, "speaker_emb") and t3_cond.speaker_emb is not None else None)
        shape_logger.info("  t3_cond.emotion_adv %s", tuple(t3_cond.emotion_adv.shape) if hasattr(t3_cond, "emotion_adv") and t3_cond.emotion_adv is not None else None)
        cpt = getattr(t3_cond, "cond_prompt_speech_tokens", None)
        shape_logger.info("  t3_cond.cond_prompt_speech_tokens %s", tuple(cpt.shape) if cpt is not None else None)

    prepare_start = time.perf_counter()
    embeds, _ = prompt_builder_t3.prepare_input_embeds(
        t3_cond=t3_cond,
        text_tokens=text_tokens,
        speech_tokens=initial_speech,
        cfg_weight=0.0,
    )
    if _trace_shapes():
        shape_logger.info("[vllm_t3_bridge] build_prompt_embeds.after_prepare_input_embeds")
        shape_logger.info("  embeds %s %s %s  (= cond + text + 1xBOS)", tuple(embeds.shape), embeds.dtype, embeds.device)

    # Append a second BOS embedding at speech_pos=0 to match native T3 inference.
    # T3.inference() builds: inputs_embeds = cat([prepare_input_embeds(...), bos_embed])
    # where bos_embed = speech_emb(BOS) + speech_pos_emb[0].  Without this extra token
    # the model context differs from training and it fails to emit EOS naturally.
    bos_ids = torch.tensor(
        [[int(prompt_builder_t3.hp.start_speech_token)]],
        dtype=torch.long,
        device=prompt_builder_t3.device,
    )
    extra_bos = prompt_builder_t3.speech_emb(bos_ids)  # (1, 1, dim)
    extra_bos = extra_bos + prompt_builder_t3.speech_pos_emb.get_fixed_embedding(0)  # (1, 1, dim)
    embeds = torch.cat([embeds, extra_bos], dim=1)  # (1, L+1, dim)

    if _trace_shapes():
        shape_logger.info("[vllm_t3_bridge] build_prompt_embeds.after_extra_bos")
        shape_logger.info("  embeds %s %s %s  (= cond + text + 1xBOS(in_prepare) + 1xBOS(extra, speech_pos=0))", tuple(embeds.shape), embeds.dtype, embeds.device)
        shape_logger.info("  total prompt seq_len sent to vLLM: %s", embeds.shape[1])

    _maybe_sync(prompt_builder_t3.device)
    prepare_s = time.perf_counter() - prepare_start

    cpu_copy_start = time.perf_counter()
    prompt_embeds = embeds.squeeze(0).detach().cpu()
    _maybe_sync(prompt_builder_t3.device)
    cpu_copy_s = time.perf_counter() - cpu_copy_start
    if not return_metadata:
        return prompt_embeds

    prompt_speech_tokens = getattr(t3_cond, "cond_prompt_speech_tokens", None)
    prompt_speech_token_len = 0 if prompt_speech_tokens is None else int(prompt_speech_tokens.shape[-1])
    text_token_len = int(text_tokens.shape[-1])
    # +1 for the extra BOS appended above to match native T3 inference contract.
    initial_speech_len = int(initial_speech.shape[-1]) + 1
    prompt_embed_seq_len = int(prompt_embeds.shape[0])
    prompt_embed_hidden_size = int(prompt_embeds.shape[1]) if prompt_embeds.ndim >= 2 else 0
    cond_seq_len = prompt_embed_seq_len - text_token_len - initial_speech_len
    metadata = {
        "prompt_speech_token_len": prompt_speech_token_len,
        "text_token_len": text_token_len,
        "initial_speech_len": initial_speech_len,
        "prompt_embed_seq_len": prompt_embed_seq_len,
        "prompt_embed_hidden_size": prompt_embed_hidden_size,
        "cond_seq_len": cond_seq_len,
        "prompt_embed_cond_to_device_s": cond_to_device_s,
        "prompt_embed_prepare_s": prepare_s,
        "prompt_embed_cpu_s": cpu_copy_s,
        "prompt_embed_total_s": time.perf_counter() - total_start,
    }
    return prompt_embeds, metadata
