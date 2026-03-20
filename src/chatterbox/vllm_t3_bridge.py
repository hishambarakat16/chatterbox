from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from .models.t3.llama_configs import LLAMA_CONFIGS
from .models.t3.modules.t3_config import T3Config


VLLM_T3_ARCHITECTURE = "ChatterboxT3ForCausalLM"
BASE_T3_FILENAME = "t3_mtl23ls_v2.safetensors"
HYDRA_HEADS_FILENAME = "t3_hydra_heads.safetensors"
REPO_ID = "ResembleAI/chatterbox"


def get_conditioning_seq_len(hp: T3Config | None = None) -> int:
    hp = hp or T3Config.multilingual()
    cond_len = 1  # speaker embedding
    if hp.use_perceiver_resampler:
        cond_len += 32
    else:
        cond_len += int(hp.speech_cond_prompt_len)
    if hp.emotion_adv:
        cond_len += 1
    return cond_len


def get_vllm_prompt_layout(hp: T3Config | None = None) -> dict[str, int]:
    hp = hp or T3Config.multilingual()
    text_token_offset = int(hp.speech_tokens_dict_size)
    conditioning_token_id = int(text_token_offset + hp.text_tokens_dict_size)
    return {
        "speech_vocab_size": int(hp.speech_tokens_dict_size),
        "text_vocab_size": int(hp.text_tokens_dict_size),
        "text_token_offset": text_token_offset,
        "conditioning_token_id": conditioning_token_id,
        "conditioning_seq_len": int(get_conditioning_seq_len(hp)),
        "input_vocab_size": int(conditioning_token_id + 1),
    }


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
    prompt_layout = get_vllm_prompt_layout(hp)
    cfg = dict(LLAMA_CONFIGS[hp.llama_config_name])
    cfg.update(
        {
            "architectures": [VLLM_T3_ARCHITECTURE],
            "vocab_size": prompt_layout["input_vocab_size"],
            "bos_token_id": hp.start_speech_token,
            "eos_token_id": hp.stop_speech_token,
            "pad_token_id": hp.stop_speech_token,
            "chatterbox_text_vocab_size": hp.text_tokens_dict_size,
            "chatterbox_speech_vocab_size": hp.speech_tokens_dict_size,
            "chatterbox_input_vocab_size": prompt_layout["input_vocab_size"],
            "chatterbox_text_token_offset": prompt_layout["text_token_offset"],
            "chatterbox_conditioning_token_id": prompt_layout["conditioning_token_id"],
            "chatterbox_conditioning_seq_len": prompt_layout["conditioning_seq_len"],
            "chatterbox_start_text_token": hp.start_text_token,
            "chatterbox_stop_text_token": hp.stop_text_token,
            "chatterbox_start_speech_token": hp.start_speech_token,
            "chatterbox_stop_speech_token": hp.stop_speech_token,
            "chatterbox_text_pos_embeddings": hp.max_text_tokens + 2,
            "chatterbox_speech_pos_embeddings": hp.max_speech_tokens + 4,
            "chatterbox_prompt_embeds_required": False,
            "chatterbox_cfg_supported": False,
            "chatterbox_hydra_supported": False,
            "chatterbox_pos_strategy": "approx_absolute_speech_positions",
            "tie_word_embeddings": False,
        }
    )
    return cfg


def export_vllm_t3_tokenizer(output_dir: str | Path, *, config: dict | None = None) -> None:
    config = config or build_vllm_t3_config(T3Config.multilingual())
    output_dir = Path(output_dir)

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    vocab_size = int(config["vocab_size"])
    vocab = {f"tok_{token_id}": token_id for token_id in range(vocab_size)}
    tokenizer_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="tok_0"))
    tokenizer_obj.pre_tokenizer = WhitespaceSplit()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        bos_token=f"tok_{int(config['bos_token_id'])}",
        eos_token=f"tok_{int(config['eos_token_id'])}",
        pad_token=f"tok_{int(config['pad_token_id'])}",
        unk_token="tok_0",
    )
    tokenizer.model_max_length = int(config.get("max_position_embeddings", 2048))
    tokenizer.save_pretrained(output_dir)


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
            "The served vLLM model reconstructs the T3 prompt internally from token ids and conditioning inputs.",
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
    export_vllm_t3_tokenizer(output_dir, config=config)

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
    enable_chunked_prefill: bool = True,
):
    LLM, _, _ = optional_import_vllm()
    register_vllm_t3_model()
    return LLM(
        model=str(model_dir),
        skip_tokenizer_init=False,
        enable_prompt_embeds=False,
        trust_remote_code=False,
        hf_overrides={"architectures": [VLLM_T3_ARCHITECTURE]},
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        dtype=dtype,
        max_model_len=max_model_len,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_mm_embeds=True,
    )


def make_sampling_params(*, options, hp: T3Config):
    _, _, SamplingParams = optional_import_vllm()
    return SamplingParams(
        temperature=float(options.temperature),
        top_p=float(options.top_p),
        min_p=float(options.min_p),
        repetition_penalty=float(options.repetition_penalty),
        max_tokens=int(options.max_new_tokens),
        stop_token_ids=[int(hp.stop_speech_token)],
        detokenize=False,
        skip_special_tokens=False,
    )


def prepare_vllm_text_tokens(*, tokenizer, text: str, language_id: str | None, device: str) -> torch.Tensor:
    normalized = punc_norm(text)
    text_tokens = tokenizer.text_to_tokens(
        normalized,
        language_id=language_id.lower() if language_id else None,
    ).to(device)
    return torch.nn.functional.pad(
        torch.nn.functional.pad(
            text_tokens,
            (1, 0),
            value=int(T3Config.multilingual().start_text_token),
        ),
        (0, 1),
        value=int(T3Config.multilingual().stop_text_token),
    )


def build_vllm_prompt(
    *,
    t3_cond,
    text_tokens: torch.Tensor,
    return_metadata: bool = False,
) -> dict | tuple[dict, dict]:
    hp = T3Config.multilingual()
    layout = get_vllm_prompt_layout(hp)

    text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device="cpu")
    text_token_ids = text_tokens.squeeze(0).tolist()
    prompt_token_ids = [
        int(layout["conditioning_token_id"]),
        *[int(layout["text_token_offset"] + token_id) for token_id in text_token_ids],
        int(hp.start_speech_token),
        int(hp.start_speech_token),
    ]

    conditioning = {
        "speaker_emb": t3_cond.speaker_emb.detach().cpu(),
        "cond_prompt_speech_tokens": None
        if getattr(t3_cond, "cond_prompt_speech_tokens", None) is None
        else t3_cond.cond_prompt_speech_tokens.detach().cpu(),
        "emotion_adv": None
        if getattr(t3_cond, "emotion_adv", None) is None
        else t3_cond.emotion_adv.detach().cpu(),
    }
    prompt = {
        "prompt_token_ids": prompt_token_ids,
        "multi_modal_data": {"conditioning": conditioning},
    }
    if not return_metadata:
        return prompt

    prompt_speech_tokens = getattr(t3_cond, "cond_prompt_speech_tokens", None)
    prompt_speech_token_len = 0 if prompt_speech_tokens is None else int(prompt_speech_tokens.shape[-1])
    text_token_len = int(text_tokens.shape[-1])
    initial_speech_len = 2
    prompt_seq_len = int(layout["conditioning_seq_len"] + text_token_len + initial_speech_len)
    metadata = {
        "prompt_speech_token_len": prompt_speech_token_len,
        "text_token_len": text_token_len,
        "initial_speech_len": initial_speech_len,
        "prompt_seq_len": prompt_seq_len,
        "prompt_hidden_size": int(hp.n_channels),
        "cond_seq_len": int(layout["conditioning_seq_len"]),
        "prompt_token_len_before_mm": int(len(prompt_token_ids)),
        "conditioning_token_id": int(layout["conditioning_token_id"]),
        "text_token_offset": int(layout["text_token_offset"]),
    }
    return prompt, metadata
