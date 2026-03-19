from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import torch

from .models.t3.llama_configs import LLAMA_CONFIGS
from .models.t3.modules.t3_config import T3Config
from .mtl_tts import punc_norm


VLLM_T3_ARCHITECTURE = "ChatterboxT3ForCausalLM"


def optional_import_vllm():
    try:
        from vllm import LLM, ModelRegistry, SamplingParams
    except ImportError as exc:
        raise ImportError(
            "vLLM is not installed in the active environment. "
            "Create the dedicated vLLM environment first and install `vllm`."
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


def export_vllm_t3_model(
    checkpoint_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    use_symlink: bool = True,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir) if output_dir is not None else Path(
        tempfile.mkdtemp(prefix="chatterbox_vllm_t3_")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_weights = checkpoint_dir / "t3_mtl23ls_v2.safetensors"
    if not source_weights.exists():
        raise FileNotFoundError(f"Missing T3 checkpoint: {source_weights}")

    config = build_vllm_t3_config(T3Config.multilingual())
    generation_config = {
        "bos_token_id": config["bos_token_id"],
        "eos_token_id": config["eos_token_id"],
        "pad_token_id": config["pad_token_id"],
    }
    export_meta = {
        "source_checkpoint_dir": str(checkpoint_dir),
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
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = False,
    dtype: str = "auto",
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


def build_prompt_embeds(
    *,
    prompt_builder_t3,
    t3_cond,
    text_tokens: torch.Tensor,
) -> torch.Tensor:
    text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=prompt_builder_t3.device)
    initial_speech = prompt_builder_t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    embeds, _ = prompt_builder_t3.prepare_input_embeds(
        t3_cond=t3_cond,
        text_tokens=text_tokens,
        speech_tokens=initial_speech,
        cfg_weight=0.0,
    )
    return embeds.squeeze(0).detach().cpu()
