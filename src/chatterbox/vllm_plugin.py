from __future__ import annotations

from vllm import ModelRegistry

from .vllm_t3_bridge import VLLM_T3_ARCHITECTURE


def register_vllm_models() -> None:
    """Register Chatterbox custom vLLM models in every spawned worker process."""
    ModelRegistry.register_model(
        VLLM_T3_ARCHITECTURE,
        "chatterbox.vllm_t3_model:ChatterboxT3ForCausalLM",
    )
