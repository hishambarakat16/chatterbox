from pathlib import Path
import os
from types import SimpleNamespace

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors

from .models.s3gen import S3Gen
from .models.t3.modules.t3_config import T3Config
from .models.tokenizers import MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .mtl_tts import Conditionals, REPO_ID, SUPPORTED_LANGUAGES
from .runtime import GenerationOptions, StreamingSession
from .runtime.worker_vllm import ChatterboxMultilingualVllmWorker
from .vllm_t3_bridge import (
    create_vllm_engine,
    export_vllm_t3_model,
    resolve_base_t3_checkpoint_dir,
)


TURBO_REPO_ID = "ResembleAI/chatterbox-turbo"


def _resolve_map_location(device: str):
    if device in ["cpu", "mps"]:
        return torch.device("cpu")
    return None


def _resolve_turbo_s3_checkpoint_dir(turbo_s3_checkpoint_dir: str | None) -> Path:
    if turbo_s3_checkpoint_dir:
        return Path(turbo_s3_checkpoint_dir)
    return Path(
        snapshot_download(
            repo_id=TURBO_REPO_ID,
            allow_patterns=["s3gen_meanflow.safetensors"],
            token=os.getenv("HF_TOKEN") or True,
        )
    )


class ChatterboxMultilingualVllmTurboS3TTS:
    def __init__(self, worker: ChatterboxMultilingualVllmWorker):
        self.sr = worker.sr
        self.worker = worker
        self.device = worker.device

    @classmethod
    def get_supported_languages(cls):
        return SUPPORTED_LANGUAGES.copy()

    @classmethod
    def from_local(
        cls,
        ckpt_dir,
        device,
        *,
        base_checkpoint_dir: str | None = None,
        turbo_s3_checkpoint_dir: str | None = None,
        vllm_model_dir: str | None = None,
        vllm_export_dir: str | None = None,
        vllm_tensor_parallel_size: int = 1,
        vllm_gpu_memory_utilization: float = 0.5,
        vllm_enforce_eager: bool = False,
        vllm_dtype: str = "auto",
        vllm_max_model_len: int = 2048,
        vllm_enable_prefix_caching: bool = False,
        vllm_enable_chunked_prefill: bool = True,
        vllm_export_copy: bool = False,
    ) -> "ChatterboxMultilingualVllmTurboS3TTS":
        ckpt_dir = Path(ckpt_dir)
        base_ckpt_dir = resolve_base_t3_checkpoint_dir(
            ckpt_dir,
            base_checkpoint_dir=base_checkpoint_dir,
        )
        turbo_s3_dir = _resolve_turbo_s3_checkpoint_dir(turbo_s3_checkpoint_dir)
        map_location = _resolve_map_location(device)

        ve = VoiceEncoder()
        ve.load_state_dict(
            torch.load(base_ckpt_dir / "ve.pt", map_location=map_location, weights_only=True)
        )
        ve.to(device).eval()

        s3gen = S3Gen(meanflow=True)
        s3gen.load_state_dict(
            load_safetensors(turbo_s3_dir / "s3gen_meanflow.safetensors"),
            strict=True,
        )
        s3gen.to(device).eval()

        tokenizer = MTLTokenizer(str(base_ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"))

        default_conds = None
        if (builtin_voice := base_ckpt_dir / "conds.pt").exists():
            default_conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        if vllm_model_dir is None:
            vllm_model_dir = export_vllm_t3_model(
                ckpt_dir,
                output_dir=vllm_export_dir,
                base_checkpoint_dir=base_ckpt_dir,
                use_symlink=(not vllm_export_copy),
            )

        engine_kwargs = {
            "model_dir": vllm_model_dir,
            "tensor_parallel_size": vllm_tensor_parallel_size,
            "gpu_memory_utilization": vllm_gpu_memory_utilization,
            "enforce_eager": vllm_enforce_eager,
            "dtype": vllm_dtype,
            "max_model_len": vllm_max_model_len,
            "enable_prefix_caching": vllm_enable_prefix_caching,
            "enable_chunked_prefill": vllm_enable_chunked_prefill,
        }
        vllm_engine = create_vllm_engine(**engine_kwargs)

        worker = ChatterboxMultilingualVllmWorker(
            vllm_engine=vllm_engine,
            s3gen=s3gen,
            ve=ve,
            tokenizer=tokenizer,
            device=device,
            default_conds=default_conds,
            t3=SimpleNamespace(hp=T3Config.multilingual()),
        )
        return cls(worker)

    @classmethod
    def from_pretrained(
        cls,
        device: torch.device,
        *,
        turbo_s3_checkpoint_dir: str | None = None,
        vllm_model_dir: str | None = None,
        vllm_export_dir: str | None = None,
        vllm_tensor_parallel_size: int = 1,
        vllm_gpu_memory_utilization: float = 0.5,
        vllm_enforce_eager: bool = False,
        vllm_dtype: str = "auto",
        vllm_max_model_len: int = 2048,
        vllm_enable_prefix_caching: bool = False,
        vllm_enable_chunked_prefill: bool = True,
        vllm_export_copy: bool = False,
    ) -> "ChatterboxMultilingualVllmTurboS3TTS":
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"

        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main",
                allow_patterns=[
                    "ve.pt",
                    "t3_mtl23ls_v2.safetensors",
                    "grapheme_mtl_merged_expanded_v1.json",
                    "conds.pt",
                    "Cangjie5_TC.json",
                ],
                token=os.getenv("HF_TOKEN"),
            )
        )
        return cls.from_local(
            ckpt_dir,
            device,
            turbo_s3_checkpoint_dir=turbo_s3_checkpoint_dir,
            vllm_model_dir=vllm_model_dir,
            vllm_export_dir=vllm_export_dir,
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            vllm_enforce_eager=vllm_enforce_eager,
            vllm_dtype=vllm_dtype,
            vllm_max_model_len=vllm_max_model_len,
            vllm_enable_prefix_caching=vllm_enable_prefix_caching,
            vllm_enable_chunked_prefill=vllm_enable_chunked_prefill,
            vllm_export_copy=vllm_export_copy,
        )

    def close(self):
        import gc
        import torch

        engine = getattr(self.worker, "vllm_engine", None)
        if engine is not None:
            # 1. Ask vLLM to shut down its EngineCore process cleanly.
            if hasattr(engine, "shutdown"):
                try:
                    engine.shutdown()
                except Exception:
                    pass
            # 2. Drop the engine reference so Python can GC its resources.
            try:
                self.worker.vllm_engine = None
            except Exception:
                pass
            del engine

        # 3. Destroy distributed process group if one was created (tp > 1 or
        #    vLLM initialised torch.distributed internally).
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

        # 4. Destroy vLLM model-parallel groups if the symbol exists
        #    (older vLLM versions; safe no-op on newer ones).
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception:
            pass

        # 5. Release any CUDA memory still cached by PyTorch's allocator.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def create_session(
        self,
        *,
        audio_prompt_path=None,
        language_id=None,
        exaggeration=0.5,
        cfg_weight=0.0,
        temperature=0.8,
        repetition_penalty=1.0,
        min_p=0.05,
        top_p=1.0,
        max_new_tokens=1000,
        auto_max_new_tokens=False,
        auto_max_new_tokens_cap=128,
        session_id=None,
    ) -> StreamingSession:
        options = GenerationOptions(
            language_id=language_id,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            auto_max_new_tokens=auto_max_new_tokens,
            auto_max_new_tokens_cap=auto_max_new_tokens_cap,
        )
        return self.worker.create_session(
            audio_prompt_path=audio_prompt_path,
            options=options,
            session_id=session_id,
        )

    def generate_with_session(
        self,
        session: StreamingSession,
        text: str,
        *,
        language_id=None,
        exaggeration=None,
        cfg_weight=None,
        temperature=None,
        repetition_penalty=None,
        min_p=None,
        top_p=None,
        max_new_tokens=None,
        auto_max_new_tokens=None,
        auto_max_new_tokens_cap=None,
    ) -> torch.Tensor:
        options = session.options.merged(
            language_id=language_id,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            auto_max_new_tokens=auto_max_new_tokens,
            auto_max_new_tokens_cap=auto_max_new_tokens_cap,
        )
        return self.worker.generate(session=session, text=text, options=options)

    def generate(
        self,
        text,
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.0,
        temperature=0.8,
        repetition_penalty=1.0,
        min_p=0.05,
        top_p=1.0,
        max_new_tokens=1000,
        auto_max_new_tokens=False,
        auto_max_new_tokens_cap=128,
        session: StreamingSession | None = None,
    ) -> torch.Tensor:
        if session is None:
            session = self.create_session(
                audio_prompt_path=audio_prompt_path,
                language_id=language_id,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                auto_max_new_tokens=auto_max_new_tokens,
                auto_max_new_tokens_cap=auto_max_new_tokens_cap,
            )

        return self.generate_with_session(
            session,
            text,
            language_id=language_id,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            auto_max_new_tokens=auto_max_new_tokens,
            auto_max_new_tokens_cap=auto_max_new_tokens_cap,
        )

    def generate_many_with_sessions(
        self,
        sessions: list[StreamingSession],
        texts: list[str],
        *,
        language_id=None,
        exaggeration=None,
        cfg_weight=None,
        temperature=None,
        repetition_penalty=None,
        min_p=None,
        top_p=None,
        max_new_tokens=None,
        auto_max_new_tokens=None,
        auto_max_new_tokens_cap=None,
    ) -> list[dict]:
        options_list = [
            session.options.merged(
                language_id=language_id,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                auto_max_new_tokens=auto_max_new_tokens,
                auto_max_new_tokens_cap=auto_max_new_tokens_cap,
            )
            for session in sessions
        ]
        return self.worker.generate_many(
            sessions=sessions,
            texts=texts,
            options_list=options_list,
        )

    def inspect_prompt_embed_with_session(
        self,
        session: StreamingSession,
        text: str,
        *,
        language_id=None,
        exaggeration=None,
        cfg_weight=None,
        temperature=None,
        repetition_penalty=None,
        min_p=None,
        top_p=None,
        max_new_tokens=None,
        auto_max_new_tokens=None,
        auto_max_new_tokens_cap=None,
    ) -> dict:
        options = session.options.merged(
            language_id=language_id,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            auto_max_new_tokens=auto_max_new_tokens,
            auto_max_new_tokens_cap=auto_max_new_tokens_cap,
        )
        return self.worker.inspect_prompt_embed(
            session=session,
            text=text,
            options=options,
        )

    def get_last_profile(self) -> dict:
        return self.worker.get_last_profile()
