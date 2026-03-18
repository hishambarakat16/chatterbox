from pathlib import Path
import os

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors

from .models.s3gen import S3Gen
from .models.t3 import T3
from .models.t3.modules.t3_config import T3Config
from .models.t3.train import load_hydra_heads_from_checkpoint
from .models.tokenizers import MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .mtl_tts import Conditionals, REPO_ID, SUPPORTED_LANGUAGES
from .runtime import GenerationOptions, StreamingSession
from .runtime.worker_scheduled import ChatterboxMultilingualScheduledWorker


class ChatterboxMultilingualScheduledTTS:
    def __init__(self, worker: ChatterboxMultilingualScheduledWorker):
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
        batching_window_ms: float = 5.0,
        text_bucket_width: int = 1,
        enable_alignment_controller: bool = False,
        hydra_checkpoint_dir: str | None = None,
        hydra_speculate_k: int = 3,
    ) -> "ChatterboxMultilingualScheduledTTS":
        ckpt_dir = Path(ckpt_dir)

        if device in ["cpu", "mps"]:
            map_location = torch.device("cpu")
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(torch.load(ckpt_dir / "ve.pt", map_location=map_location, weights_only=True))
        ve.to(device).eval()

        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(ckpt_dir / "t3_mtl23ls_v2.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(torch.load(ckpt_dir / "s3gen.pt", map_location=map_location, weights_only=True))
        s3gen.to(device).eval()

        tokenizer = MTLTokenizer(str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"))

        default_conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            default_conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        hydra_model = None
        if hydra_checkpoint_dir:
            hydra_model = load_hydra_heads_from_checkpoint(
                base_t3=t3,
                checkpoint_dir=hydra_checkpoint_dir,
                freeze_base=True,
            )

        worker = ChatterboxMultilingualScheduledWorker(
            t3=t3,
            s3gen=s3gen,
            ve=ve,
            tokenizer=tokenizer,
            device=device,
            default_conds=default_conds,
            batching_window_ms=batching_window_ms,
            text_bucket_width=text_bucket_width,
            enable_alignment_controller=enable_alignment_controller,
            hydra_model=hydra_model,
            hydra_speculate_k=hydra_speculate_k,
        )
        return cls(worker)

    @classmethod
    def from_pretrained(
        cls,
        device: torch.device,
        *,
        batching_window_ms: float = 5.0,
        text_bucket_width: int = 1,
        enable_alignment_controller: bool = False,
        hydra_checkpoint_dir: str | None = None,
        hydra_speculate_k: int = 3,
    ) -> "ChatterboxMultilingualScheduledTTS":
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
                    "s3gen.pt",
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
            batching_window_ms=batching_window_ms,
            text_bucket_width=text_bucket_width,
            enable_alignment_controller=enable_alignment_controller,
            hydra_checkpoint_dir=hydra_checkpoint_dir,
            hydra_speculate_k=hydra_speculate_k,
        )

    def create_session(
        self,
        *,
        audio_prompt_path=None,
        language_id=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
        max_new_tokens=1000,
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
        )
        return self.worker.generate(session=session, text=text, options=options)

    def generate(
        self,
        text,
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
        max_new_tokens=1000,
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
        )
