from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch


def save_wav(path: str | Path, wav, sample_rate: int) -> None:
    target = Path(path)
    audio = wav
    if torch.is_tensor(audio):
        audio = audio.detach().cpu()
        if audio.ndim == 2:
            if audio.shape[0] == 1:
                audio = audio.squeeze(0)
            else:
                audio = audio.transpose(0, 1)
        audio = audio.numpy()
    sf.write(str(target), audio, sample_rate)
