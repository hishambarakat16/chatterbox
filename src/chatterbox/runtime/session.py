from copy import deepcopy
from dataclasses import dataclass, field
from uuid import uuid4

import torch

from ..models.t3.modules.cond_enc import T3Cond
from ..mtl_tts import Conditionals
from .types import GenerationOptions, StreamingCaches


def _clone_value(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_value(inner) for key, inner in value.items()}
    return deepcopy(value)


def clone_t3_cond(t3_cond: T3Cond) -> T3Cond:
    return T3Cond(**{key: _clone_value(value) for key, value in t3_cond.__dict__.items()})


def clone_conditionals(conds: Conditionals) -> Conditionals:
    return Conditionals(
        t3=clone_t3_cond(conds.t3),
        gen={key: _clone_value(value) for key, value in conds.gen.items()},
    )


def apply_exaggeration(conds: Conditionals, exaggeration: float, device: str) -> Conditionals:
    current = conds.t3.emotion_adv
    if current is not None and float(exaggeration) == float(current.view(-1)[0].item()):
        return conds

    updated = clone_t3_cond(conds.t3)
    updated.emotion_adv = exaggeration * torch.ones(1, 1, 1)
    conds.t3 = updated.to(device=device)
    return conds


@dataclass
class StreamingSession:
    conditionals: Conditionals
    options: GenerationOptions = field(default_factory=GenerationOptions)
    session_id: str = field(default_factory=lambda: uuid4().hex)
    caches: StreamingCaches = field(default_factory=StreamingCaches)
    profile: dict = field(default_factory=dict)

    def clone_conditionals(self) -> Conditionals:
        return clone_conditionals(self.conditionals)
