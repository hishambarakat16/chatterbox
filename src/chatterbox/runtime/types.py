from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GenerationOptions:
    language_id: Optional[str] = None
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    repetition_penalty: float = 2.0
    min_p: float = 0.05
    top_p: float = 1.0
    max_new_tokens: int = 1000

    def merged(self, **overrides) -> "GenerationOptions":
        data = self.__dict__.copy()
        for key, value in overrides.items():
            if value is not None:
                data[key] = value
        return GenerationOptions(**data)


@dataclass
class StreamingCaches:
    t3: dict[str, Any] = field(default_factory=dict)
    s3: dict[str, Any] = field(default_factory=dict)
    vocoder: dict[str, Any] = field(default_factory=dict)
