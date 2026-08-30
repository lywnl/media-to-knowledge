from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from video_demo.domain.audio_plan import AudioDocumentConfig
from video_demo.domain.base import FrozenModel, LanguageCode
from video_demo.domain.speech_config import normalize_core_context, normalize_hotwords


class AudioRunConfig(FrozenModel):
    language_hints: tuple[LanguageCode, ...] = ()
    hotwords: tuple[str, ...] = Field(default=(), max_length=50)
    core_context: str | None = Field(default=None, max_length=1_000)
    document_config: AudioDocumentConfig = Field(default_factory=AudioDocumentConfig)
    result_schema_version: Literal["1.0.0"] = "1.0.0"

    @model_validator(mode="after")
    def normalize(self) -> Self:
        if len(self.language_hints) != len(set(self.language_hints)):
            raise ValueError("音频 language_hints 不得重复")
        object.__setattr__(self, "hotwords", normalize_hotwords(self.hotwords))
        object.__setattr__(self, "core_context", normalize_core_context(self.core_context))
        return self
