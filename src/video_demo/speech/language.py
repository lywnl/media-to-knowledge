from __future__ import annotations

from typing import Literal

from video_demo.domain.base import LanguageCode, Probability
from video_demo.domain.run import TimeRange


class LanguageSpan(TimeRange):
    evidence_id: str
    language: LanguageCode
    confidence: Probability | None = None
    detection_source: Literal["MODEL", "HINT"] = "MODEL"
    is_fully_evaluated_language: bool
