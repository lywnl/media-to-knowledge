from __future__ import annotations

from typing import Literal

from pydantic import Field, StrictInt

from video_demo.domain.base import FrozenModel, Sha256
from video_demo.domain.document import TranscriptSource, VideoUnderstandingResult
from video_demo.domain.evidence import DocumentEvidenceItem


class DocumentArtifactPayload(FrozenModel):
    artifact_schema_version: Literal["3.0.0"] = "3.0.0"
    result: VideoUnderstandingResult
    evidence: tuple[DocumentEvidenceItem, ...]
    stage_metrics: dict[str, StrictInt]
    stage_cache_hits: tuple[str, ...] = ()
    status: Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"]
    warnings: tuple[str, ...]
    transcript_source: TranscriptSource
    document_sha256: Sha256
    document_size_bytes: int = Field(gt=0, le=16 * 1024 * 1024)
