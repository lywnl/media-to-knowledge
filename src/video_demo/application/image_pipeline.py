from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from video_demo.application.document_rendering import RenderedDocument
from video_demo.application.image_rendering import render_image_markdown
from video_demo.domain.base import stable_identifier
from video_demo.domain.image_document import (
    ImageDocument,
    ImageSourceEvidence,
    ImageUnderstandingResult,
)
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.image_probe import probe_image


class ImageAnalyzer(Protocol):
    def analyze(self, *, image_data_url: str, title_hint: str) -> ImageDocument: ...


@dataclass(frozen=True, slots=True)
class ImagePipelineOutcome:
    result: ImageUnderstandingResult
    document: RenderedDocument
    warnings: tuple[str, ...] = ()
    status: Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"] = "SUCCEEDED"


def run_image_pipeline(
    *,
    run_id: str,
    asset_sha256: str,
    source: Path,
    relative_path: str,
    mime_type: str,
    title_hint: str,
    analyzer: ImageAnalyzer,
    runtime_root: Path,
    max_image_bytes: int = 8 * 1024 * 1024,
) -> ImagePipelineOutcome:
    probe = probe_image(source, runtime_root=runtime_root, max_bytes=max_image_bytes)
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != asset_sha256:
        raise VideoDemoError(ErrorCode.IMAGE_DIGEST_MISMATCH, "图片对象摘要校验失败")
    width, height = probe.width, probe.height
    source_evidence = ImageSourceEvidence(
        evidence_id=stable_identifier("image_source", {"sha256": digest}),
        relative_path=relative_path,
        mime_type=probe.mime_type,
        sha256=digest,
        width=width,
        height=height,
        size_bytes=len(content),
    )
    image_data_url = f"data:{probe.mime_type};base64,{base64.b64encode(content).decode('ascii')}"
    try:
        document = analyzer.analyze(image_data_url=image_data_url, title_hint=title_hint)
    except VideoDemoError:
        raise
    document = _bind_source_evidence(document, source_evidence.evidence_id)
    result = ImageUnderstandingResult(
        run_id=run_id,
        asset_sha256=asset_sha256,
        document=document,
        source=source_evidence,
    )
    return ImagePipelineOutcome(
        result=result,
        document=render_image_markdown(result),
    )


def _bind_source_evidence(document: ImageDocument, evidence_id: str) -> ImageDocument:
    blocks = tuple(
        block.model_copy(update={"evidence_refs": (evidence_id,)})
        for block in document.content_blocks
    )
    claims = tuple(
        claim.model_copy(update={"evidence_refs": (evidence_id,)})
        for claim in document.claims
    )
    return document.model_copy(
        update={
            "evidence_refs": (evidence_id,),
            "content_blocks": blocks,
            "claims": claims,
        },
    )
