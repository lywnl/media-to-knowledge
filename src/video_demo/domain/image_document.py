from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from video_demo.domain.base import FrozenModel, Probability, Sha256, StableId


class ImageSourceEvidence(FrozenModel):
    evidence_type: Literal["IMAGE_SOURCE"] = "IMAGE_SOURCE"
    evidence_id: StableId
    relative_path: str = Field(min_length=1, max_length=1024)
    mime_type: Literal["image/jpeg", "image/png", "image/webp"]
    sha256: Sha256
    width: int = Field(gt=0, le=20_000)
    height: int = Field(gt=0, le=20_000)
    size_bytes: int = Field(gt=0, le=8 * 1024 * 1024)


class ImageContentBlock(FrozenModel):
    content_type: Literal["TEXT", "DESCRIPTION", "TABLE", "DIAGRAM"]
    text: str = Field(min_length=1, max_length=16_000)
    evidence_refs: tuple[StableId, ...] = Field(min_length=1, max_length=16)


class ImageClaim(FrozenModel):
    text: str = Field(min_length=1, max_length=2_000)
    evidence_refs: tuple[StableId, ...] = Field(min_length=1, max_length=16)
    certainty: Probability


class ImageDocument(FrozenModel):
    title: str = Field(min_length=1, max_length=200)
    overview_zh: str = Field(max_length=8_000)
    content_blocks: tuple[ImageContentBlock, ...] = Field(max_length=64)
    claims: tuple[ImageClaim, ...] = Field(max_length=64)
    evidence_refs: tuple[StableId, ...] = Field(max_length=128)
    content_status: Literal["GROUNDED", "DEGRADED"] = "GROUNDED"

    @model_validator(mode="after")
    def validate_evidence_closure(self) -> ImageDocument:
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("图片文档 evidence_refs 不得重复")
        refs = set(self.evidence_refs)
        if any(not set(block.evidence_refs).issubset(refs) for block in self.content_blocks):
            raise ValueError("图片内容引用必须属于文档证据闭包")
        if any(not set(claim.evidence_refs).issubset(refs) for claim in self.claims):
            raise ValueError("图片结论引用必须属于文档证据闭包")
        if self.content_status == "GROUNDED" and not refs:
            raise ValueError("图片文档至少需要一条证据")
        return self


class ImageUnderstandingResult(FrozenModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    run_id: StableId
    asset_sha256: Sha256
    document: ImageDocument
    source: ImageSourceEvidence
