from __future__ import annotations

from typing import get_args

from pydantic import TypeAdapter

import video_demo.domain.result as result_module
from video_demo.domain.evidence import EvidenceItem


def test_result_module_only_exposes_document_3_result_contract() -> None:
    legacy_names = {
        "SemanticFields",
        "SegmentUnderstanding",
        "SummaryUnderstanding",
        "VideoSegment",
        "SummaryChapter",
        "VideoSummary",
        "normalize_keyword_fields",
        "SUPPORTED_RESULT_SCHEMA_VERSIONS",
    }

    assert result_module.RESULT_SCHEMA_VERSION == "3.0.0"
    assert result_module.VideoUnderstandingResult.model_fields["schema_version"].default == "3.0.0"
    assert legacy_names.isdisjoint(vars(result_module))


def test_formal_evidence_union_excludes_scene_and_ocr() -> None:
    evidence_schema = TypeAdapter(EvidenceItem).json_schema()
    domain_discriminators = set(evidence_schema["discriminator"]["mapping"])

    assert domain_discriminators == {
        "ASR_SEGMENT",
        "SUBTITLE_CUE",
        "KEYFRAME",
        "VISUAL_OBSERVATION",
    }


def test_legacy_result_and_evidence_are_isolated_for_old_chain() -> None:
    from video_demo.domain.legacy_result import SegmentUnderstanding
    from video_demo.fusion.merge import WindowUnderstanding

    assert SegmentUnderstanding.__module__ == "video_demo.domain.legacy_result"
    annotation = WindowUnderstanding.model_fields["understanding"].annotation
    assert get_args(annotation)[0] is SegmentUnderstanding
