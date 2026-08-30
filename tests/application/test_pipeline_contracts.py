from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

import video_demo.application.pipeline as pipeline_module
import video_demo.application.pipeline_contracts as contracts
from video_demo.domain.document_artifact import MAX_METRIC_VALUE, MODEL_METRIC_NAMES
from video_demo.domain.evidence import (
    KeyframeEvidence,
    SceneBoundary,
    SpeechSegment,
    SubtitleCue,
    VisualObservationEvidence,
)
from video_demo.errors import ErrorCode, VideoDemoError


def _speech(evidence_id: str, start_ms: int, end_ms: int) -> SpeechSegment:
    return SpeechSegment(
        evidence_id=evidence_id,
        start_ms=start_ms,
        end_ms=end_ms,
        text=evidence_id,
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )


def _subtitle(evidence_id: str, start_ms: int, end_ms: int) -> SubtitleCue:
    return SubtitleCue(
        evidence_id=evidence_id,
        start_ms=start_ms,
        end_ms=end_ms,
        text=evidence_id,
        language="zh",
        stream_index=0,
    )


def _keyframe(evidence_id: str, timestamp_ms: int) -> KeyframeEvidence:
    return KeyframeEvidence(
        evidence_id=evidence_id,
        start_ms=timestamp_ms,
        end_ms=timestamp_ms + 1,
        keyframe_id=f"frame_{evidence_id}",
        timestamp_ms=timestamp_ms,
        relative_path=f"visual/keyframes/{evidence_id}.jpg",
        mime_type="image/jpeg",
        sha256="a" * 64,
        perceptual_hash="0123456789abcdef",
        size_bytes=1,
    )


def _observation(
    evidence_id: str,
    start_ms: int,
    end_ms: int,
) -> VisualObservationEvidence:
    return VisualObservationEvidence(
        evidence_id=evidence_id,
        start_ms=start_ms,
        end_ms=end_ms,
        chapter_id="chapter_001",
        target_ids=("target_001",),
        keyframe_refs=("frame_001",),
        visual_type="GENERAL",
        caption=evidence_id,
        relation_to_transcript="INDEPENDENT",
        certainty=0.9,
    )


def test_pipeline_only_reexports_cross_stage_contracts() -> None:
    names = (
        "PipelineContext",
        "PipelineOutcome",
        "PipelineRunConfig",
        "RegisteredAsset",
        "ProbedAsset",
        "PreparedMedia",
        "SpeechBoundaryCandidate",
        "SpeechAnalysis",
        "StageMetric",
    )

    for name in names:
        assert getattr(pipeline_module, name) is getattr(contracts, name)


def test_pipeline_run_config_requires_explicit_4_schema_in_historical_snapshot() -> None:
    with pytest.raises(VideoDemoError) as raised:
        contracts.pipeline_run_config_from_snapshot(
            {"language_hints": [], "hotwords": [], "core_context": None}
        )

    assert raised.value.code == ErrorCode.RESULT_SCHEMA_UNSUPPORTED


def test_pipeline_run_config_round_trips_document_configuration_and_schema() -> None:
    config = contracts.PipelineRunConfig.model_validate(
        {
            "language_hints": ["zh"],
            "document_config": {"document_title": "知识文档"},
            "result_schema_version": "4.1.0",
        }
    )

    assert config.document_config.document_title == "知识文档"
    assert config.result_schema_version == "4.1.0"
    assert contracts.pipeline_run_config_from_snapshot(config.model_dump(mode="json")) == config


def test_scene_index_digest_covers_all_canonical_fields() -> None:
    scene = SceneBoundary(
        evidence_id="scene_001",
        start_ms=0,
        end_ms=1_000,
        transition="candidate",
        score=0.9,
    )
    payload = {
        "proxy_sha256": "a" * 64,
        "duration_ms": 1_000,
        "frame_tolerance_ms": 40,
        "scenes": [scene.model_dump(mode="json")],
    }
    expected = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    ).hexdigest()
    index_type = contracts.SceneIndex

    index = index_type(
        index_sha256=expected,
        scenes=(scene,),
        **{key: payload[key] for key in payload if key != "scenes"},
    )

    assert index.index_sha256 == expected
    with pytest.raises(ValueError, match="摘要"):
        index_type.model_validate(
            {
                **index.model_dump(exclude_computed_fields=True),
                "index_sha256": "b" * 64,
            },
        )


def test_scene_index_rejects_non_contiguous_scene_timeline() -> None:
    scenes = (
        SceneBoundary(
            evidence_id="scene_001",
            start_ms=0,
            end_ms=400,
            transition="candidate",
            score=0.9,
        ),
        SceneBoundary(
            evidence_id="scene_002",
            start_ms=500,
            end_ms=1_000,
            transition="hard_cut",
            score=0.9,
        ),
    )
    digest = contracts.scene_index_sha256(
        proxy_sha256="a" * 64,
        duration_ms=1_000,
        frame_tolerance_ms=40,
        scenes=scenes,
    )

    with pytest.raises(ValueError, match="连续"):
        contracts.SceneIndex(
            proxy_sha256="a" * 64,
            duration_ms=1_000,
            frame_tolerance_ms=40,
            scenes=scenes,
            index_sha256=digest,
        )


def test_scene_index_allows_empty_visual_result_for_text_only_continuation() -> None:
    digest = contracts.scene_index_sha256(
        proxy_sha256="a" * 64,
        duration_ms=1_000,
        frame_tolerance_ms=40,
        scenes=(),
    )

    index = contracts.SceneIndex(
        proxy_sha256="a" * 64,
        duration_ms=1_000,
        frame_tolerance_ms=40,
        scenes=(),
        index_sha256=digest,
    )

    assert index.scenes == ()


def test_scene_index_rejects_duplicate_scene_ids() -> None:
    scenes = (
        SceneBoundary(
            evidence_id="scene_duplicate",
            start_ms=0,
            end_ms=500,
            transition="candidate",
            score=0.9,
        ),
        SceneBoundary(
            evidence_id="scene_duplicate",
            start_ms=500,
            end_ms=1_000,
            transition="hard_cut",
            score=0.8,
        ),
    )
    digest = contracts.scene_index_sha256(
        proxy_sha256="a" * 64,
        duration_ms=1_000,
        frame_tolerance_ms=40,
        scenes=scenes,
    )

    with pytest.raises(ValueError, match="标识"):
        contracts.SceneIndex(
            proxy_sha256="a" * 64,
            duration_ms=1_000,
            frame_tolerance_ms=40,
            scenes=scenes,
            index_sha256=digest,
        )


def test_speech_analysis_exposes_stable_transcript_views() -> None:
    analysis = contracts.SpeechAnalysis(
        transcript_source="ASR",
        evidence=(
            _speech("asr_late", 2_000, 3_000),
            _speech("asr_early", 0, 1_000),
        ),
    )

    assert [item.evidence_id for item in analysis.transcript_evidence] == [
        "asr_early",
        "asr_late",
    ]
    assert tuple(analysis.transcript_by_id) == ("asr_early", "asr_late")


def test_speech_analysis_rejects_duplicate_transcript_ids() -> None:
    with pytest.raises(VideoDemoError) as raised:
        contracts.SpeechAnalysis(
            transcript_source="ASR",
            evidence=(
                _speech("asr_duplicate", 0, 1_000),
                _speech("asr_duplicate", 1_000, 2_000),
            ),
        )

    assert raised.value.code == ErrorCode.DUPLICATE_EVIDENCE_ID


@pytest.mark.parametrize(
    ("transcript_source", "evidence"),
    [
        ("NONE", (_speech("asr_001", 0, 1_000),)),
        ("ASR", (_subtitle("subtitle_001", 0, 1_000),)),
        ("SUBTITLE", (_speech("asr_001", 0, 1_000),)),
    ],
)
def test_speech_analysis_rejects_mismatched_transcript_source(
    transcript_source: str,
    evidence: tuple[SpeechSegment | SubtitleCue, ...],
) -> None:
    with pytest.raises(VideoDemoError) as raised:
        contracts.SpeechAnalysis(  # type: ignore[arg-type]
            transcript_source=transcript_source,
            evidence=evidence,
        )

    assert raised.value.code == ErrorCode.EVIDENCE_TYPE_MISMATCH


def test_chapter_and_integration_modules_do_not_import_pipeline_dtos() -> None:
    source_root = Path(__file__).parents[2] / "src/video_demo"
    candidates = (
        *sorted((source_root / "application").glob("chapter_*.py")),
        *sorted((source_root / "integrations").glob("*.py")),
    )

    offenders: list[str] = []
    for path in candidates:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "video_demo.application.pipeline"
            ):
                offenders.append(path.name)
            if isinstance(node, ast.Import) and any(
                alias.name == "video_demo.application.pipeline" for alias in node.names
            ):
                offenders.append(path.name)

    assert offenders == []


def test_document_evidence_merge_uses_transcript_then_typed_visual_order() -> None:
    merge = contracts.stable_merge_document_evidence
    merged = merge(
        (
            _speech("asr_late", 2_000, 3_000),
            _subtitle("subtitle_early", 0, 500),
            _speech("asr_early", 0, 500),
        ),
        (
            _observation("observation_early", 0, 100),
            _keyframe("keyframe_late", 2_000),
            _keyframe("keyframe_early", 1_000),
            _observation("observation_late", 1_000, 2_000),
        ),
    )

    assert [item.evidence_id for item in merged] == [
        "asr_early",
        "subtitle_early",
        "asr_late",
        "keyframe_early",
        "keyframe_late",
        "observation_early",
        "observation_late",
    ]


def test_document_evidence_merge_rejects_duplicate_id_across_all_inputs() -> None:
    merge = contracts.stable_merge_document_evidence

    with pytest.raises(VideoDemoError) as raised:
        merge(
            (_speech("same_id", 0, 1_000),),
            (_keyframe("same_id", 500),),
        )

    assert raised.value.code == ErrorCode.DUPLICATE_EVIDENCE_ID


@pytest.mark.parametrize("invalid", [True, -1, 1.5])
def test_model_metric_merge_rejects_non_strict_or_negative_integers(invalid: object) -> None:
    merge = contracts.merge_model_metrics

    with pytest.raises(ValueError, match="严格整数"):
        merge({"vlm_logical_analyses": invalid})


def test_model_metric_merge_rejects_unknown_name_and_overflow() -> None:
    merge = contracts.merge_model_metrics

    with pytest.raises(ValueError, match="未知"):
        merge({"unknown_metric": 1})
    with pytest.raises(ValueError, match="溢出"):
        merge(
            {"vlm_logical_analyses": MAX_METRIC_VALUE},
            {"vlm_logical_analyses": 1},
        )


def test_model_metric_merge_sums_known_stage_subsets() -> None:
    merge = contracts.merge_model_metrics

    merged = merge(
        {"vlm_logical_analyses": 1},
        {"vlm_logical_analyses": 2, "vlm_cache_hits": 1},
    )

    assert set(merged) == MODEL_METRIC_NAMES
    assert merged["vlm_logical_analyses"] == 3
    assert merged["vlm_cache_hits"] == 1
    assert all(
        value == 0
        for name, value in merged.items()
        if name not in {"vlm_logical_analyses", "vlm_cache_hits"}
    )


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5])
def test_result_evidence_budget_rejects_non_positive_strict_integer(
    invalid: object,
) -> None:
    with pytest.raises(ValueError, match="正整数"):
        contracts.require_result_evidence_budget((), invalid)  # type: ignore[arg-type]


def test_result_evidence_budget_checks_actual_complete_closure() -> None:
    evidence = (
        _speech("asr_001", 0, 1_000),
        _keyframe("keyframe_001", 500),
    )

    contracts.require_result_evidence_budget(evidence, len(evidence))
    with pytest.raises(VideoDemoError) as raised:
        contracts.require_result_evidence_budget(evidence, len(evidence) - 1)

    assert raised.value.code == ErrorCode.INPUT_BUDGET_EXCEEDED


def test_result_evidence_budget_rejects_non_unique_closure() -> None:
    evidence = (
        _speech("same_id", 0, 1_000),
        _keyframe("same_id", 500),
    )

    with pytest.raises(VideoDemoError) as raised:
        contracts.require_result_evidence_budget(evidence, len(evidence))

    assert raised.value.code == ErrorCode.DUPLICATE_EVIDENCE_ID


def test_run_status_merge_propagates_partial_success() -> None:
    merge = contracts.merge_run_statuses

    assert merge("SUCCEEDED", "SUCCEEDED") == "SUCCEEDED"
    assert merge("SUCCEEDED", "PARTIAL_SUCCEEDED", "SUCCEEDED") == (
        "PARTIAL_SUCCEEDED"
    )


def test_warning_merge_deduplicates_by_first_stage_order_and_rejects_empty_code() -> None:
    merge = contracts.stable_merge_warnings

    assert merge(("A", "B"), ("B", "C"), ("A",)) == ("A", "B", "C")
    with pytest.raises(ValueError, match="空"):
        merge(("A", " "))
