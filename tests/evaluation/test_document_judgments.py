from datetime import UTC, datetime

import pytest

from video_demo.evaluation.document_judgments import (
    ChapterDocumentJudgment,
    DocumentMetricObservation,
    DocumentQualityJudgment,
)


def _judgment(**updates: object) -> DocumentQualityJudgment:
    values: dict[str, object] = {
        "schema_version": "1.0.0",
        "evaluation_run_id": "eval_001",
        "sample_id": "sample_001",
        "annotation_sha256": "a" * 64,
        "dataset_sha256": "f" * 64,
        "authorization_sha256": "0" * 64,
        "prediction_index_sha256": "b" * 64,
        "result_sha256": "c" * 64,
        "evidence_sha256": "d" * 64,
        "document_sha256": "e" * 64,
        "reviewer": "reviewer_001",
        "reviewed_at": datetime(2026, 8, 27, tzinfo=UTC),
        "rubric_version": "document-quality-v1",
        "chapters": (
            ChapterDocumentJudgment(
                chapter_id="chapter_001",
                title_relevance="RELEVANT",
                summary_relevance="RELEVANT",
            ),
        ),
    }
    values.update(updates)
    return DocumentQualityJudgment(**values)


def test_document_judgment_requires_timezone_and_binds_required_fields() -> None:
    with pytest.raises(ValueError):
        _judgment(reviewed_at=datetime(2026, 8, 27))

    judgment = _judgment()
    assert judgment.evaluation_run_id == "eval_001"
    assert judgment.reviewed_at.tzinfo is not None


def test_chapter_judgment_requires_observation_refs_and_verdicts_in_same_order() -> None:
    with pytest.raises(ValueError, match="完整覆盖"):
        ChapterDocumentJudgment(
            chapter_id="chapter_001",
            title_relevance="RELEVANT",
            summary_relevance="RELEVANT",
            visual_observation_refs=("visual_001",),
        )

    with pytest.raises(ValueError, match="不得重复"):
        ChapterDocumentJudgment(
            chapter_id="chapter_001",
            title_relevance="RELEVANT",
            summary_relevance="RELEVANT",
            visual_observation_refs=("visual_001", "visual_001"),
            visual_supplement_judgments=(
                ("visual_001", "SUPPLEMENTS"),
                ("visual_001", "DUPLICATES"),
            ),
        )


def test_partial_metric_is_explicit_numeric_observation() -> None:
    metric = DocumentMetricObservation(value=0.5)
    assert metric.value == 0.5
    assert metric.not_run_reason is None
