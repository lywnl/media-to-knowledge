from __future__ import annotations

import hashlib
import json

from video_demo.domain.base import stable_identifier
from video_demo.domain.document_plan import FrameCandidateArtifact
from video_demo.domain.evidence import ChapterVisualObservation
from video_demo.integrations.document_port import ChapterVisionRequest, ChapterVisionResponse


def validate_chapter_vision_response(
    response: ChapterVisionResponse,
    request: ChapterVisionRequest,
    *,
    max_selected_frames: int,
    allowed_frames: tuple[str, ...] | None = None,
    allowed_targets: tuple[str, ...] | None = None,
    allowed_transcripts: tuple[str, ...] | None = None,
) -> None:
    """校验章节视觉响应对当前请求的全部引用和最终选图预算。"""

    if type(max_selected_frames) is not int or not 0 <= max_selected_frames <= 3:
        raise ValueError("max_selected_frames:invalid_budget")
    request = _validate_request_closure(
        request,
        allowed_frames,
        allowed_targets,
        allowed_transcripts,
    )
    try:
        response = ChapterVisionResponse.model_validate(response.model_dump(mode="python"))
    except (TypeError, ValueError) as error:
        raise ValueError("response:invalid") from error
    frame_by_id = {frame.frame_id: frame for frame in request.frames}
    target_ids = {target.target_id for target in request.targets}
    transcript_ids = {item.evidence_id for item in request.transcript_evidence}
    selected_ids: set[str] = set()
    observation_ids: set[str] = set()
    for observation in response.observations:
        _require_known_ids(
            observation.selected_frame_ids,
            frame_by_id,
            "observations.selected_frame_ids",
        )
        _require_known_ids(observation.target_ids, target_ids, "observations.target_ids")
        _require_known_ids(
            observation.transcript_evidence_refs,
            transcript_ids,
            "observations.transcript_evidence_refs",
        )
        covered_targets = _covered_targets(
            observation.selected_frame_ids,
            frame_by_id,
        )
        observation_targets = set(observation.target_ids)
        if not observation_targets <= covered_targets or any(
            not observation_targets.intersection(frame_by_id[frame_id].target_ids)
            for frame_id in observation.selected_frame_ids
        ):
            raise ValueError("observations.target_ids:frame_binding_mismatch")
        _validate_relations(observation, frame_by_id)
        observation_id = stable_identifier(
            "visual_observation_draft",
            observation.model_dump(mode="json"),
        )
        if observation_id in observation_ids:
            raise ValueError("observations:duplicate_equivalent")
        observation_ids.add(observation_id)
        selected_ids.update(observation.selected_frame_ids)
    if len(selected_ids) > max_selected_frames:
        raise ValueError("observations.selected_frame_ids:budget_exceeded")


def _validate_request_closure(
    request: ChapterVisionRequest,
    allowed_frames: tuple[str, ...] | None,
    allowed_targets: tuple[str, ...] | None,
    allowed_transcripts: tuple[str, ...] | None,
) -> ChapterVisionRequest:
    """在接收端重新建立请求闭包，避免 model_copy/缓存对象绕过模型校验。"""

    try:
        request_data = _without_computed_fields(
            request.model_dump(mode="python", exclude_computed_fields=True),
        )
        request = ChapterVisionRequest.model_validate(request_data)
    except (TypeError, ValueError) as error:
        raise ValueError("request:invalid") from error
    frame_ids = tuple(frame.frame_id for frame in request.frames)
    target_ids = tuple(target.target_id for target in request.targets)
    transcript_ids = tuple(item.evidence_id for item in request.transcript_evidence)
    _reject_duplicate_ids(frame_ids, "request.frames.frame_id")
    _reject_duplicate_ids(target_ids, "request.targets.target_id")
    _reject_duplicate_ids(transcript_ids, "request.transcript_evidence.evidence_id")
    target_id_set = set(target_ids)
    transcript_id_set = set(transcript_ids)
    for frame in request.frames:
        _require_known_ids(frame.target_ids, target_id_set, "request.frames.target_ids")
    for target in request.targets:
        _require_known_ids(
            target.anchor_evidence_refs,
            transcript_id_set,
            "request.targets.anchor_evidence_refs",
        )
    ordered_frames = tuple(
        frame.frame_id
        for frame in sorted(request.frames, key=lambda item: (item.timestamp_ms, item.frame_id))
    )
    _require_exact_ids(allowed_frames, ordered_frames, "allowed_frame_ids")
    _require_exact_ids(allowed_targets, target_ids, "allowed_target_ids")
    _require_exact_ids(
        allowed_transcripts,
        transcript_ids,
        "allowed_transcript_evidence_ids",
    )
    return request


def _require_exact_ids(
    actual: tuple[str, ...] | None,
    expected: tuple[str, ...],
    field: str,
) -> None:
    if actual is not None and actual != expected:
        raise ValueError(f"{field}:incomplete_or_extra")


def _reject_duplicate_ids(values: tuple[str, ...], field: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field}:duplicate")


def _without_computed_fields(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_computed_fields(item)
            for key, item in value.items()
            if key != "duration_ms"
        }
    if isinstance(value, (list, tuple)):
        return type(value)(_without_computed_fields(item) for item in value)
    return value


def chapter_vision_response_sha256(response: ChapterVisionResponse) -> str:
    """计算通过模型和语义校验后的规范章节视觉响应摘要。"""

    encoded = json.dumps(
        response.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _covered_targets(
    selected_frame_ids: tuple[str, ...],
    frame_by_id: dict[str, FrameCandidateArtifact],
) -> set[str]:
    return {
        target_id
        for frame_id in selected_frame_ids
        for target_id in frame_by_id[frame_id].target_ids
    }


def _validate_relations(
    observation: ChapterVisualObservation,
    frame_by_id: dict[str, FrameCandidateArtifact],
) -> None:
    for relation in observation.frame_relations:
        if (
            relation.from_frame_id not in frame_by_id
            or relation.to_frame_id not in frame_by_id
        ):
            raise ValueError("observations.frame_relations:unknown_reference")
        if (
            frame_by_id[relation.from_frame_id].timestamp_ms
            >= frame_by_id[relation.to_frame_id].timestamp_ms
        ):
            raise ValueError("observations.frame_relations:time_order_invalid")


def _require_known_ids(
    values: tuple[str, ...],
    allowed: dict[str, FrameCandidateArtifact] | set[str],
    field: str,
) -> None:
    if any(value not in allowed for value in values):
        raise ValueError(f"{field}:unknown_reference")
