from __future__ import annotations

import json
from collections.abc import Sequence

from video_demo.domain.base import stable_identifier
from video_demo.domain.evidence import EvidenceItem, TimelineEvidence
from video_demo.errors import ErrorCode, VideoDemoError


def canonicalize_evidence(evidence: Sequence[EvidenceItem]) -> tuple[EvidenceItem, ...]:
    """去除完全相同的输入，并拒绝同一稳定 ID 指向不同内容。"""

    by_id: dict[str, tuple[str, EvidenceItem]] = {}
    for item in evidence:
        digest_source = _canonical_json(item)
        existing = by_id.get(item.evidence_id)
        if existing is not None and existing[0] != digest_source:
            raise VideoDemoError(
                ErrorCode.DUPLICATE_EVIDENCE_ID,
                "同一证据 ID 对应了不同内容",
                {"evidence_id": item.evidence_id},
            )
        by_id[item.evidence_id] = (digest_source, item)
    ordered = sorted(
        (entry[1] for entry in by_id.values()),
        key=lambda item: (
            item.start_ms,
            item.end_ms,
            item.evidence_type,
            item.evidence_id,
        ),
    )
    return tuple(ordered)


def build_timeline(evidence: Sequence[EvidenceItem]) -> tuple[TimelineEvidence, ...]:
    canonical = canonicalize_evidence(evidence)
    grouped: dict[tuple[int, int], list[str]] = {}
    for item in canonical:
        grouped.setdefault((item.start_ms, item.end_ms), []).append(item.evidence_id)

    timeline = tuple(
        TimelineEvidence(
            timeline_id=stable_identifier(
                "timeline",
                {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "evidence_refs": refs,
                },
            ),
            start_ms=start_ms,
            end_ms=end_ms,
            evidence_refs=tuple(refs),
        )
        for (start_ms, end_ms), refs in sorted(grouped.items())
    )
    validate_timeline(timeline, canonical)
    return timeline


def validate_timeline(
    timeline: Sequence[TimelineEvidence],
    evidence: Sequence[EvidenceItem],
) -> None:
    evidence_by_id = {
        item.evidence_id: item
        for item in canonicalize_evidence(evidence)
    }
    timeline_ids: set[str] = set()
    for item in timeline:
        if item.timeline_id in timeline_ids:
            raise ValueError("timeline_id 不得重复")
        timeline_ids.add(item.timeline_id)
        for evidence_ref in item.evidence_refs:
            referenced = evidence_by_id.get(evidence_ref)
            if referenced is None:
                raise VideoDemoError(
                    ErrorCode.UNKNOWN_EVIDENCE_REFERENCE,
                    "时间轴引用了不存在的证据",
                    {"timeline_id": item.timeline_id, "evidence_id": evidence_ref},
                )
            if not item.contains(referenced):
                raise VideoDemoError(
                    ErrorCode.EVIDENCE_OUTSIDE_SEGMENT,
                    "时间轴引用的证据超出条目范围",
                    {"timeline_id": item.timeline_id, "evidence_id": evidence_ref},
                )


def _canonical_json(item: EvidenceItem) -> str:
    return json.dumps(
        item.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
