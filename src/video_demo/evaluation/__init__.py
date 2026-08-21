"""离线质量评测、阈值和可审计报告。

具体 runner 和证据类型从其定义模块导入，避免导入包时构造 live/真实媒体依赖环。
"""

from __future__ import annotations

import importlib

_PUBLIC_EXPORTS = {
    "ArtifactRole": ("video_demo.evaluation.evidence", "ArtifactRole"),
    "EvidenceStore": ("video_demo.evaluation.evidence", "EvidenceStore"),
    "LiveValidationRunner": (
        "video_demo.evaluation.live_runner",
        "LiveValidationRunner",
    ),
    "RealMediaRunner": ("video_demo.evaluation.media_runner", "RealMediaRunner"),
    "build_verified_gate_check": (
        "video_demo.evaluation.evidence",
        "build_verified_gate_check",
    ),
    "load_machine_evidence": (
        "video_demo.evaluation.evidence",
        "load_machine_evidence",
    ),
    "sha256_file": ("video_demo.evaluation.evidence", "sha256_file"),
}

__all__ = list(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> object:
    target = _PUBLIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value
