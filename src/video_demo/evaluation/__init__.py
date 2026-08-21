"""离线质量评测、阈值和可审计报告。"""

from video_demo.evaluation.evidence import (
    ArtifactRole,
    EvidenceStore,
    build_verified_gate_check,
    load_machine_evidence,
    sha256_file,
)
from video_demo.evaluation.live_runner import LiveValidationRunner
from video_demo.evaluation.media_runner import RealMediaRunner

__all__ = [
    "ArtifactRole",
    "EvidenceStore",
    "LiveValidationRunner",
    "RealMediaRunner",
    "build_verified_gate_check",
    "load_machine_evidence",
    "sha256_file",
]
