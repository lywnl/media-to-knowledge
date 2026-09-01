from __future__ import annotations

from pathlib import Path

import pytest

import video_demo.evaluation.gate as gate_module
from video_demo.evaluation.gate import FINAL_GATE_CHECKS, GateCheck
from video_demo.evaluation.report import GateStatus


def test_final_gate_contains_only_active_live_checks() -> None:
    assert "chapter_vlm_live" in FINAL_GATE_CHECKS
    assert "five_language_models" in FINAL_GATE_CHECKS
    assert "baidu_ocr_live" not in FINAL_GATE_CHECKS
    assert "qwen_live" not in FINAL_GATE_CHECKS


def test_live_implementation_digest_is_deterministic(tmp_path: Path) -> None:
    for relative_path in gate_module._LIVE_IMPLEMENTATION_FILES:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative_path.as_posix().encode())
    first = gate_module._current_live_implementation_sha256(tmp_path)
    second = gate_module._current_live_implementation_sha256(tmp_path)
    assert first == second
    assert len(first) == 64


def test_live_implementation_digest_requires_all_sources(tmp_path: Path) -> None:
    required = next(iter(gate_module._LIVE_IMPLEMENTATION_FILES))
    for relative_path in gate_module._LIVE_IMPLEMENTATION_FILES:
        if relative_path == required:
            continue
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"source")
    with pytest.raises(ValueError, match="实现文件"):
        gate_module._current_live_implementation_sha256(tmp_path)


def test_gate_check_requires_evidence_for_pass() -> None:
    with pytest.raises(ValueError):
        GateCheck(check_id="chapter_vlm_live", status=GateStatus.PASS)


def test_gate_check_requires_reason_for_not_run() -> None:
    with pytest.raises(ValueError):
        GateCheck(check_id="chapter_vlm_live", status=GateStatus.NOT_RUN)


def test_mp3_magic_rejects_truncated_id3_header() -> None:
    assert gate_module._has_mp3_signature(b"ID3x", 4) is False
    assert gate_module._has_mp3_signature(b"ID3\x04\x00\x00\x00\x00\x00\x00", 10) is True
