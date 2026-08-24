from __future__ import annotations

import hashlib
import json
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_demo.api.app import create_app
from video_demo.config import Settings
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.evaluation import gate as gate_module
from video_demo.evaluation.durability import (
    DurabilityProbe,
    DurabilityRunner,
    DurabilitySample,
    DurabilitySampler,
)
from video_demo.evaluation.evidence import EvidenceStore, build_verified_gate_check
from video_demo.evaluation.final_runner import (
    cleanup_evaluation_run,
    stage_evaluation_run_id,
)
from video_demo.evaluation.report import GateStatus


@pytest.fixture(autouse=True)
def _copy_durability_implementation(tmp_path: Path) -> None:
    """让临时工作区具备 verifier 用来计算当前实现摘要的真实源码。"""

    project_root = Path(__file__).resolve().parents[2]
    for relative in gate_module._DURABILITY_IMPLEMENTATION_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project_root / relative, target)


def test_durability_implementation_digest_covers_subtitle_first_pipeline(
    tmp_path: Path,
) -> None:
    changed_path = Path("src/video_demo/media/subtitles.py")
    assert changed_path in gate_module._DURABILITY_IMPLEMENTATION_FILES
    before = gate_module._current_durability_implementation_sha256(tmp_path)
    copied_source = tmp_path / changed_path
    copied_source.write_bytes(copied_source.read_bytes() + b"\n")

    assert gate_module._current_durability_implementation_sha256(tmp_path) != before


class ConstantSampler:
    """受控测试专用采样器，避免伪造 psutil 的内部接口。"""

    def sample(self, *_args: object, **_kwargs: object) -> tuple[int, int, int]:
        return 1, 1, 0


def _matching_probe(_path: Path, sample: object) -> DurabilityProbe:
    return DurabilityProbe(
        duration_ms=sample.duration_ms,  # type: ignore[attr-defined]
        width=sample.width,  # type: ignore[attr-defined]
        height=sample.height,  # type: ignore[attr-defined]
    )


def _failed_sample(**updates: object) -> DurabilitySample:
    values: dict[str, object] = {
        "elapsed_seconds": 1.0,
        "peak_rss_bytes": 1,
        "peak_disk_bytes": 1,
        "oom_detected": False,
        "peak_concurrency": 1,
        "outside_workspace_write_count": 0,
        "terminal_status": "FAILED",
        "failure_code": "CONTROLLED_FAILURE",
    }
    values.update(updates)
    return DurabilitySample.model_validate(values)


def _successful_metric_sample(**updates: object) -> DurabilitySample:
    values: dict[str, object] = {
        "elapsed_seconds": 1.0,
        "peak_rss_bytes": 1,
        "peak_disk_bytes": 1,
        "oom_detected": False,
        "peak_concurrency": 1,
        "outside_workspace_write_count": 0,
        "terminal_status": "SUCCEEDED",
        "failure_code": None,
    }
    values.update(updates)
    return DurabilitySample.model_validate(values)


def _write_manifest(
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    write_authorization: bool = True,
) -> Path:
    runtime = tmp_path / ".codex" / "video-rag-demo"
    eval_root = runtime / "eval"
    eval_root.mkdir(parents=True)
    encoded_rows: list[str] = []
    for row in rows:
        media = eval_root / "durability" / str(row["media_relative_path"])
        media.parent.mkdir(parents=True, exist_ok=True)
        media.write_bytes(str(row.get("content", media.name)).encode())
        row = dict(row)
        row["media_sha256"] = hashlib.sha256(media.read_bytes()).hexdigest()
        row.pop("content", None)
        encoded_rows.append(json.dumps(row, ensure_ascii=False))
    path = eval_root / "durability" / "dataset.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(encoded_rows) + "\n", encoding="utf-8")
    if write_authorization:
        authorization = {
            "schema_version": "1.0.0",
            "records": [
                {
                    "schema_version": "1.0.0",
                    "authorization_id": row["authorization_id"],
                    "source_category": "OWNED",
                    "allowed_purposes": ["VIDEO_QUALITY_EVALUATION"],
                    "confirmed_at": "2026-08-20T00:00:00Z",
                    "media_sha256": [row["media_sha256"]],
                }
                for row in map(json.loads, encoded_rows)
            ],
        }
        (eval_root / "durability" / "authorization.json").write_text(
            json.dumps(authorization, ensure_ascii=False),
            encoding="utf-8",
        )
    return path


def _row(name: str, *, duration_ms: int = 1_800_000, width: int = 1920) -> dict[str, object]:
    return {
        "sample_id": f"sample_{name}",
        "media_relative_path": f"{name}.mp4",
        "duration_ms": duration_ms,
        "width": width,
        "height": 1080,
        "authorization_id": f"authorization_{name}",
        "content": name,
    }


def test_manifest_with_fewer_than_two_samples_is_not_run_before_execution(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path, [_row("a")])
    settings = Settings(workspace_root=tmp_path)
    runner = DurabilityRunner(settings, EvidenceStore(tmp_path, settings.runtime_root))

    check = runner.run(manifest, evaluation_run_id="durability_one")

    assert check.status == GateStatus.NOT_RUN
    assert "两段" in (check.not_run_reason or "")
    assert "psutil" not in (check.not_run_reason or "").lower()
    assert not (settings.runtime_root / "runs").exists()


def test_duplicate_media_digest_is_not_run_before_execution(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, [_row("a"), _row("b", width=1920)])
    payload = json.loads(manifest.read_text(encoding="utf-8").splitlines()[1])
    first = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    payload["media_sha256"] = first["media_sha256"]
    manifest.write_text(
        "\n".join(json.dumps(item) for item in (first, payload)) + "\n",
        encoding="utf-8",
    )
    settings = Settings(workspace_root=tmp_path)

    check = DurabilityRunner(
        settings, EvidenceStore(tmp_path, settings.runtime_root)
    ).run(manifest, evaluation_run_id="durability_duplicate")

    assert check.status == GateStatus.NOT_RUN


@pytest.mark.parametrize(
    ("rows", "expected"),
    (
        ([_row("a", duration_ms=1_799_999), _row("b")], "M1_DURATION_TOO_SHORT"),
        ([_row("a", width=1919), _row("b")], "M1_RESOLUTION_TOO_SMALL"),
        ([_row("a"), _row("b"), _row("c")], "M1_SAMPLE_COUNT_INVALID"),
    ),
)
def test_invalid_sample_count_duration_or_resolution_is_not_run_before_execution(
    tmp_path: Path,
    rows: list[dict[str, object]],
    expected: str,
) -> None:
    manifest = _write_manifest(tmp_path, rows)
    settings = Settings(workspace_root=tmp_path)

    check = DurabilityRunner(
        settings,
        EvidenceStore(tmp_path, settings.runtime_root),
    ).run(manifest, evaluation_run_id=f"durability_{len(rows)}")

    assert check.status == GateStatus.NOT_RUN
    preflight = json.loads(
        (
            settings.runtime_root
            / f"eval/reports/durability_{len(rows)}/preflight.json"
        ).read_text(encoding="utf-8")
    )
    assert expected in {issue["code"] for issue in preflight["issues"]}


def test_missing_or_non_covering_authorization_is_not_run_before_execution(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(
        tmp_path,
        [_row("a"), _row("b")],
        write_authorization=False,
    )
    settings = Settings(workspace_root=tmp_path)

    check = DurabilityRunner(
        settings,
        EvidenceStore(tmp_path, settings.runtime_root),
    ).run(manifest, evaluation_run_id="durability_no_authorization")

    assert check.status == GateStatus.NOT_RUN
    assert not (settings.runtime_root / "runs").exists()


def test_authorization_that_does_not_cover_media_is_not_run(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, [_row("a"), _row("b")])
    authorization_path = manifest.with_name("authorization.json")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["records"][0]["media_sha256"] = ["0" * 64]
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    settings = Settings(workspace_root=tmp_path)

    check = DurabilityRunner(
        settings,
        EvidenceStore(tmp_path, settings.runtime_root),
    ).run(manifest, evaluation_run_id="durability_non_covering_authorization")

    assert check.status == GateStatus.NOT_RUN
    assert not (settings.runtime_root / "runs").exists()


def test_manifest_path_outside_runtime_or_through_symlink_is_not_run(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / ".codex/video-rag-demo"
    runtime.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    external_manifest = _write_manifest(outside, [_row("a"), _row("b")])
    linked = runtime / "eval/durability"
    linked.parent.mkdir(parents=True)
    linked.symlink_to(external_manifest.parent, target_is_directory=True)
    settings = Settings(workspace_root=tmp_path)

    direct = DurabilityRunner(
        settings,
        EvidenceStore(tmp_path, settings.runtime_root),
    ).run(external_manifest, evaluation_run_id="durability_external_manifest")
    linked_check = DurabilityRunner(
        settings,
        EvidenceStore(tmp_path, settings.runtime_root),
    ).run(linked / "dataset.jsonl", evaluation_run_id="durability_linked_manifest")

    assert direct.status == GateStatus.NOT_RUN
    assert linked_check.status == GateStatus.NOT_RUN
    assert not (settings.runtime_root / "runs").exists()


def test_manifest_digest_tampering_is_not_run_before_execution(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, [_row("a"), _row("b")])
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[0]["media_sha256"] = "0" * 64
    manifest.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    settings = Settings(workspace_root=tmp_path)

    check = DurabilityRunner(
        settings,
        EvidenceStore(tmp_path, settings.runtime_root),
    ).run(manifest, evaluation_run_id="durability_tampered_manifest")

    assert check.status == GateStatus.NOT_RUN
    assert not (settings.runtime_root / "runs").exists()


def test_probe_facts_must_match_manifest_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path, [_row("a"), _row("b")])
    settings = Settings(workspace_root=tmp_path)
    monkeypatch.setattr("video_demo.evaluation.durability.psutil", object())

    def mismatching_probe(_path: Path, sample: object) -> DurabilityProbe:
        return DurabilityProbe(
            duration_ms=sample.duration_ms - 1,  # type: ignore[attr-defined]
            width=sample.width,  # type: ignore[attr-defined]
            height=sample.height,  # type: ignore[attr-defined]
        )

    executed: list[str] = []
    check = DurabilityRunner(
        settings,
        EvidenceStore(tmp_path, settings.runtime_root),
        probe_media=mismatching_probe,
        execute_sample=lambda sample: executed.append(sample.sample_id),  # type: ignore[arg-type,return-value]
    ).run(manifest, evaluation_run_id="durability_probe_mismatch")

    assert check.status == GateStatus.NOT_RUN
    assert executed == []


def test_non_cpu_int8_single_worker_is_not_run_before_execution(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, [_row("a"), _row("b")])
    settings = Settings(workspace_root=tmp_path, worker_concurrency=2)
    executed: list[str] = []

    check = DurabilityRunner(
        settings,
        EvidenceStore(tmp_path, settings.runtime_root),
        execute_sample=lambda sample: executed.append(sample.sample_id),  # type: ignore[arg-type,return-value]
    ).run(manifest, evaluation_run_id="durability_wrong_settings")

    assert check.status == GateStatus.NOT_RUN
    assert executed == []


def test_sample_gate_is_evaluated_independently_and_sampler_errors_are_not_swallowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path, [_row("a"), _row("b")])
    settings = Settings(workspace_root=tmp_path)
    monkeypatch.setattr("video_demo.evaluation.durability.psutil", object())

    class ExplodingSampler(DurabilitySampler):
        def sample(self, *_args: object, **_kwargs: object) -> object:
            raise RuntimeError("采样线程失败")

    runner = DurabilityRunner(
        settings,
        EvidenceStore(tmp_path, settings.runtime_root),
        sampler=ExplodingSampler(),
        probe_media=_matching_probe,
        execute_sample=lambda _sample: DurabilitySample(
            elapsed_seconds=1.0,
            peak_rss_bytes=1,
            peak_disk_bytes=1,
            oom_detected=False,
            peak_concurrency=1,
            outside_workspace_write_count=0,
            terminal_status="SUCCEEDED",
            failure_code=None,
        ),
    )

    with pytest.raises(RuntimeError, match="采样线程失败"):
        runner.run(manifest, evaluation_run_id="durability_sampler_error")


def test_background_sampler_error_is_rethrown_on_main_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path, [_row("a"), _row("b")])
    settings = Settings(workspace_root=tmp_path)
    monkeypatch.setattr("video_demo.evaluation.durability.psutil", object())
    calls = 0
    failed = threading.Event()

    class BackgroundExplodingSampler:
        def sample(self, *_args: object, **_kwargs: object) -> tuple[int, int, int]:
            nonlocal calls
            calls += 1
            if calls == 2:
                failed.set()
                raise RuntimeError("后台采样失败")
            return 1, 1, 0

    def execute(_sample: object) -> DurabilitySample:
        assert failed.wait(timeout=1)
        return _failed_sample()

    with pytest.raises(RuntimeError, match="后台采样失败"):
        DurabilityRunner(
            settings,
            EvidenceStore(tmp_path, settings.runtime_root),
            sampler=BackgroundExplodingSampler(),
            probe_media=_matching_probe,
            execute_sample=execute,
        ).run(manifest, evaluation_run_id="durability_background_sampler_error")


def test_memory_error_is_classified_as_out_of_memory_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path, [_row("a"), _row("b")])
    settings = Settings(workspace_root=tmp_path)
    monkeypatch.setattr("video_demo.evaluation.durability.psutil", object())

    def run_out_of_memory(_sample: object) -> DurabilitySample:
        raise VideoDemoError(ErrorCode.OUT_OF_MEMORY, "任务内存不足")

    check = DurabilityRunner(
        settings,
        EvidenceStore(tmp_path, settings.runtime_root),
        sampler=ConstantSampler(),
        probe_media=_matching_probe,
        execute_sample=run_out_of_memory,
    ).run(manifest, evaluation_run_id="durability_out_of_memory")

    assert check.status == GateStatus.FAIL
    raw = json.loads(
        (
            settings.runtime_root
            / "eval/reports/durability_out_of_memory/raw.json"
        ).read_text(encoding="utf-8")
    )
    assert all(sample["oom_detected"] is True for sample in raw["samples"])
    assert all(
        sample["failure_code"] == ErrorCode.OUT_OF_MEMORY for sample in raw["samples"]
    )


def test_periodic_sampler_captures_peak_between_execution_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path, [_row("a"), _row("b")])
    settings = Settings(workspace_root=tmp_path)
    monkeypatch.setattr("video_demo.evaluation.durability.psutil", object())
    calls = 0
    peak_observed = threading.Event()

    class MidExecutionPeakSampler:
        def sample(self, *_args: object, **_kwargs: object) -> tuple[int, int, int]:
            nonlocal calls
            calls += 1
            if calls == 2:
                peak_observed.set()
                return 99, 88, 0
            return 1, 1, 0

    def execute(_sample: object) -> DurabilitySample:
        assert peak_observed.wait(timeout=1)
        return _failed_sample()

    runner = DurabilityRunner(
        settings,
        EvidenceStore(tmp_path, settings.runtime_root),
        sampler=MidExecutionPeakSampler(),
        probe_media=_matching_probe,
        execute_sample=execute,
    )

    check = runner.run(manifest, evaluation_run_id="durability_periodic_peak")

    assert check.status == GateStatus.FAIL
    raw = json.loads(
        (
            settings.runtime_root
            / "eval/reports/durability_periodic_peak/raw.json"
        ).read_text(encoding="utf-8")
    )
    assert raw["samples"][0]["peak_rss_bytes"] == 99
    assert raw["samples"][0]["peak_disk_bytes"] == 88


def test_python_audit_hook_records_write_outside_sample_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path, [_row("a"), _row("b")])
    settings = Settings(workspace_root=tmp_path)
    monkeypatch.setattr("video_demo.evaluation.durability.psutil", object())
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"

    def execute(_sample: object) -> DurabilitySample:
        outside.write_text("禁止写到样本工作区外", encoding="utf-8")
        return _failed_sample()

    try:
        check = DurabilityRunner(
            settings,
            EvidenceStore(tmp_path, settings.runtime_root),
            sampler=ConstantSampler(),
            probe_media=_matching_probe,
            execute_sample=execute,
        ).run(manifest, evaluation_run_id="durability_python_write_audit")
    finally:
        outside.unlink(missing_ok=True)

    assert check.status == GateStatus.FAIL
    raw = json.loads(
        (
            settings.runtime_root
            / "eval/reports/durability_python_write_audit/raw.json"
        ).read_text(encoding="utf-8")
    )
    assert all(
        sample["outside_workspace_write_count"] > 0 for sample in raw["samples"]
    )


def test_production_query_chain_checks_job_evidence_page_and_keyframe_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_demo.evaluation import durability as durability_module

    keyframe_bytes = b"verified-keyframe"
    keyframe_sha256 = hashlib.sha256(keyframe_bytes).hexdigest()
    responses = {
        "/api/kb/jobs/job_1": {
            "job_id": "job_1",
            "resource_id": "run_1",
            "status": "SUCCEEDED",
        },
        "/api/kb/knowledge-bases/evaluation/video-understanding-runs/run_1/result": {},
        "/api/kb/knowledge-bases/evaluation/video-understanding-runs/run_1/evidence": {
            "items": [
                {
                    "evidence_type": "KEYFRAME",
                    "evidence_id": "evidence_1",
                    "start_ms": 0,
                    "end_ms": 1,
                    "duration_ms": 1,
                    "keyframe_id": "keyframe_1",
                    "timestamp_ms": 0,
                    "mime_type": "image/jpeg",
                    "sha256": keyframe_sha256,
                    "perceptual_hash": "0" * 16,
                }
            ],
            "next_cursor": None,
        },
    }

    class Response:
        def __init__(
            self,
            payload: dict[str, object] | None = None,
            *,
            content: bytes = b"",
            content_type: str = "application/json",
        ) -> None:
            self.status_code = 200
            self._payload = payload or {}
            self.content = content
            self.headers = {"content-type": content_type}

        def json(self) -> dict[str, object]:
            return self._payload

    class Client:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get(self, url: str, **kwargs: object) -> Response:
            self.calls.append(url)
            if url.endswith("/keyframes/keyframe_1/content"):
                return Response(content=keyframe_bytes, content_type="image/jpeg")
            if url.endswith("/evidence"):
                params = kwargs.get("params")
                if isinstance(params, dict) and params.get("cursor") is None:
                    return Response({"items": [], "next_cursor": "page_2"})
            return Response(responses[url])

    result = object()
    manifest_evidence = (object(),)
    monkeypatch.setattr(durability_module, "_parse_api_result", lambda _payload: result)
    monkeypatch.setattr(
        durability_module,
        "_read_published_manifest",
        lambda *_args, **_kwargs: (
            b"manifest",
            SimpleNamespace(
                result=result,
                evidence=manifest_evidence,
                status="SUCCEEDED",
            ),
        ),
    )
    monkeypatch.setattr(
        durability_module,
        "_evidence_matches_api",
        lambda expected, actual: expected == manifest_evidence and len(actual) == 1,
        raising=False,
    )
    client = Client()

    durability_module._verify_production_queries(
        client,
        run_id="run_1",
        job_id="job_1",
        terminal="SUCCEEDED",
    )

    assert client.calls == [
        "/api/kb/jobs/job_1",
        "/api/kb/knowledge-bases/evaluation/video-understanding-runs/run_1/result",
        "/api/kb/knowledge-bases/evaluation/video-understanding-runs/run_1/evidence",
        "/api/kb/knowledge-bases/evaluation/video-understanding-runs/run_1/evidence",
        "/api/kb/knowledge-bases/evaluation/video-understanding-runs/run_1/keyframes/keyframe_1/content",
    ]


@pytest.mark.parametrize(
    ("overrides", "terminal_failure"),
    (
        ({"elapsed_seconds": 5_401.0}, False),
        ({"oom_detected": True}, False),
        ({"peak_concurrency": 2}, False),
        ({"outside_workspace_write_count": 1}, False),
        ({"terminal_status": "CANCELLED", "failure_code": "RUN_CANCELLED"}, True),
    ),
)
def test_each_sample_failure_condition_produces_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    terminal_failure: bool,
) -> None:
    manifest = _write_manifest(tmp_path, [_row("a"), _row("b")])
    settings = Settings(workspace_root=tmp_path)
    monkeypatch.setattr("video_demo.evaluation.durability.psutil", object())

    runner = DurabilityRunner(
        settings,
        EvidenceStore(tmp_path, settings.runtime_root),
        sampler=ConstantSampler(),
        probe_media=_matching_probe,
        execute_sample=lambda _sample: (
            _failed_sample(**overrides)
            if terminal_failure
            else _successful_metric_sample(**overrides)
        ),
    )

    check = runner.run(
        manifest,
        evaluation_run_id=f"durability_fail_{hash(frozenset(overrides.items())) & 0xffff}",
    )

    assert check.status == GateStatus.FAIL


def test_cleanup_removes_only_database_bound_durability_product_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cloud_asr_environment: None,
) -> None:
    manifest = _write_manifest(tmp_path, [_row("a"), _row("b")])
    settings = Settings(workspace_root=tmp_path)
    assert settings.runtime_root is not None
    stage_run_id = stage_evaluation_run_id("eval_cleanup_durability", "durability")
    app = create_app(settings)
    created: dict[str, tuple[str, str]] = {}
    from sqlalchemy import update

    from video_demo.persistence.models import (
        JobModel,
        JobStatus,
        RunStatusValue,
        VideoUnderstandingRunModel,
    )
    from video_demo.persistence.repositories import (
        JobRepository,
        Scope,
        VideoRunRepository,
    )

    with app.state.container.database.session() as session:
        scope = Scope("evaluation", "video-demo", "evaluation")
        rows = {
            row["sample_id"]: row
            for row in map(json.loads, manifest.read_text(encoding="utf-8").splitlines())
        }
        for sample in ("sample_a", "sample_b"):
            row = rows[sample]
            run_id = f"run_durability_{sample.removeprefix('sample_')}"
            job_id = f"job_durability_{sample.removeprefix('sample_')}"
            digest = hashlib.sha256(f"{stage_run_id}:{sample}".encode()).hexdigest()
            asset = VideoRunRepository(session).get_or_create_asset(
                scope=scope,
                asset_id=f"asset_{sample}",
                object_ref=f"object_{sample}",
                source_sha256=str(row["media_sha256"]),
            )
            VideoRunRepository(session).add(
                scope=scope,
                run_id=run_id,
                asset_id=asset.asset_id,
                object_ref=f"object_{sample}",
                idempotency_key=f"durability-{digest[:40]}",
                config_snapshot={},
            )
            JobRepository(session).enqueue_video_run(
                scope=scope,
                job_id=job_id,
                run_id=run_id,
            )
            created[sample] = (run_id, job_id)
        session.execute(
            update(VideoUnderstandingRunModel).values(
                status=RunStatusValue.FAILED,
                error_code="CONTROLLED_FAILURE",
            )
        )
        session.execute(
            update(JobModel).values(
                status=JobStatus.FAILED,
                error_code="CONTROLLED_FAILURE",
            )
        )
    monkeypatch.setattr("video_demo.evaluation.durability.psutil", object())

    def failed_with_bound_run(sample: object) -> DurabilitySample:
        run_id, job_id = created[sample.sample_id]  # type: ignore[attr-defined]
        return _failed_sample(production_run_id=run_id, job_id=job_id)

    check = DurabilityRunner(
        settings,
        EvidenceStore(tmp_path, settings.runtime_root),
        sampler=ConstantSampler(),
        probe_media=_matching_probe,
        execute_sample=failed_with_bound_run,
    ).run(manifest, evaluation_run_id=stage_run_id)
    assert check.status == GateStatus.FAIL
    scope_key = hashlib.sha256(
        b"evaluation\x00video-demo\x00evaluation"
    ).hexdigest()[:24]
    product_runs = tuple(
        settings.runtime_root / "runs" / scope_key / run_id
        for run_id, _job_id in created.values()
    )
    for path in product_runs:
        path.mkdir(parents=True, exist_ok=True)
        (path / "keep.txt").write_text("owned", encoding="utf-8")
    foreign = settings.runtime_root / "runs" / scope_key / "run_foreign"
    foreign.mkdir(parents=True)

    cleanup_evaluation_run(tmp_path, "eval_cleanup_durability")

    assert all(not path.exists() for path in product_runs)
    assert foreign.is_dir()


def test_published_durability_report_rejects_changed_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path, [_row("a"), _row("b")])
    settings = Settings(workspace_root=tmp_path)
    monkeypatch.setattr("video_demo.evaluation.durability.psutil", object())
    evaluation_run_id = "durability_changed_manifest"
    check = DurabilityRunner(
        settings,
        EvidenceStore(tmp_path, settings.runtime_root),
        sampler=ConstantSampler(),
        probe_media=_matching_probe,
        execute_sample=lambda _sample: _failed_sample(),
    ).run(manifest, evaluation_run_id=evaluation_run_id)
    assert check.status == GateStatus.FAIL
    manifest.write_bytes(manifest.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="无法形成可信门禁检查"):
        build_verified_gate_check(
            "m1_durability",
            settings.runtime_root
            / "eval/reports"
            / evaluation_run_id
            / "durability.json",
            workspace_root=tmp_path,
            settings=settings,
        )


def test_missing_psutil_is_specific_not_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path, [_row("a"), _row("b")])
    settings = Settings(workspace_root=tmp_path)
    monkeypatch.setattr("video_demo.evaluation.durability.psutil", None)

    check = DurabilityRunner(
        settings, EvidenceStore(tmp_path, settings.runtime_root)
    ).run(manifest, evaluation_run_id="durability_no_psutil")

    assert check.status == GateStatus.NOT_RUN
    assert "psutil" in (check.not_run_reason or "").lower()


def test_injected_successful_execution_cannot_publish_authoritative_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path, [_row("a"), _row("b")])
    settings = Settings(workspace_root=tmp_path)
    monkeypatch.setattr("video_demo.evaluation.durability.psutil", object())

    runner = DurabilityRunner(
        settings,
        EvidenceStore(tmp_path, settings.runtime_root),
        sampler=ConstantSampler(),
        probe_media=lambda _path, sample: DurabilityProbe(
            duration_ms=sample.duration_ms,
            width=sample.width,
            height=sample.height,
        ),
        execute_sample=lambda _sample: DurabilitySample(
            elapsed_seconds=1.0,
            peak_rss_bytes=1,
            peak_disk_bytes=1,
            oom_detected=False,
            peak_concurrency=1,
            outside_workspace_write_count=0,
            terminal_status="SUCCEEDED",
            failure_code=None,
        ),
    )

    with pytest.raises(VideoDemoError, match="受控"):
        runner.run(manifest, evaluation_run_id="durability_controlled_pass")
