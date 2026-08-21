from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import re
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from video_demo.application.composition import (
    ProductionDiagnosticComponents,
    ProductionModelIdentityReport,
    build_production_model_identity_report,
)
from video_demo.config import Settings
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.evaluation.annotations import load_evaluation_package
from video_demo.evaluation.evidence import (
    BaiduLiveRawReport,
    EvidenceStore,
    FiveLanguageModelsRawReport,
    LiveExecutionSummary,
    PreflightRawReport,
    PyannoteLiveRawReport,
    QwenLiveRawReport,
)
from video_demo.evaluation.report import GateStatus


def _production_diagnostics(
    settings: Settings,
    *,
    ocr_client: object | None = None,
    qwen_client: object | None = None,
    speech_models: object | None = None,
) -> ProductionDiagnosticComponents:
    class Resource:
        def close(self) -> None:
            return None

    baidu_http = Resource()
    qwen_http = Resource()
    visual_factory = SimpleNamespace(
        http_client=baidu_http,
        ocr_client=ocr_client or object(),
    )
    return ProductionDiagnosticComponents(
        ffmpeg_factory=lambda _cancel: object(),  # type: ignore[arg-type]
        speech_component_factory=lambda _media, _cancel: object(),  # type: ignore[arg-type]
        visual_component_factory=visual_factory,  # type: ignore[arg-type]
        speech_models=speech_models or SimpleNamespace(),  # type: ignore[arg-type]
        qwen_client=qwen_client or object(),  # type: ignore[arg-type]
        qwen_http_client=qwen_http,  # type: ignore[arg-type]
        model_identity_report=build_production_model_identity_report(settings),
        owned_resources=(),
    )


def test_live_validation_runner_exposes_four_check_entries() -> None:
    assert importlib.util.find_spec("video_demo.evaluation.live_runner") is not None
    module = importlib.import_module("video_demo.evaluation.live_runner")
    runner_type = getattr(module, "LiveValidationRunner", None)

    assert runner_type is not None
    assert {
        "run_baidu",
        "run_qwen",
        "run_pyannote",
        "run_local_model_stack",
    }.issubset(vars(runner_type))


def test_live_dependency_probe_keeps_tensorflow_hub_warning_off_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_demo.evaluation import live_runner as live_runner_module
    from video_demo.evaluation.live_runner import LiveValidationRunner

    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    runtime_root.mkdir(parents=True)
    runner = LiveValidationRunner(
        Settings(workspace_root=tmp_path),
        EvidenceStore(tmp_path, runtime_root),
    )

    def import_module(_name: str) -> object:
        warnings.warn(
            "pkg_resources is deprecated as an API. See upstream migration guidance.",
            UserWarning,
            stacklevel=2,
        )
        return object()

    monkeypatch.setattr(live_runner_module.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(live_runner_module.importlib, "import_module", import_module)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        assert runner._module_available("tensorflow_hub") is True

    assert captured == []


def _runner_package(
    tmp_path: Path,
    *,
    samples: tuple[tuple[str, str], ...] = (("sample-001", "zh"),),
):
    from video_demo.evaluation import gate as gate_module

    project_root = Path(__file__).resolve().parents[2]
    for relative_path in gate_module._LIVE_IMPLEMENTATION_FILES:
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project_root / relative_path, target)
    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    eval_root = runtime_root / "eval"
    manifest_records: list[dict[str, object]] = []
    authorization_records: list[dict[str, object]] = []
    for sample_id, language in samples:
        media = eval_root / "media" / f"{sample_id}.mp4"
        media.parent.mkdir(parents=True, exist_ok=True)
        media.write_bytes(f"authorized-media-{sample_id}".encode())
        media_sha256 = hashlib.sha256(media.read_bytes()).hexdigest()
        annotation_payload = {
            "schema_version": "1.0.0",
            "sample_id": sample_id,
            "media_sha256": media_sha256,
            "duration_ms": 1_000,
            "language": language,
            "reference_text": "你好",
            "words": [
                {
                    "word_id": "word-001",
                    "text": "你",
                    "start_ms": 0,
                    "end_ms": 500,
                }
            ],
            "speaker_turns": [
                {
                    "turn_id": "turn-001",
                    "speaker_id": "speaker-001",
                    "start_ms": 0,
                    "end_ms": 500,
                }
            ],
            "ocr_frames": [
                {
                    "frame_id": "frame-001",
                    "timestamp_ms": 100,
                    "text_lines": ["你好"],
                }
            ],
            "audio_events": [
                {
                    "event_id": "event-001",
                    "normalized_event": "speech",
                    "start_ms": 0,
                    "end_ms": 500,
                }
            ],
            "scene_boundaries_ms": [500],
            "semantic_boundaries_ms": [500],
            "supported_facts": [
                {"fact_id": "fact-001", "canonical_text": "问候"}
            ],
            "key_fact_ids": ["fact-001"],
            "known_people": [],
        }
        annotation = eval_root / "annotations" / f"{sample_id}.json"
        annotation.parent.mkdir(parents=True, exist_ok=True)
        annotation.write_text(
            json.dumps(annotation_payload, ensure_ascii=False),
            encoding="utf-8",
        )
        annotation_sha256 = hashlib.sha256(annotation.read_bytes()).hexdigest()
        authorization_id = f"auth-{sample_id}"
        manifest_records.append(
            {
                "sample_id": sample_id,
                "language": language,
                "authorization_id": authorization_id,
                "media_relative_path": f"media/{sample_id}.mp4",
                "media_sha256": media_sha256,
                "annotations_relative_path": f"annotations/{sample_id}.json",
                "annotations_sha256": annotation_sha256,
            }
        )
        authorization_records.append(
            {
                "schema_version": "1.0.0",
                "authorization_id": authorization_id,
                "source_category": "OWNED",
                "allowed_purposes": ["VIDEO_QUALITY_EVALUATION"],
                "confirmed_at": "2026-08-18T00:00:00Z",
                "media_sha256": [media_sha256],
            }
        )
        live_root = eval_root / "live" / "run-abc" / sample_id
        live_root.mkdir(parents=True, exist_ok=True)
        (live_root / "audio.wav").write_bytes(f"audio-{sample_id}".encode())
        (live_root / "keyframe.jpg").write_bytes(f"keyframe-{sample_id}".encode())
        (live_root / "clip.mp4").write_bytes(f"clip-{sample_id}".encode())
    manifest = eval_root / "dataset.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(record) for record in manifest_records),
        encoding="utf-8",
    )
    authorization = eval_root / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "records": authorization_records,
            }
        ),
        encoding="utf-8",
    )
    package = load_evaluation_package(
        manifest,
        authorization,
        workspace_root=tmp_path,
        runtime_root=runtime_root,
    )
    return package, runtime_root


def _write_local_stack_preflight_models(runtime_root: Path) -> None:
    model_root = runtime_root / "models"
    silero = model_root / "silero/model-id.txt"
    silero.parent.mkdir(parents=True, exist_ok=True)
    silero.write_bytes(b"silero-vad\n")
    faster_whisper = model_root / "faster-whisper"
    faster_whisper.mkdir(parents=True)
    for filename in (
        "config.json",
        "model.bin",
        "preprocessor_config.json",
        "tokenizer.json",
        "vocabulary.json",
    ):
        (faster_whisper / filename).write_bytes(b"model")
    for language in ("zh", "en", "ja", "ko", "es"):
        whisperx = model_root / "whisperx" / language
        whisperx.mkdir(parents=True)
        (whisperx / "model.bin").write_bytes(b"model")
    yamnet = model_root / "yamnet/saved_model/saved_model.pb"
    yamnet.parent.mkdir(parents=True)
    yamnet.write_bytes(b"model")
    (model_root / "yamnet/yamnet_class_map.csv").write_bytes(b"index,label\n0,speech\n")


def test_local_stack_preflight_rejects_incomplete_faster_whisper_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    _package, runtime_root = _runner_package(tmp_path)
    _write_local_stack_preflight_models(runtime_root)
    (runtime_root / "models/faster-whisper/tokenizer.json").unlink()
    runner = LiveValidationRunner(
        Settings(workspace_root=tmp_path),
        EvidenceStore(tmp_path, runtime_root),
    )
    monkeypatch.setattr(runner, "_module_available", lambda _name: True, raising=False)
    issues: list[ErrorCode] = []

    runner._collect_local_stack_environment_issues(issues)

    assert issues == [ErrorCode.FASTER_WHISPER_MODEL_UNAVAILABLE]


def _write_pyannote_preflight_files(runtime_root: Path, terms: bytes) -> None:
    model_root = runtime_root / "models/pyannote"
    model_root.mkdir(parents=True)
    (model_root / "terms-accepted.json").write_bytes(terms)
    (model_root / "model.bin").write_bytes(b"model")


def test_invalid_run_id_is_rejected_before_package_reverification(tmp_path: Path) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    runtime_root.mkdir(parents=True)
    store = EvidenceStore(tmp_path, runtime_root)
    runner = LiveValidationRunner(Settings(workspace_root=tmp_path), store)

    class ExplodingPackage:
        @property
        def dataset(self) -> object:
            raise AssertionError("非法 run ID 前不应访问 package")

    with pytest.raises(VideoDemoError) as raised:
        runner.run_baidu("../escape", ExplodingPackage())  # type: ignore[arg-type]

    assert raised.value.code == ErrorCode.INVALID_PATH_COMPONENT


def test_not_run_claim_does_not_create_real_media_marker_or_call_components(
    tmp_path: Path,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    store = EvidenceStore(tmp_path, runtime_root)
    calls: list[str] = []

    def build_components(_settings: Settings) -> object:
        calls.append("constructed")
        raise AssertionError("preflight 失败前不得构造生产组件")

    runner = LiveValidationRunner(
        Settings(workspace_root=tmp_path),
        store,
        components_factory=build_components,
    )
    check = runner.run_baidu("run-abc", package)

    assert check.status == GateStatus.NOT_RUN
    assert calls == []
    report_root = runtime_root / "eval" / "reports" / "run-abc"
    assert report_root.is_dir()
    assert not (report_root / ".real-media.incomplete").exists()


def test_existing_incomplete_run_is_an_idempotency_conflict(tmp_path: Path) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    report_root = runtime_root / "eval" / "reports" / "run-abc"
    report_root.mkdir(parents=True)
    (report_root / "unexpected.txt").write_text("partial", encoding="utf-8")
    runner = LiveValidationRunner(
        Settings(workspace_root=tmp_path),
        EvidenceStore(tmp_path, runtime_root),
    )

    with pytest.raises(VideoDemoError) as raised:
        runner.run_baidu("run-abc", package)

    assert raised.value.code == ErrorCode.IDEMPOTENCY_CONFLICT


def test_same_run_concurrent_claim_has_one_winner_and_stable_conflicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import video_demo.evaluation.live_runner as live_runner_module
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    settings = Settings(workspace_root=tmp_path)
    store = EvidenceStore(tmp_path, runtime_root)
    workers = 8
    barrier = Barrier(workers)
    real_reverify = live_runner_module.reverify_evaluation_package

    def synchronized_reverify(candidate: object) -> object:
        verified = real_reverify(candidate)  # type: ignore[arg-type]
        barrier.wait(timeout=5)
        return verified

    monkeypatch.setattr(
        live_runner_module,
        "reverify_evaluation_package",
        synchronized_reverify,
    )

    def run_once() -> GateStatus | ErrorCode:
        try:
            return LiveValidationRunner(settings, store).run_baidu(
                "run-abc",
                package,
            ).status
        except VideoDemoError as error:
            return error.code

    with ThreadPoolExecutor(max_workers=workers) as executor:
        outcomes = tuple(executor.map(lambda _index: run_once(), range(workers)))

    assert outcomes.count(GateStatus.NOT_RUN) == 1
    assert outcomes.count(ErrorCode.IDEMPOTENCY_CONFLICT) == workers - 1


@pytest.mark.parametrize(
    ("filename", "payload"),
    (
        ("preflight.json", b"ATTACKER-CONTENT"),
        ("unexpected.txt", b"UNDECLARED-CONTENT"),
    ),
)
def test_claim_rejects_concurrent_same_name_or_extra_entry_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    payload: bytes,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    store = EvidenceStore(tmp_path, runtime_root)
    real_claim = store.claim_exclusive_live_report_run

    def claim_and_inject(evaluation_run_id: str) -> object:
        session = real_claim(evaluation_run_id)
        report_root = runtime_root / "eval/reports" / evaluation_run_id
        (report_root / filename).write_bytes(payload)
        return session

    monkeypatch.setattr(store, "claim_exclusive_live_report_run", claim_and_inject)

    with pytest.raises(VideoDemoError) as raised:
        LiveValidationRunner(Settings(workspace_root=tmp_path), store).run_baidu(
            "run-abc",
            package,
        )

    report_root = runtime_root / "eval/reports/run-abc"
    assert raised.value.code == ErrorCode.IDEMPOTENCY_CONFLICT
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert (report_root / filename).read_bytes() == payload
    assert not (report_root / "baidu_ocr_live.json").exists()


def test_claimed_run_replacement_during_execution_never_writes_replacement(
    tmp_path: Path,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    settings = Settings(
        workspace_root=tmp_path,
        qwen_base_url="https://qwen.example/v1",
        qwen_api_key="qwen-secret",
        qwen_model_id="qwen3-vl-plus",
    )
    model = next(
        item
        for item in build_production_model_identity_report(settings).models
        if item.component == "qwen"
    )
    report_root = runtime_root / "eval/reports/run-abc"
    stolen_root = runtime_root / "eval/reports/run-abc-stolen"

    def execute(
        _check_id: str,
        samples: tuple[object, ...],
        _components: object,
        journal: object,
    ) -> None:
        sample = samples[0]
        journal.record_success(  # type: ignore[attr-defined]
            LiveExecutionSummary(
                schema_version="1.0.0",
                component="qwen",
                operation="capability_probe",
                evaluation_run_id="run-abc",
                model=model,
                sample_id=sample.sample_id,  # type: ignore[attr-defined]
                language=sample.language,  # type: ignore[attr-defined]
                input_kind="CLIP",
                input_sha256=sample.clip_sha256,  # type: ignore[attr-defined]
                request_id_sha256=hashlib.sha256(b"probe-request").hexdigest(),
                http_status=200,
                capabilities=("video_input", "strict_json_schema"),
                output_item_count=1,
            )
        )
        report_root.rename(stolen_root)
        report_root.mkdir()
        raise VideoDemoError(ErrorCode.QWEN_RESPONSE_INVALID, "受控响应失败")

    with pytest.raises(VideoDemoError) as raised:
        LiveValidationRunner(
            settings,
            EvidenceStore(tmp_path, runtime_root),
            components_factory=lambda _settings: object(),  # type: ignore[arg-type]
            execution_port=execute,
        ).run_qwen("run-abc", package)

    assert raised.value.code == ErrorCode.IDEMPOTENCY_CONFLICT
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert {path.name for path in report_root.iterdir()} == set()
    assert {path.name for path in stolen_root.iterdir()} == {"execution-000.json"}
    assert not (runtime_root / "eval/live-authority/run-abc/qwen_live.json").exists()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "已知延期：路径型 live verifier 存在目录 ABA 窗口；"
        "单进程 Demo 主流程不受影响，后续改为基于 writer fd 的同源验证"
    ),
)
def test_not_run_verification_rejects_directory_aba_during_entire_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import video_demo.evaluation.evidence as evidence_module
    from video_demo.evaluation.live_runner import LiveValidationRunner

    replacement_workspace = tmp_path / "replacement-workspace"
    replacement_package, replacement_runtime = _runner_package(replacement_workspace)
    LiveValidationRunner(
        Settings(
            workspace_root=replacement_workspace,
            baidu_api_key="replacement-key",
        ),
        EvidenceStore(replacement_workspace, replacement_runtime),
    ).run_baidu("run-abc", replacement_package)

    package, runtime_root = _runner_package(tmp_path)
    reports_root = runtime_root / "eval/reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    report_root = reports_root / "run-abc"
    replacement_root = reports_root / "run-abc-replacement"
    stolen_root = reports_root / "run-abc-stolen"
    (replacement_runtime / "eval/reports/run-abc").rename(replacement_root)

    real_builder = evidence_module._build_verified_gate_check
    attack_count = 0

    def build_during_directory_aba(*args: object, **kwargs: object) -> object:
        nonlocal attack_count
        if attack_count:
            return real_builder(*args, **kwargs)  # type: ignore[arg-type]
        attack_count += 1
        report_root.rename(stolen_root)
        replacement_root.rename(report_root)
        try:
            return real_builder(*args, **kwargs)  # type: ignore[arg-type]
        finally:
            report_root.rename(replacement_root)
            stolen_root.rename(report_root)

    monkeypatch.setattr(
        evidence_module,
        "_build_verified_gate_check",
        build_during_directory_aba,
    )

    with pytest.raises(VideoDemoError) as raised:
        LiveValidationRunner(
            Settings(workspace_root=tmp_path),
            EvidenceStore(tmp_path, runtime_root),
        ).run_baidu("run-abc", package)

    assert attack_count == 1
    assert raised.value.code == ErrorCode.IDEMPOTENCY_CONFLICT
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert {path.name for path in report_root.iterdir()} == {
        "preflight.json",
        "raw.json",
        "trace.stderr.txt",
        "trace.stdout.txt",
    }


@pytest.mark.parametrize("failure_mode", ("symlink", "non_directory", "oserror"))
def test_report_parent_failures_are_stable_idempotency_conflicts_without_causes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    import video_demo.evaluation.evidence as evidence_module
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    reports = runtime_root / "eval/reports"
    reports.parent.mkdir(parents=True, exist_ok=True)
    if failure_mode == "symlink":
        target = runtime_root / "eval/reports-target"
        target.mkdir()
        reports.symlink_to(target, target_is_directory=True)
    elif failure_mode == "non_directory":
        reports.write_text("not-a-directory", encoding="utf-8")
    else:
        monkeypatch.setattr(
            evidence_module,
            "_open_or_create_runtime_parent",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("sensitive-low-level-oserror")
            ),
        )

    with pytest.raises(VideoDemoError) as raised:
        LiveValidationRunner(
            Settings(workspace_root=tmp_path),
            EvidenceStore(tmp_path, runtime_root),
        ).run_baidu("run-abc", package)

    assert raised.value.code == ErrorCode.IDEMPOTENCY_CONFLICT
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert "sensitive-low-level-oserror" not in _exception_graph_text(raised.value)


def test_package_source_change_fails_before_preflight_persistence(tmp_path: Path) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    assert package.dataset_path is not None
    package.dataset_path.write_text("{}\n", encoding="utf-8")
    runner = LiveValidationRunner(
        Settings(workspace_root=tmp_path),
        EvidenceStore(tmp_path, runtime_root),
    )

    with pytest.raises(VideoDemoError) as raised:
        runner.run_baidu("run-abc", package)

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID
    assert not (runtime_root / "eval/reports/run-abc").exists()


@pytest.mark.parametrize("mismatch", ("workspace", "runtime"))
def test_runner_rejects_settings_store_root_mismatch_before_package_access(
    tmp_path: Path,
    mismatch: str,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    runtime_root = tmp_path / ".codex/video-rag-demo"
    runtime_root.mkdir(parents=True)
    store = EvidenceStore(tmp_path, runtime_root)
    if mismatch == "workspace":
        settings_root = tmp_path / "settings-workspace"
        settings_root.mkdir()
        settings = Settings(workspace_root=settings_root)
    else:
        settings = Settings(
            workspace_root=tmp_path,
            runtime_root=Path("runtime/other"),
        )

    class ExplodingPackage:
        @property
        def workspace_root(self) -> object:
            raise AssertionError("runner 配置根非法时不得访问 package")

    with pytest.raises(VideoDemoError) as raised:
        LiveValidationRunner(settings, store).run_baidu(
            "run-abc",
            ExplodingPackage(),  # type: ignore[arg-type]
        )

    assert raised.value.code == ErrorCode.INVALID_CONFIGURATION
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert not (runtime_root / "eval/reports").exists()


def test_runner_rejects_foreign_package_before_reverification_or_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import video_demo.evaluation.live_runner as live_runner_module
    from video_demo.evaluation.live_runner import LiveValidationRunner

    current_root = tmp_path / "current"
    current_runtime = current_root / ".codex/video-rag-demo"
    current_runtime.mkdir(parents=True)
    foreign_root = tmp_path / "foreign"
    foreign_package, _foreign_runtime = _runner_package(foreign_root)

    def reject_reverification(_package: object) -> object:
        raise AssertionError("foreign package 必须在重验来源前被拒绝")

    monkeypatch.setattr(
        live_runner_module,
        "reverify_evaluation_package",
        reject_reverification,
    )
    runner = LiveValidationRunner(
        Settings(workspace_root=current_root),
        EvidenceStore(current_root, current_runtime),
    )

    with pytest.raises(VideoDemoError) as raised:
        runner.run_baidu("run-abc", foreign_package)

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert not (current_runtime / "eval/reports").exists()


def test_complete_not_run_report_is_reverified_without_preflight_or_package_access(
    tmp_path: Path,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    store = EvidenceStore(tmp_path, runtime_root)
    settings = Settings(workspace_root=tmp_path)
    first = LiveValidationRunner(settings, store).run_baidu("run-abc", package)

    class ExplodingPackage:
        @property
        def dataset(self) -> object:
            raise AssertionError("完整报告重验不得重新访问调用方 package")

    second = LiveValidationRunner(
        settings,
        store,
        components_factory=lambda _settings: pytest.fail("不得重复 preflight 或构造组件"),
    ).run_baidu("run-abc", ExplodingPackage())  # type: ignore[arg-type]

    assert first == second


@pytest.mark.parametrize(
    ("method_name", "expected_codes"),
    [
        (
            "run_baidu",
            (
                ErrorCode.BAIDU_API_KEY_UNAVAILABLE,
                ErrorCode.BAIDU_SECRET_KEY_UNAVAILABLE,
            ),
        ),
        (
            "run_qwen",
            (
                ErrorCode.QWEN_ENDPOINT_UNAVAILABLE,
                ErrorCode.QWEN_API_KEY_UNAVAILABLE,
                ErrorCode.QWEN_MODEL_ID_UNAVAILABLE,
            ),
        ),
        (
            "run_pyannote",
            (
                ErrorCode.PYANNOTE_TOKEN_UNAVAILABLE,
                ErrorCode.PYANNOTE_TERMS_UNAVAILABLE,
                ErrorCode.PYANNOTE_DEPENDENCY_UNAVAILABLE,
                ErrorCode.PYANNOTE_MODEL_UNAVAILABLE,
            ),
        ),
        (
            "run_local_model_stack",
            (
                ErrorCode.SILERO_DEPENDENCY_UNAVAILABLE,
                ErrorCode.SILERO_MODEL_UNAVAILABLE,
                ErrorCode.FASTER_WHISPER_DEPENDENCY_UNAVAILABLE,
                ErrorCode.FASTER_WHISPER_MODEL_UNAVAILABLE,
                ErrorCode.WHISPERX_DEPENDENCY_UNAVAILABLE,
                ErrorCode.WHISPERX_MODEL_UNAVAILABLE,
                ErrorCode.YAMNET_DEPENDENCY_UNAVAILABLE,
                ErrorCode.YAMNET_MODEL_UNAVAILABLE,
                ErrorCode.LIVE_FIVE_LANGUAGE_AUDIO_UNAVAILABLE,
            ),
        ),
    ],
)
def test_each_preflight_collects_all_missing_items_in_contract_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    expected_codes: tuple[ErrorCode, ...],
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    calls: list[str] = []
    runner = LiveValidationRunner(
        Settings(
            workspace_root=tmp_path,
            qwen_base_url=None,
            qwen_api_key=None,
            qwen_model_id=None,
            baidu_api_key=None,
            baidu_secret_key=None,
            huggingface_token=None,
        ),
        EvidenceStore(tmp_path, runtime_root),
        components_factory=lambda _settings: calls.append("called"),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(runner, "_module_available", lambda _name: False, raising=False)

    check = getattr(runner, method_name)("run-abc", package)

    assert check.status == GateStatus.NOT_RUN
    assert calls == []
    raw = PreflightRawReport.model_validate_json(
        (runtime_root / "eval/reports/run-abc/preflight.json").read_bytes()
    )
    assert tuple(issue.code for issue in raw.issues or ()) == expected_codes


def test_new_run_recomputes_preflight_instead_of_reusing_old_not_run(
    tmp_path: Path,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    source = runtime_root / "eval/live/run-abc/sample-001"
    second = runtime_root / "eval/live/run-def/sample-001"
    second.mkdir(parents=True)
    for name in ("audio.wav", "keyframe.jpg", "clip.mp4"):
        shutil.copy2(source / name, second / name)
    first = LiveValidationRunner(
        Settings(workspace_root=tmp_path),
        EvidenceStore(tmp_path, runtime_root),
    ).run_qwen("run-abc", package)
    second_check = LiveValidationRunner(
        Settings(
            workspace_root=tmp_path,
            qwen_base_url="https://qwen.example/v1",
        ),
        EvidenceStore(tmp_path, runtime_root),
    ).run_qwen("run-def", package)

    assert first.status == second_check.status == GateStatus.NOT_RUN
    first_raw = PreflightRawReport.model_validate_json(
        (runtime_root / "eval/reports/run-abc/preflight.json").read_bytes()
    )
    second_raw = PreflightRawReport.model_validate_json(
        (runtime_root / "eval/reports/run-def/preflight.json").read_bytes()
    )
    assert tuple(issue.code for issue in first_raw.issues or ()) == (
        ErrorCode.QWEN_ENDPOINT_UNAVAILABLE,
        ErrorCode.QWEN_API_KEY_UNAVAILABLE,
        ErrorCode.QWEN_MODEL_ID_UNAVAILABLE,
    )
    assert tuple(issue.code for issue in second_raw.issues or ()) == (
        ErrorCode.QWEN_API_KEY_UNAVAILABLE,
        ErrorCode.QWEN_MODEL_ID_UNAVAILABLE,
    )


def test_preflight_rejects_input_that_cannot_form_complete_live_sample(
    tmp_path: Path,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    (runtime_root / "eval/live/run-abc/sample-001/keyframe.jpg").unlink()
    calls: list[str] = []
    runner = LiveValidationRunner(
        Settings(
            workspace_root=tmp_path,
            qwen_base_url="https://qwen.example/v1",
            qwen_api_key="qwen-secret",
            qwen_model_id="qwen3-vl-plus",
        ),
        EvidenceStore(tmp_path, runtime_root),
        components_factory=lambda _settings: calls.append("constructed"),  # type: ignore[arg-type]
    )

    check = runner.run_qwen("run-abc", package)

    assert check.status == GateStatus.NOT_RUN
    assert calls == []
    raw = PreflightRawReport.model_validate_json(
        (runtime_root / "eval/reports/run-abc/preflight.json").read_bytes()
    )
    assert tuple(issue.code for issue in raw.issues or ()) == (
        ErrorCode.LIVE_AUTHORIZED_CLIP_UNAVAILABLE,
    )


@pytest.mark.parametrize(
    "mutation",
    ("missing", "empty", "leaf_symlink", "parent_symlink", "oversize"),
)
def test_qwen_preflight_rejects_unsafe_or_unusable_clip(
    tmp_path: Path,
    mutation: str,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    sample_root = runtime_root / "eval/live/run-abc/sample-001"
    clip = sample_root / "clip.mp4"
    qwen_max_video_bytes = 16 * 1024 * 1024
    if mutation == "missing":
        clip.unlink()
    elif mutation == "empty":
        clip.write_bytes(b"")
    elif mutation == "leaf_symlink":
        target = runtime_root / "eval/live/linked-clip.mp4"
        target.write_bytes(b"linked")
        clip.unlink()
        clip.symlink_to(target)
    elif mutation == "parent_symlink":
        target = runtime_root / "eval/live/linked-sample"
        sample_root.rename(target)
        sample_root.symlink_to(target, target_is_directory=True)
    else:
        qwen_max_video_bytes = 4
        clip.write_bytes(b"too-large")
    calls: list[str] = []
    runner = LiveValidationRunner(
        Settings(
            workspace_root=tmp_path,
            qwen_base_url="https://qwen.example/v1",
            qwen_api_key="qwen-secret",
            qwen_model_id="qwen3-vl-plus",
            qwen_max_video_bytes=qwen_max_video_bytes,
        ),
        EvidenceStore(tmp_path, runtime_root),
        components_factory=lambda _settings: calls.append("called"),  # type: ignore[arg-type]
    )

    check = runner.run_qwen("run-abc", package)

    assert check.status == GateStatus.NOT_RUN
    assert calls == []
    raw = PreflightRawReport.model_validate_json(
        (runtime_root / "eval/reports/run-abc/preflight.json").read_bytes()
    )
    assert tuple(issue.code for issue in raw.issues or ()) == (
        ErrorCode.LIVE_AUTHORIZED_CLIP_UNAVAILABLE,
    )


@pytest.mark.parametrize(
    "source_kind",
    ("manifest", "authorization", "annotation", "media"),
)
def test_package_source_change_during_execution_fails_closed(
    tmp_path: Path,
    source_kind: str,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    assert package.dataset_path is not None
    assert package.authorization_path is not None
    sample = package.dataset.samples[0]
    sources = {
        "manifest": package.dataset_path,
        "authorization": package.authorization_path,
        "annotation": package.dataset.eval_root / sample.annotations_relative_path,
        "media": package.dataset.eval_root / sample.media_relative_path,
    }

    def execute(*_args: object) -> None:
        sources[source_kind].write_bytes(sources[source_kind].read_bytes() + b"\n")

    runner = LiveValidationRunner(
        Settings(
            workspace_root=tmp_path,
            qwen_base_url="https://qwen.example/v1",
            qwen_api_key="qwen-secret",
            qwen_model_id="qwen3-vl-plus",
        ),
        EvidenceStore(tmp_path, runtime_root),
        components_factory=lambda _settings: object(),  # type: ignore[arg-type]
        execution_port=execute,
    )

    with pytest.raises(VideoDemoError) as raised:
        runner.run_qwen("run-abc", package)

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID
    assert raised.value.__cause__ is None
    assert not (runtime_root / "eval/reports/run-abc/qwen_live.json").exists()
    assert not (runtime_root / "eval/live-authority/run-abc/qwen_live.json").exists()


@pytest.mark.parametrize(
    ("raised", "expected_code"),
    [
        (
            VideoDemoError(
                ErrorCode.QWEN_AUTHENTICATION_FAILED,
                "Bearer qwen-super-secret /Users/private response-body "
                "data:video/mp4;base64,AAAA request-plain-123",
                {"request_id": "request-plain-123"},
            ),
            ErrorCode.QWEN_AUTHENTICATION_FAILED,
        ),
        (
            RuntimeError(
                "Bearer qwen-super-secret /Users/private response-body "
                "data:video/mp4;base64,AAAA request-plain-123"
            ),
            ErrorCode.SYSTEM_FAILURE,
        ),
    ],
)
def test_started_qwen_failure_is_persisted_with_stable_code_and_no_sensitive_text(
    tmp_path: Path,
    raised: Exception,
    expected_code: ErrorCode,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    calls: list[str] = []

    def execute(*_args: object) -> None:
        calls.append("executed")
        raise raised

    settings = Settings(
        workspace_root=tmp_path,
        qwen_base_url="https://qwen.example/v1",
        qwen_api_key="qwen-super-secret",
        qwen_model_id="qwen3-vl-plus",
    )
    check = LiveValidationRunner(
        settings,
        EvidenceStore(tmp_path, runtime_root),
        components_factory=lambda _settings: object(),  # type: ignore[arg-type]
        execution_port=execute,
    ).run_qwen("run-abc", package)

    assert calls == ["executed"]
    assert check.status == GateStatus.FAIL
    raw = QwenLiveRawReport.model_validate_json(
        (runtime_root / "eval/reports/run-abc/raw.json").read_bytes()
    )
    assert raw.execution_started is True
    assert raw.failure_code == expected_code
    assert raw.failure_component == "qwen"
    assert raw.executions == ()
    forbidden = (
        "qwen-super-secret",
        "/Users/private",
        "response-body",
        "data:video/mp4",
        "request-plain-123",
    )
    report_root = runtime_root / "eval/reports/run-abc"
    encoded = b"\n".join(path.read_bytes() for path in report_root.rglob("*") if path.is_file())
    for value in forbidden:
        assert value.encode() not in encoded


def test_qwen_failure_keeps_only_successful_capability_probe_prefix(tmp_path: Path) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    settings = Settings(
        workspace_root=tmp_path,
        qwen_base_url="https://qwen.example/v1",
        qwen_api_key="qwen-secret",
        qwen_model_id="qwen3-vl-plus",
    )
    model = next(
        item
        for item in build_production_model_identity_report(settings).models
        if item.component == "qwen"
    )

    def execute(
        _check_id: str,
        samples: tuple[object, ...],
        _components: object,
        journal: object,
    ) -> None:
        sample = samples[0]
        journal.record_success(  # type: ignore[attr-defined]
            LiveExecutionSummary(
                schema_version="1.0.0",
                component="qwen",
                operation="capability_probe",
                evaluation_run_id="run-abc",
                model=model,
                sample_id=sample.sample_id,  # type: ignore[attr-defined]
                language=sample.language,  # type: ignore[attr-defined]
                input_kind="CLIP",
                input_sha256=sample.clip_sha256,  # type: ignore[attr-defined]
                request_id_sha256=hashlib.sha256(b"probe-request").hexdigest(),
                http_status=200,
                capabilities=("video_input", "strict_json_schema"),
                output_item_count=1,
            )
        )
        raise VideoDemoError(ErrorCode.QWEN_RESPONSE_INVALID, "供应商正文不得保留")

    check = LiveValidationRunner(
        settings,
        EvidenceStore(tmp_path, runtime_root),
        components_factory=lambda _settings: object(),  # type: ignore[arg-type]
        execution_port=execute,
    ).run_qwen("run-abc", package)

    assert check.status == GateStatus.FAIL
    raw = QwenLiveRawReport.model_validate_json(
        (runtime_root / "eval/reports/run-abc/raw.json").read_bytes()
    )
    assert tuple(fact.operation for fact in raw.executions) == ("capability_probe",)
    assert raw.failure_code == ErrorCode.QWEN_RESPONSE_INVALID
    assert raw.failure_component == "qwen"


def test_qwen_model_identity_mismatch_is_fail_without_success_fact(tmp_path: Path) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    settings = Settings(
        workspace_root=tmp_path,
        qwen_base_url="https://qwen.example/v1",
        qwen_api_key="qwen-secret",
        qwen_model_id="qwen3-vl-plus",
    )
    configured_model = next(
        item
        for item in build_production_model_identity_report(settings).models
        if item.component == "qwen"
    )
    wrong_model = configured_model.model_copy(update={"model_id": "qwen3-vl-max"})

    def execute(
        _check_id: str,
        samples: tuple[object, ...],
        _components: object,
        journal: object,
    ) -> None:
        sample = samples[0]
        journal.record_success(  # type: ignore[attr-defined]
            LiveExecutionSummary(
                schema_version="1.0.0",
                component="qwen",
                operation="capability_probe",
                evaluation_run_id="run-abc",
                model=wrong_model,
                sample_id=sample.sample_id,  # type: ignore[attr-defined]
                language=sample.language,  # type: ignore[attr-defined]
                input_kind="CLIP",
                input_sha256=sample.clip_sha256,  # type: ignore[attr-defined]
                request_id_sha256=hashlib.sha256(b"request").hexdigest(),
                http_status=200,
                capabilities=("video_input", "strict_json_schema"),
                output_item_count=1,
            )
        )

    check = LiveValidationRunner(
        settings,
        EvidenceStore(tmp_path, runtime_root),
        components_factory=lambda _settings: object(),  # type: ignore[arg-type]
        execution_port=execute,
    ).run_qwen("run-abc", package)

    assert check.status == GateStatus.FAIL
    raw = QwenLiveRawReport.model_validate_json(
        (runtime_root / "eval/reports/run-abc/raw.json").read_bytes()
    )
    assert raw.failure_code == ErrorCode.QWEN_RESPONSE_INVALID
    assert raw.failure_component == "qwen"
    assert raw.executions == ()


def test_started_baidu_failure_keeps_precise_code_without_success_fact(
    tmp_path: Path,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)

    def execute(*_args: object) -> None:
        raise VideoDemoError(ErrorCode.OCR_AUTHENTICATION_FAILED, "受控鉴权失败")

    check = LiveValidationRunner(
        Settings(
            workspace_root=tmp_path,
            baidu_api_key="baidu-key",
            baidu_secret_key="baidu-secret",
        ),
        EvidenceStore(tmp_path, runtime_root),
        components_factory=lambda _settings: object(),  # type: ignore[arg-type]
        execution_port=execute,
    ).run_baidu("run-abc", package)

    assert check.status == GateStatus.FAIL
    raw = BaiduLiveRawReport.model_validate_json(
        (runtime_root / "eval/reports/run-abc/raw.json").read_bytes()
    )
    assert raw.failure_code == ErrorCode.OCR_AUTHENTICATION_FAILED
    assert raw.failure_component == "baidu_ocr"
    assert raw.executions == ()


def test_started_pyannote_failure_keeps_precise_code_without_success_fact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    _write_pyannote_preflight_files(
        runtime_root,
        json.dumps(
            {
                "schema_version": "1.0.0",
                "model_id": "pyannote/speaker-diarization-community-1",
                "accepted": True,
            }
        ).encode(),
    )

    def execute(*_args: object) -> None:
        raise VideoDemoError(ErrorCode.PYANNOTE_AUTHENTICATION_FAILED, "受控鉴权失败")

    runner = LiveValidationRunner(
        Settings(workspace_root=tmp_path, huggingface_token="hf-secret"),
        EvidenceStore(tmp_path, runtime_root),
        components_factory=lambda _settings: object(),  # type: ignore[arg-type]
        execution_port=execute,
    )
    monkeypatch.setattr(runner, "_module_available", lambda _name: True)

    check = runner.run_pyannote("run-abc", package)

    assert check.status == GateStatus.FAIL
    raw = PyannoteLiveRawReport.model_validate_json(
        (runtime_root / "eval/reports/run-abc/raw.json").read_bytes()
    )
    assert raw.failure_code == ErrorCode.PYANNOTE_AUTHENTICATION_FAILED
    assert raw.failure_component == "pyannote"
    assert raw.executions == ()


def test_local_stack_uses_manifest_first_eligible_sample_and_keeps_success_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    languages = ("zh", "en", "ja", "ko", "es")
    sample_specs = tuple(
        sample
        for language in reversed(languages)
        for sample in (
            (f"{language}-blocked", language),
            (f"{language}-preferred", language),
            (f"{language}-alternate", language),
        )
    )
    package, runtime_root = _runner_package(tmp_path, samples=sample_specs)
    for language in languages:
        (runtime_root / f"eval/live/run-abc/{language}-blocked/audio.wav").unlink()
    _write_local_stack_preflight_models(runtime_root)
    settings = Settings(workspace_root=tmp_path)
    identities = build_production_model_identity_report(settings).models
    selected: list[tuple[str, str]] = []

    def execute(
        _check_id: str,
        samples: tuple[object, ...],
        _components: object,
        journal: object,
    ) -> None:
        selected.extend(
            (sample.language, sample.sample_id)  # type: ignore[attr-defined]
            for sample in samples
        )
        zh_sample = samples[0]
        for component, operation in (
            ("silero_vad", "vad"),
            ("faster_whisper", "transcribe"),
        ):
            model = next(item for item in identities if item.component == component)
            journal.record_success(  # type: ignore[attr-defined]
                LiveExecutionSummary(
                    schema_version="1.0.0",
                    component=component,
                    operation=operation,
                    evaluation_run_id="run-abc",
                    model=model,
                    sample_id=zh_sample.sample_id,  # type: ignore[attr-defined]
                    language=zh_sample.language,  # type: ignore[attr-defined]
                    input_kind="AUDIO",
                    input_sha256=zh_sample.audio_sha256,  # type: ignore[attr-defined]
                    output_item_count=1,
                )
            )
        raise VideoDemoError(ErrorCode.SPEECH_MODEL_UNAVAILABLE, "受控模型加载失败")

    runner = LiveValidationRunner(
        settings,
        EvidenceStore(tmp_path, runtime_root),
        components_factory=lambda _settings: object(),  # type: ignore[arg-type]
        execution_port=execute,
    )
    monkeypatch.setattr(runner, "_module_available", lambda _name: True)

    check = runner.run_local_model_stack("run-abc", package)

    assert check.status == GateStatus.FAIL
    assert selected == [
        (language, f"{language}-preferred") for language in languages
    ]
    raw = FiveLanguageModelsRawReport.model_validate_json(
        (runtime_root / "eval/reports/run-abc/raw.json").read_bytes()
    )
    assert tuple(sample.sample_id for sample in raw.samples) == tuple(
        f"{language}-preferred" for language in languages
    )
    assert tuple(fact.component for fact in raw.executions) == (
        "silero_vad",
        "faster_whisper",
    )
    assert raw.failure_code == ErrorCode.SPEECH_MODEL_UNAVAILABLE
    assert raw.failure_component == "faster_whisper"


def _record_live_failure_matrix_prefix(
    settings: Settings,
    check_id: str,
    samples: tuple[object, ...],
    journal: object,
    prefix_count: int,
) -> None:
    identities = build_production_model_identity_report(settings).models
    if check_id == "baidu_ocr_live":
        stages = (("baidu_ocr", "recognize", samples[0]),)
    elif check_id == "qwen_live":
        stages = (
            ("qwen", "capability_probe", samples[0]),
            ("qwen", "understand_segment", samples[0]),
        )
    elif check_id == "pyannote_live":
        stages = (("pyannote", "diarize", samples[0]),)
    else:
        by_language = {
            sample.language: sample  # type: ignore[attr-defined]
            for sample in samples
        }
        ordered = tuple(
            by_language[language] for language in ("zh", "en", "ja", "ko", "es")
        )
        stages = (
            ("silero_vad", "vad", ordered[0]),
            *(("faster_whisper", "transcribe", sample) for sample in ordered),
            *(("whisperx", "align", sample) for sample in ordered),
            ("yamnet", "detect", ordered[0]),
        )
    for component, operation, sample in stages[:prefix_count]:
        models = tuple(item for item in identities if item.component == component)
        model = (
            next(
                item
                for item in models
                if item.model_id
                == f"whisperx-align-{sample.language}"  # type: ignore[attr-defined]
            )
            if component == "whisperx"
            else models[0]
        )
        input_kind = (
            "KEYFRAME"
            if component == "baidu_ocr"
            else "CLIP" if component == "qwen" else "AUDIO"
        )
        input_sha256 = (
            sample.keyframe_sha256  # type: ignore[attr-defined]
            if input_kind == "KEYFRAME"
            else sample.clip_sha256  # type: ignore[attr-defined]
            if input_kind == "CLIP"
            else sample.audio_sha256  # type: ignore[attr-defined]
        )
        remote = component in {"baidu_ocr", "qwen"}
        journal.record_success(  # type: ignore[attr-defined]
            LiveExecutionSummary(
                schema_version="1.0.0",
                component=component,
                operation=operation,
                evaluation_run_id="run-abc",
                model=model,
                sample_id=sample.sample_id,  # type: ignore[attr-defined]
                language=sample.language,  # type: ignore[attr-defined]
                input_kind=input_kind,
                input_sha256=input_sha256,
                request_id_sha256=(
                    hashlib.sha256(f"{component}-{operation}".encode()).hexdigest()
                    if remote
                    else None
                ),
                http_status=200 if remote else None,
                capabilities=(
                    ("video_input", "strict_json_schema")
                    if operation == "capability_probe"
                    else ()
                ),
                output_item_count=1,
            )
        )


@pytest.mark.parametrize(
    (
        "case_id",
        "check_id",
        "raised_code",
        "prefix_count",
        "failure_component",
    ),
    (
        (
            "qwen_capability",
            "qwen_live",
            ErrorCode.QWEN_CAPABILITY_UNAVAILABLE,
            0,
            "qwen",
        ),
        (
            "qwen_network",
            "qwen_live",
            ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
            1,
            "qwen",
        ),
        ("qwen_response", "qwen_live", ErrorCode.QWEN_RESPONSE_INVALID, 1, "qwen"),
        (
            "baidu_auth",
            "baidu_ocr_live",
            ErrorCode.OCR_AUTHENTICATION_FAILED,
            0,
            "baidu_ocr",
        ),
        (
            "baidu_network",
            "baidu_ocr_live",
            ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
            0,
            "baidu_ocr",
        ),
        (
            "baidu_response",
            "baidu_ocr_live",
            ErrorCode.OCR_RESPONSE_INVALID,
            0,
            "baidu_ocr",
        ),
        (
            "pyannote_auth",
            "pyannote_live",
            ErrorCode.PYANNOTE_AUTHENTICATION_FAILED,
            0,
            "pyannote",
        ),
        (
            "pyannote_model",
            "pyannote_live",
            ErrorCode.PYANNOTE_MODEL_UNAVAILABLE,
            0,
            "pyannote",
        ),
        (
            "pyannote_dependency",
            "pyannote_live",
            ErrorCode.SPEECH_DEPENDENCY_UNAVAILABLE,
            0,
            "pyannote",
        ),
        (
            "silero_load",
            "five_language_models",
            ErrorCode.SPEECH_DEPENDENCY_UNAVAILABLE,
            0,
            "silero_vad",
        ),
        (
            "silero_inference",
            "five_language_models",
            ErrorCode.SPEECH_AUDIO_INVALID,
            0,
            "silero_vad",
        ),
        (
            "silero_output",
            "five_language_models",
            ErrorCode.SPEECH_MODEL_UNAVAILABLE,
            0,
            "silero_vad",
        ),
        (
            "faster_load",
            "five_language_models",
            ErrorCode.SPEECH_DEPENDENCY_UNAVAILABLE,
            1,
            "faster_whisper",
        ),
        (
            "faster_inference",
            "five_language_models",
            ErrorCode.SPEECH_AUDIO_INVALID,
            1,
            "faster_whisper",
        ),
        (
            "faster_output",
            "five_language_models",
            ErrorCode.SPEECH_MODEL_UNAVAILABLE,
            1,
            "faster_whisper",
        ),
        (
            "whisperx_load",
            "five_language_models",
            ErrorCode.SPEECH_DEPENDENCY_UNAVAILABLE,
            6,
            "whisperx",
        ),
        (
            "whisperx_inference",
            "five_language_models",
            ErrorCode.SPEECH_AUDIO_INVALID,
            6,
            "whisperx",
        ),
        (
            "whisperx_output",
            "five_language_models",
            ErrorCode.SPEECH_MODEL_UNAVAILABLE,
            6,
            "whisperx",
        ),
        (
            "yamnet_load",
            "five_language_models",
            ErrorCode.SPEECH_DEPENDENCY_UNAVAILABLE,
            11,
            "yamnet",
        ),
        (
            "yamnet_inference",
            "five_language_models",
            ErrorCode.SPEECH_AUDIO_INVALID,
            11,
            "yamnet",
        ),
        (
            "yamnet_output",
            "five_language_models",
            ErrorCode.SPEECH_MODEL_UNAVAILABLE,
            11,
            "yamnet",
        ),
        ("unknown", "qwen_live", None, 0, "qwen"),
        ("qwen_construct_allowed", "qwen_live", ErrorCode.QWEN_RESPONSE_INVALID, 0, "qwen"),
        ("baidu_construct_unknown", "baidu_ocr_live", None, 0, "baidu_ocr"),
        (
            "pyannote_construct_allowed",
            "pyannote_live",
            ErrorCode.PYANNOTE_MODEL_UNAVAILABLE,
            0,
            "pyannote",
        ),
        (
            "local_construct_allowed",
            "five_language_models",
            ErrorCode.SPEECH_DEPENDENCY_UNAVAILABLE,
            0,
            "silero_vad",
        ),
        ("baidu_close", "baidu_ocr_live", None, 1, "components_close"),
        ("qwen_close", "qwen_live", None, 2, "components_close"),
        ("pyannote_close", "pyannote_live", None, 1, "components_close"),
        ("local_close", "five_language_models", None, 12, "components_close"),
    ),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_runner_failure_classification_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
    check_id: str,
    raised_code: ErrorCode | None,
    prefix_count: int,
    failure_component: str,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    sample_specs = (
        (
            ("zh-001", "zh"),
            ("en-001", "en"),
            ("ja-001", "ja"),
            ("ko-001", "ko"),
            ("es-001", "es"),
        )
        if check_id == "five_language_models"
        else (("sample-001", "zh"),)
    )
    package, runtime_root = _runner_package(tmp_path, samples=sample_specs)
    settings_kwargs: dict[str, object] = {"workspace_root": tmp_path}
    if check_id == "qwen_live":
        settings_kwargs.update(
            qwen_base_url="https://qwen.example/v1",
            qwen_api_key="qwen-secret",
            qwen_model_id="qwen3-vl-plus",
        )
    elif check_id == "baidu_ocr_live":
        settings_kwargs.update(
            baidu_api_key="baidu-key",
            baidu_secret_key="baidu-secret",
        )
    elif check_id == "pyannote_live":
        settings_kwargs["huggingface_token"] = "hf-secret"
        _write_pyannote_preflight_files(
            runtime_root,
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "model_id": "pyannote/speaker-diarization-community-1",
                    "accepted": True,
                }
            ).encode(),
        )
    else:
        _write_local_stack_preflight_models(runtime_root)
    settings = Settings(**settings_kwargs)  # type: ignore[arg-type]

    execution_calls: list[str] = []

    def execute(
        current_check_id: str,
        samples: tuple[object, ...],
        _components: object,
        journal: object,
    ) -> None:
        execution_calls.append(current_check_id)
        _record_live_failure_matrix_prefix(
            settings,
            current_check_id,
            samples,
            journal,
            prefix_count,
        )
        if case_id.endswith("_close"):
            return
        message = f"sensitive-{case_id} /Users/private bearer-secret"
        if raised_code is None:
            raise RuntimeError(message)
        raise VideoDemoError(raised_code, message)

    class ClosingComponents:
        def close(self) -> None:
            raise RuntimeError(
                f"sensitive-{case_id} /Users/private bearer-secret"
            )

    def build_components(_settings: Settings) -> object:
        message = f"sensitive-{case_id} /Users/private bearer-secret"
        if "_construct_" in case_id:
            if raised_code is None:
                raise RuntimeError(message)
            raise VideoDemoError(raised_code, message)
        if case_id.endswith("_close"):
            return ClosingComponents()
        return object()

    runner = LiveValidationRunner(
        settings,
        EvidenceStore(tmp_path, runtime_root),
        components_factory=build_components,  # type: ignore[arg-type]
        execution_port=execute,
    )
    if check_id in {"pyannote_live", "five_language_models"}:
        monkeypatch.setattr(runner, "_module_available", lambda _name: True)
    method = {
        "baidu_ocr_live": runner.run_baidu,
        "qwen_live": runner.run_qwen,
        "pyannote_live": runner.run_pyannote,
        "five_language_models": runner.run_local_model_stack,
    }[check_id]

    check = method("run-abc", package)

    raw_type = {
        "baidu_ocr_live": BaiduLiveRawReport,
        "qwen_live": QwenLiveRawReport,
        "pyannote_live": PyannoteLiveRawReport,
        "five_language_models": FiveLanguageModelsRawReport,
    }[check_id]
    raw = raw_type.model_validate_json(
        (runtime_root / "eval/reports/run-abc/raw.json").read_bytes()
    )
    assert check.status == GateStatus.FAIL
    assert raw.execution_started is True
    assert raw.failure_component == failure_component
    assert raw.failure_code == (raised_code or ErrorCode.SYSTEM_FAILURE)
    assert len(raw.executions) == prefix_count
    assert len(execution_calls) == (0 if "_construct_" in case_id else 1)
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (runtime_root / "eval/reports/run-abc").iterdir()
    )
    assert f"sensitive-{case_id}" not in persisted
    assert "/Users/private" not in persisted
    assert "bearer-secret" not in persisted


@pytest.mark.parametrize("mutation", ("tamper", "extra"))
def test_existing_conflicting_complete_run_returns_stable_idempotency_error(
    tmp_path: Path,
    mutation: str,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    store = EvidenceStore(tmp_path, runtime_root)
    settings = Settings(workspace_root=tmp_path)
    LiveValidationRunner(settings, store).run_baidu("run-abc", package)
    report_root = runtime_root / "eval/reports/run-abc"
    if mutation == "tamper":
        (report_root / "preflight.json").write_text("{}", encoding="utf-8")
    else:
        (report_root / "unexpected.txt").write_text("conflict", encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        LiveValidationRunner(settings, store).run_baidu("run-abc", package)

    assert raised.value.code == ErrorCode.IDEMPOTENCY_CONFLICT
    assert raised.value.__cause__ is None


def _exception_graph_text(error: BaseException) -> str:
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    values: list[str] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        values.extend((str(current), repr(current), repr(current.args)))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return "\n".join(values)


def test_input_change_during_execution_fails_closed_without_sensitive_exception_chain(
    tmp_path: Path,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    clip = runtime_root / "eval/live/run-abc/sample-001/clip.mp4"

    def execute(*_args: object) -> None:
        clip.write_bytes(b"changed-during-call")
        raise RuntimeError(
            f"Bearer qwen-secret {tmp_path} supplier-body data:video/mp4;base64,AAAA request-123"
        )

    runner = LiveValidationRunner(
        Settings(
            workspace_root=tmp_path,
            qwen_base_url="https://qwen.example/v1",
            qwen_api_key="qwen-secret",
            qwen_model_id="qwen3-vl-plus",
        ),
        EvidenceStore(tmp_path, runtime_root),
        components_factory=lambda _settings: object(),  # type: ignore[arg-type]
        execution_port=execute,
    )

    with pytest.raises(VideoDemoError) as raised:
        runner.run_qwen("run-abc", package)

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID
    assert not (runtime_root / "eval/reports/run-abc/qwen_live.json").exists()
    assert not (runtime_root / "eval/live-authority/run-abc/qwen_live.json").exists()
    graph = _exception_graph_text(raised.value)
    for forbidden in (
        "qwen-secret",
        str(tmp_path),
        "supplier-body",
        "data:video/mp4",
        "request-123",
    ):
        assert forbidden not in graph


def test_controlled_complete_success_cannot_publish_workspace_pass(tmp_path: Path) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    settings = Settings(
        workspace_root=tmp_path,
        qwen_base_url="https://qwen.example/v1",
        qwen_api_key="qwen-secret",
        qwen_model_id="qwen3-vl-plus",
    )
    model = next(
        item
        for item in build_production_model_identity_report(settings).models
        if item.component == "qwen"
    )

    def execute(
        _check_id: str,
        samples: tuple[object, ...],
        _components: object,
        journal: object,
    ) -> None:
        sample = samples[0]
        for operation, capabilities in (
            ("capability_probe", ("video_input", "strict_json_schema")),
            ("understand_segment", ()),
        ):
            journal.record_success(  # type: ignore[attr-defined]
                LiveExecutionSummary(
                    schema_version="1.0.0",
                    component="qwen",
                    operation=operation,
                    evaluation_run_id="run-abc",
                    model=model,
                    sample_id=sample.sample_id,  # type: ignore[attr-defined]
                    language=sample.language,  # type: ignore[attr-defined]
                    input_kind="CLIP",
                    input_sha256=sample.clip_sha256,  # type: ignore[attr-defined]
                    request_id_sha256=hashlib.sha256(operation.encode()).hexdigest(),
                    http_status=200,
                    capabilities=capabilities,
                    output_item_count=1,
                )
            )

    runner = LiveValidationRunner(
        settings,
        EvidenceStore(tmp_path, runtime_root),
        components_factory=lambda _settings: object(),  # type: ignore[arg-type]
        execution_port=execute,
    )

    with pytest.raises(VideoDemoError) as raised:
        runner.run_qwen("run-abc", package)

    assert raised.value.code == ErrorCode.SYSTEM_FAILURE
    assert not (runtime_root / "eval/reports/run-abc/qwen_live.json").exists()
    assert not (runtime_root / "eval/live-authority/run-abc/qwen_live.json").exists()


def test_default_production_baidu_execution_publishes_reverifiable_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import video_demo.evaluation.live_runner as live_runner_module
    from video_demo.evaluation.live_runner import LiveValidationRunner
    from video_demo.visual.ocr import OcrProviderResponse

    package, runtime_root = _runner_package(tmp_path)
    keyframe = runtime_root / "eval/live/run-abc/sample-001/keyframe.jpg"
    keyframe.write_bytes(b"\xff\xd8\xff\xd9")
    settings = Settings(
        workspace_root=tmp_path,
        baidu_api_key="baidu-key",
        baidu_secret_key="baidu-secret",
    )

    class Ocr:
        def recognize(self, image: bytes, language: str) -> OcrProviderResponse:
            assert image == b"\xff\xd8\xff\xd9"
            assert language == "zh"
            return OcrProviderResponse(
                request_id="provider-request-plain",
                http_status=200,
                lines=(),
            )

    diagnostics = _production_diagnostics(settings, ocr_client=Ocr())
    monkeypatch.setattr(
        live_runner_module,
        "build_production_diagnostic_components",
        lambda received: diagnostics if received is settings else pytest.fail("Settings 不一致"),
    )

    check = LiveValidationRunner(
        settings,
        EvidenceStore(tmp_path, runtime_root),
    ).run_baidu("run-abc", package)

    assert check.status == GateStatus.PASS
    raw = BaiduLiveRawReport.model_validate_json(
        (runtime_root / "eval/reports/run-abc/raw.json").read_bytes()
    )
    assert raw.status == GateStatus.PASS
    assert raw.executions[0].request_id_sha256 == hashlib.sha256(
        b"provider-request-plain"
    ).hexdigest()
    persisted = b"\n".join(
        path.read_bytes()
        for path in (runtime_root / "eval/reports/run-abc").iterdir()
    )
    assert b"provider-request-plain" not in persisted


def test_default_production_qwen_execution_records_probe_and_segment_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import video_demo.evaluation.live_runner as live_runner_module
    from video_demo.domain.result import SegmentUnderstanding
    from video_demo.evaluation.live_runner import LiveValidationRunner
    from video_demo.integrations.qwen import QwenCapabilities, QwenProviderReceipt

    package, runtime_root = _runner_package(tmp_path)
    settings = Settings(
        workspace_root=tmp_path,
        qwen_base_url="https://qwen.example/v1",
        qwen_api_key="qwen-secret",
        qwen_model_id="qwen3-vl-plus",
    )

    class Qwen:
        def probe_capabilities_with_receipt(self, clip: object) -> tuple[object, object]:
            assert clip.path == runtime_root / "eval/live/run-abc/sample-001/clip.mp4"  # type: ignore[attr-defined]
            return (
                QwenCapabilities(
                    model_id="qwen3-vl-plus",
                    max_video_bytes=settings.qwen_max_video_bytes,
                    max_video_duration_ms=settings.qwen_max_video_duration_ms,
                    timeout_seconds=settings.qwen_timeout_seconds,
                ),
                QwenProviderReceipt(response_id="probe-response-plain", http_status=200),
            )

        def understand_segment_with_receipt(
            self,
            request: object,
        ) -> tuple[SegmentUnderstanding, QwenProviderReceipt]:
            evidence_id = request.evidence[0].evidence_id  # type: ignore[attr-defined]
            return (
                SegmentUnderstanding(
                    title="片段",
                    summary_zh="结构化片段摘要",
                    evidence_refs=(evidence_id,),
                ),
                QwenProviderReceipt(response_id="segment-response-plain", http_status=200),
            )

    diagnostics = _production_diagnostics(settings, qwen_client=Qwen())
    monkeypatch.setattr(
        live_runner_module,
        "build_production_diagnostic_components",
        lambda received: diagnostics if received is settings else pytest.fail("Settings 不一致"),
    )

    check = LiveValidationRunner(
        settings,
        EvidenceStore(tmp_path, runtime_root),
    ).run_qwen("run-abc", package)

    assert check.status == GateStatus.PASS
    raw = QwenLiveRawReport.model_validate_json(
        (runtime_root / "eval/reports/run-abc/raw.json").read_bytes()
    )
    assert tuple(fact.operation for fact in raw.executions) == (
        "capability_probe",
        "understand_segment",
    )
    persisted = b"\n".join(
        path.read_bytes()
        for path in (runtime_root / "eval/reports/run-abc").iterdir()
    )
    assert b"probe-response-plain" not in persisted
    assert b"segment-response-plain" not in persisted
    assert "结构化片段摘要".encode() not in persisted


def test_default_production_pyannote_execution_publishes_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import video_demo.evaluation.live_runner as live_runner_module
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    _write_pyannote_preflight_files(
        runtime_root,
        json.dumps(
            {
                "schema_version": "1.0.0",
                "model_id": "pyannote/speaker-diarization-community-1",
                "accepted": True,
            }
        ).encode(),
    )
    settings = Settings(workspace_root=tmp_path, huggingface_token="hf-secret")

    class Diarizer:
        def diarize(self, _audio: Path, **_kwargs: object) -> tuple[object, ...]:
            return (object(),)

    speech_models = SimpleNamespace(diarizer=Diarizer())
    diagnostics = _production_diagnostics(settings, speech_models=speech_models)
    monkeypatch.setattr(
        live_runner_module,
        "build_production_diagnostic_components",
        lambda received: diagnostics if received is settings else pytest.fail("Settings 不一致"),
    )
    runner = LiveValidationRunner(settings, EvidenceStore(tmp_path, runtime_root))
    monkeypatch.setattr(runner, "_module_available", lambda _name: True)

    check = runner.run_pyannote("run-abc", package)

    assert check.status == GateStatus.PASS
    raw = PyannoteLiveRawReport.model_validate_json(
        (runtime_root / "eval/reports/run-abc/raw.json").read_bytes()
    )
    assert len(raw.executions) == 1
    assert raw.executions[0].model.model_id == (
        "pyannote/speaker-diarization-community-1"
    )


def test_default_production_local_models_execute_complete_five_language_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import video_demo.evaluation.live_runner as live_runner_module
    from video_demo.evaluation.live_runner import LiveValidationRunner
    from video_demo.speech.asr import RawAsrSegment

    package, runtime_root = _runner_package(
        tmp_path,
        samples=(
            ("zh-001", "zh"),
            ("en-001", "en"),
            ("ja-001", "ja"),
            ("ko-001", "ko"),
            ("es-001", "es"),
        ),
    )
    _write_local_stack_preflight_models(runtime_root)
    settings = Settings(workspace_root=tmp_path)

    class Vad:
        def detect(self, _audio: Path, *, duration_ms: int) -> object:
            assert duration_ms == 1_000
            return SimpleNamespace(speech=(object(),))

    class Recognizer:
        def transcribe_slice(self, _audio: Path, _span: object) -> tuple[RawAsrSegment, ...]:
            return (
                RawAsrSegment(
                    start_ms=0,
                    end_ms=500,
                    text="transient-provider-text",
                    confidence=0.9,
                ),
            )

    class Aligner:
        def align(self, _audio: Path, segments: object) -> object:
            assert segments
            return SimpleNamespace(words=(object(),), warning_codes=())

    class AudioEvents:
        def detect(self, _audio: Path, *, duration_ms: int) -> tuple[object, ...]:
            assert duration_ms == 1_000
            return ()

    speech_models = SimpleNamespace(
        vad=Vad(),
        recognizer=Recognizer(),
        aligner=Aligner(),
        audio_events=AudioEvents(),
    )
    diagnostics = _production_diagnostics(settings, speech_models=speech_models)
    monkeypatch.setattr(
        live_runner_module,
        "build_production_diagnostic_components",
        lambda received: diagnostics if received is settings else pytest.fail("Settings 不一致"),
    )
    runner = LiveValidationRunner(settings, EvidenceStore(tmp_path, runtime_root))
    monkeypatch.setattr(runner, "_module_available", lambda _name: True)

    check = runner.run_local_model_stack("run-abc", package)

    assert check.status == GateStatus.PASS
    raw = FiveLanguageModelsRawReport.model_validate_json(
        (runtime_root / "eval/reports/run-abc/raw.json").read_bytes()
    )
    assert len(raw.executions) == 12
    assert tuple(fact.component for fact in raw.executions) == (
        "silero_vad",
        *("faster_whisper" for _ in range(5)),
        *("whisperx" for _ in range(5)),
        "yamnet",
    )
    persisted = b"\n".join(
        path.read_bytes()
        for path in (runtime_root / "eval/reports/run-abc").iterdir()
    )
    assert b"transient-provider-text" not in persisted


def test_complete_qwen_close_failure_persists_reverifiable_system_failure(
    tmp_path: Path,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    settings = Settings(
        workspace_root=tmp_path,
        qwen_base_url="https://qwen.example/v1",
        qwen_api_key="qwen-secret",
        qwen_model_id="qwen3-vl-plus",
    )
    model = next(
        item
        for item in build_production_model_identity_report(settings).models
        if item.component == "qwen"
    )

    class ClosingComponents:
        def close(self) -> None:
            raise RuntimeError("sensitive-close-body /Users/private qwen-secret")

    def execute(
        _check_id: str,
        samples: tuple[object, ...],
        _components: object,
        journal: object,
    ) -> None:
        sample = samples[0]
        for operation, capabilities in (
            ("capability_probe", ("video_input", "strict_json_schema")),
            ("understand_segment", ()),
        ):
            journal.record_success(  # type: ignore[attr-defined]
                LiveExecutionSummary(
                    schema_version="1.0.0",
                    component="qwen",
                    operation=operation,
                    evaluation_run_id="run-abc",
                    model=model,
                    sample_id=sample.sample_id,  # type: ignore[attr-defined]
                    language=sample.language,  # type: ignore[attr-defined]
                    input_kind="CLIP",
                    input_sha256=sample.clip_sha256,  # type: ignore[attr-defined]
                    request_id_sha256=hashlib.sha256(operation.encode()).hexdigest(),
                    http_status=200,
                    capabilities=capabilities,
                    output_item_count=1,
                )
            )

    store = EvidenceStore(tmp_path, runtime_root)
    check = LiveValidationRunner(
        settings,
        store,
        components_factory=lambda _settings: ClosingComponents(),  # type: ignore[arg-type]
        execution_port=execute,
    ).run_qwen("run-abc", package)

    assert check.status == GateStatus.FAIL
    raw = QwenLiveRawReport.model_validate_json(
        (runtime_root / "eval/reports/run-abc/raw.json").read_bytes()
    )
    assert tuple(fact.operation for fact in raw.executions) == (
        "capability_probe",
        "understand_segment",
    )
    assert raw.failure_code == ErrorCode.SYSTEM_FAILURE
    assert raw.failure_component == "components_close"

    class ExplodingPackage:
        @property
        def dataset(self) -> object:
            raise AssertionError("完整 close FAIL 重验不得访问 package")

    reloaded = LiveValidationRunner(
        settings,
        store,
        components_factory=lambda _settings: pytest.fail("不得重建组件"),
    ).run_qwen("run-abc", ExplodingPackage())  # type: ignore[arg-type]
    assert reloaded == check
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (runtime_root / "eval/reports/run-abc").iterdir()
    )
    assert "sensitive-close-body" not in persisted
    assert "/Users/private" not in persisted
    assert "qwen-secret" not in persisted


def test_package_source_removal_during_execution_has_no_absolute_path_in_context(
    tmp_path: Path,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    assert package.authorization_path is not None

    def execute(*_args: object) -> None:
        package.authorization_path.unlink()
        raise RuntimeError("Bearer secret supplier-body")

    runner = LiveValidationRunner(
        Settings(
            workspace_root=tmp_path,
            qwen_base_url="https://qwen.example/v1",
            qwen_api_key="secret",
            qwen_model_id="qwen3-vl-plus",
        ),
        EvidenceStore(tmp_path, runtime_root),
        components_factory=lambda _settings: object(),  # type: ignore[arg-type]
        execution_port=execute,
    )

    with pytest.raises(VideoDemoError) as raised:
        runner.run_qwen("run-abc", package)

    graph = _exception_graph_text(raised.value)
    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID
    assert str(tmp_path) not in graph
    assert "supplier-body" not in graph
    assert "Bearer secret" not in graph


def test_invalid_nonempty_qwen_model_id_is_still_preflight_unavailable(
    tmp_path: Path,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    check = LiveValidationRunner(
        Settings(
            workspace_root=tmp_path,
            qwen_base_url="https://qwen.example/v1",
            qwen_api_key="secret",
            qwen_model_id="caller-controlled-model",
        ),
        EvidenceStore(tmp_path, runtime_root),
    ).run_qwen("run-abc", package)

    assert check.status == GateStatus.NOT_RUN
    raw = PreflightRawReport.model_validate_json(
        (runtime_root / "eval/reports/run-abc/preflight.json").read_bytes()
    )
    assert tuple(issue.code for issue in raw.issues or ()) == (
        ErrorCode.QWEN_MODEL_ID_UNAVAILABLE,
    )


def test_demo_mode_allows_custom_qwen_model_id_to_reach_capability_probe(
    tmp_path: Path,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    check = LiveValidationRunner(
        Settings(
            workspace_root=tmp_path,
            demo_degraded_mode=True,
            qwen_base_url="https://qwen.example/v1",
            qwen_api_key="secret",
            qwen_model_id="caller-controlled-model",
        ),
        EvidenceStore(tmp_path, runtime_root),
        components_factory=lambda _settings: object(),  # type: ignore[arg-type]
        execution_port=lambda *_args: None,
    ).run_qwen("run-abc", package)

    assert check.status == GateStatus.FAIL
    assert not (runtime_root / "eval/reports/run-abc/preflight.json").exists()


@pytest.mark.parametrize(
    "terms",
    (
        json.dumps(
            {
                "schema_version": "1.0.0",
                "model_id": "pyannote/speaker-diarization-community-1",
                "accepted": True,
                "extra": "forbidden",
            }
        ).encode(),
        json.dumps(
            {
                "schema_version": "1.0.0",
                "model_id": "pyannote/speaker-diarization-3.1",
                "accepted": True,
            }
        ).encode(),
        json.dumps(
            {
                "schema_version": "1.0.0",
                "model_id": "pyannote/speaker-diarization-community-1",
                "accepted": False,
            }
        ).encode(),
        json.dumps(
            {
                "schema_version": "1.0.0",
                "model_id": "pyannote/speaker-diarization-community-1",
                "accepted": 1,
            }
        ).encode(),
        b"\xff\xfe",
    ),
)
def test_pyannote_terms_require_exact_strict_json_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terms: bytes,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    _write_pyannote_preflight_files(runtime_root, terms)
    runner = LiveValidationRunner(
        Settings(workspace_root=tmp_path, huggingface_token="hf-secret"),
        EvidenceStore(tmp_path, runtime_root),
    )
    monkeypatch.setattr(runner, "_module_available", lambda _name: True)

    check = runner.run_pyannote("run-abc", package)

    assert check.status == GateStatus.NOT_RUN
    raw = PreflightRawReport.model_validate_json(
        (runtime_root / "eval/reports/run-abc/preflight.json").read_bytes()
    )
    assert tuple(issue.code for issue in raw.issues or ()) == (
        ErrorCode.PYANNOTE_TERMS_UNAVAILABLE,
    )


@pytest.mark.parametrize("cache_kind", ("directories_only", "empty_file"))
def test_pyannote_model_tree_requires_nonempty_regular_model_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache_kind: str,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    model_root = runtime_root / "models/pyannote"
    model_root.mkdir(parents=True)
    (model_root / "terms-accepted.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "model_id": "pyannote/speaker-diarization-community-1",
                "accepted": True,
            }
        ),
        encoding="utf-8",
    )
    if cache_kind == "directories_only":
        (model_root / "snapshots/revision").mkdir(parents=True)
    else:
        (model_root / "empty.bin").write_bytes(b"")
    runner = LiveValidationRunner(
        Settings(workspace_root=tmp_path, huggingface_token="hf-secret"),
        EvidenceStore(tmp_path, runtime_root),
    )
    monkeypatch.setattr(runner, "_module_available", lambda _name: True)

    check = runner.run_pyannote("run-abc", package)

    assert check.status == GateStatus.NOT_RUN
    raw = PreflightRawReport.model_validate_json(
        (runtime_root / "eval/reports/run-abc/preflight.json").read_bytes()
    )
    assert tuple(issue.code for issue in raw.issues or ()) == (
        ErrorCode.PYANNOTE_MODEL_UNAVAILABLE,
    )


def test_pyannote_terms_file_alone_is_not_model_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    model_root = runtime_root / "models/pyannote"
    model_root.mkdir(parents=True)
    (model_root / "terms-accepted.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "model_id": "pyannote/speaker-diarization-community-1",
                "accepted": True,
            }
        ),
        encoding="utf-8",
    )
    runner = LiveValidationRunner(
        Settings(workspace_root=tmp_path, huggingface_token="hf-secret"),
        EvidenceStore(tmp_path, runtime_root),
    )
    monkeypatch.setattr(runner, "_module_available", lambda _name: True)

    check = runner.run_pyannote("run-abc", package)

    assert check.status == GateStatus.NOT_RUN
    raw = PreflightRawReport.model_validate_json(
        (runtime_root / "eval/reports/run-abc/preflight.json").read_bytes()
    )
    assert tuple(issue.code for issue in raw.issues or ()) == (
        ErrorCode.PYANNOTE_MODEL_UNAVAILABLE,
    )


def test_pyannote_model_tree_rejects_symlink_even_after_regular_cache_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_demo.evaluation.live_runner import LiveValidationRunner

    package, runtime_root = _runner_package(tmp_path)
    model_root = runtime_root / "models/pyannote"
    model_root.mkdir(parents=True)
    (model_root / "terms-accepted.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "model_id": "pyannote/speaker-diarization-community-1",
                "accepted": True,
            }
        ),
        encoding="utf-8",
    )
    (model_root / "a-model-cache.bin").write_bytes(b"model")
    target = runtime_root / "models/symlink-target.bin"
    target.write_bytes(b"target")
    (model_root / "z-linked-cache.bin").symlink_to(target)
    runner = LiveValidationRunner(
        Settings(workspace_root=tmp_path, huggingface_token="hf-secret"),
        EvidenceStore(tmp_path, runtime_root),
    )
    monkeypatch.setattr(runner, "_module_available", lambda _name: True)

    check = runner.run_pyannote("run-abc", package)

    assert check.status == GateStatus.NOT_RUN
    raw = PreflightRawReport.model_validate_json(
        (runtime_root / "eval/reports/run-abc/preflight.json").read_bytes()
    )
    assert tuple(issue.code for issue in raw.issues or ()) == (
        ErrorCode.PYANNOTE_MODEL_UNAVAILABLE,
    )


def test_production_model_identity_report_is_complete_and_redacted(tmp_path: Path) -> None:
    from importlib.metadata import version

    settings = Settings(
        workspace_root=tmp_path,
        runtime_root=Path("runtime/live"),
        inference_device="mps",
        whisper_compute_type="float16",
        qwen_base_url="https://dashscope.example/compatible-mode/v1",
        qwen_model_id="qwen3-vl-plus",
        qwen_api_key="qwen-secret",
        baidu_api_key="baidu-api-secret",
        baidu_secret_key="baidu-secret-secret",
        huggingface_token="huggingface-secret",
    )

    report = build_production_model_identity_report(settings)

    assert isinstance(report, ProductionModelIdentityReport)
    assert report.schema_version == "1.0.0"
    assert re.fullmatch(r"[0-9a-f]{64}", report.settings_fingerprint)
    assert {
        (model.component, model.provider, model.model_id, model.device, model.revision)
        for model in report.models
    } == {
        ("silero_vad", "local", "silero-vad", "cpu", version("silero-vad")),
        ("faster_whisper", "local", "large-v3", "mps", version("faster-whisper")),
        ("whisperx", "local", "whisperx-align-zh", "cpu", version("whisperx")),
        ("whisperx", "local", "whisperx-align-en", "cpu", version("whisperx")),
        ("whisperx", "local", "whisperx-align-ja", "cpu", version("whisperx")),
        ("whisperx", "local", "whisperx-align-ko", "cpu", version("whisperx")),
        ("whisperx", "local", "whisperx-align-es", "cpu", version("whisperx")),
        (
            "pyannote",
            "local",
            "pyannote/speaker-diarization-community-1",
            "cpu",
            version("pyannote.audio"),
        ),
        ("yamnet", "local", "yamnet", "cpu", version("tensorflow-hub")),
        ("baidu_ocr", "baidu_ocr", "accurate_basic", None, None),
        ("qwen", "qwen", "qwen3-vl-plus", None, None),
    }
    serialized = report.model_dump_json()
    for forbidden in (
        "qwen-secret",
        "baidu-api-secret",
        "baidu-secret-secret",
        "huggingface-secret",
        str(tmp_path),
        "data:",
    ):
        assert forbidden not in serialized


def test_model_identity_report_omits_unconfigured_qwen_identity(tmp_path: Path) -> None:
    report = build_production_model_identity_report(Settings(workspace_root=tmp_path))

    assert all(model.component != "qwen" for model in report.models)
    assert any(model.component == "baidu_ocr" for model in report.models)


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("runtime_root", Path("runtime/other")),
        ("ffmpeg_path", Path("tools/custom-ffmpeg")),
        ("ffprobe_path", Path("tools/custom-ffprobe")),
        ("inference_device", "mps"),
        ("whisper_compute_type", "float32"),
        ("max_video_bytes", 123_456),
        ("qwen_base_url", "https://other.example/compatible-mode/v1"),
        ("qwen_model_id", "qwen3-vl-max"),
        ("qwen_max_video_bytes", 12_345),
        ("qwen_max_video_duration_ms", 12_345),
        ("qwen_timeout_seconds", 45.0),
        (
            "baidu_ocr_endpoint",
            "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
        ),
    ],
)
def test_settings_fingerprint_changes_with_execution_semantics(
    tmp_path: Path,
    override: str,
    value: object,
) -> None:
    base = {
        "workspace_root": tmp_path,
        "runtime_root": Path("runtime/live"),
        "ffmpeg_path": Path("tools/ffmpeg"),
        "ffprobe_path": Path("tools/ffprobe"),
        "qwen_base_url": "https://dashscope.example/compatible-mode/v1",
        "qwen_model_id": "qwen3-vl-plus",
    }
    changed = dict(base)
    changed[override] = value

    baseline = build_production_model_identity_report(Settings(**base))
    candidate = build_production_model_identity_report(Settings(**changed))

    assert candidate.settings_fingerprint != baseline.settings_fingerprint


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("worker_concurrency", 2),
        ("process_timeout_seconds", 601),
    ],
)
def test_settings_fingerprint_ignores_non_model_runner_settings(
    tmp_path: Path,
    override: str,
    value: object,
) -> None:
    base = {
        "workspace_root": tmp_path,
        "runtime_root": Path("runtime/live"),
    }
    changed = dict(base)
    changed[override] = value

    baseline = build_production_model_identity_report(Settings(**base))
    candidate = build_production_model_identity_report(Settings(**changed))

    assert candidate.settings_fingerprint == baseline.settings_fingerprint


def test_baidu_endpoint_trailing_slash_changes_settings_fingerprint(
    tmp_path: Path,
) -> None:
    endpoint = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"
    baseline = build_production_model_identity_report(
        Settings(workspace_root=tmp_path, baidu_ocr_endpoint=endpoint),
    )
    with_slash = build_production_model_identity_report(
        Settings(workspace_root=tmp_path, baidu_ocr_endpoint=f"{endpoint}/"),
    )

    assert with_slash.settings_fingerprint != baseline.settings_fingerprint


def test_settings_fingerprint_records_faster_whisper_cache_subdirectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import video_demo.application.composition as composition

    encoded_payloads: list[bytes] = []
    real_sha256 = composition.hashlib.sha256

    def capture_sha256(payload: bytes = b"") -> object:
        encoded_payloads.append(payload)
        return real_sha256(payload)

    monkeypatch.setattr(composition.hashlib, "sha256", capture_sha256)

    build_production_model_identity_report(
        Settings(
            workspace_root=tmp_path,
            runtime_root=Path("runtime/live"),
        ),
    )

    assert len(encoded_payloads) == 1
    payload = json.loads(encoded_payloads[0])
    assert payload["model_cache"]["faster_whisper_root"] == (
        "runtime/live/models/faster-whisper"
    )
    assert payload["model_cache"]["pyannote_root"] == "runtime/live/models/pyannote"


def test_settings_fingerprint_ignores_secrets_and_absolute_workspace_location(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    common = {
        "runtime_root": Path("runtime/live"),
        "ffmpeg_path": Path("tools/ffmpeg"),
        "ffprobe_path": Path("tools/ffprobe"),
        "qwen_base_url": "https://dashscope.example/compatible-mode/v1/",
        "qwen_model_id": "qwen3-vl-plus",
    }
    first = Settings(
        workspace_root=first_root,
        qwen_api_key="first-qwen-secret",
        baidu_api_key="first-baidu-api",
        baidu_secret_key="first-baidu-secret",
        huggingface_token="first-hf-secret",
        **common,
    )
    second = Settings(
        workspace_root=second_root,
        qwen_api_key="second-qwen-secret",
        baidu_api_key="second-baidu-api",
        baidu_secret_key="second-baidu-secret",
        huggingface_token="second-hf-secret",
        **common,
    )

    first_report = build_production_model_identity_report(first)
    second_report = build_production_model_identity_report(second)

    assert first_report.settings_fingerprint == second_report.settings_fingerprint


@pytest.mark.parametrize(
    "model_id",
    (
        "data:text/plain;base64,c2VjcmV0",
        "/absolute/model/path",
        "../escaped-model",
        "qwen model with spaces",
        "token=secret",
    ),
)
def test_model_identity_report_rejects_unstable_qwen_model_id(
    tmp_path: Path,
    model_id: str,
) -> None:
    settings = Settings(workspace_root=tmp_path, qwen_model_id=model_id)

    with pytest.raises((ValueError, ValidationError)):
        build_production_model_identity_report(settings)


def test_identity_builder_does_not_accept_caller_supplied_models(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path)

    assert tuple(inspect.signature(build_production_model_identity_report).parameters) == (
        "settings",
    )
    with pytest.raises(TypeError):
        build_production_model_identity_report(settings, models=())  # type: ignore[call-arg]
