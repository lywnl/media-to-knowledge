from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import Field, TypeAdapter, ValidationError, model_validator

from video_demo.api.app import create_app
from video_demo.api.schemas import PublicEvidence, PublicKeyframeEvidence
from video_demo.application.composition import (
    build_production_model_identity_report,
    build_worker,
)
from video_demo.audio.yamnet import import_tensorflow_hub
from video_demo.config import Settings
from video_demo.domain.base import FrozenModel, Sha256, StableId
from video_demo.domain.evidence import EvidenceItem, KeyframeEvidence
from video_demo.domain.result import VideoUnderstandingResult
from video_demo.domain.result_artifact import ResultArtifactPayload
from video_demo.domain.run import ModelIdentity
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.evaluation.annotations import (
    SemanticJudgment,
    ValidatedEvaluationPackage,
    load_evaluation_package,
    load_semantic_judgment,
    reverify_evaluation_package,
)
from video_demo.evaluation.dataset import EvaluationSample, _safe_relative_file
from video_demo.evaluation.predictions import (
    EvaluationPrediction,
    PredictionRunSnapshot,
    load_verified_prediction,
)
from video_demo.evaluation.quality_runner import score_quality
from video_demo.evaluation.report import BoundQualityReport, GateStatus
from video_demo.persistence.repositories import Scope, VideoRunRepository
from video_demo.storage.workspace import safe_runtime_path, validate_path_component

_PUBLIC_EVIDENCE_ADAPTER: TypeAdapter[tuple[PublicEvidence, ...]] = TypeAdapter(
    tuple[PublicEvidence, ...]
)
_EVIDENCE_ADAPTER: TypeAdapter[tuple[EvidenceItem, ...]] = TypeAdapter(
    tuple[EvidenceItem, ...]
)
_SUCCESS_STATUSES = frozenset({"SUCCEEDED", "PARTIAL_SUCCEEDED"})
_FAILURE_STATUSES = frozenset({"FAILED", "CANCELLED"})
_TerminalStatus = Literal["SUCCEEDED", "PARTIAL_SUCCEEDED", "FAILED", "CANCELLED"]
_DEFAULT_SCOPE_HEADERS = {
    "X-Tenant-Id": "evaluation",
    "X-Application-Id": "video-demo",
}
_IMPLEMENTATION_FILES = (
    Path("src/video_demo/evaluation/prediction_runner.py"),
    Path("src/video_demo/evaluation/predictions.py"),
    Path("src/video_demo/evaluation/quality_runner.py"),
    Path("src/video_demo/application/composition.py"),
    Path("src/video_demo/application/pipeline.py"),
    Path("src/video_demo/application/queries.py"),
    Path("src/video_demo/api/app.py"),
    Path("src/video_demo/api/objects.py"),
    Path("src/video_demo/api/runs.py"),
    Path("src/video_demo/api/jobs.py"),
)


class PredictionRunReport(FrozenModel):
    schema_version: str = Field(pattern=r"^1\.0\.0$")
    evaluation_run_id: StableId
    status: GateStatus
    dataset_sha256: Sha256
    authorization_sha256: Sha256
    implementation_sha256: Sha256
    settings_fingerprint: Sha256
    prediction_index_sha256: Sha256 | None = None
    predictions: tuple[EvaluationPrediction, ...]
    not_run_reason: str | None = Field(default=None, max_length=500)
    started_at: datetime
    finished_at: datetime

    @model_validator(mode="after")
    def validate_report_contract(self) -> PredictionRunReport:
        if self.started_at.tzinfo is None or self.finished_at.tzinfo is None:
            raise ValueError("预测报告时间必须包含时区")
        if self.finished_at < self.started_at:
            raise ValueError("预测报告结束时间不得早于开始时间")
        sample_ids = tuple(item.sample_id for item in self.predictions)
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("预测报告不得包含重复样本")
        if self.status == GateStatus.NOT_RUN:
            if self.predictions or self.prediction_index_sha256 is not None:
                raise ValueError("NOT_RUN 预测报告不得包含样本预测")
            if not self.not_run_reason:
                raise ValueError("NOT_RUN 预测报告必须提供稳定原因")
        else:
            if not self.predictions or self.prediction_index_sha256 is None:
                raise ValueError("已执行预测报告必须绑定全部预测")
            if self.not_run_reason is not None:
                raise ValueError("已执行预测报告不得包含 NOT_RUN 原因")
            has_failure = any(
                item.terminal_status in _FAILURE_STATUSES
                for item in self.predictions
            )
            expected = GateStatus.FAIL if has_failure else GateStatus.PASS
            if self.status != expected:
                raise ValueError("预测报告状态与样本终态不一致")
        return self


class PredictionRunner:
    """通过产品 HTTP/API、Worker 和查询接口导出可重验预测。"""

    def __init__(
        self,
        settings: Settings,
        *,
        app_factory: Callable[[Settings], FastAPI] = create_app,
        worker_factory: Callable[..., Any] = build_worker,
        preflight: Callable[[], str | None] | None = None,
        scope_headers: Mapping[str, str] | None = None,
        knowledge_base_id: str = "evaluation",
        worker_id: str = "evaluation-worker",
    ) -> None:
        self._settings = settings
        self._app_factory = app_factory
        self._worker_factory = worker_factory
        self._preflight = preflight
        self._scope_headers = dict(scope_headers or _DEFAULT_SCOPE_HEADERS)
        self._knowledge_base_id = validate_path_component(
            knowledge_base_id,
            "knowledge_base_id",
        )
        self._worker_id = validate_path_component(worker_id, "worker_id")

    def preflight_reason(self) -> str | None:
        if self._preflight is not None:
            return self._preflight()
        return self._default_preflight_reason()

    def predict(
        self,
        package: ValidatedEvaluationPackage,
        *,
        evaluation_run_id: str,
    ) -> PredictionRunReport:
        validate_path_component(evaluation_run_id, "evaluation_run_id")
        started_at = datetime.now(UTC)
        verified_package = reverify_evaluation_package(package)
        if (
            verified_package.workspace_root is None
            or verified_package.runtime_root is None
        ):
            raise VideoDemoError(
                ErrorCode.EVALUATION_DATASET_INVALID,
                "评测包缺少可信工作区来源",
            )
        runtime_root = verified_package.runtime_root
        assert verified_package.workspace_root is not None
        assert verified_package.runtime_root is not None
        implementation_sha256 = _implementation_sha256(verified_package.workspace_root)
        identity = build_production_model_identity_report(self._settings)
        existing = self._load_existing_report(
            evaluation_run_id,
            verified_package,
            implementation_sha256,
            identity.settings_fingerprint,
        )
        if existing is not None:
            return existing
        reason = self.preflight_reason()
        if reason is not None:
            report = PredictionRunReport(
                schema_version="1.0.0",
                evaluation_run_id=evaluation_run_id,
                status=GateStatus.NOT_RUN,
                dataset_sha256=verified_package.dataset_sha256,
                authorization_sha256=verified_package.authorization_sha256,
                implementation_sha256=implementation_sha256,
                settings_fingerprint=identity.settings_fingerprint,
                prediction_index_sha256=None,
                predictions=(),
                not_run_reason=reason,
                started_at=started_at,
                finished_at=datetime.now(UTC),
            )
            self._write_report(evaluation_run_id, report, runtime_root)
            return report

        predictions: list[EvaluationPrediction] = []
        try:
            app = self._app_factory(self._settings)
            with TestClient(app) as client:
                worker = self._worker_factory(self._settings, worker_id=self._worker_id)
                try:
                    for sample in verified_package.dataset.samples:
                        predictions.append(
                            self._predict_sample(
                                client,
                                worker,
                                sample,
                                verified_package,
                                evaluation_run_id,
                                identity.models,
                            )
                        )
                finally:
                    close = getattr(worker, "close", None)
                    if callable(close):
                        close()
        except (OSError, ValueError, ValidationError, VideoDemoError):
            if not predictions:
                predictions.append(
                    self._failed_prediction_without_run(
                        verified_package.dataset.samples[0],
                        evaluation_run_id,
                        "PRODUCTION_API_UNAVAILABLE",
                        identity.models,
                        eval_root=runtime_root / "eval",
                    )
                )
            else:
                predictions.extend(
                    self._failed_prediction_without_run(
                        sample,
                        evaluation_run_id,
                        "PRODUCTION_API_UNAVAILABLE",
                        identity.models,
                        eval_root=runtime_root / "eval",
                    )
                    for sample in verified_package.dataset.samples[len(predictions) :]
                )
        report = PredictionRunReport(
            schema_version="1.0.0",
            evaluation_run_id=evaluation_run_id,
            status=(
                GateStatus.FAIL
                if any(item.terminal_status in _FAILURE_STATUSES for item in predictions)
                else GateStatus.PASS
            ),
            dataset_sha256=verified_package.dataset_sha256,
            authorization_sha256=verified_package.authorization_sha256,
            implementation_sha256=implementation_sha256,
            settings_fingerprint=identity.settings_fingerprint,
            prediction_index_sha256=_prediction_index_digest(
                runtime_root / "eval", evaluation_run_id, predictions
            ),
            predictions=tuple(predictions),
            not_run_reason=None,
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )
        self._write_report(evaluation_run_id, report, runtime_root)
        return report

    def _default_preflight_reason(self) -> str | None:
        try:
            from video_demo.capabilities import probe_runtime_capabilities

            capabilities = probe_runtime_capabilities(self._settings)
            if capabilities.issues:
                return capabilities.issues[0].code.value
            if not _has_dependency("faster_whisper"):
                return "FASTER_WHISPER_DEPENDENCY_UNAVAILABLE"
            if not _has_dependency("whisperx"):
                return "WHISPERX_DEPENDENCY_UNAVAILABLE"
            if not _has_dependency("pyannote.audio"):
                return "PYANNOTE_DEPENDENCY_UNAVAILABLE"
            if not _has_dependency("silero_vad"):
                return "SILERO_DEPENDENCY_UNAVAILABLE"
            if not _has_dependency("tensorflow_hub"):
                return "YAMNET_DEPENDENCY_UNAVAILABLE"
            if self._settings.qwen_api_key is None or not self._settings.qwen_base_url:
                return "QWEN_CREDENTIALS_UNAVAILABLE"
            if self._settings.baidu_api_key is None or self._settings.baidu_secret_key is None:
                return "BAIDU_OCR_CREDENTIALS_UNAVAILABLE"
            if self._settings.huggingface_token is None:
                return "PYANNOTE_TOKEN_UNAVAILABLE"
        except (OSError, ValueError, VideoDemoError):
            return "EVALUATION_PREFLIGHT_INVALID"
        return None

    def _predict_sample(
        self,
        client: TestClient,
        worker: Any,
        sample: EvaluationSample,
        package: ValidatedEvaluationPackage,
        evaluation_run_id: str,
        models: tuple[Any, ...],
    ) -> EvaluationPrediction:
        started_at = datetime.now(UTC)
        if package.runtime_root is None or package.workspace_root is None:
            raise VideoDemoError(
                ErrorCode.EVALUATION_DATASET_INVALID,
                "评测包缺少可信运行根",
            )
        runtime_root = package.runtime_root
        run_id: str | None = None
        job_id: str | None = None
        run_payload: dict[str, object] | None = None
        try:
            media_path = _safe_relative_file(
                package.dataset.eval_root,
                sample.media_relative_path,
                runtime_root,
            )
            mime = _mime_for_path(media_path)
            with media_path.open("rb") as stream:
                upload = client.post(
                    f"/api/kb/knowledge-bases/{self._knowledge_base_id}/video-objects",
                    headers=self._scope_headers,
                    files={"file": (media_path.name, stream, mime)},
                )
            _require_status(upload, 201)
            object_ref = str(upload.json()["object_ref"])
            idempotency = _idempotency_key(evaluation_run_id, sample.sample_id)
            created = client.post(
                f"/api/kb/knowledge-bases/{self._knowledge_base_id}/video-understanding-runs",
                headers=self._scope_headers,
                json={
                    "object_ref": object_ref,
                    "idempotency_key": idempotency,
                    "language_hints": [sample.language],
                    "min_speakers": None,
                    "max_speakers": None,
                },
            )
            _require_status(created, 202)
            created_payload = created.json()
            run_id = str(created_payload["run_id"])
            job_id = str(created_payload["job_id"])
            if not worker.run_once():
                return self._failed_prediction(
                    sample,
                    evaluation_run_id,
                    run_id,
                    job_id,
                    "WORKER_NO_JOB",
                    models,
                    started_at,
                    eval_root=runtime_root / "eval",
                )
            run_response = client.get(
                f"/api/kb/knowledge-bases/{self._knowledge_base_id}/video-understanding-runs/{run_id}",
                headers=self._scope_headers,
            )
            _require_status(run_response, 200)
            run_payload = run_response.json()
            status = str(run_payload["status"])
            job_response = client.get(
                f"/api/kb/jobs/{job_id}",
                headers=self._scope_headers,
                params={"knowledge_base_id": self._knowledge_base_id},
            )
            _require_status(job_response, 200)
            if status not in _SUCCESS_STATUSES:
                return self._failed_prediction(
                    sample,
                    evaluation_run_id,
                    run_id,
                    job_id,
                    str(run_payload.get("error_code") or "RUN_NOT_SUCCEEDED"),
                    models,
                    started_at,
                    eval_root=runtime_root / "eval",
                    run_payload=run_payload,
                )
            result_response = client.get(
                f"/api/kb/knowledge-bases/{self._knowledge_base_id}/video-understanding-runs/{run_id}/result",
                headers=self._scope_headers,
            )
            _require_status(result_response, 200)
            result = _parse_api_result(result_response.json())
            evidence = self._fetch_evidence(client, run_id, evaluation_run_id, sample.sample_id)
            _production_manifest_bytes, artifact_payload = _read_published_manifest(
                client,
                run_id=run_id,
                scope=Scope(
                    self._scope_headers["X-Tenant-Id"],
                    self._scope_headers["X-Application-Id"],
                    self._knowledge_base_id,
                ),
            )
            if (
                artifact_payload.result != result
                or not _evidence_matches_api(artifact_payload.evidence, evidence)
                or artifact_payload.status != status
            ):
                raise ValueError("查询结果与生产 artifact manifest 不一致")
            evidence = _rebind_evidence_to_manifest(
                evidence,
                artifact_payload.evidence,
                eval_root=runtime_root / "eval",
            )
            export_manifest_bytes = _export_manifest_bytes(
                artifact_payload,
                evidence,
                sample.media_sha256,
            )
            _write_sample_prediction(
                runtime_root / "eval",
                evaluation_run_id,
                sample,
                result,
                evidence,
                status,
                _warning_codes_from_payload(run_payload),
                run_payload,
                models,
                manifest_bytes=export_manifest_bytes,
            )
            return _load_index_prediction(
                runtime_root / "eval",
                evaluation_run_id,
                sample,
            )
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            ValidationError,
            VideoDemoError,
        ) as error:
            code = (
                error.code.value
                if isinstance(error, VideoDemoError)
                else "API_UNEXPECTED_RESPONSE"
            )
            return self._failed_prediction_without_run(
                sample,
                evaluation_run_id,
                code,
                models,
                started_at,
                eval_root=runtime_root / "eval",
                run_id=run_id,
                job_id=job_id,
                run_payload=run_payload,
            )

    def _fetch_evidence(
        self,
        client: TestClient,
        run_id: str,
        evaluation_run_id: str,
        sample_id: str,
    ) -> tuple[EvidenceItem, ...]:
        values: list[dict[str, object]] = []
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            params: dict[str, str | int] = {"limit": 100}
            if cursor is not None:
                if cursor in seen:
                    raise ValueError("证据分页游标重复")
                seen.add(cursor)
                params["cursor"] = cursor
            response = client.get(
                f"/api/kb/knowledge-bases/{self._knowledge_base_id}/video-understanding-runs/{run_id}/evidence",
                headers=self._scope_headers,
                params=params,
            )
            _require_status(response, 200)
            payload = response.json()
            page = payload.get("items")
            if not isinstance(page, list):
                raise ValueError("证据分页响应非法")
            values.extend(page)
            cursor = payload.get("next_cursor")
            if cursor is None:
                break
            if not isinstance(cursor, str):
                raise ValueError("证据分页游标非法")
        public_items = _PUBLIC_EVIDENCE_ADAPTER.validate_python(
            _normalize_public_evidence(values)
        )
        result: list[EvidenceItem] = []
        for public_item in public_items:
            if isinstance(public_item, PublicKeyframeEvidence):
                result.append(
                    KeyframeEvidence(
                        evidence_id=public_item.evidence_id,
                        start_ms=public_item.start_ms,
                        end_ms=public_item.end_ms,
                        keyframe_id=public_item.keyframe_id,
                        timestamp_ms=public_item.timestamp_ms,
                        relative_path=(
                            f"predictions/{evaluation_run_id}/{sample_id}/keyframes/"
                            f"{public_item.keyframe_id}."
                            f"{'png' if public_item.mime_type == 'image/png' else 'jpg'}"
                        ),
                        mime_type=public_item.mime_type,
                        sha256=public_item.sha256,
                        perceptual_hash=public_item.perceptual_hash,
                    )
                )
            else:
                parsed_item: EvidenceItem = cast(
                    EvidenceItem,
                    public_item.model_dump(mode="python"),
                )
                result.append(parsed_item)
        for item in result:
            if isinstance(item, KeyframeEvidence):
                if self._settings.runtime_root is None:
                    raise VideoDemoError(
                        ErrorCode.EVALUATION_ARTIFACT_INVALID,
                        "预测缺少可信运行根",
                    )
                content = client.get(
                    f"/api/kb/knowledge-bases/{self._knowledge_base_id}/video-understanding-runs/{run_id}/keyframes/{item.keyframe_id}/content",
                    headers=self._scope_headers,
                )
                _require_status(content, 200)
                if not content.headers.get("content-type", "").split(";", 1)[0] == item.mime_type:
                    raise ValueError("关键帧 MIME 与 API 响应不一致")
                if hashlib.sha256(content.content).hexdigest() != item.sha256:
                    raise ValueError("关键帧摘要与 API 响应不一致")
                _atomic_write_bytes(
                    self._settings.runtime_root / "eval" / item.relative_path,
                    content.content,
                )
        return tuple(result)

    def _load_existing_report(
        self,
        evaluation_run_id: str,
        package: ValidatedEvaluationPackage,
        implementation_sha256: str,
        settings_fingerprint: str,
    ) -> PredictionRunReport | None:
        assert package.runtime_root is not None
        assert package.workspace_root is not None
        path = package.runtime_root / "eval" / "reports" / evaluation_run_id / "prediction.json"
        if not path.exists():
            prediction_root = package.runtime_root / "eval" / "predictions" / evaluation_run_id
            if prediction_root.exists() and any(prediction_root.iterdir()):
                raise VideoDemoError(ErrorCode.IDEMPOTENCY_CONFLICT, "预测目录已有不完整证据")
            return None
        try:
            report = PredictionRunReport.model_validate_json(path.read_bytes())
            if (
                report.dataset_sha256 != package.dataset_sha256
                or report.authorization_sha256 != package.authorization_sha256
                or report.implementation_sha256 != implementation_sha256
                or report.settings_fingerprint != settings_fingerprint
            ):
                raise VideoDemoError(ErrorCode.IDEMPOTENCY_CONFLICT, "预测输入或实现身份已变化")
            if report.status != GateStatus.NOT_RUN:
                by_id = {item.sample_id: item for item in package.dataset.samples}
                expected_ids = tuple(by_id)
                supplied_ids = tuple(item.sample_id for item in report.predictions)
                if len(supplied_ids) != len(set(supplied_ids)) or set(supplied_ids) != set(
                    expected_ids
                ):
                    raise ValueError("预测报告未恰好覆盖当前数据集")
                for prediction in report.predictions:
                    load_verified_prediction(
                        _prediction_index_path(
                            package.runtime_root / "eval",
                            report.evaluation_run_id,
                            prediction.sample_id,
                        ),
                        eval_root=package.runtime_root / "eval",
                        workspace_root=package.workspace_root,
                        runtime_root=package.runtime_root,
                        sample=by_id[prediction.sample_id],
                    )
                if _prediction_index_digest(
                    package.runtime_root / "eval",
                    report.evaluation_run_id,
                    report.predictions,
                ) != report.prediction_index_sha256:
                    raise ValueError("预测索引摘要与当前产物不匹配")
            return report
        except (OSError, ValueError, ValidationError, KeyError):
            raise VideoDemoError(
                ErrorCode.EVALUATION_ARTIFACT_INVALID,
                "预测报告非法或无法重验",
            ) from None

    def _write_report(
        self,
        evaluation_run_id: str,
        report: PredictionRunReport,
        runtime_root: Path,
    ) -> None:
        _atomic_write_bytes(
            runtime_root / "eval" / "reports" / evaluation_run_id / "prediction.json",
            report.model_dump_json(exclude_none=True).encode("utf-8"),
        )

    @staticmethod
    def _failed_prediction(
        sample: EvaluationSample,
        evaluation_run_id: str,
        run_id: str,
        job_id: str,
        failure_code: str,
        models: tuple[ModelIdentity, ...],
        started_at: datetime,
        *,
        eval_root: Path,
        terminal_status: _TerminalStatus = "FAILED",
        current_stage: str = "REGISTER",
        run_payload: dict[str, object] | None = None,
    ) -> EvaluationPrediction:
        return _persist_failed_prediction(
            eval_root,
            evaluation_run_id=evaluation_run_id,
            sample=sample,
            run_id=run_id,
            job_id=job_id,
            terminal_status=terminal_status,
            current_stage=current_stage,
            failure_code=failure_code,
            models=models,
            started_at=started_at,
            run_payload=run_payload,
        )

    def _failed_prediction_without_run(
        self,
        sample: EvaluationSample,
        evaluation_run_id: str,
        failure_code: str,
        models: tuple[ModelIdentity, ...],
        started_at: datetime | None = None,
        *,
        eval_root: Path,
        run_id: str | None = None,
        job_id: str | None = None,
        run_payload: dict[str, object] | None = None,
    ) -> EvaluationPrediction:
        actual_run_id = run_id or f"run_failed_{sample.sample_id}"
        actual_job_id = job_id or f"job_failed_{sample.sample_id}"
        terminal_status = _terminal_status_from_payload(run_payload)
        current_stage = _stage_from_payload(run_payload)
        return self._failed_prediction(
            sample,
            evaluation_run_id,
            actual_run_id,
            actual_job_id,
            failure_code,
            models,
            started_at or datetime.now(UTC),
            eval_root=eval_root,
            terminal_status=terminal_status,
            current_stage=current_stage,
            run_payload=run_payload,
        )


def _prediction_index_path(eval_root: Path, evaluation_run_id: str, sample_id: str) -> Path:
    return eval_root / "predictions" / evaluation_run_id / sample_id / "index.json"


def _prediction_index_digest(
    eval_root: Path,
    evaluation_run_id: str,
    predictions: tuple[EvaluationPrediction, ...] | list[EvaluationPrediction],
) -> str:
    digests = []
    for prediction in predictions:
        path = _prediction_index_path(eval_root, evaluation_run_id, prediction.sample_id)
        digests.append(_sha256_file(path))
    return _canonical_digest(digests)


def _warning_codes_from_payload(payload: dict[str, object] | None) -> tuple[str, ...]:
    if payload is None:
        return ()
    warnings = payload.get("warning_codes", ())
    if not isinstance(warnings, (list, tuple)):
        return ()
    return tuple(str(item) for item in warnings)


def _terminal_status_from_payload(
    payload: dict[str, object] | None,
) -> _TerminalStatus:
    value = payload.get("status") if payload is not None else None
    return cast(_TerminalStatus, value) if value in _FAILURE_STATUSES else "FAILED"


def _stage_from_payload(payload: dict[str, object] | None) -> str:
    value = payload.get("current_stage") if payload is not None else None
    return str(value) if isinstance(value, str) and value else "REGISTER"


def _persist_failed_prediction(
    eval_root: Path,
    *,
    evaluation_run_id: str,
    sample: EvaluationSample,
    run_id: str,
    job_id: str,
    terminal_status: _TerminalStatus,
    current_stage: str,
    failure_code: str,
    models: tuple[ModelIdentity, ...],
    started_at: datetime | None = None,
    run_payload: dict[str, object] | None = None,
) -> EvaluationPrediction:
    payload_status = _terminal_status_from_payload(run_payload)
    if run_payload is not None and payload_status in _FAILURE_STATUSES:
        terminal_status = payload_status
        current_stage = _stage_from_payload(run_payload)
    error_code = (
        str(run_payload.get("error_code") or failure_code)
        if run_payload is not None
        else failure_code
    )
    snapshot = PredictionRunSnapshot(
        schema_version="1.0.0",
        run_id=run_id,
        job_id=job_id,
        terminal_status=terminal_status,
        current_stage=current_stage,
        warning_codes=_warning_codes_from_payload(run_payload),
        error_code=error_code,
        models=models,
    )
    run_bytes = snapshot.model_dump_json(exclude_none=True).encode("utf-8")
    relative = Path("predictions") / evaluation_run_id / sample.sample_id
    run_relative = (relative / "run.json").as_posix()
    _atomic_write_bytes(eval_root / run_relative, run_bytes)
    finished_at = datetime.now(UTC)
    prediction = EvaluationPrediction(
        schema_version="1.0.0",
        evaluation_run_id=evaluation_run_id,
        sample_id=sample.sample_id,
        media_sha256=sample.media_sha256,
        run_id=run_id,
        job_id=job_id,
        terminal_status=terminal_status,
        run_relative_path=run_relative,
        run_sha256=_sha256_bytes(run_bytes),
        failure_code=error_code,
        started_at=started_at or finished_at,
        finished_at=finished_at,
    )
    _atomic_write_bytes(
        eval_root / relative / "index.json",
        prediction.model_dump_json(exclude_none=True).encode("utf-8"),
    )
    return prediction


def score_prediction_run(
    evaluation_run_id: str,
    *,
    eval_root: Path,
) -> BoundQualityReport:
    """只加载已落盘预测和人工审阅并重建质量，不调用生产模型。"""

    validate_path_component(evaluation_run_id, "evaluation_run_id")
    safe_eval_root = eval_root.resolve(strict=True)
    workspace_root = safe_eval_root.parents[2]
    runtime_root = safe_eval_root.parent
    report_path = safe_eval_root / "reports" / evaluation_run_id / "prediction.json"
    try:
        report = PredictionRunReport.model_validate_json(report_path.read_bytes())
        if report.evaluation_run_id != evaluation_run_id:
            raise ValueError("预测报告评测运行 ID 不匹配")
        if report.status == GateStatus.NOT_RUN:
            raise ValueError("预测阶段未运行")
        package = load_evaluation_package(
            safe_eval_root / "dataset.jsonl",
            safe_eval_root / "authorization.json",
            workspace_root=workspace_root,
            runtime_root=runtime_root,
        )
        if (
            report.dataset_sha256 != package.dataset_sha256
            or report.authorization_sha256 != package.authorization_sha256
        ):
            raise ValueError("预测报告输入摘要与当前评测包不匹配")
        if report.implementation_sha256 != _implementation_sha256(workspace_root):
            raise ValueError("预测报告实现摘要与当前实现不匹配")
        expected_ids = {sample.sample_id for sample in package.dataset.samples}
        supplied_ids = tuple(item.sample_id for item in report.predictions)
        if len(supplied_ids) != len(set(supplied_ids)) or set(supplied_ids) != expected_ids:
            raise ValueError("预测报告未恰好覆盖当前数据集")
        by_id = {sample.sample_id: sample for sample in package.dataset.samples}
        predictions = tuple(
            load_verified_prediction(
                _prediction_index_path(safe_eval_root, evaluation_run_id, item.sample_id),
                eval_root=safe_eval_root,
                workspace_root=workspace_root,
                runtime_root=runtime_root,
                sample=by_id[item.sample_id],
            )
            for item in report.predictions
        )
        if tuple(item.index for item in predictions) != report.predictions:
            raise ValueError("预测报告与索引内容不一致")
        current_index_digest = _prediction_index_digest(
            safe_eval_root,
            evaluation_run_id,
            report.predictions,
        )
        if current_index_digest != report.prediction_index_sha256:
            raise ValueError("预测索引摘要与当前产物不匹配")
        judgments: list[SemanticJudgment] = []
        for prediction in predictions:
            path = (
                safe_eval_root
                / "judgments"
                / evaluation_run_id
                / f"{prediction.index.sample_id}.json"
            )
            if path.is_file():
                judgments.append(
                    load_semantic_judgment(
                        path,
                        workspace_root=workspace_root,
                        runtime_root=runtime_root,
                        annotation=next(
                            annotation
                            for annotation in package.annotations
                            if annotation.annotation.sample_id == prediction.index.sample_id
                        ),
                        prediction=prediction,
                    )
                )
        artifacts = score_quality(
            package,
            predictions,
            tuple(judgments),
            evaluation_run_id=evaluation_run_id,
        )
        _atomic_write_bytes(
            safe_eval_root / "reports" / evaluation_run_id / "quality.json",
            artifacts.report.model_dump_json(exclude_none=True).encode("utf-8"),
        )
        _atomic_write_bytes(
            safe_eval_root / "reports" / evaluation_run_id / "quality-details.json",
            json.dumps(
                [detail.model_dump(mode="json") for detail in artifacts.sample_details],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        )
        return artifacts.report
    except (OSError, ValueError, ValidationError, KeyError, StopIteration, VideoDemoError):
        raise VideoDemoError(ErrorCode.EVALUATION_ARTIFACT_INVALID, "预测或质量产物非法") from None


def _write_sample_prediction(
    eval_root: Path,
    evaluation_run_id: str,
    sample: EvaluationSample,
    result: VideoUnderstandingResult,
    evidence: tuple[EvidenceItem, ...],
    status: str,
    warnings: tuple[str, ...],
    run_payload: dict[str, object],
    models: tuple[ModelIdentity, ...],
    *,
    manifest_bytes: bytes,
) -> None:
    directory = eval_root / "predictions" / evaluation_run_id / sample.sample_id
    run_snapshot = PredictionRunSnapshot(
        schema_version="1.0.0",
        run_id=result.run_id,
        job_id=str(run_payload["job_id"]),
        terminal_status=cast(_TerminalStatus, status),
        current_stage=str(run_payload["current_stage"]),
        warning_codes=warnings,
        error_code=str(run_payload.get("error_code")) if run_payload.get("error_code") else None,
        models=tuple(models),
    )
    result_bytes = result.model_dump_json(exclude_computed_fields=True).encode("utf-8")
    evidence_bytes = b"".join(
        item.model_dump_json(exclude_computed_fields=True).encode("utf-8") + b"\n"
        for item in evidence
    )
    run_bytes = run_snapshot.model_dump_json(exclude_none=True).encode("utf-8")
    relative = Path("predictions") / evaluation_run_id / sample.sample_id
    result_relative = (relative / "result.json").as_posix()
    evidence_relative = (relative / "evidence.jsonl").as_posix()
    run_relative = (relative / "run.json").as_posix()
    manifest_relative = (relative / "artifact-manifest.json").as_posix()
    _atomic_write_bytes(directory / "run.json", run_bytes)
    _atomic_write_bytes(directory / "result.json", result_bytes)
    _atomic_write_bytes(directory / "evidence.jsonl", evidence_bytes)
    _atomic_write_bytes(directory / "artifact-manifest.json", manifest_bytes)
    index = EvaluationPrediction(
        schema_version="1.0.0",
        evaluation_run_id=evaluation_run_id,
        sample_id=sample.sample_id,
        media_sha256=sample.media_sha256,
        run_id=result.run_id,
        job_id=str(run_payload["job_id"]),
        terminal_status=cast(_TerminalStatus, status),
        run_relative_path=run_relative,
        run_sha256=_sha256_bytes(run_bytes),
        result_relative_path=result_relative,
        result_sha256=_sha256_bytes(result_bytes),
        evidence_relative_path=evidence_relative,
        evidence_sha256=_sha256_bytes(evidence_bytes),
        artifact_manifest_relative_path=manifest_relative,
        artifact_manifest_sha256=_sha256_bytes(manifest_bytes),
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )
    _atomic_write_bytes(
        directory / "index.json",
        index.model_dump_json(exclude_none=True).encode("utf-8"),
    )


def _read_published_manifest(
    client: TestClient,
    *,
    run_id: str,
    scope: Scope,
) -> tuple[bytes, ResultArtifactPayload]:
    container = cast(Any, client.app).state.container
    payload = container.result_query_service._read_bundle(scope, run_id)
    with container.database.session() as session:
        run = VideoRunRepository(session).get(scope, run_id)
        if run is None or run.artifact_manifest_relative_path is None:
            raise ValueError("生产结果 Manifest 不存在")
        relative_path = Path(run.artifact_manifest_relative_path)
        expected_sha256 = run.artifact_manifest_sha256
    path = safe_runtime_path(container.settings.runtime_root, relative_path)
    manifest_bytes = path.read_bytes()
    if expected_sha256 is None or _sha256_bytes(manifest_bytes) != expected_sha256:
        raise ValueError("生产结果 Manifest 摘要不匹配")
    return manifest_bytes, ResultArtifactPayload.model_validate(payload)


def _parse_api_result(payload: object) -> VideoUnderstandingResult:
    if not isinstance(payload, dict):
        raise ValueError("结果 API 响应必须是对象")
    normalized = json.loads(json.dumps(payload, ensure_ascii=False))
    segments = normalized.get("segments")
    summary = normalized.get("summary")
    if isinstance(segments, list):
        for segment in segments:
            if isinstance(segment, dict):
                segment.pop("duration_ms", None)
    if isinstance(summary, dict):
        chapters = summary.get("chapters")
        if isinstance(chapters, list):
            for chapter in chapters:
                if isinstance(chapter, dict):
                    chapter.pop("duration_ms", None)
    return VideoUnderstandingResult.model_validate(normalized)


def _normalize_public_evidence(values: list[dict[str, object]]) -> list[dict[str, object]]:
    """移除 API 序列化的 computed duration 字段，再按严格公共 Schema 解析。"""

    normalized: list[dict[str, object]] = []
    for value in values:
        item = dict(value)
        item.pop("duration_ms", None)
        normalized.append(item)
    return normalized


def _evidence_matches_api(
    left: tuple[EvidenceItem, ...],
    right: tuple[EvidenceItem, ...] | list[dict[str, object]],
) -> bool:
    if len(left) != len(right):
        return False
    for expected, actual in zip(left, right, strict=True):
        expected_payload = expected.model_dump(mode="json")
        actual_payload = (
            actual.model_dump(mode="json")
            if isinstance(actual, EvidenceItem)
            else dict(actual)
        )
        expected_payload.pop("relative_path", None)
        actual_payload.pop("relative_path", None)
        if expected_payload != actual_payload:
            return False
    return True


def _rebind_evidence_to_manifest(
    api_evidence: tuple[EvidenceItem, ...] | list[dict[str, object]],
    manifest_evidence: tuple[EvidenceItem, ...],
    *,
    eval_root: Path,
) -> tuple[EvidenceItem, ...]:
    manifest_by_id = {item.evidence_id: item for item in manifest_evidence}
    rebound: list[EvidenceItem] = []
    for item in api_evidence:
        evidence_id = (
            item.evidence_id
            if isinstance(item, EvidenceItem)
            else str(item.get("evidence_id"))
        )
        source = manifest_by_id.get(evidence_id)
        if source is None:
            raise ValueError("API 证据不在生产 Manifest 中")
        if isinstance(item, dict) and isinstance(source, KeyframeEvidence):
            relative_path = str(item.get("relative_path", ""))
            path = eval_root / relative_path
            if not path.is_file() or _sha256_file(path) != source.sha256:
                raise ValueError("关键帧导出内容与生产摘要不一致")
            rebound.append(source.model_copy(update={"relative_path": relative_path}))
        else:
            rebound.append(source)
    if len(rebound) != len(manifest_evidence):
        raise ValueError("API 证据未完整覆盖生产 Manifest")
    return tuple(rebound)


def _export_manifest_bytes(
    source: ResultArtifactPayload,
    evidence: tuple[EvidenceItem, ...],
    upstream_sha256: str,
) -> bytes:
    """保留生产阶段指标，只替换评测导出的关键帧相对路径。"""

    payload = ResultArtifactPayload(
        result=source.result,
        evidence=evidence,
        stage_metrics=source.stage_metrics,
        status=source.status,
        warnings=source.warnings,
    )
    return _envelope_bytes(
        payload.model_dump(mode="json", exclude_computed_fields=True),
        upstream_sha256,
    )


def _load_index_prediction(
    eval_root: Path,
    evaluation_run_id: str,
    sample: EvaluationSample,
) -> EvaluationPrediction:
    index_path = eval_root / "predictions" / evaluation_run_id / sample.sample_id / "index.json"
    return EvaluationPrediction.model_validate_json(index_path.read_bytes())


def _envelope_bytes(payload: dict[str, object], upstream_sha256: str) -> bytes:
    return json.dumps(
        {
            "schema_version": "1.0.0",
            "upstream_sha256": upstream_sha256,
            "payload": payload,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        with temporary.open("wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _implementation_sha256(workspace_root: Path | None) -> str:
    if workspace_root is None:
        raise VideoDemoError(ErrorCode.EVALUATION_ARTIFACT_INVALID, "预测缺少工作区来源")
    entries: list[dict[str, str]] = []
    for relative in _IMPLEMENTATION_FILES:
        path = workspace_root / relative
        if not path.is_file():
            raise VideoDemoError(ErrorCode.EVALUATION_ARTIFACT_INVALID, "预测实现文件缺失")
        entries.append({"path": relative.as_posix(), "sha256": _sha256_file(path)})
    return _canonical_digest(entries)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_digest(value: object) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _has_dependency(name: str) -> bool:
    try:
        if importlib.util.find_spec(name) is None:
            return False
        if name == "tensorflow_hub":
            import_tensorflow_hub(importer=importlib.import_module)
        else:
            importlib.import_module(name)
        return True
    except (ImportError, ModuleNotFoundError, AttributeError, OSError, ValueError):
        return False


def _mime_for_path(path: Path) -> str:
    return {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
    }.get(path.suffix.lower(), "application/octet-stream")


def _idempotency_key(evaluation_run_id: str, sample_id: str) -> str:
    return f"eval-{_sha256_text(f'{evaluation_run_id}:{sample_id}')[:40]}"


def _require_status(response: Any, expected: int) -> None:
    if response.status_code != expected:
        raise VideoDemoError(ErrorCode.SYSTEM_FAILURE, "生产 API 返回非预期状态")
