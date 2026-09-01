from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_demo.application.image_pipeline_executor import ImageStagePipelineExecutor
from video_demo.application.image_pipeline_handler import ImageJobHandler
from video_demo.application.image_rendering import render_image_markdown
from video_demo.application.media_publication import MediaPublicationService
from video_demo.application.pipeline_contracts import PipelineRunConfig
from video_demo.domain.image_document import ImageDocument
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.database import Database
from video_demo.persistence.media_repositories import MediaObjectRepository, MediaRunRepository
from video_demo.persistence.models import (
    ImageObjectModel,
    ImageUnderstandingRunModel,
    JobStatus,
)
from video_demo.persistence.repositories import ClaimedJob, JobRepository
from video_demo.persistence.scope import Scope
from video_demo.storage.artifacts import AtomicArtifactStore
from video_demo.storage.image_object_store import ImageObjectStore

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+AAAAAgABSK+kcQAAAABJRU5ErkJggg==",
)


class _Analyzer:
    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, *, image_data_url: str, title_hint: str) -> ImageDocument:
        self.calls += 1
        assert image_data_url.startswith("data:image/png;base64,")
        return ImageDocument(
            title=title_hint,
            overview_zh="图片概览",
            content_blocks=(),
            claims=(),
            evidence_refs=(),
            content_status="DEGRADED",
        )


@dataclass
class _Publication:
    calls: int = 0

    def persist(self, *args, **kwargs) -> None:
        del args, kwargs
        self.calls += 1


def _handler_for_cancellation(tmp_path: Path, publication: _Publication) -> ImageJobHandler:
    handler = ImageJobHandler(
        _DatabaseStub(),
        lambda: _Analyzer(),
        publication,
        object(),
        runtime_root=tmp_path,
        max_image_bytes=8 * 1024 * 1024,
    )
    handler._mark_running = lambda _job: None  # type: ignore[method-assign]
    handler._mark_stage = lambda _job, _stage: None  # type: ignore[method-assign]
    handler._load_input = lambda _job: (  # type: ignore[method-assign]
        tmp_path / "source.png",
        PipelineRunConfig(),
        "source.png",
        "a" * 64,
        "runs/test/source.png",
        "image/png",
    )
    return handler


class _DatabaseStub:
    def session(self):
        raise AssertionError("取消边界测试不应访问数据库")


def test_image_executor_publishes_json_and_markdown(tmp_path: Path) -> None:
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'image-publish.db'}")
    database.create_schema()
    scope = Scope("tenant", "application", "kb")
    object_store = ImageObjectStore(tmp_path, max_bytes=8 * 1024 * 1024)
    record = object_store.ingest(BytesIO(_PNG), "pixel.png", "image/png", scope)
    run_id = "run_image_publish"
    with database.session() as session:
        MediaObjectRepository(session, ImageObjectModel).add_ready(
            scope=scope,
            object_ref=record.object_ref,
            original_filename=record.original_filename,
            declared_mime=record.declared_mime,
            detected_mime=record.detected_mime,
            size_bytes=record.size_bytes,
            sha256=record.sha256,
            relative_path=record.relative_path,
        )
        MediaRunRepository(session, ImageUnderstandingRunModel).add(
            scope=scope,
            run_id=run_id,
            object_ref=record.object_ref,
            idempotency_key="image-publish-001",
            config_snapshot=PipelineRunConfig().model_dump(mode="json"),
        )
        JobRepository(session).enqueue_media_run(
            scope=scope,
            job_id="job_image_publish",
            resource_id=run_id,
            job_type="IMAGE_UNDERSTANDING",
            resource_type="IMAGE_UNDERSTANDING_RUN",
        )

    publication = MediaPublicationService(
        database,
        AtomicArtifactStore(tmp_path),
        run_model=ImageUnderstandingRunModel,
        result_type=__import__(
            "video_demo.domain.image_document",
            fromlist=["ImageUnderstandingResult"],
        ).ImageUnderstandingResult,
        render=render_image_markdown,
        resource_type="IMAGE_UNDERSTANDING_RUN",
        not_found_code=ErrorCode.IMAGE_RUN_NOT_FOUND,
        max_document_bytes=1024 * 1024,
        max_bundle_bytes=1024 * 1024,
    )
    analyzer = _Analyzer()
    handler = ImageJobHandler(
        database,
        lambda: analyzer,
        publication,
        object_store,
        runtime_root=tmp_path,
        max_image_bytes=8 * 1024 * 1024,
    )
    executor = ImageStagePipelineExecutor(database, handler, runtime_root=tmp_path)

    executor.run(scope, run_id)

    with database.session() as session:
        job = JobRepository(session).get_by_resource_type(
            scope,
            run_id,
            "IMAGE_UNDERSTANDING_RUN",
        )
        run = MediaRunRepository(session, ImageUnderstandingRunModel).get(scope, run_id)
        assert job is not None
        assert job.status == JobStatus.SUCCEEDED
        assert run is not None
        assert run.artifact_relative_path is not None
        assert run.document_relative_path is not None

    published = publication.get(scope, run_id)
    assert analyzer.calls == 1
    assert published.result.document.overview_zh == "图片概览"
    assert published.document.decode("utf-8").startswith("# pixel.png")


def test_image_handler_cancels_before_vlm_request(tmp_path: Path, monkeypatch) -> None:
    publication = _Publication()
    analyzer = _Analyzer()
    handler = ImageJobHandler(
        _DatabaseStub(),
        lambda: analyzer,
        publication,
        object(),
        runtime_root=tmp_path,
        max_image_bytes=8 * 1024 * 1024,
    )
    handler._mark_running = lambda _job: None  # type: ignore[method-assign]
    handler._mark_stage = lambda _job, _stage: None  # type: ignore[method-assign]
    handler._load_input = lambda _job: (  # type: ignore[method-assign]
        tmp_path / "source.png",
        PipelineRunConfig(),
        "source.png",
        "a" * 64,
        "runs/test/source.png",
        "image/png",
    )
    (tmp_path / "source.png").write_bytes(_PNG)
    calls = 0

    def unexpected_pipeline(**kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(result=object(), document=object(), warnings=())

    monkeypatch.setattr(
        "video_demo.application.image_pipeline_handler.run_image_pipeline",
        unexpected_pipeline,
    )
    job = ClaimedJob(
        1,
        "job-cancel-before-vlm",
        "run-cancel-before-vlm",
        "owner",
        1,
        3,
        Scope("t", "a", "k"),
    )

    with pytest.raises(VideoDemoError) as error:
        handler.process(job, is_cancel_requested=lambda: True)

    assert error.value.code == ErrorCode.JOB_CANCELLED
    assert calls == 0
    assert publication.calls == 0


def test_image_handler_cancels_after_vlm_before_publish(tmp_path: Path, monkeypatch) -> None:
    publication = _Publication()
    analyzer = _Analyzer()
    handler = ImageJobHandler(
        _DatabaseStub(),
        lambda: analyzer,
        publication,
        object(),
        runtime_root=tmp_path,
        max_image_bytes=8 * 1024 * 1024,
    )
    handler._mark_running = lambda _job: None  # type: ignore[method-assign]
    handler._mark_stage = lambda _job, _stage: None  # type: ignore[method-assign]
    handler._load_input = lambda _job: (  # type: ignore[method-assign]
        tmp_path / "source.png",
        PipelineRunConfig(),
        "source.png",
        "a" * 64,
        "runs/test/source.png",
        "image/png",
    )
    (tmp_path / "source.png").write_bytes(_PNG)
    monkeypatch.setattr(
        "video_demo.application.image_pipeline_handler.run_image_pipeline",
        lambda **kwargs: (
            kwargs["analyzer"].analyze(
                image_data_url="data:image/png;base64,AAAA",
                title_hint="source.png",
            ),
            SimpleNamespace(result=object(), document=object(), warnings=()),
        )[1],
    )
    checks = iter((False, True))
    job = ClaimedJob(
        1,
        "job-cancel-after-vlm",
        "run-cancel-after-vlm",
        "owner",
        1,
        3,
        Scope("t", "a", "k"),
    )

    with pytest.raises(VideoDemoError) as error:
        handler.process(job, is_cancel_requested=lambda: next(checks))

    assert error.value.code == ErrorCode.JOB_CANCELLED
    assert analyzer.calls == 1
    assert publication.calls == 0
