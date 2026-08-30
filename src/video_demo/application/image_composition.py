"""图片单图理解 Worker 的独立生产装配。"""

from __future__ import annotations

import httpx

from video_demo.application.image_rendering import render_image_markdown
from video_demo.application.media_publication import MediaPublicationService
from video_demo.application.media_workers import ImageJobHandler
from video_demo.config import Settings
from video_demo.domain.image_document import ImageUnderstandingResult
from video_demo.errors import ErrorCode
from video_demo.integrations.image_vlm import ImageVlmClient
from video_demo.persistence.database import Database
from video_demo.persistence.migrations import upgrade_runtime_database
from video_demo.persistence.models import ImageUnderstandingRunModel
from video_demo.storage.artifacts import AtomicArtifactStore
from video_demo.storage.image_object_store import ImageObjectStore
from video_demo.worker.runtime import ReliableWorker


def build_image_worker(settings: Settings, *, worker_id: str) -> ReliableWorker:
    """构造只领取 IMAGE_UNDERSTANDING 的图片 Worker。"""

    assert settings.runtime_root is not None
    runtime_root = settings.runtime_root
    database_url = f"sqlite+pysqlite:///{runtime_root / 'video-demo.db'}"
    upgrade_runtime_database(settings.workspace_root, runtime_root, database_url)
    database = Database(database_url)
    vision = settings.require_vlm_configuration()
    http = httpx.Client()
    analyzer = ImageVlmClient(
        http,
        base_url=vision.base_url,
        api_key=vision.api_key.get_secret_value(),
        model_id=vision.model_id,
        timeout_seconds=vision.timeout_seconds,
        max_attempts=vision.max_attempts,
        max_response_bytes=settings.model_max_response_bytes,
    )
    publication = MediaPublicationService(
        database,
        AtomicArtifactStore(runtime_root),
        run_model=ImageUnderstandingRunModel,
        result_type=ImageUnderstandingResult,
        render=render_image_markdown,
        resource_type="IMAGE_UNDERSTANDING_RUN",
        not_found_code=ErrorCode.IMAGE_RUN_NOT_FOUND,
        artifact_prefix="image",
        max_document_bytes=settings.max_document_bytes,
        max_bundle_bytes=settings.max_result_bundle_bytes,
    )
    handler = ImageJobHandler(
        database,
        lambda: analyzer,
        publication,
        ImageObjectStore(runtime_root, max_bytes=settings.max_image_bytes),
        runtime_root=runtime_root,
        max_image_bytes=settings.max_image_bytes,
    )
    return ReliableWorker(
        database,
        worker_id,
        handler,
        job_type="IMAGE_UNDERSTANDING",
        owned_resources=(http,),
    )


__all__ = ["build_image_worker"]
