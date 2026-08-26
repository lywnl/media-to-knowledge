from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from video_demo.api.dependencies import AppContainer
from video_demo.api.jobs import router as jobs_router
from video_demo.api.objects import router as objects_router
from video_demo.api.runs import router as runs_router
from video_demo.api.schemas import ErrorBody, ErrorResponse
from video_demo.application.queries import ResultQueryService
from video_demo.application.runs import RunService
from video_demo.application.uploads import UploadService
from video_demo.config import ApiRuntimeConfig, ApiRuntimeSettings, Settings
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.database import Database
from video_demo.persistence.migrations import upgrade_runtime_database
from video_demo.storage.artifacts import AtomicArtifactStore
from video_demo.storage.object_store import LocalVideoObjectStore

_NOT_FOUND_CODES = {
    ErrorCode.VIDEO_OBJECT_NOT_FOUND,
    ErrorCode.VIDEO_RUN_NOT_FOUND,
    ErrorCode.JOB_NOT_FOUND,
    ErrorCode.KEYFRAME_NOT_FOUND,
}
_CONFLICT_CODES = {
    ErrorCode.VIDEO_RESULT_NOT_READY,
    ErrorCode.IDEMPOTENCY_CONFLICT,
    ErrorCode.JOB_NOT_RETRYABLE,
    ErrorCode.RESULT_SCHEMA_UNSUPPORTED,
}
_UNAVAILABLE_CODES = {
    ErrorCode.VIDEO_FFMPEG_UNAVAILABLE,
    ErrorCode.VIDEO_FFPROBE_UNAVAILABLE,
    ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
    ErrorCode.SPEECH_SUBPROCESS_TIMEOUT,
    ErrorCode.SPEECH_SUBPROCESS_CRASHED,
}
_PUBLIC_DETAIL_KEYS = frozenset(
    {
        "evidence_id",
        "evidence_ids",
        "field",
        "language",
        "max_evidence_items",
        "model_id",
        "provider_error_code",
        "run_id",
        "segment_id",
        "status_code",
        "supported_schema_version",
    },
)


def create_app(
    settings: Settings | ApiRuntimeConfig | None = None,
) -> FastAPI:
    if settings is None:
        runtime = ApiRuntimeSettings(_env_file=None).to_runtime_config()
    elif isinstance(settings, Settings):
        runtime = settings.to_api_runtime_config()
    else:
        runtime = settings
    runtime.runtime_root.mkdir(parents=True, exist_ok=True)
    database_url = f"sqlite+pysqlite:///{runtime.runtime_root / 'video-demo.db'}"
    upgrade_runtime_database(
        runtime.workspace_root,
        runtime.runtime_root,
        database_url,
    )
    database = Database(database_url)
    object_store = LocalVideoObjectStore(
        runtime.runtime_root,
        max_video_bytes=runtime.max_video_bytes,
    )
    container = AppContainer(
        runtime_root=runtime.runtime_root,
        database=database,
        object_store=object_store,
        upload_service=UploadService(database, object_store),
        run_service=RunService(database),
        result_query_service=ResultQueryService(
            database,
            AtomicArtifactStore(runtime.runtime_root),
            max_evidence_items=runtime.max_result_evidence_items,
            max_keyframe_bytes=runtime.vlm_max_image_bytes,
            max_document_bytes=runtime.max_document_bytes,
            max_bundle_bytes=runtime.max_result_bundle_bytes,
        ),
    )

    app = FastAPI(
        title="视频理解到 retrieval_text Demo",
        version="0.1.0",
        description="独立视频上传、异步理解、证据查询与质量评测 API。",
    )
    web_root = Path(__file__).resolve().parent.parent / "web"
    app.state.container = container
    app.mount("/static", StaticFiles(directory=web_root), name="static")

    @app.get("/", include_in_schema=False)
    def frontend() -> FileResponse:
        return FileResponse(web_root / "index.html", media_type="text/html")

    app.include_router(objects_router)
    app.include_router(runs_router)
    app.include_router(jobs_router)
    app.add_exception_handler(VideoDemoError, _handle_video_demo_error)
    return app


async def _handle_video_demo_error(
    _request: Request,
    error: Exception,
) -> JSONResponse:
    assert isinstance(error, VideoDemoError)
    if error.code in _NOT_FOUND_CODES:
        status_code = 404
    elif error.code in _CONFLICT_CODES:
        status_code = 409
    elif error.code in _UNAVAILABLE_CODES:
        status_code = 503
    else:
        status_code = 422
    payload = ErrorResponse(
        error=ErrorBody(
            code=str(error.code),
            message=error.message,
            details={
                key: value
                for key, value in error.details.items()
                if key in _PUBLIC_DETAIL_KEYS
            },
        ),
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))
