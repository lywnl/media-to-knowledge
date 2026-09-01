from __future__ import annotations

import base64
from pathlib import Path

from fastapi.testclient import TestClient

from video_demo.api.app import create_app
from video_demo.config import Settings
from video_demo.persistence.models import JobModel, JobStatus


def _upload(client: TestClient, headers: dict[str, str], content: bytes) -> str:
    response = client.post(
        "/api/kb/knowledge-bases/kb-a/video-objects",
        headers=headers,
        files={"file": ("lesson.mp4", content, "video/mp4")},
    )
    assert response.status_code == 201
    return str(response.json()["object_ref"])


def _create_payload(object_ref: str) -> dict[str, object]:
    return {
        "object_ref": object_ref,
        "idempotency_key": "request-video-0001",
        "language_hints": ["zh", "en", "ja", "ko", "es"],
    }


def test_job_detail_and_cancel_are_scope_safe(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
) -> None:
    object_ref = _upload(client, scope_headers, mp4_content)
    created = client.post(
        "/api/kb/knowledge-bases/kb-a/video-understanding-runs",
        headers=scope_headers,
        json=_create_payload(object_ref),
    ).json()
    job_id = created["job_id"]

    hidden = client.get(
        f"/api/kb/jobs/{job_id}",
        headers={"X-Tenant-Id": "tenant-b", "X-Application-Id": "app-a"},
        params={"knowledge_base_id": "kb-a"},
    )
    assert hidden.status_code == 404

    cancelled = client.post(
        f"/api/kb/jobs/{job_id}/cancel",
        headers=scope_headers,
        params={"knowledge_base_id": "kb-a"},
    )
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "CANCELLED"

    run = client.get(
        f"/api/kb/knowledge-bases/kb-a/video-understanding-runs/{created['run_id']}",
        headers=scope_headers,
    )
    assert run.status_code == 200
    assert run.json()["status"] == "CANCELLED"
    assert run.json()["error_code"] == "JOB_CANCELLED"


def test_retry_requeues_failed_job_but_rejects_pending_job(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
) -> None:
    object_ref = _upload(client, scope_headers, mp4_content)
    created = client.post(
        "/api/kb/knowledge-bases/kb-a/video-understanding-runs",
        headers=scope_headers,
        json=_create_payload(object_ref),
    ).json()

    response = client.post(
        f"/api/kb/jobs/{created['job_id']}/retry",
        headers=scope_headers,
        params={"knowledge_base_id": "kb-a"},
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "JOB_NOT_RETRYABLE"


def test_image_job_operations_route_to_image_run_service(
    tmp_path: Path,
    scope_headers: dict[str, str],
    monkeypatch,
    cloud_asr_environment: None,
) -> None:
    class _PassiveImageScheduler:
        def recover(self, _items: tuple[object, ...] = ()) -> int:
            return 0

        def start(self) -> None:
            return None

        def shutdown(self, *, wait: bool = False, timeout: float | None = None) -> None:
            del wait, timeout

        def submit(self, _scope: object, _run_id: str) -> str:
            return "accepted"

    import video_demo.api.app as app_module

    monkeypatch.setattr(
        app_module,
        "build_image_scheduler",
        lambda *_args, **_kwargs: _PassiveImageScheduler(),
    )
    settings = Settings(workspace_root=tmp_path, _env_file=None)
    with TestClient(create_app(settings)) as client:
        _test_image_job_operations(client, scope_headers)


def _test_image_job_operations(client: TestClient, scope_headers: dict[str, str]) -> None:
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+AAAAAgABSK+kcQAAAABJRU5ErkJggg==",
    )
    uploaded = client.post(
        "/api/kb/knowledge-bases/kb-a/image-objects",
        headers=scope_headers,
        files={"file": ("pixel.png", png, "image/png")},
    )
    assert uploaded.status_code == 201
    object_ref = uploaded.json()["object_ref"]

    created = client.post(
        "/api/kb/knowledge-bases/kb-a/image-understanding-runs",
        headers=scope_headers,
        json={
            "object_ref": object_ref,
            "idempotency_key": "request-image-0001",
        },
    )
    assert created.status_code == 202
    job_id = created.json()["job_id"]

    detail = client.get(
        f"/api/kb/jobs/{job_id}",
        headers=scope_headers,
        params={"knowledge_base_id": "kb-a"},
    )
    assert detail.status_code == 200
    assert detail.json()["resource_id"] == created.json()["run_id"]
    assert detail.json()["status"] in {"PENDING", "RUNNING"}

    cancelled = client.post(
        f"/api/kb/jobs/{job_id}/cancel",
        headers=scope_headers,
        params={"knowledge_base_id": "kb-a"},
    )
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "CANCELLED"

    retry_created = client.post(
        "/api/kb/knowledge-bases/kb-a/image-understanding-runs",
        headers=scope_headers,
        json={
            "object_ref": object_ref,
            "idempotency_key": "request-image-0002",
        },
    )
    assert retry_created.status_code == 202
    retry_job_id = retry_created.json()["job_id"]
    with client.app.state.container.database.session() as session:
        session.query(JobModel).filter(JobModel.job_id == retry_job_id).update(
            {"status": JobStatus.FAILED, "error_code": "IMAGE_VLM_UNAVAILABLE"},
            synchronize_session=False,
        )

    retried = client.post(
        f"/api/kb/jobs/{retry_job_id}/retry",
        headers=scope_headers,
        params={"knowledge_base_id": "kb-a"},
    )
    assert retried.status_code == 200
    assert retried.json()["status"] == "PENDING"
    assert retried.json()["attempt_count"] == 0
