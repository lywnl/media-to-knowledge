from __future__ import annotations

from fastapi.testclient import TestClient


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
        "min_speakers": None,
        "max_speakers": None,
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
