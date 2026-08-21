from __future__ import annotations

from fastapi.testclient import TestClient


def _upload(
    client: TestClient,
    headers: dict[str, str],
    content: bytes,
) -> str:
    response = client.post(
        "/api/kb/knowledge-bases/kb-a/video-objects",
        headers=headers,
        files={"file": ("lesson.mp4", content, "video/mp4")},
    )
    assert response.status_code == 201
    return str(response.json()["object_ref"])


def _create_payload(
    object_ref: str,
    idempotency_key: str = "request-video-0001",
) -> dict[str, object]:
    return {
        "object_ref": object_ref,
        "idempotency_key": idempotency_key,
        "language_hints": ["zh", "en", "ja", "ko", "es"],
        "min_speakers": None,
        "max_speakers": None,
    }


def test_create_run_returns_202_and_is_idempotent(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
) -> None:
    object_ref = _upload(client, scope_headers, mp4_content)

    first = client.post(
        "/api/kb/knowledge-bases/kb-a/video-understanding-runs",
        headers=scope_headers,
        json=_create_payload(object_ref),
    )
    second = client.post(
        "/api/kb/knowledge-bases/kb-a/video-understanding-runs",
        headers=scope_headers,
        json=_create_payload(object_ref),
    )

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json() == second.json()
    assert first.json()["status"] == "PENDING"
    assert first.json()["run_id"].startswith("run_")
    assert first.json()["job_id"].startswith("job_")


def test_create_run_rejects_invalid_idempotency_language_and_speaker_range(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
) -> None:
    object_ref = _upload(client, scope_headers, mp4_content)
    invalid_payload = _create_payload(object_ref, "short")
    invalid_payload["language_hints"] = ["fr"]
    invalid_payload["min_speakers"] = 4
    invalid_payload["max_speakers"] = 2

    response = client.post(
        "/api/kb/knowledge-bases/kb-a/video-understanding-runs",
        headers=scope_headers,
        json=invalid_payload,
    )

    assert response.status_code == 422


def test_create_run_hides_object_from_other_scope(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
) -> None:
    object_ref = _upload(client, scope_headers, mp4_content)

    response = client.post(
        "/api/kb/knowledge-bases/kb-a/video-understanding-runs",
        headers={"X-Tenant-Id": "tenant-b", "X-Application-Id": "app-a"},
        json=_create_payload(object_ref),
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "VIDEO_OBJECT_NOT_FOUND"


def test_run_status_does_not_expose_internal_paths_or_secret_fields(
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

    response = client.get(
        f"/api/kb/knowledge-bases/kb-a/video-understanding-runs/{created['run_id']}",
        headers=scope_headers,
    )

    assert response.status_code == 200
    serialized = response.text.lower()
    assert response.json()["current_stage"] == "REGISTER"
    assert "relative_path" not in serialized
    assert "api_key" not in serialized
    assert "secret" not in serialized


def test_result_evidence_and_keyframe_routes_report_not_ready(
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
    run_root = f"/api/kb/knowledge-bases/kb-a/video-understanding-runs/{created['run_id']}"

    for suffix in ("result", "evidence", "keyframes/keyframe_001/content"):
        response = client.get(f"{run_root}/{suffix}", headers=scope_headers)
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "VIDEO_RESULT_NOT_READY"
