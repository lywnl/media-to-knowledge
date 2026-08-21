from __future__ import annotations

from fastapi.testclient import TestClient


def test_upload_requires_explicit_scope_headers(client: TestClient, mp4_content: bytes) -> None:
    response = client.post(
        "/api/kb/knowledge-bases/kb-a/video-objects",
        files={"file": ("lesson.mp4", mp4_content, "video/mp4")},
    )

    assert response.status_code == 422


def test_upload_returns_ready_object_without_local_path(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
) -> None:
    response = client.post(
        "/api/kb/knowledge-bases/kb-a/video-objects",
        headers=scope_headers,
        files={"file": ("lesson.mp4", mp4_content, "video/mp4")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["status"] == "READY"
    assert payload["object_ref"].startswith("obj_")
    assert payload["sha256"]
    assert "relative_path" not in payload
    assert "local_path" not in payload


def test_object_lookup_is_hidden_from_other_tenant(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
) -> None:
    uploaded = client.post(
        "/api/kb/knowledge-bases/kb-a/video-objects",
        headers=scope_headers,
        files={"file": ("lesson.mp4", mp4_content, "video/mp4")},
    ).json()

    response = client.get(
        f"/api/kb/knowledge-bases/kb-a/video-objects/{uploaded['object_ref']}",
        headers={"X-Tenant-Id": "tenant-b", "X-Application-Id": "app-a"},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "VIDEO_OBJECT_NOT_FOUND"
