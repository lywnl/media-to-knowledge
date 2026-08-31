from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from video_demo.persistence.repositories import Scope, VideoRunRepository


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
    first_payload = first.json()
    second_payload = second.json()
    assert first_payload["run_id"] == second_payload["run_id"]
    assert first_payload["job_id"] == second_payload["job_id"]
    assert first_payload["status"] in {
        "PENDING",
        "RUNNING",
        "SUCCEEDED",
        "PARTIAL_SUCCEEDED",
        "FAILED",
        "CANCELLED",
    }
    assert second_payload["status"] in {
        "PENDING",
        "RUNNING",
        "SUCCEEDED",
        "PARTIAL_SUCCEEDED",
        "FAILED",
        "CANCELLED",
    }
    assert first_payload["run_id"].startswith("run_")
    assert first_payload["job_id"].startswith("job_")


def test_list_runs_returns_video_filename_and_newest_first(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
) -> None:
    object_ref = _upload(client, scope_headers, mp4_content)
    created = client.post(
        "/api/kb/knowledge-bases/kb-a/video-understanding-runs",
        headers=scope_headers,
        json=_create_payload(object_ref),
    )
    assert created.status_code == 202

    response = client.get(
        "/api/kb/knowledge-bases/kb-a/video-understanding-runs",
        headers=scope_headers,
    )

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 1
    assert items[0]["run_id"] == created.json()["run_id"]
    assert items[0]["original_filename"] == "lesson.mp4"
    assert items[0]["status"] in {
        "PENDING",
        "RUNNING",
        "SUCCEEDED",
        "PARTIAL_SUCCEEDED",
        "FAILED",
        "CANCELLED",
    }
    assert items[0]["created_at"]


def test_list_runs_isolated_by_scope(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
) -> None:
    object_ref = _upload(client, scope_headers, mp4_content)
    created = client.post(
        "/api/kb/knowledge-bases/kb-a/video-understanding-runs",
        headers=scope_headers,
        json=_create_payload(object_ref),
    )
    assert created.status_code == 202

    response = client.get(
        "/api/kb/knowledge-bases/kb-a/video-understanding-runs",
        headers={"X-Tenant-Id": "tenant-other", "X-Application-Id": "app-a"},
    )

    assert response.status_code == 200
    assert response.json() == {"items": []}


def test_create_run_accepts_bounded_hotwords_and_core_context(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
) -> None:
    object_ref = _upload(client, scope_headers, mp4_content)
    payload = _create_payload(object_ref)
    payload.update(
        {
            "hotwords": ["  Milvus  ", "WhisperX"],
            "core_context": "  这是一个   视频检索系统的技术讲解。  ",
        }
    )

    response = client.post(
        "/api/kb/knowledge-bases/kb-a/video-understanding-runs",
        headers=scope_headers,
        json=payload,
    )

    assert response.status_code == 202


def test_create_run_persists_only_current_speech_configuration(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
) -> None:
    object_ref = _upload(client, scope_headers, mp4_content)
    response = client.post(
        "/api/kb/knowledge-bases/kb-a/video-understanding-runs",
        headers=scope_headers,
        json=_create_payload(object_ref),
    )

    assert response.status_code == 202
    run_id = response.json()["run_id"]
    stored = client.app.state.container.database
    with stored.session() as session:
        run = VideoRunRepository(session).get(Scope("tenant-a", "app-a", "kb-a"), run_id)
        assert run is not None
        assert run.config_snapshot == {
            "language_hints": ["zh", "en", "ja", "ko", "es"],
            "hotwords": [],
            "core_context": None,
            "document_config": {
                "document_title": None,
                "detail_level": "standard",
                "chapter_granularity": "standard",
                "include_verbatim_quotes": True,
                "max_visuals_per_chapter": 2,
            },
            "result_schema_version": "4.2.0",
        }


def test_create_run_sanitizes_document_title_and_includes_it_in_idempotency(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
) -> None:
    object_ref = _upload(client, scope_headers, mp4_content)
    runs_url = "/api/kb/knowledge-bases/kb-a/video-understanding-runs"
    first = {
        **_create_payload(object_ref),
        "document_config": {"document_title": "  教程/第一课\u0000  "},
        "result_schema_version": "4.2.0",
    }

    created = client.post(runs_url, headers=scope_headers, json=first)
    assert created.status_code == 202
    with client.app.state.container.database.session() as session:
        run = VideoRunRepository(session).get(
            Scope("tenant-a", "app-a", "kb-a"), created.json()["run_id"]
        )
        assert run is not None
        assert run.config_snapshot["document_config"]["document_title"] == "教程 第一课"

    conflict = client.post(
        runs_url,
        headers=scope_headers,
        json={
            **_create_payload(object_ref),
            "document_config": {"document_title": "另一课"},
            "result_schema_version": "4.2.0",
        },
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("speech_enrichment_mode", "full"),
        ("min_speakers", 1),
        ("max_speakers", 2),
    ],
)
def test_create_run_rejects_retired_speech_configuration_fields(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
    field: str,
    value: object,
) -> None:
    object_ref = _upload(client, scope_headers, mp4_content)
    response = client.post(
        "/api/kb/knowledge-bases/kb-a/video-understanding-runs",
        headers=scope_headers,
        json={**_create_payload(object_ref), field: value},
    )

    assert response.status_code == 422


def test_same_idempotency_key_rejects_changed_speech_hints(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
) -> None:
    object_ref = _upload(client, scope_headers, mp4_content)
    runs_url = "/api/kb/knowledge-bases/kb-a/video-understanding-runs"
    first = {**_create_payload(object_ref), "hotwords": ["Milvus"]}
    second = {**_create_payload(object_ref), "hotwords": ["MySQL"]}

    assert client.post(runs_url, headers=scope_headers, json=first).status_code == 202
    response = client.post(runs_url, headers=scope_headers, json=second)

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"


def test_create_run_rejects_invalid_speech_hints(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
) -> None:
    object_ref = _upload(client, scope_headers, mp4_content)
    runs_url = "/api/kb/knowledge-bases/kb-a/video-understanding-runs"
    invalid_payloads = (
        {**_create_payload(object_ref), "hotwords": ["Milvus", " Milvus "]},
        {**_create_payload(object_ref), "hotwords": ["x" * 65]},
        {**_create_payload(object_ref), "hotwords": ["术语\n注入"]},
        {**_create_payload(object_ref), "core_context": "上下文\x00注入"},
    )

    for index, payload in enumerate(invalid_payloads):
        payload["idempotency_key"] = f"invalid-speech-hint-{index:04d}"
        response = client.post(runs_url, headers=scope_headers, json=payload)
        assert response.status_code == 422


def test_create_run_rejects_invalid_idempotency_and_language(
    client: TestClient,
    scope_headers: dict[str, str],
    mp4_content: bytes,
) -> None:
    object_ref = _upload(client, scope_headers, mp4_content)
    invalid_payload = _create_payload(object_ref, "short")
    invalid_payload["language_hints"] = ["fr"]

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
    assert response.json()["current_stage"]
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
