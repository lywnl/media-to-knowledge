from __future__ import annotations

import base64
from pathlib import Path

from fastapi.testclient import TestClient

from video_demo.api.app import create_app
from video_demo.application.image_rendering import render_image_markdown
from video_demo.application.publication_contracts import ResultWriteFence
from video_demo.config import ApiRuntimeConfig
from video_demo.domain.image_document import (
    ImageDocument,
    ImageSourceEvidence,
    ImageUnderstandingResult,
)
from video_demo.persistence.repositories import JobRepository, Scope

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+AAAAAgABSK+kcQAAAABJRU5ErkJggg==",
)


def test_image_result_and_document_routes_read_published_artifacts(
    tmp_path: Path,
    scope_headers: dict[str, str],
    cloud_asr_environment: None,
) -> None:
    runtime = ApiRuntimeConfig(
        workspace_root=tmp_path,
        runtime_root=tmp_path / "runtime",
        max_video_bytes=1024 * 1024,
        max_result_bundle_bytes=64 * 1024 * 1024,
        max_document_bytes=16 * 1024 * 1024,
        max_result_evidence_items=25_000,
        vlm_max_image_bytes=8 * 1024 * 1024,
        max_audio_bytes=4 * 1024 * 1024 * 1024,
        max_audio_duration_ms=7_200_000,
        max_image_bytes=8 * 1024 * 1024,
    )
    with TestClient(create_app(runtime)) as client:
        uploaded = client.post(
            "/api/kb/knowledge-bases/kb-a/image-objects",
            headers=scope_headers,
            files={"file": ("pixel.png", _PNG, "image/png")},
        )
        assert uploaded.status_code == 201
        object_ref = uploaded.json()["object_ref"]
        asset_sha = uploaded.json()["sha256"]

        created = client.post(
            "/api/kb/knowledge-bases/kb-a/image-understanding-runs",
            headers=scope_headers,
            json={
                "object_ref": object_ref,
                "idempotency_key": "image-route-result-001",
            },
        )
        assert created.status_code == 202
        run_id = created.json()["run_id"]
        scope = Scope("tenant-a", "app-a", "kb-a")
        result = ImageUnderstandingResult(
            run_id=run_id,
            asset_sha256=asset_sha,
            document=ImageDocument(
                title="pixel.png",
                overview_zh="图片概览",
                content_blocks=(),
                claims=(),
                evidence_refs=("image_source_route",),
                content_status="DEGRADED",
            ),
            source=ImageSourceEvidence(
                evidence_id="image_source_route",
                relative_path="image_object/source/pixel.png",
                mime_type="image/png",
                sha256=asset_sha,
                width=1,
                height=1,
                size_bytes=len(_PNG),
            ),
        )
        publication = client.app.state.container.media_query_services["IMAGE"].publication
        with client.app.state.container.database.session() as session:
            claimed = JobRepository(session).claim_image_run(
                scope,
                run_id,
                "image-route-test",
                lease_seconds=60,
            )
        assert claimed is not None
        document = render_image_markdown(result)
        publication.persist(
            scope,
            result,
            document=document,
            status="SUCCEEDED",
            warnings=(),
            fence=ResultWriteFence(
                claimed.id,
                claimed.worker_id,
                claimed.attempt_count,
            ),
        )

        root = f"/api/kb/knowledge-bases/kb-a/image-understanding-runs/{run_id}"
        result_response = client.get(f"{root}/result", headers=scope_headers)
        document_response = client.get(f"{root}/document", headers=scope_headers)

    assert result_response.status_code == 200
    assert result_response.json()["schema_version"] == "1.0.0"
    assert result_response.json()["document"]["overview_zh"] == "图片概览"
    assert document_response.status_code == 200
    assert document_response.headers["content-type"] == "text/markdown; charset=utf-8"
    assert document_response.content == document.content
