from __future__ import annotations

import io
import wave

from fastapi.testclient import TestClient


def _wav() -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x00\x00" * 16_000)
    return stream.getvalue()


def test_audio_result_is_not_ready_until_audio_worker_publishes(
    client: TestClient,
    scope_headers: dict[str, str],
) -> None:
    uploaded = client.post(
        "/api/kb/knowledge-bases/kb-a/audio-objects",
        headers=scope_headers,
        files={"file": ("speech.wav", _wav(), "audio/wav")},
    )
    assert uploaded.status_code == 201
    created = client.post(
        "/api/kb/knowledge-bases/kb-a/audio-understanding-runs",
        headers=scope_headers,
        json={
            "object_ref": uploaded.json()["object_ref"],
            "idempotency_key": "audio-api-test-0001",
            "language_hints": ["zh"],
        },
    )
    assert created.status_code == 202
    root = (
        "/api/kb/knowledge-bases/kb-a/audio-understanding-runs/"
        + created.json()["run_id"]
    )
    result = client.get(f"{root}/result", headers=scope_headers)
    document = client.get(f"{root}/document", headers=scope_headers)

    assert result.status_code == 409
    assert result.json()["error"]["code"] == "AUDIO_RESULT_NOT_READY"
    assert document.status_code == 409
    assert document.json()["error"]["code"] == "AUDIO_RESULT_NOT_READY"
