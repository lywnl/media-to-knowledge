from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from video_demo.domain.document import DocumentGenerationConfig
from video_demo.domain.document_plan import FrameCandidateArtifact, VisualSearchTarget
from video_demo.domain.evidence import SpeechSegment
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.document_port import (
    ChapterVisionRepairRequest,
    ChapterVisionRequest,
    ChapterVisionResponse,
    InvalidModelResponse,
    ModelResponseValidationError,
)
from video_demo.integrations.document_prompts import (
    prompt_for_vision,
    prompt_for_vision_repair,
    vision_payload_size_upper_bound,
)
from video_demo.integrations.qwen_vl import (
    QwenVisionCallFailure,
    QwenVisionClient,
)


def _frame(root: Path, frame_id: str, timestamp_ms: int) -> FrameCandidateArtifact:
    content = b"\xff\xd8\xff" + frame_id.encode() + b"\xff\xd9"
    digest = hashlib.sha256(content).hexdigest()
    path = root / "visual" / "candidates" / f"{digest}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    (root / "visual").chmod(0o700)
    path.parent.chmod(0o700)
    path.write_bytes(content)
    path.chmod(0o600)
    return FrameCandidateArtifact(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        sha256=digest,
        size_bytes=len(content),
        relative_path=path.relative_to(root).as_posix(),
        mime_type="image/jpeg",
        perceptual_hash="0123456789abcdef",
        target_ids=("target_001",),
    )


def _request(root: Path) -> ChapterVisionRequest:
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=1_000,
        end_ms=2_000,
        text="查看画面",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    return ChapterVisionRequest(
        chapter_id="chapter_001",
        targets=(
            VisualSearchTarget(
                target_id="target_001",
                purpose="SEMANTIC",
                query_zh="画面文字",
                anchor_evidence_refs=("asr_001",),
            ),
        ),
        frames=(
            _frame(root, "frame_b", 2_000),
            _frame(root, "frame_a", 1_000),
        ),
        transcript_evidence=(speech,),
        document_config=DocumentGenerationConfig(),
        prompt_version="chapter-vlm-v1",
    )


def _client(
    runtime_root: Path,
    handler: httpx.MockTransport,
    *,
    max_response_bytes: int = 2 * 1024 * 1024,
    max_attempts: int = 1,
) -> QwenVisionClient:
    return QwenVisionClient(
        httpx.Client(transport=handler),
        base_url="https://vision.example.test/v1",
        api_key="vision-secret",
        model_id="qwen3-vl-flash",
        runtime_root=runtime_root,
        max_attempts=max_attempts,
        max_response_bytes=max_response_bytes,
        sleeper=lambda _seconds: None,
    )


def _provider_response(request: httpx.Request, body: object) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": json.dumps(body, ensure_ascii=False)}}]},
        request=request,
    )


def _valid_observation(frame_id: str = "frame_a") -> dict[str, object]:
    return {
        "observations": [
            {
                "target_ids": ["target_001"],
                "selected_frame_ids": [frame_id],
                "transcript_evidence_refs": ["asr_001"],
                "visual_type": "TEXT",
                "caption": "画面文字",
                "content_blocks": [],
                "visual_facts": [],
                "frame_relations": [],
                "relation_to_transcript": "COMPLEMENTARY",
                "certainty": 0.9,
                "quality_flags": [],
                "uncertainties": [],
            },
        ],
    }


def test_qwen_sends_sorted_jpeg_frames_in_one_request_without_local_metadata(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs" / "scope_001" / "run_001"
    run_root.mkdir(parents=True)
    request = _request(run_root)
    payloads: list[dict[str, object]] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(http_request.content))
        return _provider_response(http_request, _valid_observation())

    result = _client(tmp_path, httpx.MockTransport(handler)).analyze_chapter(
        request,
        allowed_run_root=run_root,
    )

    assert result.observations[0].selected_frame_ids == ("frame_a",)
    content = payloads[0]["messages"][1]["content"]  # type: ignore[index]
    labels = [item["text"] for item in content if item.get("type") == "text"]  # type: ignore[union-attr]
    assert labels.index("FRAME_ID=frame_a") < labels.index("FRAME_ID=frame_b")
    images = [item for item in content if item.get("type") == "image_url"]  # type: ignore[union-attr]
    assert len(images) == 2
    assert all(item["image_url"]["url"].startswith("data:image/jpeg;base64,") for item in images)  # type: ignore[index]
    untrusted = labels[0]
    assert "relative_path" not in untrusted
    assert "sha256" not in untrusted
    assert "candidates" not in untrusted
    assert "https://" not in untrusted
    assert payloads[0]["response_format"]["json_schema"]["name"] == "chapter_vlm_v2"  # type: ignore[index]


def test_qwen_accepts_content_blocks_and_explanatory_text_before_json(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "scope_001" / "run_001"
    run_root.mkdir(parents=True)
    body = _valid_observation()

    def handler(request: httpx.Request) -> httpx.Response:
        message = "我识别到以下内容：\n" + json.dumps(body, ensure_ascii=False)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "reasoning", "text": "内部思考"},
                                {"type": "text", "text": message},
                            ],
                        },
                    },
                ],
            },
            request=request,
        )

    result = _client(tmp_path, httpx.MockTransport(handler)).analyze_chapter(
        _request(run_root),
        allowed_run_root=run_root,
    )

    assert result.observations[0].caption == "画面文字"


def test_qwen_request_bytes_do_not_exceed_shared_payload_upper_bound(tmp_path: Path) -> None:
    run_root = tmp_path / "runs/scope_001/run_001"
    run_root.mkdir(parents=True)
    request = _request(run_root)
    actual_sizes: list[int] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        actual_sizes.append(len(http_request.content))
        return _provider_response(http_request, {"observations": []})

    _client(tmp_path, httpx.MockTransport(handler)).analyze_chapter(
        request,
        allowed_run_root=run_root,
    )

    ordered = sorted(request.frames, key=lambda item: (item.timestamp_ms, item.frame_id))
    upper_bound = vision_payload_size_upper_bound(
        prompt_for_vision(request),
        model_id="qwen3-vl-flash",
        schema_name="chapter_vlm_v2",
        response_schema=ChapterVisionResponse.model_json_schema(),
        ordered_frames=tuple((frame.frame_id, frame.size_bytes) for frame in ordered),
    )
    assert actual_sizes[0] <= upper_bound
    assert upper_bound - actual_sizes[0] < 16


def test_qwen_calls_attempt_callback_before_every_real_http_attempt(tmp_path: Path) -> None:
    run_root = tmp_path / "runs/scope_001/run_001"
    run_root.mkdir(parents=True)
    attempts: list[int] = []
    responses = iter((500, 200))

    def handler(http_request: httpx.Request) -> httpx.Response:
        status = next(responses)
        if status == 500:
            return httpx.Response(status, request=http_request)
        return _provider_response(http_request, {"observations": []})

    _client(tmp_path, httpx.MockTransport(handler), max_attempts=2).analyze_chapter(
        _request(run_root),
        allowed_run_root=run_root,
        on_provider_attempt=lambda: attempts.append(1),
    )

    assert attempts == [1, 1]


def test_qwen_receipt_records_final_http_and_provider_response_digest(tmp_path: Path) -> None:
    run_root = tmp_path / "runs/scope_001/run_001"
    run_root.mkdir(parents=True)
    provider_envelope = {
        "id": "provider-response-001",
        "choices": [{"message": {"content": json.dumps({"observations": []})}}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=provider_envelope, request=request)

    response, receipt = _client(
        tmp_path, httpx.MockTransport(handler)
    ).analyze_chapter_with_receipt(_request(run_root), allowed_run_root=run_root)

    assert response.observations == ()
    assert receipt.provider_attempt_count == 1
    assert receipt.final_http_status == 200
    assert receipt.response_id_sha256 == hashlib.sha256(
        b"provider-response-001"
    ).hexdigest()
    assert receipt.provider_response_sha256 == hashlib.sha256(
        json.dumps(provider_envelope, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def test_qwen_receipt_measures_real_encoded_json_request_bytes(tmp_path: Path) -> None:
    run_root = tmp_path / "runs/scope_001/run_001"
    run_root.mkdir(parents=True)
    sent_bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent_bodies.append(request.content)
        return _provider_response(request, {"observations": []})

    _response, receipt = _client(
        tmp_path,
        httpx.MockTransport(handler),
    ).analyze_chapter_with_receipt(_request(run_root), allowed_run_root=run_root)

    assert receipt.encoded_request_bytes == len(sent_bodies[0])
    assert 0 < receipt.request_json_bytes < receipt.encoded_request_bytes


def test_qwen_receipt_and_plain_call_share_identical_payload(tmp_path: Path) -> None:
    run_root = tmp_path / "runs/scope_001/run_001"
    run_root.mkdir(parents=True)
    payloads: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(request.content)
        return _provider_response(request, {"observations": []})

    plain_client = _client(tmp_path, httpx.MockTransport(handler))
    plain_client.analyze_chapter(_request(run_root), allowed_run_root=run_root)
    plain_client.analyze_chapter_with_receipt(
        _request(run_root),
        allowed_run_root=run_root,
    )

    assert len(payloads) == 2
    assert payloads[0] == payloads[1]


def test_qwen_receipt_failure_keeps_provider_attempt_and_http_status(tmp_path: Path) -> None:
    run_root = tmp_path / "runs/scope_001/run_001"
    run_root.mkdir(parents=True)

    client = _client(
        tmp_path,
        httpx.MockTransport(lambda request: httpx.Response(401, request=request)),
    )

    with pytest.raises(QwenVisionCallFailure) as raised:
        client.analyze_chapter_with_receipt(_request(run_root), allowed_run_root=run_root)

    assert raised.value.provider.provider_attempt_count == 1
    assert raised.value.provider.final_http_status == 401
    assert raised.value.provider.provider_response_sha256 is None


def test_qwen_receipt_failure_does_not_hash_bounded_error_body(tmp_path: Path) -> None:
    run_root = tmp_path / "runs/scope_001/run_001"
    run_root.mkdir(parents=True)
    body = b"provider-error"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=body, request=request)

    with pytest.raises(QwenVisionCallFailure) as raised:
        _client(
            tmp_path,
            httpx.MockTransport(handler),
            max_response_bytes=len(body),
        ).analyze_chapter_with_receipt(_request(run_root), allowed_run_root=run_root)

    assert raised.value.provider.final_http_status == 500
    assert raised.value.provider.provider_response_sha256 is None


def test_qwen_receipt_invalidates_two_xx_non_json_with_provider_digest(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs/scope_001/run_001"
    run_root.mkdir(parents=True)
    body = b"not-json"

    with pytest.raises(QwenVisionCallFailure) as raised:
        _client(
            tmp_path,
            httpx.MockTransport(
                lambda request: httpx.Response(200, content=body, request=request)
            ),
        ).analyze_chapter_with_receipt(_request(run_root), allowed_run_root=run_root)

    assert raised.value.provider.final_http_status == 200
    assert raised.value.provider.provider_response_sha256 == hashlib.sha256(body).hexdigest()


def test_qwen_receipt_oversized_error_body_keeps_status_without_digest(tmp_path: Path) -> None:
    run_root = tmp_path / "runs/scope_001/run_001"
    run_root.mkdir(parents=True)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"x" * 11, request=request)

    with pytest.raises(QwenVisionCallFailure) as raised:
        _client(
            tmp_path,
            httpx.MockTransport(handler),
            max_response_bytes=10,
        ).analyze_chapter_with_receipt(_request(run_root), allowed_run_root=run_root)

    assert raised.value.code == ErrorCode.DEPENDENCY_TEMPORARY_FAILURE
    assert raised.value.provider.final_http_status == 500
    assert raised.value.provider.provider_response_sha256 is None


def test_qwen_receipt_timeout_after_http_error_has_no_stale_status_or_digest(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs/scope_001/run_001"
    run_root.mkdir(parents=True)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(500, content=b"temporary", request=request)
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(QwenVisionCallFailure) as raised:
        _client(
            tmp_path,
            httpx.MockTransport(handler),
            max_attempts=2,
        ).analyze_chapter_with_receipt(_request(run_root), allowed_run_root=run_root)

    assert raised.value.provider.provider_attempt_count == 2
    assert raised.value.provider.final_http_status is None
    assert raised.value.provider.provider_response_sha256 is None


def test_qwen_receipt_counts_transport_retry_and_uses_final_response(tmp_path: Path) -> None:
    run_root = tmp_path / "runs/scope_001/run_001"
    run_root.mkdir(parents=True)
    statuses = iter((500, 200))
    responses: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        responses.append(status)
        if status == 500:
            return httpx.Response(status, request=request)
        return _provider_response(request, {"observations": []})

    _response, receipt = _client(
        tmp_path,
        httpx.MockTransport(handler),
        max_attempts=2,
    ).analyze_chapter_with_receipt(_request(run_root), allowed_run_root=run_root)

    assert responses == [500, 200]
    assert receipt.provider_attempt_count == 2
    assert receipt.final_http_status == 200


def test_qwen_receipt_rejects_non_string_provider_id_without_leaking_body(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs/scope_001/run_001"
    run_root.mkdir(parents=True)
    provider_envelope = {
        "id": 123,
        "choices": [{"message": {"content": json.dumps({"observations": []})}}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=provider_envelope, request=request)

    with pytest.raises(QwenVisionCallFailure) as raised:
        _client(tmp_path, httpx.MockTransport(handler)).analyze_chapter_with_receipt(
            _request(run_root),
            allowed_run_root=run_root,
        )

    assert raised.value.code == ErrorCode.QWEN_RESPONSE_INVALID
    assert raised.value.provider.provider_attempt_count == 1
    assert raised.value.provider.final_http_status == 200
    assert raised.value.provider.provider_response_sha256 is not None
    assert "123" not in raised.value.message


@pytest.mark.parametrize(
    "provider_envelope",
    (
        {
            "id": 123,
            "choices": [{"message": {"content": json.dumps({"observations": []})}}],
        },
        {
            "id": "provider-response-nan",
            "choices": [
                {"message": {"content": '{"observations": [], "score": NaN}'}},
            ],
        },
        {
            "id": "provider-response-infinity",
            "choices": [
                {"message": {"content": '{"observations": [], "score": Infinity}'}},
            ],
        },
    ),
)
def test_qwen_plain_call_keeps_invalid_provider_response_repairable(
    tmp_path: Path,
    provider_envelope: dict[str, object],
) -> None:
    run_root = tmp_path / "runs/scope_001/run_001"
    run_root.mkdir(parents=True)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=provider_envelope, request=request)

    with pytest.raises(ModelResponseValidationError) as raised:
        _client(tmp_path, httpx.MockTransport(handler)).analyze_chapter(
            _request(run_root),
            allowed_run_root=run_root,
        )

    assert raised.value.code == ErrorCode.QWEN_RESPONSE_INVALID
    assert raised.value.invalid_response.validation_errors


def test_qwen_rereads_images_before_retry_and_does_not_count_blocked_attempt(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs/scope_001/run_001"
    run_root.mkdir(parents=True)
    request = _request(run_root)
    attempts: list[int] = []
    http_attempts = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal http_attempts
        http_attempts += 1
        if http_attempts == 1:
            frame = request.frames[0]
            path = run_root / frame.relative_path
            path.unlink()
            path.write_bytes(b"\xff\xd8\xffreplaced-after-500\xff\xd9")
            path.chmod(0o600)
            return httpx.Response(500, request=http_request)
        return _provider_response(http_request, {"observations": []})

    with pytest.raises(VideoDemoError) as raised:
        _client(tmp_path, httpx.MockTransport(handler), max_attempts=2).analyze_chapter(
            request,
            allowed_run_root=run_root,
            on_provider_attempt=lambda: attempts.append(1),
        )

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID
    assert http_attempts == 1
    assert attempts == [1]


@pytest.mark.parametrize("status_code", (302, 307, 400, 409))
def test_qwen_request_rejection_is_not_response_content_invalid(
    tmp_path: Path,
    status_code: int,
) -> None:
    run_root = tmp_path / "runs/scope_001/run_001"
    run_root.mkdir(parents=True)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(status_code, request=request),
    )

    with pytest.raises(VideoDemoError) as raised:
        _client(tmp_path, transport).analyze_chapter(
            _request(run_root),
            allowed_run_root=run_root,
        )

    assert raised.value.code == ErrorCode.QWEN_REQUEST_REJECTED
    assert not isinstance(raised.value, ModelResponseValidationError)


def test_qwen_requires_per_call_run_root_inside_runtime_root(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "scope_001" / "run_001"
    outside = tmp_path.parent / "outside-run"
    run_root.mkdir(parents=True)
    outside.mkdir(exist_ok=True)
    request = _request(run_root)
    client = _client(
        tmp_path,
        httpx.MockTransport(lambda request: httpx.Response(500, request=request)),
    )

    with pytest.raises(VideoDemoError) as raised:
        client.analyze_chapter(request, allowed_run_root=outside)
    assert raised.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE


@pytest.mark.parametrize(
    ("relative_run_root", "expected_code"),
    (
        (
            Path("tmp") / "scope_001" / "run_001",
            ErrorCode.WORKSPACE_PATH_ESCAPE,
        ),
        (
            Path("arbitrary") / "scope_001" / "run_001",
            ErrorCode.WORKSPACE_PATH_ESCAPE,
        ),
        (Path("runs") / "scope_001", ErrorCode.WORKSPACE_PATH_ESCAPE),
        (
            Path("runs") / "scope_001" / "run_001" / "nested",
            ErrorCode.WORKSPACE_PATH_ESCAPE,
        ),
        (Path("runs") / "x" / "run_001", ErrorCode.INVALID_PATH_COMPONENT),
        (Path("runs") / "scope_001" / "x", ErrorCode.INVALID_PATH_COMPONENT),
    ),
)
def test_qwen_rejects_non_run_root_shape(
    tmp_path: Path,
    relative_run_root: Path,
    expected_code: ErrorCode,
) -> None:
    allowed_run_root = tmp_path / relative_run_root
    allowed_run_root.mkdir(parents=True)
    request_root = tmp_path / "runs" / "scope_001" / "run_request"
    request_root.mkdir(parents=True, exist_ok=True)
    client = _client(
        tmp_path,
        httpx.MockTransport(lambda request: httpx.Response(500, request=request)),
    )

    with pytest.raises(VideoDemoError) as raised:
        client.analyze_chapter(
            _request(request_root),
            allowed_run_root=allowed_run_root,
        )

    assert raised.value.code == expected_code


def test_qwen_rejects_symlink_and_non_jpg_extension(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "scope_001" / "run_001"
    run_root.mkdir(parents=True)
    request = _request(run_root)
    real = run_root / request.frames[0].relative_path
    link = real.with_name("link.jpg")
    link.symlink_to(real)
    linked = request.frames[0].model_copy(
        update={"relative_path": link.relative_to(run_root).as_posix()},
    )
    client = _client(
        tmp_path,
        httpx.MockTransport(lambda request: httpx.Response(500, request=request)),
    )

    with pytest.raises(VideoDemoError) as raised:
        client.analyze_chapter(
            request.model_copy(update={"frames": (linked, request.frames[1])}),
            allowed_run_root=run_root,
        )
    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID

    renamed = real.with_suffix(".jpeg")
    real.rename(renamed)
    wrong_extension = request.frames[0].model_copy(
        update={"relative_path": renamed.relative_to(run_root).as_posix()},
    )
    with pytest.raises(VideoDemoError) as raised:
        client.analyze_chapter(
            request.model_copy(update={"frames": (wrong_extension, request.frames[1])}),
            allowed_run_root=run_root,
        )
    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID


def test_qwen_rejects_file_replaced_during_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "runs" / "scope_001" / "run_001"
    run_root.mkdir(parents=True)
    request = _request(run_root)
    path = run_root / request.frames[1].relative_path
    original_open = os.open
    swapped = False

    def racing_open(
        name: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if Path(name).name == path.name and dir_fd is not None and not swapped:
            swapped = True
            replacement = path.with_suffix(".replacement")
            replacement.write_bytes(b"\xff\xd8\xffreplacement\xff\xd9")
            os.replace(replacement, path)
        return original_open(name, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", racing_open)
    client = _client(
        tmp_path,
        httpx.MockTransport(lambda request: httpx.Response(500, request=request)),
    )

    with pytest.raises(VideoDemoError) as raised:
        client.analyze_chapter(request, allowed_run_root=run_root)
    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID


def test_unknown_frame_returns_repairable_validation_error(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "scope_001" / "run_001"
    run_root.mkdir(parents=True)

    def handler(request: httpx.Request) -> httpx.Response:
        return _provider_response(request, _valid_observation("frame_unknown"))

    with pytest.raises(ModelResponseValidationError) as raised:
        _client(tmp_path, httpx.MockTransport(handler)).analyze_chapter(
            _request(run_root),
            allowed_run_root=run_root,
        )
    assert raised.value.code == ErrorCode.QWEN_RESPONSE_INVALID
    assert raised.value.invalid_response.validation_errors == (
        "observations.selected_frame_ids:unknown_reference",
    )


def test_qwen_validation_error_includes_safe_business_reason(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "scope_001" / "run_001"
    run_root.mkdir(parents=True)
    body = _valid_observation()
    observation = body["observations"][0]
    assert isinstance(observation, dict)
    observation["relation_to_transcript"] = "INDEPENDENT"

    def handler(request: httpx.Request) -> httpx.Response:
        return _provider_response(request, body)

    with pytest.raises(ModelResponseValidationError) as raised:
        _client(tmp_path, httpx.MockTransport(handler)).analyze_chapter(
            _request(run_root),
            allowed_run_root=run_root,
        )

    assert raised.value.invalid_response.validation_errors == (
        "observations.0:value_error:INDEPENDENT 草稿不得引用转写证据",
    )


def test_qwen_requires_every_claimed_target_to_have_a_selected_frame(tmp_path: Path) -> None:
    run_root = tmp_path / "runs/scope_001/run_001"
    run_root.mkdir(parents=True)
    request = _request(run_root)
    second_target = VisualSearchTarget(
        target_id="target_002",
        purpose="SEMANTIC",
        query_zh="另一处画面",
        anchor_evidence_refs=("asr_001",),
    )
    body = _valid_observation()
    body["observations"][0]["target_ids"] = ["target_001", "target_002"]  # type: ignore[index]

    def handler(http_request: httpx.Request) -> httpx.Response:
        return _provider_response(http_request, body)

    with pytest.raises(ModelResponseValidationError) as raised:
        _client(tmp_path, httpx.MockTransport(handler)).analyze_chapter(
            request.model_copy(update={"targets": (*request.targets, second_target)}),
            allowed_run_root=run_root,
        )

    assert raised.value.invalid_response.validation_errors == (
        "observations.target_ids:frame_binding_mismatch",
    )


def test_qwen_repair_resends_same_images_with_repair_prompt_and_schema(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "scope_001" / "run_001"
    run_root.mkdir(parents=True)
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return _provider_response(request, {"observations": []})

    original = _request(run_root)
    repair = ChapterVisionRepairRequest(
        request=original,
        invalid_response=InvalidModelResponse(
            content_sha256="d" * 64,
            validation_errors=("observations.selected_frame_ids:unknown_reference",),
        ),
        allowed_frame_ids=("frame_a", "frame_b"),
        allowed_target_ids=("target_001",),
        allowed_transcript_evidence_ids=("asr_001",),
        prompt_version="chapter-vlm-repair-v1",
    )

    _client(tmp_path, httpx.MockTransport(handler)).repair_chapter(
        repair,
        allowed_run_root=run_root,
    )

    payload = payloads[0]
    assert payload["response_format"]["json_schema"]["name"] == (  # type: ignore[index]
        "chapter_vlm_repair_v2"
    )
    assert "chapter-vlm-repair-v1" in payload["messages"][0]["content"]  # type: ignore[index]
    content = payload["messages"][1]["content"]  # type: ignore[index]
    labels = [item["text"] for item in content if item.get("type") == "text"]  # type: ignore[union-attr]
    assert labels.index("FRAME_ID=frame_a") < labels.index("FRAME_ID=frame_b")
    assert sum(item.get("type") == "image_url" for item in content) == 2  # type: ignore[union-attr]
    assert "d" * 64 in labels[0]

    ordered = sorted(original.frames, key=lambda item: (item.timestamp_ms, item.frame_id))
    upper_bound = vision_payload_size_upper_bound(
        prompt_for_vision_repair(repair),
        model_id="qwen3-vl-flash",
        schema_name="chapter_vlm_repair_v2",
        response_schema=ChapterVisionResponse.model_json_schema(),
        ordered_frames=tuple((frame.frame_id, frame.size_bytes) for frame in ordered),
    )
    actual_size = len(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(),
    )
    assert actual_size <= upper_bound


def test_qwen_repair_rejects_changed_or_reordered_frame_whitelist(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "scope_001" / "run_001"
    run_root.mkdir(parents=True)

    with pytest.raises(ValidationError, match="allowed_frame_ids"):
        ChapterVisionRepairRequest(
            request=_request(run_root),
            invalid_response=InvalidModelResponse(
                content_sha256="d" * 64,
                validation_errors=("response:invalid",),
            ),
            allowed_frame_ids=("frame_b", "frame_a"),
            allowed_target_ids=("target_001",),
            allowed_transcript_evidence_ids=("asr_001",),
            prompt_version="chapter-vlm-repair-v1",
        )


def test_qwen_response_uses_configured_byte_limit(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "scope_001" / "run_001"
    run_root.mkdir(parents=True)

    def handler(request: httpx.Request) -> httpx.Response:
        return _provider_response(request, {"observations": []})

    with pytest.raises(VideoDemoError) as raised:
        _client(
            tmp_path,
            httpx.MockTransport(handler),
            max_response_bytes=10,
        ).analyze_chapter(_request(run_root), allowed_run_root=run_root)
    assert raised.value.code == ErrorCode.QWEN_RESPONSE_INVALID


def test_qwen_non_json_message_hashes_message_content(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "scope_001" / "run_001"
    run_root.mkdir(parents=True)
    raw_message = "not-json-vision-content"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": raw_message}}]},
            request=request,
        )

    with pytest.raises(ModelResponseValidationError) as raised:
        _client(tmp_path, httpx.MockTransport(handler)).analyze_chapter(
            _request(run_root),
            allowed_run_root=run_root,
        )
    assert raised.value.invalid_response.content_sha256 == hashlib.sha256(
        raw_message.encode("utf-8"),
    ).hexdigest()
    assert raised.value.invalid_response.safe_json_excerpt is None


def test_qwen_rejects_non_standard_json_constants_without_safe_excerpt(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs" / "scope_001" / "run_001"
    run_root.mkdir(parents=True)
    raw_message = json.dumps(_valid_observation()).replace("0.9", "NaN")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": raw_message}}]},
            request=request,
        )

    with pytest.raises(ModelResponseValidationError) as raised:
        _client(tmp_path, httpx.MockTransport(handler)).analyze_chapter(
            _request(run_root),
            allowed_run_root=run_root,
        )

    assert raised.value.code == ErrorCode.QWEN_RESPONSE_INVALID
    assert raised.value.invalid_response.safe_json_excerpt is None


def test_qwen_authentication_status_is_not_masked_by_oversized_body(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "scope_001" / "run_001"
    run_root.mkdir(parents=True)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, content=b"x" * 100, request=request)

    with pytest.raises(VideoDemoError) as raised:
        _client(
            tmp_path,
            httpx.MockTransport(handler),
            max_response_bytes=10,
        ).analyze_chapter(_request(run_root), allowed_run_root=run_root)
    assert raised.value.code == ErrorCode.QWEN_AUTHENTICATION_FAILED
