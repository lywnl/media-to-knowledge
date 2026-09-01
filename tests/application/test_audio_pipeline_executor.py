from __future__ import annotations

import hashlib
from pathlib import Path

from video_demo.application.audio_contracts import (
    AudioTranscriptionCheckpoint,
)
from video_demo.application.audio_run_config import AudioRunConfig
from video_demo.domain.audio_plan import AudioBaseSegment
from video_demo.domain.evidence import SpeechSegment
from video_demo.media.audio_format import AUDIO_FORMAT_VERSION
from video_demo.persistence.audio_stage_repository import AudioStageRepository
from video_demo.persistence.database import Database
from video_demo.persistence.media_repositories import MediaRunRepository
from video_demo.persistence.models import (
    AudioObjectModel,
    AudioStageName,
    AudioUnderstandingRunModel,
    JobStatus,
    VideoObjectStatus,
)
from video_demo.persistence.repositories import JobRepository
from video_demo.persistence.scope import Scope


def test_audio_checkpoint_format_detection_distinguishes_mp3_and_legacy_wav() -> None:
    from video_demo.application.audio_pipeline_executor import _is_stale_audio_checkpoint

    current = {
        "schema_version": "2.0.0",
        "audio_format_version": AUDIO_FORMAT_VERSION,
    }

    assert _is_stale_audio_checkpoint(current) is False
    assert _is_stale_audio_checkpoint({**current, "schema_version": "1.0.0"}) is True
    assert _is_stale_audio_checkpoint({**current, "audio_format_version": None}) is True
    assert _is_stale_audio_checkpoint({**current, "audio_path": "media/audio.wav"}) is True
    assert _is_stale_audio_checkpoint({**current, "audio_path": "media/audio.m4a"}) is True


def test_audio_stage_executor_runs_transcription_and_persists_checkpoint(tmp_path: Path) -> None:
    from video_demo.application.audio_pipeline_executor import AudioStagePipelineExecutor

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    source = runtime_root / "audio_object" / "object.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"ID3" + b"audio")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    database = Database(f"sqlite+pysqlite:///{runtime_root / 'audio.db'}")
    database.create_schema()
    scope = Scope("tenant", "app", "kb")
    run_id = "run_audio_executor_001"
    object_ref = "obj_audio_executor_001"
    with database.session() as session:
        session.add(
            AudioObjectModel(
                tenant_id=scope.tenant_id,
                application_id=scope.application_id,
                knowledge_base_id=scope.knowledge_base_id,
                object_ref=object_ref,
                original_filename="sample.mp3",
                declared_mime="audio/mpeg",
                detected_mime="audio/mpeg",
                size_bytes=source.stat().st_size,
                sha256=digest,
                relative_path="audio_object/object.mp3",
                status=VideoObjectStatus.READY,
                scan_details={},
            ),
        )
        MediaRunRepository(session, AudioUnderstandingRunModel).add(
            scope=scope,
            run_id=run_id,
            object_ref=object_ref,
            idempotency_key="audio-executor-idempotency-001",
            config_snapshot=AudioRunConfig().model_dump(mode="json"),
        )
        JobRepository(session).enqueue_media_run(
            scope=scope,
            job_id="job_audio_executor_001",
            resource_id=run_id,
            job_type="AUDIO_UNDERSTANDING",
            resource_type="AUDIO_UNDERSTANDING_RUN",
        )
        AudioStageRepository(session).ensure(scope, run_id)

    heartbeat_state = {"active": False, "probe": False, "transcoder": False}

    class Probe:
        def probe(self, _source: Path, *, max_duration_ms: int):
            heartbeat_state["probe"] = heartbeat_state["active"]
            assert max_duration_ms == 7_200_000
            return type("ProbeResult", (), {"duration_ms": 1_000})()

    class Transcoder:
        def extract_audio(
            self,
            _source: Path,
            run_relative_root: Path,
            *,
            has_audio: bool,
            duration_ms: int,
        ):
            heartbeat_state["transcoder"] = heartbeat_state["active"]
            assert has_audio is True
            assert duration_ms == 1_000
            relative = run_relative_root / "media" / "audio.mp3"
            output = runtime_root / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"mp3")
            return type(
                "AudioArtifact",
                (),
                {
                    "relative_path": relative.as_posix(),
                    "sha256": hashlib.sha256(b"mp3").hexdigest(),
                    "size_bytes": 3,
                },
            )()

    evidence = SpeechSegment(
        evidence_id="asr_executor_001",
        start_ms=0,
        end_ms=1_000,
        text="音频内容",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    checkpoint = AudioTranscriptionCheckpoint(
        run_id=run_id,
        asset_sha256=digest,
        audio_format_version=AUDIO_FORMAT_VERSION,
        duration_ms=1_000,
        title_hint="sample",
        transcript_source="ASR",
        transcript_evidence=(evidence,),
        base_segments=(
            AudioBaseSegment(
                segment_id="audio_segment_executor_001",
                start_ms=0,
                end_ms=1_000,
                evidence_refs=(evidence.evidence_id,),
                transcript_source="ASR",
            ),
        ),
    )

    class Pipeline:
        def run_transcription(self, **kwargs: object) -> AudioTranscriptionCheckpoint:
            assert kwargs["duration_ms"] == 1_000
            assert kwargs["config"] == AudioRunConfig()
            return checkpoint

    from video_demo.storage.audio_object_store import AudioObjectStore

    factory_callbacks: list[object] = []

    def pipeline_factory(
        duration_ms: int,
        is_cancel_requested: object,
    ) -> Pipeline:
        assert duration_ms == 1_000
        factory_callbacks.append(is_cancel_requested)
        return Pipeline()

    class TrackingExecutor(AudioStagePipelineExecutor):
        def _run_with_heartbeat(self, lease, job, operation):  # type: ignore[no-untyped-def]
            heartbeat_state["active"] = True
            try:
                return operation()
            finally:
                heartbeat_state["active"] = False

    executor = TrackingExecutor(
        database,
        pipeline_factory,
        Probe(),
        Transcoder(),
        AudioObjectStore(runtime_root, max_bytes=1024 * 1024),
        runtime_root,
    )

    restored = executor.run_transcription(scope, run_id)

    assert restored == checkpoint
    assert executor.load_checkpoint(scope, run_id) == checkpoint
    assert heartbeat_state["probe"] is True
    assert heartbeat_state["transcoder"] is True
    assert len(factory_callbacks) == 1
    assert callable(factory_callbacks[0])
    with database.session() as session:
        stage = AudioStageRepository(session).get(scope, run_id, AudioStageName.TRANSCRIPTION)
        job = JobRepository(session).get(scope, "job_audio_executor_001")
        assert stage is not None and stage.status == JobStatus.SUCCEEDED
        assert stage.checkpoint_relative_path is not None
        assert job is not None and job.status == JobStatus.PENDING
        assert job.resource_type == "AUDIO_UNDERSTANDING_RUN"
