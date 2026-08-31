from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from video_demo.application.pipeline import (
    PipelineRunConfig,
    PreparedMedia,
    ProbedAsset,
    RegisteredAsset,
)
from video_demo.config import Settings
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.evaluation.evidence import (
    ArtifactRole,
    EvidenceStore,
    RealMediaCommand,
    RealMediaFile,
    RealMediaSample,
    TraceArtifact,
)
from video_demo.evaluation.real_media_source import (
    CASE_IDS,
    MAX_MEDIA_BYTES,
    CaseDirectoryConflict,
    CaseExecutionSession,
    generate_source,
    open_case_execution_session,
    regular_file_size,
)
from video_demo.media.probe import FFprobeClient, ProbeLimits
from video_demo.media.process import SafeProcessRunner
from video_demo.media.transcode import (
    AudioArtifact,
    FFmpegTranscoder,
    NoAudioArtifact,
    TranscodeLimits,
)
from video_demo.visual.candidate_artifacts import CandidateArtifactSession
from video_demo.visual.ffmpeg_frames import FFmpegFrameExtractor, FrameCandidate, FrameSample

MediaRole = Literal["SOURCE", "AUDIO", "KEYFRAME"]
MediaFormat = Literal["MP4", "WAV", "JPEG"]
MediaPhase = Literal[
    "generate",
    "probe",
    "audio",
    "ffmpeg_frame_extract",
]
MediaExecutable = Literal[
    "ffmpeg",
    "ffprobe",
    "FFmpegTranscoder",
    "FFmpegFrameExtractor",
]


class MediaJournal(Protocol):
    def begin_phase(
        self,
        *,
        case_id: str,
        phase: MediaPhase,
        executable: MediaExecutable,
        arguments: tuple[str, ...] = (),
        input_relative_paths: tuple[str, ...] = (),
        output_relative_paths: tuple[str, ...] = (),
    ) -> None: ...

    def write_current_outputs(self) -> tuple[TraceArtifact, TraceArtifact]: ...

    def record_media_file(
        self,
        case_id: str,
        media_file: RealMediaFile,
        artifact: TraceArtifact,
    ) -> None: ...

    def is_media_file_registered(self, case_id: str, relative_path: str) -> bool: ...

    def record_completed_command(
        self,
        command: RealMediaCommand,
        artifacts: tuple[TraceArtifact, ...],
    ) -> None: ...

    def finalize_sample(
        self,
        sample: RealMediaSample,
        artifacts: tuple[TraceArtifact, ...],
        verify_case_closed: Callable[[], None],
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class TransferredMedia:
    media_file: RealMediaFile
    artifact: TraceArtifact


@dataclass(frozen=True, slots=True)
class ExecutionFacts:
    samples: tuple[RealMediaSample, ...]
    artifacts: tuple[TraceArtifact, ...]


@dataclass(slots=True)
class _CaseFacts:
    files: list[RealMediaFile]
    commands: list[RealMediaCommand]
    artifacts: list[TraceArtifact]
    registered_paths: set[Path]


def transfer_media_file(
    *,
    case_id: str,
    path: Path,
    role: MediaRole,
    format_name: MediaFormat,
    store: EvidenceStore,
    journal: MediaJournal,
    max_bytes: int,
    session: CaseExecutionSession | None = None,
) -> TransferredMedia:
    """journal 返回成功才转移所有权；此前任一异常都删除当前产物。"""

    case_relative: Path | None = None
    trusted_after = None
    try:
        relative = path.relative_to(store.runtime_root)
        case_relative = path.relative_to(session.case_root) if session else None
        trusted_before = (
            session.snapshot_leaf(case_relative, max_bytes)
            if session is not None and case_relative is not None
            else None
        )
        if (
            session is not None
            and case_relative is not None
            and trusted_before is not None
            and role == "SOURCE"
        ):
            session.assert_published_source(case_relative, trusted_before, max_bytes)
        artifact_role: ArtifactRole = "INPUT_MEDIA" if role == "SOURCE" else "OUTPUT_MEDIA"
        if session is not None and case_relative is not None:
            trusted_after = session.snapshot_leaf(case_relative, max_bytes)
            if role == "SOURCE":
                session.assert_published_source(case_relative, trusted_after, max_bytes)
            if trusted_before != trusted_after:
                raise VideoDemoError(
                    ErrorCode.WORKSPACE_PATH_ESCAPE,
                    "媒体绑定对象与 case fd 叶不一致",
                )
            artifact = TraceArtifact(
                role=artifact_role,
                relative_path=path.relative_to(store.workspace_root).as_posix(),
                sha256=trusted_after.sha256,
                max_bytes=max_bytes,
            )
            size_bytes = trusted_after.size
        else:
            trusted_after = None
            artifact = store.bind_artifact(relative, artifact_role, max_bytes=max_bytes)
            size_bytes = regular_file_size(path, max_bytes)
        media_file = RealMediaFile(
            role=role,
            format=format_name,
            relative_path=artifact.relative_path,
            sha256=artifact.sha256,
            size_bytes=size_bytes,
        )
        if session is not None:
            session.assert_current()
        journal.record_media_file(case_id, media_file, artifact)
        if session is not None and case_relative is not None and trusted_after is not None:
            session.remember_registered_leaf(case_relative, trusted_after)
        return TransferredMedia(media_file=media_file, artifact=artifact)
    except BaseException:
        relative_path = path.relative_to(store.workspace_root).as_posix()
        registered = journal.is_media_file_registered(case_id, relative_path)
        if (
            registered
            and session is not None
            and case_relative is not None
            and trusted_after is not None
        ):
            session.remember_registered_leaf(case_relative, trusted_after)
        if not registered:
            if session is not None and case_relative is not None:
                session.remove_leaf(case_relative)
            else:
                _unlink_leaf(path)
        raise


def execute_real_media(
    *,
    evaluation_run_id: str,
    binaries: dict[str, Path],
    settings: Settings,
    store: EvidenceStore,
    journal: MediaJournal,
) -> ExecutionFacts:
    """按固定 case 与阶段调用生产 adapter，并只返回已登记的中立事实。"""

    assert settings.runtime_root is not None
    max_bytes = min(settings.max_video_bytes, MAX_MEDIA_BYTES)
    generator = SafeProcessRunner(
        max_output_bytes=64 * 1024,
        workspace_root=settings.workspace_root,
    )
    probe = FFprobeClient(
        binaries["ffprobe"],
        SafeProcessRunner(
            max_output_bytes=32 * 1024 * 1024,
            workspace_root=settings.workspace_root,
        ),
        "verified",
        timeout_seconds=settings.process_timeout_seconds,
    )
    transcoder = FFmpegTranscoder(
        executable=binaries["ffmpeg"],
        runner=SafeProcessRunner(
            max_output_bytes=16 * 1024 * 1024,
            workspace_root=settings.workspace_root,
        ),
        runtime_root=settings.runtime_root,
        limits=TranscodeLimits(
            max_output_bytes=max_bytes,
            timeout_seconds=settings.process_timeout_seconds,
        ),
    )
    # 评测链路保留固定六帧采样，避免历史评测样本的解码计数与生产成本策略耦合。
    extractor = FFmpegFrameExtractor.from_path(
        binaries["ffmpeg"],
        settings.runtime_root,
        workspace_root=settings.workspace_root,
        max_frame_bytes=settings.vlm_max_image_bytes,
        frame_width=settings.visual_proxy_max_edge,
        jpeg_quality=settings.keyframe_jpeg_quality,
    )
    samples: list[RealMediaSample] = []
    artifacts: list[TraceArtifact] = []
    for case_id in CASE_IDS:
        sample, sample_artifacts = _execute_case(
            case_id=case_id,
            evaluation_run_id=evaluation_run_id,
            ffmpeg=binaries["ffmpeg"],
            generator=generator,
            probe=probe,
            transcoder=transcoder,
            extractor=extractor,
            settings=settings,
            store=store,
            journal=journal,
            max_bytes=max_bytes,
        )
        samples.append(sample)
        artifacts.extend(sample_artifacts)
    return ExecutionFacts(samples=tuple(samples), artifacts=tuple(artifacts))


def _execute_case(
    *,
    case_id: str,
    evaluation_run_id: str,
    ffmpeg: Path,
    generator: SafeProcessRunner,
    probe: FFprobeClient,
    transcoder: FFmpegTranscoder,
    extractor: FFmpegFrameExtractor,
    settings: Settings,
    store: EvidenceStore,
    journal: MediaJournal,
    max_bytes: int,
) -> tuple[RealMediaSample, tuple[TraceArtifact, ...]]:
    assert settings.runtime_root is not None
    run_root = Path("eval/generated") / evaluation_run_id / case_id
    facts = _CaseFacts(files=[], commands=[], artifacts=[], registered_paths=set())
    session: CaseExecutionSession | None = None
    try:
        session = open_case_execution_session(
            settings.runtime_root,
            evaluation_run_id,
            case_id,
        )
        source = _generate_phase(
            case_id,
            ffmpeg,
            generator,
            settings,
            store,
            journal,
            facts,
            session,
        )
        session.assert_registered_leaves(facts.registered_paths)
        registered = _registered_asset(source, run_root, facts.files[0])
        probed = _probe_phase(case_id, registered, probe, journal, facts, session)
        session.assert_registered_leaves(facts.registered_paths)
        prepared = _audio_phase(
            case_id,
            probed,
            transcoder,
            store,
            journal,
            facts,
            max_bytes,
            session,
        )
        session.assert_registered_leaves(facts.registered_paths)
        selected = _ffmpeg_frame_phase(
            case_id, prepared, extractor, journal, facts, session, store, max_bytes
        )
        session.assert_registered_leaves(facts.registered_paths)
        sample = RealMediaSample(
            case_id=case_id,
            execution_status="SUCCESS",
            duration_ms=probed.duration_ms,
            has_audio=bool(probed.manifest.audio_streams),
            rotation_degrees=probed.manifest.video_stream.rotation_degrees,
            is_variable_frame_rate=probed.manifest.video_stream.is_variable_frame_rate,
            warnings=prepared.warnings,
            selected_keyframe_count=len(selected),
            files=tuple(facts.files),
            commands=tuple(facts.commands),
        )
        journal.finalize_sample(
            sample,
            tuple(facts.artifacts),
            lambda: _assert_case_media_closed(session, facts.registered_paths),
        )
        return sample, tuple(facts.artifacts)
    except CaseDirectoryConflict:
        raise
    except BaseException:
        if session is not None:
            session.cleanup_unregistered(facts.registered_paths)
        raise
    finally:
        if session is not None:
            session.close()


def _generate_phase(
    case_id: str,
    executable: Path,
    runner: SafeProcessRunner,
    settings: Settings,
    store: EvidenceStore,
    journal: MediaJournal,
    facts: _CaseFacts,
    session: CaseExecutionSession,
) -> Path:
    assert settings.runtime_root is not None
    output = (session.case_root / "source.mp4").relative_to(settings.workspace_root)
    journal.begin_phase(
        case_id=case_id,
        phase="generate",
        executable="ffmpeg",
        arguments=("lavfi", "testsrc2", _generation_label(case_id)),
        output_relative_paths=(output.as_posix(),),
    )
    generated = generate_source(
        session=session,
        case_id=case_id,
        executable=executable,
        runner=runner,
        max_bytes=settings.max_video_bytes,
        timeout_seconds=settings.process_timeout_seconds,
    )
    if generated.process_result.returncode == 0:
        transferred = _transfer_to_journal(
            case_id=case_id,
            path=generated.path,
            role="SOURCE",
            format_name="MP4",
            store=store,
            journal=journal,
            max_bytes=min(settings.max_video_bytes, MAX_MEDIA_BYTES),
            facts=facts,
            session=session,
        )
        _accept_transfer(transferred, facts)
    _complete_phase(
        journal,
        facts,
        phase="generate",
        executable="ffmpeg",
        arguments=("lavfi", "testsrc2", _generation_label(case_id)),
        inputs=(),
        outputs=(output.as_posix(),),
        exit_code=generated.process_result.returncode,
    )
    return generated.path


def _probe_phase(
    case_id: str,
    registered: RegisteredAsset,
    client: FFprobeClient,
    journal: MediaJournal,
    facts: _CaseFacts,
    session: CaseExecutionSession,
) -> ProbedAsset:
    session.assert_registered_leaves(facts.registered_paths)
    source = facts.files[0].relative_path
    journal.begin_phase(
        case_id=case_id,
        phase="probe",
        executable="ffprobe",
        arguments=("json",),
        input_relative_paths=(source,),
    )
    limits = ProbeLimits()
    source_descriptor = session.open_registered_leaf(Path("source.mp4"))
    try:
        result = client.probe(
            registered.source_path,
            object_ref=registered.object_ref,
            source_sha256=registered.source_sha256,
            source_size_bytes=registered.source_size_bytes,
            source_mime=registered.source_mime,
            limits=limits,
            input_fd=source_descriptor,
        )
    finally:
        os.close(source_descriptor)
    probed = ProbedAsset(
        asset=registered,
        manifest=result.manifest,
        limits=limits,
        warnings=result.warnings,
    )
    session.assert_registered_leaves(facts.registered_paths)
    _complete_phase(
        journal,
        facts,
        phase="probe",
        executable="ffprobe",
        arguments=("json",),
        inputs=(source,),
        outputs=(),
    )
    return probed


def _audio_phase(
    case_id: str,
    probed: ProbedAsset,
    client: FFmpegTranscoder,
    store: EvidenceStore,
    journal: MediaJournal,
    facts: _CaseFacts,
    max_bytes: int,
    session: CaseExecutionSession,
) -> PreparedMedia:
    session.assert_registered_leaves(facts.registered_paths)
    source = facts.files[0].relative_path
    expected_audio = probed.asset.run_relative_root / "media/audio.wav"
    outputs = (
        ()
        if not probed.manifest.audio_streams
        else (_runtime_file_relative(store, expected_audio),)
    )
    journal.begin_phase(
        case_id=case_id,
        phase="audio",
        executable="FFmpegTranscoder",
        arguments=("extract_audio",),
        input_relative_paths=(source,),
        output_relative_paths=outputs,
    )
    has_audio = bool(probed.manifest.audio_streams)
    if has_audio:
        source_descriptor = session.open_registered_leaf(Path("source.mp4"))
        staged = None
        try:
            staged = session.stage_output(Path("media/audio.wav"))
            audio = client.extract_audio(
                probed.asset.source_path,
                probed.asset.run_relative_root,
                has_audio=True,
                duration_ms=probed.duration_ms,
                input_fd=source_descriptor,
                output_fd=staged.descriptor,
            )
            session.publish_output(staged, max_bytes)
        finally:
            try:
                if staged is not None:
                    session.discard_output(staged)
            finally:
                os.close(source_descriptor)
    else:
        audio = client.extract_audio(
            probed.asset.source_path,
            probed.asset.run_relative_root,
            has_audio=False,
            duration_ms=probed.duration_ms,
        )
    warnings = list(probed.warnings)
    audio_path: Path | None = None
    audio_sha256: str | None = None
    if isinstance(audio, NoAudioArtifact):
        warnings.append(audio.warning_code)
    else:
        transferred = _transfer_audio(case_id, audio, store, journal, facts, max_bytes, session)
        audio_path = store.runtime_root / audio.relative_path
        audio_sha256 = transferred.media_file.sha256
    session.assert_registered_leaves(facts.registered_paths)
    _complete_phase(
        journal,
        facts,
        phase="audio",
        executable="FFmpegTranscoder",
        arguments=("extract_audio",),
        inputs=(source,),
        outputs=outputs,
    )
    return PreparedMedia(
        source=probed,
        proxy_path=store.workspace_root / facts.files[0].relative_path,
        proxy_sha256=facts.files[0].sha256,
        proxy_size_bytes=facts.files[0].size_bytes,
        audio_path=audio_path,
        audio_sha256=audio_sha256,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _ffmpeg_frame_phase(
    case_id: str,
    prepared: PreparedMedia,
    extractor: FFmpegFrameExtractor,
    journal: MediaJournal,
    facts: _CaseFacts,
    session: CaseExecutionSession,
    store: EvidenceStore,
    max_bytes: int,
) -> tuple[FrameCandidate, ...]:
    session.assert_registered_leaves(facts.registered_paths)
    source = _source_relative(facts)
    journal.begin_phase(
        case_id=case_id,
        phase="ffmpeg_frame_extract",
        executable="FFmpegFrameExtractor",
        arguments=("representative",),
        input_relative_paths=(source,),
    )
    run_root = prepared.source.asset.run_relative_root
    runtime_root = session.runtime_root
    candidate_session = CandidateArtifactSession(
        runtime_root=runtime_root,
        max_unique_bytes=max_bytes,
        max_files=16,
        max_file_bytes=max_bytes,
    )
    candidate_session.prepare_run(run_root)
    selected: tuple[FrameCandidate, ...] = ()
    outputs: tuple[str, ...] = ()
    try:
        timestamps = tuple(
            sorted(
                {
                    0,
                    prepared.source.duration_ms // 2,
                    max(0, prepared.source.duration_ms - 1),
                },
            ),
        )
        samples = tuple(
            FrameSample(f"eval-{index}", timestamp, "BASE_PRIMARY")
            for index, timestamp in enumerate(timestamps)
        )
        extracted = extractor.extract_samples(
            prepared.proxy_path,
            run_root,
            samples,
            is_cancel_requested=lambda: False,
            artifact_session=candidate_session,
        )
        selected = tuple(item.candidate for item in extracted if item.candidate is not None)
        if not selected:
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "FFmpeg 未产生候选帧")
        for frame in selected:
            transferred = _transfer_to_journal(
                case_id=case_id,
                path=runtime_root / run_root / frame.relative_path,
                role="KEYFRAME",
                format_name="JPEG",
                store=store,
                journal=journal,
                max_bytes=max_bytes,
                facts=facts,
                session=session,
            )
            _accept_transfer(transferred, facts)
        outputs = tuple(
            _runtime_file_relative(store, run_root / frame.relative_path) for frame in selected
        )
    finally:
        candidate_session.cleanup_unretained(
            frozenset(frame.sha256 for frame in selected),
        )
        candidate_session.close()
    session.assert_registered_leaves(facts.registered_paths)
    _complete_phase(
        journal,
        facts,
        phase="ffmpeg_frame_extract",
        executable="FFmpegFrameExtractor",
        arguments=("representative",),
        inputs=(source,),
        outputs=outputs,
    )
    return selected


def _complete_phase(
    journal: MediaJournal,
    facts: _CaseFacts,
    *,
    phase: MediaPhase,
    executable: MediaExecutable,
    arguments: tuple[str, ...],
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    exit_code: int = 0,
) -> None:
    stdout, stderr = journal.write_current_outputs()
    command = RealMediaCommand(
        phase=phase,
        executable=executable,
        arguments=arguments,
        input_relative_paths=inputs,
        output_relative_paths=outputs,
        exit_code=exit_code,
        stdout_relative_path=stdout.relative_path,
        stderr_relative_path=stderr.relative_path,
        stdout_sha256=stdout.sha256,
        stderr_sha256=stderr.sha256,
    )
    journal.record_completed_command(command, (stdout, stderr))
    facts.commands.append(command)
    facts.artifacts.extend((stdout, stderr))


def _registered_asset(
    source: Path,
    run_root: Path,
    media_file: RealMediaFile,
) -> RegisteredAsset:
    return RegisteredAsset(
        source_path=source,
        source_sha256=media_file.sha256,
        object_ref=f"real-media-{run_root.name}",
        source_size_bytes=media_file.size_bytes,
        source_mime="video/mp4",
        run_relative_root=run_root,
        config=PipelineRunConfig(),
    )


def _transfer_audio(
    case_id: str,
    audio: AudioArtifact,
    store: EvidenceStore,
    journal: MediaJournal,
    facts: _CaseFacts,
    max_bytes: int,
    session: CaseExecutionSession,
) -> TransferredMedia:
    transferred = _transfer_to_journal(
        case_id=case_id,
        path=store.runtime_root / audio.relative_path,
        role="AUDIO",
        format_name="WAV",
        store=store,
        journal=journal,
        max_bytes=max_bytes,
        facts=facts,
        session=session,
    )
    _accept_transfer(transferred, facts)
    return transferred


def _transfer_to_journal(
    *,
    case_id: str,
    path: Path,
    role: MediaRole,
    format_name: MediaFormat,
    store: EvidenceStore,
    journal: MediaJournal,
    max_bytes: int,
    facts: _CaseFacts,
    session: CaseExecutionSession,
) -> TransferredMedia:
    case_relative = path.relative_to(session.case_root)
    try:
        transferred = transfer_media_file(
            case_id=case_id,
            path=path,
            role=role,
            format_name=format_name,
            store=store,
            journal=journal,
            max_bytes=max_bytes,
            session=session,
        )
        facts.registered_paths.add(case_relative)
        return transferred
    except BaseException:
        relative_path = path.relative_to(store.workspace_root).as_posix()
        if journal.is_media_file_registered(case_id, relative_path):
            facts.registered_paths.add(case_relative)
        else:
            facts.registered_paths.discard(case_relative)
        raise


def _accept_transfer(transferred: TransferredMedia, facts: _CaseFacts) -> None:
    facts.files.append(transferred.media_file)
    facts.artifacts.append(transferred.artifact)


def _runtime_file_relative(store: EvidenceStore, relative: Path) -> str:
    return (store.runtime_root / relative).relative_to(store.workspace_root).as_posix()


def _workspace_relative(workspace_root: Path, path: Path) -> Path:
    return path.relative_to(workspace_root)


def _source_relative(facts: _CaseFacts) -> str:
    return next(media.relative_path for media in facts.files if media.role == "SOURCE")


def _generation_label(case_id: str) -> str:
    return {
        "normal_audio": "sine",
        "no_audio": "no_audio",
        "rotation": "rotate_90",
        "vfr": "selective_vfr",
    }[case_id]


def _assert_case_media_closed(
    session: CaseExecutionSession,
    registered: set[Path],
) -> None:
    session.assert_media_closed(registered)


def _unlink_leaf(path: Path) -> None:
    with suppress(IsADirectoryError, OSError):
        path.unlink(missing_ok=True)
