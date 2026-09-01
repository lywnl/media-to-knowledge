from pathlib import Path

from video_demo.application.checkpoint_contracts import cleanup_stale_checkpoint_artifacts


def test_cleanup_stale_checkpoint_artifacts_removes_legacy_audio_outputs(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    run_root = runtime_root / "runs/tenant_app_kb/run_legacy"
    (run_root / "media").mkdir(parents=True)
    (run_root / "speech/slices").mkdir(parents=True)
    (run_root / "speech/snapshots/asr-windows").mkdir(parents=True)
    checkpoint = run_root / "stages/transcription-checkpoint.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("legacy", encoding="utf-8")
    (run_root / "media/audio.wav").write_bytes(b"wav")
    (run_root / "speech/slices/window.wav").write_bytes(b"wav")
    (run_root / "speech/snapshots/asr-windows/window-old.json").write_text(
        "legacy",
        encoding="utf-8",
    )
    current_mp3 = run_root / "media/audio.mp3"
    current_mp3.write_bytes(b"mp3")

    cleanup_stale_checkpoint_artifacts(
        runtime_root,
        Path("runs/tenant_app_kb/run_legacy"),
        checkpoint.relative_to(runtime_root).as_posix(),
    )

    assert not checkpoint.exists()
    assert not (run_root / "media/audio.wav").exists()
    assert not (run_root / "speech/slices/window.wav").exists()
    assert not (run_root / "speech/snapshots/asr-windows/window-old.json").exists()
    assert current_mp3.exists()
