"""阶段 checkpoint 加载时的内部兼容状态。"""

from __future__ import annotations

from pathlib import Path

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage.workspace import reject_symlink_components


class CheckpointStaleError(VideoDemoError):
    """checkpoint 可解析但依赖已退役的内部音频格式，必须重新转写。"""

    def __init__(self, message: str = "转写 checkpoint 使用了已退役的音频格式") -> None:
        super().__init__(
            ErrorCode.ARTIFACT_SCHEMA_INVALID,
            message,
            {"reason": "AUDIO_FORMAT_OUTDATED"},
        )


def cleanup_stale_checkpoint_artifacts(
    runtime_root: Path,
    run_relative_root: Path,
    checkpoint_relative_path: str | None,
) -> None:
    """删除当前 Run 中已退役的音频派生产物，保留可复用的 MP3。"""

    try:
        run_root = reject_symlink_components(
            runtime_root,
            runtime_root / run_relative_root,
            message="旧 checkpoint 清理路径非法",
        )
    except VideoDemoError:
        return
    if not run_root.is_dir():
        return

    paths: list[Path] = [run_root / "media/audio.wav"]
    slices = run_root / "speech/slices"
    if slices.is_dir() and not slices.is_symlink():
        paths.extend(slices.glob("*.wav"))

    snapshots = run_root / "speech/snapshots/asr-windows"
    if snapshots.is_dir() and not snapshots.is_symlink():
        paths.extend(snapshots.glob("window-*.json"))

    if checkpoint_relative_path:
        checkpoint_path = Path(checkpoint_relative_path)
        if not checkpoint_path.is_absolute():
            try:
                candidate = reject_symlink_components(
                    runtime_root,
                    runtime_root / checkpoint_path,
                    message="旧 checkpoint 清理路径非法",
                )
            except VideoDemoError:
                candidate = None
            if candidate is not None and candidate.is_relative_to(run_root):
                paths.append(candidate)

    for path in dict.fromkeys(paths):
        if path.is_file() and not path.is_symlink():
            path.unlink()
