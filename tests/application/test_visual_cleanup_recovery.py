"""视觉恢复编排的模块职责门禁。"""

from video_demo.application import visual_cleanup, visual_cleanup_recovery


def test_recovery_is_physically_separated_from_fd_cleanup() -> None:
    assert not hasattr(visual_cleanup, "PublishedVisualCleanupRecovery")
    assert hasattr(visual_cleanup_recovery, "PublishedVisualCleanupRecovery")
