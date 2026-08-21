from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from video_demo.config import Settings, resolve_workspace_path
from video_demo.errors import ErrorCode, VideoDemoError


class SettingsTest(unittest.TestCase):
    def test_defaults_keep_runtime_inside_workspace_and_use_m1_safe_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)

            settings = Settings(workspace_root=workspace)

            self.assertEqual(
                settings.runtime_root,
                workspace.resolve() / ".codex" / "video-rag-demo",
            )
            self.assertEqual(settings.inference_device, "cpu")
            self.assertEqual(settings.whisper_compute_type, "int8")
            self.assertEqual(settings.worker_concurrency, 1)
            self.assertEqual(settings.qwen_max_video_bytes, 64 * 1024 * 1024)
            self.assertEqual(settings.qwen_max_video_duration_ms, 30_000)
            self.assertEqual(settings.qwen_timeout_seconds, 300.0)
            self.assertEqual(settings.oss_prefix, "video-demo/qwen-clips")
            self.assertEqual(settings.oss_signed_url_ttl_seconds, 3_600)
            self.assertFalse(settings.has_complete_oss_configuration())
            self.assertFalse(settings.demo_degraded_mode)

    def test_complete_oss_configuration_is_available_without_serializing_secrets(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                workspace_root=Path(directory),
                oss_endpoint="https://oss-cn-hangzhou.aliyuncs.com",
                oss_bucket="private-video-bucket",
                oss_access_key_id="test-access-key-id",
                oss_access_key_secret="test-access-key-secret",
            )

            serialized = settings.model_dump_json()

            self.assertTrue(settings.has_complete_oss_configuration())
            self.assertNotIn("test-access-key-id", serialized)
            self.assertNotIn("test-access-key-secret", serialized)

    def test_partial_oss_configuration_is_rejected(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValidationError, "OSS 配置必须全部提供或全部留空"),
        ):
            Settings(
                workspace_root=Path(directory),
                oss_endpoint="https://oss-cn-hangzhou.aliyuncs.com",
                oss_bucket="private-video-bucket",
            )

    def test_oss_prefix_and_signed_url_ttl_are_strictly_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            for values in (
                {"oss_prefix": "../qwen-clips"},
                {"oss_prefix": "video-demo//qwen-clips"},
                {"oss_prefix": "video-demo\\qwen-clips"},
                {"oss_signed_url_ttl_seconds": 59},
                {"oss_signed_url_ttl_seconds": 86_401},
            ):
                with self.subTest(values=values), self.assertRaises(ValidationError):
                    Settings(workspace_root=workspace, **values)  # type: ignore[arg-type]

    def test_demo_degraded_mode_is_explicitly_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(workspace_root=Path(directory), demo_degraded_mode=True)

            self.assertTrue(settings.demo_degraded_mode)

    def test_qwen_limits_reject_non_finite_timeout_and_duration_above_30_seconds(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            for values in (
                {"qwen_timeout_seconds": float("nan")},
                {"qwen_timeout_seconds": float("inf")},
                {"qwen_timeout_seconds": float("-inf")},
                {"qwen_timeout_seconds": 0.0},
                {"qwen_max_video_duration_ms": 30_001},
                {"qwen_max_video_duration_ms": 0},
                {"qwen_max_video_bytes": 0},
            ):
                with self.subTest(values=values), self.assertRaises(ValidationError):
                    Settings(workspace_root=workspace, **values)  # type: ignore[arg-type]

    def test_resolve_workspace_path_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)

            with self.assertRaises(VideoDemoError) as raised:
                resolve_workspace_path(workspace, Path("../outside"))

            self.assertEqual(raised.exception.code, ErrorCode.WORKSPACE_PATH_ESCAPE)

    def test_resolve_workspace_path_rejects_symlink_escape(self) -> None:
        with (
            tempfile.TemporaryDirectory() as workspace_dir,
            tempfile.TemporaryDirectory() as outside_dir,
        ):
            workspace = Path(workspace_dir)
            link = workspace / "escaped"
            link.symlink_to(Path(outside_dir), target_is_directory=True)

            with self.assertRaises(VideoDemoError) as raised:
                resolve_workspace_path(workspace, Path("escaped/result.json"))

            self.assertEqual(raised.exception.code, ErrorCode.WORKSPACE_PATH_ESCAPE)

    def test_secret_values_are_excluded_from_serialized_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            secret = "qwen-secret-value"
            settings = Settings(
                workspace_root=Path(directory),
                qwen_api_key=secret,
                baidu_api_key="baidu-api-secret",
                baidu_secret_key="baidu-secret-value",
                huggingface_token="hf-secret-value",
            )

            serialized = settings.model_dump_json()

            self.assertNotIn(secret, serialized)
            self.assertNotIn("baidu-api-secret", serialized)
            self.assertNotIn("baidu-secret-value", serialized)
            self.assertNotIn("hf-secret-value", serialized)


if __name__ == "__main__":
    unittest.main()
