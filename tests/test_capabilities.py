from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_demo.capabilities import CapabilityStatus, probe_runtime_capabilities
from video_demo.config import Settings
from video_demo.errors import ErrorCode


def _settings(workspace: Path) -> Settings:
    return Settings(
        workspace_root=workspace,
        openai_base_url="https://ai-proxy.example.test/v1",
        openai_api_key="test-openai-key",
        openai_model="openai/whisper",
        _env_file=None,
    )


class RuntimeCapabilitiesTest(unittest.TestCase):
    def test_missing_binaries_are_reported_as_explicit_errors(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = probe_runtime_capabilities(_settings(Path(directory)))

        self.assertEqual(report.status, CapabilityStatus.UNAVAILABLE)
        self.assertEqual(
            {issue.code for issue in report.issues},
            {ErrorCode.VIDEO_FFMPEG_UNAVAILABLE, ErrorCode.VIDEO_FFPROBE_UNAVAILABLE},
        )

    def test_system_path_binaries_are_ignored(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ",
            {"PATH": f"{directory}/system-tools"},
        ):
            root = Path(directory)
            system_tools = root / "system-tools"
            system_tools.mkdir()
            for name in ("ffmpeg", "ffprobe"):
                binary = system_tools / name
                binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                binary.chmod(0o755)
            workspace = root / "workspace"
            workspace.mkdir()

            report = probe_runtime_capabilities(_settings(workspace))

        self.assertEqual(report.status, CapabilityStatus.UNAVAILABLE)
        self.assertEqual(report.binaries, ())
        self.assertEqual(
            {issue.code for issue in report.issues},
            {ErrorCode.VIDEO_FFMPEG_UNAVAILABLE, ErrorCode.VIDEO_FFPROBE_UNAVAILABLE},
        )

    def test_parent_symlink_cannot_redirect_binaries_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            runtime = workspace / ".codex" / "video-rag-demo"
            runtime.mkdir(parents=True)
            external_tools = root / "external-tools"
            external_tools.mkdir()
            for name in ("ffmpeg", "ffprobe"):
                binary = external_tools / name
                binary.write_text("#!/bin/sh\necho external-version\n", encoding="utf-8")
                binary.chmod(0o755)
            (runtime / "tools").symlink_to(external_tools, target_is_directory=True)

            report = probe_runtime_capabilities(_settings(workspace))

        self.assertEqual(report.status, CapabilityStatus.UNAVAILABLE)
        self.assertEqual(report.binaries, ())
        self.assertEqual(
            {issue.code for issue in report.issues},
            {ErrorCode.VIDEO_FFMPEG_UNAVAILABLE, ErrorCode.VIDEO_FFPROBE_UNAVAILABLE},
        )

    def test_binary_version_output_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            tools = workspace / ".codex" / "video-rag-demo" / "tools"
            tools.mkdir(parents=True)
            for name in ("ffmpeg", "ffprobe"):
                binary = tools / name
                binary.write_text(
                    "#!/bin/sh\nprintf '%070000d\\n' 0\n",
                    encoding="utf-8",
                )
                binary.chmod(0o755)

            report = probe_runtime_capabilities(_settings(workspace))

        self.assertEqual(report.status, CapabilityStatus.UNAVAILABLE)
        self.assertEqual(report.binaries, ())
        self.assertEqual(
            {issue.code for issue in report.issues},
            {ErrorCode.VIDEO_BINARY_PROBE_FAILED},
        )

    @patch("video_demo.capabilities.platform.machine", return_value="arm64")
    @patch("video_demo.capabilities.platform.system", return_value="Darwin")
    @patch("video_demo.capabilities._read_binary_version", return_value="7.1")
    def test_apple_silicon_report_exposes_cloud_asr_configuration_without_secrets(
        self,
        _version: object,
        _system: object,
        _machine: object,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            tools = workspace / ".codex" / "video-rag-demo" / "tools"
            tools.mkdir(parents=True)
            for name in ("ffmpeg", "ffprobe"):
                binary = tools / name
                binary.write_bytes(b"binary")
                binary.chmod(0o755)

            report = probe_runtime_capabilities(
                _settings(workspace)
            )

        self.assertEqual(report.status, CapabilityStatus.AVAILABLE)
        self.assertEqual(report.platform, "macOS-arm64")
        self.assertEqual(report.cloud_asr_provider, "openai_compatible")
        self.assertEqual(report.cloud_asr_model, "openai/whisper")
        self.assertEqual(report.cloud_asr_base_url, "https://ai-proxy.example.test/v1")
        self.assertTrue(report.cloud_asr_configured)
        serialized = report.model_dump_json()
        self.assertNotIn("test-openai-key", serialized)
        self.assertNotIn("inference_device", serialized)
        self.assertNotIn("whisper_compute_type", serialized)

    @patch(
        "video_demo.capabilities._read_binary_version",
        side_effect=TimeoutError("probe timed out"),
    )
    def test_binary_probe_failure_is_reported_without_escaping(
        self,
        _version: object,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            tools = workspace / ".codex" / "video-rag-demo" / "tools"
            tools.mkdir(parents=True)
            for name in ("ffmpeg", "ffprobe"):
                binary = tools / name
                binary.write_bytes(b"binary")
                binary.chmod(0o755)

            report = probe_runtime_capabilities(_settings(workspace))

        self.assertEqual(report.status, CapabilityStatus.UNAVAILABLE)
        self.assertEqual(report.binaries, ())
        self.assertEqual(len(report.issues), 2)
        self.assertEqual(
            {issue.code for issue in report.issues},
            {ErrorCode.VIDEO_BINARY_PROBE_FAILED},
        )
        self.assertTrue(all("probe timed out" not in issue.message for issue in report.issues))


if __name__ == "__main__":
    unittest.main()
