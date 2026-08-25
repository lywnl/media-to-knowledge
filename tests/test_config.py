from __future__ import annotations

import tempfile
import unittest
from os import environ
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from video_demo.application.composition import build_production_model_identity_report
from video_demo.config import Settings, resolve_workspace_path
from video_demo.errors import ErrorCode, VideoDemoError


class SettingsTest(unittest.TestCase):
    def test_candidate_artifact_limits_have_safe_defaults_and_reject_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            settings = Settings(workspace_root=workspace, _env_file=None)

            self.assertEqual(settings.max_candidate_frame_files_per_run, 20_000)
            self.assertEqual(settings.candidate_directory_lock_timeout_seconds, 300.0)
            self.assertEqual(settings.max_published_keyframe_files_per_run, 20_000)
            for field, value in (
                ("max_candidate_frame_files_per_run", 0),
                ("max_published_keyframe_files_per_run", 0),
                ("candidate_directory_lock_timeout_seconds", 0),
                ("candidate_directory_lock_timeout_seconds", float("inf")),
            ):
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    Settings(
                        workspace_root=workspace,
                        _env_file=None,
                        **{field: value},
                    )

    def test_text_llm_and_vlm_configuration_are_independent_and_vlm_has_default_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                workspace_root=Path(directory),
                text_llm_base_url="https://text.example.test/v1",
                text_llm_api_key="text-secret",
                text_llm_model_id="text-model",
                vlm_base_url="https://vision.example.test/v1",
                vlm_api_key="vision-secret",
                _env_file=None,
            )

            text = settings.require_text_llm_configuration()
            vision = settings.require_vlm_configuration()

            self.assertEqual(text.base_url, "https://text.example.test/v1")
            self.assertEqual(text.model_id, "text-model")
            self.assertEqual(vision.base_url, "https://vision.example.test/v1")
            self.assertEqual(vision.model_id, "qwen3-vl-flash")

    def test_model_configuration_rejects_partial_values_and_unsafe_http(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            for values in (
                {
                    "text_llm_base_url": "https://text.example.test/v1",
                    "text_llm_api_key": "secret",
                },
                {
                    "vlm_base_url": "http://vision.example.test/v1",
                    "vlm_api_key": "secret",
                },
            ):
                with self.subTest(values=values), self.assertRaises(ValidationError):
                    Settings(workspace_root=workspace, _env_file=None, **values)

    def test_explicit_vlm_model_without_endpoint_or_key_is_partial_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValidationError):
            Settings(
                workspace_root=Path(directory),
                vlm_model_id="qwen3-vl-plus",
                _env_file=None,
            )

    def test_local_http_model_endpoint_requires_explicit_localhost_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                workspace_root=Path(directory),
                vlm_base_url="http://127.0.0.1:8080/v1",
                vlm_api_key="vision-secret",
                allow_insecure_local_model_endpoint=True,
                _env_file=None,
            )
            self.assertEqual(settings.require_vlm_configuration().base_url, "http://127.0.0.1:8080/v1")

    def test_model_secrets_are_hidden_from_serialized_settings_and_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            secret = "new-model-secret"
            settings = Settings(
                workspace_root=Path(directory),
                text_llm_base_url="https://text.example.test/v1",
                text_llm_api_key=secret,
                text_llm_model_id="text-model",
                vlm_base_url="https://vision.example.test/v1",
                vlm_api_key="vision-secret",
                _env_file=None,
            )
            serialized = (repr(settings), repr(settings.model_dump()), settings.model_dump_json())
            self.assertTrue(all(secret not in value for value in serialized))

    def test_document_model_internal_concurrency_is_capped_at_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for field in ("vlm_concurrency", "chapter_writer_concurrency"):
                with self.subTest(field=field), self.assertRaises(ValidationError):
                    Settings(
                        workspace_root=Path(directory),
                        _env_file=None,
                        **{field: 3},
                    )

    def test_vlm_inflight_budget_must_cover_all_concurrent_requests(self) -> None:
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValidationError):
            Settings(
                workspace_root=Path(directory),
                vlm_concurrency=2,
                vlm_max_encoded_request_bytes=36 * 1024 * 1024,
                vlm_max_inflight_encoded_bytes=64 * 1024 * 1024,
                _env_file=None,
            )

    def test_retired_local_model_dotenv_keys_are_ignored_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            dotenv = workspace / ".env"
            dotenv.write_text(
                "\n".join(
                    (
                        "OPENAI_BASE_URL=https://ai-proxy.example.test/v1",
                        "OPENAI_API_KEY=test-openai-key",
                        "OPENAI_MODEL=openai/whisper",
                        "VIDEO_DEMO_INFERENCE_DEVICE=cpu",
                        "VIDEO_DEMO_WHISPER_COMPUTE_TYPE=int8",
                        "VIDEO_DEMO_WHISPER_MODEL_ID=medium",
                        "VIDEO_DEMO_SPEECH_ENRICHMENT_TIMEOUT_SECONDS=3600",
                        "VIDEO_DEMO_HUGGINGFACE_TOKEN=retired-test-token",
                    ),
                )
                + "\n",
                encoding="utf-8",
            )

            settings = Settings(workspace_root=workspace, _env_file=dotenv)

            configuration = settings.require_cloud_asr_configuration()
            self.assertEqual(configuration.model, "openai/whisper")
            self.assertTrue(
                {
                    "inference_device",
                    "whisper_compute_type",
                    "whisper_model_id",
                    "speech_enrichment_timeout_seconds",
                    "huggingface_token",
                }.isdisjoint(Settings.model_fields),
            )

    def test_unknown_dotenv_key_is_rejected_without_revealing_its_value(self) -> None:
        secret = "unknown-sensitive-test-value"
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            dotenv = workspace / ".env"
            dotenv.write_text(
                f"VIDEO_DEMO_UNKNOWN_CREDENTIAL={secret}\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValidationError) as raised:
                Settings(workspace_root=workspace, _env_file=dotenv)

            self.assertIn("extra_forbidden", str(raised.exception))
            self.assertNotIn(secret, str(raised.exception))

    def test_cloud_asr_configuration_requires_all_required_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(workspace_root=Path(directory), _env_file=None)

            with self.assertRaises(VideoDemoError) as raised:
                settings.require_cloud_asr_configuration()

            self.assertEqual(raised.exception.code, ErrorCode.INVALID_CONFIGURATION)

    def test_cloud_asr_configuration_normalizes_base_url_and_fixes_window_policy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                workspace_root=Path(directory),
                openai_base_url="https://ai-proxy.example.test/v1/",
                openai_api_key="test-openai-key",
                openai_model="openai/whisper",
                openai_asr_timeout_seconds=120.5,
                openai_asr_max_attempts=4,
                _env_file=None,
            )

            configuration = settings.require_cloud_asr_configuration()

            self.assertEqual(configuration.base_url, "https://ai-proxy.example.test/v1")
            self.assertEqual(configuration.model, "openai/whisper")
            self.assertEqual(configuration.timeout_seconds, 120.5)
            self.assertEqual(configuration.max_attempts, 4)
            self.assertEqual(configuration.max_window_ms, 600_000)
            self.assertEqual(configuration.overlap_ms, 1_000)

    def test_cloud_asr_environment_uses_exact_openai_names(self) -> None:
        values = {
            "OPENAI_BASE_URL": "https://ai-proxy.example.test/v1",
            "OPENAI_API_KEY": "test-openai-key",
            "OPENAI_MODEL": "openai/whisper",
            "OPENAI_ASR_TIMEOUT_SECONDS": "42.5",
            "OPENAI_ASR_MAX_ATTEMPTS": "2",
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            environ,
            values,
            clear=False,
        ):
            settings = Settings(workspace_root=Path(directory), _env_file=None)

            configuration = settings.require_cloud_asr_configuration()

            self.assertEqual(configuration.base_url, values["OPENAI_BASE_URL"])
            self.assertEqual(configuration.model, values["OPENAI_MODEL"])
            self.assertEqual(configuration.timeout_seconds, 42.5)
            self.assertEqual(configuration.max_attempts, 2)

    def test_cloud_asr_rejects_unsafe_or_endpoint_level_base_urls(self) -> None:
        invalid_urls = (
            "http://ai-proxy.example.test/v1",
            "https://user@ai-proxy.example.test/v1",
            "https://user:password@ai-proxy.example.test/v1",
            "https://ai-proxy.example.test/v1?tenant=a",
            "https://ai-proxy.example.test/v1#section",
            "https://ai-proxy.example.test/v1/audio/transcriptions",
        )
        with tempfile.TemporaryDirectory() as directory:
            for base_url in invalid_urls:
                with self.subTest(base_url=base_url):
                    settings = Settings(
                        workspace_root=Path(directory),
                        openai_base_url=base_url,
                        openai_api_key="test-openai-key",
                        openai_model="openai/whisper",
                        _env_file=None,
                    )

                    with self.assertRaises(VideoDemoError) as raised:
                        settings.require_cloud_asr_configuration()

                    self.assertEqual(
                        raised.exception.code,
                        ErrorCode.INVALID_CONFIGURATION,
                    )

    def test_cloud_asr_rejects_blank_secret_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for values in (
                {"openai_api_key": ""},
                {"openai_api_key": "   "},
                {"openai_model": ""},
                {"openai_model": "   "},
            ):
                with self.subTest(values=values):
                    configuration_values = {
                        "openai_base_url": "https://ai-proxy.example.test/v1",
                        "openai_api_key": "test-openai-key",
                        "openai_model": "openai/whisper",
                        **values,
                    }
                    settings = Settings(
                        workspace_root=Path(directory),
                        _env_file=None,
                        **configuration_values,
                    )

                    with self.assertRaises(VideoDemoError) as raised:
                        settings.require_cloud_asr_configuration()

                    self.assertEqual(
                        raised.exception.code,
                        ErrorCode.INVALID_CONFIGURATION,
                    )

    def test_cloud_asr_timeout_and_attempt_count_are_strictly_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            for values in (
                {"openai_asr_timeout_seconds": float("nan")},
                {"openai_asr_timeout_seconds": float("inf")},
                {"openai_asr_timeout_seconds": float("-inf")},
                {"openai_asr_timeout_seconds": 0.0},
                {"openai_asr_timeout_seconds": -1.0},
                {"openai_asr_max_attempts": 0},
                {"openai_asr_max_attempts": 6},
            ):
                with self.subTest(values=values), self.assertRaises(ValidationError):
                    Settings(workspace_root=workspace, _env_file=None, **values)

    def test_cloud_asr_secret_is_excluded_from_all_configuration_surfaces(self) -> None:
        secret = "cloud-asr-secret-value"
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            settings = Settings(
                workspace_root=workspace,
                openai_base_url="https://ai-proxy.example.test/v1",
                openai_api_key=secret,
                openai_model="openai/whisper",
                _env_file=None,
            )
            configuration = settings.require_cloud_asr_configuration()
            other_key_settings = Settings(
                workspace_root=workspace,
                openai_base_url="https://ai-proxy.example.test/v1",
                openai_api_key="different-cloud-asr-key",
                openai_model="openai/whisper",
                _env_file=None,
            )
            invalid_settings = Settings(
                workspace_root=workspace,
                openai_base_url=f"https://user:{secret}@ai-proxy.example.test/v1",
                openai_api_key=secret,
                openai_model="openai/whisper",
                _env_file=None,
            )

            with self.assertRaises(VideoDemoError) as raised:
                invalid_settings.require_cloud_asr_configuration()

            serialized_surfaces = (
                repr(settings),
                repr(settings.model_dump()),
                settings.model_dump_json(),
                repr(configuration),
                repr(configuration.model_dump()),
                configuration.model_dump_json(),
                repr(raised.exception),
                str(raised.exception),
                repr(raised.exception.details),
                build_production_model_identity_report(settings).model_dump_json(),
            )
            self.assertTrue(all(secret not in value for value in serialized_surfaces))
            self.assertEqual(
                build_production_model_identity_report(settings).settings_fingerprint,
                build_production_model_identity_report(
                    other_key_settings,
                ).settings_fingerprint,
            )

    def test_defaults_keep_runtime_inside_workspace_and_use_m1_safe_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)

            settings = Settings(workspace_root=workspace, _env_file=None)

            self.assertEqual(
                settings.runtime_root,
                workspace.resolve() / ".codex" / "video-rag-demo",
            )
            self.assertTrue(
                {
                    "inference_device",
                    "whisper_compute_type",
                    "whisper_model_id",
                    "speech_enrichment_timeout_seconds",
                    "huggingface_token",
                }.isdisjoint(Settings.model_fields)
            )
            self.assertEqual(settings.worker_concurrency, 1)
            self.assertEqual(settings.qwen_max_video_bytes, 64 * 1024 * 1024)
            self.assertEqual(settings.qwen_max_video_duration_ms, 30_000)
            self.assertEqual(settings.qwen_timeout_seconds, 300.0)
            self.assertEqual(settings.speech_subprocess_timeout_seconds, 3_600)
            self.assertEqual(settings.max_video_duration_ms, 1_800_000)
            self.assertEqual(settings.oss_prefix, "video-demo/qwen-clips")
            self.assertEqual(settings.oss_signed_url_ttl_seconds, 3_600)
            self.assertFalse(settings.has_complete_oss_configuration())
            self.assertFalse(settings.demo_degraded_mode)

    def test_video_duration_and_subprocess_timeout_hard_limits_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)

            settings = Settings(
                workspace_root=workspace,
                max_video_duration_ms=7_200_000,
                process_timeout_seconds=14_400,
                speech_subprocess_timeout_seconds=14_400,
                _env_file=None,
            )

            self.assertEqual(settings.max_video_duration_ms, 7_200_000)
            for values in (
                {"max_video_duration_ms": 7_200_001},
                {"process_timeout_seconds": 14_401},
                {"speech_subprocess_timeout_seconds": 14_401},
            ):
                with self.subTest(values=values), self.assertRaises(ValidationError):
                    Settings(workspace_root=workspace, _env_file=None, **values)  # type: ignore[arg-type]

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

    def test_cloud_timeout_and_retry_change_settings_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            common = {
                "workspace_root": workspace,
                "openai_base_url": "https://ai-proxy.example.test/v1",
                "openai_api_key": "test-openai-key",
                "openai_model": "openai/whisper",
                "_env_file": None,
            }
            baseline = build_production_model_identity_report(
                Settings(**common),
            )
            timeout_changed = build_production_model_identity_report(
                Settings(**common, openai_asr_timeout_seconds=301),
            )
            retry_changed = build_production_model_identity_report(
                Settings(**common, openai_asr_max_attempts=4),
            )
            key_changed = build_production_model_identity_report(
                Settings(**{**common, "openai_api_key": "different-test-key"}),
            )

            self.assertNotEqual(
                baseline.settings_fingerprint,
                timeout_changed.settings_fingerprint,
            )
            self.assertNotEqual(
                baseline.settings_fingerprint,
                retry_changed.settings_fingerprint,
            )
            self.assertEqual(
                baseline.settings_fingerprint,
                key_changed.settings_fingerprint,
            )

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
                openai_api_key="openai-secret-value",
            )

            serialized = settings.model_dump_json()

            self.assertNotIn(secret, serialized)
            self.assertNotIn("baidu-api-secret", serialized)
            self.assertNotIn("baidu-secret-value", serialized)
            self.assertNotIn("openai-secret-value", serialized)


if __name__ == "__main__":
    unittest.main()
