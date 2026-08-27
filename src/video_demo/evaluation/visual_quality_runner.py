"""代表性视觉质量评测的真实执行与幂等持久化入口。"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from video_demo.application.composition import (
    build_production_model_identity_report,
    production_tool_path,
)
from video_demo.config import Settings
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.evaluation.annotations import (
    ValidatedEvaluationPackage,
    reverify_evaluation_package,
)
from video_demo.evaluation.chapter_vlm_input import (
    evaluation_run_id_for_input,
    prepare_chapter_vlm_input,
)
from video_demo.evaluation.chapter_vlm_live import (
    build_visual_text_score_fact,
    execute_chapter_vlm_live,
    has_selected_frame_selection,
    has_visual_text_projection,
)
from video_demo.evaluation.gate import _current_live_implementation_sha256
from video_demo.evaluation.visual_quality import (
    VisualQualityCase,
    VisualQualityReport,
    VisualQualitySet,
    build_visual_quality_report,
    build_visual_quality_set,
    verify_visual_quality_report,
    visual_quality_case_id,
)
from video_demo.integrations.qwen_vl import QwenVisionCallFailure, QwenVisionClient


class VisualQualityRunner:
    """执行代表性视觉 case，并只持久化可重验的脱敏事实。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build_quality_set(
        self,
        package: ValidatedEvaluationPackage,
        *,
        evaluation_run_id: str,
        proxy_max_edge: int = 1_920,
    ) -> VisualQualitySet:
        return build_visual_quality_set(
            reverify_evaluation_package(package),
            parent_evaluation_run_id=evaluation_run_id,
            proxy_max_edge=proxy_max_edge,
            jpeg_quality=self._settings.keyframe_jpeg_quality,
        )

    def run(
        self,
        package: ValidatedEvaluationPackage,
        *,
        evaluation_run_id: str,
        proxy_max_edge: int = 1_920,
    ) -> VisualQualityReport:
        verified_package = reverify_evaluation_package(package)
        quality_set = self.build_quality_set(
            verified_package, evaluation_run_id=evaluation_run_id, proxy_max_edge=proxy_max_edge
        )
        existing = self._read_existing_report(evaluation_run_id, proxy_max_edge)
        if existing is not None:
            verify_visual_quality_report(existing, quality_set, verified_package)
            return existing
        if quality_set.status == "NOT_RUN":
            report = self.report_not_run(quality_set)
            self.write_report(
                report,
                evaluation_run_id=evaluation_run_id,
                filename=(
                    "visual-quality.json"
                    if proxy_max_edge == 1_920
                    else f"visual-quality-{proxy_max_edge}.json"
                ),
            )
            return report
        assert self._settings.runtime_root is not None
        vision = self._settings.require_vlm_configuration()
        http_client = httpx.Client()
        try:
            client = QwenVisionClient(
                http_client,
                base_url=vision.base_url,
                api_key=vision.api_key.get_secret_value(),
                model_id=vision.model_id,
                runtime_root=self._settings.runtime_root,
                timeout_seconds=vision.timeout_seconds,
                max_attempts=vision.max_attempts,
                max_image_bytes=vision.max_image_bytes,
                max_request_image_bytes=vision.max_request_image_bytes,
                max_encoded_request_bytes=vision.max_encoded_request_bytes,
                max_response_bytes=self._settings.model_max_response_bytes,
            )
            cases = tuple(
                self._run_case(
                    verified_package,
                    quality_set,
                    sample.sample_id,
                    sample.requested_reference_frame_ids,
                    client,
                )
                for sample in quality_set.samples
            )
        finally:
            http_client.close()
        report = build_visual_quality_report(quality_set, verified_package, cases)
        self.write_report(
            report,
            evaluation_run_id=evaluation_run_id,
            filename=(
                "visual-quality.json"
                if proxy_max_edge == 1_920
                else f"visual-quality-{proxy_max_edge}.json"
            ),
        )
        return report

    def _run_case(
        self,
        package: ValidatedEvaluationPackage,
        quality_set: VisualQualitySet,
        sample_id: str,
        requested_frame_ids: tuple[str, ...],
        client: QwenVisionClient,
    ) -> VisualQualityCase:
        verified = next(
            item for item in package.annotations if item.annotation.sample_id == sample_id
        )
        sample = next(item for item in package.dataset.samples if item.sample_id == sample_id)
        product_run_id = evaluation_run_id_for_input(
            quality_set.parent_evaluation_run_id,
            sample_id,
            sample.media_sha256,
            verified.sha256,
            quality_set.proxy_max_edge,
            quality_set.jpeg_quality,
            requested_frame_ids,
        )
        identity = build_production_model_identity_report(self._settings)
        implementation = _current_live_implementation_sha256(self._settings.workspace_root)
        model = next(item for item in identity.models if item.component == "chapter_vlm")
        base: dict[str, Any] = {
            "case_id": visual_quality_case_id(
                quality_set.parent_evaluation_run_id,
                sample_id,
                requested_frame_ids,
                quality_set.proxy_max_edge,
                quality_set.jpeg_quality,
            ),
            "parent_evaluation_run_id": quality_set.parent_evaluation_run_id,
            "sample_id": sample_id,
            "requested_reference_frame_ids": requested_frame_ids,
            "proxy_max_edge": quality_set.proxy_max_edge,
            "jpeg_quality": quality_set.jpeg_quality,
            "implementation_sha256": implementation,
            "settings_fingerprint": identity.settings_fingerprint,
            "resolution_settings_fingerprint": identity.settings_fingerprint,
        }
        started = time.monotonic_ns()
        assert self._settings.runtime_root is not None
        try:
            preparation = prepare_chapter_vlm_input(
                package,
                parent_evaluation_run_id=quality_set.parent_evaluation_run_id,
                proxy_max_edge=quality_set.proxy_max_edge,
                jpeg_quality=quality_set.jpeg_quality,
                max_video_bytes=self._settings.max_video_bytes,
                vlm_max_image_bytes=self._settings.vlm_max_image_bytes,
                max_candidate_frame_bytes_per_run=self._settings.max_candidate_frame_bytes_per_run,
                max_candidate_frame_files_per_run=self._settings.max_candidate_frame_files_per_run,
                ffprobe=self._ffprobe(),
                transcoder=self._transcoder(),
                frame_extractor=self._extractor(),
                runtime_root=self._settings.runtime_root,
                sample_id=sample_id,
                requested_reference_frame_ids=requested_frame_ids,
            )
        except (OSError, TypeError, ValueError, VideoDemoError):
            preparation = None
        elapsed_ms = max(0, (time.monotonic_ns() - started) // 1_000_000)
        if preparation is None or preparation.status != "READY" or preparation.manifest is None:
            code = (
                preparation.error_code
                if preparation is not None and preparation.error_code is not None
                else ErrorCode.ARTIFACT_SCHEMA_INVALID
            )
            if preparation is not None and not preparation.execution_started:
                return VisualQualityCase(**base, case_status="NOT_RUN")
            return VisualQualityCase(
                **base,
                evaluation_run_id=product_run_id,
                case_status="FAIL",
                error_code=code,
                model=model,
                proxy_elapsed_ms=elapsed_ms,
            )
        manifest = preparation.manifest
        fields = self._manifest_fields(manifest, elapsed_ms)
        try:
            if preparation.context is None:
                raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "准备结果缺少已验证上下文")
            response, receipt = execute_chapter_vlm_live(
                manifest,
                context=preparation.context,
                expected_parent_evaluation_run_id=quality_set.parent_evaluation_run_id,
                expected_evaluation_run_id=product_run_id,
                vision_client=client,
            )
            response_sha = receipt.response_sha256
            failure_code: ErrorCode | None = None
            score = None
            if not has_selected_frame_selection(response, manifest):
                failure_code = ErrorCode.VISUAL_RESULT_INVALID
            else:
                score = build_visual_text_score_fact(
                    manifest, verified, response, response_sha256=response_sha
                )
                if not has_visual_text_projection(response, manifest):
                    failure_code = ErrorCode.VISUAL_RESULT_INVALID
            fields.update(
                request_json_bytes=receipt.request_json_bytes,
                encoded_request_bytes=receipt.encoded_request_bytes,
                vlm_elapsed_ms=receipt.vlm_elapsed_ms,
            )
            return VisualQualityCase(
                **base,
                evaluation_run_id=product_run_id,
                case_status="READY" if failure_code is None else "FAIL",
                error_code=failure_code,
                manifest_sha256=preparation.manifest_sha256,
                call_receipt=receipt,
                response_sha256=response_sha,
                score_fact=score,
                model=model,
                **fields,
            )
        except QwenVisionCallFailure as error:
            return VisualQualityCase(
                **base,
                evaluation_run_id=product_run_id,
                case_status="FAIL",
                error_code=error.code,
                failure_receipt=error.provider,
                model=model,
                manifest_sha256=preparation.manifest_sha256,
                **fields,
            )
        except (OSError, TypeError, ValueError, VideoDemoError) as error:
            code = (
                error.code
                if isinstance(error, VideoDemoError)
                else ErrorCode.ARTIFACT_SCHEMA_INVALID
            )
            return VisualQualityCase(
                **base,
                evaluation_run_id=product_run_id,
                case_status="FAIL",
                error_code=code,
                model=model,
                manifest_sha256=preparation.manifest_sha256,
                **fields,
            )

    def _manifest_fields(self, manifest: Any, elapsed_ms: int) -> dict[str, Any]:
        return {
            "proxy_width": manifest.proxy_width,
            "proxy_height": manifest.proxy_height,
            "proxy_frame_rate": manifest.proxy_frame_rate,
            "proxy_size_bytes": manifest.proxy_size_bytes,
            "proxy_elapsed_ms": elapsed_ms,
        }

    def _ffprobe(self) -> Any:
        from video_demo.media.probe import FFprobeClient

        assert self._settings.runtime_root is not None
        return FFprobeClient.from_path(
            production_tool_path(self._settings, "ffprobe"),
            workspace_root=self._settings.workspace_root,
        )

    def _transcoder(self) -> Any:
        from video_demo.media.transcode import FFmpegTranscoder

        assert self._settings.runtime_root is not None
        return FFmpegTranscoder.from_path(
            production_tool_path(self._settings, "ffmpeg"),
            self._settings.runtime_root,
            workspace_root=self._settings.workspace_root,
        )

    def _extractor(self) -> Any:
        from video_demo.visual.keyframes import OpenCvFrameExtractor

        assert self._settings.runtime_root is not None
        return OpenCvFrameExtractor(
            self._settings.runtime_root,
            max_frame_bytes=self._settings.vlm_max_image_bytes,
            jpeg_quality=self._settings.keyframe_jpeg_quality,
        )

    def report_not_run(
        self, quality_set: VisualQualitySet, *, reason: str | None = None
    ) -> VisualQualityReport:
        if reason and quality_set.status != "NOT_RUN":
            quality_set = quality_set.model_copy(
                update={"status": "NOT_RUN", "not_run_reason": reason}
            )
        cases = tuple(
            VisualQualityCase(
                case_id=visual_quality_case_id(
                    quality_set.parent_evaluation_run_id,
                    sample.sample_id,
                    sample.requested_reference_frame_ids,
                    quality_set.proxy_max_edge,
                    quality_set.jpeg_quality,
                ),
                parent_evaluation_run_id=quality_set.parent_evaluation_run_id,
                sample_id=sample.sample_id,
                requested_reference_frame_ids=sample.requested_reference_frame_ids,
                proxy_max_edge=quality_set.proxy_max_edge,
                jpeg_quality=quality_set.jpeg_quality,
                case_status="NOT_RUN",
                implementation_sha256="0" * 64,
                settings_fingerprint="0" * 64,
            )
            for sample in quality_set.samples
        )
        return build_visual_quality_report(quality_set, object(), cases)

    def write_report(
        self,
        report: Any,
        *,
        evaluation_run_id: str,
        filename: str = "visual-quality.json",
    ) -> Any:
        assert self._settings.runtime_root is not None
        path = self._settings.runtime_root / "eval" / "reports" / evaluation_run_id / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2
            ),
            encoding="utf-8",
        )
        return path

    def _read_existing_report(
        self, evaluation_run_id: str, proxy_max_edge: int
    ) -> VisualQualityReport | None:
        assert self._settings.runtime_root is not None
        filename = (
            "visual-quality.json"
            if proxy_max_edge == 1_920
            else f"visual-quality-{proxy_max_edge}.json"
        )
        path = self._settings.runtime_root / "eval" / "reports" / evaluation_run_id / filename
        if not path.is_file():
            return None
        return VisualQualityReport.model_validate_json(path.read_bytes())


__all__ = ["VisualQualityRunner"]
