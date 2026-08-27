from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_RETIRED_FILES = (
    "src/video_demo/application/adaptive_ocr.py",
    "src/video_demo/application/production_visual.py",
    "src/video_demo/application/legacy_composition.py",
    "src/video_demo/integrations/baidu_ocr.py",
    "src/video_demo/integrations/oss.py",
    "src/video_demo/integrations/qwen.py",
    "src/video_demo/integrations/prompts.py",
    "src/video_demo/integrations/video_port.py",
    "src/video_demo/domain/legacy_result.py",
    "src/video_demo/domain/legacy_result_artifact.py",
    "src/video_demo/fusion/merge.py",
    "src/video_demo/fusion/result_builder.py",
    "src/video_demo/fusion/retrieval_text.py",
    "src/video_demo/fusion/timeline.py",
    "src/video_demo/visual/ocr.py",
    "src/video_demo/visual/ocr_budget.py",
)
_RETIRED_MODULES = tuple(path.removesuffix(".py").replace("/", ".") for path in _RETIRED_FILES)
_RETIRED_SYMBOLS = (
    "baidu_ocr_live",
    "qwen_live",
    "OcrEvidence",
    "OcrLine",
    "LegacyEvidenceItem",
    "TimelineEvidence",
    "WholeVideoUnderstanding",
    "understand_video",
    "data:video",
    "video_url",
    "VIDEO_DEMO_OSS_",
    "VIDEO_DEMO_BAIDU_",
    "VIDEO_DEMO_QWEN_",
    "qwen_base_url",
    "qwen_model_id",
    "qwen_api_key",
    "baidu_api_key",
    "baidu_secret_key",
    "oss_endpoint",
    "oss_bucket",
)


def test_retired_visual_chain_is_physically_removed() -> None:
    assert all(not (_ROOT / relative).exists() for relative in _RETIRED_FILES)


def test_production_source_has_no_retired_visual_imports_or_symbols() -> None:
    violations: list[str] = []
    for path in sorted((_ROOT / "src/video_demo").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        relative = path.relative_to(_ROOT).as_posix()
        if any(symbol in source for symbol in _RETIRED_SYMBOLS):
            violations.append(f"{relative}:retired-symbol")
        tree = ast.parse(source, filename=relative)
        for node in ast.walk(tree):
            modules: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = (node.module, *(f"{node.module}.{alias.name}" for alias in node.names))
            for module in modules:
                if any(
                    module == prefix or module.startswith(prefix + ".")
                    for prefix in _RETIRED_MODULES
                ):
                    violations.append(f"{relative}:{node.lineno}:{module}")
    assert not violations, "\n".join(violations)
