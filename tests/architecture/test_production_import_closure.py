from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from video_demo.implementation import implementation_import_closure

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_ENTRYPOINTS = (
    (Path("src/video_demo/main.py"), "video_demo.main"),
    (Path("src/video_demo/api/app.py"), "video_demo.api.app"),
    (Path("src/video_demo/application/pipeline.py"), "video_demo.application.pipeline"),
    (
        Path("src/video_demo/application/document_pipeline.py"),
        "video_demo.application.document_pipeline",
    ),
    (
        Path("src/video_demo/application/composition.py"),
        "video_demo.application.composition",
    ),
    (Path("src/video_demo/application/queries.py"), "video_demo.application.queries"),
    (
        Path("src/video_demo/persistence/repositories.py"),
        "video_demo.persistence.repositories",
    ),
    (
        Path("src/video_demo/application/video_scheduler.py"),
        "video_demo.application.video_scheduler",
    ),
    (Path("src/video_demo/worker/runtime.py"), "video_demo.worker.runtime"),
    (
        Path("src/video_demo/evaluation/prediction_runner.py"),
        "video_demo.evaluation.prediction_runner",
    ),
    (
        Path("src/video_demo/evaluation/predictions.py"),
        "video_demo.evaluation.predictions",
    ),
    (
        Path("src/video_demo/evaluation/quality_runner.py"),
        "video_demo.evaluation.quality_runner",
    ),
)
_FORBIDDEN_FILES = frozenset(
    {
        Path("src/video_demo/domain/legacy_result.py"),
        Path("src/video_demo/domain/legacy_result_artifact.py"),
        Path("src/video_demo/persistence/legacy_result_repository.py"),
        Path("src/video_demo/application/legacy_composition.py"),
        Path("src/video_demo/application/production_visual.py"),
        Path("src/video_demo/fusion/merge.py"),
        Path("src/video_demo/fusion/result_builder.py"),
        Path("src/video_demo/fusion/retrieval_text.py"),
    }
)
_FORBIDDEN_MODULES = frozenset(
    path.with_suffix("").relative_to("src").as_posix().replace("/", ".")
    for path in _FORBIDDEN_FILES
)


@pytest.mark.parametrize(("entry_file", "_module_name"), _ENTRYPOINTS)
def test_production_static_import_closure_excludes_legacy_chain(
    entry_file: Path,
    _module_name: str,
) -> None:
    closure = set(
        implementation_import_closure(
            _WORKSPACE_ROOT,
            (entry_file,),
            extra_files=(),
        )
    )

    if entry_file != Path("src/video_demo/application/video_scheduler.py"):
        assert Path("src/video_demo/domain/__init__.py") in closure
    assert closure.isdisjoint(_FORBIDDEN_FILES)


@pytest.mark.parametrize(("_entry_file", "module_name"), _ENTRYPOINTS)
def test_production_fresh_process_import_excludes_legacy_modules(
    _entry_file: Path,
    module_name: str,
) -> None:
    script = """
import importlib
import json
import sys

importlib.import_module(sys.argv[1])
forbidden = set(json.loads(sys.argv[2]))
print(json.dumps(sorted(forbidden.intersection(sys.modules))))
"""
    temporary_parent = _WORKSPACE_ROOT / ".codex"
    temporary_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="pytest-production-import-",
        dir=temporary_parent,
    ) as temporary_root:
        runtime_root = Path(temporary_root) / "runtime"
        runtime_root.mkdir()
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(_WORKSPACE_ROOT / "src"),
                "VIDEO_DEMO_WORKSPACE_ROOT": str(_WORKSPACE_ROOT),
                "VIDEO_DEMO_RUNTIME_ROOT": str(runtime_root),
            }
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                module_name,
                json.dumps(sorted(_FORBIDDEN_MODULES)),
            ],
            cwd=_WORKSPACE_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []
