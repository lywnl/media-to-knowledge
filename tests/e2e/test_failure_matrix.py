from __future__ import annotations

import ast
from pathlib import Path

from video_demo.evaluation.gate import FAILURE_SCENARIO_TESTS, REQUIRED_FAILURE_SCENARIOS


def test_failure_matrix_contains_all_design_scenarios() -> None:
    assert set(REQUIRED_FAILURE_SCENARIOS) == {
        "corrupted_media",
        "spoofed_mime",
        "vfr",
        "rotation",
        "no_audio",
        "no_speech",
        "black_frames",
        "five_language_switch",
        "malformed_json",
        "cancellation",
        "retry",
        "restart_resume",
        "disk_insufficient",
        "cross_tenant",
        "redaction",
        "prompt_injection",
    }


def test_each_failure_scenario_points_to_existing_test_functions() -> None:
    assert set(FAILURE_SCENARIO_TESTS) == set(REQUIRED_FAILURE_SCENARIOS)
    project_root = Path(__file__).parents[2]

    for scenario, node_ids in FAILURE_SCENARIO_TESTS.items():
        assert node_ids, scenario
        for node_id in node_ids:
            relative_path, collected_name = node_id.split("::", maxsplit=1)
            function_name = collected_name.split("[", maxsplit=1)[0]
            path = project_root / relative_path
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            functions = {
                node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            assert function_name in functions, f"{scenario}: {node_id}"
