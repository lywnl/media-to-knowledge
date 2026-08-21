from __future__ import annotations

from pathlib import Path

from video_demo.evaluation.no_indexing import audit_no_indexing_capability


def test_source_and_run_stages_do_not_add_embedding_milvus_or_bm25() -> None:
    project_root = Path(__file__).parents[2]

    assert audit_no_indexing_capability(project_root) == ()
