from pathlib import Path

_ROOT = Path(__file__).parents[2]


def test_frontend_renders_text_without_loading_keyframe_images() -> None:
    app_source = (_ROOT / "src/video_demo/web/app.js").read_text(encoding="utf-8")
    styles_source = (_ROOT / "src/video_demo/web/styles.css").read_text(encoding="utf-8")
    index_source = (_ROOT / "src/video_demo/web/index.html").read_text(encoding="utf-8")

    assert "fetchEvidence" not in app_source
    assert "/keyframes/" not in app_source
    assert "renderKeyframeFigure" not in app_source
    assert ".chapter-keyframe" not in styles_source
    assert ".retrieval-text" not in styles_source
    assert index_source.count('id="history-panel"') == 1
    assert index_source.count('id="history-list"') == 1
