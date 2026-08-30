from video_demo.application.image_rendering import render_image_markdown
from video_demo.domain.image_document import (
    ImageClaim,
    ImageContentBlock,
    ImageDocument,
    ImageSourceEvidence,
    ImageUnderstandingResult,
)


def test_description_block_keeps_content_without_redundant_heading() -> None:
    result = ImageUnderstandingResult(
        run_id="run_image_rendering",
        asset_sha256="a" * 64,
        source=ImageSourceEvidence(
            evidence_id="image_source",
            relative_path="runs/scope/run/input/source.png",
            mime_type="image/png",
            sha256="a" * 64,
            width=100,
            height=100,
            size_bytes=100,
        ),
        document=ImageDocument(
            title="图片标题",
            overview_zh="图片概览",
            content_blocks=(
                ImageContentBlock(
                    content_type="DESCRIPTION",
                    text="图片内容正文保留",
                    evidence_refs=("image_source",),
                ),
            ),
            claims=(
                ImageClaim(
                    text="关键结论",
                    evidence_refs=("image_source",),
                    certainty=0.9,
                ),
            ),
            evidence_refs=("image_source",),
        ),
    )

    markdown = render_image_markdown(result).content.decode("utf-8")

    assert "### DESCRIPTION" not in markdown
    assert "图片内容正文保留" in markdown
    assert "## 图片内容" in markdown
    assert "## 关键结论" in markdown
