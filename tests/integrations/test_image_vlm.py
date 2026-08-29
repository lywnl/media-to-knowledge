import json

import httpx

from video_demo.integrations.image_vlm import ImageVlmClient


def test_image_vlm_accepts_json_wrapped_model_message() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "title": "架构图",
                            "overview_zh": "展示系统结构",
                            "content_blocks": [
                                {
                                    "content_type": "DIAGRAM",
                                    "text": "包含三个模块",
                                    "evidence_refs": ["image_source_001"],
                                },
                            ],
                            "claims": [],
                            "evidence_refs": ["image_source_001"],
                        },
                        ensure_ascii=False,
                    ),
                },
            },
        ],
    }
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=body))
    with httpx.Client(transport=transport) as client:
        result = ImageVlmClient(
            client,
            base_url="https://vlm.example.test/v1",
            api_key="secret",
            model_id="qwen3-vl-flash",
        ).analyze(image_data_url="data:image/png;base64,AAAA", title_hint="图")
    assert result.title == "架构图"
