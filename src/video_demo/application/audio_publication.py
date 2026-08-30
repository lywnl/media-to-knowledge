"""音频结果的独立发布服务入口。"""

from __future__ import annotations

from video_demo.application.media_publication import MediaPublicationService


class AudioPublicationService(MediaPublicationService):
    """固定绑定音频结果类型的发布服务，避免调用方误配为图片。"""
