"""生产音频产物的统一 MP3 编码契约。"""

from __future__ import annotations

from typing import Final

AUDIO_FORMAT_VERSION: Final = "mp3-192k-v1"
AUDIO_OUTPUT_EXTENSION: Final = ".mp3"
AUDIO_OUTPUT_MIME: Final = "audio/mpeg"
AUDIO_CODEC: Final = "mp3"
AUDIO_ENCODER: Final = "libmp3lame"
AUDIO_SAMPLE_RATE_HZ: Final = 16_000
AUDIO_CHANNELS: Final = 1
AUDIO_BITRATE_BPS: Final = 192_000
AUDIO_BITRATE: Final = "192k"
_MP3_CONTAINER_OVERHEAD_BYTES: Final = 64 * 1024


def estimate_audio_output_bytes(duration_ms: int) -> int:
    """估算 MP3 输出容量，并为容器头和帧边界保留安全余量。"""

    if type(duration_ms) is not int or duration_ms < 1:
        raise ValueError("音频时长必须是正整数毫秒")
    encoded_bytes = (duration_ms * (AUDIO_BITRATE_BPS // 8) + 999) // 1_000
    return encoded_bytes + _MP3_CONTAINER_OVERHEAD_BYTES
