from video_demo.speech.audio_fixed_asr import (
    AUDIO_ASR_CHUNK_DURATION_MS,
    AUDIO_ASR_CONCURRENCY,
    build_fixed_audio_asr_windows,
)


def test_audio_asr_uses_fixed_ten_minute_windows() -> None:
    windows = build_fixed_audio_asr_windows(1_200_000)

    assert AUDIO_ASR_CHUNK_DURATION_MS == 600_000
    assert AUDIO_ASR_CONCURRENCY == 1
    assert [(item.chunk_index, item.upload_range.start_ms, item.upload_range.end_ms)
            for item in windows] == [(0, 0, 600_000), (1, 600_000, 1_200_000)]


def test_audio_asr_last_window_is_clamped_to_duration() -> None:
    windows = build_fixed_audio_asr_windows(600_001)

    assert windows[-1].upload_range.end_ms == 600_001
    assert windows[-1].owned_range == windows[-1].upload_range
