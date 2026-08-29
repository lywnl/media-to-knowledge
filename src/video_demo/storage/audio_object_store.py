from __future__ import annotations

from pathlib import Path

from video_demo.storage.media_object_store import BinaryMediaObjectStore

_AUDIO_EXTENSIONS = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
}


def detect_audio_mime(header: bytes) -> str | None:
    if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
        return "audio/wav"
    if header.startswith(b"ID3") or header.startswith(b"\xff\xfb"):
        return "audio/mpeg"
    if header.startswith(b"fLaC"):
        return "audio/flac"
    if header.startswith(b"OggS"):
        return "audio/ogg"
    if len(header) >= 12 and header[4:8] == b"ftyp":
        return "audio/mp4"
    return None


class AudioObjectStore(BinaryMediaObjectStore):
    def __init__(self, runtime_root: Path, *, max_bytes: int) -> None:
        super().__init__(
            runtime_root,
            max_bytes=max_bytes,
            media_kind="AUDIO_OBJECT",
            allowed_mimes=_AUDIO_EXTENSIONS.values(),
            extension_to_mime=_AUDIO_EXTENSIONS,
            detector=detect_audio_mime,
            error_prefix="AUDIO",
        )
