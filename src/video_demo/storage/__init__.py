"""工作区内隔离存储。"""

from video_demo.storage.artifacts import AtomicArtifactStore
from video_demo.storage.object_store import LocalVideoObjectStore

__all__ = ["AtomicArtifactStore", "LocalVideoObjectStore"]
