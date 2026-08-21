"""SQLite 持久化实现。"""

from video_demo.persistence.database import Database
from video_demo.persistence.repositories import Scope

__all__ = ["Database", "Scope"]
