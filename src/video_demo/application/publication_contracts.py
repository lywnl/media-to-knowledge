"""媒体结果发布共用的中性租约与作用域契约。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from video_demo.persistence.scope import Scope


@dataclass(frozen=True, slots=True)
class ResultWriteFence:
    """限制结果写入只能由当前任务租约提交。"""

    job_pk: int
    worker_id: str
    attempt_count: int


def scope_key(scope: Scope) -> str:
    """将租户作用域映射为稳定、不可逆的运行目录名。"""

    encoded = "\x00".join(
        (scope.tenant_id, scope.application_id, scope.knowledge_base_id),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]
