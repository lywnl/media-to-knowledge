"""持久化层共用的租户作用域值对象。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Scope:
    tenant_id: str
    application_id: str
    knowledge_base_id: str
