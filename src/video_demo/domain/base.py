from __future__ import annotations

import hashlib
import json
import math
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
StableId = Annotated[str, Field(min_length=3, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")]
LanguageCode = Annotated[str, Field(min_length=2, max_length=16, pattern=r"^[a-z]{2,3}$|^und$")]


class FrozenModel(BaseModel):
    """拒绝未知字段且不可变的公共契约基类。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
        str_strip_whitespace=True,
    )


class UniqueStringTuplesMixin(BaseModel):
    """为结构化结果中的集合型 tuple 提供统一去重校验。"""

    @field_validator(
        "speakers",
        "languages",
        "topics",
        "entities",
        "actions",
        "keywords",
        "original_keywords",
        "evidence_refs",
        "segment_ids",
        check_fields=False,
    )
    @classmethod
    def reject_duplicates(cls, value: tuple[str, ...], info: Any) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} 不得重复")
        return value


def stable_identifier(prefix: str, payload: dict[str, object]) -> str:
    """使用规范 JSON 生成与运行顺序无关的稳定标识。"""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:24]}"


def require_finite(value: float, field_name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} 必须是有限数值")
    return value
