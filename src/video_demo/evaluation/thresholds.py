from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class MetricThreshold:
    limit: float
    direction: Literal["max", "min"]


SEMANTIC_BOUNDARY_TOLERANCE_MS = 2_000


QUALITY_THRESHOLDS: dict[str, MetricThreshold] = {
    "zh_cer": MetricThreshold(0.15, "max"),
    "ja_cer": MetricThreshold(0.20, "max"),
    "ko_cer": MetricThreshold(0.20, "max"),
    "en_wer": MetricThreshold(0.18, "max"),
    "es_wer": MetricThreshold(0.18, "max"),
    "visual_text_accuracy": MetricThreshold(0.90, "min"),
    "visual_key_field_recall": MetricThreshold(0.90, "min"),
    "semantic_boundary_f1": MetricThreshold(0.75, "min"),
    "fact_support_rate": MetricThreshold(0.95, "min"),
    "key_fact_recall": MetricThreshold(0.85, "min"),
    "unknown_evidence_count": MetricThreshold(0, "max"),
    "fabricated_name_count": MetricThreshold(0, "max"),
    "schema_time_valid_rate": MetricThreshold(1.0, "min"),
    "rtf": MetricThreshold(3.0, "max"),
}
