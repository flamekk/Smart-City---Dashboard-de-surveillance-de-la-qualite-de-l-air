from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.utils.helpers import max_status, utc_now_iso


@dataclass
class Threshold:
    warning: float
    critical: float


class AlertManager:
    """Rule-based alert engine with anti-false-positive logic."""

    def __init__(self, thresholds: Dict[str, Dict[str, float]], consecutive_breaches: int = 3) -> None:
        self.thresholds: Dict[str, Threshold] = {
            metric: Threshold(
                warning=float(values.get("warning", 0.0)),
                critical=float(values.get("critical", 0.0)),
            )
            for metric, values in thresholds.items()
        }
        self.consecutive_breaches = max(1, int(consecutive_breaches))
        self._breach_counts: Dict[str, int] = {metric: 0 for metric in self.thresholds}
        self._active_levels: Dict[str, str] = {metric: "normal" for metric in self.thresholds}

    def _metric_level(self, metric: str, value: Optional[float]) -> str:
        if value is None:
            return "normal"
        threshold = self.thresholds[metric]
        if value >= threshold.critical:
            return "critical"
        if value >= threshold.warning:
            return "warning"
        return "normal"

    def instant_status(self, measurements: Dict[str, Any]) -> str:
        levels = [self._metric_level(metric, measurements.get(metric)) for metric in self.thresholds]
        return max_status(levels)

    def active_status(self) -> str:
        return max_status(self._active_levels.values())

    def evaluate(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        measurements: Dict[str, Any] = payload.get("measurements", {})
        timestamp = payload.get("timestamp", utc_now_iso())
        alerts: List[Dict[str, Any]] = []

        for metric, threshold in self.thresholds.items():
            value_raw = measurements.get(metric)
            value = float(value_raw) if value_raw is not None else None
            level = self._metric_level(metric, value)
            previous_active = self._active_levels[metric]

            if level == "normal":
                self._breach_counts[metric] = 0
                if previous_active != "normal":
                    # Recovery message when metric goes back to normal range.
                    alerts.append(
                        {
                            "timestamp": timestamp,
                            "metric": metric,
                            "value": value,
                            "threshold": threshold.warning,
                            "level": "normal",
                            "message": f"Metric {metric} recovered to normal range.",
                        }
                    )
                self._active_levels[metric] = "normal"
                continue

            self._breach_counts[metric] += 1
            breach_count = self._breach_counts[metric]

            if breach_count < self.consecutive_breaches:
                continue

            # Trigger when a level becomes active the first time or escalates.
            should_trigger = previous_active == "normal" or (
                previous_active == "warning" and level == "critical"
            )

            self._active_levels[metric] = level
            if not should_trigger:
                continue

            threshold_value = threshold.critical if level == "critical" else threshold.warning
            alerts.append(
                {
                    "timestamp": timestamp,
                    "metric": metric,
                    "value": value,
                    "threshold": threshold_value,
                    "level": level,
                    "message": (
                        f"{level.upper()} alert for {metric}: value={value:.2f} exceeded "
                        f"threshold={threshold_value:.2f} for {breach_count} consecutive samples."
                    ),
                }
            )

        return alerts

