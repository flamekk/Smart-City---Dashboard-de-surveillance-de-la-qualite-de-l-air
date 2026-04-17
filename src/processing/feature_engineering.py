from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from src.utils.helpers import coerce_float

# (concentration_low, concentration_high, index_low, index_high)
PM25_BREAKPOINTS: List[Tuple[float, float, int, int]] = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 500.4, 301, 500),
]

PM10_BREAKPOINTS: List[Tuple[float, float, int, int]] = [
    (0.0, 54.0, 0, 50),
    (55.0, 154.0, 51, 100),
    (155.0, 254.0, 101, 150),
    (255.0, 354.0, 151, 200),
    (355.0, 424.0, 201, 300),
    (425.0, 604.0, 301, 500),
]


def _sub_index(concentration: Optional[float], breakpoints: List[Tuple[float, float, int, int]]) -> Optional[int]:
    if concentration is None:
        return None
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= concentration <= c_high:
            ratio = (concentration - c_low) / (c_high - c_low)
            return int(round(i_low + ratio * (i_high - i_low)))
    return 500 if concentration > breakpoints[-1][1] else None


def compute_simple_aqi(pm25: Optional[float], pm10: Optional[float]) -> Optional[int]:
    pm25_idx = _sub_index(pm25, PM25_BREAKPOINTS)
    pm10_idx = _sub_index(pm10, PM10_BREAKPOINTS)
    candidates = [idx for idx in [pm25_idx, pm10_idx] if idx is not None]
    return max(candidates) if candidates else None


def classify_aqi(aqi: Optional[int]) -> str:
    if aqi is None:
        return "unknown"
    if aqi <= 50:
        return "good"
    if aqi <= 100:
        return "moderate"
    if aqi <= 150:
        return "unhealthy_sensitive"
    if aqi <= 200:
        return "unhealthy"
    if aqi <= 300:
        return "very_unhealthy"
    return "hazardous"


def estimate_co2_equivalent(co2: Optional[float], co: Optional[float], no2: Optional[float]) -> Optional[float]:
    co2_val = coerce_float(co2)
    if co2_val is not None:
        return co2_val
    co_val = coerce_float(co) or 0.0
    no2_val = coerce_float(no2) or 0.0
    return round(420.0 + (co_val * 110.0) + (no2_val * 0.7), 2)


def estimate_tvoc(tvoc: Optional[float], co: Optional[float], no2: Optional[float], o3: Optional[float]) -> Optional[float]:
    tvoc_val = coerce_float(tvoc)
    if tvoc_val is not None:
        return tvoc_val
    co_val = coerce_float(co) or 0.0
    no2_val = coerce_float(no2) or 0.0
    o3_val = coerce_float(o3) or 0.0
    return round((co_val * 45.0) + (no2_val * 1.2) + (o3_val * 0.5), 2)


def compute_risk_score(sample: Dict[str, Optional[float]], aqi: Optional[int]) -> float:
    """Simple weighted risk score in [0, 100] for dashboard summarization."""
    pm25 = sample.get("pm25") or 0.0
    pm10 = sample.get("pm10") or 0.0
    co = sample.get("co") or 0.0
    no2 = sample.get("no2") or 0.0
    humidity = sample.get("humidity") or 0.0

    pollutant_component = min(
        100.0,
        (pm25 / 55.0) * 35.0 + (pm10 / 150.0) * 25.0 + (co / 9.0) * 15.0 + (no2 / 100.0) * 10.0,
    )
    humidity_component = 0.0 if 30 <= humidity <= 70 else min(10.0, abs(humidity - 50.0) / 5.0)
    aqi_component = min(30.0, (aqi or 0) / 500.0 * 30.0)

    score = pollutant_component + humidity_component + aqi_component
    return round(min(100.0, score), 2)


def status_from_thresholds(values: Iterable[str]) -> str:
    if "critical" in values:
        return "critical"
    if "warning" in values:
        return "warning"
    return "normal"

