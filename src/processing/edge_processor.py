from __future__ import annotations

from typing import Any, Dict, Optional

from src.ml.predict import MLPredictor
from src.processing.feature_engineering import (
    classify_aqi,
    compute_risk_score,
    compute_simple_aqi,
    estimate_co2_equivalent,
    estimate_tvoc,
)
from src.processing.preprocess import RowPreprocessor, build_preprocessor_from_settings
from src.utils.config import get_nested
from src.utils.helpers import utc_now_iso


class EdgeProcessor:
    """Simulated edge-computing stage for each sensor message."""

    def __init__(self, settings: Dict[str, Any], ml_predictor: Optional[MLPredictor] = None) -> None:
        self.settings = settings
        self.sensor_id = get_nested(settings, "project", "sensor_id", default="SIM-AQ-01")
        self.preprocessor: RowPreprocessor = build_preprocessor_from_settings(settings)
        self.ml_predictor = ml_predictor

    def _base_status(self, aqi_value: Optional[int]) -> str:
        if aqi_value is None:
            return "normal"
        if aqi_value >= 151:
            return "critical"
        if aqi_value >= 101:
            return "warning"
        return "normal"

    def process(self, raw_row: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = self.preprocessor.transform(raw_row)
        aqi_estimated = compute_simple_aqi(cleaned.get("pm25"), cleaned.get("pm10"))
        aqi_category = classify_aqi(aqi_estimated)
        co2_equivalent = estimate_co2_equivalent(cleaned.get("co2"), cleaned.get("co"), cleaned.get("no2"))
        tvoc_estimated = estimate_tvoc(cleaned.get("tvoc"), cleaned.get("co"), cleaned.get("no2"), cleaned.get("o3"))
        risk_score = compute_risk_score(cleaned, aqi_estimated)

        payload: Dict[str, Any] = {
            "timestamp": utc_now_iso(),
            "sensor_id": self.sensor_id,
            "measurements": {
                "pm25": cleaned.get("pm25"),
                "pm10": cleaned.get("pm10"),
                "co": cleaned.get("co"),
                "no2": cleaned.get("no2"),
                "so2": cleaned.get("so2"),
                "o3": cleaned.get("o3"),
                "temperature": cleaned.get("temperature"),
                "humidity": cleaned.get("humidity"),
                "wind_speed": cleaned.get("wind_speed"),
                "co2_equivalent": co2_equivalent,
                "tvoc_estimated": tvoc_estimated,
            },
            "aqi_estimated": aqi_estimated,
            "aqi_category": aqi_category,
            "risk_score": risk_score,
            "status": self._base_status(aqi_estimated),
            "data_quality": {
                "source_date": cleaned.get("source_date"),
                "co2_available": cleaned.get("co2") is not None,
                "tvoc_available": cleaned.get("tvoc") is not None,
            },
        }

        if self.ml_predictor is not None:
            ml_pm25 = self.ml_predictor.predict_pm25(payload["measurements"])
            ml_aqi = compute_simple_aqi(ml_pm25, cleaned.get("pm10"))
            payload["ml_prediction"] = {
                "predicted_pm25": ml_pm25,
                "predicted_aqi": ml_aqi,
                "predicted_category": classify_aqi(ml_aqi),
            }
        return payload

