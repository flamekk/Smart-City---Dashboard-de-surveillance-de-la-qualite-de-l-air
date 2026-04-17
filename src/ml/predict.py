from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from src.processing.feature_engineering import classify_aqi, compute_simple_aqi
from src.utils.config import get_nested, load_settings, resolve_path


class MLPredictor:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = resolve_path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        payload = joblib.load(self.model_path)
        self.pipeline = payload["pipeline"]
        self.feature_columns = payload["feature_columns"]
        self.target_column = payload["target_column"]

    def predict_pm25(self, measurements: Dict[str, Any]) -> Optional[float]:
        row = {feature: measurements.get(feature) for feature in self.feature_columns}
        df = pd.DataFrame([row], columns=self.feature_columns)
        prediction = self.pipeline.predict(df)[0]
        return float(round(prediction, 3))

    def predict_enriched(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        predicted_pm25 = self.predict_pm25(measurements)
        predicted_aqi = compute_simple_aqi(predicted_pm25, measurements.get("pm10"))
        return {
            "predicted_pm25": predicted_pm25,
            "predicted_aqi": predicted_aqi,
            "predicted_category": classify_aqi(predicted_aqi),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference using trained optional ML model.")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings YAML.")
    parser.add_argument("--model-path", default=None, help="Model path. Defaults to ml.model_path.")
    parser.add_argument(
        "--sample-json",
        default="",
        help="JSON string containing measurement fields (pm10,no2,so2,co,o3,temperature,humidity,wind_speed).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    model_path = args.model_path or get_nested(settings, "ml", "model_path", default="models/aqi_pm25_model.joblib")
    predictor = MLPredictor(model_path)

    if args.sample_json:
        sample = json.loads(args.sample_json)
    else:
        sample = {
            "pm10": 120.0,
            "no2": 45.0,
            "so2": 11.0,
            "co": 1.4,
            "o3": 39.0,
            "temperature": 29.0,
            "humidity": 71.0,
            "wind_speed": 3.0,
        }

    prediction = predictor.predict_enriched(sample)
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()

