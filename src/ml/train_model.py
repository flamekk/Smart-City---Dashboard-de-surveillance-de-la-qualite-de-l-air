from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.processing.preprocess import build_preprocessor_from_settings, load_raw_dataframe, preprocess_dataframe
from src.utils.config import get_nested, load_settings, resolve_path
from src.utils.helpers import utc_now_iso


def train_model(
    settings: Dict[str, Any],
    input_csv: str | Path,
    model_path: str | Path,
    metrics_path: str | Path,
) -> Dict[str, float]:
    preprocessor = build_preprocessor_from_settings(settings)
    raw_df = load_raw_dataframe(input_csv)
    df = preprocess_dataframe(raw_df, preprocessor)

    feature_columns: List[str] = get_nested(settings, "ml", "feature_columns", default=[])
    target_column: str = get_nested(settings, "ml", "target_column", default="pm25")

    for col in feature_columns + [target_column]:
        if col not in df.columns:
            df[col] = pd.NA

    train_df = df[feature_columns + [target_column]].dropna(subset=[target_column]).copy()
    X = train_df[feature_columns]
    y = train_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=1)),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
        "n_rows": int(len(train_df)),
    }

    model_payload = {
        "trained_at": utc_now_iso(),
        "target_column": target_column,
        "feature_columns": feature_columns,
        "pipeline": pipeline,
    }

    model_file = resolve_path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_payload, model_file)

    metrics_file = resolve_path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train optional ML model for simulated air-quality pipeline.")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to YAML settings file.")
    parser.add_argument("--input-csv", default=None, help="Training CSV path. Defaults to data.source_csv.")
    parser.add_argument("--model-path", default=None, help="Output model path. Defaults to ml.model_path.")
    parser.add_argument("--metrics-path", default=None, help="Output metrics JSON path. Defaults to ml.metrics_path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    input_csv = args.input_csv or get_nested(settings, "data", "source_csv", default="data/raw/air_quality_combined_source.csv")
    model_path = args.model_path or get_nested(settings, "ml", "model_path", default="models/aqi_pm25_model.joblib")
    metrics_path = args.metrics_path or get_nested(settings, "ml", "metrics_path", default="models/aqi_pm25_metrics.json")

    metrics = train_model(settings=settings, input_csv=input_csv, model_path=model_path, metrics_path=metrics_path)
    print("Model training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
