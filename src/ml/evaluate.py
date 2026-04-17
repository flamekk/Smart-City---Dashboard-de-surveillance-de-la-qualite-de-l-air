from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.processing.preprocess import build_preprocessor_from_settings, load_raw_dataframe, preprocess_dataframe
from src.utils.config import get_nested, load_settings, resolve_path


def evaluate_model(settings: Dict[str, Any], input_csv: str | Path, model_path: str | Path) -> Dict[str, float]:
    preprocessor = build_preprocessor_from_settings(settings)
    df = preprocess_dataframe(load_raw_dataframe(input_csv), preprocessor)

    model_payload = joblib.load(resolve_path(model_path))
    pipeline = model_payload["pipeline"]
    feature_columns: List[str] = model_payload["feature_columns"]
    target_column: str = model_payload["target_column"]

    for col in feature_columns + [target_column]:
        if col not in df.columns:
            df[col] = pd.NA

    eval_df = df[feature_columns + [target_column]].dropna(subset=[target_column]).copy()
    X = eval_df[feature_columns]
    y = eval_df[target_column]
    pred = pipeline.predict(X)

    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": float(r2_score(y, pred)),
        "n_rows": int(len(eval_df)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model on a dataset.")
    parser.add_argument("--config", default="config/settings.yaml", help="Settings YAML path.")
    parser.add_argument("--input-csv", default=None, help="Evaluation CSV path. Defaults to data.source_csv.")
    parser.add_argument("--model-path", default=None, help="Model path. Defaults to ml.model_path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    input_csv = args.input_csv or get_nested(settings, "data", "source_csv", default="data/raw/air_quality_combined_source.csv")
    model_path = args.model_path or get_nested(settings, "ml", "model_path", default="models/aqi_pm25_model.joblib")
    metrics = evaluate_model(settings, input_csv=input_csv, model_path=model_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

