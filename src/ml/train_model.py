from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.processing.preprocess import build_preprocessor_from_settings, load_raw_dataframe, preprocess_dataframe
from src.utils.config import PROJECT_ROOT, get_nested, load_settings, resolve_path
from src.utils.helpers import utc_now_iso

MODEL_LABELS = {
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "extra_trees": "Extra Trees",
    "svr_rbf": "Support Vector Regression (RBF)",
    "mlp_regressor": "Neural Network (MLP)",
}


def _path_for_metadata(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _select_candidate_models(settings: Dict[str, Any]) -> Iterable[Tuple[str, str, Any]]:
    configured_models = get_nested(settings, "ml", "candidate_models", default={})

    linear_config = configured_models.get("linear_regression", {})
    if linear_config.get("enabled", True):
        yield "linear_regression", MODEL_LABELS["linear_regression"], LinearRegression(
            fit_intercept=bool(linear_config.get("fit_intercept", True))
        )

    forest_config = configured_models.get("random_forest", {})
    if forest_config.get("enabled", True):
        yield "random_forest", MODEL_LABELS["random_forest"], RandomForestRegressor(
            n_estimators=int(forest_config.get("n_estimators", 250)),
            max_depth=None if forest_config.get("max_depth", None) is None else int(forest_config["max_depth"]),
            min_samples_leaf=int(forest_config.get("min_samples_leaf", 1)),
            max_features=forest_config.get("max_features", 1.0),
            random_state=int(forest_config.get("random_state", 42)),
            n_jobs=int(forest_config.get("n_jobs", 1)),
        )

    boosting_config = configured_models.get("gradient_boosting", {})
    if boosting_config.get("enabled", True):
        yield "gradient_boosting", MODEL_LABELS["gradient_boosting"], GradientBoostingRegressor(
            n_estimators=int(boosting_config.get("n_estimators", 250)),
            learning_rate=float(boosting_config.get("learning_rate", 0.05)),
            max_depth=int(boosting_config.get("max_depth", 3)),
            subsample=float(boosting_config.get("subsample", 1.0)),
            random_state=int(boosting_config.get("random_state", 42)),
        )

    extra_trees_config = configured_models.get("extra_trees", {})
    if extra_trees_config.get("enabled", True):
        yield "extra_trees", MODEL_LABELS["extra_trees"], ExtraTreesRegressor(
            n_estimators=int(extra_trees_config.get("n_estimators", 250)),
            max_depth=None if extra_trees_config.get("max_depth", None) is None else int(extra_trees_config["max_depth"]),
            min_samples_leaf=int(extra_trees_config.get("min_samples_leaf", 1)),
            max_features=extra_trees_config.get("max_features", 1.0),
            random_state=int(extra_trees_config.get("random_state", 42)),
            n_jobs=int(extra_trees_config.get("n_jobs", 1)),
        )

    svr_config = configured_models.get("svr_rbf", {})
    if svr_config.get("enabled", True):
        yield "svr_rbf", MODEL_LABELS["svr_rbf"], SVR(
            kernel="rbf",
            C=float(svr_config.get("C", 25.0)),
            epsilon=float(svr_config.get("epsilon", 0.2)),
            gamma=svr_config.get("gamma", "scale"),
        )

    mlp_config = configured_models.get("mlp_regressor", {})
    if mlp_config.get("enabled", True):
        hidden_layers = mlp_config.get("hidden_layer_sizes", [64])
        yield "mlp_regressor", MODEL_LABELS["mlp_regressor"], MLPRegressor(
            hidden_layer_sizes=tuple(int(size) for size in hidden_layers),
            alpha=float(mlp_config.get("alpha", 0.001)),
            learning_rate_init=float(mlp_config.get("learning_rate_init", 0.001)),
            max_iter=int(mlp_config.get("max_iter", 1400)),
            early_stopping=bool(mlp_config.get("early_stopping", True)),
            random_state=int(mlp_config.get("random_state", 42)),
        )


def _build_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def _compute_metrics(y_true: pd.Series, predictions: Any, n_rows: int, n_train_rows: int, n_test_rows: int) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(mean_squared_error(y_true, predictions) ** 0.5),
        "r2": float(r2_score(y_true, predictions)),
        "n_rows": int(n_rows),
        "n_train_rows": int(n_train_rows),
        "n_test_rows": int(n_test_rows),
    }


def _ranking_key(metrics: Dict[str, float], selection_metric: str) -> Tuple[float, float, float]:
    if selection_metric == "r2":
        return (-metrics["r2"], metrics["rmse"], metrics["mae"])
    return (metrics[selection_metric], metrics["mae"], -metrics["r2"])


def train_model(
    settings: Dict[str, Any],
    input_csv: str | Path,
    model_path: str | Path,
    metrics_path: str | Path,
) -> Dict[str, Any]:
    preprocessor = build_preprocessor_from_settings(settings)
    raw_df = load_raw_dataframe(input_csv)
    df = preprocess_dataframe(raw_df, preprocessor)

    feature_columns: List[str] = get_nested(settings, "ml", "feature_columns", default=[])
    target_column: str = get_nested(settings, "ml", "target_column", default="pm25")
    selection_metric: str = str(get_nested(settings, "ml", "selection_metric", default="rmse")).lower()

    if selection_metric not in {"mae", "rmse", "r2"}:
        raise ValueError(f"Unsupported selection metric: {selection_metric}")

    for col in feature_columns + [target_column]:
        if col not in df.columns:
            df[col] = pd.NA

    train_df = df[feature_columns + [target_column]].dropna(subset=[target_column]).copy()
    X = train_df[feature_columns]
    y = train_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_file = resolve_path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    candidate_dir = model_file.parent / f"{model_file.stem}_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    trained_at = utc_now_iso()
    candidate_results: Dict[str, Dict[str, Any]] = {}
    trained_payloads: Dict[str, Dict[str, Any]] = {}

    for model_name, model_label, estimator in _select_candidate_models(settings):
        pipeline = _build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        candidate_metrics = _compute_metrics(
            y_true=y_test,
            predictions=predictions,
            n_rows=len(train_df),
            n_train_rows=len(X_train),
            n_test_rows=len(X_test),
        )

        candidate_payload = {
            "trained_at": trained_at,
            "model_name": model_name,
            "model_label": model_label,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "pipeline": pipeline,
        }

        artifact_path = candidate_dir / f"{model_name}.joblib"
        joblib.dump(candidate_payload, artifact_path)
        trained_payloads[model_name] = candidate_payload
        candidate_results[model_name] = {
            "label": model_label,
            "artifact_path": _path_for_metadata(artifact_path),
            **candidate_metrics,
        }

    if not candidate_results:
        raise ValueError("No ML candidate model is enabled in settings.")

    ranking = sorted(candidate_results.items(), key=lambda item: _ranking_key(item[1], selection_metric))
    selected_model_name = ranking[0][0]
    selected_model_summary = candidate_results[selected_model_name]
    joblib.dump(trained_payloads[selected_model_name], model_file)

    metrics_file = resolve_path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_summary = {
        "trained_at": trained_at,
        "selection_metric": selection_metric,
        "selected_model": selected_model_name,
        "selected_model_label": selected_model_summary["label"],
        "selected_model_path": _path_for_metadata(model_file),
        "target_column": target_column,
        "feature_columns": feature_columns,
        "mae": selected_model_summary["mae"],
        "rmse": selected_model_summary["rmse"],
        "r2": selected_model_summary["r2"],
        "n_rows": selected_model_summary["n_rows"],
        "n_train_rows": selected_model_summary["n_train_rows"],
        "n_test_rows": selected_model_summary["n_test_rows"],
        "models": candidate_results,
        "ranking": [model_name for model_name, _ in ranking],
    }
    metrics_file.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
    return metrics_summary


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
