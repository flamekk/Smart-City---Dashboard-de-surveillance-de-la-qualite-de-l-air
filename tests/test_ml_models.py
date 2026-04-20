import csv
from pathlib import Path
from uuid import uuid4

from src.ml.evaluate import evaluate_all_models, export_actual_vs_predicted
from src.ml.predict import MLPredictor
from src.ml.train_model import train_model


def _build_training_csv(path: Path, rows: int = 72) -> Path:
    fieldnames = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "temperature", "humidity", "wind_speed"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(rows):
            pm10 = 45 + (index % 12) * 4
            no2 = 12 + (index % 7) * 3
            so2 = 4 + (index % 5)
            co = round(0.4 + (index % 6) * 0.2, 3)
            o3 = 18 + (index % 8) * 2
            temperature = 21 + (index % 10)
            humidity = 48 + (index % 9) * 4
            wind_speed = round(1.2 + (index % 5) * 0.5, 3)
            noise = ((index % 4) - 1.5) * 1.1
            pm25 = round(
                0.42 * pm10
                + 0.22 * no2
                + 0.08 * humidity
                - 0.65 * wind_speed
                + noise,
                3,
            )
            writer.writerow(
                {
                    "PM2.5": pm25,
                    "PM10": pm10,
                    "NO2": no2,
                    "SO2": so2,
                    "CO": co,
                    "O3": o3,
                    "temperature": temperature,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                }
            )
    return path


def _build_settings() -> dict:
    return {
        "preprocessing": {
            "enforce_non_negative": True,
            "defaults": {
                "pm25": 20.0,
                "pm10": 40.0,
                "co": 0.5,
                "no2": 20.0,
                "so2": 5.0,
                "o3": 25.0,
                "temperature": 26.0,
                "humidity": 60.0,
                "wind_speed": 2.5,
            },
            "column_aliases": {
                "PM2.5": "pm25",
                "PM10": "pm10",
                "CO": "co",
                "NO2": "no2",
                "SO2": "so2",
                "O3": "o3",
                "temperature": "temperature",
                "humidity": "humidity",
                "wind_speed": "wind_speed",
            },
        },
        "ml": {
            "selection_metric": "rmse",
            "target_column": "pm25",
            "feature_columns": [
                "pm10",
                "no2",
                "so2",
                "co",
                "o3",
                "temperature",
                "humidity",
                "wind_speed",
            ],
            "candidate_models": {
                "linear_regression": {"enabled": True},
                "random_forest": {"enabled": True, "n_estimators": 25, "random_state": 42, "n_jobs": 1},
                "gradient_boosting": {
                    "enabled": True,
                    "n_estimators": 40,
                    "learning_rate": 0.08,
                    "max_depth": 2,
                    "random_state": 42,
                },
                "extra_trees": {"enabled": True, "n_estimators": 25, "random_state": 42, "n_jobs": 1},
                "svr_rbf": {"enabled": True, "C": 10.0, "epsilon": 0.2, "gamma": "scale"},
                "mlp_regressor": {
                    "enabled": True,
                    "hidden_layer_sizes": [32],
                    "alpha": 0.001,
                    "learning_rate_init": 0.001,
                    "max_iter": 800,
                    "early_stopping": True,
                    "random_state": 42,
                },
            },
        },
    }


def test_train_model_compares_multiple_regressors() -> None:
    base_dir = Path("data/sample") / f"ml_test_{uuid4().hex}"
    base_dir.mkdir(parents=True, exist_ok=True)
    csv_path = _build_training_csv(base_dir / "air_quality_train.csv")
    model_path = base_dir / "aqi_pm25_model.joblib"
    metrics_path = base_dir / "aqi_pm25_metrics.json"
    settings = _build_settings()

    try:
        metrics = train_model(settings=settings, input_csv=csv_path, model_path=model_path, metrics_path=metrics_path)

        expected_models = {
            "linear_regression",
            "random_forest",
            "gradient_boosting",
            "extra_trees",
            "svr_rbf",
            "mlp_regressor",
        }

        assert metrics["selected_model"] in expected_models
        assert metrics["ranking"]
        assert set(metrics["models"]) == expected_models
        assert model_path.exists()
        assert metrics_path.exists()

        for saved_model in metrics["models"].values():
            assert Path(saved_model["artifact_path"]).exists()

        predictor = MLPredictor(model_path)
        prediction = predictor.predict_pm25(
            {
                "pm10": 78.0,
                "no2": 18.0,
                "so2": 6.0,
                "co": 1.1,
                "o3": 24.0,
                "temperature": 28.0,
                "humidity": 64.0,
                "wind_speed": 2.0,
            }
        )
        assert isinstance(prediction, float)

        comparison = evaluate_all_models(settings=settings, input_csv=csv_path, metrics_path=metrics_path)
        assert comparison["selected_model"] == metrics["selected_model"]
        assert set(comparison["models"]) == expected_models

        artifacts = export_actual_vs_predicted(
            settings=settings,
            input_csv=csv_path,
            plot_path=base_dir / "actual_vs_predicted.html",
            predictions_csv_path=base_dir / "actual_vs_predicted.csv",
            metrics_path=metrics_path,
            include_all_models=True,
        )
        assert Path(artifacts["plot_path"]).exists()
        assert Path(artifacts["predictions_csv_path"]).exists()
        html_content = Path(artifacts["plot_path"]).read_text(encoding="utf-8")
        csv_content = Path(artifacts["predictions_csv_path"]).read_text(encoding="utf-8")
        assert "Valeurs reelles" in html_content
        assert "prediction_linear_regression" in csv_content
    finally:
        for file_path in sorted(base_dir.rglob("*"), reverse=True):
            if file_path.is_file():
                file_path.unlink(missing_ok=True)
            elif file_path.is_dir():
                file_path.rmdir()
        base_dir.rmdir()
