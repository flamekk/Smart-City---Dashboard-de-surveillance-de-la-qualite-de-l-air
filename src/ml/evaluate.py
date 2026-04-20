from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import altair as alt
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.processing.preprocess import build_preprocessor_from_settings, load_raw_dataframe, preprocess_dataframe
from src.utils.config import get_nested, load_settings, resolve_path

alt.data_transformers.disable_max_rows()


def _build_eval_frame(df: pd.DataFrame, model_payload: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, Any]:
    feature_columns: List[str] = model_payload["feature_columns"]
    target_column: str = model_payload["target_column"]
    working_df = df.copy()

    for col in feature_columns + [target_column]:
        if col not in working_df.columns:
            working_df[col] = pd.NA

    eval_df = working_df[feature_columns + [target_column]].dropna(subset=[target_column]).copy()
    X = eval_df[feature_columns]
    y = eval_df[target_column]
    pred = model_payload["pipeline"].predict(X)
    return eval_df, y, pred


def _evaluate_payload(df: pd.DataFrame, model_payload: Dict[str, Any]) -> Dict[str, float]:
    eval_df, y, pred = _build_eval_frame(df, model_payload)

    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": float(r2_score(y, pred)),
        "n_rows": int(len(eval_df)),
    }


def _prediction_frame(df: pd.DataFrame, model_payload: Dict[str, Any]) -> pd.DataFrame:
    _, y, pred = _build_eval_frame(df, model_payload)
    model_name = model_payload.get("model_name", "selected_model")
    model_label = model_payload.get("model_label", model_name)
    frame = pd.DataFrame(
        {
            "sample_index": list(range(len(y))),
            "actual_pm25": y.to_numpy(),
            "predicted_pm25": pred,
            "residual": y.to_numpy() - pred,
            "model_name": model_name,
            "model_label": model_label,
        }
    )
    return frame


def _build_line_chart(long_df: pd.DataFrame, title: str) -> alt.Chart:
    palette = ["#111827", "#C62828", "#1565C0", "#2E7D32", "#8E24AA", "#EF6C00"]
    series_order = list(dict.fromkeys(long_df["series"].tolist()))
    return (
        alt.Chart(long_df)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("sample_index:Q", title="Echantillon"),
            y=alt.Y("value:Q", title="PM2.5"),
            color=alt.Color(
                "series:N",
                sort=series_order,
                scale=alt.Scale(domain=series_order, range=palette[: len(series_order)]),
                title="Serie",
            ),
            tooltip=[
                alt.Tooltip("sample_index:Q", title="Echantillon"),
                alt.Tooltip("series:N", title="Serie"),
                alt.Tooltip("value:Q", title="Valeur", format=".2f"),
            ],
        )
        .properties(height=360, title=title)
        .interactive()
    )


def _write_chart_html(chart: alt.Chart, output_path: Path, title: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spec_json = chart.to_json(indent=None)
    page_title = html.escape(title)
    html_body = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{page_title}</title>
  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
</head>
<body style="font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #fafafa;">
  <h2 style="margin-bottom: 8px;">{page_title}</h2>
  <div style="margin-bottom: 16px; color: #4b5563;">Comparaison des valeurs reelles et des predictions.</div>
  <div id="vis"></div>
  <script>
    const spec = {spec_json};
    vegaEmbed("#vis", spec, {{actions: false}});
  </script>
</body>
</html>
"""
    output_path.write_text(html_body, encoding="utf-8")
    return output_path


def export_actual_vs_predicted(
    settings: Dict[str, Any],
    input_csv: str | Path,
    plot_path: str | Path,
    predictions_csv_path: str | Path,
    model_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    include_all_models: bool = False,
) -> Dict[str, Any]:
    preprocessor = build_preprocessor_from_settings(settings)
    df = preprocess_dataframe(load_raw_dataframe(input_csv), preprocessor)

    prediction_frames: List[pd.DataFrame] = []
    if include_all_models:
        if metrics_path is None:
            raise ValueError("metrics_path is required when include_all_models=True")
        metrics_payload = json.loads(resolve_path(metrics_path).read_text(encoding="utf-8"))
        for model_name, saved_metrics in metrics_payload.get("models", {}).items():
            artifact_path = saved_metrics.get("artifact_path")
            if not artifact_path:
                continue
            model_payload = joblib.load(resolve_path(artifact_path))
            prediction_frames.append(_prediction_frame(df, model_payload))
    else:
        if model_path is None:
            raise ValueError("model_path is required when include_all_models=False")
        model_payload = joblib.load(resolve_path(model_path))
        prediction_frames.append(_prediction_frame(df, model_payload))

    if not prediction_frames:
        raise ValueError("No prediction data available to build the comparison plot.")

    actual_series = prediction_frames[0][["sample_index", "actual_pm25"]].rename(columns={"actual_pm25": "Valeurs reelles"})
    wide_df = actual_series.copy()
    long_frames = [
        pd.DataFrame(
            {
                "sample_index": prediction_frames[0]["sample_index"],
                "series": "Valeurs reelles",
                "value": prediction_frames[0]["actual_pm25"],
            }
        )
    ]

    for frame in prediction_frames:
        model_name = str(frame["model_name"].iloc[0])
        model_label = str(frame["model_label"].iloc[0])
        predicted_column = f"prediction_{model_name}"
        wide_df[predicted_column] = frame["predicted_pm25"].to_numpy()
        long_frames.append(
            pd.DataFrame(
                {
                    "sample_index": frame["sample_index"],
                    "series": f"Prediction - {model_label}",
                    "value": frame["predicted_pm25"],
                }
            )
        )

    plot_file = resolve_path(plot_path)
    csv_file = resolve_path(predictions_csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(csv_file, index=False)

    long_df = pd.concat(long_frames, ignore_index=True)
    title = "Comparaison valeurs reelles vs predictions"
    if include_all_models:
        title = "Comparaison valeurs reelles vs predictions des modeles"
    _write_chart_html(_build_line_chart(long_df, title), plot_file, title)

    return {
        "plot_path": str(plot_file),
        "predictions_csv_path": str(csv_file),
        "series": list(dict.fromkeys(long_df["series"].tolist())),
        "n_rows": int(len(wide_df)),
    }


def evaluate_model(settings: Dict[str, Any], input_csv: str | Path, model_path: str | Path) -> Dict[str, float]:
    preprocessor = build_preprocessor_from_settings(settings)
    df = preprocess_dataframe(load_raw_dataframe(input_csv), preprocessor)

    model_payload = joblib.load(resolve_path(model_path))
    metrics = _evaluate_payload(df, model_payload)
    metrics["model_name"] = model_payload.get("model_name", "selected_model")
    metrics["model_label"] = model_payload.get("model_label", metrics["model_name"])
    return metrics


def evaluate_all_models(
    settings: Dict[str, Any],
    input_csv: str | Path,
    metrics_path: str | Path,
) -> Dict[str, Any]:
    preprocessor = build_preprocessor_from_settings(settings)
    df = preprocess_dataframe(load_raw_dataframe(input_csv), preprocessor)

    metrics_payload = json.loads(resolve_path(metrics_path).read_text(encoding="utf-8"))
    comparison: Dict[str, Any] = {}
    for model_name, saved_metrics in metrics_payload.get("models", {}).items():
        artifact_path = saved_metrics.get("artifact_path")
        if not artifact_path:
            continue
        model_payload = joblib.load(resolve_path(artifact_path))
        comparison[model_name] = {
            "label": saved_metrics.get("label", model_name),
            "artifact_path": artifact_path,
            **_evaluate_payload(df, model_payload),
        }

    selected_model = metrics_payload.get("selected_model")
    selected_metrics = comparison.get(selected_model, {})
    selection_metric = metrics_payload.get("selection_metric", "rmse")

    def ranking_key(name: str) -> Any:
        if selection_metric == "r2":
            return (-comparison[name]["r2"], comparison[name]["rmse"], comparison[name]["mae"])
        return (comparison[name][selection_metric], comparison[name]["mae"], -comparison[name]["r2"])

    return {
        "selected_model": selected_model,
        "selected_model_label": metrics_payload.get("selected_model_label", selected_model),
        "selection_metric": selection_metric,
        "mae": selected_metrics.get("mae"),
        "rmse": selected_metrics.get("rmse"),
        "r2": selected_metrics.get("r2"),
        "n_rows": selected_metrics.get("n_rows"),
        "models": comparison,
        "ranking": sorted(comparison.keys(), key=ranking_key),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model on a dataset.")
    parser.add_argument("--config", default="config/settings.yaml", help="Settings YAML path.")
    parser.add_argument("--input-csv", default=None, help="Evaluation CSV path. Defaults to data.source_csv.")
    parser.add_argument("--model-path", default=None, help="Model path. Defaults to ml.model_path.")
    parser.add_argument("--metrics-path", default=None, help="Metrics JSON path. Defaults to ml.metrics_path.")
    parser.add_argument("--all-models", action="store_true", help="Evaluate every saved candidate model for comparison.")
    parser.add_argument("--plot-path", default=None, help="Optional HTML output path for actual-vs-predicted curve.")
    parser.add_argument("--predictions-csv-path", default=None, help="Optional CSV output path for actual/predicted values.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    input_csv = args.input_csv or get_nested(settings, "data", "source_csv", default="data/raw/air_quality_combined_source.csv")
    model_path = args.model_path or get_nested(settings, "ml", "model_path", default="models/aqi_pm25_model.joblib")
    metrics_path = args.metrics_path or get_nested(settings, "ml", "metrics_path", default="models/aqi_pm25_metrics.json")
    plot_path = args.plot_path or get_nested(settings, "ml", "comparison_plot_path", default="models/actual_vs_predicted.html")
    predictions_csv_path = args.predictions_csv_path or get_nested(
        settings,
        "ml",
        "predictions_csv_path",
        default="models/actual_vs_predicted.csv",
    )
    if args.all_models:
        metrics = evaluate_all_models(settings, input_csv=input_csv, metrics_path=metrics_path)
    else:
        metrics = evaluate_model(settings, input_csv=input_csv, model_path=model_path)
    artifacts = export_actual_vs_predicted(
        settings=settings,
        input_csv=input_csv,
        plot_path=plot_path,
        predictions_csv_path=predictions_csv_path,
        model_path=model_path,
        metrics_path=metrics_path,
        include_all_models=args.all_models,
    )
    metrics["artifacts"] = artifacts
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

