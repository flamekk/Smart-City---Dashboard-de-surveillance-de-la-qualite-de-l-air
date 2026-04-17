from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.config import get_nested, load_settings, resolve_path
from src.utils.helpers import coerce_float

NUMERIC_FIELDS = ["pm25", "pm10", "co", "no2", "so2", "o3", "temperature", "humidity", "wind_speed", "co2", "tvoc"]


def _normalized_alias_map(config_aliases: Dict[str, str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for key, value in config_aliases.items():
        aliases[key] = value
        aliases[key.lower()] = value
    return aliases


@dataclass
class RowPreprocessor:
    aliases: Dict[str, str]
    defaults: Dict[str, float]
    enforce_non_negative: bool = True
    last_valid: Dict[str, float] = field(default_factory=dict)

    def _map_key(self, key: str) -> str:
        if key in self.aliases:
            return self.aliases[key]
        lower_key = key.lower()
        if lower_key in self.aliases:
            return self.aliases[lower_key]
        sanitized = (
            lower_key.replace(" ", "_")
            .replace("-", "_")
            .replace(".", "")
            .replace("(", "")
            .replace(")", "")
        )
        return self.aliases.get(sanitized, sanitized)

    def _fill_missing(self, field: str, value: Optional[float]) -> Optional[float]:
        if value is not None:
            self.last_valid[field] = value
            return value
        if field in self.last_valid:
            return self.last_valid[field]
        return self.defaults.get(field)

    def transform(self, raw_row: Dict[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for raw_key, raw_value in raw_row.items():
            mapped_key = self._map_key(raw_key)
            cleaned[mapped_key] = raw_value

        # Harmonize PM keys if dataset has different naming style.
        if "pm25" not in cleaned and "pm2_5" in cleaned:
            cleaned["pm25"] = cleaned["pm2_5"]

        numeric_values: Dict[str, Optional[float]] = {}
        for field in NUMERIC_FIELDS:
            value = coerce_float(cleaned.get(field))
            if value is not None and self.enforce_non_negative and value < 0:
                value = 0.0
            numeric_values[field] = value

        # Derivation strategy when a metric is missing in source data.
        if numeric_values.get("pm25") is None and numeric_values.get("pm10") is not None:
            numeric_values["pm25"] = round(float(numeric_values["pm10"]) * 0.6, 2)
        if numeric_values.get("pm10") is None and numeric_values.get("pm25") is not None:
            numeric_values["pm10"] = round(float(numeric_values["pm25"]) * 1.5, 2)

        for field in NUMERIC_FIELDS:
            value = self._fill_missing(field, numeric_values.get(field))
            if value is not None and self.enforce_non_negative and value < 0:
                value = 0.0
            cleaned[field] = value

        # Keep date-like value if present for traceability.
        source_date = cleaned.get("source_date")
        if source_date is not None:
            cleaned["source_date"] = str(source_date)

        return cleaned


def build_preprocessor_from_settings(settings: Dict[str, Any]) -> RowPreprocessor:
    aliases = _normalized_alias_map(get_nested(settings, "preprocessing", "column_aliases", default={}))
    defaults = get_nested(settings, "preprocessing", "defaults", default={})
    enforce_non_negative = bool(get_nested(settings, "preprocessing", "enforce_non_negative", default=True))
    return RowPreprocessor(aliases=aliases, defaults=defaults, enforce_non_negative=enforce_non_negative)


def load_raw_dataframe(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(resolve_path(csv_path))


def preprocess_dataframe(df: pd.DataFrame, preprocessor: RowPreprocessor) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        records.append(preprocessor.transform(row))
    output = pd.DataFrame(records)
    output = output.drop_duplicates().reset_index(drop=True)
    return output


def run_preprocessing(input_csv: str | Path, output_csv: str | Path, settings_path: str | Path | None = None) -> Path:
    settings = load_settings(settings_path)
    preprocessor = build_preprocessor_from_settings(settings)

    df_raw = load_raw_dataframe(input_csv)
    df_processed = preprocess_dataframe(df_raw, preprocessor)

    output_path = resolve_path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess air quality dataset for the simulated IoT pipeline.")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings YAML file.")
    parser.add_argument("--input-csv", default=None, help="Input CSV path. Defaults to settings:data.source_csv.")
    parser.add_argument(
        "--output-csv",
        default="data/processed/air_quality_clean.csv",
        help="Output cleaned CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    input_csv = args.input_csv or get_nested(settings, "data", "source_csv", default="data/raw/air_quality_combined_source.csv")
    out_path = run_preprocessing(input_csv=input_csv, output_csv=args.output_csv, settings_path=args.config)
    print(f"Preprocessed data saved to: {out_path}")


if __name__ == "__main__":
    main()
