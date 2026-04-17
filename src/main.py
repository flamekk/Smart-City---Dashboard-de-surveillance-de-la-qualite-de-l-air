from __future__ import annotations

"""Main runtime for the simulated IoT pipeline.

In this academic version, CSV records are replayed to emulate sensor acquisition
because physical sensors were not available during implementation.
"""

import argparse
from typing import Any, Dict, Optional

from src.alerts.alert_manager import AlertManager
from src.ingestion.simulated_sensor import SimulatedSensor
from src.ml.predict import MLPredictor
from src.mqtt.publisher import MQTTPublisher
from src.processing.edge_processor import EdgeProcessor
from src.utils.config import get_nested, load_settings
from src.utils.helpers import max_status
from src.utils.logger import AlertHistoryWriter, append_jsonl, get_logger


def _maybe_load_ml_predictor(settings: Dict[str, Any], force_enabled: Optional[bool], logger) -> Optional[MLPredictor]:
    enabled = force_enabled if force_enabled is not None else bool(get_nested(settings, "ml", "enabled_by_default", default=False))
    if not enabled:
        logger.info("ML predictor disabled.")
        return None
    model_path = get_nested(settings, "ml", "model_path", default="models/aqi_pm25_model.joblib")
    try:
        predictor = MLPredictor(model_path)
        logger.info("ML predictor loaded from %s", model_path)
        return predictor
    except Exception as exc:
        logger.warning("Unable to load ML predictor (%s). Continuing without ML.", exc)
        return None


def run_pipeline(
    settings: Dict[str, Any],
    csv_path: Optional[str] = None,
    interval_seconds: Optional[float] = None,
    max_records: Optional[int] = None,
    loop: Optional[bool] = None,
    demo_mode: bool = False,
    force_ml_enabled: Optional[bool] = None,
) -> None:
    logger = get_logger()
    ml_predictor = _maybe_load_ml_predictor(settings, force_ml_enabled, logger)

    default_csv = get_nested(settings, "data", "source_csv", default="data/raw/air_quality_combined_source.csv")
    fallback_csv = get_nested(settings, "data", "fallback_csv", default="data/raw/low_cost_sensor_data.csv")
    input_csv = csv_path or default_csv

    configured_interval = float(get_nested(settings, "simulation", "interval_seconds", default=2.0))
    demo_interval = float(get_nested(settings, "simulation", "demo_interval_seconds", default=0.4))
    interval = interval_seconds if interval_seconds is not None else (demo_interval if demo_mode else configured_interval)

    configured_loop = bool(get_nested(settings, "simulation", "loop", default=True))
    use_loop = configured_loop if loop is None else loop
    use_max_records = max_records if max_records is not None else get_nested(settings, "simulation", "max_records", default=None)

    alert_thresholds = get_nested(settings, "alerts", "thresholds", default={})
    consecutive = int(get_nested(settings, "alerts", "consecutive_breaches", default=3))

    data_log_path = get_nested(settings, "dashboard", "data_log_path", default="logs/stream_data.jsonl")
    alert_csv_path = get_nested(settings, "dashboard", "alert_log_path", default="logs/alerts_history.csv")

    alert_manager = AlertManager(thresholds=alert_thresholds, consecutive_breaches=consecutive)
    edge_processor = EdgeProcessor(settings=settings, ml_predictor=ml_predictor)
    mqtt_publisher = MQTTPublisher(settings=settings, logger=logger)
    alert_writer = AlertHistoryWriter(alert_csv_path)

    try:
        sensor = SimulatedSensor(
            csv_path=input_csv,
            interval_seconds=float(interval),
            loop=bool(use_loop),
            max_records=use_max_records,
            inject_timestamp=True,
        )
    except FileNotFoundError:
        logger.warning("Configured source CSV not found (%s). Falling back to %s", input_csv, fallback_csv)
        sensor = SimulatedSensor(
            csv_path=fallback_csv,
            interval_seconds=float(interval),
            loop=bool(use_loop),
            max_records=use_max_records,
            inject_timestamp=True,
        )

    def on_message(raw_row: Dict[str, Any]) -> None:
        processed = edge_processor.process(raw_row)
        instant_status = alert_manager.instant_status(processed["measurements"])
        alerts = alert_manager.evaluate(processed)
        active_status = alert_manager.active_status()
        processed["status"] = max_status([processed.get("status", "normal"), instant_status, active_status])
        processed["alerts_triggered"] = len(alerts)

        append_jsonl(data_log_path, processed)
        mqtt_publisher.publish_data(processed)
        mqtt_publisher.publish_status(
            processed["status"],
            details={
                "aqi_estimated": processed.get("aqi_estimated"),
                "risk_score": processed.get("risk_score"),
                "alerts_triggered": len(alerts),
            },
        )

        if alerts:
            for alert in alerts:
                mqtt_publisher.publish_alert(alert)
                alert_writer.append(alert)
                logger.warning("ALERT | %s", alert["message"])

        logger.info(
            "DATA | pm25=%.2f pm10=%.2f aqi=%s status=%s",
            processed["measurements"].get("pm25") or 0.0,
            processed["measurements"].get("pm10") or 0.0,
            processed.get("aqi_estimated"),
            processed["status"],
        )

    logger.info("Pipeline started. Press CTRL+C to stop.")
    mqtt_publisher.publish_status("online", details={"mode": "demo" if demo_mode else "normal"})

    try:
        sensor.run(on_message=on_message)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
    finally:
        mqtt_publisher.publish_status("offline", details={"reason": "stopped"})
        mqtt_publisher.close()
        logger.info("Pipeline stopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simulated Smart City air quality IoT pipeline.")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to YAML settings.")
    parser.add_argument("--csv-path", default=None, help="CSV source for sensor simulation.")
    parser.add_argument("--interval", type=float, default=None, help="Sample interval in seconds.")
    parser.add_argument("--max-records", type=int, default=None, help="Stop after N samples.")
    parser.add_argument("--no-loop", action="store_true", help="Disable looping over CSV.")
    parser.add_argument("--demo", action="store_true", help="Use fast demo interval from config.")
    parser.add_argument("--ml", action="store_true", help="Enable optional ML predictor even if disabled by default.")
    parser.add_argument("--no-ml", action="store_true", help="Disable optional ML predictor.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    force_ml = True if args.ml else (False if args.no_ml else None)
    run_pipeline(
        settings=settings,
        csv_path=args.csv_path,
        interval_seconds=args.interval,
        max_records=args.max_records,
        loop=False if args.no_loop else None,
        demo_mode=args.demo,
        force_ml_enabled=force_ml,
    )


if __name__ == "__main__":
    main()
