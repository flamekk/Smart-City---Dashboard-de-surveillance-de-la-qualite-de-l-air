import csv
from pathlib import Path
from uuid import uuid4

from src.ingestion.simulated_sensor import SimulatedSensor


def _build_csv() -> Path:
    base_dir = Path("data/sample")
    base_dir.mkdir(parents=True, exist_ok=True)
    file_path = base_dir / f"sensor_test_{uuid4().hex}.csv"
    with file_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["PM2.5", "PM10"])
        writer.writeheader()
        writer.writerow({"PM2.5": 10, "PM10": 20})
        writer.writerow({"PM2.5": 11, "PM10": 21})
    return file_path


def test_sensor_stream_non_loop() -> None:
    csv_path = _build_csv()
    try:
        sensor = SimulatedSensor(csv_path=csv_path, interval_seconds=0, loop=False)
        rows = list(sensor.stream_rows())
        assert len(rows) == 2
        assert rows[0]["PM2.5"] == "10"
    finally:
        csv_path.unlink(missing_ok=True)


def test_sensor_stream_loop_with_max_records() -> None:
    csv_path = _build_csv()
    try:
        sensor = SimulatedSensor(csv_path=csv_path, interval_seconds=0, loop=True, max_records=3)
        rows = list(sensor.stream_rows())
        assert len(rows) == 3
        assert rows[2]["PM10"] == "20"
    finally:
        csv_path.unlink(missing_ok=True)
