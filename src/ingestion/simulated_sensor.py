from __future__ import annotations

"""CSV-based sensor simulator for the academic Smart City IoT workflow."""

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from src.utils.config import get_nested, load_settings, resolve_path
from src.utils.helpers import utc_now_iso


class SimulatedSensor:
    """Replay CSV rows as if they were live sensor readings."""

    def __init__(
        self,
        csv_path: str | Path,
        interval_seconds: float = 2.0,
        loop: bool = True,
        max_records: Optional[int] = None,
        inject_timestamp: bool = True,
    ) -> None:
        self.csv_path = resolve_path(csv_path)
        self.interval_seconds = max(0.0, interval_seconds)
        self.loop = loop
        self.max_records = max_records
        self.inject_timestamp = inject_timestamp
        self._rows = self._load_rows()

    def _load_rows(self) -> List[Dict[str, Any]]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if not rows:
            raise ValueError(f"CSV is empty: {self.csv_path}")
        return rows

    def stream_rows(self) -> Iterator[Dict[str, Any]]:
        emitted = 0
        while True:
            for idx, row in enumerate(self._rows, start=1):
                event = dict(row)
                event["sim_index"] = idx
                if self.inject_timestamp:
                    event["simulated_at"] = utc_now_iso()
                yield event
                emitted += 1
                if self.max_records is not None and emitted >= self.max_records:
                    return
            if not self.loop:
                return

    def run(self, on_message: Callable[[Dict[str, Any]], None]) -> None:
        for event in self.stream_rows():
            on_message(event)
            time.sleep(self.interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a CSV dataset as simulated IoT sensor stream.")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings file.")
    parser.add_argument("--csv-path", default=None, help="CSV path to replay.")
    parser.add_argument("--interval", type=float, default=None, help="Delay between samples (seconds).")
    parser.add_argument("--loop", action="store_true", help="Force loop mode.")
    parser.add_argument("--no-loop", action="store_true", help="Disable loop mode.")
    parser.add_argument("--max-records", type=int, default=None, help="Stop after N records.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    csv_path = args.csv_path or get_nested(settings, "data", "source_csv", default="data/raw/air_quality_combined_source.csv")
    interval = args.interval if args.interval is not None else float(
        get_nested(settings, "simulation", "interval_seconds", default=2.0)
    )
    loop_default = bool(get_nested(settings, "simulation", "loop", default=True))
    loop = False if args.no_loop else (True if args.loop else loop_default)

    sensor = SimulatedSensor(csv_path=csv_path, interval_seconds=interval, loop=loop, max_records=args.max_records)

    def _printer(message: Dict[str, Any]) -> None:
        print(message)

    sensor.run(_printer)


if __name__ == "__main__":
    main()
