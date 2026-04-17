from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.utils.config import resolve_path

SEVERITY_ORDER = {"normal": 0, "warning": 1, "critical": 2}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    value_str = str(value).strip()
    if not value_str:
        return None
    try:
        return float(value_str)
    except ValueError:
        return None


def max_status(levels: Iterable[str]) -> str:
    selected = "normal"
    for level in levels:
        if SEVERITY_ORDER.get(level, 0) > SEVERITY_ORDER.get(selected, 0):
            selected = level
    return selected


def read_jsonl_tail(path: str | Path, limit: int = 200) -> List[Dict[str, Any]]:
    file_path = resolve_path(path)
    if not file_path.exists():
        return []
    lines = file_path.read_text(encoding="utf-8").splitlines()
    tail = lines[-limit:] if limit > 0 else lines
    output: List[Dict[str, Any]] = []
    for line in tail:
        try:
            output.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return output

