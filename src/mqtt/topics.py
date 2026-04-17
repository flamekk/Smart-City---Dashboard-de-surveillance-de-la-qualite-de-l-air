from __future__ import annotations

from typing import Any, Dict

from src.utils.config import get_nested

DEFAULT_TOPICS = {
    "data": "air_quality/data",
    "alerts": "air_quality/alerts",
    "status": "air_quality/status",
}


def resolve_topics(settings: Dict[str, Any]) -> Dict[str, str]:
    configured = get_nested(settings, "mqtt", "topics", default={}) or {}
    merged = dict(DEFAULT_TOPICS)
    merged.update(configured)
    return merged

