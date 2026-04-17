from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def resolve_path(path_like: str | Path) -> Path:
    """Resolve a project-relative path to an absolute path."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def get_nested(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _env_overrides() -> Dict[str, Any]:
    mqtt_enabled = os.getenv("MQTT_ENABLED")
    sim_interval = os.getenv("SIM_INTERVAL_SECONDS")
    sim_loop = os.getenv("SIM_LOOP")

    updates: Dict[str, Any] = {"mqtt": {}, "simulation": {}}
    if mqtt_enabled is not None:
        updates["mqtt"]["enabled"] = mqtt_enabled.lower() in {"1", "true", "yes", "on"}
    if os.getenv("MQTT_HOST"):
        updates["mqtt"]["host"] = os.getenv("MQTT_HOST")
    if os.getenv("MQTT_PORT"):
        updates["mqtt"]["port"] = int(os.getenv("MQTT_PORT", "1883"))
    if os.getenv("MQTT_USERNAME"):
        updates["mqtt"]["username"] = os.getenv("MQTT_USERNAME", "")
    if os.getenv("MQTT_PASSWORD"):
        updates["mqtt"]["password"] = os.getenv("MQTT_PASSWORD", "")
    if sim_interval is not None:
        updates["simulation"]["interval_seconds"] = float(sim_interval)
    if sim_loop is not None:
        updates["simulation"]["loop"] = sim_loop.lower() in {"1", "true", "yes", "on"}

    return updates


def load_settings(settings_path: str | Path | None = None) -> Dict[str, Any]:
    """Load YAML settings and apply .env overrides."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    path = resolve_path(settings_path or DEFAULT_SETTINGS_PATH)
    with path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}

    updates = _env_overrides()
    return _deep_update(settings, updates)

