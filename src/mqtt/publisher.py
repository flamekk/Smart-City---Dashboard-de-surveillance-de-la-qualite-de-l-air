from __future__ import annotations

import json
import logging
from typing import Any, Dict

from src.mqtt.topics import resolve_topics
from src.utils.config import get_nested
from src.utils.helpers import utc_now_iso
from src.utils.logger import append_jsonl

try:
    import paho.mqtt.client as mqtt
except Exception:  # pragma: no cover - import guard for lightweight environments.
    mqtt = None


class MQTTPublisher:
    def __init__(self, settings: Dict[str, Any], logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.topics = resolve_topics(settings)
        self.enabled = bool(get_nested(settings, "mqtt", "enabled", default=True))
        self.fallback_to_file = bool(get_nested(settings, "mqtt", "fallback_to_file", default=True))
        self.fallback_file = "logs/mqtt_fallback.jsonl"

        self._client = None
        self._connected = False

        if self.enabled:
            self._connect()
        else:
            self.logger.info("MQTT disabled in config; using file-based fallback only.")

    @property
    def connected(self) -> bool:
        return self._connected

    def _connect(self) -> None:
        if mqtt is None:
            self.logger.warning("paho-mqtt unavailable. MQTT disabled and fallback mode enabled.")
            self._connected = False
            return

        host = get_nested(self.settings, "mqtt", "host", default="localhost")
        port = int(get_nested(self.settings, "mqtt", "port", default=1883))
        keepalive = int(get_nested(self.settings, "mqtt", "keepalive", default=60))
        client_id = get_nested(self.settings, "mqtt", "client_id", default="smartcity-air-quality-publisher")
        username = get_nested(self.settings, "mqtt", "username", default="")
        password = get_nested(self.settings, "mqtt", "password", default="")

        try:
            self._client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)
            if username:
                self._client.username_pw_set(username, password)
            self._client.connect(host, port, keepalive=keepalive)
            self._client.loop_start()
            self._connected = True
            self.logger.info("Connected to MQTT broker at %s:%s", host, port)
        except Exception as exc:
            self._connected = False
            self.logger.warning("MQTT connection failed (%s). Fallback mode enabled.", exc)

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        if self._connected and self._client is not None:
            message = json.dumps(payload, ensure_ascii=False)
            try:
                result = self._client.publish(topic, message)
                if result.rc != 0:
                    raise RuntimeError(f"publish rc={result.rc}")
            except Exception as exc:
                self.logger.warning("MQTT publish failed on topic %s: %s", topic, exc)
                self._fallback(topic, payload)
        else:
            self._fallback(topic, payload)

    def publish_data(self, payload: Dict[str, Any]) -> None:
        self.publish(self.topics["data"], payload)

    def publish_alert(self, payload: Dict[str, Any]) -> None:
        self.publish(self.topics["alerts"], payload)

    def publish_status(self, status: str, details: Dict[str, Any] | None = None) -> None:
        body = {"timestamp": utc_now_iso(), "status": status}
        if details:
            body["details"] = details
        self.publish(self.topics["status"], body)

    def _fallback(self, topic: str, payload: Dict[str, Any]) -> None:
        if not self.fallback_to_file:
            return
        append_jsonl(
            self.fallback_file,
            {
                "timestamp": utc_now_iso(),
                "topic": topic,
                "payload": payload,
            },
        )

    def close(self) -> None:
        if self._client is not None and self._connected:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
        self._connected = False

