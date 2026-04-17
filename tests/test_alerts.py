from src.alerts.alert_manager import AlertManager


def test_alert_requires_consecutive_breaches() -> None:
    manager = AlertManager(
        thresholds={"pm25": {"warning": 35.0, "critical": 55.0}},
        consecutive_breaches=3,
    )
    base_payload = {"timestamp": "2026-04-17T10:00:00+00:00", "measurements": {"pm25": 0.0}}

    payloads = [36.0, 38.0, 40.0]
    alerts = []
    for value in payloads:
        payload = dict(base_payload)
        payload["measurements"] = {"pm25": value}
        alerts.extend(manager.evaluate(payload))

    warning_alerts = [a for a in alerts if a["level"] == "warning"]
    assert len(warning_alerts) == 1


def test_alert_escalates_to_critical_and_recovers() -> None:
    manager = AlertManager(
        thresholds={"pm25": {"warning": 35.0, "critical": 55.0}},
        consecutive_breaches=2,
    )

    events = [
        {"timestamp": "t1", "measurements": {"pm25": 40.0}},
        {"timestamp": "t2", "measurements": {"pm25": 41.0}},  # warning triggered
        {"timestamp": "t3", "measurements": {"pm25": 60.0}},  # escalate to critical
        {"timestamp": "t4", "measurements": {"pm25": 20.0}},  # recovery
    ]
    all_alerts = []
    for event in events:
        all_alerts.extend(manager.evaluate(event))

    levels = [a["level"] for a in all_alerts]
    assert "warning" in levels
    assert "critical" in levels
    assert "normal" in levels

