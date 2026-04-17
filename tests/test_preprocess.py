from src.processing.preprocess import RowPreprocessor


def test_preprocess_maps_columns_and_derives_pm25() -> None:
    processor = RowPreprocessor(
        aliases={"PM10": "pm10", "CO": "co", "temperature": "temperature", "humidity": "humidity"},
        defaults={"pm25": 20.0, "pm10": 40.0, "co": 0.5, "temperature": 25.0, "humidity": 50.0},
    )
    row = {"PM10": "80", "CO": "1.1", "temperature": "31", "humidity": "70"}
    cleaned = processor.transform(row)
    assert cleaned["pm10"] == 80.0
    assert cleaned["pm25"] == 48.0  # derived from PM10
    assert cleaned["co"] == 1.1


def test_preprocess_uses_forward_fill_then_default() -> None:
    processor = RowPreprocessor(
        aliases={"PM2.5": "pm25"},
        defaults={"pm25": 12.0, "pm10": 30.0, "co": 0.4},
    )
    first = processor.transform({"PM2.5": "25"})
    second = processor.transform({"PM2.5": ""})
    assert first["pm25"] == 25.0
    assert second["pm25"] == 25.0

