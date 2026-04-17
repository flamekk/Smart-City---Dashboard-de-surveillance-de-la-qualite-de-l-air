from src.main import run_pipeline
from src.utils.config import load_settings


def main() -> None:
    settings = load_settings("config/settings.yaml")
    run_pipeline(
        settings=settings,
        demo_mode=True,
        max_records=120,
        loop=False,
        force_ml_enabled=False,
    )


if __name__ == "__main__":
    main()

