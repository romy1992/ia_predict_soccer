import logging
import os


def setup_logging(level: int = logging.INFO) -> None:
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join("logs", "app.log"), encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )

