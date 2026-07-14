import logging
import pathlib
from datetime import datetime

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"


class CustomFormatter(logging.Formatter):
    COLORS = {
        "INFO": "\033[94m",     # Blue
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",    # Red
        "CRITICAL": "\033[91m",
        "RESET": "\033[0m",
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        message = super().format(record)
        return f"{color}{message}{self.COLORS['RESET']}"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that prints to console and writes to logs/tidez_{name}_<timestamp>.log."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_file = LOG_DIR / f"tidez_{name}_{timestamp}.log"

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter("[%(levelname)s] at %(asctime)s: \033[97m%(message)s\033[0m"))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("[%(levelname)s] at %(asctime)s: %(message)s"))
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
