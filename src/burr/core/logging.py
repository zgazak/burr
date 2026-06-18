import logging
import logging.config
from enum import Enum
from typing import Any, Dict, Literal
from pathlib import Path
from loguru import logger as loguru_logger
import sys

from burr.core.constants import LOG_PATH

std_logger = logging.getLogger(__name__)

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class LogMode(str, Enum):
    LOCAL = "local"


def get_local_config() -> Dict[str, Any]:
    """Configuration for local development with timestamps and color"""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "colored": {
                "()": "colorlog.ColoredFormatter",
                "format": "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "log_colors": {
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
            },
            "file": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "datefmt": "%Y-%m-%d %H:%M:%S"},
            "uvicorn": {
                "()": "uvicorn.logging.DefaultFormatter",
                "format": "%(levelprefix)s %(name)s - %(message)s",
                "use_colors": True,
            },
            "uvicorn.access": {
                "()": "uvicorn.logging.AccessFormatter",
                "format": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
                "use_colors": True,
            },
        },
        "handlers": {
            "console": {"class": "colorlog.StreamHandler", "formatter": "colored", "stream": "ext://sys.stdout"},
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "file",
                "filename": str(LOG_PATH),
                "maxBytes": 10485760,
                "backupCount": 5,
            },
            "uvicorn": {"class": "logging.StreamHandler", "formatter": "uvicorn", "stream": "ext://sys.stdout"},
            "uvicorn.access": {
                "class": "logging.StreamHandler",
                "formatter": "uvicorn.access",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn": {"handlers": ["uvicorn", "file"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["uvicorn.access", "file"], "level": "INFO", "propagate": False},
        },
    }


def setup_logging(
    level: str = "INFO", disabled_loggers: list[str] = None, run_output_dir: Path | None = None, force: bool = False
) -> None:
    """Configure logging based on level and context

    Args:
        level: The default logging level for all loggers
        disabled_loggers: List of logger names to disable or set to a higher level
        run_output_dir: Optional run output directory to store logs in
        force: If True, reconfigure logging even if already configured
    """
    # Check if logging is already configured with our settings
    root_logger = logging.getLogger()
    if not force and root_logger.handlers and hasattr(root_logger, "burr_logging_configured"):
        std_logger.info("Logging already configured, skipping setup")
        return

    # Remove any existing handlers if forcing reconfiguration
    if force and root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    # Determine log file path
    if run_output_dir is not None:
        log_path = run_output_dir / "logs" / "app.log"
    else:
        log_path = LOG_PATH

    log_path.parent.mkdir(parents=True, exist_ok=True)
    config = get_local_config()

    # Update log file path in config
    config["handlers"]["file"]["filename"] = str(log_path)

    # Update log levels
    for logger_ in config["loggers"].values():
        logger_["level"] = level.upper()

    # Disable specific loggers if requested
    if disabled_loggers:
        for logger_name in disabled_loggers:
            # If the logger is already in the config, set it to WARNING or higher
            if logger_name in config["loggers"]:
                config["loggers"][logger_name]["level"] = "WARNING"
            # Otherwise add a new logger config
            else:
                config["loggers"][logger_name] = {
                    "handlers": ["console", "file"],
                    "level": "WARNING",
                    "propagate": False,
                }

            # Also directly set the level for any already-created loggers
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    logging.config.dictConfig(config)

    # Mark logging as configured
    root_logger.burr_logging_configured = True
    std_logger.info(f"Completed logging setup with level `{level}`")


def set_log_level(level: LogLevel) -> None:
    """Change the log level of all configured loggers.

    Args:
        level: The new logging level to set
    """
    root_logger = logging.getLogger()
    if not hasattr(root_logger, "burr_logging_configured"):
        std_logger.warning("Logging not configured with setup_logging(), setting level may have limited effect")

    # Convert string to uppercase for consistency
    level_upper = level.upper()

    # Set level for root logger
    root_logger.setLevel(level_upper)

    # Set level for all existing loggers
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(level_upper)

    std_logger.info(f"Changed logging level to `{level_upper}`")


def setup_loguru(level: str = "INFO", run_output_dir: Path | None = None) -> None:
    """Configure loguru logging based on level and context.

    Args:
        level: The default logging level (default: INFO)
        run_output_dir: Optional run output directory to store logs in
    """
    # Remove the default handler
    loguru_logger.remove()

    # Determine log file path
    if run_output_dir is not None:
        log_path = run_output_dir / "logs" / "app.log"
    else:
        log_path = LOG_PATH

    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Add stdout handler with a clean format
    loguru_logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level.upper(),
        colorize=True,
    )

    # Add file handler with the same format for consistency
    loguru_logger.add(
        str(log_path),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level.upper(),
        rotation="10 MB",
        retention="10 days",
    )

    loguru_logger.info(f"Logging setup complete. File logs will be written to {log_path}")
