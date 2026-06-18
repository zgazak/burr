"""
Explicit engine initialization.

Call initialize_engine() from CLI entrypoints before doing any work.
Importing burr no longer triggers side effects.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from burr.utils.astropy import setup_astropy
from burr.utils.ephemeris import ensure_ephemeris

_initialized = False


def load_environment():
    """Load environment variables from .env file.

    Looks for .env in the following locations in order:
    1. Path specified by BURR_ENV_PATH environment variable
    2. Current working directory
    3. Project root directory (parent of cwd)
    4. User's home directory
    """
    env_path = Path(os.getenv("BURR_ENV_PATH", ".env"))
    if env_path.exists():
        logger.info(f"Loading environment variables from {env_path}")
        load_dotenv(env_path)
        return

    env_path = Path(os.getcwd()) / ".env"
    if env_path.exists():
        logger.info(f"Loading environment variables from {env_path}")
        load_dotenv(env_path)
        return

    env_path = Path(os.getcwd()).parent / ".env"
    if env_path.exists():
        logger.info(f"Loading environment variables from {env_path}")
        load_dotenv(env_path)
        return

    env_path = Path.home() / ".env"
    if env_path.exists():
        logger.info(f"Loading environment variables from {env_path}")
        load_dotenv(env_path)
        return

    logger.debug("No .env file found in any of the search locations")


def initialize_engine():
    """Initialize all required engine components.

    Safe to call multiple times - subsequent calls are no-ops.
    """
    global _initialized
    if _initialized:
        return

    logger.info("Initializing engine...")
    load_environment()
    setup_astropy()
    ensure_ephemeris()
    _initialized = True
    logger.info("Engine initialized.")
