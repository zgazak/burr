"""
Run management utilities for burr astronomical observation processing.

Handles data persistence and run resumability using the Pydantic BurrRun model.
"""

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from loguru import logger

from burr.models.site import SiteMetadata, LightingSchedule
from burr.core.config import get_config, initialize_config, AppConfig
from burr.core.constants import LOCAL_APP_CONFIG_OVERRIDE
from burr.run import (
    BurrRun,
    create_burr_run,
    find_existing_runs,
    load_run_state,
    log_command,
)


def get_or_create_run(
    site: SiteMetadata = None,
    observation_date: datetime | None = None,
    verbose: bool = False,
    new_config: "AppConfig" = None,
) -> BurrRun:
    """
    Get existing run or create a new one.

    Parameters
    ----------
    site : SiteMetadata, optional
        Site information, uses config.site if None
    observation_date : datetime, optional
        Observation date, uses appropriate midnight if None
    verbose : bool
        Enable verbose output
    new_config : AppConfig, optional
        New configuration to apply to loaded run

    Returns
    -------
    BurrRun
        Existing or new run object
    """
    config = get_config()

    # Use site from config if not provided
    if site is None:
        site = config.site

    # If observation_date not provided, calculate appropriate midnight
    if observation_date is None:
        # Get current UTC time
        current_utc = datetime.now(ZoneInfo("UTC"))

        # Get timezone using site's get_timezone method
        try:
            # Get timezone string
            timezone_str = str(site.get_timezone())
            # Convert to local time
            current_local = current_utc.astimezone(ZoneInfo(timezone_str))
            logger.info(f"🌍 Using timezone {timezone_str} for site {site.name}")
        except Exception as e:
            # If timezone conversion fails, use UTC
            current_local = current_utc
            logger.warning(f"⚠️ Could not use timezone for site {site.name}, using UTC: {str(e)}")

        # Calculate target midnight based on current local time
        if current_local.hour >= 12:
            # After noon - use tonight's midnight (start of next day)
            next_day = current_local + timedelta(days=1)
            observation_date = next_day.replace(hour=0, minute=0, second=0, microsecond=0)
            logger.info("After local noon, using tonight's midnight")
        else:
            # Before noon - use last night's midnight (start of current day)
            observation_date = current_local.replace(hour=0, minute=0, second=0, microsecond=0)
            logger.info("Before local noon, using last night's midnight")

    logger.info(f"🕒 Using observation date: {observation_date}")

    # Calculate lighting schedule for observation date
    schedule = LightingSchedule.calculate_twilights(site, observation_date)

    # Add detailed debug logging
    logger.info(f"📅 Schedule start (UTC): {schedule.schedule_start_utc}")
    logger.info(f"📅 Schedule end (UTC): {schedule.schedule_end_utc}")
    logger.info(f"🌙 Schedule local midnight: {schedule.local_midnight}")

    # Look for existing runs
    existing_runs = find_existing_runs()

    if existing_runs:
        logger.info(f"🔍 Found {len(existing_runs)} existing run(s)")

        # Try to load the most recent run
        latest_run_dir = existing_runs[0]
        run = load_run_state(latest_run_dir, new_config=new_config)

        if run:
            # Compare local midnight dates instead of raw observation dates
            if run.lighting_schedule.local_midnight.date() == schedule.local_midnight.date():
                logger.info(f"🔄 Resuming existing run: {run.run_id}")
                logger.info(f"   📊 Status: {run.status}")
                logger.info(f"   📊 Progress: {run.progress * 100:.1f}%")
                logger.info(f"   📁 Files: {len(run.generated_files)}")
                logger.info(f"   💻 Commands: {len(run.executed_commands)}")

                # Log the resumption
                log_command(run, "run resumed", f"Resumed processing from {run.current_stage}")

                return run
            else:
                logger.info(
                    f"🆕 Found existing run {run.run_id} but local midnight {run.lighting_schedule.local_midnight.date()}"
                )
                logger.info(f"   doesn't match current local midnight {schedule.local_midnight.date()}")
                logger.info("🆕 Creating new run instead...")
        else:
            logger.error(f"❌ Failed to load run from {latest_run_dir}")
            logger.info("🆕 Creating new run instead...")

    # Create new run
    logger.info(f"🆕 Creating new run for site: {site.name}")

    run = create_burr_run(site, observation_date)

    logger.info(f"🚀 Created run: {run.run_id}")
    logger.info(f"📁 Output directory: {run.output_directory}")
    logger.info(f"🌙 Moon phase: {run.lighting_schedule.moon_phase:.3f}")

    # Log the creation
    log_command(run, "run started", "New processing run created")

    return run


def get_run(config: Path = LOCAL_APP_CONFIG_OVERRIDE) -> BurrRun:
    """
    Get or create a run for a service.

    Parameters
    ----------
    config : Path
        Path to the configuration file

    Returns
    -------
    BurrRun
        The run object, either loaded from an existing run or newly created
    """
    # Initialize configuration first
    app_config = initialize_config(config)

    # Get current time in site's timezone
    current_utc = datetime.now(ZoneInfo("UTC"))
    try:
        timezone_str = str(app_config.site.get_timezone())
        current_local = current_utc.astimezone(ZoneInfo(timezone_str))
        logger.info(f"🌍 Using timezone {timezone_str} for site {app_config.site.name}")
    except Exception as e:
        current_local = current_utc
        logger.warning(f"⚠️ Could not use timezone for site {app_config.site.name}, using UTC: {str(e)}")

    # Calculate target midnight based on current local time
    if current_local.hour >= 12:
        # After noon - use tonight's midnight (start of next day)
        next_day = current_local + timedelta(days=1)
        target_midnight = next_day.replace(hour=0, minute=0, second=0, microsecond=0)
        logger.info("After local noon, using tonight's midnight")
    else:
        # Before noon - use last night's midnight (start of current day)
        target_midnight = current_local.replace(hour=0, minute=0, second=0, microsecond=0)
        logger.info("Before local noon, using last night's midnight")

    logger.info(f"🎯 Target midnight: {target_midnight}")

    # Look for existing runs
    existing_runs = find_existing_runs()

    if existing_runs:
        logger.info(f"Found {len(existing_runs)} existing runs")
        for run_dir in existing_runs:
            logger.info(f"  - {run_dir}")

        # Try to find a run matching our target midnight
        for run_dir in existing_runs:
            logger.info(f"Checking run in: {run_dir}")
            run = load_run_state(run_dir, new_config=app_config)

            if run:
                logger.info(f"Run local midnight: {run.lighting_schedule.local_midnight}")
                if run.lighting_schedule.local_midnight.date() == target_midnight.date():
                    logger.info(f"Found matching run: {run.run_id}")
                    return run
                else:
                    logger.info(f"Run {run.run_id} does not match target midnight {target_midnight.date()}")
            else:
                logger.warning(f"Failed to load run from {run_dir}")

        logger.info("No matching run found for target midnight")

    # If no matching run found or loading failed, create new run
    logger.info("Creating new run...")
    # Use target_midnight as the observation date to ensure correct schedule
    run = get_or_create_run(app_config.site, target_midnight, verbose=True)
    logger.info(f"Created new run with ID: {run.run_id}")
    logger.info(f"New run observation date: {run.observation_date}")
    logger.info(f"New run local midnight: {run.lighting_schedule.local_midnight}")
    logger.info(
        f"New run schedule: {run.lighting_schedule.schedule_start_utc} to {run.lighting_schedule.schedule_end_utc}"
    )
    return run
