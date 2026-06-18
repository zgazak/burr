"""
Burr astronomical observation run management.

This module provides functionality for creating, managing, and resuming
astronomical observation runs with data persistence and state management.
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any
from loguru import logger

from burr.models.run import BurrRun, sanitize_site_prefix
from burr.models.site import SiteMetadata, LightingSchedule
from burr.core.config import get_config, AppConfig


def create_burr_run(site: SiteMetadata, observation_date: datetime) -> BurrRun:
    """
    Create a new BurrRun object with initialized state.

    Parameters
    ----------
    site : SiteMetadata
        Site information
    observation_date : datetime
        Date of observation

    Returns
    -------
    BurrRun
        Initialized run object
    """
    # Get config from the existing config system
    config = get_config()

    # Create a copy of config with the specified site
    config_dict = config.model_dump()
    config_dict["site"] = site.model_dump()
    modified_config = AppConfig.model_validate(config_dict)

    # Create run using the Pydantic model's factory method
    run = BurrRun.from_config(config=modified_config, observation_date=observation_date)

    # Save initial run state
    save_run_state(run)

    # Generate initial lighting schedule plot if enabled
    if config.plotting.generate_plots:
        schedule_plot_path = run.output_directory / "plots" / f"lighting_schedule.{config.plotting.plot_format}"
        run.lighting_schedule.plot_elevation(site, save_path=str(schedule_plot_path))
        run.generated_files.append(str(schedule_plot_path.relative_to(run.output_directory)))

    logger.info(f"Created new burr run: {run.run_id}")
    return run


def save_run_state(run: BurrRun) -> None:
    """
    Save the current run state to disk for resumability using atomic writes.
    This prevents corruption if the process is interrupted during the write.

    Parameters
    ----------
    run : BurrRun
        Run object to save
    """
    state_file = run.output_directory / "metadata" / "run_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)

    # Use the Pydantic model's JSON serialization method
    state_dict = run.model_dump_for_json()

    # Create a temporary file in the same directory as the target file
    # This ensures atomic move works (must be on same filesystem)
    with tempfile.NamedTemporaryFile(mode="w", dir=state_file.parent, suffix=".tmp", delete=False) as tf:
        # Write to temporary file
        json.dump(state_dict, tf, indent=2, default=str)
        temp_path = tf.name

    try:
        # Atomic move of temporary file to final location
        shutil.move(temp_path, state_file)
        logger.debug(f"Saved run state to {state_file}")
    except Exception as e:
        # Clean up temp file if move fails
        Path(temp_path).unlink(missing_ok=True)
        raise e


def load_run_state(run_dir: Path, new_config: AppConfig | None = None) -> Optional[BurrRun]:
    """
    Load run state from a run directory.

    Parameters
    ----------
    run_dir : Path
        Path to run directory
    new_config : AppConfig, optional
        New configuration to apply to loaded run

    Returns
    -------
    Optional[BurrRun]
        Loaded run object or None if loading fails
    """
    logger.info(f"Loading run state from: {run_dir}")
    state_file = run_dir / "metadata" / "run_state.json"

    if not state_file.exists():
        logger.warning(f"Run state file not found: {state_file}")
        return None

    try:
        with open(state_file, "r") as f:
            data = json.load(f)
            logger.info("Successfully loaded run state JSON")
            logger.info(f"Run ID from state: {data.get('run_id')}")
            logger.info(f"Observation date from state: {data.get('observation_date')}")

        # Create run from loaded data
        run = BurrRun.from_json_dict(data)
        logger.info("Successfully created BurrRun object from state")
        logger.info(f"Run observation date: {run.observation_date}")
        logger.info(f"Run local midnight: {run.lighting_schedule.local_midnight}")

        # Apply new configuration if provided
        if new_config:
            logger.info("Applying new configuration to run")
            run.config = new_config
            # Recompute the lighting schedule from the current site + observation
            # date instead of trusting the deserialized one. The persisted
            # schedule was computed by the same Burr binary that created the
            # run dir — if that startup hit a bug (IERS staleness fallback,
            # an off-by-one in the fallback path, etc.) the bad schedule is
            # baked into run_state.json and silently survives every retry,
            # causing _on_attach to publish an inverted offer window. The
            # config is being re-applied here anyway, so recomputing keeps
            # all derived state in sync with it.
            run.lighting_schedule = LightingSchedule.calculate_twilights(
                new_config.site, run.observation_date
            )
            logger.info(
                f"Recomputed lighting schedule for {new_config.site.name} "
                f"at {run.observation_date}"
            )

        return run

    except Exception as e:
        logger.error(f"Error loading run state: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def find_existing_runs() -> List[Path]:
    """
    Find existing runs in the configured output directory for the current site.

    Filters by `{site_prefix}_` so runs from other sites are not resumed.
    """
    config = get_config()
    output_dir = Path(config.runtime.output_dir)
    site_prefix = sanitize_site_prefix(config.site)
    prefix = f"{site_prefix}_"
    logger.info(f"Looking for existing runs in: {output_dir} (prefix: {prefix})")

    if not output_dir.exists():
        logger.info("Output directory does not exist")
        return []

    run_dirs = []
    for path in output_dir.iterdir():
        if not path.is_dir():
            continue
        if not path.name.startswith(prefix):
            logger.debug(f"Skipping directory {path} - does not match site prefix {prefix}")
            continue
        state_file = path / "metadata" / "run_state.json"
        if state_file.exists():
            logger.info(f"Found potential run directory: {path}")
            run_dirs.append(path)
        else:
            logger.debug(f"Skipping directory {path} - no run_state.json found in metadata/")

    if not run_dirs:
        logger.info("No existing runs found")
        return []

    # Sort by creation time (most recent first)
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    logger.info("Found runs (sorted by modification time, most recent first):")
    for run_dir in run_dirs:
        mod_time = datetime.fromtimestamp(run_dir.stat().st_mtime)
        logger.info(f"  - {run_dir.name} (modified: {mod_time})")

    return run_dirs


def log_command(run: BurrRun, command: str, result: Any = None, error: str = None) -> None:
    """
    Log a command execution to the run history.

    Parameters
    ----------
    run : BurrRun
        Run object to update
    command : str
        Command that was executed
    result : Any, optional
        Command result/output
    error : str, optional
        Error message if command failed
    """
    command_entry = {
        "timestamp": datetime.now().isoformat(),
        "command": command,
        "result": str(result) if result is not None else None,
        "error": error,
        "stage": run.current_stage,
    }

    run.executed_commands.append(command_entry)

    if error:
        run.error_count += 1
        run.last_error = error
        logger.error(f"Command failed in {run.run_id}: {command} - {error}")

    # Save updated state
    save_run_state(run)


def add_generated_file(run: BurrRun, file_path: Path) -> None:
    """
    Add a generated file to the run's file list.

    Parameters
    ----------
    run : BurrRun
        Run object to update
    file_path : Path
        Path to the generated file
    """
    # Store relative to output directory
    try:
        relative_path = file_path.relative_to(run.output_directory)
        run.generated_files.append(str(relative_path))
        save_run_state(run)
        logger.debug(f"Added generated file: {relative_path}")
    except ValueError:
        # File is outside output directory - store absolute path
        run.generated_files.append(str(file_path))
        save_run_state(run)
        logger.warning(f"Generated file outside output directory: {file_path}")


def update_progress(run: BurrRun, stage: str, progress: float, status: str = None) -> None:
    """
    Update run progress and stage.

    Parameters
    ----------
    run : BurrRun
        Run object to update
    stage : str
        Current processing stage
    progress : float
        Progress from 0.0 to 1.0
    status : str, optional
        Optional status update
    """
    run.current_stage = stage
    run.progress = max(0.0, min(1.0, progress))  # Clamp to [0,1]

    if status:
        run.status = status

    save_run_state(run)
    logger.info(f"Progress update for {run.run_id}: {stage} ({progress * 100:.1f}%)")


def cleanup_run(run: BurrRun, keep_outputs: bool = True) -> None:
    """
    Clean up run directory and resources.

    Parameters
    ----------
    run : BurrRun
        Run object to clean up
    keep_outputs : bool, optional
        Whether to keep generated output files (default: True)
    """
    if not keep_outputs:
        if run.output_directory.exists():
            shutil.rmtree(run.output_directory)
            logger.info(f"Cleaned up run directory: {run.output_directory}")
    else:
        logger.info(f"Run outputs preserved in: {run.output_directory}")


__all__ = [
    # Core run management
    "BurrRun",
    "create_burr_run",
    "save_run_state",
    "load_run_state",
    "find_existing_runs",
    # Run operations
    "log_command",
    "add_generated_file",
    "update_progress",
    "cleanup_run",
]
