from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from nats.aio.client import Client as NATS
import json

import burr
from burr.core.config import AppConfig, get_config
from burr.models.site import LightingSchedule
from burr.models.analysis import AnalysisResults
from loguru import logger


def sanitize_site_prefix(site) -> str:
    """Filesystem-safe prefix derived from site.name, falling back to BURR."""
    if not site.name:
        return "BURR"
    sanitized = site.name.replace(" ", "_").replace("-", "_")
    sanitized = "".join(c for c in sanitized if c.isalnum() or c == "_")
    return sanitized or "BURR"


def generate_run_id(site, observation_date: datetime) -> str:
    """Generate run ID based on site and local midnight date."""
    site_name = sanitize_site_prefix(site)
    lighting_schedule = LightingSchedule.calculate_twilights(site, observation_date)
    date_str = lighting_schedule.local_midnight.strftime("%Y%m%d")
    return f"{site_name}_{date_str}"


def generate_run_id_from_schedule(site, lighting_schedule: LightingSchedule) -> str:
    """Generate run ID using an already calculated lighting schedule."""
    site_name = sanitize_site_prefix(site)
    date_str = lighting_schedule.local_midnight.strftime("%Y%m%d")
    return f"{site_name}_{date_str}"


def create_output_directory(site, observation_date: datetime) -> Path:
    """Create output directory based on site and observation date."""
    config = get_config()
    run_id = generate_run_id(site, observation_date)
    output_dir = Path(config.runtime.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create standard subdirectories
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "calibration").mkdir(exist_ok=True)
    (output_dir / "processing").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)

    return output_dir


def create_output_directory_from_run_id(run_id: str) -> Path:
    """Create output directory based on run ID."""
    config = get_config()
    output_dir = Path(config.runtime.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create standard subdirectories
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "calibration").mkdir(exist_ok=True)
    (output_dir / "processing").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)

    return output_dir


class BurrRun(BaseModel):
    """
    Represents a complete burr observation run with all metadata and state.
    """

    # Core identification
    version: str = Field(default=burr.__version__, description="Burr version")
    run_id: str = Field(description="Unique run identifier")
    observation_date: datetime = Field(description="Date of observation")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(ZoneInfo("UTC")), description="Run creation timestamp"
    )

    # Configuration and site
    config: AppConfig = Field(description="Application configuration")
    lighting_schedule: LightingSchedule = Field(description="Calculated lighting schedule")

    # Run state management
    status: str = Field(
        default="initialized", description="Run status: initialized, running, completed, failed, paused"
    )
    current_stage: str = Field(
        default="setup",
        description="Current processing stage: setup, operations, shutdown",
    )
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Progress from 0.0 to 1.0")

    # Output management
    output_directory: Path = Field(description="Run output directory path")
    generated_files: List[str] = Field(
        default_factory=list, description="List of generated file paths relative to output_directory"
    )

    # Command history and logging
    executed_commands: List[Dict[str, Any]] = Field(default_factory=list, description="Command history with timestamps")

    # Error handling
    last_error: Optional[str] = Field(default=None, description="Last error message")
    error_count: int = Field(default=0, ge=0, description="Total error count")

    # Analysis results
    results: AnalysisResults = Field(default_factory=AnalysisResults, description="Analysis results")

    # Add config history tracking
    config_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="History of configs used in this run"
    )

    # NATS client for real-time updates (not included in serialization)
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _nc: Optional[NATS] = None
    _connected: bool = False

    async def connect_nats(self, nats_url: str = "nats://nats:4222"):
        """Connect to NATS server and setup subscriptions if enabled"""
        if self._connected:
            return

        try:
            self._nc = NATS()
            await self._nc.connect(nats_url)

            # Subscribe to result updates
            await self._nc.subscribe(
                "burr.results.updates",
                cb=self._handle_update,
                queue="burr_results",  # Load balance if multiple subscribers
            )

            self._connected = True
            logger.info("Connected to NATS server for real-time updates")

        except Exception as e:
            logger.error(f"Failed to connect to NATS: {e}")
            raise

    async def disconnect_nats(self):
        """Disconnect from NATS server if connected"""
        if self._nc:
            await self._nc.drain()
            self._connected = False
            logger.info("Disconnected from NATS server")

    async def _handle_update(self, msg):
        """Handle incoming result updates"""
        try:
            data = json.loads(msg.data.decode())
            key = data["key"]
            value = data["value"]
            timestamp = datetime.fromisoformat(data["timestamp"])

            # Update local state
            self.results.set_value(key, value, timestamp)

            logger.debug(f"Updated {key} with new value from NATS")

        except Exception as e:
            logger.error(f"Error handling NATS update: {e}")

    async def publish_result(self, key: str, value: Any):
        """Publish a new result to all subscribers if NATS is enabled"""
        if not self._connected:
            logger.debug("NATS not connected, skipping result publication")
            # Still update local state even if NATS isn't connected
            self.results.set_value(key, value)
            return

        try:
            timestamp = datetime.now(ZoneInfo("UTC"))
            data = {"key": key, "value": value, "timestamp": timestamp.isoformat()}

            # Publish to NATS
            await self._nc.publish("burr.results.updates", json.dumps(data).encode())

            # Update local state
            self.results.set_value(key, value, timestamp)

            logger.debug(f"Published new value for {key} to NATS")

        except Exception as e:
            logger.error(f"Error publishing result to NATS: {e}")
            raise

    @classmethod
    def from_config(
        cls,
        config: AppConfig | None = None,
        observation_date: datetime | None = None,
        run_id: str | None = None,
        output_directory: Path | None = None,
    ) -> "BurrRun":
        """
        Create a BurrRun from configuration.

        Parameters
        ----------
        config : AppConfig, optional
            Application config, uses global config if None
        observation_date : datetime, optional
            Observation date, uses current time if None
        run_id : str, optional
            Run identifier, auto-generated if None
        output_directory : Path, optional
            Output directory, auto-generated if None

        Returns
        -------
        BurrRun
            Initialized run object
        """
        if config is None:
            config = get_config()
        if observation_date is None:
            observation_date = datetime.now(ZoneInfo("UTC"))

        # Calculate lighting schedule once
        lighting = LightingSchedule.calculate_twilights(config.site, observation_date)

        # Generate run ID using the already calculated lighting schedule
        if run_id is None:
            run_id = generate_run_id_from_schedule(config.site, lighting)

        # Generate output directory using the run_id (no need to recalculate lighting)
        if output_directory is None:
            output_directory = create_output_directory_from_run_id(run_id)

        # Create initial run object
        run = cls(
            run_id=run_id,
            observation_date=observation_date,
            config=config,
            lighting_schedule=lighting,
            output_directory=output_directory,
            results=AnalysisResults(),
        )

        # Add config to history
        run.config_history.append(
            {
                "timestamp": datetime.now(ZoneInfo("UTC")).isoformat(),
                "config": config.model_dump(),
                "source": "initial" if not run.config_history else "update",
            }
        )

        return run

    def update_config(self, new_config: AppConfig) -> None:
        """
        Update the run's configuration, preserving history.

        Parameters
        ----------
        new_config : AppConfig
            New configuration to apply
        """
        # Add to history before updating
        self.config_history.append(
            {
                "timestamp": datetime.now(ZoneInfo("UTC")).isoformat(),
                "config": self.config.model_dump(),
                "source": "previous",
            }
        )

        # Update current config
        self.config = new_config

        # Add new config to history
        self.config_history.append(
            {
                "timestamp": datetime.now(ZoneInfo("UTC")).isoformat(),
                "config": new_config.model_dump(),
                "source": "update",
            }
        )

        logger.info(f"Updated run configuration (history size: {len(self.config_history)})")

    @property
    def site(self):
        """Convenience property to access site from config."""
        return self.config.site

    def model_dump_for_json(self) -> Dict[str, Any]:
        """
        Dump model to dict suitable for JSON serialization.

        Returns
        -------
        Dict[str, Any]
            Serializable dictionary
        """
        data = self.model_dump(exclude={"_nc", "_connected"})

        # Convert Path to string
        data["output_directory"] = str(self.output_directory)

        # Convert datetime objects to ISO strings
        data["observation_date"] = self.observation_date.isoformat()
        data["created_at"] = self.created_at.isoformat()

        return data

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "BurrRun":
        """
        Create BurrRun from JSON-serialized dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary from JSON

        Returns
        -------
        BurrRun
            Reconstructed run object
        """
        # Convert ISO strings back to datetime objects
        if isinstance(data["observation_date"], str):
            data["observation_date"] = datetime.fromisoformat(data["observation_date"])
        if isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])

        # Convert string back to Path
        if isinstance(data["output_directory"], str):
            data["output_directory"] = Path(data["output_directory"])

        return cls.model_validate(data)

    def log_command(self, command: str, result: Any = None, metadata: Dict[str, Any] = None, error: str = None) -> None:
        """
        Log a command execution to the run history.

        Parameters
        ----------
        command : str
            Command that was executed
        result : Any, optional
            Command result/output
        metadata : Dict[str, Any], optional
            Additional metadata to store with the command
        error : str, optional
            Error message if command failed
        """
        command_entry = {
            "timestamp": datetime.now(ZoneInfo("UTC")).isoformat(),
            "command": command,
            "result": str(result) if result is not None else None,
            "metadata": metadata,
            "error": error,
            "stage": self.current_stage,
        }

        self.executed_commands.append(command_entry)

        if error:
            self.error_count += 1
            self.last_error = error
            logger.error(f"Command failed in {self.run_id}: {command} - {error}")

        # Save updated state
        from burr.run import save_run_state

        save_run_state(self)
