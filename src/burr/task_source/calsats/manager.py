"""
Calibration satellite task source manager.

Implements the TaskSource protocol for scheduling calibration satellite
observations during astronomical twilight and night periods.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

try:
    from loguru import logger as _loguru_logger
except ImportError:  # pragma: no cover
    _loguru_logger = None

from logging import Logger, getLogger

from burr.core.constants import SUPPORTED_CALSATS
from burr.models.observation import ExposureSpec, ObservationRequest, TLETarget
from burr.models.run import BurrRun
from burr.models.site import LightingCondition
from burr.models.tracking import TrackingMode
from burr.utils.spacetrack import TLECatalog, find_tles_that_are_visible
from burr.utils.spacetrack import fresh_tles as _fresh_tles_spacetrack
from burr.utils.spacebook import fresh_tles as _fresh_tles_spacebook


def _fetch_tles(provider: str) -> str:
    """Fetch a fresh TLE file from the configured provider, returning its path."""
    name = (provider or "spacetrack").strip().lower()
    if name == "spacebook":
        return _fresh_tles_spacebook()
    if name == "spacetrack":
        return _fresh_tles_spacetrack()
    raise ValueError(f"Unknown TLE provider {provider!r}; expected 'spacetrack' or 'spacebook'")


class CalibrationSatelliteManager:
    """
    Manager for calibration satellite observations.

    Implements the :class:`~burr.task_source.protocol.TaskSource` protocol.

    This class handles the scheduling and management of calibration satellite
    observations during astronomical twilight and night periods. It aims to
    observe a diverse set of satellites across different sky positions and
    exposure times.
    """

    # -- TaskSource protocol: properties ------------------------------------

    @property
    def name(self) -> str:
        """Short identifier for this task source."""
        return "calsats"

    @property
    def suitable_conditions(self) -> list[LightingCondition]:
        """Lighting conditions under which this source can generate tasks."""
        return [
            LightingCondition(w.lower()) for w in self.config.windows
        ]

    # -- construction -------------------------------------------------------

    def __init__(
        self,
        run: BurrRun,
        logger: Optional[Logger] = None,
    ):
        """
        Initialize the calibration satellite manager.

        Parameters
        ----------
        run : BurrRun
            The current observing run.
        logger : Logger, optional
            Stdlib logger used as fallback when loguru is unavailable.
        """
        self.run = run
        self.config = run.config.schedule.calsats

        # Prefer loguru; fall back to stdlib logger
        if _loguru_logger is not None:
            self.logger = _loguru_logger
        else:
            self.logger = logger or getLogger(__name__)

        self.min_altitude = max(
            self.config.minimum_altitude_degrees,
            run.config.site.minimum_altitude_degrees,
        )
        self.max_altitude = self.config.maximum_altitude_degrees
        self.min_exposure = self.config.minimum_exposure_seconds
        self.max_exposure = self.config.maximum_exposure_seconds

        # Create catalog directory in run output
        self.catalog_dir = self.run.output_directory / "catalogs"
        self.catalog_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TLE catalog
        self.catalog: Optional[TLECatalog] = None
        self.last_catalog_update = None

        # Initialize state from run history
        self.last_observation_time = None
        self.observation_count = 0
        self._restore_state_from_history()

    # -- state management ---------------------------------------------------

    def _restore_state_from_history(self) -> None:
        """Restore manager state from run command history."""
        self.logger.info("Restoring calsat observation state from run history")

        for cmd in self.run.executed_commands:
            if cmd["command"] == "calsat_catalog_updated":
                # Restore catalog update time
                self.last_catalog_update = datetime.fromisoformat(cmd["metadata"]["update_time"])
                self.logger.debug(f"Restored catalog update time: {self.last_catalog_update}")

            elif cmd["command"] == "calsat_observed":
                # Restore observation time and count
                observation_time = datetime.fromisoformat(cmd["metadata"]["observation_time"])
                if self.last_observation_time is None or observation_time > self.last_observation_time:
                    self.last_observation_time = observation_time
                self.observation_count += 1

        self.logger.info(
            f"Restored state: {self.observation_count} observations, last observation at {self.last_observation_time}"
        )

    def _update_catalog(self) -> None:
        """Update the TLE catalog if needed."""
        now = datetime.now(ZoneInfo("UTC"))

        # Update catalog if it's too old or doesn't exist
        if (
            self.catalog is None
            or self.last_catalog_update is None
            or (now - self.last_catalog_update) > timedelta(hours=12)
        ):
            try:
                # Get fresh TLEs from the configured provider
                tle_file = _fetch_tles(self.config.tle_provider)
                self.logger.info(
                    f"Downloaded fresh TLEs from {self.config.tle_provider}: {tle_file}"
                )

                # Load and filter to calsats
                full_catalog = TLECatalog.from_file(tle_file)
                self.catalog = full_catalog.filter_satellites(SUPPORTED_CALSATS)
                self.last_catalog_update = now

                # Save filtered catalog to run directory
                catalog_file = self.catalog_dir / f"calsats_{now.strftime('%Y%m%d_%H%M%S')}.tle"
                self.catalog.to_file(str(catalog_file))
                self.logger.info(f"Saved filtered catalog with {len(self.catalog.tles)} satellites to {catalog_file}")

                # Log catalog update
                self.run.log_command(
                    "calsat_catalog_updated",
                    f"Updated TLE catalog with {len(self.catalog.tles)} satellites",
                    metadata={
                        "update_time": now.isoformat(),
                        "catalog_file": str(catalog_file),
                        "satellite_count": len(self.catalog.tles),
                    },
                )

            except Exception as e:
                self.logger.error(f"Error updating TLE catalog: {e}")
                raise

    # -- status -------------------------------------------------------------

    def get_status_summary(self) -> str:
        """Get a summary of the manager's current state."""
        now = datetime.now(ZoneInfo("UTC"))

        # Format time since last observation
        last_obs_str = "never"
        if self.last_observation_time:
            time_since = now - self.last_observation_time
            if time_since < timedelta(minutes=60):
                last_obs_str = f"{time_since.seconds // 60}m ago"
            else:
                last_obs_str = f"{time_since.seconds // 3600}h ago"

        # Format catalog age
        catalog_age_str = "no catalog"
        if self.last_catalog_update:
            catalog_age = now - self.last_catalog_update
            if catalog_age < timedelta(hours=1):
                catalog_age_str = f"{catalog_age.seconds // 60}m old"
            else:
                catalog_age_str = f"{catalog_age.seconds // 3600}h old"

        return f"{self.observation_count} obs (last: {last_obs_str}), catalog: {catalog_age_str}"

    # -- TaskSource protocol: generate_request ------------------------------

    def generate_request(self, task_id: str, binning: int = 1) -> ObservationRequest | None:
        """Generate the next observation request.

        Parameters
        ----------
        task_id : str
            Unique identifier for this task.
        binning : int
            Current camera binning factor (unused for calsats, kept for
            protocol compatibility).

        Returns
        -------
        ObservationRequest | None
            A request if a suitable satellite is visible, None otherwise.
        """
        try:
            now = datetime.now(ZoneInfo("UTC"))

            # Update catalog if needed
            self._update_catalog()
            if not self.catalog or not self.catalog.tles:
                self.logger.warning("No TLE catalog available")
                return None

            # Find visible satellites
            try:
                visible_tles = find_tles_that_are_visible(
                    list(self.catalog.tles.values()),
                    site=self.run.site,
                    time=now,
                    min_altitude_deg=self.min_altitude,
                )
            except Exception as e:
                self.logger.error(f"Error finding visible satellites: {e}")
                return None

            if not visible_tles:
                self.logger.info("No suitable satellites visible")
                return None

            # Select a random visible satellite
            selected_tle = random.choice(visible_tles)
            self.logger.info(f"Selected satellite {selected_tle.name} (NORAD ID: {selected_tle.norad_id})")

            # Validate TLE format
            try:
                _ = selected_tle.satellite
            except ValueError as e:
                self.logger.error(f"Invalid TLE format for satellite {selected_tle.name}: {e}")
                return None

            # Generate random exposure time
            exposure_time = random.uniform(self.min_exposure, self.max_exposure)
            self.logger.info(f"Selected exposure time: {exposure_time:.2f}s")

            # Create exposure map: RATE RATE RATE SIDEREAL
            exposure_map = [ExposureSpec(tracking_mode=TrackingMode.RATE)] * 3 + [
                ExposureSpec(tracking_mode=TrackingMode.SIDEREAL)
            ]

            # Build the observation request
            request = ObservationRequest(
                task_id=task_id,
                source_name="calsats",
                target=TLETarget(tle_line1=selected_tle.line1, tle_line2=selected_tle.line2),
                exposure_seconds=[exposure_time],
                exposure_map=exposure_map,
                rates=None,  # No rates for TLE tasks - mount handles tracking
                metadata={
                    "satellite_name": selected_tle.name,
                    "norad_id": selected_tle.norad_id,
                },
            )

            # Log the observation
            self.run.log_command(
                "calsat_observed",
                f"Observing satellite {selected_tle.name} with {exposure_time:.2f}s exposure",
                metadata={
                    "observation_time": now.isoformat(),
                    "satellite_name": selected_tle.name,
                    "norad_id": selected_tle.norad_id,
                    "exposure_time": exposure_time,
                    "tracking_modes": [spec.tracking_mode.value for spec in exposure_map],
                },
            )

            # Update state
            self.last_observation_time = now
            self.observation_count += 1

            self.logger.info(f"Generated task for satellite {selected_tle.name}")
            return request

        except Exception as e:
            self.logger.error(f"Error generating observation request: {e}")
            return None

    # -- TaskSource protocol: collection / callbacks ------------------------

    def is_collection_complete(self) -> bool:
        """Check if this source has completed its collection goals.

        Calibration satellites run on an interval schedule and are never
        "complete" -- there are always more passes to observe.
        """
        return False

    def on_request_completed(
        self, request: ObservationRequest, duration_seconds: float
    ) -> None:
        """Called when an observation request completes successfully.

        Parameters
        ----------
        request : ObservationRequest
            The completed request.
        duration_seconds : float
            Wall-clock time the request took.
        """
        sat_name = request.metadata.get("satellite_name", "unknown")
        self.logger.info(
            f"Calsat observation of {sat_name} completed in {duration_seconds:.1f}s"
        )

    def on_request_failed(
        self, request: ObservationRequest, error: str
    ) -> None:
        """Called when an observation request fails.

        Parameters
        ----------
        request : ObservationRequest
            The failed request.
        error : str
            Description of the failure.
        """
        sat_name = request.metadata.get("satellite_name", "unknown")
        self.logger.warning(
            f"Calsat observation of {sat_name} failed: {error}"
        )

