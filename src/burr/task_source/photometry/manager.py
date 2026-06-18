"""
Photometric standards task source manager.

Implements the TaskSource protocol for scheduling and managing standard star
observations during astronomical twilight and night periods.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, NamedTuple
from zoneinfo import ZoneInfo

import numpy as np
from loguru import logger

from burr.core.constants import LANDOLT_STARS_PATH
from burr.models.observation import (
    ExposureSpec,
    ObservationRequest,
    RADecTarget,
)
from burr.models.run import BurrRun
from burr.models.site import LightingCondition
from burr.models.tracking import (
    Rates,
    RatesFromStreakDistribution,
    TrackingMode,
)
from burr.task_source.photometry.catalog import load_catalog


class Star(NamedTuple):
    """Representation of a standard star."""

    name: str
    ra_deg: float
    dec_deg: float
    vmag: float


class PhotometricStandardsManager:
    """
    Manager for photometric standard star observations.

    This class handles the scheduling and management of standard star observations
    during astronomical twilight and night periods. It aims to observe a diverse
    set of standard stars across different sky positions and exposure times.

    Implements the ``TaskSource`` protocol.
    """

    def __init__(
        self,
        run: BurrRun,
    ):
        """
        Initialize the photometric standards manager.

        Parameters
        ----------
        run : BurrRun
            The current observing run
        """
        self.run = run
        self.config = run.config.schedule.photometric_standards

        self.min_altitude = max(
            self.config.minimum_altitude_degrees,
            run.config.site.minimum_altitude_degrees,
        )
        self.max_altitude = self.config.maximum_altitude_degrees
        self.min_exposure = self.config.minimum_exposure_seconds
        self.max_exposure = self.config.maximum_exposure_seconds
        self.num_exposures = self.config.n_exposures
        # Load and prepare the Landolt catalog
        self.catalog = load_catalog(Path(LANDOLT_STARS_PATH))
        logger.info(f"Loaded {len(self.catalog)} standard stars from catalog")

    # ------------------------------------------------------------------
    # TaskSource protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Short identifier for this task source."""
        return "photometric_standards"

    @property
    def suitable_conditions(self) -> list[LightingCondition]:
        """Lighting conditions under which this source can generate tasks."""
        return [
            LightingCondition(window.lower())
            for window in self.config.windows
        ]

    def generate_request(
        self, task_id: str, binning: int = 1
    ) -> ObservationRequest | None:
        """Generate the next observation request for a photometric standard star.

        Parameters
        ----------
        task_id : str
            Unique identifier for this task.
        binning : int
            Current camera binning factor.

        Returns
        -------
        ObservationRequest | None
            A request if a suitable star is found, None otherwise.
        """
        now = datetime.now(ZoneInfo("UTC"))

        exposure_times = np.linspace(
            self.min_exposure,
            self.max_exposure,
            self.num_exposures,
        )

        # Convert catalog to list for random selection
        catalog_list = list(self.catalog)

        # Select a random star from the catalog
        star = None
        while star is None and catalog_list:
            star = random.choice(catalog_list)  # Use Python's random instead of numpy
            try:
                position = self.run.site.calculate_altitude(ra_deg=star.ra_deg, dec_deg=star.dec_deg, time=now)
                if position < self.min_altitude or position > self.max_altitude:
                    logger.debug(
                        f"Star {star.name} outside altitude limits ({position:.1f} not in [{self.min_altitude:.1f}, {self.max_altitude:.1f}])"
                    )
                    catalog_list.remove(star)  # Remove this star from consideration
                    star = None
            except ValueError as e:
                logger.warning(f"Error calculating altitude for star {star.name}: {e}")
                catalog_list.remove(star)  # Remove star with invalid coordinates
                star = None

        if star is None:
            logger.warning("No suitable stars found within altitude limits")
            return None

        if self.run.config.schedule.streaks.collect:
            # Get base pixel scale and adjust for current binning
            base_pixel_scale = (
                self.run.config.hardware.telescope.ifov_no_binning_arcsec if self.run.config.hardware.telescope else 0.5
            )
            pixel_scale = base_pixel_scale * binning

            rates: Rates | RatesFromStreakDistribution = RatesFromStreakDistribution(
                streak_length_pixels_min=self.run.config.schedule.streaks.minimum_length_pixel,
                streak_length_pixels_max=self.run.config.schedule.streaks.maximum_length_pixel,
                streak_orientation_min_deg=self.run.config.schedule.streaks.minimum_orientation_deg,
                streak_orientation_max_deg=self.run.config.schedule.streaks.maximum_orientation_deg,
                plate_scale_arcsec_per_pixel=pixel_scale,
            )
            # For streaks, alternate between SIDEREAL and RATE modes
            exposure_map_specs = [ExposureSpec(tracking_mode=TrackingMode.SIDEREAL)]
            for _ in range(self.run.config.schedule.streaks.streaks_per_collect):
                exposure_map_specs.append(ExposureSpec(tracking_mode=TrackingMode.RATE))
        else:
            rates = Rates(
                right_ascension_rate_arcsec_s=0.0,
                declination_rate_arcsec_s=0.0,
            )
            exposure_map_specs = [ExposureSpec(tracking_mode=TrackingMode.SIDEREAL)]

        request = ObservationRequest(
            task_id=task_id,
            source_name="photometric_standards",
            target=RADecTarget(right_ascension_deg=star.ra_deg, declination_deg=star.dec_deg),
            exposure_seconds=list(exposure_times),
            exposure_map=exposure_map_specs,
            rates=rates,
            metadata={"star_name": star.name, "vmag": star.vmag},
        )

        logger.info(
            f"Generated observation request for star {star.name} "
            f"(vmag={star.vmag:.1f}) with {len(exposure_times)} exposure times"
        )
        return request

    def is_collection_complete(self) -> bool:
        """Check if this source has completed its collection goals.

        Photometric standards use time_share scheduling and are never
        "complete" -- they run whenever conditions permit.
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
        star_name = request.metadata.get("star_name", "unknown")
        logger.info(
            f"Photometric standard observation completed: {star_name} "
            f"in {duration_seconds:.1f}s"
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
        star_name = request.metadata.get("star_name", "unknown")
        logger.warning(
            f"Photometric standard observation failed for {star_name}: {error}"
        )

    # ------------------------------------------------------------------
    # Existing helpers (unchanged)
    # ------------------------------------------------------------------

    @property
    def plate_scale(self) -> float:
        """
        Get the current plate scale, falling back to default if not available
        or too old
        """
        return self.run.results.get_latest_value(
            "plate_scale",
            max_age=timedelta(hours=12),  # Only use values from last 24h
            default=1.0,  # Default fallback value
        )

    def generate_observation_periods(
        self,
        useful_conditions: List[LightingCondition] = [
            LightingCondition.ASTRONOMICAL_TWILIGHT,
            LightingCondition.NIGHT,
        ],
    ) -> List[tuple[datetime, datetime]]:
        """
        Generate a list of time periods suitable for standard star observations.

        Parameters
        ----------
        useful_conditions : List[LightingCondition]
            List of lighting conditions suitable for standard star observations.
            Defaults to astronomical twilight and night.

        Returns
        -------
        List[tuple[datetime, datetime]]
            List of (start_time, end_time) tuples for observation periods
        """
        # Ensure we have a timezone-aware current time
        now = datetime.now(ZoneInfo("UTC"))
        logger.info(f"Current time: {now}")
        schedule = self.run.lighting_schedule
        logger.info(f"Schedule bounds: {schedule.schedule_start_utc} to {schedule.schedule_end_utc}")

        # Get all time boundaries for the requested conditions
        boundaries = []
        for condition in useful_conditions:
            condition_boundaries = schedule.get_condition_boundaries(condition)
            logger.info(f"Found {len(condition_boundaries)} periods for {condition}:")
            for start, end in condition_boundaries:
                # Ensure timezone awareness for all datetime objects
                if start.tzinfo is None:
                    start = start.replace(tzinfo=ZoneInfo("UTC"))
                if end.tzinfo is None:
                    end = end.replace(tzinfo=ZoneInfo("UTC"))
                logger.info(f"  {start} to {end} ({condition})")
                boundaries.append((start, end, condition))

        # Sort boundaries by start time
        boundaries.sort(key=lambda x: x[0])
        logger.info("All boundaries after sorting:")
        for start, end, condition in boundaries:
            logger.info(f"  {start} to {end} ({condition})")

        # Merge overlapping or adjacent periods
        if not boundaries:
            logger.warning("No boundaries found for any conditions")
            return []

        merged_periods = []
        current_start, current_end, current_condition = boundaries[0]

        for start, end, condition in boundaries[1:]:
            # If this period starts within 1 minute of the previous end,
            # extend the current period
            if (start - current_end) <= timedelta(minutes=1):
                logger.info(f"Extending period to {end} (was {current_end})")
                current_end = max(current_end, end)
            else:
                logger.info(f"Creating new period: {current_start} to {current_end} (gap > 1 min)")
                merged_periods.append((current_start, current_end))
                current_start, current_end = start, end
                current_condition = condition

        # Add the last period
        merged_periods.append((current_start, current_end))
        logger.info(f"Added final period: {current_start} to {current_end}")

        logger.info("Merged periods:")
        for start, end in merged_periods:
            logger.info(f"  {start} to {end}")

        # Filter out periods that are completely in the past
        future_periods = [(start, end) for start, end in merged_periods if end > now]
        logger.info(f"Found {len(future_periods)} future periods out of {len(merged_periods)} total")

        # For periods that span current time, update start time to now
        result = []
        for start, end in future_periods:
            if start < now:
                logger.info(f"Updating period start from {start} to {now} (in past)")
                result.append((now, end))
            else:
                result.append((start, end))

        logger.info("Final periods after filtering:")
        for start, end in result:
            logger.info(f"  {start} to {end}")

        return result

    def select_targets(
        self,
        start_time: datetime,
        end_time: datetime,
        n_targets: int = 100,
    ) -> List[Star]:
        """
        Select suitable standard star targets for a given time period.

        Parameters
        ----------
        start_time : datetime
            Start time for target selection
        end_time : datetime
            End time for target selection
        n_targets : int, optional
            Number of targets to select (default: 100)

        Returns
        -------
        List[Star]
            List of selected standard stars
        """
        # Ensure times are timezone-aware
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=ZoneInfo("UTC"))
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=ZoneInfo("UTC"))

        # Filter catalog for stars that will be above minimum altitude during the period
        mid_time = start_time + (end_time - start_time) / 2
        visible_stars = []
        invalid_coords = 0
        below_altitude = 0
        above_altitude = 0
        error_count = 0

        logger.info(f"Checking {len(self.catalog)} stars for visibility at {mid_time}")

        for star in self.catalog:
            try:
                # Skip stars with invalid coordinates
                if not (-90 <= star.dec_deg <= 90) or not (0 <= star.ra_deg <= 360):
                    logger.debug(
                        f"Skipping star {star.name} with invalid coordinates: RA={star.ra_deg}, Dec={star.dec_deg}"
                    )
                    invalid_coords += 1
                    continue

                alt = self.run.site.calculate_altitude(ra_deg=star.ra_deg, dec_deg=star.dec_deg, time=mid_time)
                if self.min_altitude <= alt <= self.max_altitude:
                    visible_stars.append(star)
                elif alt < self.min_altitude:
                    logger.debug(f"Star {star.name} below minimum altitude: {alt:.1f}° < {self.min_altitude}°")
                    below_altitude += 1
                else:
                    logger.debug(f"Star {star.name} above maximum altitude: {alt:.1f}° > {self.max_altitude}°")
                    above_altitude += 1
            except Exception as e:
                logger.warning(f"Error calculating altitude for star {star.name}: {str(e)}")
                error_count += 1
                continue

        logger.info(
            f"Star filtering results:\n"
            f"  - Total stars: {len(self.catalog)}\n"
            f"  - Invalid coordinates: {invalid_coords}\n"
            f"  - Below min altitude ({self.min_altitude}°): {below_altitude}\n"
            f"  - Above max altitude ({self.max_altitude}°): {above_altitude}\n"
            f"  - Calculation errors: {error_count}\n"
            f"  - Visible stars: {len(visible_stars)}"
        )

        if not visible_stars:
            logger.warning("No suitable standard stars found for the given period")
            return []

        # Sort by magnitude to ensure we get a good range
        visible_stars.sort(key=lambda x: x.vmag)
        mag_range = visible_stars[-1].vmag - visible_stars[0].vmag
        logger.info(
            f"Visible stars magnitude range: {visible_stars[0].vmag:.1f} to {visible_stars[-1].vmag:.1f} ({mag_range:.1f} mag spread)"
        )

        # Try to select stars with good spacing in RA
        selected_stars = []
        target_ra_spacing = 360.0 / n_targets

        # Start with the brightest star
        selected_stars.append(visible_stars[0])
        visible_stars = visible_stars[1:]

        while len(selected_stars) < n_targets and visible_stars:
            # Find the star that best fills a gap in RA
            best_score = float("inf")
            best_star = None

            for star in visible_stars:
                # Calculate minimum angular distance to any selected star
                min_dist = float("inf")
                for selected in selected_stars:
                    dist = abs(star.ra_deg - selected.ra_deg)
                    if dist > 180:  # Handle RA wrap-around
                        dist = 360 - dist
                    min_dist = min(min_dist, dist)

                # Score is how far this deviates from ideal spacing
                score = abs(min_dist - target_ra_spacing)

                if score < best_score:
                    best_score = score
                    best_star = star

            if best_star is not None:
                selected_stars.append(best_star)
                visible_stars.remove(best_star)
            else:
                break

        # Now that we have our selected stars, calculate timing parameters
        period_length = (end_time - start_time).total_seconds()
        if len(selected_stars) > 0:
            time_step = period_length / len(selected_stars)
        else:
            logger.warning("No stars were selected, skipping timing calculations")
            return []

        # Calculate magnitude range for exposure time scaling
        min_vmag = min(star.vmag for star in selected_stars)
        max_vmag = max(star.vmag for star in selected_stars)
        mag_range = max_vmag - min_vmag

        final_stars = []
        for i, star in enumerate(selected_stars):
            # Calculate exposure time based on magnitude
            # Brighter stars get shorter exposures
            if mag_range > 0:
                exposure_fraction = (star.vmag - min_vmag) / mag_range
            else:
                exposure_fraction = 0.5

            exposure_time = self.min_exposure + exposure_fraction * (self.max_exposure - self.min_exposure)

            # Space observations throughout the period
            obs_time = start_time + timedelta(seconds=i * time_step)

            final_stars.append(star)

        return final_stars

    def make_standard_star_offers(self) -> None:
        """
        Create observation offers for standard stars during suitable periods.

        This method:
        1. Identifies suitable observation periods
        2. Selects appropriate standard stars for each period
        3. Creates observation offers with varied exposure times
        """
        logger.info("Starting make_standard_star_offers")

        try:
            # Get observation periods - windows are already validated by Pydantic
            logger.info(f"Config: {self.run.config.schedule.photometric_standards}")
            useful_conditions = self.suitable_conditions
            logger.info(f"Useful conditions: {useful_conditions}")
            periods = self.generate_observation_periods(useful_conditions=useful_conditions)
            logger.info(f"Generated {len(periods)} observation periods")

            # For each period, select targets and create offers
            all_targets = []
            for start_time, end_time in periods:
                # Ensure times are timezone-aware
                if start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=ZoneInfo("UTC"))
                if end_time.tzinfo is None:
                    end_time = end_time.replace(tzinfo=ZoneInfo("UTC"))

                logger.info(f"Processing period {start_time} to {end_time}")
                targets = self.select_targets(start_time, end_time)
                logger.info(f"Selected {len(targets)} targets")
                all_targets.extend(targets)

            logger.info(f"Selected {len(all_targets)} total targets")

        except Exception as e:
            logger.error(f"Error in make_standard_star_offers: {e}")
            raise
