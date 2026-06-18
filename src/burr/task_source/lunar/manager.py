"""
Lunar background task source manager.
"""

from datetime import datetime
from typing import Set
from zoneinfo import ZoneInfo
import random
import numpy as np

from loguru import logger

from burr.models.observation import (
    ExposureSpec,
    LunarTarget,
    ObservationRequest,
)
from burr.models.run import BurrRun
from burr.models.site import LightingCondition
from burr.models.tracking import (
    Rates,
    RatesFromStreakDistribution,
    TrackingMode,
)


class LunarBackgroundManager:
    """
    Manager for lunar background measurements.

    This class handles taking measurements at different angular separations from the moon
    to characterize lunar background and its effects on observations.

    Implements the TaskSource protocol.
    """

    def __init__(
        self,
        run: BurrRun,
    ):
        """
        Initialize the lunar background manager.

        Parameters
        ----------
        run : BurrRun
            The current observing run
        """
        self.run = run
        self.config = run.config.schedule.lunar_background

        # Load configuration
        self.min_moon_altitude = self.config.minimum_moon_altitude_degrees
        self.max_moon_altitude = self.config.maximum_moon_altitude_degrees
        self.min_moon_phase = self.config.minimum_moon_phase
        self.moon_separations = self.config.moon_separation_degrees
        self.min_exposure = self.config.minimum_exposure_seconds
        self.max_exposure = self.config.maximum_exposure_seconds
        self.num_exposures = self.config.n_exposures
        # Site-wide horizon floor for the measurement position itself.
        # Lunar has no per-source minimum_altitude_degrees because the
        # pointing is derived from moon+separation, so the only floor
        # available is the site one.
        self.min_measurement_altitude = run.config.site.minimum_altitude_degrees

        # Initialize completion tracking
        self.completed_separations = self._get_completed_separations_from_history()
        self.total_separations = len(self.moon_separations)

        logger.info("Initialized LunarBackgroundManager")
        logger.info(f"  Target separations: {self.moon_separations}")
        logger.info(f"  Completed separations: {sorted(self.completed_separations)}")
        logger.info(f"  Remaining separations: {sorted(self.get_remaining_separations())}")
        logger.info(f"  Progress: {len(self.completed_separations)}/{self.total_separations}")

        run_moon_phase = self.run.lighting_schedule.moon_phase
        if run_moon_phase < self.min_moon_phase:
            logger.info(
                f"  Moon phase {run_moon_phase:.3f} is below minimum {self.min_moon_phase:.3f} — "
                "lunar background will be skipped this run"
            )

    # ------------------------------------------------------------------
    # TaskSource protocol: properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Short identifier for this task source."""
        return "lunar_background"

    @property
    def suitable_conditions(self) -> list[LightingCondition]:
        """Lighting conditions under which this source can generate tasks."""
        return [LightingCondition(w) for w in self.config.windows]

    # ------------------------------------------------------------------
    # TaskSource protocol: lifecycle callbacks
    # ------------------------------------------------------------------

    def on_request_completed(
        self, request: ObservationRequest, duration_seconds: float
    ) -> None:
        """Called when an observation request completes successfully.

        Marks the separation from the request metadata as complete.
        """
        separation = request.metadata.get("separation_deg")
        if separation is not None:
            self.mark_separation_complete(float(separation))
            logger.info(
                f"Lunar background request {request.task_id} completed in "
                f"{duration_seconds:.1f}s (separation={separation})"
            )
        else:
            logger.warning(
                f"Completed request {request.task_id} has no separation_deg in metadata"
            )

    def on_request_failed(
        self, request: ObservationRequest, error: str
    ) -> None:
        """Called when an observation request fails."""
        separation = request.metadata.get("separation_deg")
        logger.warning(
            f"Lunar background request {request.task_id} failed "
            f"(separation={separation}): {error}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_completed_separations_from_history(self) -> Set[float]:
        """
        Reconstruct completed separations from run command history.
        """
        completed = set()
        separation_commands = 0
        for cmd in self.run.executed_commands:
            if cmd["command"] == "lunar_background_separation_completed":
                separation_commands += 1
                if "metadata" in cmd and "moon_separation_deg" in cmd["metadata"]:
                    separation = float(cmd["metadata"]["moon_separation_deg"])
                    completed.add(separation)
                    logger.debug(f"Found completed separation from history: {separation}")

        logger.debug(f"Processed {separation_commands} separation completion commands from history")
        logger.debug(f"Total completed separations from history: {sorted(completed)}")
        return completed

    def is_collection_complete(self) -> bool:
        """Return True when no more lunar measurements will be attempted this run.

        True when either all target separations have been measured, or the
        run's moon phase is below the configured minimum (in which case no
        measurement can succeed tonight regardless of separation, since
        generate_request gates on the same threshold). The second case
        prevents this one-shot source from staying at top priority forever
        and starving every other task source — TaskSlot.is_suitable has no
        other signal that generate_request will always return None tonight.
        """
        if len(self.completed_separations) >= self.total_separations:
            return True
        if self.run.lighting_schedule.moon_phase < self.min_moon_phase:
            return True
        return False

    def get_remaining_separations(self) -> list[float]:
        """Get list of separations that haven't been completed yet."""
        return [sep for sep in self.moon_separations if sep not in self.completed_separations]

    def mark_separation_complete(self, separation: float) -> None:
        """Mark a separation as completed and log it."""
        logger.info(f"mark_separation_complete called with separation: {separation}")
        logger.info(f"Current completed separations before: {sorted(self.completed_separations)}")

        if separation not in self.completed_separations:
            self.completed_separations.add(separation)

            # Log completion to run history
            self.run.log_command(
                "lunar_background_separation_completed",
                f"Completed lunar background at {separation} separation ({len(self.completed_separations)}/{self.total_separations})",
                {
                    "moon_separation_deg": separation,
                    "completed_count": len(self.completed_separations),
                    "total_count": self.total_separations,
                },
            )

            logger.info(f"Completed lunar background at {separation} separation")
            logger.info(
                f"Progress: {len(self.completed_separations)}/{self.total_separations} separations completed"
            )
            logger.info(f"Current completed separations after: {sorted(self.completed_separations)}")
        else:
            logger.warning(f"Separation {separation} was already marked as complete")

    def _calculate_measurement_position(
        self, moon_alt: float, moon_az: float, separation_deg: float
    ) -> tuple[float, float]:
        """
        Calculate the position to take a measurement at a given angular separation from the moon.

        This function handles the special case near zenith by choosing a direction that
        maintains the true angular separation.

        Parameters
        ----------
        moon_alt : float
            Moon's altitude in degrees
        moon_az : float
            Moon's azimuth in degrees
        separation_deg : float
            Desired angular separation in degrees

        Returns
        -------
        tuple[float, float]
            (altitude, azimuth) in degrees for the measurement position
        """
        logger.info(
            f"Calculating measurement position: Moon at Alt={moon_alt:.2f}, Az={moon_az:.2f}, separation={separation_deg:.2f}"
        )

        # Convert to radians for calculations
        moon_alt_rad = np.radians(moon_alt)
        moon_az_rad = np.radians(moon_az)
        sep_rad = np.radians(separation_deg)

        # For small separations (< 5 degrees), use a simpler approach
        # Just move in a random direction from the moon
        if separation_deg < 5.0:
            # Choose a random direction (0-360 degrees)
            direction_angle = np.random.uniform(0, 360)
            direction_rad = np.radians(direction_angle)

            # For small angles, we can use the small angle approximation
            # Move in altitude and azimuth to achieve the desired separation
            delta_alt = separation_deg * np.cos(direction_rad)
            delta_az = separation_deg * np.sin(direction_rad) / np.cos(moon_alt_rad)  # Correct for latitude compression

            new_alt = moon_alt + delta_alt
            new_az = (moon_az + delta_az) % 360

            # Ensure we don't go below horizon or above zenith
            new_alt = max(0, min(90, new_alt))

            logger.info(
                f"Small separation calculation: direction={direction_angle:.1f}, delta_alt={delta_alt:.3f}, delta_az={delta_az:.3f}"
            )
            logger.info(f"Result: Alt={new_alt:.2f}, Az={new_az:.2f}")

            return new_alt, new_az

        # If moon is near zenith (> 80 degrees), use altitude-only movement
        elif moon_alt > 80:
            # Move primarily in altitude to maintain true angular separation
            # Calculate how much we need to move in altitude to achieve the separation
            new_alt = moon_alt - separation_deg
            # Keep the same azimuth
            new_az = moon_az

            logger.info(f"Near-zenith calculation: moved {separation_deg} down in altitude")
        else:
            # Use spherical trigonometry to calculate new position
            # This maintains true angular separation even at high altitudes

            # Choose a random direction around the moon
            direction_angle = np.random.uniform(0, 360)
            direction_rad = np.radians(direction_angle)

            # Use the spherical law of cosines to find the new position
            # cos(c) = cos(a)cos(b) + sin(a)sin(b)cos(C)
            # where a = 90 - moon_alt, b = 90 - new_alt, c = separation, C = azimuth difference

            # We want to find new_alt and azimuth difference
            # Rearranging: cos(90 - new_alt) = cos(90 - moon_alt)cos(separation) + sin(90 - moon_alt)sin(separation)cos(direction)
            # sin(new_alt) = sin(moon_alt)cos(separation) + cos(moon_alt)sin(separation)cos(direction)

            sin_new_alt = np.sin(moon_alt_rad) * np.cos(sep_rad) + np.cos(moon_alt_rad) * np.sin(sep_rad) * np.cos(
                direction_rad
            )
            new_alt_rad = np.arcsin(np.clip(sin_new_alt, -1, 1))
            new_alt = np.degrees(new_alt_rad)

            # Calculate azimuth difference using spherical law of sines
            # sin(A)/sin(a) = sin(B)/sin(b)
            # sin(daz)/sin(separation) = sin(direction)/sin(90 - new_alt)
            if np.abs(np.cos(new_alt_rad)) > 1e-10:  # Avoid division by zero
                sin_daz = np.sin(sep_rad) * np.sin(direction_rad) / np.cos(new_alt_rad)
                sin_daz = np.clip(sin_daz, -1, 1)
                daz_rad = np.arcsin(sin_daz)
                new_az = (moon_az + np.degrees(daz_rad)) % 360
            else:
                new_az = moon_az

            logger.info(
                f"Spherical trigonometry: direction={direction_angle:.1f}, new_alt={new_alt:.2f}, new_az={new_az:.2f}"
            )

        # Verify the separation is correct
        actual_separation = self._calculate_angular_separation(moon_alt, moon_az, new_alt, new_az)
        logger.info(f"Verification: requested={separation_deg:.3f}, actual={actual_separation:.3f}")

        return new_alt, new_az

    def _calculate_angular_separation(self, alt1: float, az1: float, alt2: float, az2: float) -> float:
        """Calculate the angular separation between two points in alt/az coordinates."""
        # Convert to radians
        alt1_rad, az1_rad = np.radians(alt1), np.radians(az1)
        alt2_rad, az2_rad = np.radians(alt2), np.radians(az2)

        # Use the spherical law of cosines
        cos_sep = np.sin(alt1_rad) * np.sin(alt2_rad) + np.cos(alt1_rad) * np.cos(alt2_rad) * np.cos(az2_rad - az1_rad)
        cos_sep = np.clip(cos_sep, -1, 1)
        separation = np.degrees(np.arccos(cos_sep))

        return separation

    # ------------------------------------------------------------------
    # TaskSource protocol: generate_request
    # ------------------------------------------------------------------

    def generate_request(
        self, task_id: str, binning: int = 1
    ) -> ObservationRequest | None:
        """
        Generate an observation request for lunar background measurements.

        Parameters
        ----------
        task_id : str
            Unique identifier for this task.
        binning : int
            Current camera binning factor.

        Returns
        -------
        ObservationRequest | None
            A request if one is available, None otherwise.
        """
        try:
            # Check if collection is complete
            if self.is_collection_complete():
                logger.info("Lunar background collection is complete")
                return None

            # Get remaining separations
            remaining_separations = self.get_remaining_separations()
            if not remaining_separations:
                logger.info("No remaining separations to complete")
                return None

            now = datetime.now(ZoneInfo("UTC"))

            # Calculate current moon position using astropy
            try:
                from astropy.coordinates import EarthLocation, get_body, AltAz
                from astropy.time import Time
                import astropy.units as u

                # Create location object
                location = EarthLocation(
                    lat=self.run.site.latitude * u.deg,
                    lon=self.run.site.longitude * u.deg,
                    height=(self.run.site.altitude_km or 0) * u.km,
                )

                # Calculate moon position
                astropy_time = Time(now.replace(tzinfo=None))  # Strip timezone for astropy
                moon_coords = get_body("moon", astropy_time)
                moon_altaz = moon_coords.transform_to(AltAz(obstime=astropy_time, location=location))

                moon_alt = float(moon_altaz.alt.deg)
                moon_az = float(moon_altaz.az.deg)

            except ImportError:
                logger.error("astropy is required for moon position calculations")
                return None
            except Exception as e:
                logger.error(f"Error calculating moon position: {e}")
                return None

            # Get moon phase from lighting schedule
            moon_phase = self.run.lighting_schedule.moon_phase

            # Check if conditions are suitable
            if moon_alt < self.min_moon_altitude or moon_alt > self.max_moon_altitude:
                logger.debug(
                    f"Moon altitude {moon_alt:.1f} outside limits [{self.min_moon_altitude:.1f}, {self.max_moon_altitude:.1f}]"
                )
                return None

            if moon_phase < self.min_moon_phase:
                logger.debug(f"Moon phase {moon_phase:.2f} below minimum {self.min_moon_phase:.2f}")
                return None

            # Choose a random separation from the remaining list
            separation = random.choice(remaining_separations)

            logger.info(f"Selected separation: {separation} from remaining: {sorted(remaining_separations)}")
            logger.info(f"Completed separations: {sorted(self.completed_separations)}")

            # Calculate measurement position
            alt, az = self._calculate_measurement_position(moon_alt, moon_az, separation)

            # The measurement direction is randomized within the great-circle
            # ring at `separation` from the moon, so for sufficient separations
            # it can land below the site's horizon floor. Reject this attempt
            # and let the next scheduler cycle re-pick a (separation, direction)
            # combination — randomness across calls should land inside the
            # allowed region soon enough.
            if alt < self.min_measurement_altitude:
                logger.debug(
                    f"Lunar measurement position alt={alt:.1f}° below site floor "
                    f"{self.min_measurement_altitude:.1f}° (moon at {moon_alt:.1f}°, "
                    f"separation {separation}°) — skipping this cycle"
                )
                return None

            # Generate exposure times
            exposure_times = np.linspace(
                self.min_exposure,
                self.max_exposure,
                self.num_exposures,
            )

            # Set up streak parameters if enabled
            if self.run.config.schedule.streaks.collect:
                # Get base pixel scale and adjust for current binning
                base_pixel_scale = (
                    self.run.config.hardware.telescope.ifov_no_binning_arcsec
                    if self.run.config.hardware.telescope
                    else 0.5
                )
                pixel_scale = base_pixel_scale * binning

                rates = RatesFromStreakDistribution(
                    streak_length_pixels_min=self.run.config.schedule.streaks.minimum_length_pixel,
                    streak_length_pixels_max=self.run.config.schedule.streaks.maximum_length_pixel,
                    streak_orientation_min_deg=self.run.config.schedule.streaks.minimum_orientation_deg,
                    streak_orientation_max_deg=self.run.config.schedule.streaks.maximum_orientation_deg,
                    plate_scale_arcsec_per_pixel=pixel_scale,
                )
                # For streaks, one SIDEREAL then N RATE frames
                exposure_map_specs = [ExposureSpec(tracking_mode=TrackingMode.SIDEREAL)]
                for _ in range(self.run.config.schedule.streaks.streaks_per_collect):
                    exposure_map_specs.append(ExposureSpec(tracking_mode=TrackingMode.RATE))
            else:
                rates = Rates(
                    right_ascension_rate_arcsec_s=0.0,
                    declination_rate_arcsec_s=0.0,
                )
                exposure_map_specs = [ExposureSpec(tracking_mode=TrackingMode.SIDEREAL)]

            # Build the ObservationRequest
            request = ObservationRequest(
                task_id=task_id,
                source_name="lunar_background",
                target=LunarTarget(
                    altitude_deg=alt,
                    azimuth_deg=az,
                    moon_altitude_deg=moon_alt,
                    moon_azimuth_deg=moon_az,
                    moon_separation_deg=separation,
                    moon_phase=moon_phase,
                ),
                exposure_seconds=list(exposure_times),
                exposure_map=exposure_map_specs,
                rates=rates,
                metadata={
                    "separation_deg": separation,
                    "remaining": sorted(remaining_separations),
                },
            )

            logger.info(
                f"Generated lunar background request at {separation} separation "
                f"with {len(exposure_times)} exposure times"
            )
            logger.info(f"Remaining separations: {sorted(remaining_separations)}")

            return request

        except Exception as e:
            logger.error(f"Error generating observation request: {e}")
            return None

