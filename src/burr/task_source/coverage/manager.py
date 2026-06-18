from datetime import datetime, timedelta
from typing import List, Optional, NamedTuple, Tuple, Dict, Set
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib.pyplot as plt
from logging import getLogger

from burr.models.run import BurrRun
from burr.models.site import LightingCondition
from burr.models.tracking import (
    RatesFromStreakDistribution,
    Rates,
    TrackingMode,
)
from burr.models.observation import (
    ObservationRequest,
    AltAzTarget,
    ExposureSpec,
)
from .alt_sampling import generate_sky_points


class SkyPointing(NamedTuple):
    """Representation of a sky pointing in alt/az coordinates."""

    pixel_id: int
    altitude_deg: float
    azimuth_deg: float
    priority: float = 1.0
    observed: bool = False
    last_observed: Optional[datetime] = None


class SparseMap:
    """Represents a sparse sampling of the sky using Fibonacci sphere method."""

    def __init__(
        self, points_per_map: int, min_altitude: float, max_altitude: float, map_id: int, min_separation: float = 0
    ):
        """
        Initialize a sparse sky map.

        Parameters
        ----------
        points_per_map : int
            Number of points to generate for this map
        min_altitude : float
            Minimum altitude in degrees
        max_altitude : float
            Maximum altitude in degrees
        map_id : int
            Unique identifier for this map
        min_separation : float, optional
            Minimum separation between points in degrees
        """
        self.points_per_map = points_per_map
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.map_id = map_id

        # Generate sky points
        points = generate_sky_points(points_per_map, min_altitude, max_altitude, min_separation=min_separation)

        # Create pointings
        self.pointings = []
        for i, (alt, az) in enumerate(points):
            pointing = SkyPointing(pixel_id=i, altitude_deg=alt, azimuth_deg=az, priority=1.0, observed=False)
            self.pointings.append(pointing)

        # Track which points have been observed
        self.observed_pixels: Set[int] = set()
        self.observation_times: Dict[int, datetime] = {}

    def mark_observed(self, pixel_id: int, observation_time: datetime):
        """Mark a pixel as observed."""
        self.observed_pixels.add(pixel_id)
        self.observation_times[pixel_id] = observation_time

        # Update the pointing
        for i, pointing in enumerate(self.pointings):
            if pointing.pixel_id == pixel_id:
                self.pointings[i] = pointing._replace(observed=True, last_observed=observation_time)
                break

    def get_completion_fraction(self) -> float:
        """Get the fraction of this sparse map that has been observed."""
        return len(self.observed_pixels) / len(self.pointings)

    def is_complete(self) -> bool:
        """Check if this sparse map is fully observed."""
        return len(self.observed_pixels) == len(self.pointings)

    def plot(self, save_path: Optional[str] = None):
        """Plot the current state of the map."""
        # Separate observed and unobserved points
        observed_points = [(p.altitude_deg, p.azimuth_deg) for p in self.pointings if p.observed]
        unobserved_points = [(p.altitude_deg, p.azimuth_deg) for p in self.pointings if not p.observed]

        # Create polar projection
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="polar")

        # Plot unobserved points
        if unobserved_points:
            alts, azs = zip(*unobserved_points)
            r = [90 - alt for alt in alts]  # Convert altitude to zenith distance
            theta = np.radians(azs)  # Convert to radians
            ax.scatter(theta, r, s=40, alpha=0.6, c="blue", label="Unobserved")

        # Plot observed points
        if observed_points:
            alts, azs = zip(*observed_points)
            r = [90 - alt for alt in alts]  # Convert altitude to zenith distance
            theta = np.radians(azs)  # Convert to radians
            ax.scatter(theta, r, s=40, alpha=0.6, c="green", label="Observed")

        # Customize the plot
        ax.set_theta_zero_location("N")  # 0° at North
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_rlim(0, 90)  # Set radial limits (0° to max zenith distance)

        # Set up azimuth ticks every 30 degrees
        az_ticks = np.arange(0, 360, 30)
        ax.set_xticks(np.radians(az_ticks))
        ax.set_xticklabels([f"{int(az)}°" for az in az_ticks])

        # Add altitude circles and labels
        alt_circles = [0, 30, 60]  # Altitudes to mark
        for alt in alt_circles:
            radius = 90 - alt
            # Draw circle at this altitude using polar plot's native circle
            ax.add_artist(
                plt.Circle(
                    (0, 0), radius, fill=False, linestyle=":", alpha=0.3, color="gray", transform=ax.transData._b
                )
            )
            # Add label along the 90° azimuth line (East)
            ax.text(np.radians(90), radius, f"{alt}°", ha="left", va="center", alpha=0.5)

        # Customize grid
        ax.grid(True, alpha=0.3)
        ax.set_rticks([])  # Remove radial ticks
        ax.legend()

        ax.set_title(f"Sky Coverage Map {self.map_id}\n{self.get_completion_fraction():.1%} Complete", pad=20)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        else:
            plt.show()
        plt.close()


class TravelingSalesmanOptimizer:
    """Optimize the order of observations to minimize slew time."""

    @staticmethod
    def angular_distance(alt1: float, az1: float, alt2: float, az2: float) -> float:
        """
        Calculate angular distance between two alt/az positions.

        Parameters
        ----------
        alt1, az1 : float
            First position in degrees
        alt2, az2 : float
            Second position in degrees

        Returns
        -------
        float
            Angular distance in degrees
        """
        # Convert to radians
        alt1_rad, az1_rad = np.radians(alt1), np.radians(az1)
        alt2_rad, az2_rad = np.radians(alt2), np.radians(az2)

        # Haversine formula for angular distance
        dalt = alt2_rad - alt1_rad
        daz = az2_rad - az1_rad

        a = np.sin(dalt / 2) ** 2 + np.cos(alt1_rad) * np.cos(alt2_rad) * np.sin(daz / 2) ** 2

        return 2 * np.degrees(np.arcsin(np.sqrt(a)))

    @staticmethod
    def slew_cost(alt1: float, az1: float, alt2: float, az2: float) -> float:
        """
        Calculate slew cost between two positions.

        Includes penalty for meridian crossing which requires telescope flip.
        """
        base_distance = TravelingSalesmanOptimizer.angular_distance(alt1, az1, alt2, az2)

        # Add penalty for meridian crossing (azimuth ~180 degrees apart)
        az_diff = abs(az2 - az1)
        if az_diff > 180:
            az_diff = 360 - az_diff

        meridian_penalty = 0
        if 150 < az_diff < 210:  # Near meridian crossing
            meridian_penalty = 50  # Large penalty for meridian flip

        return base_distance + meridian_penalty

    @classmethod
    def optimize_sequence(
        cls, pointings: List[SkyPointing], current_alt: float = 45.0, current_az: float = 0.0
    ) -> List[SkyPointing]:
        """
        Optimize the sequence of pointings to minimize total slew time.

        Uses a greedy nearest-neighbor approach with meridian crossing penalties.
        """
        if not pointings:
            return []

        # Start with current telescope position
        remaining = pointings.copy()
        optimized = []

        current_position = (current_alt, current_az)

        while remaining:
            # Find nearest unvisited pointing
            best_pointing = None
            best_cost = float("inf")

            for pointing in remaining:
                cost = cls.slew_cost(
                    current_position[0], current_position[1], pointing.altitude_deg, pointing.azimuth_deg
                )

                if cost < best_cost:
                    best_cost = cost
                    best_pointing = pointing

            # Move to best pointing
            optimized.append(best_pointing)
            remaining.remove(best_pointing)
            current_position = (best_pointing.altitude_deg, best_pointing.azimuth_deg)

        return optimized


class SkyCoverageManager:
    """
    Manager for systematic sky coverage observations.

    This class handles the scheduling and management of sky coverage observations
    using HEALPix tessellation for systematic sky sampling. It creates sparse maps
    that are observed over time to build up complete sky coverage while minimizing
    telescope slew time.

    Implements the :class:`~burr.task_source.protocol.TaskSource` protocol.
    """

    def __init__(self, run: BurrRun):
        """
        Initialize the sky coverage manager.

        Parameters
        ----------
        run : BurrRun
            The current observing run
        """
        self.run = run
        self.config = run.config.schedule.coverage
        self.logger = getLogger(__name__)

        # Get configuration
        self.min_altitude = max(
            self.config.minimum_altitude_degrees,
            run.config.site.minimum_altitude_degrees,
        )
        self.max_altitude = self.config.maximum_altitude_degrees
        self.min_exposure = self.config.minimum_exposure_seconds
        self.max_exposure = self.config.maximum_exposure_seconds
        self.n_exposures = self.config.n_exposures
        self.points_per_map = self.config.points_per_map
        self.min_separation = self.config.min_separation_degrees
        self.enable_plotting = self.config.enable_plotting

        # Storage for sparse maps
        self.sparse_maps: Dict[int, SparseMap] = {}
        self.current_map_id = 0

        # Current telescope position (for slew optimization)
        self.last_pointing: Optional[SkyPointing] = None

        # Restore state from run history
        self._restore_state_from_history()

        self.logger.info(
            f"Initialized sky coverage manager with {self.points_per_map} points per map, "
            f"altitude range {self.min_altitude}° to {self.max_altitude}°"
        )

    def _restore_state_from_history(self) -> None:
        """Restore manager state from run command history."""
        self.logger.info("Restoring sky coverage state from run history")

        # Track the highest map ID we've seen
        max_map_id = 0

        for cmd in self.run.executed_commands:
            if cmd["command"] == "coverage_map_created":
                # Create map from saved state
                map_id = cmd["metadata"]["map_id"]
                max_map_id = max(max_map_id, map_id)

                if map_id not in self.sparse_maps:
                    self.sparse_maps[map_id] = SparseMap(
                        points_per_map=cmd["metadata"]["points_per_map"],
                        min_altitude=cmd["metadata"]["min_altitude"],
                        max_altitude=cmd["metadata"]["max_altitude"],
                        map_id=map_id,
                        min_separation=cmd["metadata"]["min_separation"],
                    )
                    self.logger.info(f"Restored map {map_id} from history")

            elif cmd["command"] == "coverage_point_observed":
                # Restore observed point
                map_id = cmd["metadata"]["map_id"]
                pixel_id = cmd["metadata"]["pixel_id"]
                observation_time = datetime.fromisoformat(cmd["metadata"]["observation_time"])

                if map_id in self.sparse_maps:
                    self.sparse_maps[map_id].mark_observed(pixel_id, observation_time)
                    self.logger.debug(f"Restored observation of pixel {pixel_id} in map {map_id}")

        # Set current map ID to the highest completed map + 1
        self.current_map_id = max_map_id
        self.logger.info(f"Restored {len(self.sparse_maps)} maps, current map ID: {self.current_map_id}")

    # --- TaskSource protocol implementation ---

    @property
    def name(self) -> str:
        """Short identifier for this task source."""
        return "coverage"

    @property
    def suitable_conditions(self) -> list[LightingCondition]:
        """Lighting conditions under which this source can generate tasks."""
        return [LightingCondition(window.lower()) for window in self.config.windows]

    def is_collection_complete(self) -> bool:
        """Coverage is never 'complete' -- there is always more sky to map."""
        return False

    def on_request_completed(
        self, request: ObservationRequest, duration_seconds: float
    ) -> None:
        """Called when an observation request completes successfully."""
        self.logger.info(
            f"Coverage request {request.task_id} completed in {duration_seconds:.1f}s "
            f"(pixel {request.metadata.get('pixel_id')}, "
            f"map {request.metadata.get('map_id')})"
        )

    def on_request_failed(
        self, request: ObservationRequest, error: str
    ) -> None:
        """Called when an observation request fails.

        Unmarks the pixel so it can be retried on a subsequent pass.
        """
        map_id = request.metadata.get("map_id")
        pixel_id = request.metadata.get("pixel_id")
        self.logger.warning(
            f"Coverage request {request.task_id} failed: {error} "
            f"(pixel {pixel_id}, map {map_id})"
        )
        # Attempt to unmark the pixel so it can be retried
        if map_id is not None and pixel_id is not None and map_id in self.sparse_maps:
            sparse_map = self.sparse_maps[map_id]
            sparse_map.observed_pixels.discard(pixel_id)
            for i, pointing in enumerate(sparse_map.pointings):
                if pointing.pixel_id == pixel_id:
                    sparse_map.pointings[i] = pointing._replace(
                        observed=False, last_observed=None
                    )
                    break
            self.logger.info(f"Unmarked pixel {pixel_id} in map {map_id} for retry")

    def get_or_create_current_map(self) -> SparseMap:
        """Get the current active sparse map, creating one if needed."""
        # Check if current map exists and is not complete
        if self.current_map_id in self.sparse_maps and not self.sparse_maps[self.current_map_id].is_complete():
            return self.sparse_maps[self.current_map_id]

        # Create new sparse map
        self.current_map_id += 1
        new_map = SparseMap(
            points_per_map=self.points_per_map,
            min_altitude=self.min_altitude,
            max_altitude=self.max_altitude,
            map_id=self.current_map_id,
            min_separation=self.min_separation,
        )
        self.sparse_maps[self.current_map_id] = new_map

        # Log the creation of the new map
        self.run.log_command(
            "coverage_map_created",
            f"Created new sparse map {self.current_map_id} with {len(new_map.pointings)} pointings",
            {
                "map_id": self.current_map_id,
                "points_per_map": self.points_per_map,
                "min_altitude": self.min_altitude,
                "max_altitude": self.max_altitude,
                "min_separation": self.min_separation,
                "n_pointings": len(new_map.pointings),
                "pointings": [
                    {
                        "pixel_id": p.pixel_id,
                        "altitude_deg": p.altitude_deg,
                        "azimuth_deg": p.azimuth_deg,
                    }
                    for p in new_map.pointings
                ],
            },
        )

        self.logger.info(f"Created new sparse map {self.current_map_id} with {len(new_map.pointings)} pointings")

        # Generate initial plot for the new map
        if self.enable_plotting:
            self.plot_all_maps()

        return new_map

    def filter_observable_pointings(
        self, pointings: List[SkyPointing], observation_time: datetime
    ) -> List[SkyPointing]:
        """
        Filter pointings to only include those observable at the given time.

        Parameters
        ----------
        pointings : List[SkyPointing]
            List of pointings to filter
        observation_time : datetime
            Time for observability check

        Returns
        -------
        List[SkyPointing]
            Filtered list of observable pointings
        """
        observable = []

        for pointing in pointings:
            # Skip already observed pointings
            if pointing.observed:
                continue

            # Check altitude limits
            if pointing.altitude_deg < self.min_altitude or pointing.altitude_deg > self.max_altitude:
                continue

            # TODO: Add more sophisticated observability checks
            # - Check for obstacles (buildings, trees, etc.)
            # - Check for target separation from moon/sun
            # - Check for atmospheric conditions

            observable.append(pointing)

        return observable

    def generate_request(self, task_id: str, binning: int = 1) -> ObservationRequest | None:
        """Generate the next observation request.

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
        # Get base pixel scale and adjust for current binning
        base_pixel_scale = (
            self.run.config.hardware.telescope.ifov_no_binning_arcsec if self.run.config.hardware.telescope else 0.5
        )
        pixel_scale = base_pixel_scale * binning

        now = datetime.now(ZoneInfo("UTC"))

        # Get current sparse map
        current_map = self.get_or_create_current_map()

        # Filter for observable pointings
        observable = self.filter_observable_pointings(current_map.pointings, now)

        if not observable:
            self.logger.warning("No observable pointings found in current sparse map")
            return None

        # Optimize observing sequence
        if self.last_pointing:
            # Start from last telescope position
            optimized = TravelingSalesmanOptimizer.optimize_sequence(
                observable, self.last_pointing.altitude_deg, self.last_pointing.azimuth_deg
            )
        else:
            # Start from arbitrary position (zenith)
            optimized = TravelingSalesmanOptimizer.optimize_sequence(observable, 90.0, 0.0)

        if not optimized:
            return None

        # Select the first (optimal) pointing
        target_pointing = optimized[0]

        # Generate exposure times
        exposure_times = np.linspace(
            self.min_exposure,
            self.max_exposure,
            self.n_exposures,
        )

        # Create tracking rates and exposure map specs
        if self.run.config.schedule.streaks.collect:
            # Get base pixel scale and adjust for current binning
            base_pixel_scale = (
                self.run.config.hardware.telescope.ifov_no_binning_arcsec if self.run.config.hardware.telescope else 0.5
            )
            current_binning = (
                self.run.config.hardware.camera.default_binning if self.run.config.hardware.camera else 1
            )  # Use default binning from config
            pixel_scale = base_pixel_scale * current_binning

            rates = RatesFromStreakDistribution(
                streak_length_pixels_min=self.run.config.schedule.streaks.minimum_length_pixel,
                streak_length_pixels_max=self.run.config.schedule.streaks.maximum_length_pixel,
                streak_orientation_min_deg=self.run.config.schedule.streaks.minimum_orientation_deg,
                streak_orientation_max_deg=self.run.config.schedule.streaks.maximum_orientation_deg,
                plate_scale_arcsec_per_pixel=pixel_scale,
            )
            # For streaks, one SIDEREAL followed by n RATE frames
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
            source_name="coverage",
            target=AltAzTarget(
                altitude_deg=target_pointing.altitude_deg,
                azimuth_deg=target_pointing.azimuth_deg,
            ),
            exposure_seconds=list(exposure_times),
            exposure_map=exposure_map_specs,
            rates=rates,
            metadata={
                "map_id": current_map.map_id,
                "pixel_id": target_pointing.pixel_id,
                "completion": current_map.get_completion_fraction(),
            },
        )

        # Mark pointing as observed and update last position
        current_map.mark_observed(target_pointing.pixel_id, now)
        self.last_pointing = target_pointing

        # Log the observation to run history
        self.run.log_command(
            "coverage_point_observed",
            f"Observed pixel {target_pointing.pixel_id} in map {current_map.map_id}",
            {
                "map_id": current_map.map_id,
                "pixel_id": target_pointing.pixel_id,
                "altitude_deg": target_pointing.altitude_deg,
                "azimuth_deg": target_pointing.azimuth_deg,
                "observation_time": now.isoformat(),
                "exposure_times": list(exposure_times),
                "tracking_modes": [spec.tracking_mode.value for spec in exposure_map_specs],
                "completion": current_map.get_completion_fraction(),
            },
        )

        self.logger.info(
            f"Generated coverage task for pixel {target_pointing.pixel_id} "
            f"at alt={target_pointing.altitude_deg:.1f}°, az={target_pointing.azimuth_deg:.1f}°"
        )
        self.logger.info(f"Map {current_map.map_id} completion: {current_map.get_completion_fraction():.1%}")

        # Update plots after each observation
        if self.enable_plotting:
            self.plot_all_maps()

        return request

    def generate_observation_periods(
        self,
        useful_conditions: List[LightingCondition] = [
            LightingCondition.ASTRONOMICAL_TWILIGHT,
            LightingCondition.NIGHT,
        ],
    ) -> List[Tuple[datetime, datetime]]:
        """
        Generate a list of time periods suitable for sky coverage observations.
        """
        # Use the run's observation date instead of current time
        check_time = self.run.observation_date
        self.logger.info(f"Using observation date: {check_time}")
        schedule = self.run.lighting_schedule
        self.logger.info(f"Schedule bounds: {schedule.schedule_start_utc} to {schedule.schedule_end_utc}")

        # Get all time boundaries for the requested conditions
        boundaries = []
        for condition in useful_conditions:
            condition_boundaries = schedule.get_condition_boundaries(condition)
            self.logger.info(f"Found {len(condition_boundaries)} periods for {condition}:")
            for start, end in condition_boundaries:
                # Ensure timezone awareness for all datetime objects
                if start.tzinfo is None:
                    start = start.replace(tzinfo=ZoneInfo("UTC"))
                if end.tzinfo is None:
                    end = end.replace(tzinfo=ZoneInfo("UTC"))
                self.logger.info(f"  {start} to {end} ({condition})")
                boundaries.append((start, end, condition))

        # Sort boundaries by start time
        boundaries.sort(key=lambda x: x[0])

        # Merge overlapping or adjacent periods
        if not boundaries:
            self.logger.warning("No boundaries found for any conditions")
            return []

        merged_periods = []
        current_start, current_end, current_condition = boundaries[0]

        for start, end, condition in boundaries[1:]:
            # If this period starts within 1 minute of the previous end,
            # extend the current period
            if (start - current_end) <= timedelta(minutes=1):
                current_end = max(current_end, end)
            else:
                merged_periods.append((current_start, current_end))
                current_start, current_end = start, end

        # Add the last period
        merged_periods.append((current_start, current_end))

        # Filter out periods that are completely in the past relative to observation date
        future_periods = [(start, end) for start, end in merged_periods if end > check_time]

        # For periods that span observation time, update start time
        result = []
        for start, end in future_periods:
            if start < check_time:
                result.append((check_time, end))
            else:
                result.append((start, end))

        return result

    def plot_coverage_map(self, map_id: Optional[int] = None, save_path: Optional[str] = None):
        """
        Plot the coverage map showing planned vs observed pointings.

        Parameters
        ----------
        map_id : int, optional
            ID of sparse map to plot. If None, plots current map.
        save_path : str, optional
            Path to save the plot. If None, displays interactively.
        """
        if not self.enable_plotting:
            return

        # Get the map to plot
        if map_id is None:
            if not self.sparse_maps:
                self.logger.warning("No sparse maps available for plotting")
                return
            sparse_map = self.sparse_maps[self.current_map_id]
        else:
            if map_id not in self.sparse_maps:
                self.logger.warning(f"Sparse map {map_id} not found")
                return
            sparse_map = self.sparse_maps[map_id]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), subplot_kw=dict(projection="polar"))

        # Plot 1: All planned pointings
        ax1.set_title(
            f"Sparse Map {sparse_map.map_id}: Planned Coverage\n({len(sparse_map.pointings)} pointings)", pad=20
        )

        # Convert to plot coordinates
        alts = [p.altitude_deg for p in sparse_map.pointings]
        azs = [p.azimuth_deg for p in sparse_map.pointings]
        r = [90 - alt for alt in alts]  # Convert altitude to zenith distance
        theta = np.radians(azs)  # Convert to radians

        ax1.scatter(theta, r, s=40, alpha=0.6)

        # Customize first plot
        ax1.set_theta_zero_location("N")
        ax1.set_theta_direction(-1)
        ax1.set_rlim(0, 90)
        ax1.grid(True, alpha=0.3)

        # Set up azimuth ticks every 30 degrees
        az_ticks = np.arange(0, 360, 30)
        ax1.set_xticks(np.radians(az_ticks))
        ax1.set_xticklabels([f"{int(az)}°" for az in az_ticks])
        ax1.set_rticks([])

        # Add altitude circles and labels to first plot
        for alt in [0, 30, 60]:
            radius = 90 - alt
            ax1.add_artist(
                plt.Circle(
                    (0, 0), radius, fill=False, linestyle=":", alpha=0.3, color="gray", transform=ax1.transData._b
                )
            )
            ax1.text(np.radians(90), radius, f"{alt}°", ha="left", va="center", alpha=0.5)

        # Plot 2: Observed vs unobserved
        ax2.set_title(
            f"Sparse Map {sparse_map.map_id}: Progress\n"
            f"({len(sparse_map.observed_pixels)}/{len(sparse_map.pointings)} "
            f"observed, {sparse_map.get_completion_fraction():.1%} complete)",
            pad=20,
        )

        # Separate observed and unobserved points
        observed = [(p.altitude_deg, p.azimuth_deg) for p in sparse_map.pointings if p.observed]
        unobserved = [(p.altitude_deg, p.azimuth_deg) for p in sparse_map.pointings if not p.observed]

        # Plot unobserved points
        if unobserved:
            alts, azs = zip(*unobserved)
            r = [90 - alt for alt in alts]
            theta = np.radians(azs)
            ax2.scatter(theta, r, s=40, alpha=0.6, c="blue", label="Unobserved")

        # Plot observed points
        if observed:
            alts, azs = zip(*observed)
            r = [90 - alt for alt in alts]
            theta = np.radians(azs)
            ax2.scatter(theta, r, s=40, alpha=0.6, c="green", label="Observed")

        # Customize second plot
        ax2.set_theta_zero_location("N")
        ax2.set_theta_direction(-1)
        ax2.set_rlim(0, 90)
        ax2.grid(True, alpha=0.3)

        # Set up azimuth ticks every 30 degrees
        az_ticks = np.arange(0, 360, 30)
        ax2.set_xticks(np.radians(az_ticks))
        ax2.set_xticklabels([f"{int(az)}°" for az in az_ticks])
        ax2.set_rticks([])

        # Add altitude circles and labels to second plot
        for alt in [0, 30, 60]:
            radius = 90 - alt
            ax2.add_artist(
                plt.Circle(
                    (0, 0), radius, fill=False, linestyle=":", alpha=0.3, color="gray", transform=ax2.transData._b
                )
            )
            ax2.text(np.radians(90), radius, f"{alt}°", ha="left", va="center", alpha=0.5)

        ax2.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Saved coverage plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_all_maps(self, output_dir: Optional[Path] = None):
        """Plot all active sparse maps."""
        if not self.enable_plotting or not self.sparse_maps:
            return

        if output_dir is None:
            output_dir = self.run.output_directory / "plots" / "coverage"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Only plot maps that have some observations or are current
        for map_id, sparse_map in self.sparse_maps.items():
            if sparse_map.observed_pixels or map_id == self.current_map_id:
                save_path = output_dir / f"coverage_map_{map_id:03d}.png"
                sparse_map.plot(str(save_path))  # Use SparseMap's plot method directly
                self.logger.info(f"Updated coverage plot for map {map_id}")

    def get_status_summary(self) -> Dict:
        """Get a summary of coverage status."""
        total_pixels_planned = sum(len(m.pointings) for m in self.sparse_maps.values())
        total_pixels_observed = sum(len(m.observed_pixels) for m in self.sparse_maps.values())

        return {
            "total_maps": len(self.sparse_maps),
            "current_map_id": self.current_map_id,
            "total_pixels_planned": total_pixels_planned,
            "total_pixels_observed": total_pixels_observed,
            "overall_completion": total_pixels_observed / total_pixels_planned if total_pixels_planned > 0 else 0.0,
            "points_per_map": self.points_per_map,
            "min_altitude": self.min_altitude,
            "max_altitude": self.max_altitude,
        }

    def make_coverage_offers(self) -> None:
        """
        Create observation offers for sky coverage during suitable periods.

        This method identifies suitable observation periods and prepares
        the manager for generating coverage tasks.
        """
        self.logger.info("Starting make_coverage_offers")

        try:
            # Get observation periods
            useful_conditions = self.suitable_conditions
            self.logger.info(f"Useful conditions: {useful_conditions}")
            periods = self.generate_observation_periods(useful_conditions=useful_conditions)
            self.logger.info(f"Generated {len(periods)} observation periods")

            # Initialize current sparse map if needed
            current_map = self.get_or_create_current_map()
            self.logger.info(
                f"Current sparse map {current_map.map_id} has "
                f"{len(current_map.pointings)} pointings, "
                f"{current_map.get_completion_fraction():.1%} complete"
            )

            # No need to plot here as the runner will call plot_all_maps separately

        except Exception as e:
            self.logger.error(f"Error in make_coverage_offers: {e}")
            raise
