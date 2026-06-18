"""
Flat field collection management during twilight.

This module provides tools for managing the collection of flat field calibration
frames during twilight, with automatic exposure time adjustment based on sky
brightness changes.

Implements the TaskSource protocol defined in burr.task_source.protocol.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List  # retained for generate_twilight_periods signature

import numpy as np
from astropy.io import fits
from loguru import logger
from pydantic import BaseModel
from zoneinfo import ZoneInfo

from burr.models.observation import (
    AltAzTarget,
    ExposureSpec,
    ObservationRequest,
)
from burr.models.run import BurrRun
from burr.models.site import LightingCondition
from burr.models.tracking import TrackingMode


@dataclass
class ExposureRecord:
    """Record of a single flat field exposure."""

    timestamp: datetime
    exposure_time: float
    median_counts: float
    saturation_percentage: float
    is_good: bool


class Twilight(BaseModel):
    start_time: datetime
    end_time: datetime


@dataclass
class CollectRequest:
    """Request for collecting a flat field frame."""

    exposure_seconds: float
    altitude: float = 75.0  # Point at 75 degrees by default
    azimuth: float = 0.0


class FlatFieldManager:
    """
    Manages the collection of flat field calibration frames during twilight.

    This class handles:
    - Determining optimal exposure times based on sky brightness
    - Tracking the quality of collected frames
    - Adjusting exposure times based on measured counts
    """

    def __init__(self, run: BurrRun, sk_data_root: Path | None = None):
        """
        Initialize the flat field manager.

        Parameters
        ----------
        run : BurrRun
            The run configuration and state.
        sk_data_root : Path, optional
            Directory where the SK camera data graph writes FITS frames.
            When set, ``on_request_completed`` reads the most recently
            written frame matching ``*twilight_flats*.fits`` and feeds
            it into ``process_exposure`` so the exposure-adjustment
            loop closes. Direct mode leaves this ``None`` because its
            executor calls ``process_exposure`` directly with the temp
            FITS path before invoking ``on_request_completed``.
        """
        self.run = run
        self.config = run.config.schedule.twilight_flats
        self.target_flats = self.config.n_frames
        self.min_counts = self.config.min_counts
        self.max_counts = self.config.max_counts
        self.long_exposure_threshold = self.config.long_exposure_threshold
        self.minimum_exposure = self.config.minimum_exposure_seconds
        self.max_saturation_percentage = self.config.max_saturation_percentage
        self.saturation_threshold = self.config.saturation_threshold
        self.sk_data_root = Path(sk_data_root) if sk_data_root else None

        # Track the last in-flight request's start time so on_request_completed
        # can ignore stale FITS files left over from a previous run when
        # globbing the SK data directory.
        self._request_started_at: datetime | None = None

        # Initialize state from run history if it exists
        self.good_frames = self._get_good_frames_from_history()
        self.current_exposure = self.minimum_exposure  # Start with minimum exposure
        self.exposures: List[ExposureRecord] = []

        logger.info(f"Initialized FlatFieldManager (target: {self.target_flats} flats)")
        logger.info(f"├── Count range: {self.min_counts:,} - {self.max_counts:,} ADU")
        logger.info(f"├── Long exposure threshold: {self.long_exposure_threshold}s")
        logger.info(f"├── Minimum exposure: {self.minimum_exposure}s")
        logger.info(f"├── Max saturation: {self.max_saturation_percentage}%")
        logger.info(f"├── Saturation threshold: {self.saturation_threshold or 'auto-detect'}")
        logger.info(f"└── Resuming from {self.good_frames} existing good frames")

    # ------------------------------------------------------------------
    # TaskSource protocol properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Short identifier for this task source."""
        return "twilight_flats"

    @property
    def suitable_conditions(self) -> list[LightingCondition]:
        """Lighting conditions under which this source can generate tasks.

        Reads from the ``windows`` list in the twilight_flats config section,
        e.g. ``["nautical_twilight", "astronomical_twilight"]``.
        """
        return [
            LightingCondition(window.lower())
            for window in self.config.windows
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_good_frames_from_history(self) -> int:
        """
        Reconstruct state from run command history.
        """
        good_frames = 0
        for cmd in self.run.executed_commands:
            if cmd["command"] == "flat_field_saved":
                good_frames = int(cmd["result"].split("/")[0])  # Extract X from "X/Y flats collected"
        return good_frames

    def _calculate_sky_brightness_trend(self, window_size: int = 3) -> float | None:
        """Calculate the rate of change in sky brightness from recent exposures.

        Args:
            window_size: Number of recent exposures to consider

        Returns:
            Rate of change in counts per second, or None if insufficient data
        """
        if len(self.exposures) < window_size:
            return None

        recent = self.exposures[-window_size:]

        # Calculate counts/second for each exposure
        brightness = [(exp.median_counts / exp.exposure_time, exp.timestamp) for exp in recent]

        # Calculate time differences in seconds
        time_diffs = [(b[1] - brightness[0][1]).total_seconds() for b in brightness]
        brightness_diffs = [b[0] - brightness[0][0] for b in brightness]

        if not time_diffs[1:]:  # Need at least 2 points
            return None

        # Use numpy to fit a line to the data
        try:
            slope, _ = np.polyfit(time_diffs, brightness_diffs, 1)
            return slope
        except Exception as e:
            logger.warning(f"Failed to calculate brightness trend: {e}")
            return None

    def _adjust_exposure_time(self, current_counts: float, exposure_time: float) -> float:
        """Calculate the next exposure time based on current counts and sky brightness trend.

        Args:
            current_counts: Median counts in the current frame
            exposure_time: Current exposure time in seconds

        Returns:
            New exposure time in seconds
        """
        # For short exposures, use simple linear scaling
        if exposure_time < self.long_exposure_threshold:
            new_exposure = exposure_time * (self.max_counts / current_counts)
            logger.debug(f"Short exposure adjustment: {exposure_time:.3f}s → {new_exposure:.3f}s")
        # For longer exposures, consider sky brightness trend
        elif exposure_time >= self.long_exposure_threshold:
            trend = self._calculate_sky_brightness_trend()
            if trend is not None:
                logger.info(f"Sky brightness trend: {trend:,.1f} counts/s²")

                # If sky is darkening significantly
                if trend < -10:  # threshold for considering significant darkening
                    # Calculate how much darker it will be by the time we finish the next exposure
                    # Use a more aggressive adjustment to compensate for darkening
                    predicted_darkening = abs(trend * exposure_time)
                    adjustment_factor = (self.max_counts + predicted_darkening) / current_counts
                    new_exposure = exposure_time * adjustment_factor

                    logger.info("Long exposure with darkening sky:")
                    logger.info(f"├── Current counts: {current_counts:,.0f}")
                    logger.info(f"├── Predicted darkening: {predicted_darkening:,.0f} counts/s")
                    logger.info(f"├── Adjustment factor: {adjustment_factor:.3f}")
                    logger.info(f"└── New exposure: {new_exposure:.3f}s")
                else:
                    # Default to standard linear scaling if trend is not significant
                    new_exposure = exposure_time * (self.max_counts / current_counts)
                    logger.debug(f"Standard linear adjustment: {exposure_time:.3f}s → {new_exposure:.3f}s")
            else:
                # Default to standard linear scaling if we can't calculate trend
                new_exposure = exposure_time * (self.max_counts / current_counts)
                logger.debug(f"Standard linear adjustment: {exposure_time:.3f}s → {new_exposure:.3f}s")

        # Ensure exposure time never goes below minimum
        if new_exposure < self.minimum_exposure:
            logger.info(f"Calculated exposure ({new_exposure:.3f}s) below minimum ({self.minimum_exposure}s)")
            logger.info(f"Setting exposure to minimum: {self.minimum_exposure}s")
            new_exposure = self.minimum_exposure

        return new_exposure

    def process_exposure(self, frame_path: str, device_control=None) -> bool:
        """
        Process a collected flat field frame.

        Parameters
        ----------
        frame_path : str
            Path to the FITS file produced by the camera.
        device_control : DirectDeviceControl, optional
            Direct-mode device control. When supplied (direct path),
            good frames are re-written via ``device_control.save_fits``
            using the camera's most recent image and full live metadata.
            When ``None`` (SK path), the FITS at ``frame_path`` is
            already the canonical SK output, so we copy it into the
            run's flats directory and stamp the flat-specific headers
            into the copy.

        Returns
        -------
        bool
            True if the frame meets quality criteria
        """
        try:
            # Convert frame_path to a proper Path object and resolve it
            frame_path = Path(frame_path).resolve()
            logger.info(f"Processing flat field frame: {frame_path}")

            # Load the frame
            with fits.open(str(frame_path)) as hdul:
                data = hdul[0].data
                header = hdul[0].header

            # Get exposure time and timestamp
            exposure_time = float(header.get("EXPTIME", 0))

            # Handle timestamp - use current time if not available or invalid
            date_obs = header.get("DATE-OBS")
            if date_obs and isinstance(date_obs, str) and date_obs.strip():
                try:
                    # Try different timestamp formats
                    if "." in date_obs:
                        timestamp = datetime.strptime(date_obs, "%Y-%m-%dT%H:%M:%S.%f")
                    else:
                        timestamp = datetime.strptime(date_obs, "%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    logger.warning(f"Could not parse DATE-OBS '{date_obs}', using current time")
                    timestamp = datetime.now(ZoneInfo("UTC"))
            else:
                logger.debug("No valid DATE-OBS in header, using current time")
                timestamp = datetime.now(ZoneInfo("UTC"))

            logger.debug(f"Frame metadata: exposure={exposure_time:.3f}s, timestamp={timestamp}")

            # Calculate statistics
            median_counts = float(np.median(data))
            mean_counts = float(np.mean(data))
            std_counts = float(np.std(data))
            min_counts = float(np.min(data))
            max_counts = float(np.max(data))

            # Calculate saturation percentage
            saturation_percentage, saturation_threshold_used = self._calculate_saturation_percentage(data)

            logger.info("Frame statistics:")
            logger.info(f"├── Median: {median_counts:,.0f} ADU")
            logger.info(f"├── Mean: {mean_counts:,.0f} ADU")
            logger.info(f"├── Std Dev: {std_counts:,.0f} ADU")
            logger.info(f"├── Min: {min_counts:,.0f} ADU")
            logger.info(f"├── Max: {max_counts:,.0f} ADU")
            logger.info(f"└── Saturation: {saturation_percentage:.2f}% (threshold: {saturation_threshold_used})")

            # Check if it's a good frame - both count range AND saturation criteria
            counts_good = self.min_counts <= median_counts <= self.max_counts
            saturation_good = saturation_percentage <= self.max_saturation_percentage
            is_good = counts_good and saturation_good

            # Create and store record
            record = ExposureRecord(
                timestamp=timestamp,
                exposure_time=exposure_time,
                median_counts=median_counts,
                saturation_percentage=saturation_percentage,
                is_good=is_good,
            )
            self.exposures.append(record)

            # If counts are too low, adjust exposure time using our smart adjustment
            if median_counts < self.min_counts:
                old_exposure = self.current_exposure
                self.current_exposure = self._adjust_exposure_time(median_counts, exposure_time)
                logger.warning(f"Counts too low ({median_counts:,.0f} < {self.min_counts:,.0f})")
                logger.info(f"Adjusting exposure: {old_exposure:.3f}s → {self.current_exposure:.3f}s")
            elif median_counts > self.max_counts:
                # For high counts, still use smart adjustment
                old_exposure = self.current_exposure
                self.current_exposure = self._adjust_exposure_time(median_counts, exposure_time)

                if old_exposure == self.minimum_exposure and self.current_exposure == self.minimum_exposure:
                    # We're already at minimum exposure and counts are still too high
                    logger.warning(f"Counts too high ({median_counts:,.0f} > {self.max_counts:,.0f})")
                    logger.info(f"Already at minimum exposure ({self.minimum_exposure}s) - waiting for sky to darken")
                else:
                    logger.warning(f"Counts too high ({median_counts:,.0f} > {self.max_counts:,.0f})")
                    logger.info(f"Adjusting exposure: {old_exposure:.3f}s → {self.current_exposure:.3f}s")

            # Only save and increment good_frames if the frame meets our criteria
            if is_good:
                # Good frame, save it to flats directory with proper metadata
                self.good_frames += 1
                output_dir = Path(self.run.output_directory) / "data" / "flats"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"flat_{self.good_frames:03d}.fits"

                logger.success(f"Good flat frame ({self.good_frames}/{self.target_flats}):")
                logger.info(f"├── Median counts: {median_counts:,.0f}")
                logger.info(f"├── Saturation: {saturation_percentage:.2f}%")
                logger.info(f"├── Exposure time: {exposure_time:.3f}s")
                logger.info(f"└── Saving to: {output_path}")

                # Log the successful flat collection to run history
                self.run.log_command(
                    "flat_field_saved",
                    f"{self.good_frames}/{self.target_flats} flats collected",
                    {
                        "frame_number": self.good_frames,
                        "median_counts": median_counts,
                        "saturation_percentage": saturation_percentage,
                        "exposure_time": exposure_time,
                        "output_path": str(output_path.relative_to(self.run.output_directory)),
                    },
                )

                # Add flat-specific metadata
                extra_headers = {
                    "IMAGETYP": ("Flat Field", "Type of image"),
                    "MEDIAN": (median_counts, "Median counts in frame"),
                    "MEAN": (mean_counts, "Mean counts in frame"),
                    "STDDEV": (std_counts, "Standard deviation of counts"),
                    "MINCNTS": (min_counts, "Minimum counts in frame"),
                    "MAXCNTS": (max_counts, "Maximum counts in frame"),
                    "FLATSNUM": (self.good_frames, "Flat field sequence number"),
                    "SATPCT": (saturation_percentage, "Percentage of saturated pixels"),
                    "SATTHRES": (saturation_threshold_used, "Saturation threshold used"),
                }

                # Save using device control's save_fits to include all metadata
                # (direct path), or copy the SK-written FITS and stamp
                # flat-specific headers into it (SK path).
                if device_control is not None:
                    device_control.save_fits(
                        output_path=output_path,
                        tracking_mode=TrackingMode.OFF,
                        rates=None,
                        extra_headers=extra_headers,
                    )
                else:
                    self._copy_and_stamp_flat(
                        src=frame_path,
                        dst=output_path,
                        extra_headers=extra_headers,
                    )

                # Calculate next exposure time if we're in long exposure mode
                if exposure_time >= self.long_exposure_threshold:
                    old_exposure = self.current_exposure
                    self.current_exposure = self._adjust_exposure_time(median_counts, exposure_time)
                    logger.info("Preemptively adjusting next exposure for darkening sky:")
                    logger.info(f"├── Current exposure: {old_exposure:.3f}s")
                    logger.info(f"└── Next exposure: {self.current_exposure:.3f}s")
            else:
                # Provide detailed reason for rejection
                rejection_reasons = []
                if not counts_good:
                    rejection_reasons.append(
                        f"counts ({median_counts:,.0f}) outside range [{self.min_counts:,.0f}, {self.max_counts:,.0f}]"
                    )
                if not saturation_good:
                    rejection_reasons.append(
                        f"saturation ({saturation_percentage:.2f}%) exceeds limit ({self.max_saturation_percentage}%)"
                    )

                reason = " and ".join(rejection_reasons)
                logger.warning(f"Frame rejected - {reason}")

                # Log the rejected frame to run history
                self.run.log_command(
                    "flat_field_rejected",
                    f"Frame rejected: {reason}",
                    {
                        "median_counts": median_counts,
                        "saturation_percentage": saturation_percentage,
                        "exposure_time": exposure_time,
                        "min_counts": self.min_counts,
                        "max_counts": self.max_counts,
                        "max_saturation_percentage": self.max_saturation_percentage,
                        "rejection_reason": reason,
                    },
                )

            return is_good

        except Exception as e:
            logger.error(f"Error processing flat field frame: {str(e)}")
            logger.exception(e)  # Log full traceback
            return False

    def generate_request(
        self, task_id: str, binning: int = 1
    ) -> ObservationRequest | None:
        """Generate the next observation request.

        Parameters
        ----------
        task_id : str
            Unique identifier for this task.
        binning : int
            Current camera binning factor (unused for flats but required by protocol).

        Returns
        -------
        ObservationRequest | None
            A request if one is available, None otherwise.
        """
        # Check if we're done
        if self.is_collection_complete():
            logger.success(f"Flat collection complete! Collected {self.good_frames} good frames")
            return None

        # Stamp the request's start time so on_request_completed can
        # filter out stale FITS files when globbing the SK data dir.
        self._request_started_at = datetime.now(ZoneInfo("UTC"))

        # Generate request with current exposure time
        request = ObservationRequest(
            task_id=task_id,
            source_name="twilight_flats",
            target=AltAzTarget(altitude_deg=75.0, azimuth_deg=0.0),
            exposure_seconds=[self.current_exposure],
            exposure_map=[ExposureSpec(tracking_mode=TrackingMode.OFF)],
            rates=None,
            metadata={"good_frames": self.good_frames, "target_flats": self.target_flats},
        )

        logger.info(f"Generated flat field request ({self.good_frames + 1}/{self.target_flats}):")
        logger.info(f"├── Task ID: {task_id}")
        logger.info(f"├── Exposure: {self.current_exposure:.3f}s")
        logger.info(f"├── Pointing: Alt={75.0}, Az={0.0}")
        logger.info("└── Tracking: OFF")

        return request

    def is_collection_complete(self) -> bool:
        """Check if we have enough good flats."""
        return self.good_frames >= self.target_flats

    def _copy_and_stamp_flat(
        self,
        src: Path,
        dst: Path,
        extra_headers: dict,
    ) -> None:
        """Copy an SK-written FITS into the flats dir and add flat headers.

        The source FITS already carries SK's full per-frame metadata
        (camera, mount, site headers written by SK's data graph). We
        only need to stamp burr's flat-specific keys onto the primary
        HDU and write the result to the canonical flats path.
        """
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        with fits.open(dst, mode="update") as hdul:
            header = hdul[0].header
            for key, value in extra_headers.items():
                if isinstance(value, tuple):
                    header[key] = value
                else:
                    header[key] = (value, f"Added header {key}")

    def on_request_completed(
        self, request: ObservationRequest, duration_seconds: float
    ) -> None:
        """Called when an observation request completes successfully.

        In direct mode, ``process_exposure`` was already invoked by the
        executor before this callback fires, so this just logs.

        In SK mode, the camera's data graph writes the FITS to disk
        but burr never sees the path. We close the loop here by
        globbing ``sk_data_root`` for the most recently written
        ``*twilight_flats*.fits`` and feeding it into
        ``process_exposure`` so ``current_exposure`` and
        ``good_frames`` advance.

        Parameters
        ----------
        request : ObservationRequest
            The completed request.
        duration_seconds : float
            Wall-clock time the request took.
        """
        logger.info(
            f"Flat request {request.task_id} completed in {duration_seconds:.1f}s "
            f"({self.good_frames}/{self.target_flats} good frames so far)"
        )

        if self.sk_data_root is None:
            return

        frame_path = self._find_latest_sk_flat(request)
        if frame_path is None:
            logger.warning(
                f"No flat FITS found under {self.sk_data_root} for request "
                f"{request.task_id} — exposure adjustment skipped"
            )
            return

        logger.info(f"Reading back SK-written flat for analysis: {frame_path}")
        self.process_exposure(str(frame_path), device_control=None)

    def _find_latest_sk_flat(self, request: ObservationRequest) -> Path | None:
        """Locate the FITS the SK data graph wrote for this request.

        SK's WriteFile uses the ``file_name`` template burr injects in
        ``_build_sk_tasks`` (see ``burr_program.py``), which embeds the source
        name (``BurrContext.mode``) into the filename. We pick the newest file
        under ``sk_data_root`` whose name contains the source name and
        whose mtime falls inside the request's wall-clock window.

        Returns ``None`` if no matching frame is found.
        """
        if self.sk_data_root is None or not self.sk_data_root.exists():
            return None

        # Restrict to files whose mtime is at or after when this request
        # started — protects against a stale leftover frame matching
        # *twilight_flats*.fits from a previous run/session.
        cutoff = self._request_started_at
        candidates = []
        pattern = f"*{request.source_name}*.fits"
        for path in self.sk_data_root.rglob(pattern):
            try:
                mtime = datetime.fromtimestamp(
                    path.stat().st_mtime, tz=ZoneInfo("UTC")
                )
            except OSError:
                continue
            if cutoff is not None and mtime < cutoff:
                continue
            candidates.append((mtime, path))

        if not candidates:
            return None

        candidates.sort(key=lambda t: t[0], reverse=True)
        return candidates[0][1]

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
        logger.warning(
            f"Flat request {request.task_id} failed: {error} "
            f"({self.good_frames}/{self.target_flats} good frames so far)"
        )

    # ------------------------------------------------------------------
    # Twilight period helpers
    # ------------------------------------------------------------------

    def generate_twilight_periods(
        self,
        useful_conditions: List[LightingCondition] = [
            LightingCondition.ASTRONOMICAL_TWILIGHT,
            LightingCondition.NAUTICAL_TWILIGHT,
        ],
    ) -> List[Twilight]:
        """
        Generate a list of twilight periods where lighting conditions are suitable for flat collection.

        Parameters
        ----------
        useful_conditions : List[LightingCondition]
            List of lighting conditions that are suitable for flat collection.
            Defaults to astronomical and nautical twilight.

        Returns
        -------
        List[Twilight]
            List of Twilight objects containing start and end times for suitable periods
        """
        # Get current time in UTC
        now = datetime.now(ZoneInfo("UTC"))
        logger.info(f"Generating twilight periods from {now} UTC")
        logger.info(f"Useful conditions: {[c.name for c in useful_conditions]}")

        # Use the pre-calculated lighting schedule from the run
        schedule = self.run.lighting_schedule
        logger.info(f"Schedule bounds: {schedule.schedule_start_utc} to {schedule.schedule_end_utc}")

        # Get all time boundaries for each useful condition
        all_boundaries = []
        for condition in useful_conditions:
            boundaries = schedule.get_condition_boundaries(condition)
            logger.info(f"Found {len(boundaries)} periods for {condition.name}:")
            for start, end in boundaries:
                # Ensure boundary times are timezone-aware
                if start.tzinfo is None:
                    start = start.replace(tzinfo=ZoneInfo("UTC"))
                if end.tzinfo is None:
                    end = end.replace(tzinfo=ZoneInfo("UTC"))
                logger.info(f"  ├── {start} UTC")
                logger.info(f"  └── {end} UTC")
                all_boundaries.append((start, end))

        if not all_boundaries:
            logger.warning("No boundaries found for any conditions")
            return []

        # Sort boundaries by start time
        all_boundaries.sort(key=lambda x: x[0])

        # Merge overlapping or adjacent periods
        merged_periods = []
        current_start, current_end = all_boundaries[0]

        for start, end in all_boundaries[1:]:
            # If this period starts after current period ends (with a small gap tolerance)
            if start > current_end + timedelta(minutes=1):
                # Save the current period and start a new one
                merged_periods.append(Twilight(start_time=current_start, end_time=current_end))
                current_start = start
                current_end = end
            else:
                # Extend current period
                current_end = max(current_end, end)

        # Add the last period
        merged_periods.append(Twilight(start_time=current_start, end_time=current_end))

        # Filter out periods that are completely in the past
        future_periods = [period for period in merged_periods if period.end_time > now]

        # For periods that span current time, update start time to now
        for period in future_periods:
            if period.start_time < now:
                period.start_time = now

        logger.info(f"Generated {len(future_periods)} usable twilight periods:")
        for i, period in enumerate(future_periods, 1):
            logger.info(f"Period {i}:")
            logger.info(f"├── Start: {period.start_time} UTC")
            logger.info(f"└── End: {period.end_time} UTC")

        return future_periods

    def _calculate_saturation_percentage(self, data: np.ndarray) -> tuple[float, int]:
        """Calculate the percentage of saturated pixels in the frame.

        Args:
            data: The image data array

        Returns:
            Tuple of (saturation_percentage, saturation_threshold_used)
        """
        # Auto-detect saturation threshold if not configured
        if self.saturation_threshold is None:
            # For 16-bit data, saturation is typically at 2^16 - 1 = 65535
            # For 8-bit data, it would be 2^8 - 1 = 255
            # We'll detect based on the data type and maximum value
            if data.dtype == np.uint16:
                threshold = 65535
            elif data.dtype == np.uint8:
                threshold = 255
            else:
                # For other data types, use 99.9% of the maximum value in the frame
                threshold = int(np.max(data) * 0.999)
        else:
            threshold = self.saturation_threshold

        # Count saturated pixels
        saturated_pixels = np.sum(data >= threshold)
        total_pixels = data.size
        saturation_percentage = (saturated_pixels / total_pixels) * 100.0

        return saturation_percentage, threshold
