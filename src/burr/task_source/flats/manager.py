"""
Flat field collection management during twilight.

This module provides tools for managing the collection of flat field calibration
frames during twilight, with automatic exposure time adjustment based on sky
brightness changes.

Exposure control follows the huntsman-pocs ``FlatFieldSequence`` scheme: the
next exposure scales the last one to hit the count target, then corrects for
the sky brightening/darkening by ``2 ** (elapsed / sky_doubling_seconds)``
since that frame was taken. Counts are a sigma-clipped mean over a central
cutout, which keeps vignetted corners out of the statistic.

Multi-camera: one request fans out to every camera on the controller (SK
mode). Each camera keeps its own exposure state, keyed by the FITS ``UNITID``
header (falling back to the frame's parent directory), and the next request
uses an exposure that serves the cameras still short of ``n_frames``. A
camera whose ideal exposure is far from the shared one is simply rejected
that cycle and recovers as the sky changes; per-camera exposure times need
sensorkit support for per-instrument collect parameters.

Implements the TaskSource protocol defined in burr.task_source.protocol.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List  # retained for generate_twilight_periods signature

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
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

# Camera key used when a frame carries no unit identity (e.g. direct mode,
# which drives a single camera).
DEFAULT_CAMERA = "default"


@dataclass
class ExposureRecord:
    """Record of a single flat field exposure."""

    timestamp: datetime
    exposure_time: float
    median_counts: float
    saturation_percentage: float
    is_good: bool


@dataclass
class CameraFlatState:
    """Per-camera flat collection state."""

    good_frames: int = 0
    next_exposure: float | None = None
    exposures: List[ExposureRecord] = field(default_factory=list)


@dataclass
class FrameStats:
    """Count statistics for one flat frame."""

    average_counts: float
    median_counts: float
    mean_counts: float
    std_counts: float
    min_counts: float
    max_counts: float
    saturation_percentage: float
    saturation_threshold: int


class Twilight(BaseModel):
    start_time: datetime
    end_time: datetime


def load_image_hdu(path: Path) -> tuple[np.ndarray, fits.Header]:
    """Return ``(data, header)`` of the first HDU in ``path`` holding an image.

    SK's ``compress_fits`` writes an empty primary HDU followed by a
    ``CompImageHDU``, so the pixels are not in HDU 0. The returned header
    is the image HDU's own (where SK puts EXPTIME, UNITID, IMAGETYP, ...).
    """
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                return np.asarray(hdu.data), hdu.header.copy()
    raise ValueError(f"no image data in any HDU of {path}")


def _parse_date_obs(date_obs) -> datetime | None:
    """Parse a FITS DATE-OBS string as UTC, or None if absent/invalid."""
    if not (date_obs and isinstance(date_obs, str) and date_obs.strip()):
        return None
    try:
        parsed = datetime.fromisoformat(date_obs.strip())
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=ZoneInfo("UTC"))


def _is_flat_frame(path: Path) -> bool:
    """True if any HDU header in ``path`` has ``IMAGETYP = flat``.

    Reads headers only, so a compressed image is not decompressed just to
    classify it.
    """
    try:
        with fits.open(path) as hdul:
            return any(str(hdu.header.get("IMAGETYP", "")).strip().lower() == "flat" for hdu in hdul)
    except Exception as e:
        logger.debug(f"Skipping unreadable FITS {path}: {e}")
        return False


def _central_cutout(data: np.ndarray, size: int | None) -> np.ndarray:
    """Central ``size``x``size`` box of ``data`` (whole frame if None/too big)."""
    if size is None:
        return data
    ny, nx = data.shape[-2:]
    if size >= ny and size >= nx:
        return data
    y0, x0 = max((ny - size) // 2, 0), max((nx - size) // 2, 0)
    return data[..., y0 : y0 + size, x0 : x0 + size]


class FlatFieldManager:
    """
    Manages the collection of flat field calibration frames during twilight.

    This class handles:
    - Determining optimal exposure times based on sky brightness
    - Tracking the quality of collected frames, per camera
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
            When set, ``on_request_completed`` finds every flat frame the
            cameras wrote for the request (by ``IMAGETYP`` and write time,
            not filename) and feeds each into ``process_exposure`` so the
            exposure-adjustment loop closes. Direct mode leaves this
            ``None`` because its executor calls ``process_exposure``
            directly with the temp FITS path before invoking
            ``on_request_completed``.
        """
        self.run = run
        self.config = run.config.schedule.twilight_flats
        self.target_flats = self.config.n_frames
        self.min_counts = self.config.min_counts
        self.max_counts = self.config.max_counts
        self.target_counts = (self.min_counts + self.max_counts) / 2
        self.minimum_exposure = self.config.minimum_exposure_seconds
        self.maximum_exposure = self.config.maximum_exposure_seconds
        self.sky_doubling_seconds = self.config.sky_doubling_seconds
        self.cutout_size = self.config.cutout_size
        self.frame_settle_seconds = self.config.frame_settle_seconds
        self.max_saturation_percentage = self.config.max_saturation_percentage
        self.saturation_threshold = self.config.saturation_threshold
        self.sk_data_root = Path(sk_data_root) if sk_data_root else None

        # Track the last in-flight request's start time so on_request_completed
        # only picks up frames written for this request.
        self._request_started_at: datetime | None = None
        # Cameras that wrote a frame for the last completed request; used to
        # decide how many frames to wait for on the next one.
        self._expected_cameras: int = 0

        # Initialize state from run history if it exists
        self.cameras: dict[str, CameraFlatState] = self._get_camera_state_from_history()
        self.current_exposure = self.minimum_exposure  # Start with minimum exposure

        logger.info(f"Initialized FlatFieldManager (target: {self.target_flats} flats per camera)")
        logger.info(f"├── Count range: {self.min_counts:,} - {self.max_counts:,} ADU (target {self.target_counts:,.0f})")
        logger.info(f"├── Exposure range: {self.minimum_exposure}s - {self.maximum_exposure}s")
        logger.info(f"├── Sky doubling time: {self.sky_doubling_seconds}s")
        logger.info(f"├── Stats cutout: {self.cutout_size or 'full frame'}")
        logger.info(f"├── Max saturation: {self.max_saturation_percentage}%")
        logger.info(f"├── Saturation threshold: {self.saturation_threshold or 'auto-detect'}")
        logger.info(f"└── Resuming from {self.good_frames} existing good frames ({len(self.cameras)} camera(s))")

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

    @property
    def good_frames(self) -> int:
        """Good frames of the camera furthest behind (collection is done when it is)."""
        if not self.cameras:
            return 0
        return min(state.good_frames for state in self.cameras.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_camera_state_from_history(self) -> dict[str, CameraFlatState]:
        """
        Reconstruct per-camera good-frame counts from run command history.
        """
        cameras: dict[str, CameraFlatState] = {}
        for cmd in self.run.executed_commands:
            if cmd["command"] != "flat_field_saved":
                continue
            metadata = cmd.get("metadata") or {}
            camera = metadata.get("camera", DEFAULT_CAMERA)
            good = metadata.get("frame_number")
            if good is None:
                good = int(cmd["result"].split("/")[0])  # Extract X from "X/Y flats collected"
            cameras.setdefault(camera, CameraFlatState()).good_frames = int(good)
        return cameras

    def _is_dawn(self, when: datetime) -> bool:
        """True if ``when`` is after local midnight, i.e. the sky is brightening."""
        midnight = self.run.lighting_schedule.local_midnight
        if midnight.tzinfo is None:
            midnight = midnight.replace(tzinfo=ZoneInfo("UTC"))
        return when >= midnight

    def _next_exposure_for(self, last_exposure: float, average_counts: float, taken_at: datetime) -> float:
        """Exposure that should hit ``target_counts`` now, given the last frame.

        Scales the last exposure to the count target, then compensates for
        the sky changing since that frame by ``2 ** (elapsed / doubling)``
        (longer at dusk as it darkens, shorter at dawn as it brightens).
        A frame from the other twilight (last evening's, seen at dawn) says
        nothing about the sky now, so the sequence restarts at the minimum.
        """
        now = datetime.now(ZoneInfo("UTC"))
        dawn = self._is_dawn(now)
        if dawn != self._is_dawn(taken_at):
            return self.minimum_exposure
        exposure = last_exposure * (self.target_counts / max(average_counts, 1.0))
        elapsed = max((now - taken_at).total_seconds(), 0.0)
        # Cap the doublings so a long gap clips to the limits instead of overflowing.
        sky_factor = 2.0 ** min(elapsed / self.sky_doubling_seconds, 64.0)
        exposure = exposure / sky_factor if dawn else exposure * sky_factor
        return float(np.clip(exposure, self.minimum_exposure, self.maximum_exposure))

    def _shared_exposure(self) -> float:
        """One exposure for the next request, serving cameras still collecting.

        Takes the median of the per-camera ideal exposures (re-projected to
        now for sky change) over cameras that are not yet complete.
        """
        candidates = []
        for state in self.cameras.values():
            if state.good_frames >= self.target_flats or not state.exposures:
                continue
            last = state.exposures[-1]
            candidates.append(self._next_exposure_for(last.exposure_time, last.median_counts, last.timestamp))
        if not candidates:
            return self.current_exposure
        return float(np.median(candidates))

    def _frame_stats(self, data: np.ndarray) -> FrameStats:
        """Count statistics over the central cutout, plus full-frame saturation."""
        cutout = _central_cutout(data, self.cutout_size).astype(np.float32)
        average, median, std = sigma_clipped_stats(cutout)
        saturation_percentage, threshold = self._calculate_saturation_percentage(data)
        return FrameStats(
            average_counts=float(average),
            median_counts=float(median),
            mean_counts=float(np.mean(cutout)),
            std_counts=float(std),
            min_counts=float(np.min(cutout)),
            max_counts=float(np.max(cutout)),
            saturation_percentage=saturation_percentage,
            saturation_threshold=threshold,
        )

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
            frame_path = Path(frame_path).resolve()
            logger.info(f"Processing flat field frame: {frame_path}")

            data, header = load_image_hdu(frame_path)
            camera = str(header.get("UNITID") or (frame_path.parent.name if device_control is None else DEFAULT_CAMERA))
            state = self.cameras.setdefault(camera, CameraFlatState())

            exposure_time = float(header.get("EXPTIME", 0) or self.current_exposure)
            timestamp = _parse_date_obs(header.get("DATE-OBS"))
            if timestamp is None:
                logger.debug("No valid DATE-OBS in header, using current time")
                timestamp = datetime.now(ZoneInfo("UTC"))

            stats = self._frame_stats(data)
            counts = stats.average_counts

            logger.info(f"Frame statistics [{camera}] ({exposure_time:.3f}s):")
            logger.info(f"├── Clipped mean: {counts:,.0f} ADU (median {stats.median_counts:,.0f}, std {stats.std_counts:,.0f})")
            logger.info(f"├── Cutout min/max: {stats.min_counts:,.0f} / {stats.max_counts:,.0f} ADU")
            logger.info(f"└── Saturation: {stats.saturation_percentage:.2f}% (threshold: {stats.saturation_threshold})")

            counts_good = self.min_counts <= counts <= self.max_counts
            saturation_good = stats.saturation_percentage <= self.max_saturation_percentage
            # A camera that is already complete still updates its exposure
            # model, but its extra frames are not kept.
            is_good = counts_good and saturation_good and state.good_frames < self.target_flats

            state.exposures.append(
                ExposureRecord(
                    timestamp=timestamp,
                    exposure_time=exposure_time,
                    median_counts=counts,
                    saturation_percentage=stats.saturation_percentage,
                    is_good=is_good,
                )
            )
            state.next_exposure = self._next_exposure_for(exposure_time, counts, timestamp)
            logger.info(f"[{camera}] ideal next exposure: {state.next_exposure:.3f}s")

            if is_good:
                self._save_good_frame(camera, state, frame_path, stats, exposure_time, device_control)
            elif state.good_frames >= self.target_flats:
                logger.info(f"[{camera}] already has {state.good_frames} flats — frame not kept")
            else:
                self._log_rejection(camera, stats, exposure_time, counts_good, saturation_good)

            return is_good

        except Exception as e:
            logger.error(f"Error processing flat field frame: {str(e)}")
            logger.exception(e)  # Log full traceback
            return False

    def _save_good_frame(
        self,
        camera: str,
        state: CameraFlatState,
        frame_path: Path,
        stats: FrameStats,
        exposure_time: float,
        device_control,
    ) -> None:
        """Save a good flat to the run's flats directory and log it to history."""
        state.good_frames += 1
        output_dir = Path(self.run.output_directory) / "data" / "flats"
        if camera != DEFAULT_CAMERA:
            output_dir = output_dir / camera
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"flat_{state.good_frames:03d}.fits"

        logger.success(f"Good flat frame [{camera}] ({state.good_frames}/{self.target_flats}):")
        logger.info(f"├── Clipped mean counts: {stats.average_counts:,.0f}")
        logger.info(f"├── Saturation: {stats.saturation_percentage:.2f}%")
        logger.info(f"├── Exposure time: {exposure_time:.3f}s")
        logger.info(f"└── Saving to: {output_path}")

        self.run.log_command(
            "flat_field_saved",
            f"{state.good_frames}/{self.target_flats} flats collected",
            {
                "camera": camera,
                "frame_number": state.good_frames,
                "median_counts": stats.average_counts,
                "saturation_percentage": stats.saturation_percentage,
                "exposure_time": exposure_time,
                "source_path": str(frame_path),
                "output_path": str(output_path.relative_to(self.run.output_directory)),
            },
        )

        extra_headers = {
            "IMAGETYP": ("Flat Field", "Type of image"),
            "MEDIAN": (stats.median_counts, "Clipped median counts in cutout"),
            "MEAN": (stats.average_counts, "Clipped mean counts in cutout"),
            "STDDEV": (stats.std_counts, "Clipped std dev of counts in cutout"),
            "MINCNTS": (stats.min_counts, "Minimum counts in cutout"),
            "MAXCNTS": (stats.max_counts, "Maximum counts in cutout"),
            "FLATSNUM": (state.good_frames, "Flat field sequence number"),
            "SATPCT": (stats.saturation_percentage, "Percentage of saturated pixels"),
            "SATTHRES": (stats.saturation_threshold, "Saturation threshold used"),
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
            self._copy_and_stamp_flat(src=frame_path, dst=output_path, extra_headers=extra_headers)

    def _log_rejection(
        self, camera: str, stats: FrameStats, exposure_time: float, counts_good: bool, saturation_good: bool
    ) -> None:
        """Log why a flat frame was rejected, to the log and run history."""
        rejection_reasons = []
        if not counts_good:
            rejection_reasons.append(
                f"counts ({stats.average_counts:,.0f}) outside range [{self.min_counts:,.0f}, {self.max_counts:,.0f}]"
            )
        if not saturation_good:
            rejection_reasons.append(
                f"saturation ({stats.saturation_percentage:.2f}%) exceeds limit ({self.max_saturation_percentage}%)"
            )

        reason = " and ".join(rejection_reasons)
        logger.warning(f"Frame rejected [{camera}] - {reason}")

        self.run.log_command(
            "flat_field_rejected",
            f"Frame rejected: {reason}",
            {
                "camera": camera,
                "median_counts": stats.average_counts,
                "saturation_percentage": stats.saturation_percentage,
                "exposure_time": exposure_time,
                "min_counts": self.min_counts,
                "max_counts": self.max_counts,
                "max_saturation_percentage": self.max_saturation_percentage,
                "rejection_reason": reason,
            },
        )

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
            logger.success(f"Flat collection complete! Collected {self.good_frames} good frames per camera")
            return None

        # Re-project every camera's ideal exposure to now (the sky has moved
        # on since the last frames were analysed) and share one exposure.
        self.current_exposure = self._shared_exposure()

        # Stamp the request's start time so on_request_completed only
        # picks up frames written for this request.
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
        """Check if every camera seen so far has enough good flats."""
        return bool(self.cameras) and self.good_frames >= self.target_flats

    def _copy_and_stamp_flat(
        self,
        src: Path,
        dst: Path,
        extra_headers: dict,
    ) -> None:
        """Copy an SK-written FITS into the flats dir and add flat headers.

        The source FITS already carries SK's full per-frame metadata
        (camera, mount, site headers written by SK's data graph). We
        only need to stamp burr's flat-specific keys onto the image HDU
        (the compressed HDU when SK compressed the frame) and write the
        result to the canonical flats path.
        """
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        with fits.open(dst, mode="update") as hdul:
            image_hdus = [hdu for hdu in hdul if hdu.data is not None]
            header = (image_hdus[0] if image_hdus else hdul[0]).header
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

        In SK mode, each camera's data graph writes its FITS to disk
        asynchronously, after the collect task returns, and burr never
        sees the paths. We close the loop here by waiting for the flat
        frames written since the request started to land under
        ``sk_data_root`` and feeding each into ``process_exposure`` so
        every camera's exposure model and good-frame count advance.

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

        frame_paths = self._wait_for_sk_flats(request)
        if not frame_paths:
            logger.warning(
                f"No flat FITS found under {self.sk_data_root} for request "
                f"{request.task_id} — exposure adjustment skipped"
            )
            return

        logger.info(f"Reading back {len(frame_paths)} SK-written flat(s) for analysis")
        for frame_path in frame_paths:
            self.process_exposure(str(frame_path), device_control=None)
        self._expected_cameras = max(self._expected_cameras, len(frame_paths))

        logger.info(
            "Per-camera flats: "
            + ", ".join(f"{cam}={state.good_frames}" for cam, state in sorted(self.cameras.items()))
        )

    def _wait_for_sk_flats(self, request: ObservationRequest) -> list[Path]:
        """Wait for the cameras' flat frames for ``request`` to land, then return them.

        Polls until as many frames as the previous request produced have
        appeared and the set has been stable for a couple of seconds, or
        ``frame_settle_seconds`` elapses.
        """
        deadline = time.monotonic() + self.frame_settle_seconds
        found: list[Path] = []
        stable_since = time.monotonic()
        while True:
            current = self._find_sk_flats(request)
            if len(current) != len(found):
                found, stable_since = current, time.monotonic()
            settled = time.monotonic() - stable_since >= 2.0
            if found and settled and len(found) >= self._expected_cameras:
                return found
            if time.monotonic() >= deadline:
                return found
            time.sleep(0.5)

    def _find_sk_flats(self, request: ObservationRequest) -> list[Path]:
        """Locate the flat FITS the SK data graph wrote for this request.

        Matches on content rather than filename (the controller's
        ``FileNameTemplate`` may override burr's): a ``*.fits`` file under
        ``sk_data_root`` whose mtime is at or after the request start and
        whose image header has ``IMAGETYP = flat``. Excludes burr's own
        processed copies and in-progress ``.tmp`` writes.
        """
        if self.sk_data_root is None or not self.sk_data_root.exists():
            return []

        cutoff = self._request_started_at
        output_root = Path(self.run.output_directory).resolve()
        frames = []
        for path in self.sk_data_root.rglob("*.fits"):
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=ZoneInfo("UTC"))
            except OSError:
                continue
            if cutoff is not None and mtime < cutoff:
                continue
            if path.resolve().is_relative_to(output_root):
                continue
            if _is_flat_frame(path):
                frames.append(path)

        return sorted(frames)

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
            if data.dtype == np.uint8:
                threshold = 255
            elif np.issubdtype(data.dtype, np.integer):
                # SK writes int32 even for 12/14/16-bit sensors (the ASI183's
                # 12-bit ADC is scaled to 16 bits, saturating at 65520), so
                # take the full scale of the smallest bit depth that holds the
                # frame's maximum and allow 0.5% headroom below it.
                bits = max(int(np.ceil(np.log2(float(np.max(data)) + 1.0))), 8)
                threshold = int((2**bits - 1) * 0.995)
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
