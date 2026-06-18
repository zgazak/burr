"""
Direct task executor - pattern-matched dispatch on Target type.

Extracted from runner.py's _execute_single_task method.
Executes one ObservationRequest at a time via DirectDeviceControl.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from loguru import logger

from burr.hardware.direct.device import DirectDeviceControl
from burr.hardware.direct.fits_writer import generate_fits_filename, build_frame_headers
from burr.models.observation import (
    ObservationRequest,
    RADecTarget,
    AltAzTarget,
    TLETarget,
    LunarTarget,
)
from burr.models.tracking import TrackingMode, Rates
from burr.models.run import BurrRun


class DirectTaskExecutor:
    """Executes ObservationRequests via direct ASCOM hardware control.

    Uses pattern matching on Target type to dispatch to the appropriate
    slew/track method.
    """

    def __init__(
        self,
        device_control: DirectDeviceControl,
        run: BurrRun,
        flat_processor: Optional[Callable] = None,
    ):
        """
        Parameters
        ----------
        device_control : DirectDeviceControl
            Hardware control interface.
        run : BurrRun
            Current observation run.
        flat_processor : callable, optional
            Callback for flat field processing: (frame_path, device_control) -> bool
        """
        self.device_control = device_control
        self.run = run
        self.flat_processor = flat_processor

    async def execute(
        self,
        request: ObservationRequest,
        shutdown_check: Callable[[], bool] | None = None,
    ) -> bool:
        """Execute a complete ObservationRequest.

        Iterates through exposure_seconds, and for each exposure iterates
        through the exposure_map frames.

        Parameters
        ----------
        request : ObservationRequest
            The observation request to execute.
        shutdown_check : callable, optional
            Returns True if shutdown has been requested.

        Returns
        -------
        bool
            True if all exposures completed successfully.
        """
        if not self.device_control:
            logger.error("Device control not initialized")
            return False

        should_stop = shutdown_check or (lambda: False)
        collect_id = str(uuid.uuid4())
        start_time = datetime.now(ZoneInfo("UTC"))
        success = False

        try:
            target = request.target
            total_exposures = request.total_exposures

            for exp_num, exposure_time in enumerate(request.exposure_seconds, 1):
                if should_stop():
                    logger.info("Shutdown requested, stopping execution")
                    return False

                # Resolve concrete rates for this exposure time
                concrete_rates = None
                if request.rates is not None:
                    if hasattr(request.rates, "to_concrete_rates"):
                        concrete_rates = request.rates.to_concrete_rates(exposure_time)
                    elif isinstance(request.rates, Rates):
                        concrete_rates = request.rates

                ok = await self._execute_single_exposure(
                    request=request,
                    target=target,
                    exposure_time=exposure_time,
                    exposure_num=exp_num,
                    total_exposures=total_exposures,
                    collect_id=collect_id,
                    rates=concrete_rates,
                    should_stop=should_stop,
                )

                if not ok:
                    return False

            success = True
            return True

        except Exception as e:
            logger.error(f"Error executing request: {e}")
            logger.exception(e)
            return False

        finally:
            end_time = datetime.now(ZoneInfo("UTC"))
            duration = (end_time - start_time).total_seconds()

            if success:
                self.run.log_command(
                    f"{request.source_name}_task_completed",
                    f"Task completed in {duration:.1f}s",
                    metadata={
                        "task_id": request.task_id,
                        "collect_id": collect_id,
                        "task_duration": duration,
                    },
                )
            else:
                self.run.log_command(
                    f"{request.source_name}_task_failed",
                    f"Task failed after {duration:.1f}s",
                    metadata={
                        "task_id": request.task_id,
                        "collect_id": collect_id,
                        "task_duration": duration,
                    },
                )

    async def _execute_single_exposure(
        self,
        request: ObservationRequest,
        target,
        exposure_time: float,
        exposure_num: int,
        total_exposures: int,
        collect_id: str,
        rates: Rates | None,
        should_stop: Callable[[], bool],
    ) -> bool:
        """Execute frames for a single exposure time within a request."""

        total_frames = len(request.exposure_map)

        match target:
            case RADecTarget():
                return await self._execute_radec(
                    request, target, exposure_time, exposure_num,
                    total_exposures, total_frames, collect_id, rates, should_stop,
                )
            case LunarTarget():
                # LunarTarget extends AltAzTarget - must match before AltAzTarget
                return await self._execute_altaz(
                    request, target, exposure_time, exposure_num,
                    total_exposures, total_frames, collect_id, rates, should_stop,
                )
            case AltAzTarget():
                return await self._execute_altaz(
                    request, target, exposure_time, exposure_num,
                    total_exposures, total_frames, collect_id, rates, should_stop,
                )
            case TLETarget():
                return await self._execute_tle(
                    request, target, exposure_time, exposure_num,
                    total_exposures, total_frames, collect_id, rates, should_stop,
                )
            case _:
                logger.error(f"Unknown target type: {type(target)}")
                return False

    async def _execute_radec(
        self, request, target: RADecTarget, exposure_time, exposure_num,
        total_exposures, total_frames, collect_id, rates, should_stop,
    ) -> bool:
        """Execute RA/Dec observation - slew once, then iterate frames."""
        if should_stop():
            return False

        logger.info(f"Slewing to RA={target.right_ascension_deg}°, Dec={target.declination_deg}°")
        slew_metadata = await self.device_control.goto_ra_dec(
            target.right_ascension_deg, target.declination_deg, TrackingMode.OFF, None
        )
        self._log_slew(slew_metadata)

        return await self._capture_frames(
            request, exposure_time, exposure_num, total_exposures,
            total_frames, collect_id, rates, should_stop,
        )

    async def _execute_altaz(
        self, request, target: AltAzTarget, exposure_time, exposure_num,
        total_exposures, total_frames, collect_id, rates, should_stop,
    ) -> bool:
        """Execute Alt/Az observation - slew once, then iterate frames."""
        if should_stop():
            return False

        logger.info(f"Slewing to Alt={target.altitude_deg}°, Az={target.azimuth_deg}°")
        slew_metadata = await self.device_control.goto_alt_az(
            target.altitude_deg, target.azimuth_deg, TrackingMode.OFF, None
        )
        self._log_slew(slew_metadata)

        is_flat = request.source_name == "twilight_flats"
        return await self._capture_frames(
            request, exposure_time, exposure_num, total_exposures,
            total_frames, collect_id, rates, should_stop,
            process_as_flat=is_flat,
        )

    async def _execute_tle(
        self, request, target: TLETarget, exposure_time, exposure_num,
        total_exposures, total_frames, collect_id, rates, should_stop,
    ) -> bool:
        """Execute TLE tracking observation."""
        if should_stop():
            return False

        previous_tracking_mode = None

        for frame_num, frame_spec in enumerate(request.exposure_map, 1):
            if should_stop():
                return False

            tracking_mode = frame_spec.tracking_mode
            logger.info(f"Executing TLE frame {frame_num}/{total_frames} mode={tracking_mode}")

            slew_metadata = await self.device_control.track_tle(
                target.tle_line1, target.tle_line2,
                tracking_mode, rates, previous_tracking_mode,
            )
            self._log_slew(slew_metadata)
            previous_tracking_mode = tracking_mode

            extra_headers = build_frame_headers(
                request.task_id, collect_id, frame_num, total_frames,
                exposure_num, total_exposures, request.source_name,
            )

            filename = generate_fits_filename(
                request.source_name, request.task_id, collect_id,
                tracking_mode.value, frame_num, total_frames,
                exposure_num, total_exposures,
            )
            output_path = Path(self.run.output_directory) / "data" / filename

            success = await self.device_control.take_exposure(exposure_time, str(output_path))
            if not success:
                logger.error(f"Failed to take TLE exposure {exposure_time}s")
                return False

            self.device_control.save_fits(
                output_path=output_path,
                tracking_mode=tracking_mode,
                rates=rates,
                extra_headers=extra_headers,
            )

        return True

    async def _capture_frames(
        self, request, exposure_time, exposure_num, total_exposures,
        total_frames, collect_id, rates, should_stop,
        process_as_flat: bool = False,
    ) -> bool:
        """Capture frames according to the exposure map."""

        for frame_num, frame_spec in enumerate(request.exposure_map, 1):
            if should_stop():
                return False

            tracking_mode = frame_spec.tracking_mode
            logger.info(f"Executing frame {frame_num}/{total_frames} mode={tracking_mode}")

            await self.device_control._set_tracking_mode(tracking_mode, rates)

            extra_headers = build_frame_headers(
                request.task_id, collect_id, frame_num, total_frames,
                exposure_num, total_exposures, request.source_name,
            )

            filename = generate_fits_filename(
                request.source_name, request.task_id, collect_id,
                tracking_mode.value, frame_num, total_frames,
                exposure_num, total_exposures,
            )
            output_path = Path(self.run.output_directory) / "data" / filename

            if process_as_flat and self.flat_processor:
                # Flat field: take to temp, analyze, save if good
                temp_path = output_path.with_suffix(".tmp.fits")
                temp_path = temp_path.resolve()
                temp_path.parent.mkdir(parents=True, exist_ok=True)

                success = await self.device_control.take_exposure(exposure_time, str(temp_path))
                if not success:
                    logger.error(f"Failed to take flat exposure {exposure_time}s")
                    return False

                is_good = self.flat_processor(str(temp_path), self.device_control)
                if is_good:
                    output_path = output_path.resolve()
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    self.device_control.save_fits(
                        output_path=output_path,
                        tracking_mode=tracking_mode,
                        rates=rates,
                        extra_headers=extra_headers,
                    )
                temp_path.unlink(missing_ok=True)
            else:
                success = await self.device_control.take_exposure(exposure_time, str(output_path))
                if not success:
                    logger.error(f"Failed to take exposure {exposure_time}s mode={tracking_mode}")
                    return False

                self.device_control.save_fits(
                    output_path=output_path,
                    tracking_mode=tracking_mode,
                    rates=rates,
                    extra_headers=extra_headers,
                )

        return True

    def _log_slew(self, slew_metadata) -> None:
        """Log slew completion."""
        if slew_metadata:
            self.run.log_command(
                "slew_completed",
                f"Slew completed in {slew_metadata.duration_seconds:.2f}s",
                metadata={
                    "slew_type": slew_metadata.slew_type,
                    "duration_seconds": slew_metadata.duration_seconds,
                    "start_alt_deg": slew_metadata.start_alt_deg,
                    "start_az_deg": slew_metadata.start_az_deg,
                    "end_alt_deg": slew_metadata.end_alt_deg,
                    "end_az_deg": slew_metadata.end_az_deg,
                },
            )
