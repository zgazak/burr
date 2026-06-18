"""
Rewritten DirectRunner - thin orchestrator (~200 lines).

Uses Scheduler, DirectTaskExecutor, WeatherMonitor instead of
inlining all logic in one 1600-line file.
"""

import asyncio
import os
import signal
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from loguru import logger

from burr.core.config import get_or_initialize_config
from burr.core.logging import setup_loguru
from burr.hardware.direct.device import DirectDeviceControl
from burr.hardware.direct.executor import DirectTaskExecutor
from burr.hardware.direct.weather import WeatherMonitor
from burr.hardware.factory import create_devices
from burr.models.run import BurrRun
from burr.models.site import LightingCondition
from burr.scheduler.scheduler import Scheduler
from burr.scheduler.factory import build_scheduler
from burr.task_source.protocol import TaskSource

# Configure matplotlib cache directory
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"


class DirectRunner:
    """Direct hardware control runner using the new architecture.

    Orchestrates Scheduler, DirectTaskExecutor, and WeatherMonitor.
    """

    def __init__(self, run: BurrRun, config_path: Path | list[Path]):
        self.run = run
        self.config_path = config_path
        self.config = get_or_initialize_config(config_path)
        self.device_control: Optional[DirectDeviceControl] = None
        self.scheduler: Optional[Scheduler] = None
        self.executor: Optional[DirectTaskExecutor] = None
        self.weather_monitor: Optional[WeatherMonitor] = None
        self._running = False
        self._shutdown_requested = False
        self._last_condition: Optional[LightingCondition] = None
        self._dome_windows: list[LightingCondition] = []
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        def handler(signum, frame):
            if not self._shutdown_requested:
                logger.info(f"Received signal {signum}, requesting shutdown...")
                self._shutdown_requested = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    async def initialize_hardware(self):
        """Initialize and connect to all hardware devices."""
        setup_loguru(level=self.config.logging.level, run_output_dir=self.run.output_directory)

        devices = create_devices(self.config)
        self.device_control = DirectDeviceControl(
            camera=devices.camera,
            dome=devices.dome,
            mount=devices.mount,
            weather=devices.weather,
            config=self.config,
        )

        try:
            await self.device_control.connect()
        except Exception as e:
            logger.error(f"Failed to connect to devices: {e}")
            self.device_control = None
            raise RuntimeError(f"Failed to initialize hardware: {e}")

        # Configure dome windows
        if self.config.hardware and self.config.hardware.dome and self.config.hardware.dome.open_windows:
            self._dome_windows = [
                LightingCondition(w.lower()) for w in self.config.hardware.dome.open_windows
            ]

        # Initial dome state
        condition = self._get_condition()
        if condition and self._dome_windows:
            if condition in self._dome_windows:
                await self.device_control.open_dome()
            else:
                await self.device_control.close_dome()

        logger.info("Hardware initialization complete")

    async def initialize_task_sources(self):
        """Initialize task sources and build the scheduler."""
        sources: dict[str, TaskSource] = {}

        # Import task sources
        from burr.task_source.flats.manager import FlatFieldManager
        from burr.task_source.calsats.manager import CalibrationSatelliteManager
        from burr.task_source.photometry.manager import PhotometricStandardsManager
        from burr.task_source.coverage.manager import SkyCoverageManager
        from burr.task_source.lunar.manager import LunarBackgroundManager

        sched = self.config.schedule

        if sched.twilight_flats.collect:
            sources["twilight_flats"] = FlatFieldManager(run=self.run)

        if sched.lunar_background.collect:
            sources["lunar_background"] = LunarBackgroundManager(run=self.run)

        if sched.photometric_standards.collect:
            sources["photometric_standards"] = PhotometricStandardsManager(run=self.run)

        if sched.coverage.collect:
            sources["coverage"] = SkyCoverageManager(run=self.run)

        if sched.calsats.collect:
            sources["calsats"] = CalibrationSatelliteManager(run=self.run)

        self.scheduler = build_scheduler(self.config, self.run, sources)

        # Set up executor with flat processor callback
        flat_source = sources.get("twilight_flats")
        flat_processor = flat_source.process_exposure if flat_source and hasattr(flat_source, "process_exposure") else None

        self.executor = DirectTaskExecutor(
            device_control=self.device_control,
            run=self.run,
            flat_processor=flat_processor,
        )

        self.weather_monitor = WeatherMonitor(
            device_control=self.device_control,
            run=self.run,
        )

        logger.info(f"Initialized {len(sources)} task sources")

    def _get_condition(self) -> Optional[LightingCondition]:
        """Get current lighting condition."""
        try:
            now = datetime.now(ZoneInfo("UTC"))
            return self.run.lighting_schedule.get_lighting_condition(now)
        except Exception as e:
            logger.warning(f"Could not determine lighting condition: {e}")
            return None

    async def _handle_lighting_transition(self, old: Optional[LightingCondition], new: Optional[LightingCondition]):
        """Handle dome when lighting conditions change."""
        if not self.device_control or not self.device_control.dome.active:
            return
        if new is None:
            await self.device_control.close_dome()
            return
        if old is None:
            return

        old_open = old in self._dome_windows if self._dome_windows else True
        new_open = new in self._dome_windows if self._dome_windows else True

        if old_open != new_open:
            if new_open:
                logger.info(f"Opening dome for {new} condition")
                await self.device_control.open_dome()
            else:
                logger.info(f"Closing dome for {new} condition")
                await self.device_control.close_dome()

    async def execute_next_task(self) -> bool:
        """Execute one task from the best available source."""
        if self._shutdown_requested or not self.device_control or not self.scheduler:
            return False

        if self.weather_monitor and self.weather_monitor.is_suspended:
            await asyncio.sleep(10)
            return False

        condition = self._get_condition()
        if not condition:
            await asyncio.sleep(10)
            return False

        # Handle lighting transitions
        if condition != self._last_condition:
            logger.info(f"Lighting condition changed to: {condition.value.upper()}")
            await self._handle_lighting_transition(self._last_condition, condition)
            self._last_condition = condition

        candidates = self.scheduler.ranked_suitable(condition)
        if not candidates:
            logger.debug(
                f"no suitable task source for condition={condition.value}"
            )
            await asyncio.sleep(1)
            return False

        binning = self.device_control.camera.bin_x if self.device_control.camera else 1

        # Walk suitable slots in priority order. A slot whose generate_request
        # returns None this cycle (e.g. no calsat currently above the horizon)
        # shouldn't burn the whole iteration — try the next-best slot before
        # giving up the cycle, so lower-priority sources don't starve.
        slot = None
        request = None
        for candidate in candidates:
            task_id = str(uuid.uuid4())
            try:
                req = candidate.source.generate_request(
                    task_id=task_id, binning=binning
                )
            except Exception as e:
                logger.exception(
                    f"{candidate.name}.generate_request raised: {e} — "
                    "recording failure, trying next slot"
                )
                candidate.record_failure()
                continue

            if req is None:
                logger.info(
                    f"{candidate.name}.generate_request returned None — "
                    "trying next slot"
                )
                candidate.record_failure()
                continue

            slot = candidate
            request = req
            break

        if slot is None or request is None:
            logger.info(
                "every suitable slot returned None this cycle — nothing to run"
            )
            await asyncio.sleep(1)
            return False

        logger.info(
            f"selected {slot.name} → executing request {request.task_id} "
            f"target={type(request.target).__name__} "
            f"exposures={len(request.exposure_seconds)}"
        )

        start_time = datetime.now(ZoneInfo("UTC"))
        success = await self.executor.execute(
            request, shutdown_check=lambda: self._shutdown_requested
        )
        duration = (datetime.now(ZoneInfo("UTC")) - start_time).total_seconds()

        if success:
            slot.record_success(duration, condition)
            slot.source.on_request_completed(request, duration)
            offer_intervals = self.run.lighting_schedule.merged_windows_for(
                {c for s in self.scheduler.slots for c in s.source.suitable_conditions}
            )
            self.scheduler.log_status(condition, offer_intervals=offer_intervals)
        else:
            slot.record_failure()
            slot.source.on_request_failed(request, "Execution failed")

        return success

    async def start(self):
        """Main entry point - setup, operate, shutdown."""
        heartbeat_task = None
        try:
            # Setup
            logger.info("=== STAGE 1: SETUP ===")
            self.run.current_stage = "setup"
            self.run.status = "running"

            await self.initialize_hardware()
            await self.initialize_task_sources()
            await self.weather_monitor.start()

            # Operations
            logger.info("=== STAGE 2: OPERATIONS ===")
            self.run.current_stage = "operations"
            self._running = True
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            while self._running and not self._shutdown_requested:
                try:
                    await self.execute_next_task()
                except Exception as e:
                    logger.error(f"Error in execute_next_task: {e}")
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in runner: {e}")
            raise
        finally:
            # Shutdown
            logger.info("=== STAGE 3: SHUTDOWN ===")
            self.run.current_stage = "shutdown"
            self._running = False

            if self.weather_monitor:
                await self.weather_monitor.stop()

            if heartbeat_task:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

            if self.device_control:
                await self.device_control.park_mount()
                await self.device_control.close_dome()
                await self.device_control.disconnect()

            self.run.status = "completed"
            logger.info("Runner stopped")

    async def stop(self):
        """Stop the runner."""
        self._running = False
        if self.weather_monitor:
            await self.weather_monitor.stop()
        if self.device_control:
            await self.device_control.disconnect()

    async def _heartbeat_loop(self):
        """Periodic status logging."""
        while self._running:
            try:
                condition = self._get_condition()
                condition_str = condition.value.upper() if condition else "UNKNOWN"
                now = datetime.now(ZoneInfo("UTC"))
                weather_status = self.weather_monitor.get_status() if self.weather_monitor else "N/A"
                logger.info(
                    f"Heartbeat - UTC: {now.strftime('%H:%M')}, "
                    f"Condition: {condition_str}, Weather: {weather_status}"
                )
            except Exception as e:
                logger.error(f"Error in heartbeat: {e}")
            await asyncio.sleep(300)
