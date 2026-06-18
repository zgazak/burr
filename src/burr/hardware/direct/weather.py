"""
Weather monitoring extracted from runner.py.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from loguru import logger

from burr.hardware.direct.device import DirectDeviceControl
from burr.models.run import BurrRun


class WeatherMonitor:
    """Monitors weather conditions and manages dome safety."""

    def __init__(self, device_control: DirectDeviceControl, run: BurrRun):
        self.device_control = device_control
        self.run = run
        self._monitoring_task: Optional[asyncio.Task] = None
        self._weather_bad_since: Optional[datetime] = None
        self._weather_good_since: Optional[datetime] = None
        self._operations_suspended = False

    @property
    def is_suspended(self) -> bool:
        """Check if operations are suspended due to weather."""
        return self._operations_suspended

    async def start(self) -> None:
        """Start weather monitoring loop."""
        if self._monitoring_task is not None:
            logger.warning("Weather monitoring already started")
            return
        logger.info("Starting weather monitoring")
        self._monitoring_task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """Stop weather monitoring loop."""
        if self._monitoring_task is not None:
            logger.info("Stopping weather monitoring")
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

    async def _monitor_loop(self) -> None:
        """Main weather monitoring loop."""
        try:
            while True:
                await self._check_conditions()
                check_interval = self.run.config.weather.check_interval_seconds
                await asyncio.sleep(check_interval)
        except asyncio.CancelledError:
            logger.info("Weather monitoring cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in weather monitoring loop: {e}")
            raise

    async def _check_conditions(self) -> None:
        """Check weather and take appropriate action."""
        try:
            if not self.device_control:
                return

            weather_status = self.device_control.check_weather_conditions()
            if weather_status["should_close_dome"]:
                await self._handle_bad_weather(weather_status)
            else:
                await self._handle_good_weather(weather_status)
        except Exception as e:
            logger.error(f"Error checking weather conditions: {e}")

    async def _handle_bad_weather(self, weather_status: dict) -> None:
        """Handle bad weather - close dome and suspend operations."""
        issues = weather_status["issues"]
        logger.warning(f"Bad weather detected: {', '.join(issues)}")

        if self._weather_bad_since is None:
            self._weather_bad_since = datetime.now(ZoneInfo("UTC"))
            self._weather_good_since = None

        if self.device_control and self.device_control.dome.active:
            try:
                if not self.device_control.dome.is_closed():
                    logger.warning("Closing dome due to bad weather")
                    await self.device_control.close_dome()
                    self.run.log_command(
                        "weather_dome_closed",
                        f"Dome closed due to bad weather: {', '.join(issues)}",
                        metadata={
                            "weather_issues": issues,
                            "rain_rate": weather_status.get("rain_rate"),
                            "humidity": weather_status.get("humidity"),
                        },
                    )
            except Exception as e:
                logger.error(f"Failed to close dome for weather: {e}")

        if not self._operations_suspended:
            logger.warning("Suspending operations due to bad weather")
            self._operations_suspended = True
            self.run.log_command(
                "weather_operations_suspended",
                f"Operations suspended due to bad weather: {', '.join(issues)}",
                metadata={
                    "weather_issues": issues,
                    "rain_rate": weather_status.get("rain_rate"),
                    "humidity": weather_status.get("humidity"),
                },
            )

    async def _handle_good_weather(self, weather_status: dict) -> None:
        """Handle good weather - track recovery timer."""
        if weather_status["status"] == "simulated":
            if self._weather_bad_since is not None or self._weather_good_since is not None:
                logger.info("Simulated weather detected, resetting recovery timer")
                self._weather_bad_since = None
                self._weather_good_since = None
                self._operations_suspended = False
            return

        if self._weather_good_since is None:
            self._weather_good_since = datetime.now(ZoneInfo("UTC"))
            logger.info("Weather conditions improved, starting recovery timer")
            self.run.log_command(
                "weather_improved",
                "Weather conditions improved, starting recovery timer",
                metadata={
                    "rain_rate": weather_status.get("rain_rate"),
                    "humidity": weather_status.get("humidity"),
                    "recovery_duration_minutes": self.run.config.weather.recovery_duration_minutes,
                },
            )
        else:
            recovery_duration = timedelta(minutes=self.run.config.weather.recovery_duration_minutes)
            time_since_good = datetime.now(ZoneInfo("UTC")) - self._weather_good_since

            if time_since_good >= recovery_duration:
                self._weather_bad_since = None
                self._weather_good_since = None
                logger.info("Weather recovery period completed")
            else:
                remaining = recovery_duration - time_since_good
                logger.info(
                    f"Weather good for {time_since_good.total_seconds() / 60:.1f} minutes, "
                    f"need {remaining.total_seconds() / 60:.1f} more minutes"
                )

        if self._operations_suspended:
            logger.info("Resuming operations - weather conditions are good")
            self._operations_suspended = False
            self.run.log_command(
                "weather_operations_resumed",
                "Operations resumed - weather conditions are good",
                metadata={
                    "rain_rate": weather_status.get("rain_rate"),
                    "humidity": weather_status.get("humidity"),
                },
            )

    def get_status(self) -> str:
        """Get weather monitoring status string."""
        if self._operations_suspended:
            return "SUSPENDED"
        elif self._weather_bad_since is not None:
            if self._weather_good_since:
                time_since_good = (datetime.now(ZoneInfo("UTC")) - self._weather_good_since).total_seconds() / 60
                recovery_needed = self.run.config.weather.recovery_duration_minutes
                return f"RECOVERING ({time_since_good:.1f}/{recovery_needed}m)"
            else:
                return "WAITING_FOR_IMPROVEMENT"
        return "NORMAL"
