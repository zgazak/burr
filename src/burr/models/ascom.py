"""ASCOM device implementations."""

from alpaca.dome import Dome, ShutterState
from alpaca.camera import CameraStates, SensorType
from alpaca.observingconditions import ObservingConditions
import asyncio
from typing import Optional, List, Any
import sys
import argparse
from pathlib import Path
from pydantic import BaseModel, Field
from burr.core.notifications import notifier, NotificationLevel
from burr.core.config import get_or_initialize_config
from burr.models.hardware import DomeConfig, PWI4Config, WeatherConfig
from loguru import logger
from requests.exceptions import ReadTimeout
from burr.hardware.direct.pwi4_client import PWI4
import numpy as np
from burr.core.logging import setup_loguru
import time
from datetime import datetime


class SlewMetadata(BaseModel):
    """Metadata for mount slew operations."""

    # Timing
    start_time: datetime = Field(..., description="When the slew started")
    end_time: datetime = Field(..., description="When the slew completed")
    duration_seconds: float = Field(..., description="Total slew duration in seconds")

    # Start position
    start_alt_deg: float = Field(..., description="Starting altitude in degrees")
    start_az_deg: float = Field(..., description="Starting azimuth in degrees")
    start_ra_hours: Optional[float] = Field(None, description="Starting RA in hours (if available)")
    start_dec_deg: Optional[float] = Field(None, description="Starting declination in degrees (if available)")

    # End position
    end_alt_deg: float = Field(..., description="Final altitude in degrees")
    end_az_deg: float = Field(..., description="Final azimuth in degrees")
    end_ra_hours: Optional[float] = Field(None, description="Final RA in hours (if available)")
    end_dec_deg: Optional[float] = Field(None, description="Final declination in degrees (if available)")

    # Target information
    target_ra_hours: Optional[float] = Field(None, description="Target RA in hours (for RA/Dec slews)")
    target_dec_deg: Optional[float] = Field(None, description="Target declination in degrees (for RA/Dec slews)")
    target_alt_deg: Optional[float] = Field(None, description="Target altitude in degrees (for Alt/Az slews)")
    target_az_deg: Optional[float] = Field(None, description="Target azimuth in degrees (for Alt/Az slews)")

    # Slew type
    slew_type: str = Field(..., description="Type of slew: 'ra_dec', 'alt_az', or 'tle'")

    # Distance calculations
    alt_distance_deg: float = Field(..., description="Total altitude distance traveled in degrees")
    az_distance_deg: float = Field(..., description="Total azimuth distance traveled in degrees")

    def __post_init__(self):
        """Calculate distances after initialization."""
        self.alt_distance_deg = abs(self.end_alt_deg - self.start_alt_deg)
        self.az_distance_deg = abs(self.end_az_deg - self.start_az_deg)


class DomeEntity(DomeConfig):
    """ASCOM Dome device implementation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._device = None

    @property
    def device(self) -> Dome:
        """Returns an Alpaca Dome instance configured with this entity's parameters."""
        if self._device is None:
            self._device = Dome(f"{self.address}:{self.port}", self.device_number)
        return self._device

    def connect(self):
        """Connect to the dome device."""
        self.device.Connected = True

    def disconnect(self):
        """Disconnect from the dome device."""
        if self._device is not None:
            self.device.Connected = False
            self._device = None

    def is_open(self) -> bool:
        """Returns True if the dome is open, False otherwise."""
        return self.device.ShutterStatus == ShutterState.shutterOpen

    def is_closed(self) -> bool:
        """Returns True if the dome is closed, False otherwise."""
        return self.device.ShutterStatus == ShutterState.shutterClosed

    def is_moving(self) -> bool:
        """Returns True if the dome is moving, False otherwise."""
        return self.device.ShutterStatus in [ShutterState.shutterOpening, ShutterState.shutterClosing]

    def is_error(self) -> bool:
        """Returns True if the dome is parked, False otherwise."""
        return self.device.ShutterStatus == ShutterState.shutterError

    def open(self):
        """Open the dome."""
        try:
            self.device.OpenShutter()
        except ReadTimeout:
            pass

    def close(self):
        """Close the dome."""
        try:
            self.device.CloseShutter()
        except ReadTimeout:
            pass

    async def monitor_dome(
        self,
        slack_channel: Optional[str] = None,
        interval: float = 10.0,
        stop_event: Optional[asyncio.Event] = None,
    ):
        """
        Continuously monitor the dome for state changes.

        Args:
            slack_channel: Optional Slack channel to send notifications to
            interval: How often to check dome state in seconds
            stop_event: Optional event to stop monitoring. If not provided, runs indefinitely
        """
        last_error_state = None
        last_shutter_state = None

        while True:
            if stop_event and stop_event.is_set():
                break

            # Check error state
            current_error_state = self.is_error()
            if current_error_state != last_error_state:
                if current_error_state:
                    notifier.notify(
                        message="Dome has entered an error state",
                        level=NotificationLevel.ERROR,
                        title="Dome Error",
                        channel=slack_channel,
                    )
                else:
                    notifier.notify(
                        message="Dome has recovered from error state",
                        level=NotificationLevel.SUCCESS,
                        title="Dome Recovery",
                        channel=slack_channel,
                    )
                last_error_state = current_error_state

            # Check shutter state
            current_shutter_state = self.device.ShutterStatus
            if current_shutter_state != last_shutter_state and not current_error_state:
                if last_shutter_state is not None:  # Skip first state change detection
                    if current_shutter_state == ShutterState.shutterOpen:
                        notifier.notify(
                            message="Dome shutter has opened",
                            level=NotificationLevel.INFO,
                            title="Dome Status",
                            channel=slack_channel,
                        )
                    elif current_shutter_state == ShutterState.shutterClosed:
                        notifier.notify(
                            message="Dome shutter has closed",
                            level=NotificationLevel.INFO,
                            title="Dome Status",
                            channel=slack_channel,
                        )
                    elif current_shutter_state in [ShutterState.shutterOpening, ShutterState.shutterClosing]:
                        action = "opening" if current_shutter_state == ShutterState.shutterOpening else "closing"
                        notifier.notify(
                            message=f"Dome shutter is {action}",
                            level=NotificationLevel.INFO,
                            title="Dome Status",
                            channel=slack_channel,
                        )
                last_shutter_state = current_shutter_state

            await asyncio.sleep(interval)


class CameraEntity(BaseModel):
    """ASCOM camera device entity."""

    address: str
    port: int
    device_number: int = 0
    active: bool = True
    default_binning: int = 1
    binning: List[int] = Field(default_factory=list)
    readout: List[int | str] = Field(default_factory=list)
    _device: Optional[Any] = None
    _image_ready: bool = False
    _image_array: Optional[np.ndarray] = None
    _last_exposure_duration: Optional[float] = None
    _last_exposure_start_time: Optional[str] = None

    @property
    def device(self) -> Any:
        """Get the ASCOM device instance."""
        if self._device is None:
            raise RuntimeError("Camera not connected")
        return self._device

    @property
    def image_ready(self) -> bool:
        """Check if an image is ready."""
        return self._image_ready

    @image_ready.setter
    def image_ready(self, value: bool):
        """Set the image ready flag."""
        self._image_ready = value

    @property
    def image_array(self) -> Optional[np.ndarray]:
        """Get the current image array."""
        return self._image_array

    @image_array.setter
    def image_array(self, value: np.ndarray):
        """Set the image array."""
        self._image_array = value

    @property
    def last_exposure_duration(self) -> Optional[float]:
        """Get the last exposure duration."""
        return self._last_exposure_duration

    @last_exposure_duration.setter
    def last_exposure_duration(self, value: float):
        """Set the last exposure duration."""
        self._last_exposure_duration = value

    @property
    def last_exposure_start_time(self) -> Optional[str]:
        """Get the last exposure start time."""
        return self._last_exposure_start_time

    @last_exposure_start_time.setter
    def last_exposure_start_time(self, value: str):
        """Set the last exposure start time."""
        self._last_exposure_start_time = value

    def connect(self):
        """Connect to the camera."""
        if not self.active:
            return

        try:
            # Import here to avoid dependency if not using ASCOM
            from alpaca.camera import Camera

            # Just pass address:port and device_number
            self._device = Camera(f"{self.address}:{self.port}", self.device_number)
            self._device.Connected = True

            # Check camera state and handle any ongoing exposures
            state = self.camera_state
            logger.info(f"Camera state on connect: {state}")

            if state == CameraStates.cameraExposing:
                logger.warning("Camera is currently exposing - attempting to abort")
                if self.can_abort_exposure:
                    self.abort_exposure()
                    logger.info("Successfully aborted ongoing exposure")
                else:
                    logger.warning("Camera does not support aborting exposures - waiting for completion")
                    while self.camera_state == CameraStates.cameraExposing:
                        logger.info(f"Waiting for exposure to complete ({self.percent_completed}% done)")
                        time.sleep(0.5)
            elif state == CameraStates.cameraError:
                logger.warning("Camera is in error state - attempting to reset")
                self._device.Connected = False
                time.sleep(1)
                self._device.Connected = True
                if self.camera_state == CameraStates.cameraError:
                    raise RuntimeError("Camera still in error state after reset attempt")

            # Set default binning
            if self.default_binning > 0:
                logger.info(f"Setting default binning to {self.default_binning}x{self.default_binning}")
                self.bin_x = self.default_binning
                self.bin_y = self.default_binning

            # Set up cooling
            logger.info("Setting up camera cooling")
            self._device.CoolerOn = False  # Turn off first to ensure clean state
            self._device.SetCCDTemperature = -10.0  # Set target temp to -10C
            self._device.CoolerOn = True  # Turn cooler on
            logger.info(
                f"Camera cooling initialized. Current temp: {self._device.CCDTemperature:.1f}°C, Target: -10.0°C"
            )

        except Exception as e:
            raise RuntimeError(f"Failed to connect to camera: {e}")

    def disconnect(self):
        """Disconnect from the camera."""
        if not self.active or self._device is None:
            return

        try:
            # Turn off cooling before disconnecting
            if self._device.CoolerOn:
                logger.info("Turning off camera cooling")
                self._device.CoolerOn = False

            self._device.Connected = False
            self._device = None
        except Exception as e:
            raise RuntimeError(f"Failed to disconnect from camera: {e}")

    @property
    def sensor_name(self) -> str:
        """Get the sensor name."""
        if not self.active:
            return "Simulated Sensor"
        return self.device.SensorName

    @property
    def bin_x(self) -> int:
        """Get the X binning factor."""
        if not self.active:
            return 1
        return self.device.BinX

    @bin_x.setter
    def bin_x(self, value: int):
        """Set the X binning factor."""
        if not self.active:
            return
        self.device.BinX = value

    @property
    def bin_y(self) -> int:
        """Get the Y binning factor."""
        if not self.active:
            return 1
        return self.device.BinY

    @bin_y.setter
    def bin_y(self, value: int):
        """Set the Y binning factor."""
        if not self.active:
            return
        self.device.BinY = value

    @property
    def gain(self) -> float:
        """Get the camera gain."""
        if not self.active:
            return 1.0
        return self.device.Gain

    @property
    def readout_mode(self) -> str:
        """Get the readout mode."""
        if not self.active:
            return "Normal"
        return self.device.ReadoutMode

    def start_exposure(self, duration: float, light_frame: bool = True):
        """Start an exposure.

        Args:
            duration: The exposure duration in seconds
            light_frame: True for light frames, False for dark frames

        Raises:
            RuntimeError: If the camera is not in a state to start an exposure
        """
        try:
            # Diagnostic checks
            logger.debug("Running pre-exposure diagnostic checks...")

            # Get full diagnostic info
            diag = self.get_diagnostic_info()
            logger.debug("Camera diagnostic information:")
            for category, info in diag.items():
                logger.debug(f"  {category}: {info}")

            # Check connection
            logger.debug("Checking camera connection...")
            if not self.is_connected:
                raise RuntimeError("Camera is not connected")
            logger.debug("Camera is connected")

            # Check camera state
            state = self.camera_state
            logger.debug(f"Current camera state: {state}")
            if state == CameraStates.cameraError:
                raise RuntimeError("Camera is in error state")
            elif state != CameraStates.cameraIdle:
                raise RuntimeError(f"Camera is not idle (state: {state})")
            logger.debug("Camera is in valid state for exposure")

            # Check if we're pulse guiding
            if self.is_pulse_guiding:
                raise RuntimeError("Cannot start exposure while pulse guiding")

            # Check exposure parameters
            logger.debug(f"Checking if exposure duration {duration}s is within limits...")
            if not (self.exposure_min <= duration <= self.exposure_max):
                raise RuntimeError(
                    f"Exposure duration {duration}s outside valid range ({self.exposure_min}s to {self.exposure_max}s)"
                )
            logger.debug("Exposure duration is within valid range")

            # Set ROI based on binning
            logger.debug("Setting ROI based on binning...")
            full_width = self.camera_x_size
            full_height = self.camera_y_size
            binned_width = full_width // self.bin_x
            binned_height = full_height // self.bin_y

            # Set ROI to full binned frame
            self.device.StartX = 0
            self.device.StartY = 0
            self.device.NumX = binned_width
            self.device.NumY = binned_height
            logger.debug(f"Set ROI to full binned frame: {binned_width}x{binned_height}")

            # Start the exposure
            logger.debug(f"Attempting to start {'light' if light_frame else 'dark'} exposure...")
            try:
                self.device.StartExposure(duration, light_frame)
                logger.info(
                    f"Successfully started {'light' if light_frame else 'dark'} exposure for {duration} seconds"
                )
            except Exception as e:
                logger.error(f"ASCOM driver error during StartExposure: {str(e)}")
                # Try to get more information about the camera state
                try:
                    state = self.camera_state
                    logger.debug(f"Camera state after failed start: {state}")
                    # Get updated diagnostics after failure
                    diag = self.get_diagnostic_info()
                    logger.debug("Camera diagnostic information after failure:")
                    for category, info in diag.items():
                        logger.debug(f"  {category}: {info}")
                except Exception:
                    pass
                raise

        except Exception as e:
            error_msg = f"Failed to start exposure: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def stop_exposure(self):
        """Stop the current exposure."""
        if self.device.CanStopExposure:
            self.device.StopExposure()

    def abort_exposure(self):
        """Abort the current exposure."""
        if self.device.CanAbortExposure:
            self.device.AbortExposure()

    @property
    def percent_completed(self) -> int:
        """Returns the percentage completion of the current operation."""
        return self.device.PercentCompleted

    @property
    def camera_state(self) -> CameraStates:
        """Returns the current camera state."""
        return self.device.CameraState

    @property
    def image_array_info(self):
        """Returns metadata about the last exposure's image data."""
        return self.device.ImageArrayInfo

    @property
    def sensor_type(self) -> SensorType:
        """Returns the type of the sensor used in the camera."""
        return self.device.SensorType

    @property
    def camera_x_size(self) -> int:
        """Returns the width of the camera sensor in pixels."""
        return self.device.CameraXSize

    @property
    def camera_y_size(self) -> int:
        """Returns the height of the camera sensor in pixels."""
        return self.device.CameraYSize

    @property
    def max_adu(self) -> int:
        """Returns the maximum ADU value the camera can produce."""
        return self.device.MaxADU

    @property
    def gains(self) -> List[str]:
        """Returns the valid gain values that this camera supports.

        Returns an empty list if the driver doesn't implement this property.
        """
        try:
            return self.device.Gains
        except Exception as e:
            if "not implemented" in str(e).lower():
                logger.debug("Camera driver does not implement Gains property")
                return []
            raise  # Re-raise other exceptions

    @property
    def readout_modes(self) -> List[str]:
        """Returns the valid readout modes that this camera supports.

        Returns an empty list if the driver doesn't implement this property.
        """
        try:
            return self.device.ReadoutModes
        except Exception as e:
            if "not implemented" in str(e).lower():
                logger.debug("Camera driver does not implement ReadoutModes property")
                return []
            raise  # Re-raise other exceptions

    async def monitor_exposure(self, slack_channel: Optional[str] = None, interval: float = 0.5):
        """Monitor an exposure until it completes.

        Args:
            slack_channel: Optional Slack channel to send notifications to
            interval: How often to check exposure state in seconds
        """
        while not self.image_ready:
            state = self.camera_state
            percent = self.percent_completed

            if state == CameraStates.cameraError:
                notifier.notify(
                    message="Camera has entered an error state during exposure",
                    level=NotificationLevel.ERROR,
                    title="Camera Error",
                    channel=slack_channel,
                )
                break

            if slack_channel:
                notifier.notify(
                    message=f"Exposure {percent}% complete",
                    level=NotificationLevel.INFO,
                    title="Camera Status",
                    channel=slack_channel,
                )

            await asyncio.sleep(interval)

        if self.image_ready:
            notifier.notify(
                message="Exposure complete, image ready",
                level=NotificationLevel.SUCCESS,
                title="Camera Status",
                channel=slack_channel,
            )

    @property
    def can_async_image(self) -> bool:
        """Returns whether the camera supports async imaging.

        Note: This is not a standard ASCOM property, so we assume True if not available.
        """
        try:
            return self.device.CanAsyncImaging
        except AttributeError:
            # This property isn't part of the standard ASCOM interface
            logger.debug("Camera driver does not have CanAsyncImaging property - assuming async capable")
            return True
        except Exception as e:
            logger.warning(f"Error checking async imaging capability: {e}")
            return True  # Assume it can if we can't check

    @property
    def exposure_min(self) -> float:
        """Returns the minimum exposure time in seconds."""
        try:
            return self.device.ExposureMin
        except Exception as e:
            if "not implemented" in str(e).lower():
                logger.debug("Camera driver does not implement ExposureMin property")
                return 0.0  # Assume no minimum if not specified
            raise

    @property
    def exposure_max(self) -> float:
        """Returns the maximum exposure time in seconds."""
        try:
            return self.device.ExposureMax
        except Exception as e:
            if "not implemented" in str(e).lower():
                logger.debug("Camera driver does not implement ExposureMax property")
                return float("inf")  # Assume no maximum if not specified
            raise

    @property
    def is_connected(self) -> bool:
        """Returns whether the camera is connected."""
        try:
            return self.device.Connected
        except Exception as e:
            logger.error(f"Failed to check camera connection: {e}")
            return False

    @property
    def can_pulse_guide(self) -> bool:
        """Returns whether the camera supports pulse guiding."""
        try:
            return self.device.CanPulseGuide
        except Exception:
            return False

    @property
    def can_stop_exposure(self) -> bool:
        """Returns whether the camera supports stopping exposures."""
        try:
            return self.device.CanStopExposure
        except Exception:
            return False

    @property
    def can_abort_exposure(self) -> bool:
        """Returns whether the camera supports aborting exposures."""
        try:
            return self.device.CanAbortExposure
        except Exception:
            return False

    @property
    def is_pulse_guiding(self) -> bool:
        """Returns whether the camera is currently pulse guiding."""
        try:
            return self.device.IsPulseGuiding
        except Exception:
            return False

    @property
    def has_shutter(self) -> bool:
        """Returns whether the camera has a mechanical shutter."""
        try:
            return self.device.HasShutter
        except Exception:
            return False

    def get_diagnostic_info(self) -> dict:
        """Get diagnostic information about the camera.

        Returns:
            dict: Dictionary of camera capabilities and state
        """
        info = {
            "connected": self.is_connected,
            "state": self.camera_state,
            "sensor": {
                "name": "Unknown",
                "type": "Unknown",
                "size": (0, 0),
                "max_adu": 0,
            },
            "capabilities": {
                "can_pulse_guide": self.can_pulse_guide,
                "can_stop_exposure": self.can_stop_exposure,
                "can_abort_exposure": self.can_abort_exposure,
                "has_shutter": self.has_shutter,
            },
            "exposure": {
                "min": self.exposure_min,
                "max": self.exposure_max,
                "is_pulse_guiding": self.is_pulse_guiding,
            },
            "binning": {
                "x": 1,
                "y": 1,
            },
        }

        # Try to get sensor info
        try:
            info["sensor"].update(
                {
                    "name": self.sensor_name,
                    "type": self.sensor_type,
                    "size": (self.camera_x_size, self.camera_y_size),
                    "max_adu": self.max_adu,
                }
            )
        except Exception as e:
            logger.debug(f"Could not get complete sensor info: {e}")

        # Try to get current binning
        try:
            info["binning"].update({"x": self.bin_x, "y": self.bin_y})
        except Exception as e:
            logger.debug(f"Could not get binning info: {e}")

        return info


class PWI4Entity(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8220


class PWI4MountEntity(PWI4Config):
    """PWI4 mount implementation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._device = None

    @property
    def device(self) -> PWI4:
        """Returns a PWI4 instance configured with this entity's parameters."""
        if self._device is None:
            self._device = PWI4(host=self.host, port=self.port)
        return self._device

    def connect(self):
        """Connect to the mount."""
        status = self.device.mount_connect()
        if not status.mount.is_connected:
            raise RuntimeError("Failed to connect to mount")
        logger.info("Connected to PWI4 mount")

    def disconnect(self):
        """Disconnect from the mount."""
        try:
            self.device.mount_disconnect()
            logger.info("Disconnected from PWI4 mount")
        except Exception as e:
            logger.error(f"Error disconnecting from mount: {e}")

    def goto_ra_dec_j2000(self, ra_hours: float, dec_degs: float):
        """Go to J2000 coordinates.

        Args:
            ra_hours: Right ascension in hours
            dec_degs: Declination in degrees
        """
        status = self.device.mount_goto_ra_dec_j2000(ra_hours, dec_degs)
        logger.info(f"Slewing to RA {ra_hours:.4f}h Dec {dec_degs:.4f}°")
        return status

    def goto_alt_az(self, alt_degs: float, az_degs: float):
        """Go to altitude/azimuth coordinates.

        Args:
            alt_degs: Altitude in degrees
            az_degs: Azimuth in degrees
        """
        status = self.device.mount_goto_alt_az(alt_degs, az_degs)
        logger.info(f"Slewing to Alt {alt_degs:.4f}° Az {az_degs:.4f}°")
        return status

    def stop(self):
        """Stop all mount motion."""
        status = self.device.mount_stop()
        logger.info("Stopped mount motion")
        return status

    def park(self):
        """Park the mount."""
        status = self.device.mount_park()
        logger.info("Parking mount")
        return status

    def tracking_on(self):
        """Enable mount tracking."""
        status = self.device.mount_tracking_on()
        logger.info("Enabled mount tracking")
        return status

    def tracking_off(self):
        """Disable mount tracking."""
        status = self.device.mount_tracking_off()
        logger.info("Disabled mount tracking")
        return status

    def mount_follow_tle(self, tle_line0: str, tle_line1: str, tle_line2: str):
        """Follow a satellite using TLE elements.

        Args:
            tle_line0: TLE line 0 (name)
            tle_line1: TLE line 1
            tle_line2: TLE line 2

        Returns:
            PWI4 status object
        """
        status = self.device.mount_follow_tle(tle_line0, tle_line1, tle_line2)
        logger.info(f"Following satellite with TLE: {tle_line0}")
        return status

    async def monitor_slew(self, interval: float = 0.1):
        """Monitor mount until slew completes.

        Args:
            interval: How often to check mount state in seconds

        Returns:
            float: Duration of the slew in seconds
        """
        import time

        # Record start time
        start_time = time.time()

        # First wait a bit to let the slew command be processed
        await asyncio.sleep(0.5)

        # Then monitor until slew completes
        while True:
            status = self.device.status()

            # Check if mount is done slewing - trust PWI4's is_slewing flag
            if not status.mount.is_slewing:
                break

            # Log current position
            logger.info(
                f"Mount at RA {status.mount.ra_j2000_hours:.4f}h "
                f"Dec {status.mount.dec_j2000_degs:.4f}° "
                f"(Alt {status.mount.altitude_degs:.1f}° "
                f"Az {status.mount.azimuth_degs:.1f}°)"
            )

            await asyncio.sleep(interval)

        # Calculate and log slew duration
        slew_duration = time.time() - start_time
        logger.info(f"Slew complete in {slew_duration:.1f} seconds")

        return slew_duration


async def test_pwi4_mount(host: str = "127.0.0.1", port: int = 8220):
    """Test basic PWI4 mount operations."""
    mount = PWI4MountEntity(host=host, port=port)
    logger.info(f"Testing PWI4 mount at {host}:{port}")

    try:
        logger.info("Connecting to mount...")
        mount.connect()

        # Print mount info
        status = mount.device.status()
        logger.info("\nMount Information:")
        logger.info(f"Connected: {status.mount.is_connected}")
        logger.info(f"Tracking: {status.mount.is_tracking}")
        logger.info(f"Position: RA {status.mount.ra_j2000_hours:.4f}h Dec {status.mount.dec_j2000_degs:.4f}°")
        logger.info(f"Alt/Az: {status.mount.altitude_degs:.1f}°/{status.mount.azimuth_degs:.1f}°")

        # Test tracking
        logger.info("\nTesting tracking control...")
        mount.tracking_on()
        await asyncio.sleep(2)
        mount.tracking_off()

        # Test slew
        logger.info("\nTesting slew to coordinates...")
        mount.goto_alt_az(45, 180)  # Slew to 45° altitude, 180° azimuth
        await mount.monitor_slew()

        # Test park
        logger.info("\nTesting park...")
        mount.park()
        await mount.monitor_slew()

    except Exception as e:
        logger.error(f"Error during mount test: {str(e)}")
    finally:
        logger.info("\nDisconnecting from mount...")
        mount.disconnect()


async def test_dome(host: str = "127.0.0.1", port: int = 11111, slack_channel: Optional[str] = None):
    """Test basic dome operations."""
    dome = DomeEntity(address=host, port=port)
    logger.info(f"Testing dome at {host}:{port}")

    try:
        logger.info("Connecting to dome...")
        try:
            dome.connect()
            logger.info("Connected successfully!")
        except Exception as e:
            error_msg = f"Failed to connect to dome: {str(e)}"
            logger.error(error_msg)
            if slack_channel:
                notifier.notify(
                    message=error_msg,
                    level=NotificationLevel.ERROR,
                    title="Dome Test Error",
                    channel=slack_channel,
                )
            return

        # Wait a moment to ensure connection is stable
        await asyncio.sleep(2)

        logger.info("Checking initial state:")
        try:
            if dome.is_error():
                logger.error("Dome is in error state!")
                return
            state = "open" if dome.is_open() else "closed"
            logger.info(f"Dome is {state}")
        except Exception as e:
            error_msg = f"Failed to check dome state: {str(e)}"
            logger.error(error_msg)
            if slack_channel:
                notifier.notify(
                    message=error_msg,
                    level=NotificationLevel.ERROR,
                    title="Dome Test Error",
                    channel=slack_channel,
                )
            return

        if not dome.is_open():
            logger.info("Opening dome...")
            dome.open()

            # Wait for dome to finish opening
            while not dome.is_open() and not dome.is_error():
                if dome.is_moving():
                    logger.info("Dome is moving...")
                await asyncio.sleep(1)

            if dome.is_error():
                logger.error("Dome entered error state while opening!")
                return
            logger.info("Dome opened successfully!")

            # Send notification if slack channel is configured
            if slack_channel:
                notifier.notify(
                    message="Dome opened successfully during test",
                    level=NotificationLevel.SUCCESS,
                    title="Dome Test",
                    channel=slack_channel,
                )

        # Wait a bit before closing
        await asyncio.sleep(5)

        logger.info("Closing dome...")
        dome.close()

        # Wait for dome to finish closing
        while not dome.is_closed() and not dome.is_error():
            if dome.is_moving():
                logger.info("Dome is moving...")
            await asyncio.sleep(1)

        if dome.is_error():
            logger.error("Dome entered error state while closing!")
            if slack_channel:
                notifier.notify(
                    message="Dome entered error state while closing during test",
                    level=NotificationLevel.ERROR,
                    title="Dome Test Error",
                    channel=slack_channel,
                )
            return

        logger.info("Dome closed successfully!")
        if slack_channel:
            notifier.notify(
                message="Dome closed successfully during test",
                level=NotificationLevel.SUCCESS,
                title="Dome Test",
                channel=slack_channel,
            )

    except Exception as e:
        error_msg = f"Error during dome test: {str(e)}"
        logger.error(error_msg)
        if slack_channel:
            notifier.notify(
                message=error_msg,
                level=NotificationLevel.ERROR,
                title="Dome Test Error",
                channel=slack_channel,
            )
    finally:
        logger.info("Disconnecting from dome...")
        dome.disconnect()
        logger.info("Disconnected!")


async def run_dome_monitor(host: str = "127.0.0.1", port: int = 11111, slack_channel: Optional[str] = None):
    """Run the dome monitor as a standalone process.

    Args:
        host: The host address of the dome device
        port: The port number of the dome device
        slack_channel: Optional Slack channel to send notifications to
    """
    dome = DomeEntity(address=host, port=port)

    try:
        print(f"Starting dome monitor for {host}:{port}")
        print("Connecting to dome...")
        dome.connect()
        print("Connected successfully!")

        # Create a stop event that we'll never set (run forever)
        stop_event = asyncio.Event()

        # Run the monitor
        await dome.monitor_dome(slack_channel=slack_channel, interval=10.0, stop_event=stop_event)
    except KeyboardInterrupt:
        print("\nShutting down dome monitor...")
    except Exception as e:
        print(f"ERROR: {str(e)}")
    finally:
        print("Disconnecting from dome...")
        dome.disconnect()
        print("Disconnected!")


async def test_camera(host: str = "127.0.0.1", port: int = 11111):
    """Test basic camera operations."""
    from burr.core.config import get_or_initialize_config

    # Get config and setup logging
    config = get_or_initialize_config()
    setup_loguru(level=config.logging.level, run_output_dir=Path("./temp"))

    camera = CameraEntity(address=host, port=port)
    logger.info(f"Testing camera at {host}:{port}")

    try:
        logger.info("Connecting to camera...")
        try:
            camera.connect()
            logger.info("Connected successfully!")
        except Exception as e:
            error_msg = f"Failed to connect to camera: {str(e)}"
            logger.error(error_msg)
            return

        # Print camera info
        logger.info("\nCamera Information:")
        try:
            logger.info(f"Sensor: {camera.sensor_name} ({camera.sensor_type})")
            logger.info(f"Resolution: {camera.camera_x_size}x{camera.camera_y_size}")
            logger.info(f"Max ADU: {camera.max_adu}")
            logger.info(f"Available gains: {camera.gains}")
            logger.info(f"Gain: {camera.gain}")
            logger.info(f"Available readout modes: {camera.readout_modes}")
            logger.info(f"Readout mode: {camera.readout_mode}")
        except Exception as e:
            logger.warning(f"Failed to get some camera information: {str(e)}")

        # Set up binning
        logger.info("\nSetting up binning...")
        try:
            camera.bin_x = 1
            camera.bin_y = 1
            logger.info(f"Binning set to {camera.bin_x}x{camera.bin_y}")
        except Exception as e:
            logger.error(f"Failed to set binning: {str(e)}")
            return

        # Create a DirectDeviceControl instance for FITS saving
        from burr.hardware.direct.device import DirectDeviceControl
        from burr.hardware.device_interface import TrackingMode, Rates

        device_control = DirectDeviceControl(camera=camera)

        # Take a test exposure
        logger.info("\nTaking a 2 second test exposure...")
        try:
            # Take exposure using device control
            success = await device_control.take_exposure(2.0, "test_exposure.fits")
            if not success:
                logger.error("Failed to take exposure")
                return

            # Now save the image using device control
            try:
                # Create temp directory if it doesn't exist
                temp_dir = Path("./temp")
                temp_dir.mkdir(exist_ok=True)
                logger.info(f"\nSaving test files to {temp_dir.absolute()}")

                # Test saving as FITS with different tracking modes
                # Test sidereal tracking
                sidereal_path = temp_dir / "test_sidereal.fits"
                device_control.save_fits(
                    output_path=sidereal_path,
                    tracking_mode=TrackingMode.SIDEREAL,
                    rates=Rates(right_ascension_rate_arcsec_s=0.0, declination_rate_arcsec_s=0.0),
                    extra_headers={"OBJECT": "Test Star", "TRGRA": 123.45, "TRGDEC": -45.67},
                )
                logger.info(f"Saved sidereal tracking FITS to {sidereal_path}")

                # Test custom tracking rates
                rates_path = temp_dir / "test_rates.fits"
                device_control.save_fits(
                    output_path=rates_path,
                    tracking_mode=TrackingMode.RATE,
                    rates=Rates(right_ascension_rate_arcsec_s=15.041, declination_rate_arcsec_s=1.23),
                    extra_headers={
                        "OBJECT": "Test Satellite",
                        "TLE1": "Example TLE Line 1",
                        "TLE2": "Example TLE Line 2",
                    },
                )
                logger.info(f"Saved custom rates FITS to {rates_path}")

            except Exception as e:
                logger.error(f"Failed to save image: {str(e)}", exc_info=True)

        except Exception as e:
            logger.error(f"Error during exposure: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error during camera test: {str(e)}")
    finally:
        logger.info("\nDisconnecting from camera...")
        try:
            camera.disconnect()
            logger.info("Disconnected!")
        except Exception as e:
            logger.error(f"Error disconnecting from camera: {str(e)}")


class WeatherEntity(WeatherConfig):
    """ASCOM Observing Conditions device implementation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._device = None
        self._last_rain_rate = None
        self._last_humidity = None
        self._consecutive_errors = 0

    @property
    def device(self) -> ObservingConditions:
        """Returns an Alpaca ObservingConditions instance configured with this entity's parameters."""
        if self._device is None:
            self._device = ObservingConditions(f"{self.address}:{self.port}", self.device_number)
        return self._device

    def connect(self):
        """Connect to the weather device."""
        if not self.active:
            return

        try:
            self.device.Connected = True
            logger.info("Connected to weather device")
        except Exception as e:
            logger.error(f"Failed to connect to weather device: {e}")
            raise

    def disconnect(self):
        """Disconnect from the weather device."""
        if not self.active or self._device is None:
            return

        try:
            self.device.Connected = False
            self._device = None
            logger.info("Disconnected from weather device")
        except Exception as e:
            logger.error(f"Error disconnecting from weather device: {e}")

    def get_rain_rate(self) -> Optional[float]:
        """Get the current rain rate in mm/hr.

        Returns:
            Rain rate in mm/hr, or None if unavailable or error
        """
        if not self.active:
            return None

        try:
            rain_rate = self.device.RainRate
            self._last_rain_rate = rain_rate
            self._consecutive_errors = 0  # Reset error counter on success
            return rain_rate
        except Exception as e:
            self._consecutive_errors += 1
            logger.warning(f"Failed to get rain rate (error {self._consecutive_errors}): {e}")
            return None

    def get_humidity(self) -> Optional[float]:
        """Get the current humidity in percent.

        Returns:
            Humidity in percent, or None if unavailable or error
        """
        if not self.active:
            return None

        try:
            humidity = self.device.Humidity
            self._last_humidity = humidity
            self._consecutive_errors = 0  # Reset error counter on success
            return humidity
        except Exception as e:
            self._consecutive_errors += 1
            logger.warning(f"Failed to get humidity (error {self._consecutive_errors}): {e}")
            return None

    def is_connected(self) -> bool:
        """Check if the weather device is connected."""
        if not self.active:
            return False

        try:
            return self.device.Connected
        except Exception as e:
            logger.error(f"Failed to check weather device connection: {e}")
            return False

    def has_too_many_errors(self, max_errors: int) -> bool:
        """Check if we have too many consecutive errors.

        Args:
            max_errors: Maximum number of consecutive errors allowed

        Returns:
            True if we have exceeded the error threshold
        """
        return self._consecutive_errors >= max_errors

    def get_status_summary(self) -> dict:
        """Get a summary of current weather conditions.

        Returns:
            Dictionary with current weather status
        """
        return {
            "connected": self.is_connected(),
            "rain_rate": self._last_rain_rate,
            "humidity": self._last_humidity,
            "consecutive_errors": self._consecutive_errors,
            "active": self.active,
        }


def get_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description="Test ASCOM devices")
    parser.add_argument("--config", type=Path, help="Path to config file")
    parser.add_argument("--monitor", action="store_true", help="Run dome monitor")
    parser.add_argument("--camera", action="store_true", help="Test camera instead of dome")
    parser.add_argument("--pwi4", action="store_true", help="Test PWI4 mount")
    parser.add_argument("--host", help="Override host from config")
    parser.add_argument("--port", type=int, help="Override port from config")
    parser.add_argument("--channel", help="Override Slack channel from config")
    return parser


def run_dome_test(host: str, port: int, channel: Optional[str] = None):
    """Run the dome test with the given parameters."""
    logger.info(f"Testing dome at {host}:{port}")
    asyncio.run(test_dome(host, port, channel))


def run_camera_test(host: str, port: int):
    """Run the camera test with the given parameters."""
    logger.info(f"Testing camera at {host}:{port}")
    asyncio.run(test_camera(host, port))


def run_dome_monitor_test(host: str, port: int, channel: Optional[str] = None):
    """Run the dome monitor with the given parameters."""
    logger.info(f"Starting dome monitor at {host}:{port}")
    asyncio.run(run_dome_monitor(host, port, channel))


def main(args=None):
    """Main entry point for device testing."""
    parser = get_parser()
    args = parser.parse_args(args)

    # Load configuration
    config = get_or_initialize_config(args.config)

    # Get connection details from config or command line
    if args.pwi4:
        host = args.host or config.hardware.pwi4.host if config.hardware.pwi4 else "127.0.0.1"
        port = args.port or config.hardware.pwi4.port if config.hardware.pwi4 else 8220
        asyncio.run(test_pwi4_mount(host, port))
    elif args.monitor:
        host = args.host or config.hardware.dome.address if config.hardware.dome else "127.0.0.1"
        port = args.port or config.hardware.dome.port if config.hardware.dome else 11111
        channel = args.channel or config.notifications.slack_channel
        run_dome_monitor_test(host, port, channel)
    elif args.camera:
        host = args.host or config.hardware.camera.address if config.hardware.camera else "127.0.0.1"
        port = args.port or config.hardware.camera.port if config.hardware.camera else 11111
        run_camera_test(host, port)
    else:
        host = args.host or config.hardware.dome.address if config.hardware.dome else "127.0.0.1"
        port = args.port or config.hardware.dome.port if config.hardware.dome else 11111
        channel = args.channel or config.notifications.slack_channel
        run_dome_test(host, port, channel)


if __name__ == "__main__":
    sys.exit(main())
