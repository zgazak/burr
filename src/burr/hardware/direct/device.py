from typing import Optional, Dict, Any
import asyncio
from loguru import logger
from pathlib import Path
import numpy as np
from astropy.io import fits
from requests.exceptions import ReadTimeout
from datetime import datetime

from burr.models.ascom import PWI4MountEntity, CameraEntity, DomeEntity, WeatherEntity, CameraStates, SlewMetadata
from burr.hardware.device_interface import DeviceInterface, TrackingMode, Rates
from burr.core.config import AppConfig


class DirectDeviceControl(DeviceInterface):
    """Direct device control implementation using ASCOM."""

    def __init__(
        self,
        camera: Optional[CameraEntity] = None,
        dome: Optional[DomeEntity] = None,
        mount: Optional[PWI4MountEntity] = None,
        weather: Optional[WeatherEntity] = None,
        config: Optional[AppConfig] = None,
    ):
        """Initialize device control.

        Args:
            camera: Optional camera device. If None, a simulated camera will be used.
            dome: Optional dome device. If None, a simulated dome will be used.
            mount: Optional mount device. If None, a simulated mount will be used.
            weather: Optional weather device. If None, a simulated weather device will be used.
            config: Optional application configuration for metadata
        """
        # Create simulated devices if none provided
        self.camera = camera or CameraEntity(address="127.0.0.1", port=11111)
        self.dome = dome or DomeEntity(address="127.0.0.1", port=11111)
        self.mount = mount or PWI4MountEntity(host="127.0.0.1", port=8220)
        self.weather = weather or WeatherEntity(address="127.0.0.1", port=5580)
        self.config = config

        # Track which devices are real vs simulated
        self.camera.active = camera.active if camera else False
        self.dome.active = dome.active if dome else False
        self.mount.active = mount.active if mount else False
        self.weather.active = weather.active if weather else False

        if not any([camera, dome, mount, weather]):
            logger.info("All devices are simulated")
        else:
            active = []
            if camera:
                active.append("camera")
            if dome:
                active.append("dome")
            if mount:
                active.append("mount")
            if weather:
                active.append("weather")
            logger.info(f"Active devices: {', '.join(active)}")

        self._simulated_tracking = False
        self._simulated_dome_open = False

    async def connect(self):
        """Connect to all devices."""
        try:
            # Connect to mount
            if self.mount.active:
                logger.info("Connecting to mount")
                self.mount.connect()
                logger.info("Connected to mount")

                # Wait for mount to initialize
                logger.info("Waiting for mount initialization...")
                await asyncio.sleep(2)

                # Check mount status
                status = self.mount.device.status()
                if not status.mount.is_connected:
                    raise RuntimeError("Mount not connected after initialization")

                # Enable motors with retries
                max_retries = 3
                if not status.mount.axis[0].is_enabled or not status.mount.axis[1].is_enabled:
                    for attempt in range(max_retries):
                        try:
                            # Enable altitude motor
                            status = self.mount.device.mount_enable(0)
                            await asyncio.sleep(1)
                            if not status.mount.axis[0].is_enabled:
                                if attempt == max_retries - 1:
                                    raise RuntimeError("Failed to enable altitude motor")
                                logger.warning(f"Altitude motor enable failed, attempt {attempt + 1}/{max_retries}")
                                await asyncio.sleep(2)
                                continue

                            # Enable azimuth motor
                            status = self.mount.device.mount_enable(1)
                            await asyncio.sleep(1)
                            if not status.mount.axis[1].is_enabled:
                                if attempt == max_retries - 1:
                                    raise RuntimeError("Failed to enable azimuth motor")
                                logger.warning(f"Azimuth motor enable failed, attempt {attempt + 1}/{max_retries}")
                                await asyncio.sleep(2)
                                continue

                            logger.info("Enabled altitude and azimuth motors")
                            break
                        except Exception as e:
                            if attempt == max_retries - 1:
                                raise RuntimeError(f"Error enabling motors: {e}")
                            logger.warning(f"Error enabling motors, attempt {attempt + 1}/{max_retries}: {e}")
                            await asyncio.sleep(2)
                else:
                    logger.info("Mount motors already enabled")

                # Find home position
                logger.info("Finding home position...")
                self.mount.device.mount_find_home()
                await self.mount.monitor_slew()
                logger.info("Mount homed successfully")

            else:
                logger.info("Mount in simulation mode")

            # Connect to camera
            if self.camera.active:
                logger.info("Connecting to camera")
                self.camera.connect()
                logger.info("Connected to camera")

                # Set camera binning
                if self.camera.default_binning > 0:
                    logger.info(
                        f"Setting camera binning to {self.camera.default_binning}x{self.camera.default_binning}"
                    )
                    self.camera.bin_x = self.camera.default_binning
                    self.camera.bin_y = self.camera.default_binning

            else:
                logger.info("Camera in simulation mode")

            # Connect to dome
            if self.dome.active:
                logger.info("Connecting to dome")
                self.dome.connect()
                logger.info("Connected to dome")
            else:
                logger.info("Dome in simulation mode")

            # Connect to weather device
            if self.weather.active:
                logger.info("Connecting to weather device")
                self.weather.connect()
                logger.info("Connected to weather device")
            else:
                logger.info("Weather device in simulation mode")

        except Exception as e:
            logger.error(f"Error connecting to devices: {e}")
            # Clean up any partially connected devices
            try:
                if self.mount.active and self.mount.device.Connected:
                    self.mount.disconnect()
            except:
                pass
            try:
                if self.camera.active and self.camera.device.Connected:
                    self.camera.disconnect()
            except:
                pass
            try:
                if self.dome.active and self.dome.device.Connected:
                    self.dome.disconnect()
            except:
                pass
            try:
                if self.weather.active and self.weather.device.Connected:
                    self.weather.disconnect()
            except:
                pass
            raise

    async def disconnect(self):
        """Disconnect from all devices."""
        try:
            # Disconnect mount if active
            if self.mount.active:
                # First park the mount (while motors are still enabled)
                self.mount.park()
                logger.info("Parking mount")
                await self.mount.monitor_slew()
                logger.info("Mount parked successfully")

                # Now disable tracking and motors
                self.mount.device.mount_tracking_off()
                logger.info("Disabled mount tracking")
                self.mount.device.mount_disable(0)  # Disable altitude motor
                self.mount.device.mount_disable(1)  # Disable azimuth motor
                logger.info("Disabled altitude and azimuth motors")

                # Finally disconnect
                self.mount.disconnect()
                logger.info("Disconnected from mount")

            # Disconnect camera if active
            if self.camera.active:
                self.camera.disconnect()
                logger.info("Disconnected from camera")

            # Disconnect dome if active
            if self.dome.active:
                self.dome.disconnect()
                logger.info("Disconnected from dome")

            # Disconnect weather device if active
            if self.weather.active:
                self.weather.disconnect()
                logger.info("Disconnected from weather device")

        except Exception as e:
            logger.error(f"Error disconnecting from devices: {e}")
            raise

    async def disconnect_without_parking(self):
        """Disconnect from all devices without parking (assumes mount is already parked)."""
        try:
            # Disconnect mount if active - no parking since it should already be done
            if self.mount.active:
                # Just disable tracking and motors, then disconnect
                self.mount.device.mount_tracking_off()
                logger.info("Disabled mount tracking")
                self.mount.device.mount_disable(0)  # Disable altitude motor
                self.mount.device.mount_disable(1)  # Disable azimuth motor
                logger.info("Disabled altitude and azimuth motors")

                # Disconnect without parking
                self.mount.disconnect()
                logger.info("Disconnected from mount")

            # Disconnect camera if active
            if self.camera.active:
                self.camera.disconnect()
                logger.info("Disconnected from camera")

            # Disconnect dome if active
            if self.dome.active:
                self.dome.disconnect()
                logger.info("Disconnected from dome")

            # Disconnect weather device if active
            if self.weather.active:
                self.weather.disconnect()
                logger.info("Disconnected from weather device")

        except Exception as e:
            logger.error(f"Error disconnecting from devices: {e}")
            raise

    async def emergency_disconnect(self):
        """Emergency disconnect from all devices with minimal operations."""
        try:
            # Quick disconnect from all devices with error handling
            if self.mount.active:
                try:
                    self.mount.device.mount_tracking_off()
                    logger.info("Emergency: Disabled mount tracking")
                except Exception as e:
                    logger.warning(f"Emergency: Could not disable tracking: {e}")

                try:
                    self.mount.device.mount_disable(0)  # Disable altitude motor
                    self.mount.device.mount_disable(1)  # Disable azimuth motor
                    logger.info("Emergency: Disabled mount motors")
                except Exception as e:
                    logger.warning(f"Emergency: Could not disable motors: {e}")

                try:
                    self.mount.disconnect()
                    logger.info("Emergency: Disconnected from mount")
                except Exception as e:
                    logger.warning(f"Emergency: Could not disconnect mount: {e}")

            if self.camera.active:
                try:
                    self.camera.disconnect()
                    logger.info("Emergency: Disconnected from camera")
                except Exception as e:
                    logger.warning(f"Emergency: Could not disconnect camera: {e}")

            if self.dome.active:
                try:
                    self.dome.disconnect()
                    logger.info("Emergency: Disconnected from dome")
                except Exception as e:
                    logger.warning(f"Emergency: Could not disconnect dome: {e}")

            if self.weather.active:
                try:
                    self.weather.disconnect()
                    logger.info("Emergency: Disconnected from weather device")
                except Exception as e:
                    logger.warning(f"Emergency: Could not disconnect weather device: {e}")

        except Exception as e:
            logger.error(f"Error during emergency disconnect: {e}")
            # Don't re-raise in emergency mode, just log and continue

    async def open_dome(self):
        """Open the dome."""
        try:
            if self.dome.active:
                self.dome.open()
                logger.info("Opening dome")

                # Wait for dome to open with timeout handling
                timeout = 30  # 30 second timeout
                start_time = asyncio.get_event_loop().time()

                while not self.dome.is_open():
                    try:
                        if asyncio.get_event_loop().time() - start_time > timeout:
                            raise TimeoutError("Dome failed to open within timeout period")

                        # Check if dome is in error state
                        if self.dome.is_error():
                            raise RuntimeError("Dome entered error state while opening")

                        await asyncio.sleep(1)

                    except ReadTimeout:
                        # Ignore read timeouts and continue waiting
                        logger.warning("Read timeout while checking dome status, continuing to wait...")
                        continue

                logger.info("Dome opened successfully")
            else:
                logger.info("[SIMULATION] Opening dome")
                await asyncio.sleep(3)  # Simulate dome movement time
                self._simulated_dome_open = True
                logger.info("[SIMULATION] Dome opened successfully")

        except ReadTimeout:
            # If we get a timeout during the initial open command, assume it worked
            logger.warning("Read timeout during dome open command - this is expected with this driver")
            if self.dome.active:
                logger.info("Continuing with dome operation...")
        except Exception as e:
            logger.error(f"Error opening dome: {e}")
            raise

    async def close_dome(self):
        """Close the dome."""
        try:
            if self.dome.active:
                self.dome.close()
                logger.info("Closing dome")
                while not self.dome.is_closed():
                    await asyncio.sleep(1)
                logger.info("Dome closed successfully")
            else:
                logger.info("[SIMULATION] Closing dome")
                await asyncio.sleep(3)  # Simulate dome movement time
                self._simulated_dome_open = False
                logger.info("[SIMULATION] Dome closed successfully")

        except Exception as e:
            logger.error(f"Error closing dome: {e}")
            raise

    async def _set_tracking_mode(self, tracking_mode: TrackingMode, rates: Optional[Rates] = None):
        """Set the mount tracking mode.

        Args:
            tracking_mode: The tracking mode to use
            rates: Optional tracking rates for RATE mode
        """
        if isinstance(tracking_mode, str):
            # Convert string to enum if needed
            tracking_mode = TrackingMode(tracking_mode)

        if tracking_mode == TrackingMode.SIDEREAL:
            # Set rates to 0,0 for sidereal tracking
            self.mount.device.mount_offset(ra_set_rate_arcsec_per_sec=0, dec_set_rate_arcsec_per_sec=0)
            self.mount.tracking_on()
            logger.info("Enabled sidereal tracking")
            await asyncio.sleep(1)  # Let mount settle into new rates
        elif tracking_mode == TrackingMode.RATE:
            # Set custom tracking rates
            if hasattr(rates, "right_ascension_rate_arcsec_s") and hasattr(rates, "declination_rate_arcsec_s"):
                self.mount.device.mount_offset(
                    ra_set_rate_arcsec_per_sec=rates.right_ascension_rate_arcsec_s,
                    dec_set_rate_arcsec_per_sec=rates.declination_rate_arcsec_s,
                )
                self.mount.tracking_on()
                logger.info(
                    f"Set custom tracking rates: RA={rates.right_ascension_rate_arcsec_s}, Dec={rates.declination_rate_arcsec_s} arcsec/s"
                )
                await asyncio.sleep(1)  # Let mount settle into new rates
            else:
                # Invalid rates, fall back to sidereal
                logger.error(
                    f"TrackingMode.RATE specified but rates object is invalid: {rates}, falling back to sidereal"
                )
                self.mount.device.mount_offset(ra_set_rate_arcsec_per_sec=0, dec_set_rate_arcsec_per_sec=0)
                self.mount.tracking_on()
                logger.info("Falling back to sidereal tracking")
                await asyncio.sleep(1)  # Let mount settle into new rates
        elif tracking_mode == TrackingMode.OFF:
            # Disable tracking completely
            self.mount.device.mount_offset(ra_set_rate_arcsec_per_sec=0, dec_set_rate_arcsec_per_sec=0)
            self.mount.tracking_off()
            logger.info("Disabled tracking")
            await asyncio.sleep(1)  # Let mount settle into new rates
        else:
            # Unknown tracking mode, fall back to sidereal
            logger.error(f"Unknown tracking mode: {tracking_mode}, falling back to sidereal")
            self.mount.device.mount_offset(ra_set_rate_arcsec_per_sec=0, dec_set_rate_arcsec_per_sec=0)
            self.mount.tracking_on()
            logger.info("Falling back to sidereal tracking")
            await asyncio.sleep(1)  # Let mount settle into new rates

    async def goto_ra_dec(
        self, ra_deg: float, dec_deg: float, tracking_mode: TrackingMode, rates: Optional[Rates] = None
    ) -> SlewMetadata:
        """Go to RA/Dec coordinates.

        Args:
            ra_deg: Right ascension in degrees
            dec_deg: Declination in degrees
            tracking_mode: Tracking mode to use
            rates: Optional tracking rates

        Returns:
            SlewMetadata: Information about the slew operation
        """
        try:
            if self.mount.active:
                # Convert RA from degrees to hours
                ra_hours = ra_deg / 15.0

                # Get current mount position before slew
                status_before = self.mount.device.status()
                start_time = datetime.now()

                # Go to target
                self.mount.goto_ra_dec_j2000(ra_hours, dec_deg)
                logger.info(f"Slewing to RA {ra_hours:.4f}h Dec {dec_deg:.4f}°")

                # Monitor slew and get duration
                slew_duration = await self.mount.monitor_slew()

                # Get final mount position
                status_after = self.mount.device.status()
                end_time = datetime.now()

                # Set tracking mode
                await self._set_tracking_mode(tracking_mode, rates)

                # Calculate distances
                alt_distance_deg = abs(status_after.mount.altitude_degs - status_before.mount.altitude_degs)
                az_distance_deg = abs(status_after.mount.azimuth_degs - status_before.mount.azimuth_degs)

                # Create SlewMetadata object
                slew_metadata = SlewMetadata(
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=slew_duration,
                    start_alt_deg=status_before.mount.altitude_degs,
                    start_az_deg=status_before.mount.azimuth_degs,
                    start_ra_hours=status_before.mount.ra_j2000_hours,
                    start_dec_deg=status_before.mount.dec_j2000_degs,
                    end_alt_deg=status_after.mount.altitude_degs,
                    end_az_deg=status_after.mount.azimuth_degs,
                    end_ra_hours=status_after.mount.ra_j2000_hours,
                    end_dec_deg=status_after.mount.dec_j2000_degs,
                    target_ra_hours=ra_hours,
                    target_dec_deg=dec_deg,
                    slew_type="ra_dec",
                    alt_distance_deg=alt_distance_deg,
                    az_distance_deg=az_distance_deg,
                )
                return slew_metadata
            else:
                logger.info(f"[SIMULATION] Slewing to RA {ra_deg / 15.0:.4f}h Dec {dec_deg:.4f}°")
                start_time = datetime.now()
                await asyncio.sleep(2)  # Simulate slew time
                end_time = datetime.now()
                logger.info("[SIMULATION] Slew complete")
                self._simulated_tracking = tracking_mode == TrackingMode.SIDEREAL

                # Create simulated slew metadata
                slew_metadata = SlewMetadata(
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=2.0,
                    start_alt_deg=45.0,  # Simulated values
                    start_az_deg=180.0,
                    start_ra_hours=ra_deg / 15.0,
                    start_dec_deg=dec_deg,
                    end_alt_deg=45.0,
                    end_az_deg=180.0,
                    end_ra_hours=ra_deg / 15.0,
                    end_dec_deg=dec_deg,
                    target_ra_hours=ra_deg / 15.0,
                    target_dec_deg=dec_deg,
                    slew_type="ra_dec",
                    alt_distance_deg=0.0,  # No movement in simulation
                    az_distance_deg=0.0,
                )
                return slew_metadata

        except Exception as e:
            logger.error(f"Error going to RA/Dec: {e}")
            raise

    async def goto_alt_az(
        self, alt_deg: float, az_deg: float, tracking_mode: TrackingMode, rates: Optional[Rates] = None
    ) -> SlewMetadata:
        """Go to Alt/Az coordinates.

        Args:
            alt_deg: Altitude in degrees
            az_deg: Azimuth in degrees
            tracking_mode: Tracking mode to use
            rates: Optional tracking rates

        Returns:
            SlewMetadata: Information about the slew operation
        """
        try:
            if self.mount.active:
                # Get current mount position before slew
                status_before = self.mount.device.status()
                start_time = datetime.now()

                # Go to target
                self.mount.goto_alt_az(alt_deg, az_deg)
                logger.info(f"Slewing to Alt {alt_deg:.4f}° Az {az_deg:.4f}°")

                # Monitor slew and get duration
                slew_duration = await self.mount.monitor_slew()

                # Get final mount position
                status_after = self.mount.device.status()
                end_time = datetime.now()

                # Set tracking mode
                await self._set_tracking_mode(tracking_mode, rates)

                # Calculate distances
                alt_distance_deg = abs(status_after.mount.altitude_degs - status_before.mount.altitude_degs)
                az_distance_deg = abs(status_after.mount.azimuth_degs - status_before.mount.azimuth_degs)

                # Create SlewMetadata object
                slew_metadata = SlewMetadata(
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=slew_duration,
                    start_alt_deg=status_before.mount.altitude_degs,
                    start_az_deg=status_before.mount.azimuth_degs,
                    start_ra_hours=status_before.mount.ra_j2000_hours,
                    start_dec_deg=status_before.mount.dec_j2000_degs,
                    end_alt_deg=status_after.mount.altitude_degs,
                    end_az_deg=status_after.mount.azimuth_degs,
                    end_ra_hours=status_after.mount.ra_j2000_hours,
                    end_dec_deg=status_after.mount.dec_j2000_degs,
                    target_alt_deg=alt_deg,
                    target_az_deg=az_deg,
                    slew_type="alt_az",
                    alt_distance_deg=alt_distance_deg,
                    az_distance_deg=az_distance_deg,
                )
                return slew_metadata
            else:
                logger.info(f"[SIMULATION] Slewing to Alt {alt_deg:.4f}° Az {az_deg:.4f}°")
                start_time = datetime.now()
                await asyncio.sleep(2)  # Simulate slew time
                end_time = datetime.now()
                logger.info("[SIMULATION] Slew complete")
                self._simulated_tracking = tracking_mode == TrackingMode.SIDEREAL

                # Create simulated slew metadata
                slew_metadata = SlewMetadata(
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=2.0,
                    start_alt_deg=45.0,  # Simulated values
                    start_az_deg=180.0,
                    start_ra_hours=0.0,
                    start_dec_deg=0.0,
                    end_alt_deg=alt_deg,
                    end_az_deg=az_deg,
                    end_ra_hours=0.0,
                    end_dec_deg=0.0,
                    target_alt_deg=alt_deg,
                    target_az_deg=az_deg,
                    slew_type="alt_az",
                    alt_distance_deg=abs(alt_deg - 45.0),  # Calculate distance from start to target
                    az_distance_deg=abs(az_deg - 180.0),
                )
                return slew_metadata

        except Exception as e:
            logger.error(f"Error going to Alt/Az: {e}")
            raise

    async def track_tle(
        self,
        tle_line1: str,
        tle_line2: str,
        tracking_mode: TrackingMode,
        rates: Optional[Rates] = None,
        previous_tracking_mode: Optional[TrackingMode] = None,
    ) -> SlewMetadata:
        """Track a satellite using TLE elements.

        Args:
            tle_line1: First line of TLE
            tle_line2: Second line of TLE
            tracking_mode: Desired tracking mode
            rates: Optional tracking rates (ignored for TLE tasks as PWI4 handles tracking)
            previous_tracking_mode: The tracking mode from the previous frame, if any.
                                  None indicates this is the first frame or a new target.

        Returns:
            SlewMetadata: Information about the slew operation
        """
        try:
            if self.mount.active:
                # Basic TLE validation
                if not tle_line1 or not tle_line2:
                    raise ValueError("TLE lines cannot be empty")

                # Determine if we need to call mount_follow_tle
                need_follow_tle = False

                # Case 1: No previous tracking state (first call or new target)
                if previous_tracking_mode is None:
                    need_follow_tle = True
                    logger.debug("First TLE tracking call or new target, initializing tracking")

                # Case 2: Going from non-RATE mode to RATE mode
                elif tracking_mode == TrackingMode.RATE and previous_tracking_mode != TrackingMode.RATE:
                    need_follow_tle = True
                    logger.debug("Switching to RATE mode, reinitializing tracking")

                # Execute mount_follow_tle if needed
                if need_follow_tle:
                    # Get current mount position before slew
                    status_before = self.mount.device.status()
                    start_time = datetime.now()

                    tle_line0 = "0 CALSAT"
                    status = self.mount.device.mount_follow_tle(tle_line0, tle_line1, tle_line2)
                    logger.info("Started TLE tracking")

                    # Wait for mount to get to position and get duration
                    slew_duration = await self.mount.monitor_slew()
                    logger.info("Mount in position")

                    # Get final mount position
                    status_after = self.mount.device.status()
                    end_time = datetime.now()

                    # Calculate distances
                    alt_distance_deg = abs(status_after.mount.altitude_degs - status_before.mount.altitude_degs)
                    az_distance_deg = abs(status_after.mount.azimuth_degs - status_before.mount.azimuth_degs)

                    # Create SlewMetadata object for the TLE slew
                    slew_metadata = SlewMetadata(
                        start_time=start_time,
                        end_time=end_time,
                        duration_seconds=slew_duration,
                        start_alt_deg=status_before.mount.altitude_degs,
                        start_az_deg=status_before.mount.azimuth_degs,
                        start_ra_hours=status_before.mount.ra_j2000_hours,
                        start_dec_deg=status_before.mount.dec_j2000_degs,
                        end_alt_deg=status_after.mount.altitude_degs,
                        end_az_deg=status_after.mount.azimuth_degs,
                        end_ra_hours=status_after.mount.ra_j2000_hours,
                        end_dec_deg=status_after.mount.dec_j2000_degs,
                        slew_type="tle",
                        alt_distance_deg=alt_distance_deg,
                        az_distance_deg=az_distance_deg,
                    )
                else:
                    # No slew needed, create metadata for tracking mode change only
                    status_current = self.mount.device.status()
                    current_time = datetime.now()
                    slew_metadata = SlewMetadata(
                        start_time=current_time,
                        end_time=current_time,
                        duration_seconds=0.0,
                        start_alt_deg=status_current.mount.altitude_degs,
                        start_az_deg=status_current.mount.azimuth_degs,
                        start_ra_hours=status_current.mount.ra_j2000_hours,
                        start_dec_deg=status_current.mount.dec_j2000_degs,
                        end_alt_deg=status_current.mount.altitude_degs,
                        end_az_deg=status_current.mount.azimuth_degs,
                        end_ra_hours=status_current.mount.ra_j2000_hours,
                        end_dec_deg=status_current.mount.dec_j2000_degs,
                        slew_type="tle",
                        alt_distance_deg=0.0,  # No movement
                        az_distance_deg=0.0,
                    )

                # Set final tracking state if not RATE mode
                if tracking_mode.value != TrackingMode.RATE.value:  # Compare the actual enum values
                    if tracking_mode.value == TrackingMode.SIDEREAL.value:
                        # Break TLE tracking and switch to sidereal
                        # First stop tracking to break the TLE tracking state
                        self.mount.device.mount_tracking_off()
                        # Then enable sidereal tracking with 0,0 rates
                        self.mount.device.mount_offset(ra_set_rate_arcsec_per_sec=0.0, dec_set_rate_arcsec_per_sec=0.0)
                        self.mount.device.mount_tracking_on()
                        logger.info("Set to sidereal tracking")
                    else:  # OFF
                        self.mount.device.mount_tracking_off()
                        logger.info("Tracking disabled")

                return slew_metadata

            else:
                # Simulation mode
                logger.info(f"[SIMULATION] Moving to satellite position with tracking mode {tracking_mode}")
                start_time = datetime.now()
                await asyncio.sleep(5.0)  # Simulate slew time
                end_time = datetime.now()
                self._simulated_tracking = tracking_mode in [TrackingMode.RATE, TrackingMode.SIDEREAL]

                # Create simulated slew metadata
                slew_metadata = SlewMetadata(
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=5.0,
                    start_alt_deg=45.0,  # Simulated values
                    start_az_deg=180.0,
                    start_ra_hours=0.0,
                    start_dec_deg=0.0,
                    end_alt_deg=45.0,
                    end_az_deg=180.0,
                    end_ra_hours=0.0,
                    end_dec_deg=0.0,
                    slew_type="tle",
                    alt_distance_deg=0.0,  # No movement in simulation
                    az_distance_deg=0.0,
                )
                return slew_metadata

        except Exception as e:
            logger.error(f"Error tracking TLE: {e}")
            raise

    async def take_exposure(self, exposure_seconds: float, file_path: str) -> bool:
        """Take an exposure.

        Args:
            exposure_seconds: Exposure time in seconds
            file_path: Path to save the image

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.camera.active:
                # Set timing properties as close to exposure start as possible
                from datetime import datetime
                from zoneinfo import ZoneInfo

                start_time = datetime.now(ZoneInfo("UTC"))
                self.camera.last_exposure_start_time = start_time.isoformat()
                self.camera.last_exposure_duration = exposure_seconds

                # Start exposure
                logger.info(f"Starting {exposure_seconds}s exposure")
                self.camera.start_exposure(exposure_seconds, True)  # True = Light frame

                # Wait for exposure to complete and image to be ready
                while not self.camera.device.ImageReady:
                    state = self.camera.camera_state
                    if state == CameraStates.cameraError:
                        logger.error("Camera reported error during exposure")
                        return False
                    elif state == CameraStates.cameraIdle:
                        # Sometimes the camera goes idle before ImageReady is True
                        await asyncio.sleep(0.1)
                        if self.camera.device.ImageReady:
                            break
                        logger.error("Camera returned to idle without image ready")
                        return False
                    elif state == CameraStates.cameraExposing:
                        logger.debug(f"Exposure {self.camera.percent_completed}% complete")
                    elif state == CameraStates.cameraWaiting:
                        logger.info("Camera is waiting (possibly downloading image)")
                    await asyncio.sleep(0.1)

                # Get the image data
                logger.info("Exposure complete, retrieving image data")
                self.camera.image_ready = True
                self.camera.image_array = self.camera.device.ImageArray

                # Save the image data to file
                logger.info(f"Saving image data to {file_path}")
                file_path = Path(file_path)
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # Create a new FITS file with the image data
                hdu = fits.PrimaryHDU(self.camera.image_array)
                hdu.header["EXPTIME"] = exposure_seconds
                hdu.header["DATE-OBS"] = self.camera.last_exposure_start_time
                hdu.writeto(file_path, overwrite=True)

                logger.info("Image data retrieved and saved successfully")
                return True
            else:
                # Set timing properties for simulation mode too
                from datetime import datetime
                from zoneinfo import ZoneInfo

                start_time = datetime.now(ZoneInfo("UTC"))
                self.camera.last_exposure_start_time = start_time.isoformat()
                self.camera.last_exposure_duration = exposure_seconds

                logger.info(f"[SIMULATION] Taking {exposure_seconds}s exposure")
                await asyncio.sleep(exposure_seconds)
                logger.info("[SIMULATION] Image ready")
                # In simulation mode, create a fake image array
                self.camera.image_ready = True
                self.camera.image_array = np.random.normal(32000, 1000, (2048, 2048))  # Simulate a 2k x 2k image

                # Save the simulated image
                logger.info(f"[SIMULATION] Saving simulated image to {file_path}")
                file_path = Path(file_path)
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # Create a new FITS file with the simulated image data
                hdu = fits.PrimaryHDU(self.camera.image_array)
                hdu.header["EXPTIME"] = exposure_seconds
                hdu.header["DATE-OBS"] = self.camera.last_exposure_start_time
                hdu.header["SIMULATED"] = True
                hdu.writeto(file_path, overwrite=True)

                return True

        except Exception as e:
            logger.error(f"Error taking exposure: {e}")
            logger.exception(e)  # Log the full traceback
            return False

    async def park_mount(self):
        """Park the mount."""
        try:
            if self.mount.active:
                self.mount.park()
                logger.info("Parking mount")

                # Wait for park to complete
                await self.mount.monitor_slew()
                logger.info("Mount parked successfully")
            else:
                # Simulate parking
                logger.info("[SIMULATION] Parking mount")
                await asyncio.sleep(5.0)  # Simulate 5 second park time
                self._simulated_tracking = False
                logger.info("[SIMULATION] Mount parked successfully")

        except Exception as e:
            logger.error(f"Error parking mount: {e}")
            raise

    async def home_mount(self):
        """Home the mount."""
        try:
            if self.mount.active:
                self.mount.device.mount_find_home()
                logger.info("Finding mount home position")

                # Wait for home to complete
                await self.mount.monitor_slew()
                logger.info("Mount homed successfully")
            else:
                # Simulate homing
                logger.info("[SIMULATION] Finding mount home position")
                await asyncio.sleep(5.0)  # Simulate 5 second home time
                logger.info("[SIMULATION] Mount homed successfully")

        except Exception as e:
            logger.error(f"Error homing mount: {e}")
            raise

    def save_fits(
        self,
        output_path: Path,
        tracking_mode: Optional[TrackingMode] = None,
        rates: Optional[Rates] = None,
        extra_headers: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save the current camera image as a FITS file with full metadata.

        Args:
            output_path: Where to save the FITS file
            tracking_mode: The tracking mode used for this exposure
            rates: The tracking rates used for this exposure
            extra_headers: Optional additional header keywords to include

        Returns:
            Path to the saved FITS file

        Raises:
            RuntimeError: If no image is available or required metadata is missing
        """
        if not self.camera.image_ready:
            raise RuntimeError("No image available to save")

        # Get the image data as numpy array
        image_data = np.array(self.camera.image_array)

        # Create primary HDU
        primary_hdu = fits.PrimaryHDU(image_data)
        hdul = fits.HDUList([primary_hdu])

        # Add camera metadata
        primary_hdu.header["EXPTIME"] = (self.camera.last_exposure_duration, "Exposure time in seconds")
        primary_hdu.header["DATE-OBS"] = (self.camera.last_exposure_start_time, "Start time of observation")
        primary_hdu.header["IMAGETYP"] = ("Light Frame", "Type of image")  # TODO: Handle dark frames
        primary_hdu.header["INSTRUME"] = (self.camera.sensor_name, "Instrument/sensor name")
        primary_hdu.header["XBINNING"] = (self.camera.bin_x, "Binning factor in X")
        primary_hdu.header["YBINNING"] = (self.camera.bin_y, "Binning factor in Y")

        try:
            primary_hdu.header["GAIN"] = (self.camera.gain, "Camera gain setting")
        except:
            logger.debug("Could not get camera gain")

        try:
            primary_hdu.header["READOUTM"] = (self.camera.readout_mode, "Camera readout mode")
        except:
            logger.debug("Could not get readout mode")

        # Add mount metadata if available
        if self.mount:
            try:
                status = self.mount.device.status()
                primary_hdu.header["RA"] = (status.mount.ra_j2000_hours, "[hours] Telescope right ascension (J2000)")
                primary_hdu.header["DEC"] = (status.mount.dec_j2000_degs, "[deg] Telescope declination (J2000)")
                primary_hdu.header["ALT"] = (status.mount.altitude_degs, "[deg] Telescope altitude")
                primary_hdu.header["AZ"] = (status.mount.azimuth_degs, "[deg] Telescope azimuth")
                primary_hdu.header["TRACKING"] = (status.mount.is_tracking, "Mount tracking enabled")
            except Exception as e:
                logger.warning(f"Could not get mount metadata: {e}")

        # Add tracking mode if provided
        if tracking_mode is not None:
            primary_hdu.header["TRKMODE"] = (tracking_mode.value, "Tracking mode")

        # Add tracking rates if provided
        if rates is not None:
            primary_hdu.header["RARATE"] = (rates.right_ascension_rate_arcsec_s, "[arcsec/s] RA tracking rate")
            primary_hdu.header["DECRATE"] = (rates.declination_rate_arcsec_s, "[arcsec/s] Dec tracking rate")

        # Add site metadata if config is available
        if self.config and self.config.site:
            primary_hdu.header["SITENAME"] = (self.config.site.name, "Observatory site name")
            primary_hdu.header["SITELAT"] = (self.config.site.latitude, "[deg] Observatory latitude")
            primary_hdu.header["SITELONG"] = (self.config.site.longitude, "[deg] Observatory longitude")
            primary_hdu.header["SITEELEV"] = (self.config.site.altitude_km * 1000, "[m] Observatory elevation")

        # Add any extra headers
        if extra_headers:
            for key, value in extra_headers.items():
                if isinstance(value, tuple):
                    primary_hdu.header[key] = value  # Assume (value, comment) tuple
                else:
                    primary_hdu.header[key] = (value, f"Added header {key}")

        # Save the file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        hdul.writeto(output_path, overwrite=True)
        logger.info(f"Saved FITS file to {output_path}")

        return output_path

    def check_weather_conditions(self) -> dict:
        """Check current weather conditions.

        Returns:
            Dictionary with weather status and any issues found
        """
        if not self.weather.active:
            return {
                "status": "simulated",
                "issues": [],
                "should_close_dome": False,
                "rain_rate": 0.0,
                "humidity": 30.0,
                "consecutive_errors": 0,
            }

        issues = []
        should_close_dome = False

        # Check rain rate
        rain_rate = self.weather.get_rain_rate()
        if rain_rate is not None:
            if rain_rate > 0.0:
                issues.append(f"Rain detected: {rain_rate:.2f} mm/hr")
                should_close_dome = True
        else:
            issues.append("Could not read rain rate")

        # Check humidity
        humidity = self.weather.get_humidity()
        if humidity is not None:
            if self.config and humidity > self.config.weather.humidity_threshold:
                issues.append(f"High humidity: {humidity:.1f}% (threshold: {self.config.weather.humidity_threshold}%)")
                should_close_dome = True
        else:
            issues.append("Could not read humidity")

        # Check for too many consecutive errors
        if self.config and self.weather.has_too_many_errors(self.config.weather.max_consecutive_errors):
            issues.append(f"Too many consecutive errors: {self.weather._consecutive_errors}")
            should_close_dome = True

        return {
            "status": "active",
            "issues": issues,
            "should_close_dome": should_close_dome,
            "rain_rate": rain_rate,
            "humidity": humidity,
            "consecutive_errors": self.weather._consecutive_errors,
        }
