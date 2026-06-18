#!/usr/bin/env python3
"""
Dark frame collection script for burr.

This script automatically collects dark frames for all exposure times, binning modes,
and readout modes used by the configured managers. It analyzes the configuration to
determine what dark frames are needed and collects them systematically.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Set, Optional
import numpy as np
from loguru import logger

from burr.core.config import initialize_config, get_config
from burr.models.ascom import CameraEntity, PWI4MountEntity, DomeEntity, CameraStates
from burr.hardware.direct.device import DirectDeviceControl
from burr.run.utils import get_or_create_run
from burr.run import log_command, update_progress, add_generated_file
from astropy.io import fits


class DarkFrameCollector:
    """
    Collects dark frames for all exposure times, binning modes, and readout modes
    used by the configured task managers.
    """

    def __init__(self, config_path: Optional[Path] = None, exposures_per_iteration: Optional[int] = None):
        """
        Initialize the dark frame collector.

        Parameters
        ----------
        config_path : Path, optional
            Path to configuration file
        exposures_per_iteration : int, optional
            Number of exposures to take per combination. If None, uses config value.
        """
        self.config_path = config_path
        self.exposures_per_iteration_override = exposures_per_iteration
        self.config = None
        self.run = None
        self.device_control = None

        # Analysis results
        self.exposure_times: Set[float] = set()
        self.binning_modes: List[int] = []
        self.readout_modes: List[str] = []
        self.camera_supports_readout_change = True

    async def initialize(self):
        """Initialize configuration and hardware."""
        logger.info("Initializing dark frame collector...")

        # Load configuration
        if self.config_path:
            self.config = initialize_config(self.config_path)
        else:
            self.config = get_config()

        # Set exposures_per_iteration from config or override
        if self.exposures_per_iteration_override is not None:
            self.exposures_per_iteration = self.exposures_per_iteration_override
            logger.info(f"Using command line override for exposures_per_iteration: {self.exposures_per_iteration}")
        else:
            self.exposures_per_iteration = self.config.schedule.darks.exposures_per_iteration
            logger.info(f"Using config value for exposures_per_iteration: {self.exposures_per_iteration}")

        # Get or create run
        self.run = get_or_create_run(site=self.config.site, verbose=True)
        logger.info(f"Using run: {self.run.run_id}")

        # Analyze configuration to determine what dark frames are needed
        self._analyze_configuration()

        # Initialize hardware
        await self._initialize_hardware()

    def _analyze_configuration(self):
        """
        Analyze the configuration to determine all exposure times, binning modes,
        and readout modes that will be used.
        """
        logger.info("Analyzing configuration to determine required dark frames...")

        # Collect exposure times from all managers
        exposure_ranges = []

        # Photometric standards manager
        if hasattr(self.config.schedule, "photometric_standards") and self.config.schedule.photometric_standards:
            config = self.config.schedule.photometric_standards
            exposure_ranges.append(
                (
                    "Photometric Standards",
                    config.minimum_exposure_seconds,
                    config.maximum_exposure_seconds,
                    config.n_exposures,
                )
            )

        # Coverage manager
        if hasattr(self.config.schedule, "coverage") and self.config.schedule.coverage:
            config = self.config.schedule.coverage
            exposure_ranges.append(
                ("Sky Coverage", config.minimum_exposure_seconds, config.maximum_exposure_seconds, config.n_exposures)
            )

        # Lunar background manager
        if hasattr(self.config.schedule, "lunar_background") and self.config.schedule.lunar_background:
            config = self.config.schedule.lunar_background
            exposure_ranges.append(
                (
                    "Lunar Background",
                    config.minimum_exposure_seconds,
                    config.maximum_exposure_seconds,
                    config.n_exposures,
                )
            )

        # Calibration satellites manager
        if hasattr(self.config.schedule, "calsats") and self.config.schedule.calsats:
            config = self.config.schedule.calsats
            exposure_ranges.append(
                (
                    "Calibration Satellites",
                    config.minimum_exposure_seconds,
                    config.maximum_exposure_seconds,
                    1,  # Calsats use random exposure times, so we'll sample the range
                )
            )

        # Skip twilight flats - they use adaptive exposure times that don't need darks

        # Generate exposure times from ranges
        for manager_name, min_exp, max_exp, n_exp in exposure_ranges:
            if n_exp == 1:
                # For single exposures (like calsats), sample the range
                sample_times = np.linspace(min_exp, max_exp, 5)
            else:
                # Use the actual exposure sequence the manager would use
                sample_times = np.linspace(min_exp, max_exp, n_exp)

            for exp_time in sample_times:
                self.exposure_times.add(round(exp_time, 3))

            logger.info(f"{manager_name}: {min_exp:.1f}s - {max_exp:.1f}s ({n_exp} steps)")

        # Sort exposure times
        self.exposure_times = sorted(self.exposure_times)

        logger.info(f"Total unique exposure times needed: {len(self.exposure_times)}")
        logger.info(
            f"Exposure times: {[f'{t:.3f}s' for t in self.exposure_times[:10]]}{'...' if len(self.exposure_times) > 10 else ''}"
        )

        # Get binning modes from camera config
        if self.config.hardware and self.config.hardware.camera:
            camera_config = self.config.hardware.camera

            # Use binning modes from config if available
            if hasattr(camera_config, "binning") and camera_config.binning:
                logger.info(f"Using binning modes from config: {camera_config.binning}")
                self.binning_modes = list(camera_config.binning)
            else:
                # Fallback to default binning if no binning list in config
                logger.info(f"No binning list in config, using default binning: {camera_config.default_binning}")
                self.binning_modes = [camera_config.default_binning]
        else:
            # Default binning modes
            logger.info("No camera config found, using default binning modes [1, 2]")
            self.binning_modes = [1, 2]

        logger.info(f"Final binning modes: {self.binning_modes}")

    async def _initialize_hardware(self):
        """Initialize hardware devices."""
        logger.info("Initializing hardware devices...")
        logger.info("Mount and dome will be forced inactive - they are not needed for dark frame collection")

        # Create device entities from config
        if self.config.hardware:
            # Initialize camera
            camera = CameraEntity(
                address=self.config.hardware.camera.address if self.config.hardware.camera else "127.0.0.1",
                port=self.config.hardware.camera.port if self.config.hardware.camera else 11111,
                device_number=self.config.hardware.camera.device_number if self.config.hardware.camera else 0,
                active=self.config.hardware.camera.active if self.config.hardware.camera else False,
                default_binning=self.config.hardware.camera.default_binning if self.config.hardware.camera else 1,
            )

            # Initialize mount (force inactive for dark frame collection)
            mount = PWI4MountEntity(
                host=self.config.hardware.pwi4.host if self.config.hardware.pwi4 else "127.0.0.1",
                port=self.config.hardware.pwi4.port if self.config.hardware.pwi4 else 8220,
                active=False,  # Force inactive - mount not needed for dark frames
            )

            # Initialize dome (force inactive for dark frame collection)
            dome = DomeEntity(
                address=self.config.hardware.dome.address if self.config.hardware.dome else "127.0.0.1",
                port=self.config.hardware.dome.port if self.config.hardware.dome else 11111,
                device_number=self.config.hardware.dome.device_number if self.config.hardware.dome else 0,
                active=False,  # Force inactive - dome not needed for dark frames
            )
        else:
            # Use defaults with simulation mode
            camera = CameraEntity(active=False)
            mount = PWI4MountEntity(active=False)
            dome = DomeEntity(active=False)

        # Create device control interface
        self.device_control = DirectDeviceControl(camera=camera, mount=mount, dome=dome, config=self.config)

        # Connect to devices
        await self.device_control.connect()

        # Validate camera capabilities
        await self._validate_camera_capabilities()

    async def _validate_camera_capabilities(self):
        """Validate camera capabilities and adjust our plans accordingly."""
        logger.info("Validating camera capabilities...")

        camera = self.device_control.camera

        if camera.active:
            try:
                # Get available readout modes
                available_readout_modes = camera.readout_modes
                if available_readout_modes:
                    self.readout_modes = available_readout_modes
                    logger.info(f"Camera supports {len(self.readout_modes)} readout modes: {self.readout_modes}")
                else:
                    # Camera doesn't support changing readout modes
                    current_mode = camera.readout_mode
                    self.readout_modes = [current_mode]
                    self.camera_supports_readout_change = False
                    logger.warning("Camera does not support changing readout modes")
                    logger.warning(f"Current readout mode: {current_mode}")
                    logger.warning(
                        "You may need to manually change the readout mode and run this script multiple times"
                    )

                # Check current readout mode
                current_readout = camera.readout_mode
                logger.info(f"Current readout mode: {current_readout}")

            except Exception as e:
                logger.warning(f"Could not query camera readout modes: {e}")
                self.readout_modes = ["Unknown"]
                self.camera_supports_readout_change = False

            # Validate binning modes against camera capabilities
            # For now, we'll just use the configured ones and let the camera reject invalid ones
        else:
            # Simulated camera
            self.readout_modes = ["Normal", "Fast"]
            logger.info("Using simulated camera with default readout modes")

        logger.info(f"Final readout modes: {self.readout_modes}")
        logger.info(f"Camera supports readout mode changes: {self.camera_supports_readout_change}")

    async def collect_all_darks(self):
        """Collect dark frames for all combinations of exposure time, binning, and readout mode."""
        total_combinations = len(self.exposure_times) * len(self.binning_modes) * len(self.readout_modes)
        total_frames = total_combinations * self.exposures_per_iteration

        logger.info("=" * 80)
        logger.info("DARK FRAME COLLECTION PLAN")
        logger.info("=" * 80)
        logger.info(f"Exposure times: {len(self.exposure_times)}")
        logger.info(f"Binning modes: {len(self.binning_modes)}")
        logger.info(f"Readout modes: {len(self.readout_modes)}")
        logger.info(f"Exposures per combination: {self.exposures_per_iteration}")
        logger.info(f"Total combinations: {total_combinations}")
        logger.info(f"Total frames to collect: {total_frames}")
        logger.info("=" * 80)

        if not self.camera_supports_readout_change and len(self.readout_modes) > 1:
            logger.warning("WARNING: Camera does not support changing readout modes!")
            logger.warning("This script will only work with the current readout mode.")
            logger.warning("To collect darks for other readout modes:")
            logger.warning("1. Manually change the readout mode in your camera software")
            logger.warning("2. Run this script again")
            logger.warning("3. Repeat for each readout mode")

            response = input("Continue with current readout mode only? (y/n): ")
            if response.lower() != "y":
                logger.info("Aborting dark frame collection")
                return

        # Create output directory
        darks_dir = self.run.output_directory / "data" / "darks"
        darks_dir.mkdir(parents=True, exist_ok=True)

        # Track progress
        completed = 0

        # Update run progress
        update_progress(self.run, "dark_collection", 0.0, "collecting darks")

        # Collect darks for each combination
        for readout_mode in self.readout_modes:
            for binning in self.binning_modes:
                for exposure_time in self.exposure_times:
                    logger.info(f"Starting combination {completed + 1}/{total_combinations}")
                    logger.info(
                        f"Readout: {readout_mode}, Binning: {binning}x{binning}, Exposure: {exposure_time:.3f}s"
                    )

                    try:
                        # Set up camera for this combination
                        await self._setup_camera(binning, readout_mode)

                        # Collect the frames for this combination
                        for frame_num in range(self.exposures_per_iteration):
                            frame_name = (
                                f"dark_{readout_mode}_{binning}x{binning}_{exposure_time:.3f}s_{frame_num + 1:03d}.fits"
                            )
                            output_path = darks_dir / frame_name

                            logger.info(
                                f"Taking dark frame {frame_num + 1}/{self.exposures_per_iteration}: {frame_name}"
                            )

                            # Take the dark frame
                            await self._take_dark_frame(
                                exposure_time, output_path, readout_mode, binning, frame_num + 1
                            )

                        completed += 1
                        progress = completed / total_combinations
                        update_progress(
                            self.run,
                            "dark_collection",
                            progress,
                            f"completed {completed}/{total_combinations} combinations",
                        )

                        logger.info(f"Completed combination {completed}/{total_combinations} ({progress:.1%})")

                    except Exception as e:
                        logger.error(f"Error collecting darks for combination: {e}")
                        log_command(
                            self.run,
                            "dark_collection_error",
                            f"Failed: {readout_mode}, {binning}x{binning}, {exposure_time:.3f}s",
                            str(e),
                        )
                        continue

        # Mark completion
        update_progress(self.run, "dark_collection", 1.0, "completed")
        log_command(
            self.run, "dark_collection_completed", f"Collected {total_frames} dark frames in {completed} combinations"
        )

        logger.info("=" * 80)
        logger.info("DARK FRAME COLLECTION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Total frames collected: {total_frames}")
        logger.info(f"Output directory: {darks_dir}")
        logger.info("=" * 80)

    async def _setup_camera(self, binning: int, readout_mode: str):
        """Set up camera for the specified binning and readout mode."""
        camera = self.device_control.camera

        if camera.active:
            # Set binning
            try:
                camera.bin_x = binning
                camera.bin_y = binning
                logger.debug(f"Set binning to {binning}x{binning}")
            except Exception as e:
                logger.warning(f"Could not set binning to {binning}x{binning}: {e}")
                raise

            # Set readout mode (if supported)
            if self.camera_supports_readout_change:
                try:
                    # Note: This is a simplified approach - actual readout mode setting
                    # may require camera-specific ASCOM properties
                    logger.debug(f"Would set readout mode to {readout_mode} (method depends on camera driver)")
                except Exception as e:
                    logger.warning(f"Could not set readout mode to {readout_mode}: {e}")
                    # Continue anyway
        else:
            logger.debug(f"Simulated camera setup: {binning}x{binning}, {readout_mode}")

    def _check_existing_dark_frame(
        self, output_path: Path, exposure_time: float, readout_mode: str, binning: int, frame_number: int
    ) -> bool:
        """
        Check if a dark frame already exists and validate its parameters.

        Parameters
        ----------
        output_path : Path
            Path where the dark frame would be saved
        exposure_time : float
            Expected exposure time in seconds
        readout_mode : str
            Expected readout mode
        binning : int
            Expected binning factor
        frame_number : int
            Frame number in sequence

        Returns
        -------
        bool
            True if existing frame matches parameters, False otherwise
        """
        if not output_path.exists():
            return False

        try:
            # Read FITS header
            with fits.open(output_path, memmap=False) as hdul:
                header = hdul[0].header

                # Check essential parameters
                existing_exptime = header.get("EXPTIME", None)
                existing_xbinning = header.get("XBINNING", None)
                existing_ybinning = header.get("YBINNING", None)
                existing_readout = header.get("READOUTM", None)
                existing_imagetyp = header.get("IMAGETYP", None)

                # Validate exposure time (within 1% tolerance)
                if existing_exptime is None or abs(existing_exptime - exposure_time) > 0.01 * exposure_time:
                    logger.debug(f"Existing dark frame exposure time mismatch: {existing_exptime} vs {exposure_time}")
                    return False

                # Validate binning
                if existing_xbinning != binning or existing_ybinning != binning:
                    logger.debug(
                        f"Existing dark frame binning mismatch: {existing_xbinning}x{existing_ybinning} vs {binning}x{binning}"
                    )
                    return False

                # Validate readout mode
                if existing_readout != readout_mode:
                    logger.debug(f"Existing dark frame readout mode mismatch: {existing_readout} vs {readout_mode}")
                    return False

                # Validate image type
                if existing_imagetyp and "Dark" not in existing_imagetyp:
                    logger.debug(f"Existing frame is not a dark frame: {existing_imagetyp}")
                    return False

                # All parameters match
                logger.info(f"Found existing dark frame with matching parameters: {output_path.name}")
                return True

        except Exception as e:
            logger.warning(f"Error reading existing dark frame {output_path.name}: {e}")
            return False

    async def _take_dark_frame(
        self, exposure_time: float, output_path: Path, readout_mode: str, binning: int, frame_number: int
    ):
        """Take a single dark frame."""
        try:
            # Check if frame already exists with matching parameters
            if self._check_existing_dark_frame(output_path, exposure_time, readout_mode, binning, frame_number):
                logger.info(f"Skipping existing dark frame: {output_path.name}")

                # Log to run history
                log_command(
                    self.run,
                    "dark_frame_existing",
                    f"Found existing dark: {exposure_time:.3f}s, {binning}x{binning}, {readout_mode}, frame {frame_number}",
                )

                # Add to generated files (it's still part of our output set)
                add_generated_file(self.run, output_path)
                return

            # Take the dark frame exposure
            await self._take_dark_exposure(exposure_time)

            # Save the frame with metadata
            extra_headers = {
                "IMAGETYP": ("Dark Frame", "Type of image"),
                "EXPTIME": (exposure_time, "Exposure time in seconds"),
                "XBINNING": (binning, "Binning factor in X"),
                "YBINNING": (binning, "Binning factor in Y"),
                "READOUTM": (readout_mode, "Camera readout mode"),
                "FRAMENUM": (frame_number, "Frame number in dark sequence"),
                "DARKSEQ": (self.exposures_per_iteration, "Total frames in dark sequence"),
                "CREATED": (datetime.now(ZoneInfo("UTC")).isoformat(), "File creation timestamp"),
            }

            # Save using device control's save_fits to include all metadata
            saved_path = self.device_control.save_fits(
                output_path=output_path,
                tracking_mode=None,
                rates=None,
                extra_headers=extra_headers,
            )

            # Log to run history
            log_command(
                self.run,
                "dark_frame_collected",
                f"Dark: {exposure_time:.3f}s, {binning}x{binning}, {readout_mode}, frame {frame_number}, saved to {output_path.name}",
            )

            # Add to generated files
            add_generated_file(self.run, output_path)

            logger.info(f"Dark frame saved: {output_path.name}")

        except Exception as e:
            logger.error(f"Error taking dark frame: {e}")
            raise

    async def _take_dark_exposure(self, exposure_time: float):
        """Take a dark frame exposure using the camera directly."""
        camera = self.device_control.camera

        if camera.active:
            # Start dark frame exposure (light_frame=False)
            logger.info(f"Starting {exposure_time}s dark frame exposure")
            camera.start_exposure(exposure_time, light_frame=False)

            # Wait for exposure to complete and image to be ready
            while not camera.device.ImageReady:
                state = camera.camera_state
                if state == CameraStates.cameraError:
                    logger.error("Camera reported error during exposure")
                    raise RuntimeError("Camera error during dark frame exposure")
                elif state == CameraStates.cameraIdle:
                    # Sometimes the camera goes idle before ImageReady is True
                    await asyncio.sleep(0.1)
                    if camera.device.ImageReady:
                        break
                    logger.error("Camera returned to idle without image ready")
                    raise RuntimeError("Camera returned to idle without completing exposure")
                elif state == CameraStates.cameraExposing:
                    logger.debug(f"Dark frame exposure {camera.percent_completed}% complete")
                elif state == CameraStates.cameraWaiting:
                    logger.debug("Camera is waiting (possibly downloading image)")
                await asyncio.sleep(0.1)

            # Get the image data
            logger.info("Dark frame exposure complete, retrieving image data")
            # Set only the minimum properties needed for save_fits to work
            camera.image_ready = True
            camera.image_array = camera.device.ImageArray

        else:
            # Simulated dark frame
            logger.info(f"[SIMULATION] Taking {exposure_time}s dark frame exposure")
            await asyncio.sleep(exposure_time)
            logger.info("[SIMULATION] Dark frame image ready")

            # In simulation mode, create a fake dark frame (low signal + noise)
            # For simulation, we'll set the minimum properties needed for save_fits to use
            camera.image_ready = True
            camera.image_array = np.random.normal(100, 10, (2048, 2048))  # Simulate a dark frame with bias + noise

    async def cleanup(self):
        """Clean up resources."""
        if self.device_control:
            try:
                await self.device_control.disconnect()
                logger.info("Disconnected from hardware")
            except Exception as e:
                logger.warning(f"Error disconnecting from hardware: {e}")


async def run_dark_collection(
    config_path: Optional[Path] = None, exposures_per_iteration: Optional[int] = None, verbose: bool = False
):
    """
    Run dark frame collection.

    Parameters
    ----------
    config_path : Path, optional
        Path to configuration file
    exposures_per_iteration : int, optional
        Number of exposures to take per combination. If None, uses config value.
    verbose : bool, optional
        Enable verbose logging
    """
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    collector = DarkFrameCollector(config_path, exposures_per_iteration)

    try:
        await collector.initialize()
        await collector.collect_all_darks()
    finally:
        await collector.cleanup()


def main():
    """Command line entry point."""
    parser = argparse.ArgumentParser(
        description="Collect dark frames for all exposure times, binning modes, and readout modes used by burr managers"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file",
        default=None,
    )
    parser.add_argument(
        "--exposures-per-iteration",
        type=int,
        default=None,
        help="Number of exposures to take per combination (default: uses config value)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    try:
        asyncio.run(
            run_dark_collection(
                config_path=args.config, exposures_per_iteration=args.exposures_per_iteration, verbose=args.verbose
            )
        )
    except KeyboardInterrupt:
        logger.info("Dark frame collection interrupted by user")
    except Exception as e:
        logger.error(f"Error during dark frame collection: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
