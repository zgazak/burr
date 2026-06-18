"""
CLI entrypoint for burr-shutdown using the hardware factory.
"""

import argparse
import asyncio
from pathlib import Path

from loguru import logger

from burr.bootstrap import initialize_engine
from burr.core.config import get_or_initialize_config
from burr.core.logging import setup_loguru
from burr.hardware.direct.device import DirectDeviceControl
from burr.hardware.factory import create_devices


async def shutdown_hardware(config_path: Path | list[Path]) -> None:
    """Safely shutdown all hardware devices."""
    initialize_engine()
    device_control = None

    try:
        logger.info("Starting hardware shutdown procedure...")
        config = get_or_initialize_config(config_path)
        setup_loguru(level=config.logging.level)

        devices = create_devices(config)
        device_control = DirectDeviceControl(
            camera=devices.camera,
            dome=devices.dome,
            mount=devices.mount,
            weather=devices.weather,
            config=config,
        )

        if devices.any_active:
            logger.info("Connecting to active hardware devices...")
            await device_control.connect()
        else:
            logger.info("All devices are simulated")

        # Step 1: Close dome
        if devices.dome.active:
            logger.info("Step 1: Closing dome...")
            await device_control.close_dome()
        else:
            logger.info("Step 1: [SIMULATION] Closing dome...")

        # Step 2-3: Verification (same as original)
        if devices.mount.active and devices.camera.active:
            logger.info("Step 2: Finding bright star for verification...")
            from burr.core.constants import LANDOLT_STARS_PATH
            from burr.task_source.photometry.catalog import load_catalog
            from datetime import datetime
            from zoneinfo import ZoneInfo

            try:
                catalog = load_catalog(Path(LANDOLT_STARS_PATH))
                now = datetime.now(ZoneInfo("UTC"))

                suitable = []
                for star in catalog:
                    try:
                        alt = config.site.calculate_altitude(ra_deg=star.ra_deg, dec_deg=star.dec_deg, time=now)
                        if alt >= 30.0 and star.vmag <= 6.0:
                            suitable.append((star, alt))
                    except Exception:
                        continue

                if suitable:
                    suitable.sort(key=lambda x: (-x[1], x[0].vmag))
                    star, alt = suitable[0]
                    logger.info(f"Selected {star.name} (V={star.vmag:.1f}, Alt={alt:.1f})")

                    from burr.models.tracking import TrackingMode
                    await device_control.goto_ra_dec(star.ra_deg, star.dec_deg, TrackingMode.SIDEREAL)

                    logger.info("Step 3: Taking verification image...")
                    await device_control.take_exposure(1.0, "verification_dome_closed.fits")
            except Exception as e:
                logger.warning(f"Skipping verification: {e}")
        else:
            logger.info("Step 2-3: [SIMULATION] Verification...")

        # Step 4: Park mount
        if devices.mount.active:
            logger.info("Step 4: Parking mount...")
            await device_control.park_mount()
        else:
            logger.info("Step 4: [SIMULATION] Parking mount...")

        # Step 5: Disconnect
        if devices.any_active:
            logger.info("Step 5: Disconnecting...")
            await device_control.disconnect()
        else:
            logger.info("Step 5: [SIMULATION] Disconnecting...")

        logger.info("Hardware shutdown complete!")

    except KeyboardInterrupt:
        logger.warning("Shutdown interrupted - emergency disconnect...")
        if device_control:
            try:
                await device_control.emergency_disconnect()
            except Exception:
                pass
        raise
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        if device_control:
            try:
                await device_control.emergency_disconnect()
            except Exception:
                pass
        raise


def main():
    """CLI entrypoint for burr-shutdown."""
    parser = argparse.ArgumentParser(description="Safely shutdown all hardware devices")
    parser.add_argument("--config", type=Path, default=None,
                        help="Single YAML config file (legacy). Mutually exclusive with --site/--preset.")
    parser.add_argument("--site", type=Path, default=None,
                        help="Site YAML (hardware, lat/lon). Combine with --preset.")
    parser.add_argument("--preset", type=Path, default=None,
                        help="Mode preset YAML (scheduling). Deep-merged on top of --site.")
    args = parser.parse_args()

    if args.site or args.preset:
        if args.config:
            parser.error("--config cannot be combined with --site/--preset")
        if not args.site or not args.preset:
            parser.error("--site and --preset must be supplied together")
        config_path: Path | list[Path] = [args.site, args.preset]
    else:
        config_path = args.config

    try:
        asyncio.run(shutdown_hardware(config_path))
    except KeyboardInterrupt:
        logger.warning("Shutdown interrupted by user")
    except Exception as e:
        logger.error(f"Shutdown failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
