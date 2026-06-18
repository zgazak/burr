"""
CLI entrypoint for burr-direct using the new architecture.
"""

import argparse
import asyncio
import signal
from pathlib import Path

from loguru import logger as loguru_logger

from burr.bootstrap import initialize_engine
from burr.core.config import get_or_initialize_config
from burr.hardware.direct.runner import DirectRunner
from burr.run.utils import get_or_create_run


class DirectRunnerWrapper:
    """Wrapper for DirectRunner with graceful shutdown handling."""

    def __init__(self, config_path: Path | list[Path]):
        self.config_path = config_path
        self.runner = None
        self._shutdown_requested = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        def handler(signum, frame):
            loguru_logger.info(f"Interrupt signal received (signal {signum})")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    async def run(self):
        try:
            config = get_or_initialize_config(self.config_path)
            run = get_or_create_run(site=config.site, verbose=True, new_config=config)
            self.runner = DirectRunner(run, self.config_path)
            await self.runner.start()
        except Exception as e:
            loguru_logger.error(f"Error in direct hardware control: {e}")
            raise
        finally:
            await self._cleanup()

    async def _cleanup(self):
        if not self.runner:
            return
        try:
            await self.runner.stop()
            if self._shutdown_requested and self.runner.device_control:
                await self._handle_safe_shutdown()
        except Exception as e:
            loguru_logger.error(f"Error during cleanup: {e}")
            if self.runner and self.runner.device_control:
                try:
                    await self.runner.device_control.disconnect()
                except Exception:
                    pass

    async def _handle_safe_shutdown(self):
        try:
            print("\n What would you like to do?")
            print("  Y) Close dome and stow mount, then disconnect (SAFE)")
            print("  N) Just disconnect without moving hardware")

            try:
                response = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, input, "Choice [Y/n]: "),
                    timeout=10.0,
                )
                response = response.strip().lower()
            except asyncio.TimeoutError:
                response = "y"

            if response == "" or response.startswith("y"):
                loguru_logger.info("Performing safe shutdown...")
                await self.runner.device_control.park_mount()
                await self.runner.device_control.close_dome()
                await self.runner.device_control.disconnect()
            else:
                await self.runner.device_control.disconnect()
        except Exception as e:
            loguru_logger.error(f"Error during shutdown: {e}")
            if self.runner.device_control:
                try:
                    await self.runner.device_control.disconnect()
                except Exception:
                    pass


def main():
    """CLI entrypoint for burr-direct."""
    parser = argparse.ArgumentParser(description="Direct hardware control for burr")
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

    initialize_engine()

    try:
        wrapper = DirectRunnerWrapper(config_path)
        asyncio.run(wrapper.run())
    except KeyboardInterrupt:
        loguru_logger.info("Exiting due to keyboard interrupt")
    except Exception as e:
        loguru_logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
