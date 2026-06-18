import asyncio
from pathlib import Path
import signal
from loguru import logger
from burr.run.utils import get_run
import sys
import json
from datetime import datetime
from zoneinfo import ZoneInfo


class ProcessingService:
    """Service for processing astronomical data"""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.running = True
        self.run = None

    async def start(self):
        """Start the processing service"""
        try:
            # Initialize the run using the existing get_run_sk function
            self.run = get_run(self.config_path)
            # Connect to NATS for real-time updates
            await self.run.connect_nats()

            # Subscribe to frame collection messages
            await self.run._nc.subscribe(
                "burr.frames.collected",
                cb=self._handle_frame_collected,
                queue="burr_frames",  # Load balance if multiple subscribers
            )

            # Setup signal handlers
            for sig in (signal.SIGTERM, signal.SIGINT):
                asyncio.get_event_loop().add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.shutdown(s)))

            logger.info("Processing service started")

            while self.running:
                try:
                    # Process any pending tasks
                    await self.process_pending()
                    await asyncio.sleep(1)  # Or appropriate interval

                except Exception as e:
                    logger.error(f"Error in processing loop: {e}")

        except Exception as e:
            logger.error(f"Failed to start processing service: {e}")
            raise

        finally:
            if self.run:
                await self.run.disconnect_nats()

    async def shutdown(self, sig=None):
        """Shutdown the processing service"""
        if sig:
            logger.info(f"Received signal {sig.name}, shutting down...")
        self.running = False

    async def process_pending(self):
        """Process any pending tasks"""
        # Here you would check for new data files, database entries, etc.
        # For now, this is a placeholder
        pass

    async def _handle_frame_collected(self, msg):
        """Handle incoming frame collection messages"""
        try:
            data = json.loads(msg.data.decode())
            frame_path = data.get("frame_path")
            frame_type = data.get("frame_type")
            metadata = data.get("metadata", {})

            logger.info(f"Received frame collection message: {frame_type} frame at {frame_path}")
            logger.info(f"Frame metadata: {metadata}")

            # Here we would process the frame based on its type
            # For now just acknowledge receipt
            await self.run.publish_result(
                "last_frame_processed",
                {
                    "path": frame_path,
                    "type": frame_type,
                    "metadata": metadata,
                    "timestamp": datetime.now(ZoneInfo("UTC")).isoformat(),
                },
            )

        except Exception as e:
            logger.error(f"Error handling frame collection message: {e}")
            logger.exception(e)  # Log the full traceback

    async def process_plate_scale(self, data_path: Path):
        """Process plate scale from an image"""
        try:
            # Here you would implement the actual plate scale calculation
            # For now, just a placeholder
            plate_scale = 1.23  # arcsec/pixel

            # Publish the result
            await self.run.publish_result("plate_scale", plate_scale)

        except Exception as e:
            logger.error(f"Error processing plate scale from {data_path}: {e}")


def main():
    """Main entry point when run as a module"""
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    else:
        config_path = Path("burr_config.yaml")

    service = ProcessingService(config_path)
    asyncio.run(service.start())


if __name__ == "__main__":
    main()
