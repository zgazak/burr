"""
Run state manager - handles IO operations extracted from BurrRun.

BurrRun is now a pure Pydantic data model. RunStateManager wraps it
with NATS connectivity, file persistence, and command logging.
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from loguru import logger
from nats.aio.client import Client as NATS

from burr.models.run import BurrRun


class RunStateManager:
    """Manages IO operations for a BurrRun instance.

    Handles:
    - NATS connectivity for real-time updates
    - Command logging with atomic persistence
    - Result publishing
    """

    def __init__(self, run: BurrRun):
        self.run = run
        self._nc: Optional[NATS] = None
        self._connected: bool = False

    async def connect_nats(self, nats_url: str = "nats://nats:4222") -> None:
        """Connect to NATS server for real-time updates."""
        if self._connected:
            return

        try:
            self._nc = NATS()
            await self._nc.connect(nats_url)
            await self._nc.subscribe(
                "burr.results.updates",
                cb=self._handle_update,
                queue="burr_results",
            )
            self._connected = True
            logger.info("Connected to NATS server for real-time updates")
        except Exception as e:
            logger.error(f"Failed to connect to NATS: {e}")
            raise

    async def disconnect_nats(self) -> None:
        """Disconnect from NATS server."""
        if self._nc:
            await self._nc.drain()
            self._connected = False
            logger.info("Disconnected from NATS server")

    async def _handle_update(self, msg) -> None:
        """Handle incoming result updates from NATS."""
        try:
            data = json.loads(msg.data.decode())
            key = data["key"]
            value = data["value"]
            timestamp = datetime.fromisoformat(data["timestamp"])
            self.run.results.set_value(key, value, timestamp)
            logger.debug(f"Updated {key} with new value from NATS")
        except Exception as e:
            logger.error(f"Error handling NATS update: {e}")

    async def publish_result(self, key: str, value: Any) -> None:
        """Publish a result to NATS and update local state."""
        self.run.results.set_value(key, value)

        if not self._connected:
            logger.debug("NATS not connected, skipping publication")
            return

        try:
            timestamp = datetime.now(ZoneInfo("UTC"))
            data = {"key": key, "value": value, "timestamp": timestamp.isoformat()}
            await self._nc.publish("burr.results.updates", json.dumps(data).encode())
            logger.debug(f"Published new value for {key} to NATS")
        except Exception as e:
            logger.error(f"Error publishing result to NATS: {e}")
            raise

    def log_command(
        self,
        command: str,
        result: Any = None,
        metadata: Dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Log a command execution and persist state.

        This is the preferred way to log commands - delegates to BurrRun.log_command()
        which handles persistence via save_run_state.
        """
        self.run.log_command(command, result, metadata, error)
