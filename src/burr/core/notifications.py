"""Notification service for burr that handles both logging and optional Slack integration."""

import importlib.util
from enum import Enum
from typing import Optional, Tuple
import os

from loguru import logger


class NotificationLevel(str, Enum):
    """Levels of notification importance."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


class NotificationService:
    """Service for sending notifications to both logger and optionally Slack."""

    def __init__(self):
        self._slack_client = None
        self._setup_slack()

    def _setup_slack(self):
        """Attempt to setup Slack client if slack_sdk is installed and token is available."""
        if importlib.util.find_spec("slack_sdk") is not None:
            from slack_sdk import WebClient

            token = os.environ.get("SLACK_BOT_TOKEN")
            if token:
                self._slack_client = WebClient(token=token)
                logger.info("Slack notifications enabled")
            else:
                logger.warning("Slack SDK installed but SLACK_BOT_TOKEN not found in environment")

    def _ensure_channel(self, channel: str) -> Tuple[bool, str]:
        """Ensure a channel exists, create it if possible.

        Args:
            channel: Channel name (with or without #)

        Returns:
            Tuple of (success, message)
        """
        if not self._slack_client:
            return False, "Slack client not initialized"

        # Strip # if present for API calls
        channel_name = channel.lstrip("#")

        try:
            # First try to find the channel
            response = self._slack_client.conversations_list(types="public_channel")
            for ch in response["channels"]:
                if ch["name"] == channel_name:
                    return True, f"Channel #{channel_name} already exists"

            # Channel not found, try to create it
            logger.info(f"Channel #{channel_name} not found, attempting to create it...")
            try:
                result = self._slack_client.conversations_create(name=channel_name)
                if result["ok"]:
                    logger.info(f"Created Slack channel #{channel_name}")
                    return True, f"Created channel #{channel_name}"
                else:
                    return False, f"Failed to create channel: {result.get('error', 'unknown error')}"
            except Exception as e:
                error_msg = str(e)
                if "name_taken" in error_msg:
                    # Race condition - channel was created between our check and create
                    return True, f"Channel #{channel_name} already exists"
                elif "not_allowed" in error_msg:
                    return (
                        False,
                        f"Bot doesn't have permission to create channels. Please create #{channel_name} manually and invite the bot.",
                    )
                else:
                    return False, f"Failed to create channel: {error_msg}"

        except Exception as e:
            return False, f"Error checking/creating channel: {str(e)}"

    def _get_slack_color(self, level: NotificationLevel) -> str:
        """Get the Slack attachment color for a notification level."""
        return {
            NotificationLevel.INFO: "#2196F3",  # Blue
            NotificationLevel.WARNING: "#FFC107",  # Amber
            NotificationLevel.ERROR: "#F44336",  # Red
            NotificationLevel.SUCCESS: "#4CAF50",  # Green
        }[level]

    def notify(
        self,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        title: Optional[str] = None,
        channel: Optional[str] = None,
    ):
        """
        Send a notification.

        Args:
            message: The main notification message
            level: Importance level of the notification
            title: Optional title for the notification (mainly for Slack)
            channel: Optional Slack channel override (defaults to configured channel)
        """
        # Always log
        log_func = {
            NotificationLevel.INFO: logger.info,
            NotificationLevel.WARNING: logger.warning,
            NotificationLevel.ERROR: logger.error,
            NotificationLevel.SUCCESS: logger.success,
        }[level]

        log_func(f"{title + ': ' if title else ''}{message}")

        # Send to Slack if available
        if self._slack_client and channel:
            # Ensure channel starts with # for public channels
            if not channel.startswith(("#", "@")):
                channel = f"#{channel}"

            # Try to ensure channel exists first
            success, channel_msg = self._ensure_channel(channel)
            if not success:
                logger.error(channel_msg)
                return

            # Create message text that includes title if present
            text_message = f"{title + ': ' if title else ''}{message}"

            try:
                # Send the message with all required fields
                self._slack_client.chat_postMessage(
                    channel=channel,
                    text=text_message,
                    blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": text_message}}],
                    attachments=[
                        {
                            "color": self._get_slack_color(level),
                            "title": title or "Notification",
                            "text": message,
                            "fallback": text_message,
                            "footer": "burr notification service",
                        }
                    ],
                )
            except Exception as e:
                error_msg = str(e)
                if "channel_not_found" in error_msg:
                    logger.error(f"Slack channel '{channel}' exists but bot needs to be invited. Use /invite @bot-name")
                else:
                    logger.error(f"Failed to send Slack notification: {error_msg}")


# Global notification service instance
notifier = NotificationService()
