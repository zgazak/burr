"""
TaskSlot wraps a TaskSource with scheduling state and strategy.
"""

from datetime import datetime
from zoneinfo import ZoneInfo
from loguru import logger

from burr.models.site import LightingCondition
from burr.task_source.protocol import TaskSource
from burr.scheduler.strategies import SchedulingStrategy, OneShotStrategy


class TaskSlot:
    """Wraps a TaskSource with scheduling metadata and strategy."""

    def __init__(
        self,
        source: TaskSource,
        priority: int,
        strategy: SchedulingStrategy,
    ):
        self.source = source
        self.priority = priority
        self.strategy = strategy
        self.last_used: datetime | None = None
        self.success_count: int = 0
        self.failure_count: int = 0

    @property
    def name(self) -> str:
        return self.source.name

    def is_suitable(self, condition: LightingCondition) -> bool:
        """Check if this slot is suitable for the given lighting condition."""
        if condition not in self.source.suitable_conditions:
            return False

        # For one-shot strategies, the source's collection-complete check is
        # the source of truth when available — completed_conditions is only
        # populated on live record_success and won't reflect prior runs
        # restored from history, which can leave a finished source eligible
        # forever (priority 0 sources then starve everything else).
        if isinstance(self.strategy, OneShotStrategy):
            if hasattr(self.source, "is_collection_complete"):
                return not self.source.is_collection_complete()
            if condition in self.strategy.completed_conditions:
                return False

        return True

    def get_priority_score(
        self,
        condition: LightingCondition,
        all_slots: list["TaskSlot"],
        is_only_option: bool = False,
    ) -> float:
        """Get priority score for current conditions."""
        if not self.is_suitable(condition):
            return float("inf")

        return self.strategy.get_priority_score(
            condition, self.priority, all_slots, is_only_option
        )

    def record_success(self, duration_seconds: float, condition: LightingCondition) -> None:
        """Record successful task execution."""
        self.last_used = datetime.now(ZoneInfo("UTC"))
        self.success_count += 1
        self.strategy.record_execution(duration_seconds, condition)

        # For one-shot, check if collection is now complete
        if isinstance(self.strategy, OneShotStrategy):
            if hasattr(self.source, "is_collection_complete") and self.source.is_collection_complete():
                self.strategy.mark_condition_complete(condition)

    def record_failure(self) -> None:
        """Record failed task execution."""
        self.failure_count += 1

    def restore_state(self, commands: list[dict]) -> None:
        """Restore state from run command history."""
        self.strategy.restore_from_history(commands, self.name)

        # Restore success/failure counts
        for cmd in commands:
            if not cmd["command"].startswith(self.name):
                continue
            timestamp = datetime.fromisoformat(cmd["timestamp"])
            if self.last_used is None or timestamp > self.last_used:
                self.last_used = timestamp
            if "error" in cmd and cmd["error"]:
                self.failure_count += 1
            else:
                self.success_count += 1

        logger.info(
            f"Restored {self.name} state: "
            f"{self.success_count}/{self.success_count + self.failure_count} successful"
        )

    def status_line(self, all_slots: list["TaskSlot"]) -> str:
        """Generate a status line for logging."""
        from burr.scheduler.strategies import TimeShareStrategy, IntervalStrategy

        if isinstance(self.strategy, TimeShareStrategy):
            actual = self.strategy.get_time_share_percentage(all_slots)
            target = self.strategy.get_normalized_target(all_slots)
            return (
                f"{self.name} (time_share) - {actual:.1f}% used "
                f"(target: {target:.1f}%) | "
                f"{self.success_count}/{self.success_count + self.failure_count} successful"
            )
        elif isinstance(self.strategy, IntervalStrategy):
            time_since = (
                "never"
                if self.strategy.last_used is None
                else f"{(datetime.now(ZoneInfo('UTC')) - self.strategy.last_used).total_seconds() / 60:.1f}m"
            )
            return (
                f"{self.name} (interval) - {time_since} since last "
                f"(min: {self.strategy.minimum_interval.total_seconds() / 60:.1f}m) | "
                f"{self.success_count}/{self.success_count + self.failure_count} successful"
            )
        else:
            # One-shot
            if hasattr(self.source, "is_collection_complete"):
                status = "completed" if self.source.is_collection_complete() else "in progress"
            else:
                status = "completed" if self.strategy.completed_conditions else "not completed"
            return (
                f"{self.name} (one_shot) - {status} | "
                f"{self.success_count}/{self.success_count + self.failure_count} successful"
            )
