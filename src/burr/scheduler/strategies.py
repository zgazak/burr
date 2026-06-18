"""
Scheduling strategies extracted from runner.py's TaskManager.

Three strategies control how task sources are scheduled:
- TimeShareStrategy: percentage-based allocation
- IntervalStrategy: minimum interval between runs
- OneShotStrategy: once per lighting condition
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from burr.models.site import LightingCondition


class SchedulingStrategy(ABC):
    """Base class for scheduling strategies."""

    @abstractmethod
    def get_priority_score(
        self,
        current_condition: LightingCondition,
        base_priority: int,
        all_slots: list,
        is_only_option: bool = False,
    ) -> float:
        """Calculate priority score (lower is higher priority).

        Parameters
        ----------
        current_condition : LightingCondition
            Current lighting condition.
        base_priority : int
            Base priority of the task slot.
        all_slots : list
            All registered TaskSlots (for relative calculations).
        is_only_option : bool
            Whether this is the only suitable slot.

        Returns
        -------
        float
            Priority score. Lower = higher priority. inf = not eligible.
        """
        ...

    @abstractmethod
    def is_eligible(self, current_condition: LightingCondition) -> bool:
        """Check if this strategy allows execution now."""
        ...

    @abstractmethod
    def record_execution(self, duration_seconds: float, condition: LightingCondition) -> None:
        """Record that a task was executed."""
        ...

    @abstractmethod
    def restore_from_history(self, commands: list[dict], source_name: str) -> None:
        """Restore state from run command history."""
        ...


class TimeShareStrategy(SchedulingStrategy):
    """Percentage-based time allocation."""

    def __init__(self, target_percentage: float):
        self.target_percentage = target_percentage
        self.time_used: float = 0.0

    def get_time_share_percentage(self, all_slots: list) -> float:
        """Calculate actual time share relative to all time_share slots."""
        total = sum(
            s.strategy.time_used
            for s in all_slots
            if isinstance(s.strategy, TimeShareStrategy)
        )
        if total == 0:
            return 0.0
        return (self.time_used / total) * 100.0

    def get_normalized_target(self, all_slots: list) -> float:
        """Get normalized target percentage (ensures all time_share targets sum to 100%)."""
        total_target = sum(
            s.strategy.target_percentage
            for s in all_slots
            if isinstance(s.strategy, TimeShareStrategy)
        )
        if total_target == 0:
            return 0.0
        return (self.target_percentage / total_target) * 100.0

    def get_priority_score(
        self,
        current_condition: LightingCondition,
        base_priority: int,
        all_slots: list,
        is_only_option: bool = False,
    ) -> float:
        actual = self.get_time_share_percentage(all_slots)
        target = self.get_normalized_target(all_slots)
        diff = actual - target

        if diff < 0:
            return diff  # Under target -> high priority (negative)
        else:
            return diff * 2  # Over target -> penalized

    def is_eligible(self, current_condition: LightingCondition) -> bool:
        return True  # Always eligible if condition matches

    def record_execution(self, duration_seconds: float, condition: LightingCondition) -> None:
        self.time_used += duration_seconds

    def restore_from_history(self, commands: list[dict], source_name: str) -> None:
        for cmd in commands:
            if not cmd["command"].startswith(source_name):
                continue
            if "metadata" in cmd and cmd["metadata"] and "task_duration" in cmd["metadata"]:
                self.time_used += float(cmd["metadata"]["task_duration"])


class IntervalStrategy(SchedulingStrategy):
    """Minimum interval between executions."""

    def __init__(self, minimum_interval_minutes: int, fill_gaps: bool = True):
        self.minimum_interval = timedelta(minutes=minimum_interval_minutes)
        self.fill_gaps = fill_gaps
        self.last_used: datetime | None = None

    def get_priority_score(
        self,
        current_condition: LightingCondition,
        base_priority: int,
        all_slots: list,
        is_only_option: bool = False,
    ) -> float:
        if self.last_used is None:
            return -100.0  # Never used -> highest priority

        time_since = datetime.now(ZoneInfo("UTC")) - self.last_used
        if time_since < self.minimum_interval:
            if not self.fill_gaps and not is_only_option:
                return float("inf")
            return base_priority + 2.0  # Low priority fill
        else:
            intervals_passed = time_since.total_seconds() / self.minimum_interval.total_seconds()
            if intervals_passed > 2:
                return -50.0 * (intervals_passed - 1)
            else:
                return base_priority - (intervals_passed - 1)

    def is_eligible(self, current_condition: LightingCondition) -> bool:
        if self.last_used is None:
            return True
        time_since = datetime.now(ZoneInfo("UTC")) - self.last_used
        return time_since >= self.minimum_interval or self.fill_gaps

    def record_execution(self, duration_seconds: float, condition: LightingCondition) -> None:
        self.last_used = datetime.now(ZoneInfo("UTC"))

    def restore_from_history(self, commands: list[dict], source_name: str) -> None:
        for cmd in commands:
            if not cmd["command"].startswith(source_name):
                continue
            timestamp = datetime.fromisoformat(cmd["timestamp"])
            if self.last_used is None or timestamp > self.last_used:
                self.last_used = timestamp


class OneShotStrategy(SchedulingStrategy):
    """Execute once per lighting condition."""

    def __init__(self):
        self.completed_conditions: set[LightingCondition] = set()

    def get_priority_score(
        self,
        current_condition: LightingCondition,
        base_priority: int,
        all_slots: list,
        is_only_option: bool = False,
    ) -> float:
        if current_condition not in self.completed_conditions:
            score = base_priority - 2.0
            if base_priority <= 1:
                return -1000.0 + score
            return score
        return float("inf")

    def is_eligible(self, current_condition: LightingCondition) -> bool:
        return current_condition not in self.completed_conditions

    def record_execution(self, duration_seconds: float, condition: LightingCondition) -> None:
        # Don't mark complete here - the slot checks source.is_collection_complete()
        pass

    def mark_condition_complete(self, condition: LightingCondition) -> None:
        """Explicitly mark a condition as complete."""
        self.completed_conditions.add(condition)

    def restore_from_history(self, commands: list[dict], source_name: str) -> None:
        for cmd in commands:
            if not cmd["command"].startswith(source_name):
                continue
            if cmd["command"].endswith("_completed"):
                if "metadata" in cmd and cmd["metadata"]:
                    condition_str = cmd["metadata"].get("lighting_condition")
                    if condition_str:
                        self.completed_conditions.add(LightingCondition(condition_str.lower()))
