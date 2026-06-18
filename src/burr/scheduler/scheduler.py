"""
Central Scheduler that selects the best task source for current conditions.

Used identically by both DirectRunner and BurrProgram.
"""

from loguru import logger

from burr.models.site import LightingCondition
from burr.task_source.protocol import TaskSource
from burr.scheduler.strategies import SchedulingStrategy
from burr.scheduler.slot import TaskSlot


class Scheduler:
    """Selects the best task source based on lighting conditions and scheduling strategies."""

    def __init__(self):
        self._slots: list[TaskSlot] = []

    @property
    def slots(self) -> list[TaskSlot]:
        return self._slots

    def register(
        self,
        source: TaskSource,
        priority: int,
        strategy: SchedulingStrategy,
    ) -> TaskSlot:
        """Register a task source with its scheduling strategy.

        Parameters
        ----------
        source : TaskSource
            The task source to register.
        priority : int
            Base priority (lower = higher priority).
        strategy : SchedulingStrategy
            The scheduling strategy for this source.

        Returns
        -------
        TaskSlot
            The created slot wrapper.
        """
        slot = TaskSlot(source=source, priority=priority, strategy=strategy)
        self._slots.append(slot)
        logger.info(f"Registered task source '{source.name}' with priority {priority}")
        return slot

    def select_best(self, condition: LightingCondition) -> TaskSlot | None:
        """Select the best task slot for the given lighting condition.

        Parameters
        ----------
        condition : LightingCondition
            Current lighting condition.

        Returns
        -------
        TaskSlot | None
            The best slot, or None if no suitable slot is available.
        """
        ranked = self.ranked_suitable(condition)
        return ranked[0] if ranked else None

    def ranked_suitable(self, condition: LightingCondition) -> list[TaskSlot]:
        """All slots suitable for ``condition``, sorted by priority score.

        Callers should iterate this list and try each slot's
        ``generate_request`` in order — a single slot returning ``None`` (no
        target currently producible) shouldn't consume the whole dispatch
        cycle while lower-priority slots still have work to do.

        Parameters
        ----------
        condition : LightingCondition
            Current lighting condition.

        Returns
        -------
        list[TaskSlot]
            Sorted best-first. Empty if no slot is suitable.
        """
        suitable = [s for s in self._slots if s.is_suitable(condition)]
        if not suitable:
            return []

        is_only = len(suitable) == 1
        suitable.sort(
            key=lambda s: s.get_priority_score(condition, self._slots, is_only)
        )
        return suitable

    def restore_state(self, executed_commands: list[dict]) -> None:
        """Restore all slot states from run command history.

        Parameters
        ----------
        executed_commands : list[dict]
            Command history from BurrRun.
        """
        for slot in self._slots:
            slot.restore_state(executed_commands)

    def log_status(
        self,
        condition: LightingCondition | None = None,
        offer_intervals: list[tuple] | None = None,
    ) -> None:
        """Log current status of all slots.

        Parameters
        ----------
        condition : LightingCondition, optional
            Current lighting condition; logged on a trailer line.
        offer_intervals : list[tuple[datetime, datetime]], optional
            Program-level offer windows currently published to SK (or
            equivalent for direct mode). Each is logged on its own line so
            operators can see at a glance what windows burr is actually
            telling the harness it can work in — useful for diagnosing
            "schedule item asks for day but burr never runs during day".
        """
        logger.info("Task Scheduler Status:")
        for i, slot in enumerate(self._slots):
            prefix = "└──" if i == len(self._slots) - 1 else "├──"
            logger.info(f"{prefix} {slot.status_line(self._slots)}")
        if condition:
            logger.info(f"└── Current condition: {condition.value}")
        if offer_intervals:
            logger.info(f"└── Offer windows ({len(offer_intervals)}):")
            for start, end in offer_intervals:
                logger.info(f"    • {start} → {end}")
