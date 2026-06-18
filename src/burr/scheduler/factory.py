"""
Factory for building a Scheduler from configuration.

Called identically by both DirectRunner and BurrProgram.
"""

from loguru import logger

from burr.core.config import AppConfig
from burr.models.run import BurrRun
from burr.models.site import LightingCondition
from burr.scheduler.scheduler import Scheduler
from burr.scheduler.strategies import (
    TimeShareStrategy,
    IntervalStrategy,
    OneShotStrategy,
)
from burr.task_source.protocol import TaskSource


def _parse_conditions(windows: list[str]) -> list[LightingCondition]:
    """Convert config window strings to LightingCondition enums."""
    return [LightingCondition(w.lower()) for w in windows]


def build_scheduler(
    config: AppConfig,
    run: BurrRun,
    sources: dict[str, TaskSource],
) -> Scheduler:
    """Build a Scheduler from configuration and task sources.

    Parameters
    ----------
    config : AppConfig
        Application configuration.
    run : BurrRun
        Current run (for restoring state from history).
    sources : dict[str, TaskSource]
        Map of source name -> TaskSource instance.

    Returns
    -------
    Scheduler
        Configured scheduler with all sources registered.
    """
    scheduler = Scheduler()

    # Register twilight flats (one_shot, priority 1)
    if "twilight_flats" in sources:
        scheduler.register(
            source=sources["twilight_flats"],
            priority=1,
            strategy=OneShotStrategy(),
        )

    # Register lunar background (one_shot, priority 0 - highest)
    if "lunar_background" in sources:
        scheduler.register(
            source=sources["lunar_background"],
            priority=0,
            strategy=OneShotStrategy(),
        )

    # Register photometric standards (time_share, priority 2)
    if "photometric_standards" in sources:
        sched = config.schedule.photometric_standards.scheduling
        scheduler.register(
            source=sources["photometric_standards"],
            priority=2,
            strategy=TimeShareStrategy(target_percentage=sched.target_percentage),
        )

    # Register coverage (time_share, priority 2)
    if "coverage" in sources:
        sched = config.schedule.coverage.scheduling
        scheduler.register(
            source=sources["coverage"],
            priority=2,
            strategy=TimeShareStrategy(target_percentage=sched.target_percentage),
        )

    # Register calsats (interval, priority 3)
    if "calsats" in sources:
        sched = config.schedule.calsats.scheduling
        scheduler.register(
            source=sources["calsats"],
            priority=3,
            strategy=IntervalStrategy(
                minimum_interval_minutes=sched.minimum_interval_minutes,
                fill_gaps=sched.fill_gaps,
            ),
        )

    # Restore state from run history
    scheduler.restore_state(run.executed_commands)

    logger.info(f"Built scheduler with {len(scheduler.slots)} task sources")
    return scheduler
