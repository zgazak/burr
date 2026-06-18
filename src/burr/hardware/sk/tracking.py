"""
Shared completion tracking for the SK integration.

The SK wire protocol strips a burr ``ObservationRequest`` down to a
``BurrCollectPayload`` that rides in ``StandardCollectTask.context``, so
``BurrController._collect`` only sees the payload — not the original
request. But the scheduler slot and task-source ``on_request_completed``
hook both need the original request to update per-source progress
(coverage pixels observed, photometry stars measured, flats counted).

``SkTracker`` bridges the two: ``BurrProgram._generate`` registers the
slot+request keyed by ``task_id`` at dispatch time, and
``BurrController._collect`` calls ``record_completion`` with the same
``task_id`` at finish time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID

from loguru import logger

from burr.models.observation import ObservationRequest
from burr.models.run import BurrRun
from burr.models.site import LightingCondition
from burr.scheduler.scheduler import Scheduler
from burr.scheduler.slot import TaskSlot


@dataclass
class _InflightEntry:
    slot: TaskSlot
    request: ObservationRequest
    started_at: datetime


@dataclass
class SkTracker:
    """Shared state between BurrProgram and BurrController."""

    scheduler: Scheduler
    run: BurrRun
    _inflight: dict[UUID, _InflightEntry] = field(default_factory=dict)

    def register(
        self,
        task_id: UUID,
        slot: TaskSlot,
        request: ObservationRequest,
    ) -> None:
        self._inflight[task_id] = _InflightEntry(
            slot=slot,
            request=request,
            started_at=datetime.now(timezone.utc),
        )

    def record_completion(
        self,
        task_id: UUID,
        success: bool,
        condition: LightingCondition,
        error: str | None = None,
    ) -> None:
        entry = self._inflight.pop(task_id, None)
        if entry is None:
            logger.warning(f"SkTracker: no inflight entry for task {task_id}")
            return
        duration = (datetime.now(timezone.utc) - entry.started_at).total_seconds()
        if success:
            entry.slot.record_success(duration, condition)
            entry.slot.source.on_request_completed(entry.request, duration)
        else:
            entry.slot.record_failure()
            entry.slot.source.on_request_failed(entry.request, error or "unknown")
