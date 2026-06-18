"""Burr SensorKit program — functional-style declaration.

This module is the canonical burr SK entry point. It uses sensorkit's
declarative functional API (``sk.declare_program`` + per-callback
decorators) rather than a class-based program, matching sensorkit's
preferred idiom.

Launch modes (pick one):

  # From inside the sensorkit venv, as a standalone service:
  sensorkit service run burr_service burr.hardware.sk.burr_program:main

  # Or registered in services.yaml alongside alpaca/pwi4/controller/agent/webapi:
  - name: burr_service
    module: burr.hardware.sk.burr_program
    func: main

  # Or via burr's own CLI (which defers to ``main`` under the hood):
  burr-sk --site ... --preset ...

The name you give the service (``burr_service`` above, or whatever you
pass) becomes the name of the SK program entity as well — burr's program
declaration is nameless, so sensorkit's single-entity-service rule
inherits the service name. Don't pick a name that collides with another
entity already on the bus.

Either way, burr needs its site+preset config. When invoked from
``sensorkit service run`` or a services.yaml entry, point ``BURR_CONFIG``
at the YAML path(s), semicolon-separated for site+preset composition::

  $env:BURR_CONFIG = "deploy/example/site.yaml;deploy/presets/sda.yaml"
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from collections import deque
from collections.abc import Iterator

import sensorkit.api as sk
from loguru import logger
from sensorkit.astro.common import ReferenceFrame
from sensorkit.common.keyword import KeywordDict
from sensorkit.std.collect import CameraParameterSet, StandardCollectTask

from burr.bootstrap import initialize_engine
from burr.core.config import AppConfig, get_or_initialize_config
from burr.core.logging import setup_loguru
from burr.hardware.sk.tasks import (
    BurrContext,
    to_sk_ratetarget,
    to_sk_sidereal_target,
    to_sk_target,
)
from burr.hardware.sk.tracking import SkTracker
from burr.models.observation import ObservationRequest, TLETarget
from burr.models.run import BurrRun
from burr.models.site import LightingCondition
from burr.models.tracking import Rates, TrackingMode
from burr.run.utils import get_or_create_run
from burr.scheduler.factory import build_scheduler
from burr.scheduler.scheduler import Scheduler
from burr.task_source.protocol import TaskSource


@dataclass
class _ProgramContext:
    run: BurrRun
    config: AppConfig
    controller_entity: str
    scheduler: Scheduler
    tracker: SkTracker


# Populated by main() before service.run() fires any program callbacks.
_ctx: _ProgramContext | None = None


@dataclass
class _PendingTask:
    """A single SK task waiting to be handed to the executor, plus its
    request-level bookkeeping so ``_next_task`` knows when to record
    completion."""
    task: StandardCollectTask
    slot: "object"
    request: ObservationRequest
    is_last_of_request: bool


# Queue of SK tasks produced from burr ObservationRequests but not yet
# handed to the tasking loop. The SK executor calls ``_next_task`` once
# per task dispatch (it expects exactly one yield per invocation), so
# when a request fans out into multiple SK tasks (e.g. photometric
# multi-exposure, or sidereal-then-rate splits) we enqueue them here
# and pop one per factory call. Cleared at main() start.
_pending_queue: deque[_PendingTask] = deque()


def _required_conditions() -> set[LightingCondition]:
    """Union of suitable_conditions across all registered task sources."""
    assert _ctx is not None
    conditions: set[LightingCondition] = set()
    for slot in _ctx.scheduler.slots:
        conditions.update(slot.source.suitable_conditions)
    return conditions


def _offer_intervals() -> list[tuple[datetime, datetime]]:
    """Merged offer windows for the currently registered sources."""
    assert _ctx is not None
    return _ctx.run.lighting_schedule.merged_windows_for(_required_conditions())


async def _on_attach() -> None:
    """Publish burr's observing-window offers when the entity joins the bus.

    The offer is the union of every registered task source's
    ``suitable_conditions`` mapped onto the run's lighting schedule. SK uses
    this program-level window to decide when to hand burr a turn at all —
    per-task lighting gating in ``scheduler.select_best`` only fires inside
    it, so anything not covered here can never run regardless of preset.
    """
    assert _ctx is not None, "burr_program attached before main() populated _ctx"
    prog = sk.program()

    schedule = _ctx.run.lighting_schedule
    intervals = _offer_intervals()

    if not intervals:
        # No registered source declared any suitable condition. Fall back to
        # the original night-only window so attach doesn't silently publish
        # an empty offer list.
        intervals = [(schedule.evening_civil_start, schedule.morning_civil_end)]
        logger.warning(
            "No task sources declared suitable_conditions; "
            "falling back to civil-to-civil night offer"
        )

    prog.clear_offers()
    for start, end in intervals:
        prog.add_offer(start, end)
    await prog.publish_offers()
    for start, end in intervals:
        logger.info(f"burr published offer {start} → {end}")

    # Dump status so operators have one place that shows: what slots exist,
    # what condition we're in, and what windows SK will see.
    _ctx.scheduler.log_status(_current_condition(), offer_intervals=intervals)


async def _on_detach() -> None:
    logger.info("burr detaching from bus")


async def _on_enable() -> None:
    logger.info("burr enabled by agent")


async def _on_disable() -> None:
    logger.info("burr disabled by agent")


async def _next_task():
    """Yield exactly one StandardCollectTask per invocation.

    SK's executor calls this factory once per task dispatch and expects
    the generator to yield exactly once (then exit on the next asend).
    When an ObservationRequest fans out into multiple SK tasks, we
    enqueue them in ``_pending_queue`` and pop one per call. That keeps
    the generator short-lived per dispatch and avoids starving the
    lease-refresh loop during long multi-task collects.

    When the queue is empty we walk the scheduler's suitable slots in
    priority order until one produces a request. A slot returning ``None``
    from ``generate_request`` (e.g. calsats with nothing currently above
    the horizon, lunar reroll below the floor) is recorded as a failure
    and we fall through to the next candidate, instead of consuming the
    whole dispatch cycle and silently starving every lower-priority slot
    until SK polls again.
    """
    assert _ctx is not None

    if not _pending_queue:
        condition = _current_condition()
        logger.info(f"SK requested next task (condition={condition.value})")

        candidates = _ctx.scheduler.ranked_suitable(condition)
        if not candidates:
            logger.info(
                f"no suitable task source for condition={condition.value} — "
                "nothing to generate this cycle"
            )
            return  # StopAsyncIteration → NoTaskAvailable

        logger.info(
            f"considering {len(candidates)} suitable slot(s) in priority order: "
            f"{[s.name for s in candidates]}"
        )

        chosen_slot = None
        chosen_request = None
        for slot in candidates:
            task_id = str(uuid4())
            try:
                request = slot.source.generate_request(task_id=task_id)
            except Exception as e:
                logger.exception(
                    f"{slot.name}.generate_request raised: {e} — "
                    "recording failure, trying next slot"
                )
                slot.record_failure()
                continue

            if request is None:
                logger.info(
                    f"{slot.name}.generate_request returned None "
                    "(nothing to produce right now) — trying next slot"
                )
                slot.record_failure()
                continue

            chosen_slot = slot
            chosen_request = request
            break

        if chosen_slot is None:
            logger.info(
                "every suitable slot returned None this cycle — "
                "nothing to generate"
            )
            return

        logger.info(
            f"selected {chosen_slot.name} → generating request {chosen_request.task_id} "
            f"target={type(chosen_request.target).__name__} "
            f"exposures={len(chosen_request.exposure_seconds)} "
            f"modes={[s.tracking_mode.value for s in chosen_request.exposure_map]}"
        )

        try:
            sk_tasks = list(_build_sk_tasks(chosen_request))
        except Exception as e:
            logger.exception(
                f"could not build SK tasks for {chosen_slot.name} request: {e}"
            )
            chosen_slot.record_failure()
            return

        logger.info(
            f"{chosen_slot.name} request {chosen_request.task_id}: enqueued "
            f"{len(sk_tasks)} SK task(s) from {len(chosen_request.exposure_seconds)} "
            f"exposure(s) × {len(chosen_request.exposure_map)} mode(s)"
        )

        # One tracker entry per request, keyed on the FINAL SK task id
        # so record_completion (fired when that task returns) matches.
        _ctx.tracker.register(sk_tasks[-1].task_id, chosen_slot, chosen_request)

        for i, t in enumerate(sk_tasks):
            _pending_queue.append(
                _PendingTask(
                    task=t,
                    slot=chosen_slot,
                    request=chosen_request,
                    is_last_of_request=(i == len(sk_tasks) - 1),
                )
            )

    pending = _pending_queue.popleft()
    logger.info(
        f"yielding SK task {pending.task.task_id} for {pending.slot.name} "
        f"(request {pending.request.task_id}, "
        f"{'final' if pending.is_last_of_request else 'intermediate'} of fan-out)"
    )

    # Yield exactly once. The executor will resume us (via asend) with
    # the TaskExecutionResult after the controller finishes this task,
    # then expect StopAsyncIteration.
    result = yield pending.task

    if pending.is_last_of_request:
        # The request's last SK task has completed — record completion.
        # Treat any non-None result as success; the result type may or
        # may not expose success flags depending on sensorkit version.
        success = result is not None
        error = None
        if result is not None:
            if getattr(result, "success", None) is False:
                success = False
            error = getattr(result, "error", None)

        _ctx.tracker.record_completion(
            task_id=pending.task.task_id,
            success=success,
            condition=_current_condition(),
            error=error,
        )


def _current_condition() -> LightingCondition:
    assert _ctx is not None
    now = datetime.now(ZoneInfo("UTC"))
    try:
        return _ctx.run.lighting_schedule.get_lighting_condition(now)
    except Exception as e:
        logger.warning(f"could not determine lighting condition: {e}")
        return LightingCondition.NIGHT


def _build_sk_tasks(request: ObservationRequest) -> Iterator[StandardCollectTask]:
    """Translate an ObservationRequest into one or more StandardCollectTask instances.

    Burr's nested frame model (``exposure_seconds`` × ``exposure_map``) maps
    to one or more SK tasks per request:

    - **Uniform SIDEREAL**: one task per exposure, target=ICRSTarget,
      frame_count=len(exposure_map).
    - **Uniform RATE**: one task per exposure, target=RateTarget (concrete
      rates resolved for that exposure duration), frame_count=len(exposure_map).
    - **[RATE..., SIDEREAL...]** (rates first): one task per exposure,
      target=RateTarget, ``sidereal_track_from_frame`` set to the first
      sidereal frame index. Generic controller switches to ICRF at that frame.
    - **[SIDEREAL..., RATE...]** (sidereal first): two tasks per exposure —
      N sidereal frames at ICRSTarget, then M rate frames at RateTarget.
      The generic controller's ``sidereal_track_from_frame`` only switches
      TO sidereal, so we can't express this with one task.
    - Any other interleaving raises ``ValueError``.

    Targets feeding SIDEREAL frames are resolved through
    ``to_sk_sidereal_target``, which freezes an AltAz/Lunar pointing to ICRS at
    emit time (sensorkit tracks AltAzTarget as fixed). OFF-only requests (flats)
    keep their AltAz target.
    """
    assert _ctx is not None
    end_time = _ctx.run.lighting_schedule.schedule_end_utc
    controller_id = _ctx.controller_entity

    modes = [spec.tracking_mode for spec in request.exposure_map]
    frame_count = len(modes)
    # Per-request identity (target name + source); seq is added per exposure.
    targ, mode = _burr_identity(request)

    # Target for SIDEREAL/OFF frames. When any frame tracks sidereally we
    # freeze an AltAz/Lunar pointing to the ICRS coords it maps to *now*
    # (emit time): sensorkit tracks an AltAzTarget as FIXED, so coverage and
    # lunar (which map in alt/az but want sidereal tracking) would otherwise
    # hold the horizon coordinate and let the sky drift. Flats use OFF-only,
    # so they stay an AltAz target via to_sk_target and keep fixed tracking.
    if any(m == TrackingMode.SIDEREAL for m in modes):
        sidereal_target = to_sk_sidereal_target(
            request.target, site=_ctx.config.site
        )
    else:
        sidereal_target = to_sk_target(request.target)

    # A TLE target carries its own motion — PWI4's follow_tle handles the
    # rate. For non-TLE targets, request.rates must supply the rate vector.
    is_tle_target = isinstance(request.target, TLETarget)
    any_rate = any(m == TrackingMode.RATE for m in modes)
    if any_rate and not is_tle_target and request.rates is None:
        raise ValueError(
            f"{request.source_name}: exposure_map contains RATE frames, "
            f"target is not a TLETarget, and request.rates is None"
        )

    for exp_seconds in request.exposure_seconds:
        # A fresh BurrContext.seq per exposure: every SK task, and thus every
        # frame, fanned out from THIS exposure shares it — e.g. the ICRS
        # sidereal frame and the RateTarget rate frames of one sidereal-then-
        # rate collect carry the same seq — so downstream fitting regroups the
        # frames of one collect. Distinct exposure_seconds entries differ.
        # file_name stays a flat top-level key because sensorkit's
        # filesys.WriteFile reads ``context["file_name"]`` literally; its
        # template resolves the identity via ``BurrContext.<field>``.
        exposure_context = KeywordDict(
            BurrContext(seq=str(uuid4()), targ=targ, mode=mode),
            file_name=_FILE_NAME_TEMPLATE,
        )

        # Resolve rates for THIS exposure (non-TLE rate case only).
        # RatesFromStreakDistribution samples a fresh rate per call,
        # which is what we want for streak-training distribution.
        concrete_rates: Rates | None = None
        if any_rate and not is_tle_target:
            if isinstance(request.rates, Rates):
                concrete_rates = request.rates
            else:
                concrete_rates = request.rates.to_concrete_rates(exp_seconds)

        # Target for RATE-mode frames, built lazily — for all-OFF /
        # all-SIDEREAL requests (e.g. twilight flats) concrete_rates is
        # None and to_sk_ratetarget would crash, so we only construct
        # it on the branches that actually consume it.
        # - TLE target → follow the TLE (PWI4 follow_tle)
        # - otherwise → RateTarget built from request's concrete rates
        # (to_sk_ratetarget converts AltAz/Lunar to ICRS using site+now)
        def _rate_target():
            if is_tle_target:
                return sidereal_target  # already SKTLETarget via to_sk_target
            frame = (
                ReferenceFrame.CIRF
                if _ctx.config.sk.rate_target_frame == "cirf"
                else ReferenceFrame.ICRF
            )
            return to_sk_ratetarget(
                request.target,
                concrete_rates,
                site=_ctx.config.site,
                frame=frame,
            )

        cam = CameraParameterSet(
            integration_time_seconds=exp_seconds,
            frame_count=frame_count,
        )

        # SIDEREAL and OFF both route through the non-rate target path:
        # the PWI4 generic controller points to ICRSTarget / AltAzTarget
        # without applying a custom rate. Twilight flats use OFF, most
        # other sources use SIDEREAL; from our perspective they're the
        # same "no custom rate" case for SK task emission.
        if all(m != TrackingMode.RATE for m in modes):
            yield StandardCollectTask(
                task_id=uuid4(),
                controller_id=controller_id,
                target=sidereal_target,
                camera_params=cam,
                end_time=end_time,
                context=exposure_context,
            )
        elif all(m == TrackingMode.RATE for m in modes):
            yield StandardCollectTask(
                task_id=uuid4(),
                controller_id=controller_id,
                target=_rate_target(),
                camera_params=cam,
                end_time=end_time,
                context=exposure_context,
            )
        elif _is_rates_then_nonrate(modes):
            first_nonrate = next(
                i for i, m in enumerate(modes) if m != TrackingMode.RATE
            )
            yield StandardCollectTask(
                task_id=uuid4(),
                controller_id=controller_id,
                target=_rate_target(),
                camera_params=cam,
                end_time=end_time,
                sidereal_track_from_frame=first_nonrate,
                context=exposure_context,
            )
        elif _is_nonrate_then_rates(modes):
            first_rate = modes.index(TrackingMode.RATE)
            yield StandardCollectTask(
                task_id=uuid4(),
                controller_id=controller_id,
                target=sidereal_target,
                camera_params=CameraParameterSet(
                    integration_time_seconds=exp_seconds,
                    frame_count=first_rate,
                ),
                end_time=end_time,
                context=exposure_context,
            )
            yield StandardCollectTask(
                task_id=uuid4(),
                controller_id=controller_id,
                target=_rate_target(),
                camera_params=CameraParameterSet(
                    integration_time_seconds=exp_seconds,
                    frame_count=frame_count - first_rate,
                ),
                end_time=end_time,
                context=exposure_context,
            )
        else:
            raise ValueError(
                f"{request.source_name}: unsupported exposure_map pattern "
                f"{[m.value for m in modes]} — only uniform or "
                f"rates-then-non-rate / non-rate-then-rates are implemented"
            )


# FITS filename template carried as a flat ``file_name`` context key.
# sensorkit.data.filesys.WriteFile reads ``context["file_name"]`` literally,
# then resolves it per-frame via ``_resolve_template`` → ``context.eval``
# (``f"``-prefixed → evaluated with ``datetime``/``UTC`` in globals and the
# task context as locals). It reaches the identity through the BurrContext
# keyword, e.g. ``BurrContext.mode`` / ``BurrContext.targ``; ``frame_num`` is
# injected by the std controller per frame.
# Produces e.g. ``20260424T031530_photometric_standards_110-364_f0.fits``.
_FILE_NAME_TEMPLATE = (
    'f"{datetime.now(UTC):%Y%m%dT%H%M%S}'
    '_{BurrContext.mode}_{BurrContext.targ}_f{frame_num}.fits"'
)


def _burr_identity(request: ObservationRequest) -> tuple[str, str]:
    """Resolve ``(targ, mode)`` for a request's BurrContext keyword.

    ``mode`` is the task source name verbatim. ``targ`` is the best target
    name the source supplied, picked from its metadata and sanitized for
    filesystem safety (spaces / path separators → underscores) since it also
    lands in the FITS filename. Falls back to the request's ``task_id`` when a
    source supplies no name (e.g. flats, lunar).
    """
    metadata = request.metadata or {}
    targ = (
        metadata.get("target_name")
        or metadata.get("star_name")        # photometry
        or metadata.get("satellite_name")   # calsats
        or metadata.get("pixel_id")         # coverage
        or metadata.get("norad_id")
        or request.task_id
    )
    safe_targ = (
        str(targ)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )
    return safe_targ, request.source_name


def _is_rates_then_nonrate(modes: list[TrackingMode]) -> bool:
    """True iff ``modes`` starts with >=1 RATE followed by >=1 non-RATE frames."""
    try:
        first_nonrate = next(
            i for i, m in enumerate(modes) if m != TrackingMode.RATE
        )
    except StopIteration:
        return False
    if first_nonrate == 0:
        return False
    return all(m != TrackingMode.RATE for m in modes[first_nonrate:])


def _is_nonrate_then_rates(modes: list[TrackingMode]) -> bool:
    """True iff ``modes`` starts with >=1 non-RATE followed by >=1 RATE frames."""
    try:
        first_rate = modes.index(TrackingMode.RATE)
    except ValueError:
        return False
    if first_rate == 0:
        return False
    return (
        all(m != TrackingMode.RATE for m in modes[:first_rate])
        and all(m == TrackingMode.RATE for m in modes[first_rate:])
    )


def build_task_sources(run: BurrRun, config: AppConfig) -> dict[str, TaskSource]:
    """Build the full set of configured task sources."""
    from burr.task_source.calsats.manager import CalibrationSatelliteManager
    from burr.task_source.coverage.manager import SkyCoverageManager
    from burr.task_source.flats.manager import FlatFieldManager
    from burr.task_source.lunar.manager import LunarBackgroundManager
    from burr.task_source.photometry.manager import PhotometricStandardsManager

    sched = config.schedule
    sources: dict[str, TaskSource] = {}

    if sched.twilight_flats.collect:
        sources["twilight_flats"] = FlatFieldManager(
            run=run,
            sk_data_root=config.sk.data_root,
        )
    if sched.lunar_background.collect:
        sources["lunar_background"] = LunarBackgroundManager(run=run)
    if sched.photometric_standards.collect:
        sources["photometric_standards"] = PhotometricStandardsManager(run=run)
    if sched.coverage.collect:
        sources["coverage"] = SkyCoverageManager(run=run)
    if sched.calsats.collect:
        sources["calsats"] = CalibrationSatelliteManager(run=run)

    return sources


def _resolve_config_path() -> Path | list[Path] | None:
    """Resolve burr config path(s) from ``BURR_CONFIG`` env var.

    Accepts a single path or multiple paths separated by ``;`` (or ``:``
    for legacy compatibility on POSIX) for composed site+preset configs.
    Windows paths contain ``:`` after the drive letter, so on Windows we
    only split on ``;``.
    """
    value = os.getenv("BURR_CONFIG")
    if not value:
        return None
    sep = ";" if os.name == "nt" else ":"
    paths = [Path(p) for p in value.split(sep) if p]
    return paths[0] if len(paths) == 1 else paths


@sk.service_entrypoint(version=sk.VERSION)
async def main(service: sk.Service) -> None:
    """SensorKit service entrypoint for burr.

    Boots burr's config + run + scheduler, registers the controller(s),
    adds the burr program to the service, and hands control to the SK
    service runtime. See module docstring for invocation recipes.
    """
    global _ctx

    initialize_engine()

    config = get_or_initialize_config(_resolve_config_path())
    setup_loguru(level=config.logging.level)
    logger.info(f"burr service starting for site {config.site.name}")

    run = get_or_create_run(site=config.site, verbose=True, new_config=config)
    logger.info(f"run: {run.run_id}  ({run.output_directory})")

    sources = build_task_sources(run, config)
    scheduler = build_scheduler(config, run, sources)
    tracker = SkTracker(scheduler=scheduler, run=run)

    _ctx = _ProgramContext(
        run=run,
        config=config,
        controller_entity=config.sk.burr_controller_entity,
        scheduler=scheduler,
        tracker=tracker,
    )

    # Reset any stale SK tasks queued from a previous run (relevant when
    # the SK service lifecycle restarts main() without reloading the
    # Python module).
    _pending_queue.clear()

    # Burr runs as a program only — no controller. The generic controller
    # (sensorkit.std.sensor, named ``controller1`` in services.yaml) owns
    # the mount/camera/dome leases and handles weather safety. Burr emits
    # StandardCollectTask with ``controller_id=config.sk.burr_controller_entity``
    # so the agent dispatches to that controller.

    # Build the program declaration fresh on every main() invocation — the
    # ``DeclaredProgram`` object is stateful (name + impl populated during
    # register), so reusing one across restarts raises DeclarationError.
    # See sensorkit.api.declarative.Service.add and DeclaredEntity.__init__.
    burr_program = sk.declare_program()
    sk.on_attach(burr_program)(_on_attach)
    sk.on_detach(burr_program)(_on_detach)
    sk.on_enable(burr_program)(_on_enable)
    sk.on_disable(burr_program)(_on_disable)
    sk.task_factory(burr_program)(_next_task)
    # Leave the program entity nameless so it inherits the service name
    # (set by the launcher via ``sensorkit service run <name> ...`` or the
    # ``name:`` field in services.yaml). This collapses service + program
    # into a single entity and avoids the EntityLease collision that
    # occurs when service name == program entity name.
    service.add(burr_program)

    logger.info("burr service running")
    try:
        await service.run()
    except BaseException:
        logger.exception("burr service.run() raised — full traceback:")
        raise
