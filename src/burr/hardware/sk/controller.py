"""
SensorKit controllers for Burr observations.

BurrController handles BurrCollectTask by dispatching on target type and
sequencing frames per the exposure map. All slews go through FollowTarget;
per-frame mode switching uses FrameTarget(ICRF) for sidereal and the primary
astro target for rate-follow.

DomeController handles dome open/close via SK's Init/Shutdown tasks.
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone

import sensorkit.api as sk
from loguru import logger
from sensorkit.astro.common import ReferenceFrame
from sensorkit.astro.target import FrameTarget
from sensorkit.common.keyword import KeywordDict
from sensorkit.models.devices import (
    AltAzPointing,
    AxisRates,
    CameraCapture,
    Close,
    Connect,
    Deinit,
    FollowTarget,
    Init,
    Open,
    RADecPointing,
    SitePosition,
    Stop,
)
from sensorkit.std.collect import StandardCollectTask
from sensorkit.std.sensor import Capabilities as SensorCapabilities, SensorDevices

from burr.hardware.sk.tasks import PAYLOAD_KEY, BurrCollectPayload, to_sk_target
from burr.hardware.sk.tracking import SkTracker
from burr.models.site import LightingCondition, SiteMetadata
from burr.models.tracking import TrackingMode


@sk.declare_controller
class BurrController:
    """SensorKit controller that executes burr observations.

    Parameters
    ----------
    mount_entity, camera_entity : str
        Entity names of the mount and camera devices.
    dome_entity : str | None
        Optional dome entity name. When provided, this controller also
        opens the dome during Init and closes it during Shutdown.
    """

    def __init__(
        self,
        mount_entity: str,
        camera_entity: str,
        site: SiteMetadata,
        tracker: SkTracker,
        dome_entity: str | None = None,
    ):
        self.mount_entity = mount_entity
        self.camera_entity = camera_entity
        self.dome_entity = dome_entity
        self.site = site
        self.tracker = tracker
        self.run = tracker.run
        self.mount = None
        self.camera = None
        self.dome = None

    @sk.on_attach
    async def _attach(self):
        controller = sk.controller()
        # Subscribe to the mount's pointing keywords so SensorView (and the
        # agent's context builder) can observe where the telescope is looking.
        self.mount = controller.use_device(
            self.mount_entity,
            subscribe=[AltAzPointing, RADecPointing, AxisRates],
        )
        self.camera = controller.use_device(self.camera_entity)
        if self.dome_entity:
            self.dome = controller.use_device(self.dome_entity)

        # The SK agent's driver blocks on the controller's SitePosition
        # before it will schedule tasks for us.
        await controller.kv_put_model(
            SitePosition(
                latitude_degrees=self.site.latitude,
                longitude_degrees=self.site.longitude,
                altitude_km=self.site.altitude_km,
            )
        )

        # Publish a Capabilities record so SensorView (and other UIs) can
        # discover which devices back this controller — in particular, that
        # the mount entity owns the pointing telemetry.
        await controller.kv_put_model(
            SensorCapabilities(
                tasks=[h.__name__ for h in controller._task_handlers.keys()],
                devices=SensorDevices(
                    mount=self.mount_entity,
                    camera=self.camera_entity,
                    dome=self.dome_entity,
                ),
            )
        )

        logger.info(
            "BurrController attached "
            f"(mount={self.mount_entity}, camera={self.camera_entity}, dome={self.dome_entity})"
        )

    @sk.task_handler
    async def _init(self, task: sk.InitTask):
        # Safety net: the program's offer window should already keep the
        # SK agent from dispatching InitTask during daytime, but if the
        # offer gate is ever bypassed we don't want the dome opening to
        # full sun. Skip dome Open and mount Init during DAY; the agent
        # will retry init when the offer window actually begins.
        condition = self._current_lighting_condition()
        if condition == LightingCondition.DAY:
            logger.warning(
                f"BurrController init skipped (lighting={condition.value}); "
                "waiting for observing window"
            )
            return

        logger.info(f"BurrController init (lighting={condition.value})")
        if self.dome is not None:
            await self.dome.command(Open())
        await self.mount.command(Init())

    def _current_lighting_condition(self) -> LightingCondition:
        now = datetime.now(timezone.utc)
        try:
            return self.run.lighting_schedule.get_lighting_condition(now)
        except Exception as e:
            logger.warning(f"Could not determine lighting condition ({e}); assuming DAY")
            return LightingCondition.DAY

    @sk.task_handler
    async def _shutdown(self, task: sk.ShutdownTask):
        logger.info("BurrController shutdown")
        with contextlib.suppress(Exception):
            await self.mount.command(Stop())
        with contextlib.suppress(Exception):
            await self.mount.command(Deinit())
        if self.dome is not None:
            with contextlib.suppress(Exception):
                await self.dome.command(Close())

    @sk.task_handler
    async def _collect(self, task: StandardCollectTask):
        """Execute a burr collect.

        The task arrives as a canonical ``StandardCollectTask`` (so the agent
        and UIs can inspect ``camera_params``). Burr's richer per-frame
        sequencing plan rides in ``task.context['burr']`` — we drive the
        collect from that, not from ``task.camera_params``.
        """
        payload_data = None
        ctx = task.get_context() if hasattr(task, "get_context") else task.context
        if ctx is not None:
            payload_data = ctx.get(PAYLOAD_KEY)

        if payload_data is None:
            logger.warning(
                f"CollectTask {task.task_id} has no burr payload in context; "
                "ignoring (this task was not generated by burr)"
            )
            return

        # On the wire, unknown KeywordDict values arrive as UnknownKeyword
        # pydantic models. Unwrap back to a plain dict before validating.
        if hasattr(payload_data, "model_dump"):
            payload_data = payload_data.model_dump()

        payload = BurrCollectPayload.model_validate(payload_data)
        logger.info(f"Handling {payload}")

        primary = to_sk_target(payload.target)
        sidereal = FrameTarget(frame=ReferenceFrame.ICRF)

        success = False
        error_msg: str | None = None
        try:
            # Initial slew/follow.
            await self.mount.command(FollowTarget(target=primary))

            current_mode: TrackingMode | None = None

            for exposure_time in payload.exposure_seconds:
                for mode_str in payload.exposure_map:
                    mode = TrackingMode(mode_str)

                    if mode != current_mode:
                        await self._apply_tracking_mode(mode, primary, sidereal)
                        current_mode = mode

                    logger.info(
                        f"Capturing {exposure_time}s frame "
                        f"(mode={mode.value}, source={payload.source_name})"
                    )
                    capture_context = KeywordDict()
                    capture_context["source_name"] = payload.source_name
                    capture_context["tracking_mode"] = mode.value
                    capture_context["task_id"] = str(task.task_id)
                    capture_context.update(payload.metadata)
                    await self.camera.command(
                        CameraCapture(
                            integration_time=exposure_time,
                            context=capture_context,
                        )
                    )
            success = True
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.exception(f"Collect failed ({payload.source_name}): {e}")
            raise
        finally:
            # Stop motion when collection completes or errors out.
            with contextlib.suppress(Exception):
                await self.mount.command(Stop())
            # Close the loop with the scheduler and task source so
            # success counts, pacing, and per-source completion state
            # (coverage pixels, photometric stars, flats taken) are
            # updated and persisted to run_state.json for resume.
            with contextlib.suppress(Exception):
                self.tracker.record_completion(
                    task_id=task.task_id,
                    success=success,
                    condition=self._current_lighting_condition(),
                    error=error_msg,
                )

    @sk.task_handler
    async def _recover(self, task: sk.RecoverTask):
        """Reconnect devices and stop motion after an error."""
        logger.warning("BurrController recover")

        async def _try(coro_factory, label: str):
            try:
                await coro_factory()
            except Exception as e:
                logger.warning(f"recover: {label} failed ({type(e).__name__}: {e})")

        await _try(lambda: self.mount.command(Connect()), "mount.Connect")
        await _try(lambda: self.camera.command(Connect()), "camera.Connect")
        if self.dome is not None:
            await _try(lambda: self.dome.command(Connect()), "dome.Connect")
        await _try(lambda: self.mount.command(Stop()), "mount.Stop")

    async def _apply_tracking_mode(
        self,
        mode: TrackingMode,
        primary,
        sidereal_target: FrameTarget,
    ):
        """Switch mount tracking to the given mode."""
        match mode:
            case TrackingMode.RATE:
                await self.mount.command(FollowTarget(target=primary))
            case TrackingMode.SIDEREAL:
                await self.mount.command(FollowTarget(target=sidereal_target))
            case TrackingMode.OFF:
                await self.mount.command(Stop())


@sk.declare_controller
class DomeController:
    """SensorKit controller for dome open/close."""

    def __init__(self, dome_entity: str):
        self.dome_entity = dome_entity
        self.dome = None

    @sk.on_attach
    async def _attach(self):
        controller = sk.controller()
        self.dome = controller.use_device(self.dome_entity)
        logger.info(f"DomeController attached (dome={self.dome_entity})")

    @sk.task_handler
    async def _init(self, task: sk.InitTask):
        logger.info("Opening dome")
        await self.dome.command(Open())

    @sk.task_handler
    async def _shutdown(self, task: sk.ShutdownTask):
        logger.info("Closing dome")
        await self.dome.command(Close())
