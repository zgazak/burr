"""
Hardware device factory.

Eliminates the 3x duplication of device initialization in runner.py,
shutdown.py, and SK service.
"""

from dataclasses import dataclass

from loguru import logger

from burr.core.config import AppConfig
from burr.models.ascom import PWI4MountEntity, CameraEntity, DomeEntity, WeatherEntity


@dataclass
class DeviceSet:
    """Collection of hardware devices."""

    mount: PWI4MountEntity
    camera: CameraEntity
    dome: DomeEntity
    weather: WeatherEntity

    @property
    def any_active(self) -> bool:
        return any([self.mount.active, self.camera.active, self.dome.active, self.weather.active])


def create_devices(config: AppConfig) -> DeviceSet:
    """Create hardware device entities from configuration.

    When a device config section is missing or active=False, creates a
    simulated (inactive) device.

    Parameters
    ----------
    config : AppConfig
        Application configuration.

    Returns
    -------
    DeviceSet
        Collection of initialized device entities.
    """
    hw = config.hardware

    if not hw:
        logger.warning("No hardware configuration found, using defaults with simulation mode")
        return DeviceSet(
            mount=PWI4MountEntity(active=False),
            camera=CameraEntity(active=False),
            dome=DomeEntity(active=False),
            weather=WeatherEntity(active=False),
        )

    mount = PWI4MountEntity(
        host=hw.pwi4.host if hw.pwi4 else "127.0.0.1",
        port=hw.pwi4.port if hw.pwi4 else 8220,
        active=hw.pwi4.active if hw.pwi4 else False,
    )

    camera = CameraEntity(
        address=hw.camera.address if hw.camera else "127.0.0.1",
        port=hw.camera.port if hw.camera else 11111,
        device_number=hw.camera.device_number if hw.camera else 0,
        active=hw.camera.active if hw.camera else False,
        default_binning=hw.camera.default_binning if hw.camera else 1,
    )

    dome = DomeEntity(
        address=hw.dome.address if hw.dome else "127.0.0.1",
        port=hw.dome.port if hw.dome else 11111,
        device_number=hw.dome.device_number if hw.dome else 0,
        active=hw.dome.active if hw.dome else False,
    )

    weather = WeatherEntity(
        address=hw.weather.address if hw.weather else "127.0.0.1",
        port=hw.weather.port if hw.weather else 5580,
        device_number=hw.weather.device_number if hw.weather else 0,
        active=hw.weather.active if hw.weather else False,
    )

    active_names = []
    if mount.active:
        active_names.append("mount")
    if camera.active:
        active_names.append("camera")
    if dome.active:
        active_names.append("dome")
    if weather.active:
        active_names.append("weather")

    if active_names:
        logger.info(f"Created devices - active: {', '.join(active_names)}")
    else:
        logger.info("All devices are simulated")

    return DeviceSet(mount=mount, camera=camera, dome=dome, weather=weather)
