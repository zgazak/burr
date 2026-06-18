"""Hardware-related base models."""

from typing import List
from pydantic import BaseModel, Field


class TelescopeConfig(BaseModel):
    """Telescope optical configuration with measured values."""

    ifov_no_binning_arcsec: float = Field(description="Measured pixel scale in arcseconds per pixel")
    field_of_view_degree: float = Field(description="Measured field of view in degrees")


class DomeConfig(BaseModel):
    """ASCOM Dome configuration.

    When active=False, the dome will operate in simulation mode without making actual hardware calls.
    """

    address: str = "127.0.0.1"
    port: int = 11111
    device_number: int = 0
    active: bool = False
    open_windows: List[str] = Field(
        default=["civil_twilight", "nautical_twilight", "astronomical_twilight", "night"],
        description="List of lighting conditions during which the dome should be open",
    )


class CameraConfig(BaseModel):
    """ASCOM Camera configuration.

    When active=False, the camera will operate in simulation mode without making actual hardware calls.
    """

    address: str = "127.0.0.1"
    port: int = 11111
    device_number: int = 0
    active: bool = False
    default_binning: int = 1
    binning: List[int] = Field(default_factory=list, description="Available binning modes")
    readout: List[int | str] = Field(default_factory=list, description="Available readout modes")


class PWI4Config(BaseModel):
    """PWI4 configuration.

    When active=False, the mount will operate in simulation mode without making actual hardware calls.
    """

    host: str = "127.0.0.1"
    port: int = 8220
    active: bool = False


class WeatherDeviceConfig(BaseModel):
    """ASCOM Observing Conditions hardware configuration.

    When active=False, the weather device will operate in simulation mode without making actual hardware calls.

    Note: Renamed from WeatherConfig to avoid collision with core.config.WeatherConfig
    (which holds monitoring thresholds). WeatherConfig alias preserved for YAML backward compat.
    """

    address: str = "127.0.0.1"
    port: int = 5580
    device_number: int = 0
    active: bool = False


# Backward-compatible alias for YAML configs that use 'weather' key with old field names
WeatherConfig = WeatherDeviceConfig


class HardwareConfig(BaseModel):
    """Hardware configuration.

    Each device can be independently set to active or simulation mode using its active flag.
    When a device is not active, it will operate in simulation mode without making actual hardware calls.
    """

    telescope: TelescopeConfig | None = None
    dome: DomeConfig | None = None
    camera: CameraConfig | None = None
    pwi4: PWI4Config | None = None
    weather: WeatherDeviceConfig | None = None

    @property
    def any_active(self) -> bool:
        """Check if any hardware device is active."""
        return any(device and device.active for device in (self.dome, self.camera, self.pwi4, self.weather))

    @property
    def all_active(self) -> bool:
        """Check if all configured hardware devices are active."""
        return all(device.active for device in (self.dome, self.camera, self.pwi4, self.weather) if device is not None)
