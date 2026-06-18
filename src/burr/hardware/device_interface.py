from abc import ABC, abstractmethod
from typing import Optional, Protocol

from burr.models.tracking import TrackingMode, Rates


class Mount(Protocol):
    """Protocol for mount device control."""

    def connect(self) -> None:
        """Connect to the mount."""
        ...

    def disconnect(self) -> None:
        """Disconnect from the mount."""
        ...

    def park(self) -> None:
        """Park the mount."""
        ...

    def tracking_on(self) -> None:
        """Enable sidereal tracking."""
        ...

    def tracking_off(self) -> None:
        """Disable tracking."""
        ...

    def goto_ra_dec_j2000(self, ra_hours: float, dec_deg: float) -> None:
        """Go to J2000 RA/Dec coordinates."""
        ...

    def goto_alt_az(self, alt_deg: float, az_deg: float) -> None:
        """Go to altitude/azimuth coordinates."""
        ...

    async def monitor_slew(self) -> None:
        """Wait for slew to complete."""
        ...


class Camera(Protocol):
    """Protocol for camera device control."""

    def connect(self) -> None:
        """Connect to the camera."""
        ...

    def disconnect(self) -> None:
        """Disconnect from the camera."""
        ...

    def start_exposure(self, duration: float, light: bool = True) -> None:
        """Start an exposure."""
        ...

    async def monitor_exposure(self) -> None:
        """Wait for exposure to complete."""
        ...

    @property
    def image_ready(self) -> bool:
        """Check if an image is ready to download."""
        ...

    def save_image(self, path: str) -> None:
        """Save the current image to a file."""
        ...


class Dome(Protocol):
    """Protocol for dome device control."""

    def connect(self) -> None:
        """Connect to the dome."""
        ...

    def disconnect(self) -> None:
        """Disconnect from the dome."""
        ...

    def open(self) -> None:
        """Open the dome."""
        ...

    def close(self) -> None:
        """Close the dome."""
        ...

    def is_open(self) -> bool:
        """Check if the dome is open."""
        ...

    def is_closed(self) -> bool:
        """Check if the dome is closed."""
        ...

    def is_moving(self) -> bool:
        """Check if the dome is moving."""
        ...

    def is_error(self) -> bool:
        """Check if the dome is in an error state."""
        ...


class DeviceInterface(ABC):
    """Base interface for direct hardware control implementations."""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to all devices."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from all devices."""
        pass

    @abstractmethod
    async def goto_ra_dec(
        self, ra_deg: float, dec_deg: float, tracking_mode: TrackingMode, rates: Optional[Rates] = None
    ) -> None:
        """Go to RA/Dec coordinates."""
        pass

    @abstractmethod
    async def goto_alt_az(
        self, alt_deg: float, az_deg: float, tracking_mode: TrackingMode, rates: Optional[Rates] = None
    ) -> None:
        """Go to altitude/azimuth coordinates."""
        pass

    @abstractmethod
    async def track_tle(
        self, tle_line1: str, tle_line2: str, tracking_mode: TrackingMode, rates: Optional[Rates] = None
    ) -> None:
        """Track a satellite using TLE elements."""
        pass

    @abstractmethod
    async def take_exposure(self, exposure_seconds: float, file_path: str) -> bool:
        """Take an exposure."""
        pass

    @abstractmethod
    async def open_dome(self) -> None:
        """Open the dome."""
        pass

    @abstractmethod
    async def close_dome(self) -> None:
        """Close the dome."""
        pass

    @abstractmethod
    async def park_mount(self) -> None:
        """Park the mount."""
        pass
