"""
Universal observation request types.

ObservationRequest is the single type returned by all task sources and consumed
by all executors (direct and SK). Target is a discriminated union that executors
pattern-match on.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from burr.models.tracking import TrackingMode, RateSpec


# --- Target types (discriminated union) ---


class Target(BaseModel):
    """Base class for observation targets."""

    type: str


class RADecTarget(Target):
    """Target specified in J2000 RA/Dec coordinates."""

    type: Literal["radec"] = "radec"
    right_ascension_deg: float
    declination_deg: float


class AltAzTarget(Target):
    """Target specified in altitude/azimuth coordinates."""

    type: Literal["altaz"] = "altaz"
    altitude_deg: float
    azimuth_deg: float


class TLETarget(Target):
    """Target specified by TLE orbital elements."""

    type: Literal["tle"] = "tle"
    tle_line1: str
    tle_line2: str


class LunarTarget(AltAzTarget):
    """Target for lunar background measurements."""

    type: Literal["lunar"] = "lunar"
    moon_altitude_deg: float
    moon_azimuth_deg: float
    moon_separation_deg: float
    moon_phase: float


# Discriminated union of all target types
TargetUnion = RADecTarget | AltAzTarget | TLETarget | LunarTarget


# --- Exposure specification ---


class ExposureSpec(BaseModel):
    """Specification for a single frame within an observation."""

    tracking_mode: TrackingMode


# --- ObservationRequest ---


class ObservationRequest(BaseModel):
    """Universal observation request returned by all task sources.

    This replaces the various MixedTrackingTask* types and is consumed
    by both DirectTaskExecutor and BurrController.
    """

    task_id: str
    source_name: str = Field(description="Task source: 'photometry', 'coverage', 'calsats', 'flats', 'lunar'")
    target: TargetUnion = Field(discriminator="type")
    exposure_seconds: list[float] = Field(description="Exposure durations for the task set")
    exposure_map: list[ExposureSpec] = Field(description="Tracking mode per frame within each exposure")
    rates: RateSpec | None = None
    metadata: dict[str, Any] = Field(default_factory=dict, description="Source-specific metadata")

    @property
    def total_exposures(self) -> int:
        """Total number of exposures in the set."""
        return len(self.exposure_seconds)

    @property
    def total_frames_per_exposure(self) -> int:
        """Number of frames per exposure (from the exposure map)."""
        return len(self.exposure_map)

    @classmethod
    def create_exposure_map(cls, n_streaks: int) -> list[ExposureSpec]:
        """Create a standard exposure map: one SIDEREAL followed by n RATE frames."""
        return [ExposureSpec(tracking_mode=TrackingMode.SIDEREAL)] + [
            ExposureSpec(tracking_mode=TrackingMode.RATE) for _ in range(n_streaks)
        ]
