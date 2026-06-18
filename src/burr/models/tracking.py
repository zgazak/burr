"""
Tracking mode and rate definitions for telescope control.

This is the single source of truth for TrackingMode and rate types.
All other modules should import from here.
"""

from enum import Enum

import numpy as np
import random
from pydantic import BaseModel


class TrackingMode(str, Enum):
    """Tracking modes for telescope control."""

    RATE = "rate"  # Track at custom rates
    SIDEREAL = "sidereal"  # Track at sidereal rate
    OFF = "off"  # Tracking disabled


class Rates(BaseModel):
    """Concrete tracking rates for telescope control."""

    right_ascension_rate_arcsec_s: float = 0.0
    declination_rate_arcsec_s: float = 0.0

    def get_rates(self, exposure_seconds: float) -> tuple[float, float]:
        """Get the rates as a tuple."""
        return (
            self.right_ascension_rate_arcsec_s,
            self.declination_rate_arcsec_s,
        )

    def __str__(self):
        return (
            f"Rates(ra={self.right_ascension_rate_arcsec_s:.4f}, "
            f"dec={self.declination_rate_arcsec_s:.4f})"
        )


class RatesFromStreak(BaseModel):
    """Calculate tracking rates to achieve a desired streak length."""

    streak_length_pixels: float
    streak_orientation_deg: float
    plate_scale_arcsec_per_pixel: float = 0.5

    def get_rates(self, exposure_seconds: float) -> tuple[float, float]:
        """Calculate rates to achieve desired streak length over the exposure time."""
        velocity_arcsec_s = (
            self.streak_length_pixels * self.plate_scale_arcsec_per_pixel
        ) / exposure_seconds

        angle_rad = np.radians(self.streak_orientation_deg)
        ra_rate = velocity_arcsec_s * np.cos(angle_rad)
        dec_rate = velocity_arcsec_s * np.sin(angle_rad)

        return (float(ra_rate), float(dec_rate))

    def to_concrete_rates(self, exposure_seconds: float) -> Rates:
        """Convert to concrete Rates for a given exposure time."""
        ra_rate, dec_rate = self.get_rates(exposure_seconds)
        return Rates(
            right_ascension_rate_arcsec_s=ra_rate,
            declination_rate_arcsec_s=dec_rate,
        )

    def __str__(self):
        return (
            f"RatesFromStreak(length={self.streak_length_pixels}px, "
            f"orientation={self.streak_orientation_deg}°)"
        )


class RatesFromStreakDistribution(BaseModel):
    """Calculate tracking rates for a random streak within a distribution."""

    streak_length_pixels_min: float = 5
    streak_length_pixels_max: float = 100
    streak_orientation_min_deg: float = 0
    streak_orientation_max_deg: float = 180
    plate_scale_arcsec_per_pixel: float = 0.5

    def get_rates(self, exposure_seconds: float) -> tuple[float, float]:
        """Calculate rates for a random streak within the distribution."""
        streak_length = random.uniform(
            self.streak_length_pixels_min, self.streak_length_pixels_max
        )
        streak_orientation = random.uniform(
            self.streak_orientation_min_deg, self.streak_orientation_max_deg
        )

        rates = RatesFromStreak(
            streak_length_pixels=streak_length,
            streak_orientation_deg=streak_orientation,
            plate_scale_arcsec_per_pixel=self.plate_scale_arcsec_per_pixel,
        )
        return rates.get_rates(exposure_seconds)

    def to_concrete_rates(self, exposure_seconds: float) -> Rates:
        """Convert to concrete Rates for a given exposure time."""
        ra_rate, dec_rate = self.get_rates(exposure_seconds)
        return Rates(
            right_ascension_rate_arcsec_s=ra_rate,
            declination_rate_arcsec_s=dec_rate,
        )

    def __str__(self):
        return (
            f"RatesFromStreakDistribution(length=[{self.streak_length_pixels_min}, "
            f"{self.streak_length_pixels_max}]px, "
            f"orientation=[{self.streak_orientation_min_deg}, "
            f"{self.streak_orientation_max_deg}]°)"
        )


# Type alias for any rate specification
RateSpec = Rates | RatesFromStreak | RatesFromStreakDistribution
