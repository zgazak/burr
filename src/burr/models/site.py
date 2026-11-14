"""Site model and utilities."""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
from enum import Enum
from pydantic import Field, model_validator
import numpy as np
from burr.core.constants import TWILIGHT_ANGLES
from typing import Optional, List, Tuple
from zoneinfo import ZoneInfo
from loguru import logger


def _format_time_dual_zone(utc_time: datetime, local_tz, tz_name: str) -> str:
    """
    Format a UTC time to show both UTC and local timezone.

    Parameters
    ----------
    utc_time : datetime
        UTC time (assumed to be timezone-naive)
    local_tz : timezone
        Local timezone object
    tz_name : str
        Local timezone name for display

    Returns
    -------
    str
        Formatted string like "14:45 [UTC] 04:45 [America/Honolulu]"
    """
    import datetime as dt

    # Convert UTC to local
    utc_with_tz = utc_time.replace(tzinfo=dt.timezone.utc)
    local_time = utc_with_tz.astimezone(local_tz)

    return f"{utc_time.strftime('%H:%M')} [UTC] {local_time.strftime('%H:%M')} [{tz_name}]"


# Lazy imports for timezone and astropy to avoid import overhead
def _get_timezone_modules():
    """Lazy import timezone modules to avoid overhead when not needed."""
    try:
        from timezonefinder import TimezoneFinder
        from zoneinfo import ZoneInfo

        return TimezoneFinder, ZoneInfo
    except ImportError:
        try:
            from timezonefinder import TimezoneFinder
            from pytz import timezone as ZoneInfo

            return TimezoneFinder, ZoneInfo
        except ImportError:
            raise ImportError(
                "timezonefinder and zoneinfo/pytz are required for timezone calculations. Install with: pip install timezonefinder"
            )


# Lazy imports for astropy to avoid import overhead
def _get_astropy_modules():
    """Lazy import astropy modules to avoid overhead when not needed."""
    try:
        from astropy.coordinates import EarthLocation, get_sun
        from astropy.coordinates.solar_system import get_body
        from astropy.time import Time
        from astropy import units as u
        from astropy.coordinates import AltAz

        return EarthLocation, get_sun, get_body, Time, u, AltAz
    except ImportError:
        raise ImportError("astropy is required for twilight calculations. Install with: pip install astropy")


class LightingCondition(Enum):
    ASTRONOMICAL_TWILIGHT = "astronomical_twilight"
    CIVIL_TWILIGHT = "civil_twilight"
    NAUTICAL_TWILIGHT = "nautical_twilight"
    DAY = "day"
    NIGHT = "night"


class LightingSchedule(BaseModel):
    """
    Lighting schedule for astronomical observations.

    Covers a 24-hour period centered on local midnight, containing:
    - Two day periods (before evening twilights and after morning twilights)
    - Evening twilights (civil → nautical → astronomical)
    - One night period (darkest time)
    - Morning twilights (astronomical → nautical → civil)
    """

    # Schedule bounds
    schedule_start_utc: datetime = Field(..., description="Start of 24-hour schedule (local midnight - 12h)")
    schedule_end_utc: datetime = Field(..., description="End of 24-hour schedule (local midnight + 12h)")
    local_midnight: datetime = Field(..., description="Local midnight at center of schedule")

    # First day period (before evening twilights)
    day_period_1_start: datetime = Field(..., description="Start of first day period")
    day_period_1_end: datetime = Field(..., description="End of first day period (sunset)")

    # Evening twilights (sun setting)
    evening_civil_start: datetime = Field(..., description="Start of evening civil twilight")
    evening_civil_end: datetime = Field(..., description="End of evening civil twilight")
    evening_nautical_start: datetime = Field(..., description="Start of evening nautical twilight")
    evening_nautical_end: datetime = Field(..., description="End of evening nautical twilight")
    evening_astronomical_start: datetime = Field(..., description="Start of evening astronomical twilight")
    evening_astronomical_end: datetime = Field(..., description="End of evening astronomical twilight")

    # Night period (darkest time)
    night_start: datetime = Field(..., description="Start of night (end of evening astronomical twilight)")
    night_end: datetime = Field(..., description="End of night (start of morning astronomical twilight)")

    # Morning twilights (sun rising)
    morning_astronomical_start: datetime = Field(..., description="Start of morning astronomical twilight")
    morning_astronomical_end: datetime = Field(..., description="End of morning astronomical twilight")
    morning_nautical_start: datetime = Field(..., description="Start of morning nautical twilight")
    morning_nautical_end: datetime = Field(..., description="End of morning nautical twilight")
    morning_civil_start: datetime = Field(..., description="Start of morning civil twilight")
    morning_civil_end: datetime = Field(..., description="End of morning civil twilight")

    # Second day period (after morning twilights)
    day_period_2_start: datetime = Field(..., description="Start of second day period (sunrise)")
    day_period_2_end: datetime = Field(..., description="End of second day period")

    # Lunar information
    moon_phase: float = Field(..., description="Lunar phase from 0 (new) to 1 (full)", ge=0, le=1)
    moon_rise: Optional[datetime] = Field(default=None, description="Moon rise time (if occurs during schedule)")
    moon_set: Optional[datetime] = Field(default=None, description="Moon set time (if occurs during schedule)")
    moon_is_waxing: bool = Field(..., description="True if moon is waxing (growing), False if waning (shrinking)")

    class Config:
        """Pydantic model configuration."""

        json_encoders = {datetime: lambda dt: dt.isoformat()}

    @model_validator(mode="before")
    def ensure_timezone_aware(cls, data):
        """Ensure all datetime fields are timezone-aware with UTC timezone."""
        for field_name, value in data.items():
            if isinstance(value, datetime):
                if value.tzinfo is None:
                    data[field_name] = value.replace(tzinfo=ZoneInfo("UTC"))
                else:
                    data[field_name] = value.astimezone(ZoneInfo("UTC"))
        return data

    @classmethod
    def calculate_twilights(cls, site: "SiteMetadata", date: datetime) -> "LightingSchedule":
        """
        Calculate precise twilight times for a given site and date.

        Creates a 24-hour schedule centered on local midnight (±12 hours).

        Parameters
        ----------
        site : SiteMetadata
            Location information
        date : datetime
            Any time in the observing night to schedule. Nights run local noon to noon, so a
            time before local noon belongs to the night ending that morning and a time from noon
            on to the night starting that evening; a local midnight selects the night centered on
            it. Naive datetimes are taken as site-local.

        Returns
        -------
        LightingSchedule
            Complete lighting schedule with transition times
        """
        try:
            EarthLocation, get_sun, get_body, Time, u, AltAz = _get_astropy_modules()

            # Get local timezone and calculate midnight with smart logic
            local_tz = site.get_timezone()

            current_utc = datetime.now(ZoneInfo("UTC"))
            current_local = current_utc.astimezone(local_tz)

            # Convert input date to local timezone for reference
            if date.tzinfo is None:
                # Assume input is in local time if no timezone specified
                date_local = local_tz.localize(date) if hasattr(local_tz, "localize") else date.replace(tzinfo=local_tz)
            else:
                # Convert to local timezone
                date_local = date.astimezone(local_tz)

            # Determine which observing night `date` falls in. The "observing night" starts at noon
            # and ends at noon the next day, so evening twilights are at the start of the schedule.
            # This must follow `date`, not the wall clock: a run resumed on a later night recomputes
            # its schedule from its own observation date, and has to keep that night's schedule.
            local_noon = date_local.replace(hour=12, minute=0, second=0, microsecond=0)

            if date_local < local_noon:
                # Before noon - the morning after an observing night: the midnight just passed
                local_midnight = date_local.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                # Noon or later - the afternoon/evening before a night: the coming midnight
                local_midnight = date_local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

            # Ensure timezone awareness
            if local_midnight.tzinfo is None:
                local_midnight = (
                    local_tz.localize(local_midnight)
                    if hasattr(local_tz, "localize")
                    else local_midnight.replace(tzinfo=local_tz)
                )

            # Convert to UTC for schedule boundaries
            local_midnight_utc = local_midnight.astimezone(ZoneInfo("UTC"))

            # Schedule bounds: ±12 hours from local midnight
            schedule_start_utc = local_midnight_utc - timedelta(hours=12)
            schedule_end_utc = local_midnight_utc + timedelta(hours=12)

            logger.info(f"Local midnight: {local_midnight}")
            logger.info(f"Schedule start (UTC): {schedule_start_utc}")
            logger.info(f"Schedule end (UTC): {schedule_end_utc}")
            logger.info(f"Current time: {current_utc} ({current_local})")
            logger.info(f"Observation date: {date} ({date_local})")
            logger.info(f"Local noon: {local_noon}")

            # Create location object for astropy
            location = EarthLocation(
                lat=site.latitude * u.deg,
                lon=site.longitude * u.deg,
                height=site.altitude_km * u.km if site.altitude_km else 0 * u.km,
            )

            # Calculate sun positions throughout the schedule period
            time_steps = np.linspace(0, 24, 721)  # Increased from 145 to 721 steps for ~2-minute resolution
            times = []
            for hour in time_steps:
                # Create datetime for each step
                step_time = schedule_start_utc + timedelta(hours=float(hour))
                # Strip timezone for Astropy (it assumes UTC)
                times.append(step_time.replace(tzinfo=None))

            # Create a single Time object for the sequence
            astropy_times = Time(times)

            frame = AltAz(obstime=astropy_times, location=location)
            sun_altaz = get_sun(astropy_times).transform_to(frame)
            sun_alts = sun_altaz.alt.deg

            logger.info(f"Sun altitude range: {sun_alts.min():.1f}° to {sun_alts.max():.1f}°")
            logger.info(f"Sun altitude at schedule start: {sun_alts[0]:.1f}°")
            logger.info(f"Sun altitude at schedule end: {sun_alts[-1]:.1f}°")

            # Debug: log some key times and altitudes
            debug_indices = [0, 180, 360, 540, 720]  # Updated for new resolution: noon, 6pm, midnight, 6am, noon
            for i in debug_indices:
                if i < len(sun_alts):
                    debug_time_utc = astropy_times[i].datetime.replace(tzinfo=ZoneInfo("UTC"))
                    debug_time_local = debug_time_utc.astimezone(site.get_timezone())
                    logger.info(f"Debug: {debug_time_local.strftime('%H:%M %Z')} -> Sun altitude: {sun_alts[i]:.1f}°")

            # Find transition times by detecting crossings
            transitions = {}
            for name, angle in TWILIGHT_ANGLES.items():
                logger.info(f"Looking for {name} transitions at {angle}°")

                # Find all crossings of the threshold
                # Look for sign changes in (sun_alts - angle)
                diff = sun_alts - angle
                sign_changes = np.where(np.diff(np.sign(diff)))[0]

                logger.info(f"  {name}: found {len(sign_changes)} sign changes around {angle}°")

                evening_time = None
                morning_time = None
                all_crossings = []

                for idx in sign_changes:
                    # Use linear interpolation to find more precise crossing time
                    time_before = astropy_times[idx].datetime
                    time_after = astropy_times[idx + 1].datetime
                    alt_before = sun_alts[idx]
                    alt_after = sun_alts[idx + 1]

                    # Linear interpolation to find exact crossing time
                    # When sun_alt crosses 'angle': solve for t where alt_before + t*(alt_after - alt_before) = angle
                    if alt_after != alt_before:  # Avoid division by zero
                        t = (angle - alt_before) / (alt_after - alt_before)
                        # Clamp t to [0, 1] range
                        t = max(0, min(1, t))
                    else:
                        t = 0.5  # If altitudes are equal, use midpoint

                    # Interpolate the crossing time
                    time_delta = (time_after - time_before) * t
                    crossing_time = time_before + time_delta

                    crossing_time_utc = crossing_time.replace(tzinfo=ZoneInfo("UTC"))
                    crossing_time_local = crossing_time_utc.astimezone(site.get_timezone())

                    if diff[idx] > 0 and diff[idx + 1] < 0:
                        all_crossings.append(("down", idx, crossing_time_utc, crossing_time_local))
                        logger.info(
                            f"  Downward crossing at index {idx}, time {crossing_time_utc} ({crossing_time_local.strftime('%H:%M %Z')})"
                        )
                        logger.info(f"    Sun altitude: {sun_alts[idx]:.1f}° -> {sun_alts[idx + 1]:.1f}°")
                    elif diff[idx] < 0 and diff[idx + 1] > 0:
                        all_crossings.append(("up", idx, crossing_time_utc, crossing_time_local))
                        logger.info(
                            f"  Upward crossing at index {idx}, time {crossing_time_utc} ({crossing_time_local.strftime('%H:%M %Z')})"
                        )
                        logger.info(f"    Sun altitude: {sun_alts[idx]:.1f}° -> {sun_alts[idx + 1]:.1f}°")

                # Pick evening and morning crossings based on order in 24-hour cycle
                # Since schedule runs noon-to-noon, first crossing = evening, second = morning
                down_crossings = [c for c in all_crossings if c[0] == "down"]
                up_crossings = [c for c in all_crossings if c[0] == "up"]

                if len(down_crossings) >= 1:
                    evening_time = down_crossings[0][2]
                    logger.info(f"  -> Selected as evening {name}: {down_crossings[0][3].strftime('%H:%M %Z')}")

                if len(up_crossings) >= 1:
                    morning_time = up_crossings[0][2]
                    logger.info(f"  -> Selected as morning {name}: {up_crossings[0][3].strftime('%H:%M %Z')}")

                # Set transitions or use fallbacks
                if evening_time:
                    transitions[f"evening_{name.lower()}"] = evening_time
                else:
                    logger.info(f"  Evening {name}: no downward crossing found, using schedule start")
                    transitions[f"evening_{name.lower()}"] = schedule_start_utc

                if morning_time:
                    transitions[f"morning_{name.lower()}"] = morning_time
                else:
                    logger.info(f"  Morning {name}: no upward crossing found, using schedule end")
                    transitions[f"morning_{name.lower()}"] = schedule_end_utc

            # Night transitions are the same as astronomical twilight transitions
            transitions["evening_night"] = transitions["evening_astronomical"]
            transitions["morning_night"] = transitions["morning_astronomical"]

            # Log all transitions for debugging
            logger.info(f"Calculated lighting transitions for {site.name}:")
            for key, time in sorted(transitions.items()):
                local_time = time.astimezone(site.get_timezone())
                logger.info(f"  {key}: {local_time.strftime('%H:%M %Z')} ({time.strftime('%H:%M UTC')})")

            # Calculate lunar information
            moon_phase, moon_rise, moon_set, moon_is_waxing = _calculate_lunar_info(
                location, local_midnight_utc, schedule_start_utc, schedule_end_utc
            )

            return LightingSchedule(
                # Schedule metadata
                schedule_start_utc=schedule_start_utc,
                schedule_end_utc=schedule_end_utc,
                local_midnight=local_midnight_utc,
                # First day period (from schedule start until sunset)
                day_period_1_start=schedule_start_utc,
                day_period_1_end=transitions["evening_day"],  # End of day at sunset
                # Evening twilights (sun setting)
                evening_civil_start=transitions["evening_day"],  # Civil starts at sunset (0°)
                evening_civil_end=transitions["evening_civil"],  # Civil ends at -6°
                evening_nautical_start=transitions["evening_civil"],  # Nautical starts when civil ends
                evening_nautical_end=transitions["evening_nautical"],  # Nautical ends at -12°
                evening_astronomical_start=transitions["evening_nautical"],  # Astronomical starts when nautical ends
                evening_astronomical_end=transitions["evening_astronomical"],  # Astronomical ends at -18°
                # Night period (darkest time)
                night_start=transitions["evening_astronomical"],  # Night starts when astronomical ends
                night_end=transitions["morning_astronomical"],  # Night ends when morning astronomical starts
                # Morning twilights (sun rising)
                morning_astronomical_start=transitions["morning_astronomical"],  # Morning astronomical starts at -18°
                morning_astronomical_end=transitions["morning_nautical"],  # Morning astronomical ends at -12°
                morning_nautical_start=transitions["morning_nautical"],  # Morning nautical starts at -12°
                morning_nautical_end=transitions["morning_civil"],  # Morning nautical ends at -6°
                morning_civil_start=transitions["morning_civil"],  # Morning civil starts at -6°
                morning_civil_end=transitions["morning_day"],  # Morning civil ends at sunrise (0°)
                # Second day period (from sunrise until schedule end)
                day_period_2_start=transitions["morning_day"],  # Second day starts at sunrise
                day_period_2_end=schedule_end_utc,
                # Lunar information
                moon_phase=moon_phase,
                moon_rise=moon_rise,
                moon_set=moon_set,
                moon_is_waxing=moon_is_waxing,
            )

        except Exception as e:
            logger.error(f"Failed to calculate precise schedule, using fallback: {str(e)}")
            import traceback

            logger.error(f"Exception traceback: {traceback.format_exc()}")
            return _create_approximate_schedule(site, date)

    def to_dict(self, site: "SiteMetadata" = None) -> dict[str, str]:
        """
        Convert to dictionary with ISO format datetime strings.

        Parameters
        ----------
        site : SiteMetadata, optional
            Site information to include timezone. If not provided, timezone won't be included.

        Returns
        -------
        dict: Dictionary with datetime fields as ISO format strings
        """
        result = {
            "schedule_start_utc": self.schedule_start_utc.isoformat(),
            "schedule_end_utc": self.schedule_end_utc.isoformat(),
            "local_midnight": self.local_midnight.isoformat(),
            "day_period_1_start": self.day_period_1_start.isoformat(),
            "day_period_1_end": self.day_period_1_end.isoformat(),
            "evening_civil_start": self.evening_civil_start.isoformat(),
            "evening_civil_end": self.evening_civil_end.isoformat(),
            "evening_nautical_start": self.evening_nautical_start.isoformat(),
            "evening_nautical_end": self.evening_nautical_end.isoformat(),
            "evening_astronomical_start": self.evening_astronomical_start.isoformat(),
            "evening_astronomical_end": self.evening_astronomical_end.isoformat(),
            "night_start": self.night_start.isoformat(),
            "night_end": self.night_end.isoformat(),
            "morning_astronomical_start": self.morning_astronomical_start.isoformat(),
            "morning_astronomical_end": self.morning_astronomical_end.isoformat(),
            "morning_nautical_start": self.morning_nautical_start.isoformat(),
            "morning_nautical_end": self.morning_nautical_end.isoformat(),
            "morning_civil_start": self.morning_civil_start.isoformat(),
            "morning_civil_end": self.morning_civil_end.isoformat(),
            "day_period_2_start": self.day_period_2_start.isoformat(),
            "day_period_2_end": self.day_period_2_end.isoformat(),
            "moon_phase": self.moon_phase,
            "moon_rise": self.moon_rise.isoformat() if self.moon_rise else None,
            "moon_set": self.moon_set.isoformat() if self.moon_set else None,
        }

        if site:
            result["local_timezone"] = site.local_timezone

        return result

    def plot_elevation(self, site: "SiteMetadata", figsize=(12, 6), save_path=None):
        """
        Plot sun elevation vs time with lighting condition shading.

        Parameters
        ----------
        site : SiteMetadata
            Site information for calculating sun positions and timezone
        figsize : tuple, optional
            Figure size (width, height) in inches
        save_path : str or Path, optional
            Path to save the plot. If None, shows the plot.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import numpy as np

            # Configure matplotlib for better readability
            plt.rcParams.update(
                {
                    "font.size": 14,  # Base font size
                    "axes.titlesize": 16,  # Title font size
                    "axes.labelsize": 14,  # Axis label font size
                    "xtick.labelsize": 14,  # X-axis tick label font size
                    "ytick.labelsize": 14,  # Y-axis tick label font size
                    "legend.fontsize": 14,  # Legend font size
                    "figure.titlesize": 16,  # Figure title font size
                    "figure.labelsize": 14,  # Figure label font size
                    "lines.linewidth": 1.5,  # Line width
                    "axes.linewidth": 0.8,  # Axis line width
                    "grid.linewidth": 0.5,  # Grid line width
                    "grid.alpha": 0.3,  # Grid transparency
                }
            )
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        try:
            EarthLocation, get_sun, get_body, Time, u, AltAz = _get_astropy_modules()

            # Create location object
            location = EarthLocation(
                lat=site.latitude * u.deg, lon=site.longitude * u.deg, height=(site.altitude_km or 0) * u.km
            )

            # Calculate local midnight bounds for plotting (12 hours each side of local midnight)
            local_tz = site.get_timezone()
            import datetime as dt

            # Convert local midnight to local timezone
            local_midnight_tz = self.local_midnight.astimezone(local_tz)

            # Calculate noon-to-noon bounds in local time
            local_start = local_midnight_tz.replace(hour=12, minute=0, second=0, microsecond=0) - timedelta(days=1)
            local_end = local_midnight_tz.replace(hour=12, minute=0, second=0, microsecond=0)

            # Convert to UTC for plotting
            plot_start = local_start.astimezone(ZoneInfo("UTC"))
            plot_end = local_end.astimezone(ZoneInfo("UTC"))

            # Create time array for x-axis
            time_steps = np.linspace(0, (plot_end - plot_start).total_seconds(), 100)
            times = [plot_start + timedelta(seconds=t) for t in time_steps]

            # Convert to Astropy Time objects (strip timezone as Astropy assumes UTC)
            astropy_times = Time([t.replace(tzinfo=None).isoformat() for t in times])

            # Set up plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_ylim(-10, 90)
            ax.set_xlabel("Time (UTC)")
            ax.set_ylabel("Elevation (degrees)")
            ax.grid(True, alpha=0.3)

            # Calculate sun elevations
            sun_coords = get_sun(astropy_times)
            sun_altaz = sun_coords.transform_to(AltAz(obstime=astropy_times, location=location))
            sun_elevations = sun_altaz.alt.deg

            # Calculate moon elevations
            moon_coords = get_body("moon", astropy_times)
            moon_altaz = moon_coords.transform_to(AltAz(obstime=astropy_times, location=location))
            moon_elevations = moon_altaz.alt.deg

            # Plot sun and moon elevations
            ax.plot(times, sun_elevations, "orange", linewidth=2, label="Sun Elevation")
            ax.plot(times, moon_elevations, "lightblue", linewidth=2, label="Moon Elevation")

            # Set exact x-axis limits
            ax.set_xlim(plot_start, plot_end)

            # Add horizontal lines for twilight boundaries
            for name, angle in TWILIGHT_ANGLES.items():
                ax.axhline(y=angle, color="gray", linestyle="--", alpha=0.5)
                ax.text(times[5], angle + 0.5, name, fontsize=8, alpha=0.7)

            # Define periods with distinct colors in chronological order
            # The sequence follows the sun's actual progression through the 24-hour period
            periods = [
                # First day period (from schedule start until evening twilights)
                (self.day_period_1_start, self.day_period_1_end, "Day", "#FFDD88", 0.4),
                # Evening twilights (sun setting)
                (self.evening_civil_start, self.evening_civil_end, "Civil Twilight", "#0055AA", 0.4),
                (self.evening_nautical_start, self.evening_nautical_end, "Nautical Twilight", "#003377", 0.4),
                (
                    self.evening_astronomical_start,
                    self.evening_astronomical_end,
                    "Astronomical Twilight",
                    "#001155",
                    0.4,
                ),
                # Night period (darkest time)
                (self.night_start, self.night_end, "Night", "#000033", 0.4),
                # Morning twilights (sun rising)
                (
                    self.morning_astronomical_start,
                    self.morning_astronomical_end,
                    "Astronomical Twilight",
                    "#001155",
                    0.4,
                ),
                (self.morning_nautical_start, self.morning_nautical_end, "Nautical Twilight", "#003377", 0.4),
                (self.morning_civil_start, self.morning_civil_end, "Civil Twilight", "#0055AA", 0.4),
                # Second day period (from morning twilights until schedule end)
                (self.day_period_2_start, self.day_period_2_end, "Day", "#FFDD88", 0.4),
            ]

            # Track unique labels for legend
            added_labels = set()

            # Add period shading
            for period_start, period_end, label, color, alpha in periods:
                if period_start and period_end and period_start < period_end:
                    # Ensure times are timezone-aware
                    if period_start.tzinfo is None:
                        period_start = period_start.replace(tzinfo=ZoneInfo("UTC"))
                    if period_end.tzinfo is None:
                        period_end = period_end.replace(tzinfo=ZoneInfo("UTC"))

                    # Only shade if period overlaps with plot range
                    if period_end > plot_start and period_start < plot_end:
                        # Clip times to plot range
                        shade_start = max(period_start, plot_start)
                        shade_end = min(period_end, plot_end)

                        # Only add label to legend once per unique label
                        legend_label = label if label not in added_labels else ""
                        if legend_label:
                            added_labels.add(label)

                        # Add the shading
                        ax.axvspan(shade_start, shade_end, color=color, alpha=alpha, label=legend_label, zorder=1)

            # Format x-axis for UTC
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))

            # Add local time axis on top
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())

            # Create local time ticks every 2 hours
            local_tick_times = []
            local_tick_labels = []

            # Start from noon on first day and go to noon next day
            current_time = local_start
            while current_time <= local_end:
                utc_time = current_time.astimezone(ZoneInfo("UTC"))
                if current_time.hour % 2 == 0:  # Every even hour
                    local_tick_times.append(utc_time)
                    local_tick_labels.append(current_time.strftime("%H:%M"))
                current_time += timedelta(hours=1)

            ax2.set_xticks(local_tick_times)
            ax2.set_xticklabels(local_tick_labels)
            ax2.set_xlabel(f"Local Time ({site.local_timezone})")

            # Format date range for title
            if local_start.date() == local_end.date():
                local_date = local_start.strftime("%B") + " " + str(local_start.day) + " " + local_start.strftime("%Y")
            elif local_start.month == local_end.month:
                local_date = (
                    f"{local_start.strftime('%B')} {local_start.day} - {local_end.day}, {local_end.strftime('%Y')}"
                )
            else:
                local_date = f"{local_start.strftime('%B')} {local_start.day} - {local_end.strftime('%B')} {local_end.day}, {local_end.strftime('%Y')}"

            # Add title and legend
            site_name = site.name or f"Site ({site.latitude:.3f}, {site.longitude:.3f})"
            # Show date range since plot spans two calendar days (midnight ±12h)
            local_midnight_tz = self.local_midnight.replace(tzinfo=dt.timezone.utc).astimezone(local_tz)
            start_date_local = local_midnight_tz - timedelta(hours=12)
            end_date_local = local_midnight_tz + timedelta(hours=12)

            # Format date range
            if start_date_local.date() == end_date_local.date():
                # Same day (shouldn't happen with ±12h, but just in case)
                local_date = (
                    start_date_local.strftime("%B")
                    + " "
                    + str(start_date_local.day)
                    + " "
                    + start_date_local.strftime("%Y")
                )
            elif start_date_local.month == end_date_local.month:
                # Same month: "June 10 - 11, 2025"
                # Add one day to start_date_local to get the correct range
                plot_start_date = start_date_local + timedelta(days=1)
                plot_end_date = end_date_local + timedelta(days=1)
                local_date = f"{plot_start_date.strftime('%B')} {plot_start_date.day} - {plot_end_date.day}, {plot_end_date.strftime('%Y')}"
            else:
                # Different months: "May 31 - June 1, 2025"
                plot_start_date = start_date_local + timedelta(days=1)
                plot_end_date = end_date_local + timedelta(days=1)
                local_date = f"{plot_start_date.strftime('%B')} {plot_start_date.day} - {plot_end_date.strftime('%B')} {plot_end_date.day}, {plot_end_date.strftime('%Y')}"

            # Format moon phase description
            phase_pct = self.moon_phase * 100
            waxing = getattr(self, "moon_is_waxing", True)
            if self.moon_phase < 0.05:
                phase_desc = "New Moon"
            elif self.moon_phase > 0.95:
                phase_desc = "Full Moon"
            elif self.moon_phase < 0.25:
                phase_desc = ("Waxing Crescent Moon" if waxing else "Waning Crescent Moon") + f" ({phase_pct:.0f}%)"
            elif self.moon_phase < 0.45:
                phase_desc = ("First Quarter Moon" if waxing else "Last Quarter Moon") + f" ({phase_pct:.0f}%)"
            elif self.moon_phase < 0.75:
                phase_desc = ("Waxing Gibbous Moon" if waxing else "Waning Gibbous Moon") + f" ({phase_pct:.0f}%)"
            else:
                phase_desc = ("Waning Gibbous Moon" if not waxing else "Waxing Gibbous Moon") + f" ({phase_pct:.0f}%)"

            fig.suptitle(f"{site_name} - {local_date}, {phase_desc}", fontsize=14)
            # Create legend
            ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.99), fontsize=10)

            # Set y-axis limits to show twilight angles clearly
            y_min = -20
            y_max = 90
            ax.set_ylim(y_min, y_max)

            # Adjust layout
            plt.tight_layout()

            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Plot saved to: {save_path}")
            else:
                plt.show()

            return fig

        except Exception as e:
            logger.error(f"Error creating elevation plot: {e}")
            import traceback

            traceback.print_exc()
            return None

    def get_lighting_condition(self, time: datetime) -> LightingCondition:
        """
        Get the lighting condition at a specific time.

        Parameters
        ----------
        time : datetime
            Time to check

        Returns
        -------
        LightingCondition
            Current lighting condition

        Raises
        ------
        ValueError
            If time is outside schedule bounds
        """
        # Check if time is within schedule bounds
        if time < self.schedule_start_utc or time > self.schedule_end_utc:
            raise ValueError(
                f"Time {time} is outside schedule bounds (schedule: {self.schedule_start_utc} to {self.schedule_end_utc})"
            )

        # Ensure time is timezone-aware
        if time.tzinfo is None:
            time = time.replace(tzinfo=ZoneInfo("UTC"))

        # Check each period in chronological order
        if self.day_period_1_start <= time <= self.day_period_1_end:
            return LightingCondition.DAY
        elif self.evening_civil_start <= time <= self.evening_civil_end:
            return LightingCondition.CIVIL_TWILIGHT
        elif self.evening_nautical_start <= time <= self.evening_nautical_end:
            return LightingCondition.NAUTICAL_TWILIGHT
        elif self.evening_astronomical_start <= time <= self.evening_astronomical_end:
            return LightingCondition.ASTRONOMICAL_TWILIGHT
        elif self.night_start <= time <= self.night_end:
            return LightingCondition.NIGHT
        elif self.morning_astronomical_start <= time <= self.morning_astronomical_end:
            return LightingCondition.ASTRONOMICAL_TWILIGHT
        elif self.morning_nautical_start <= time <= self.morning_nautical_end:
            return LightingCondition.NAUTICAL_TWILIGHT
        elif self.morning_civil_start <= time <= self.morning_civil_end:
            return LightingCondition.CIVIL_TWILIGHT
        elif self.day_period_2_start <= time <= self.day_period_2_end:
            return LightingCondition.DAY
        else:
            # This should never happen if schedule is properly constructed
            raise ValueError(f"Time {time} not found in any lighting period")

    def get_condition_boundaries(self, condition: LightingCondition) -> List[Tuple[datetime, datetime]]:
        """
        Get all time boundaries for a specific lighting condition.

        Parameters
        ----------
        condition : LightingCondition
            The lighting condition to find boundaries for

        Returns
        -------
        List[Tuple[datetime, datetime]]
            List of (start, end) time tuples for all occurrences of the condition
        """
        boundaries = []

        if condition == LightingCondition.DAY:
            if self.day_period_1_start <= self.day_period_1_end:
                boundaries.append((self.day_period_1_start, self.day_period_1_end))
            if self.day_period_2_start <= self.day_period_2_end:
                boundaries.append((self.day_period_2_start, self.day_period_2_end))
        elif condition == LightingCondition.CIVIL_TWILIGHT:
            if self.evening_civil_start <= self.evening_civil_end:
                boundaries.append((self.evening_civil_start, self.evening_civil_end))
            if self.morning_civil_start <= self.morning_civil_end:
                boundaries.append((self.morning_civil_start, self.morning_civil_end))
        elif condition == LightingCondition.NAUTICAL_TWILIGHT:
            if self.evening_nautical_start <= self.evening_nautical_end:
                boundaries.append((self.evening_nautical_start, self.evening_nautical_end))
            if self.morning_nautical_start <= self.morning_nautical_end:
                boundaries.append((self.morning_nautical_start, self.morning_nautical_end))
        elif condition == LightingCondition.ASTRONOMICAL_TWILIGHT:
            if self.evening_astronomical_start <= self.evening_astronomical_end:
                boundaries.append((self.evening_astronomical_start, self.evening_astronomical_end))
            if self.morning_astronomical_start <= self.morning_astronomical_end:
                boundaries.append((self.morning_astronomical_start, self.morning_astronomical_end))
        elif condition == LightingCondition.NIGHT:
            if self.night_start <= self.night_end:
                boundaries.append((self.night_start, self.night_end))

        return boundaries

    def merged_windows_for(
        self, conditions
    ) -> List[Tuple[datetime, datetime]]:
        """Return sorted, non-overlapping (start, end) windows covering the
        union of the given lighting conditions on this schedule.

        Adjacent or overlapping intervals are merged so a caller asking for
        ``{CIVIL, NAUTICAL, ASTRONOMICAL, NIGHT}`` gets one continuous
        evening-to-morning window instead of six fragments. The merge is
        keyed on ``end == next.start`` since :meth:`get_condition_boundaries`
        emits boundaries whose endpoints coincide between adjacent periods.

        Parameters
        ----------
        conditions : Iterable[LightingCondition]
            Conditions whose boundaries to union.

        Returns
        -------
        List[Tuple[datetime, datetime]]
            Merged windows in chronological order.
        """
        raw: List[Tuple[datetime, datetime]] = []
        for cond in set(conditions):
            raw.extend(self.get_condition_boundaries(cond))
        if not raw:
            return []

        raw.sort(key=lambda iv: iv[0])
        merged: List[Tuple[datetime, datetime]] = [raw[0]]
        for start, end in raw[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged

    def get_next_transition(self, time: datetime) -> Optional[Tuple[datetime, LightingCondition]]:
        """
        Get the next lighting condition transition time after the given time.

        Parameters
        ----------
        time : datetime
            The time to find the next transition after

        Returns
        -------
        Optional[Tuple[datetime, LightingCondition]]
            A tuple of (transition_time, next_condition), or None if no transitions remain in the schedule
        """
        # Ensure time is timezone-aware
        if time.tzinfo is None:
            time = time.replace(tzinfo=ZoneInfo("UTC"))

        # Check if time is within schedule bounds
        if time < self.schedule_start_utc or time > self.schedule_end_utc:
            return None

        # Define all transitions with their resulting conditions
        transitions = [
            (self.day_period_1_start, LightingCondition.DAY),
            (self.day_period_1_end, LightingCondition.CIVIL_TWILIGHT),  # End of day -> start of civil
            (self.evening_civil_start, LightingCondition.CIVIL_TWILIGHT),  # Start of civil twilight
            (self.evening_civil_end, LightingCondition.NAUTICAL_TWILIGHT),  # End of civil -> start of nautical
            (self.evening_nautical_start, LightingCondition.NAUTICAL_TWILIGHT),  # Start of nautical twilight
            (
                self.evening_nautical_end,
                LightingCondition.ASTRONOMICAL_TWILIGHT,
            ),  # End of nautical -> start of astronomical
            (
                self.evening_astronomical_start,
                LightingCondition.ASTRONOMICAL_TWILIGHT,
            ),  # Start of astronomical twilight
            (self.evening_astronomical_end, LightingCondition.NIGHT),  # End of astronomical -> start of night
            (self.night_start, LightingCondition.NIGHT),  # Start of night
            (self.night_end, LightingCondition.ASTRONOMICAL_TWILIGHT),  # End of night -> start of morning astronomical
            (self.morning_astronomical_start, LightingCondition.ASTRONOMICAL_TWILIGHT),  # Start of morning astronomical
            (
                self.morning_astronomical_end,
                LightingCondition.NAUTICAL_TWILIGHT,
            ),  # End of astronomical -> start of morning nautical
            (self.morning_nautical_start, LightingCondition.NAUTICAL_TWILIGHT),  # Start of morning nautical
            (self.morning_nautical_end, LightingCondition.CIVIL_TWILIGHT),  # End of nautical -> start of morning civil
            (self.morning_civil_start, LightingCondition.CIVIL_TWILIGHT),  # Start of morning civil
            (self.morning_civil_end, LightingCondition.DAY),  # End of civil -> start of day 2
            (self.day_period_2_start, LightingCondition.DAY),  # Start of second day period
            (self.day_period_2_end, None),  # End of schedule
        ]

        # Ensure all transitions are timezone-aware
        transitions = [(t.replace(tzinfo=ZoneInfo("UTC")) if t.tzinfo is None else t, c) for t, c in transitions]

        # Sort transitions by time
        transitions.sort(key=lambda x: x[0])

        # Find the next transition after the given time
        for transition_time, next_condition in transitions:
            if transition_time > time:
                if next_condition is not None:
                    return transition_time, next_condition
                else:
                    return None

        return None


def _estimate_sunrise_sunset(latitude, date):
    """
    Rough estimate of sunrise/sunset times for initial approximation.

    Parameters
    ----------
    latitude : float
        Latitude in degrees
    date : datetime
        Date for calculation

    Returns
    -------
    tuple[datetime, datetime]
        Approximate sunrise and sunset times
    """
    # Simple approximation - actual times will be refined by astropy
    # Assumes 6 AM sunrise, 6 PM sunset, adjusted by season and latitude

    # Day of year for seasonal variation
    day_of_year = date.timetuple().tm_yday

    # Seasonal adjustment (very rough)
    season_offset = 2 * np.sin(2 * np.pi * (day_of_year - 81) / 365.25)  # Peak at summer solstice

    # Latitude adjustment (rough approximation)
    lat_factor = abs(latitude) / 90.0

    sunrise_hour = 6 - season_offset - lat_factor * 2
    sunset_hour = 18 + season_offset + lat_factor * 2

    sunrise = date.replace(hour=int(sunrise_hour), minute=int((sunrise_hour % 1) * 60), second=0, microsecond=0)
    sunset = date.replace(hour=int(sunset_hour), minute=int((sunset_hour % 1) * 60), second=0, microsecond=0)

    return sunrise, sunset


def _calculate_lunar_info(location, local_midnight_utc, schedule_start_utc, schedule_end_utc):
    """
    Calculate lunar phase and rise/set times.

    Parameters
    ----------
    location : EarthLocation
        Observer location
    local_midnight_utc : datetime
        Local midnight in UTC
    schedule_start_utc : datetime
        Start of schedule in UTC
    schedule_end_utc : datetime
        End of schedule in UTC

    Returns
    -------
    tuple[float, datetime|None, datetime|None]
        Moon phase (0-1), moon rise time (if any), moon set time (if any)
    """
    try:
        EarthLocation, get_sun, get_body, Time, u, AltAz = _get_astropy_modules()

        # Calculate moon phase at local midnight
        # Strip timezone for Astropy (it assumes UTC)
        midnight_time = Time(local_midnight_utc.replace(tzinfo=None).isoformat())
        moon_coords = get_body("moon", midnight_time)
        sun_coords = get_sun(midnight_time)

        # Calculate phase from sun-moon angle
        # Phase = 0.5 * (1 - cos(angle_between_sun_and_moon))
        # 0° separation (same direction) = new moon (phase 0)
        # 180° separation (opposite) = full moon (phase 1)
        import numpy as np

        sun_moon_angle = sun_coords.separation(moon_coords).deg
        moon_phase = 0.5 * (1 - np.cos(np.radians(sun_moon_angle)))

        # Determine if moon is waxing or waning by comparing phases at different times
        # Check phase 6 hours later to see if it's increasing (waxing) or decreasing (waning)
        future_time = Time((local_midnight_utc + timedelta(hours=6)).replace(tzinfo=None).isoformat())
        future_moon_coords = get_body("moon", future_time)
        future_sun_coords = get_sun(future_time)
        future_sun_moon_angle = future_sun_coords.separation(future_moon_coords).deg
        future_moon_phase = 0.5 * (1 - np.cos(np.radians(future_sun_moon_angle)))

        # If future phase > current phase, moon is waxing (growing)
        # If future phase < current phase, moon is waning (shrinking)
        is_waxing = future_moon_phase > moon_phase

        # Find moon rise and set times within schedule
        moon_rise, moon_set = None, None

        # Search for moon rise/set by checking elevation crossings
        search_time = schedule_start_utc
        step = timedelta(minutes=30)

        while search_time < schedule_end_utc:
            # Check if moon elevation crosses 0° in next 30 minutes
            next_time = search_time + step

            # Calculate elevations at start and end of window
            # Strip timezone for Astropy
            start_astropy_time = Time(search_time.replace(tzinfo=None).isoformat())
            end_astropy_time = Time(next_time.replace(tzinfo=None).isoformat())

            moon_coords_start = get_body("moon", start_astropy_time)
            moon_coords_end = get_body("moon", end_astropy_time)

            moon_altaz_start = moon_coords_start.transform_to(AltAz(obstime=start_astropy_time, location=location))
            moon_altaz_end = moon_coords_end.transform_to(AltAz(obstime=end_astropy_time, location=location))

            start_elevation = moon_altaz_start.alt.deg
            end_elevation = moon_altaz_end.alt.deg

            # Check if elevation crosses 0° (either direction)
            if (start_elevation <= 0 <= end_elevation) or (end_elevation <= 0 <= start_elevation):
                # Find precise crossing time using binary search
                crossing_time = _find_transition_time(
                    location, search_time + timedelta(minutes=15), 0, start_elevation < end_elevation
                )
                if crossing_time:
                    # Ensure timezone awareness
                    if crossing_time.tzinfo is None:
                        crossing_time = crossing_time.replace(tzinfo=ZoneInfo("UTC"))
                    if start_elevation < end_elevation:  # Rising
                        moon_rise = crossing_time
                    else:  # Setting
                        moon_set = crossing_time

            search_time = next_time

        return moon_phase, moon_rise, moon_set, is_waxing

    except Exception as e:
        logger.warning(f"Warning: Lunar calculation failed ({str(e)}), using defaults")
        return 0.5, None, None, True  # Default to waxing for fallback


def _find_transition_time(location, approximate_time, target_elevation, is_morning):
    """
    Find precise transition time using binary search.

    Parameters
    ----------
    location : EarthLocation
        Observer location
    approximate_time : datetime
        Approximate time of transition
    target_elevation : float
        Target sun elevation angle in degrees
    is_morning : bool
        True for morning transitions, False for evening

    Returns
    -------
    datetime
        Precise transition time
    """
    try:
        EarthLocation, get_sun, Time, u, AltAz = _get_astropy_modules()

        # Search window: 3 hours before/after approximate time
        if is_morning:
            start_time = approximate_time - timedelta(hours=3)
            end_time = approximate_time + timedelta(hours=1)
        else:
            start_time = approximate_time - timedelta(hours=1)
            end_time = approximate_time + timedelta(hours=3)

        # Binary search for precise time
        tolerance = 60  # 1 minute tolerance

        while (end_time - start_time).total_seconds() > tolerance:
            mid_time = start_time + (end_time - start_time) / 2

            # Calculate sun elevation at mid time
            astropy_time = Time(mid_time.isoformat())
            sun_coords = get_sun(astropy_time)
            sun_altaz = sun_coords.transform_to(AltAz(obstime=astropy_time, location=location))
            current_elevation = sun_altaz.alt.deg

            if is_morning:
                # For morning, we want elevation increasing through target
                if current_elevation < target_elevation:
                    start_time = mid_time
                else:
                    end_time = mid_time
            else:
                # For evening, we want elevation decreasing through target
                if current_elevation > target_elevation:
                    start_time = mid_time
                else:
                    end_time = mid_time

        return start_time + (end_time - start_time) / 2

    except Exception:
        # Fallback if binary search fails
        return approximate_time


def _create_approximate_schedule(site, date):
    """
    Create approximate twilight schedule when astropy calculations fail.

    Parameters
    ----------
    site : SiteMetadata
        Site information
    date : datetime
        Date for calculation (assumed local)

    Returns
    -------
    LightingSchedule
        Approximate schedule with UTC times
    """
    logger.info(f"Creating approximate schedule for {site.name} at {date}")
    try:
        # Get timezone and calculate in local time
        local_tz = site.get_timezone()

        # Convert to local time if needed
        if date.tzinfo is None:
            date_local = local_tz.localize(date) if hasattr(local_tz, "localize") else date.replace(tzinfo=local_tz)
        else:
            date_local = date.astimezone(local_tz)

        # Calculate local noon for this date
        local_noon = date_local.replace(hour=12, minute=0, second=0, microsecond=0)

        # If we're before noon, use the midnight that just passed (start of current day)
        # If we're after noon, use the midnight that's coming up (start of next day)
        if date_local < local_noon:
            local_midnight = date_local.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            # Use midnight at end of current day (start of next day)
            local_midnight = date_local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

        # Ensure timezone awareness
        if local_midnight.tzinfo is None:
            local_midnight = (
                local_tz.localize(local_midnight)
                if hasattr(local_tz, "localize")
                else local_midnight.replace(tzinfo=local_tz)
            )

        # Convert to UTC
        local_midnight_utc = local_midnight.astimezone(ZoneInfo("UTC"))

        # Schedule bounds: ±12 hours from local midnight
        schedule_start_utc = local_midnight_utc - timedelta(hours=12)
        schedule_end_utc = local_midnight_utc + timedelta(hours=12)

        # Very rough estimates based on location. Sunset belongs to the
        # evening *before* local_midnight (the night we're observing), so
        # estimate it for the prior day. Sunrise is the morning of
        # local_midnight. Using a single date for both yielded sunset on the
        # same calendar day as sunrise, producing an inverted offer window
        # (sunset > sunrise in UTC) that crashes SK's IntervalTree on
        # _on_attach.
        local_date = local_midnight.replace(tzinfo=None)
        _, sunset_approx = _estimate_sunrise_sunset(
            site.latitude, local_date - timedelta(days=1)
        )
        sunrise_approx, _ = _estimate_sunrise_sunset(site.latitude, local_date)

        # Convert to UTC
        sunrise_local = (
            local_tz.localize(sunrise_approx)
            if hasattr(local_tz, "localize")
            else sunrise_approx.replace(tzinfo=local_tz)
        )
        sunset_local = (
            local_tz.localize(sunset_approx)
            if hasattr(local_tz, "localize")
            else sunset_approx.replace(tzinfo=local_tz)
        )

        sunrise_utc = sunrise_local.astimezone(ZoneInfo("UTC"))
        sunset_utc = sunset_local.astimezone(ZoneInfo("UTC"))

        # Standard offsets for different twilight types (in hours)
        offsets = {
            "astronomical": 1.5,  # ~90 minutes before/after
            "nautical": 1.0,  # ~60 minutes before/after
            "civil": 0.5,  # ~30 minutes before/after
        }

        return LightingSchedule(
            # Schedule metadata
            schedule_start_utc=schedule_start_utc,
            schedule_end_utc=schedule_end_utc,
            local_midnight=local_midnight_utc,
            # First day period (from schedule start until sunset)
            day_period_1_start=schedule_start_utc,
            day_period_1_end=sunset_utc,
            # Evening twilights (sun setting) - each transition is when sun crosses that elevation
            evening_civil_start=sunset_utc,
            evening_civil_end=sunset_utc + timedelta(hours=offsets["civil"]),
            evening_nautical_start=sunset_utc + timedelta(hours=offsets["civil"]),
            evening_nautical_end=sunset_utc + timedelta(hours=offsets["nautical"]),
            evening_astronomical_start=sunset_utc + timedelta(hours=offsets["nautical"]),
            evening_astronomical_end=sunset_utc + timedelta(hours=offsets["astronomical"]),
            # Night period (darkest time)
            night_start=sunset_utc + timedelta(hours=offsets["astronomical"]),
            night_end=sunrise_utc - timedelta(hours=offsets["astronomical"]),
            # Morning twilights (sun rising) - each transition is when sun crosses that elevation
            morning_astronomical_start=sunrise_utc - timedelta(hours=offsets["astronomical"]),
            morning_astronomical_end=sunrise_utc - timedelta(hours=offsets["nautical"]),
            morning_nautical_start=sunrise_utc - timedelta(hours=offsets["nautical"]),
            morning_nautical_end=sunrise_utc - timedelta(hours=offsets["civil"]),
            morning_civil_start=sunrise_utc - timedelta(hours=offsets["civil"]),
            morning_civil_end=sunrise_utc,
            # Second day period (from sunrise until schedule end)
            day_period_2_start=sunrise_utc,
            day_period_2_end=schedule_end_utc,
            # Lunar information (defaults for fallback)
            moon_phase=0.5,  # Half moon default
            moon_rise=None,
            moon_set=None,
            moon_is_waxing=True,
        )

    except Exception as e:
        logger.warning(f"Failed to create precise schedule, using fallback: {str(e)}")
        # Ultimate fallback with fixed UTC times
        # Convert to local time if needed
        if date.tzinfo is None:
            date_local = local_tz.localize(date) if hasattr(local_tz, "localize") else date.replace(tzinfo=local_tz)
        else:
            date_local = date.astimezone(local_tz)

        # If we're after noon, use midnight at start of next day
        # If we're before noon, use midnight at start of current day
        if date_local.hour >= 12:
            base_date = date_local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        else:
            base_date = date_local.replace(hour=0, minute=0, second=0, microsecond=0)

        # Convert to UTC
        base_date_utc = base_date.astimezone(ZoneInfo("UTC"))
        schedule_start = base_date_utc - timedelta(hours=12)
        schedule_end = base_date_utc + timedelta(hours=12)

        return LightingSchedule(
            # Schedule metadata
            schedule_start_utc=schedule_start,
            schedule_end_utc=schedule_end,
            local_midnight=base_date_utc,
            # First day period (from schedule start until evening twilights begin)
            day_period_1_start=schedule_start,
            day_period_1_end=base_date_utc + timedelta(hours=19),  # 7 PM sunset
            # Evening twilights (sun setting)
            evening_civil_start=base_date_utc + timedelta(hours=19),
            evening_civil_end=base_date_utc + timedelta(hours=20),
            evening_nautical_start=base_date_utc + timedelta(hours=20),
            evening_nautical_end=base_date_utc + timedelta(hours=21),
            evening_astronomical_start=base_date_utc + timedelta(hours=21),
            evening_astronomical_end=base_date_utc + timedelta(hours=22),
            # Night period (darkest time)
            night_start=base_date_utc + timedelta(hours=22),
            night_end=base_date_utc + timedelta(hours=4),  # 4 AM next day
            # Morning twilights (sun rising)
            morning_astronomical_start=base_date_utc + timedelta(hours=4),
            morning_astronomical_end=base_date_utc + timedelta(hours=5),
            morning_nautical_start=base_date_utc + timedelta(hours=5),
            morning_nautical_end=base_date_utc + timedelta(hours=6),
            morning_civil_start=base_date_utc + timedelta(hours=6),
            morning_civil_end=base_date_utc + timedelta(hours=7),  # 7 AM sunrise
            # Second day period (from morning twilights end until schedule end)
            day_period_2_start=base_date_utc + timedelta(hours=7),
            day_period_2_end=schedule_end,
            # Lunar information (defaults for fallback)
            moon_phase=0.5,  # Half moon default
            moon_rise=None,
            moon_set=None,
            moon_is_waxing=True,
        )


class SiteMetadata(BaseModel):
    name: str = Field(description="Site name", default=None)
    latitude: float = Field(description="Latitude", ge=-90, le=90)
    longitude: float = Field(description="Longitude", ge=-180, le=180)
    altitude_km: float = Field(description="Altitude in km", default=None, ge=0)
    minimum_altitude_degrees: float = Field(
        default=0.0,
        ge=0,
        le=90,
        description=(
            "Site-wide horizon floor in degrees. Every task source uses "
            "max(its own minimum_altitude_degrees, this value) as its "
            "effective minimum. Set this to the lowest altitude the mount "
            "can safely point at — protects against any preset overriding "
            "below a hardware limit. Default 0 preserves prior behavior."
        ),
    )

    # Private cached timezone fields (excluded from serialization)
    timezone_name_cache: Optional[str] = Field(default=None, exclude=True)
    timezone_obj_cache: Optional[object] = Field(default=None, exclude=True)

    def calculate_altitude(self, ra_deg: float, dec_deg: float, time: datetime) -> float:
        """
        Calculate the altitude of a celestial object at this site.

        Parameters
        ----------
        ra_deg : float
            Right ascension in degrees (0 to 360)
        dec_deg : float
            Declination in degrees (-90 to +90)
        time : datetime
            Time of observation (must be timezone-aware)

        Returns
        -------
        float
            Altitude in degrees above horizon

        Raises
        ------
        ValueError
            If inputs are invalid or out of range
        """
        try:
            from astropy.coordinates import EarthLocation, SkyCoord, AltAz
            from astropy.time import Time
            import astropy.units as u

            # Validate inputs
            if not -90 <= dec_deg <= 90:
                raise ValueError(f"Declination {dec_deg} must be between -90 and +90 degrees")

            # Normalize RA to 0-360 range
            ra_deg = ra_deg % 360

            # Check for NaN/inf values
            if np.any(np.isnan([ra_deg, dec_deg])) or np.any(np.isinf([ra_deg, dec_deg])):
                raise ValueError(f"Invalid coordinates: RA={ra_deg}, Dec={dec_deg}")

            # Ensure time is timezone-aware
            if time.tzinfo is None:
                raise ValueError("Time must be timezone-aware")

            # Convert time to UTC for astropy
            utc_time = time.astimezone(timezone.utc)

            # Create location object
            location = EarthLocation(
                lat=self.latitude * u.deg, lon=self.longitude * u.deg, height=(self.altitude_km or 0) * u.km
            )

            # Create sky coordinates with explicit frame
            target = SkyCoord(
                ra=ra_deg * u.deg,
                dec=dec_deg * u.deg,
                frame="icrs",  # Explicitly use ICRS frame
            )

            # Create time object
            t = Time(utc_time)

            # Calculate alt/az
            altaz_frame = AltAz(obstime=t, location=location)
            altaz = target.transform_to(altaz_frame)

            # Return altitude as float
            return float(altaz.alt.deg)

        except ImportError:
            raise ImportError("astropy is required for altitude calculations. Install with: pip install astropy")
        except Exception as e:
            logger.error(f"Error calculating altitude: {str(e)}")
            raise

    def get_timezone(self):
        """
        Get the timezone for this site location (cached).

        Returns
        -------
        timezone object
            Timezone object for the site location
        """
        if self.timezone_obj_cache is not None:
            return self.timezone_obj_cache

        TimezoneFinder, ZoneInfo = _get_timezone_modules()

        # Auto-detect timezone from lat/lon
        tf = TimezoneFinder()
        tz_name = tf.timezone_at(lat=self.latitude, lng=self.longitude)

        if tz_name is None:
            # Fallback to UTC if no timezone found (e.g., ocean locations)
            tz_name = "UTC"

        # Cache both the name and the object
        self.timezone_name_cache = tz_name
        self.timezone_obj_cache = ZoneInfo(tz_name)

        logger.info(f"Detected and cached timezone {self.timezone_obj_cache} for site {self.name}")

        return self.timezone_obj_cache

    def get_timezone_name(self) -> str:
        """Get the timezone name as a string (cached)."""
        if self.timezone_name_cache is not None:
            return self.timezone_name_cache

        # This will populate both timezone_name_cache and timezone_obj_cache
        self.get_timezone()
        return self.timezone_name_cache

    @property
    def local_timezone(self) -> str:
        """Get the local timezone name as a string property (cached)."""
        return self.get_timezone_name()

    def model_post_init(self, __context) -> None:
        """Initialize timezone cache after model creation."""
        # Optionally pre-populate timezone on creation
        # Comment out if you prefer lazy loading
        try:
            self.get_timezone()
        except Exception as e:
            logger.warning(f"Could not pre-populate timezone for site {self.name}: {e}")

    def __str__(self) -> str:
        """String representation including timezone."""
        try:
            tz = self.local_timezone
            return f"Site({self.name}, {self.latitude:.4f}, {self.longitude:.4f}, {tz})"
        except Exception:
            return f"Site({self.name}, {self.latitude:.4f}, {self.longitude:.4f})"
