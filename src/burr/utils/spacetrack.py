import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from burr.utils.astro import calculate_position
import numpy as np
import json
from collections import deque

import skyfield
from functools import lru_cache
from skyfield.api import Loader, EarthSatellite, Topos
from skyfield.iokit import load_file

from burr.core.constants import SPACETRACK_DIR, SUPPORTED_CALSATS
from burr.utils.ephemeris import EPHEMERIS_FILE, get_ephemeris_file


@lru_cache(maxsize=1)
def _ephemeris():
    """Load the planetary ephemeris once per process, downloading it if absent.

    skyfield's ``load()`` mishandles an absolute path (it treats the whole path
    as a remote filename, builds a bogus URL and 404s). ``load_file`` opens a
    known path directly, and ``get_ephemeris_file`` guarantees the file exists
    even when ``initialize_engine()`` never ran — e.g. burr driven as an SK
    plugin, where nothing calls ``ensure_ephemeris()``.
    """
    return load_file(str(get_ephemeris_file(EPHEMERIS_FILE)))


# Define some well-known LEO satellites
WELL_KNOWN_LEOS = [
    25544,  # ISS (International Space Station)
    43013,  # Starlink-1007
    44713,  # Starlink-1130
    47129,  # Starlink-1341
    48274,  # Starlink-1463
    49032,  # Starlink-1575
    51850,  # Starlink-2062
    52738,  # Starlink-2230
    53755,  # Starlink-3090
    54684,  # Starlink-3582
]


class VisibilityWindow(BaseModel):
    """Represents a visibility window for a satellite."""

    start_time: datetime = Field(..., description="Start time of visibility window")
    end_time: datetime = Field(..., description="End time of visibility window")
    max_altitude: float = Field(..., description="Maximum altitude during window (degrees)")
    max_altitude_time: datetime = Field(..., description="Time of maximum altitude")
    duration_minutes: float = Field(..., description="Duration of visibility window in minutes")

    @property
    def duration_timedelta(self) -> timedelta:
        """Get duration as a timedelta object."""
        return self.end_time - self.start_time


class TLE(BaseModel):
    """Represents a single TLE (Two Line Element) entry."""

    norad_id: str = Field(..., description="NORAD Catalog Number (5-char designator: numeric '00000' or Alpha-5 'A0000')")
    name: str = Field(..., description="Satellite name")
    line0: str = Field(..., description="TLE line 0 (name)")
    line1: str = Field(..., description="TLE line 1")
    line2: str = Field(..., description="TLE line 2")

    @property
    def satellite(self) -> EarthSatellite:
        """Create a Skyfield satellite object from this TLE."""
        try:
            # Validate TLE format
            if not self.line1 or not self.line2:
                raise ValueError("TLE lines cannot be empty")

            # Line 1 validation
            if len(self.line1) != 69:
                raise ValueError(f"TLE line 1 must be 69 characters long, got {len(self.line1)}")
            if not self.line1.startswith("1 "):
                raise ValueError("TLE line 1 must start with '1 '")

            # Line 2 validation
            if len(self.line2) != 69:
                raise ValueError(f"TLE line 2 must be 69 characters long, got {len(self.line2)}")
            if not self.line2.startswith("2 "):
                raise ValueError("TLE line 2 must start with '2 '")

            # Try to create the satellite object to validate the TLE data
            return EarthSatellite(self.line1, self.line2, self.name)
        except Exception as e:
            logger.error(f"Invalid TLE format for satellite {self.name} (NORAD ID: {self.norad_id}): {e}")
            raise ValueError(f"Invalid TLE format: {e}")

    @property
    def orbit_type(self) -> str:
        """Determine orbit type based on mean motion."""
        try:
            # Extract mean motion from line 2 (columns 52-63)
            mean_motion = float(self.line2[52:63])

            if mean_motion < 0.5:
                return "HEO"  # Highly Elliptical Orbit
            elif mean_motion < 3.0:
                return "GEO"  # Geostationary/Geosynchronous
            elif mean_motion < 8.0:
                return "MEO"  # Medium Earth Orbit
            else:
                return "LEO"  # Low Earth Orbit
        except Exception:
            return "UNKNOWN"

    def is_visible(self, site=None, time=None, min_altitude_deg=10.0) -> bool:
        """
        Check if this satellite is visible from the given site at the given time.

        Parameters
        ----------
        site : SiteMetadata, optional
            Observer location. If None, uses default Hawaii location.
        time : datetime, optional
            Time for visibility calculation. If None, uses current time.
        min_altitude_deg : float, optional
            Minimum altitude in degrees for visibility (default: 10.0)

        Returns
        -------
        bool
            True if satellite is visible (above min_altitude_deg and sunlit)
        """
        try:
            # Set up Skyfield objects
            loader = Loader(".")
            ts = loader.timescale(builtin=True)
            eph_data = _ephemeris()
            sun = eph_data["sun"]
            earth = eph_data["earth"]

            # Use provided site or default to Hawaii
            if site is None:
                # Default Hawaii location (Haleakala Observatory)
                observatory = Topos(
                    latitude_degrees=20.745889,
                    longitude_degrees=-156.43172,
                    elevation_m=150,
                )
                tz_string = "US/Hawaii"
            else:
                observatory = Topos(
                    latitude_degrees=site.latitude,
                    longitude_degrees=site.longitude,
                    elevation_m=(site.altitude_km or 0) * 1000,  # Convert km to m
                )
                tz_string = site.get_timezone_name()

            # Use provided time or current time
            if time is None:
                current_time = datetime.now(ZoneInfo(tz_string))
            else:
                current_time = time

            # Calculate satellite position
            sat = earth + self.satellite

            # Check if the current time is within a reasonable range for this TLE
            try:
                # Convert current time to Skyfield time and check if it's valid
                skyfield_time = ts.utc(current_time)

                # Attempt to calculate the position
                coords = calculate_position(
                    skyfield_time,
                    sun,
                    earth,
                    sat,
                    observatory,
                    tz_string,
                )

                # Check visibility criteria:
                # 1. Above minimum altitude
                # 2. Satellite is sunlit (visible against dark sky)
                altitude_ok = coords["alt"].degrees >= min_altitude_deg
                sunlit_ok = coords["sunlit"]

                return altitude_ok and sunlit_ok

            except Exception as time_error:
                # If there's a time-related error (like ephemeris range), log it but don't crash
                if "ephemeris segment only covers dates" in str(time_error):
                    logger.debug(f"Satellite {self.norad_id} TLE epoch outside ephemeris range: {time_error}")
                else:
                    logger.warning(f"Error calculating position for satellite {self.norad_id}: {time_error}")
                return False

        except Exception as e:
            logger.warning(f"Error calculating visibility for satellite {self.norad_id}: {e}")
            return False

    def get_visibility_windows(
        self, site=None, start_time=None, duration_hours=24, min_altitude_deg=10.0, min_duration_minutes=1.0
    ) -> List[VisibilityWindow]:
        """
        Get visibility windows for this satellite over a specified time period.

        Parameters
        ----------
        site : SiteMetadata, optional
            Observer location. If None, uses default Hawaii location.
        start_time : datetime, optional
            Start time for search. If None, uses current time.
        duration_hours : float, optional
            Duration to search in hours (default: 24)
        min_altitude_deg : float, optional
            Minimum altitude in degrees for visibility (default: 10.0)
        min_duration_minutes : float, optional
            Minimum duration for a visibility window in minutes (default: 1.0)

        Returns
        -------
        List[VisibilityWindow]
            List of visibility windows
        """
        try:
            # Set up Skyfield objects
            loader = Loader(".")
            ts = loader.timescale(builtin=True)
            eph_data = _ephemeris()
            sun = eph_data["sun"]
            earth = eph_data["earth"]

            # Use provided site or default to Hawaii
            if site is None:
                # Default Hawaii location (Haleakala Observatory)
                observatory = Topos(
                    latitude_degrees=20.745889,
                    longitude_degrees=-156.43172,
                    elevation_m=150,
                )
                tz_string = "US/Hawaii"
            else:
                observatory = Topos(
                    latitude_degrees=site.latitude,
                    longitude_degrees=site.longitude,
                    elevation_m=(site.altitude_km or 0) * 1000,  # Convert km to m
                )
                tz_string = site.get_timezone_name()

            # Use provided time or current time
            if start_time is None:
                current_time = datetime.now(ZoneInfo(tz_string))
            else:
                current_time = start_time

            # Calculate end time
            end_time = current_time + timedelta(hours=duration_hours)

            # Sample the satellite position at regular intervals
            sample_interval_minutes = 1.0  # Sample every minute
            sample_times = []
            t = current_time
            while t <= end_time:
                sample_times.append(t)
                t += timedelta(minutes=sample_interval_minutes)

            # Calculate positions and visibility
            sat = earth + self.satellite
            visibility_data = []

            for sample_time in sample_times:
                try:
                    skyfield_time = ts.utc(sample_time)
                    coords = calculate_position(
                        skyfield_time,
                        sun,
                        earth,
                        sat,
                        observatory,
                        tz_string,
                    )

                    is_visible = coords["alt"].degrees >= min_altitude_deg and coords["sunlit"]
                    visibility_data.append(
                        {"time": sample_time, "altitude": coords["alt"].degrees, "visible": is_visible}
                    )
                except Exception as e:
                    # Handle ephemeris range errors gracefully
                    if "ephemeris segment only covers dates" in str(e):
                        logger.debug(f"Time {sample_time} outside ephemeris range for satellite {self.norad_id}")
                    else:
                        logger.warning(f"Error calculating position at {sample_time}: {e}")
                    visibility_data.append({"time": sample_time, "altitude": 0.0, "visible": False})

            # Find visibility windows
            windows = []
            window_start = None
            window_data = []

            for i, data in enumerate(visibility_data):
                if data["visible"]:
                    if window_start is None:
                        # Start of new window
                        window_start = data["time"]
                        window_data = [data]
                    else:
                        # Continue existing window
                        window_data.append(data)
                else:
                    if window_start is not None:
                        # End of window
                        window_end = visibility_data[i - 1]["time"]

                        # Check if window is long enough
                        duration = (window_end - window_start).total_seconds() / 60.0
                        if duration >= min_duration_minutes:
                            # Find maximum altitude and its time
                            max_alt_data = max(window_data, key=lambda x: x["altitude"])

                            window = VisibilityWindow(
                                start_time=window_start,
                                end_time=window_end,
                                max_altitude=max_alt_data["altitude"],
                                max_altitude_time=max_alt_data["time"],
                                duration_minutes=duration,
                            )
                            windows.append(window)

                        window_start = None
                        window_data = []

            # Handle case where satellite is visible at the end of the time period
            if window_start is not None and window_data:
                window_end = window_data[-1]["time"]
                duration = (window_end - window_start).total_seconds() / 60.0
                if duration >= min_duration_minutes:
                    max_alt_data = max(window_data, key=lambda x: x["altitude"])

                    window = VisibilityWindow(
                        start_time=window_start,
                        end_time=window_end,
                        max_altitude=max_alt_data["altitude"],
                        max_altitude_time=max_alt_data["time"],
                        duration_minutes=duration,
                    )
                    windows.append(window)

            return windows

        except Exception as e:
            logger.error(f"Error calculating visibility windows for satellite {self.norad_id}: {e}")
            return []

    @classmethod
    def from_tle_lines(cls, lines: tuple[str, str, str]) -> "TLE":
        """Create a TLE object from three lines of TLE data."""
        name, line1, line2 = [line.strip() for line in lines]
        # Keep the catalog number as its 5-char designator (Alpha-5 like 'A0000' isn't an int);
        # zero-pad short numerics so it matches Space-Track's on-wire form.
        norad_id = line2.split()[1].zfill(5)
        return cls(
            norad_id=norad_id, name=name[2:] if name.startswith("0 ") else name, line0=name, line1=line1, line2=line2
        )


class TLECatalog(BaseModel):
    """A collection of TLEs indexed by NORAD ID."""

    tles: Dict[str, TLE] = Field(default_factory=dict, description="TLEs indexed by NORAD ID")
    date: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d"),
        description="Date of the catalog in YYYYMMDD format",
    )
    catalog_type: str = Field(default="MIXED", description="Type of catalog (GEO, LEO, MIXED)")

    def add_tle(self, tle: TLE) -> None:
        """Add a TLE to the catalog."""
        self.tles[tle.norad_id] = tle

    def get_tle(self, norad_id: str) -> Optional[TLE]:
        """Get a TLE by NORAD ID."""
        return self.tles.get(norad_id)

    def filter_satellites(self, norad_ids: List[str]) -> "TLECatalog":
        """Create a new catalog with only the specified satellites."""
        filtered = TLECatalog(date=self.date, catalog_type=self.catalog_type)
        for norad_id in norad_ids:
            if tle := self.get_tle(norad_id):
                filtered.add_tle(tle)
        return filtered

    def filter_by_orbit_type(self, orbit_types: List[str]) -> "TLECatalog":
        """Create a new catalog with only satellites of specified orbit types."""
        filtered = TLECatalog(date=self.date, catalog_type=f"FILTERED_{'+'.join(orbit_types)}")
        for tle in self.tles.values():
            if tle.orbit_type in orbit_types:
                filtered.add_tle(tle)
        return filtered

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the catalog."""
        orbit_counts = {}
        for tle in self.tles.values():
            orbit_type = tle.orbit_type
            orbit_counts[orbit_type] = orbit_counts.get(orbit_type, 0) + 1

        return {"total": len(self.tles), "by_orbit_type": orbit_counts}

    @classmethod
    def from_file(cls, filename: str, catalog_type: str = "MIXED") -> "TLECatalog":
        """Load TLEs from a file into a catalog."""
        catalog = cls(catalog_type=catalog_type)

        with open(filename, "r") as f:
            lines = []
            for line in f:
                if line.strip():
                    lines.append(line)
                    if len(lines) == 3:
                        tle = TLE.from_tle_lines(tuple(lines))
                        catalog.add_tle(tle)
                        lines = []

        return catalog

    def to_file(self, filename: str) -> None:
        """Save the catalog to a file in TLE format."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            for tle in self.tles.values():
                f.write(f"{tle.line0}\n{tle.line1}\n{tle.line2}\n")

    @classmethod
    def combine_catalogs(cls, catalogs: List["TLECatalog"]) -> "TLECatalog":
        """Combine multiple catalogs into one."""
        combined = cls(catalog_type="COMBINED")
        for catalog in catalogs:
            for tle in catalog.tles.values():
                combined.add_tle(tle)
        return combined


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for special types used in the codebase."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (deque, set)):
            return list(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, (bool, np.bool_)):
            return int(obj)
        if isinstance(obj, skyfield.units.Angle):
            return obj.degrees if hasattr(obj, "degrees") else obj.hours
        if isinstance(obj, TLE):
            return obj.model_dump()
        return super().default(obj)


def get_spacetrack_credentials() -> tuple[str, str]:
    """Get SpaceTrack credentials from environment variables or .env file."""
    env_path = Path(os.getenv("BURR_ENV_PATH", ".env"))
    if env_path.exists():
        load_dotenv(env_path)

    username = os.getenv("SPACETRACK_USERNAME")
    password = os.getenv("SPACETRACK_PASSWORD")

    if not username or not password:
        raise ValueError(
            "SpaceTrack credentials not found. Please set SPACETRACK_USERNAME and "
            "SPACETRACK_PASSWORD environment variables or add them to your .env file."
        )

    return username, password


def get_session() -> requests.Session:
    """Get an authenticated SpaceTrack session."""
    username, password = get_spacetrack_credentials()

    session = requests.Session()
    logger.debug("Authenticating with SpaceTrack...")
    resp = session.post("https://www.space-track.org/ajaxauth/login", data={"identity": username, "password": password})
    resp.raise_for_status()
    logger.debug("Successfully authenticated with SpaceTrack")
    return session


def load_tles(filename):
    tles = []
    reading = True
    with open(filename, "r") as f:
        while reading:
            tle0 = f.readline()
            tle1 = f.readline()
            tle2 = f.readline()
            if len(tle1) > 0 and tle1[0] == "1" and tle2[0] == "2":
                tles.append((tle0.strip(), tle1.strip(), tle2.strip()))
            else:
                reading = False

    return tles


def tle_num_to_str(tle_num):
    return "%.5i" % tle_num


def tle_list_coords(tle_list: List[str] = SUPPORTED_CALSATS, site=None, time=None, min_altitude_deg=25.0) -> List[dict]:
    """
    Get coordinates for a list of satellites.

    Parameters
    ----------
    tle_list : List[int]
        List of NORAD IDs to get coordinates for
    site : SiteMetadata, optional
        Observer location. If None, uses default Hawaii location.
    time : datetime, optional
        Time for calculation. If None, uses current time.
    min_altitude_deg : float, optional
        Minimum altitude in degrees for inclusion (default: 25.0)

    Returns
    -------
    List[dict]
        List of coordinate dictionaries for visible satellites
    """
    # Set up Skyfield objects
    loader = Loader(".")
    ts = loader.timescale(builtin=True)
    eph_data = _ephemeris()
    sun = eph_data["sun"]
    earth = eph_data["earth"]

    # Use provided site or default to Hawaii
    if site is None:
        # Default Hawaii location (Haleakala Observatory)
        observatory = Topos(
            latitude_degrees=20.745889,
            longitude_degrees=-156.43172,
            elevation_m=150,
        )
        tz_string = "US/Hawaii"
    else:
        observatory = Topos(
            latitude_degrees=site.latitude,
            longitude_degrees=site.longitude,
            elevation_m=(site.altitude_km or 0) * 1000,  # Convert km to m
        )
        tz_string = site.get_timezone_name()

    local_tz = ZoneInfo(tz_string)

    # Get TLEs and filter to desired satellites
    tle_file = fresh_tles()
    catalog = TLECatalog.from_file(tle_file)
    filtered_catalog = catalog.filter_satellites(tle_list)

    # Calculate positions
    visible_sats = []
    if time is None:
        current_time = datetime.now(local_tz)
    else:
        current_time = time

    for tle in filtered_catalog.tles.values():
        sat = earth + tle.satellite
        coords = calculate_position(
            ts.utc(current_time),
            sun,
            earth,
            sat,
            observatory,
            tz_string,
        )

        if coords["alt"].degrees > min_altitude_deg:
            logger.info(f"Satellite {tle.norad_id} is visible")
            coords["tle"] = tle.model_dump()
            visible_sats.append(coords)

    # Save results
    with open("calsats.json", "w") as fp:
        json.dump(visible_sats, fp, cls=CustomJSONEncoder, indent=4)

    return visible_sats


def find_tles_that_are_visible(tles: List[TLE], site=None, time=None, min_altitude_deg=10.0) -> List[TLE]:
    """
    Find TLEs that are visible at the given time and location.

    Parameters
    ----------
    tles : List[TLE]
        List of TLE objects to check for visibility
    site : SiteMetadata, optional
        Observer location. If None, uses default Hawaii location.
    time : datetime, optional
        Time for visibility calculation. If None, uses current time.
    min_altitude_deg : float, optional
        Minimum altitude in degrees for visibility (default: 10.0)

    Returns
    -------
    List[TLE]
        List of visible TLE objects
    """
    visible_tles = []

    for tle in tles:
        try:
            if tle.is_visible(site=site, time=time, min_altitude_deg=min_altitude_deg):
                visible_tles.append(tle)
                logger.debug(f"Satellite {tle.name} ({tle.norad_id}) is visible")
            else:
                logger.debug(f"Satellite {tle.name} ({tle.norad_id}) is not visible")
        except Exception as e:
            logger.warning(f"Error checking visibility for {tle.name} ({tle.norad_id}): {e}")

    logger.info(f"Found {len(visible_tles)} visible satellites out of {len(tles)} total")
    return visible_tles


def fresh_tles() -> str:
    """
    Download fresh TLEs for geostationary satellites.

    Queries Space-Track.org for satellites with:
    - Mean motion between 0.9 and 3 revs/day
    - Eccentricity < 0.1
    - Object type = payload
    - Epoch within last 30 days

    Returns
    -------
    str
        Path to the TLE file containing the downloaded data
    """
    logger.info("Downloading fresh TLEs for GEO satellites")

    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    tle_file = os.path.join(SPACETRACK_DIR, f"{date}.tle")
    latest_file = os.path.join(SPACETRACK_DIR, "latest.tle")

    # Check if we already have today's TLEs
    if os.path.exists(tle_file):
        catalog = TLECatalog.from_file(tle_file, catalog_type="GEO")
        if catalog.tles:
            logger.info(f"Already loaded TLEs for {date}, {len(catalog.tles)} satellites in {tle_file}")
            logger.info(f"Updating {latest_file}")
            catalog.to_file(latest_file)
            return tle_file

    # Query parameters for GEO satellites
    query_url = (
        "https://www.space-track.org/basicspacedata/query/class/gp/"
        "EPOCH/>now-30/MEAN_MOTION/0.9--3/ECCENTRICITY/<0.1/"
        "OBJECT_TYPE/payload/orderby/NORAD_CAT_ID,EPOCH/format/3le"
    )

    # Use the session management
    session = get_session()
    resp = session.get(query_url)
    resp.raise_for_status()

    # Write the TLEs to file
    os.makedirs(SPACETRACK_DIR, exist_ok=True)
    with open(tle_file, "w") as fp:
        fp.write("\n".join(resp.text.splitlines()))

    # Verify download and update latest.tle
    if os.path.exists(tle_file):
        catalog = TLECatalog.from_file(tle_file, catalog_type="GEO")
        if catalog.tles:
            logger.info(f"Download complete: {len(catalog.tles)} satellites in {tle_file}")
            logger.info(f"Updating {latest_file}")
            catalog.to_file(latest_file)
            return tle_file

    # Handle failure case
    if os.path.exists(latest_file):
        catalog = TLECatalog.from_file(latest_file, catalog_type="GEO")
        logger.warning(f"TLE download failed. Falling back to {len(catalog.tles)} satellites in {latest_file}")
        return latest_file

    raise RuntimeError("Failed to download TLEs and no fallback file available")


def fresh_leo_tles() -> str:
    """
    Download fresh TLEs for low Earth orbit satellites.

    Queries Space-Track.org for satellites with:
    - Mean motion between 10 and 17 revs/day (typical LEO range)
    - Eccentricity < 0.25
    - Object type = payload
    - Epoch within last 7 days (LEO TLEs decay faster)

    Returns
    -------
    str
        Path to the TLE file containing the downloaded data
    """
    logger.info("Downloading fresh TLEs for LEO satellites")

    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    tle_file = os.path.join(SPACETRACK_DIR, f"{date}_leo.tle")
    latest_file = os.path.join(SPACETRACK_DIR, "latest_leo.tle")

    # Check if we already have today's TLEs
    if os.path.exists(tle_file):
        catalog = TLECatalog.from_file(tle_file, catalog_type="LEO")
        if catalog.tles:
            logger.info(f"Already loaded LEO TLEs for {date}, {len(catalog.tles)} satellites in {tle_file}")
            logger.info(f"Updating {latest_file}")
            catalog.to_file(latest_file)
            return tle_file

    # Query parameters for LEO satellites
    query_url = (
        "https://www.space-track.org/basicspacedata/query/class/gp/"
        "EPOCH/>now-7/MEAN_MOTION/10--17/ECCENTRICITY/<0.25/"
        "OBJECT_TYPE/payload/orderby/NORAD_CAT_ID,EPOCH/format/3le"
    )

    # Use the session management
    session = get_session()
    resp = session.get(query_url)
    resp.raise_for_status()

    # Write the TLEs to file
    os.makedirs(SPACETRACK_DIR, exist_ok=True)
    with open(tle_file, "w") as fp:
        fp.write("\n".join(resp.text.splitlines()))

    # Verify download and update latest_leo.tle
    if os.path.exists(tle_file):
        catalog = TLECatalog.from_file(tle_file, catalog_type="LEO")
        if catalog.tles:
            logger.info(f"Download complete: {len(catalog.tles)} LEO satellites in {tle_file}")
            logger.info(f"Updating {latest_file}")
            catalog.to_file(latest_file)
            return tle_file

    # Handle failure case
    if os.path.exists(latest_file):
        catalog = TLECatalog.from_file(latest_file, catalog_type="LEO")
        logger.warning(f"LEO TLE download failed. Falling back to {len(catalog.tles)} satellites in {latest_file}")
        return latest_file

    raise RuntimeError("Failed to download LEO TLEs and no fallback file available")


def fresh_combined_tles() -> Tuple[str, str]:
    """
    Download fresh TLEs for both GEO and LEO satellites.

    Returns
    -------
    Tuple[str, str]
        Tuple of (GEO TLE file path, LEO TLE file path)
    """
    logger.info("Downloading fresh TLEs for both GEO and LEO satellites")

    geo_file = fresh_tles()
    leo_file = fresh_leo_tles()

    return geo_file, leo_file


def get_specific_satellite_tle(norad_id: int) -> Optional[TLE]:
    """
    Get TLE for a specific satellite by NORAD ID.

    Parameters
    ----------
    norad_id : int
        NORAD catalog ID of the satellite

    Returns
    -------
    Optional[TLE]
        TLE object if found, None otherwise
    """
    logger.info(f"Downloading TLE for satellite {norad_id}")

    try:
        session = get_session()

        # First try to get the satellite name using the json format
        name_resp = session.get(
            f"https://www.space-track.org/basicspacedata/query/class/tle_latest/ORDINAL/1/NORAD_CAT_ID/"
            f"{norad_id}/format/json"
        )
        name_resp.raise_for_status()

        # Parse satellite name from JSON response
        satellite_name = f"NORAD {norad_id}"  # Default name
        try:
            import json

            json_data = json.loads(name_resp.text)
            if json_data and len(json_data) > 0:
                satellite_name = json_data[0].get("OBJECT_NAME", f"NORAD {norad_id}")
        except:
            logger.warning(f"Could not parse satellite name for {norad_id}, using default")

        # Query Space-Track.org for the TLE lines
        resp = session.get(
            f"https://www.space-track.org/basicspacedata/query/class/tle_latest/ORDINAL/1/NORAD_CAT_ID/"
            f"{norad_id}/orderby/TLE_LINE1%20ASC/format/tle"
        )
        resp.raise_for_status()

        # Parse TLE - the response contains only lines 1 and 2, so we need to add the name
        lines = resp.text.strip().split("\n")

        # Clean up lines (remove \r characters)
        lines = [line.strip() for line in lines if line.strip()]

        if len(lines) == 2:
            # Add the satellite name as line 0
            line0 = satellite_name
            line1 = lines[0]
            line2 = lines[1]

            # Validate the lines
            if line1.startswith("1 ") and line2.startswith("2 "):
                tle = TLE.from_tle_lines((line0, line1, line2))
                logger.info(f"Retrieved TLE for {tle.name} ({norad_id})")
                return tle
            else:
                logger.warning(
                    f"Invalid TLE line format for satellite {norad_id}: line1='{line1[:10]}...', line2='{line2[:10]}...'"
                )
                return None
        elif len(lines) == 3:
            # Standard 3-line format
            tle = TLE.from_tle_lines(tuple(lines))
            logger.info(f"Retrieved TLE for {tle.name} ({norad_id})")
            return tle
        else:
            logger.warning(
                f"Invalid TLE format received for satellite {norad_id}: got {len(lines)} lines, expected 2 or 3"
            )
            return None

    except Exception as e:
        logger.error(f"Error getting TLE for satellite {norad_id}: {e}")
        return None


def get_visible_satellites_at_time(
    time: datetime, site=None, min_altitude_deg=10.0, include_geo=True, include_leo=True, specific_satellites=None
) -> List[dict]:
    """
    Get all visible satellites at a specific time.

    Parameters
    ----------
    time : datetime
        The specific time to check for visibility
    site : SiteMetadata, optional
        Observer location. If None, uses default Hawaii location.
    min_altitude_deg : float, optional
        Minimum altitude in degrees for visibility (default: 10.0)
    include_geo : bool, optional
        Whether to include GEO satellites (default: True)
    include_leo : bool, optional
        Whether to include LEO satellites (default: True)
    specific_satellites : List[int], optional
        List of specific NORAD IDs to check. If provided, overrides include_geo/leo.

    Returns
    -------
    List[dict]
        List of dictionaries containing satellite information and coordinates for visible satellites.
    """
    logger.info(f"Checking satellite visibility at {time}")

    try:
        # Get catalogs
        catalogs = []

        if specific_satellites:
            # Load both catalogs and filter to specific satellites
            if include_geo:
                geo_file = fresh_tles()
                geo_catalog = TLECatalog.from_file(geo_file, catalog_type="GEO")
                catalogs.append(geo_catalog)

            if include_leo:
                leo_file = fresh_leo_tles()
                leo_catalog = TLECatalog.from_file(leo_file, catalog_type="LEO")
                catalogs.append(leo_catalog)

            # Combine and filter
            combined_catalog = TLECatalog.combine_catalogs(catalogs)
            filtered_catalog = combined_catalog.filter_satellites(specific_satellites)

        else:
            # Load requested catalog types
            if include_geo:
                geo_file = fresh_tles()
                geo_catalog = TLECatalog.from_file(geo_file, catalog_type="GEO")
                catalogs.append(geo_catalog)

            if include_leo:
                leo_file = fresh_leo_tles()
                leo_catalog = TLECatalog.from_file(leo_file, catalog_type="LEO")
                catalogs.append(leo_catalog)

            filtered_catalog = TLECatalog.combine_catalogs(catalogs)

        if not filtered_catalog.tles:
            logger.warning("No satellites found in catalogs")
            return []

        # Convert to list of TLE objects
        tles = list(filtered_catalog.tles.values())

        # Find visible satellites
        visible_tles = find_tles_that_are_visible(tles, site=site, time=time, min_altitude_deg=min_altitude_deg)

        # Get detailed coordinates for visible satellites
        if visible_tles:
            visible_coords = tle_list_coords(
                [tle.norad_id for tle in visible_tles], site=site, time=time, min_altitude_deg=min_altitude_deg
            )

            logger.info(f"Found {len(visible_coords)} visible satellites at {time}")
            return visible_coords
        else:
            logger.info("No visible satellites found")
            return []

    except Exception as e:
        logger.error(f"Error getting visible satellites at time {time}: {e}")
        return []


def get_next_visible_passes(
    norad_id: int, site=None, start_time=None, duration_hours=24, min_altitude_deg=10.0, min_duration_minutes=1.0
) -> List[VisibilityWindow]:
    """
    Get the next visible passes for a specific satellite.

    Parameters
    ----------
    norad_id : int
        NORAD catalog ID of the satellite
    site : SiteMetadata, optional
        Observer location. If None, uses default Hawaii location.
    start_time : datetime, optional
        Start time for search. If None, uses current time.
    duration_hours : float, optional
        Duration to search in hours (default: 24)
    min_altitude_deg : float, optional
        Minimum altitude in degrees for visibility (default: 10.0)
    min_duration_minutes : float, optional
        Minimum duration for a visibility window in minutes (default: 1.0)

    Returns
    -------
    List[VisibilityWindow]
        List of visibility windows for the satellite
    """
    logger.info(f"Getting next visible passes for satellite {norad_id}")

    try:
        # Get TLE for the specific satellite
        tle = get_specific_satellite_tle(norad_id)
        if not tle:
            logger.warning(f"Could not get TLE for satellite {norad_id}")
            return []

        # Get visibility windows
        windows = tle.get_visibility_windows(
            site=site,
            start_time=start_time,
            duration_hours=duration_hours,
            min_altitude_deg=min_altitude_deg,
            min_duration_minutes=min_duration_minutes,
        )

        logger.info(f"Found {len(windows)} visibility windows for satellite {norad_id}")
        return windows

    except Exception as e:
        logger.error(f"Error getting visibility windows for satellite {norad_id}: {e}")
        return []


def update_tles() -> None:
    """Update TLE files for calibration satellites."""
    try:
        session = get_session()
        catalog = TLECatalog()

        for sat in SUPPORTED_CALSATS:
            logger.info(f"Updating TLEs for {sat}")

            # Query Space-Track.org
            resp = session.get(
                "https://www.space-track.org/basicspacedata/query/class/tle_latest/ORDINAL/1/NORAD_CAT_ID/"
                + str(sat)
                + "/orderby/TLE_LINE1%20ASC/format/tle"
            )
            resp.raise_for_status()

            # Parse and store TLE
            lines = resp.text.strip().split("\n")
            if len(lines) == 3:
                tle = TLE.from_tle_lines(tuple(lines))
                catalog.add_tle(tle)

                # Write individual TLE file
                tle_file = os.path.join(SPACETRACK_DIR, f"{sat}.txt")
                with open(tle_file, "w") as f:
                    f.write(resp.text)
                logger.info(f"Wrote TLE to {tle_file}")

                # Update ephemeris
                logger.info(f"Updating ephemeris for {sat}")
                calculate_position(tle.satellite, EPHEMERIS_FILE)
                logger.info(f"Updated ephemeris in {EPHEMERIS_FILE}")

    except Exception as e:
        logger.error(f"Error updating TLEs: {str(e)}")
        raise


def get_visible_satellites_for_site(site=None, time=None, min_altitude_deg=10.0, tle_list=None) -> List[dict]:
    """
    Get all visible satellites for a given site and time with full coordinate information.

    This is a convenience function that combines TLE loading, filtering, and visibility checking.

    Parameters
    ----------
    site : SiteMetadata, optional
        Observer location. If None, uses default Hawaii location.
    time : datetime, optional
        Time for visibility calculation. If None, uses current time.
    min_altitude_deg : float, optional
        Minimum altitude in degrees for visibility (default: 10.0)
    tle_list : List[int], optional
        List of NORAD IDs to check. If None, uses SUPPORTED_CALSATS.

    Returns
    -------
    List[dict]
        List of dictionaries containing satellite information and coordinates for visible satellites.
        Each dict contains: name, norad_id, altitude, azimuth, range, sunlit status, etc.
    """
    if tle_list is None:
        tle_list = SUPPORTED_CALSATS

    logger.info(f"Checking visibility for {len(tle_list)} satellites")

    # Get fresh TLEs
    tle_file = fresh_tles()
    catalog = TLECatalog.from_file(tle_file)
    filtered_catalog = catalog.filter_satellites(tle_list)

    # Convert to list of TLE objects
    tles = list(filtered_catalog.tles.values())

    # Find visible satellites
    visible_tles = find_tles_that_are_visible(tles, site=site, time=time, min_altitude_deg=min_altitude_deg)

    # Get detailed coordinates for visible satellites
    if visible_tles:
        visible_coords = tle_list_coords(
            [tle.norad_id for tle in visible_tles], site=site, time=time, min_altitude_deg=min_altitude_deg
        )

        logger.info(f"Found {len(visible_coords)} visible satellites with coordinates")
        return visible_coords
    else:
        logger.info("No visible satellites found")
        return []


# Example usage function
def example_usage():
    """
    Example of how to use the visibility functions, including new LEO functionality.
    """
    from burr.models.site import SiteMetadata
    from datetime import datetime
    from zoneinfo import ZoneInfo

    # Example 1: Use default Hawaii location (original functionality)
    logger.info("=== Example 1: Default Hawaii location (GEO satellites) ===")
    visible_sats = get_visible_satellites_for_site()
    logger.info(f"Visible GEO satellites at Hawaii: {len(visible_sats)}")

    # Example 2: LEO satellites at Hawaii
    logger.info("=== Example 2: LEO satellites at Hawaii ===")
    hawaii_site = SiteMetadata(name="Haleakala", latitude=20.745889, longitude=-156.43172, altitude_km=0.15)

    leo_sats = get_visible_satellites_at_time(
        time=datetime.now(ZoneInfo("UTC")),
        site=hawaii_site,
        min_altitude_deg=5.0,  # Lower threshold for LEOs
        include_geo=False,
        include_leo=True,
    )
    logger.info(f"Visible LEO satellites at Hawaii: {len(leo_sats)}")

    # Example 3: ISS visibility windows
    logger.info("=== Example 3: ISS visibility windows ===")
    iss_windows = get_next_visible_passes(
        norad_id=25544,  # ISS
        site=hawaii_site,
        duration_hours=24,
        min_altitude_deg=10.0,
    )
    logger.info(f"ISS visibility windows in next 24 hours: {len(iss_windows)}")

    if iss_windows:
        next_window = iss_windows[0]
        logger.info(f"Next ISS pass: {next_window.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(f"Duration: {next_window.duration_minutes:.1f} minutes")
        logger.info(f"Max altitude: {next_window.max_altitude:.1f}°")

    # Example 4: Combined GEO and LEO at specific time
    logger.info("=== Example 4: Combined GEO and LEO at specific time ===")
    future_time = datetime.now(ZoneInfo("UTC")) + timedelta(hours=6)

    combined_sats = get_visible_satellites_at_time(
        time=future_time, site=hawaii_site, min_altitude_deg=10.0, include_geo=True, include_leo=True
    )

    leo_count = sum(1 for sat in combined_sats if sat.get("tle", {}).get("orbit_type") == "LEO")
    geo_count = sum(1 for sat in combined_sats if sat.get("tle", {}).get("orbit_type") == "GEO")

    logger.info(f"At {future_time.strftime('%Y-%m-%d %H:%M:%S UTC')}:")
    logger.info(f"  LEO satellites: {leo_count}")
    logger.info(f"  GEO satellites: {geo_count}")
    logger.info(f"  Total visible: {len(combined_sats)}")

    # Example 5: Specific satellites (well-known LEOs)
    logger.info("=== Example 5: Specific satellites (well-known LEOs) ===")
    specific_sats = get_visible_satellites_at_time(
        time=datetime.now(ZoneInfo("UTC")), site=hawaii_site, min_altitude_deg=5.0, specific_satellites=WELL_KNOWN_LEOS
    )
    logger.info(f"Visible well-known LEO satellites: {len(specific_sats)}")

    # Print details of visible satellites
    for sat in specific_sats[:3]:  # Show first 3
        logger.info(
            f"  {sat['target']}: Alt={sat['alt_deg']:.1f}°, Az={sat['az_deg']:.1f}°, "
            f"Range={sat['km_range']:.0f}km, Sunlit={sat['sunlit']}, Type={sat['tle'].get('orbit_type', '?')}"
        )

    # Example 6: Catalog statistics
    logger.info("=== Example 6: Catalog statistics ===")
    try:
        geo_file, leo_file = fresh_combined_tles()
        geo_catalog = TLECatalog.from_file(geo_file, catalog_type="GEO")
        leo_catalog = TLECatalog.from_file(leo_file, catalog_type="LEO")

        combined_catalog = TLECatalog.combine_catalogs([geo_catalog, leo_catalog])
        stats = combined_catalog.get_statistics()

        logger.info(f"Total satellites in combined catalog: {stats['total']}")
        logger.info("By orbit type:")
        for orbit_type, count in stats["by_orbit_type"].items():
            logger.info(f"  {orbit_type}: {count} satellites")

    except Exception as e:
        logger.error(f"Error getting catalog statistics: {e}")

    # Example 7: Individual satellite TLE
    logger.info("=== Example 7: Individual satellite TLE (ISS) ===")
    iss_tle = get_specific_satellite_tle(25544)
    if iss_tle:
        logger.info(f"Retrieved TLE for {iss_tle.name}")
        logger.info(f"  Orbit type: {iss_tle.orbit_type}")
        logger.info(f"  Currently visible: {iss_tle.is_visible(site=hawaii_site)}")
    else:
        logger.warning("Could not retrieve ISS TLE")


if __name__ == "__main__":
    # Run example if script is executed directly
    example_usage()
