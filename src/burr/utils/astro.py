from typing import Any, Dict
from zoneinfo import ZoneInfo
import numpy as np
from numpy.typing import NDArray
from skyfield.timelib import Time
from skyfield.positionlib import Geocentric
from skyfield.toposlib import Topos


def calculate_position(
    time: Time,
    sun: Geocentric,
    earth: Geocentric,
    sat: Geocentric,
    obs: Topos,
    local_tz: str,
) -> Dict[str, Any]:
    """
    Calculate the position and visibility of a satellite.

    Parameters
    ----------
    time : Time
        Skyfield time object for the calculation
    sun : Geocentric
        Skyfield sun object
    earth : Geocentric
        Skyfield earth object
    sat : Geocentric
        Skyfield satellite object
    obs : Topos
        Observer location on Earth
    local_tz : str
        Local timezone name (e.g. 'US/Hawaii')

    Returns
    -------
    Dict[str, Any]
        Dictionary containing satellite position and visibility information:
        - target: Satellite name/ID
        - utc: UTC timestamp
        - local: Local time
        - sunlit: Whether satellite is in sunlight
        - ra: Right ascension
        - dec: Declination
        - alt: Altitude
        - az: Azimuth
        - alt_deg: Altitude in degrees
        - az_deg: Azimuth in degrees
        - km_range: Distance to satellite in km
        - type: Object type ('rso' for resident space object)
    """
    # Get positions of bodies in km
    sunpos, earthpos, satpos = [body.at(time).position.km for body in (sun, earth, sat)]

    # Calculate vectors from sun to earth and satellite
    sunearth: NDArray = earthpos - sunpos
    sunsat: NDArray = satpos - sunpos

    # Normalize vectors
    sunearthnorm: NDArray = sunearth / np.sqrt((sunearth**2).sum(axis=0))
    sunsatnorm: NDArray = sunsat / np.sqrt((sunsat**2).sum(axis=0))

    # Calculate angles and distances
    angle = np.arccos((sunearthnorm * sunsatnorm).sum(axis=0))
    sunearthdistance = np.sqrt((sunearth**2).sum(axis=0))
    sunsatdistance = np.sqrt((sunsat**2).sum(axis=0))

    # Calculate Earth's limb angle (using Earth's radius in km)
    limbangle = np.arctan2(6378.137, sunearthdistance)

    # Convert times
    utc_dt = time.utc_datetime()
    local_dt = utc_dt.astimezone(ZoneInfo(local_tz))

    # Determine if satellite is sunlit
    sunlit = bool((angle > limbangle) or (sunsatdistance < sunearthdistance))

    # Calculate topocentric coordinates
    coords = (earth + obs).at(time).observe(sat).apparent()
    ra, dec, dist = coords.radec()
    alt, az, dist = coords.altaz()

    return {
        "target": str(sat.target)[2:],  # Remove leading "0 "
        "utc": utc_dt,
        "local": local_dt,
        "sunlit": sunlit,
        "ra": ra,
        "dec": dec,
        "alt": alt,
        "az": az,
        "alt_deg": alt.degrees,
        "az_deg": az.degrees,
        "km_range": dist.km,
        "type": "rso",
    }
