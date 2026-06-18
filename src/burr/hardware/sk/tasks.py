"""Burr-side helpers for building SensorKit target types.

Burr emits canonical ``StandardCollectTask`` instances to the SensorKit
controller. The per-frame tracking mode is encoded directly in the task's
``target`` field — ``ICRSTarget`` for sidereal, ``RateTarget`` for custom
rate-follow, ``TLETarget`` for TLE-driven follow. Everything a downstream
SK client needs to render the task lives in standard fields.
"""

from __future__ import annotations

from datetime import datetime, timezone

import astropy.units as u
import sensorkit.api as sk
from astropy.coordinates import AltAz as AstropyAltAz, EarthLocation, SkyCoord
from astropy.time import Time
from pydantic import BaseModel
from sensorkit.astro.common import Equatorial, Horizontal, ReferenceFrame, TLE
from sensorkit.astro.target import (
    AltAzTarget as SKAltAzTarget,
    ICRSTarget,
    RateTarget,
    TLETarget as SKTLETarget,
    Target,
)

from burr.models.observation import (
    AltAzTarget,
    LunarTarget,
    RADecTarget,
    TargetUnion,
    TLETarget,
)
from burr.models.site import SiteMetadata
from burr.models.tracking import Rates


@sk.declare_keyword
class BurrContext(BaseModel):
    """Burr's identity keyword, attached to every SK collect task burr emits.

    Rides in ``StandardCollectTask.context`` as a *declared* keyword (rather
    than loose flat keys), so it round-trips on the wire as a typed object and
    is reachable from sensorkit FITS-header exprs and filename templates via
    ``context.eval`` as ``BurrContext.<field>`` (e.g. ``BurrContext.seq``).
    Being namespaced also stops these values from colliding with the generic
    flat keys the std controller injects (``target_id``, ``track_mode``, ...),
    which otherwise override anything burr puts at the top level.

    Fields
    ------
    seq
        Per-exposure collect id — a fresh value for each entry in the
        request's ``exposure_seconds``. Every SK task and every frame fanned
        out from one exposure (e.g. the sidereal reference frame plus its rate
        frames) shares one ``seq``; distinct exposures differ. This is the unit
        downstream fitting groups on ("I will fit this").
    targ
        Target name for this collect (filename-safe). The manager's best name
        for the thing observed — star name, satellite name, coverage pixel, etc.
    mode
        The task source that produced the collect: ``coverage``, ``calsats``,
        ``twilight_flats``, ``photometric_standards``, ``lunar_background``.
    """

    seq: str
    targ: str
    mode: str


def to_sk_target(burr_target) -> Target:
    """Convert a burr target to the corresponding sensorkit astro target."""
    match burr_target:
        case RADecTarget():
            return ICRSTarget(
                coords=Equatorial(
                    ra=burr_target.right_ascension_deg,
                    dec=burr_target.declination_deg,
                )
            )
        # LunarTarget extends AltAzTarget - match before AltAzTarget.
        case LunarTarget():
            return SKAltAzTarget(
                coords=Horizontal(
                    az=burr_target.azimuth_deg,
                    alt=burr_target.altitude_deg,
                )
            )
        case AltAzTarget():
            return SKAltAzTarget(
                coords=Horizontal(
                    az=burr_target.azimuth_deg,
                    alt=burr_target.altitude_deg,
                )
            )
        case TLETarget():
            # Synthesize a line0 ("name line") from the NORAD catalog number
            # in line1[2:7]. SK's newer std controller (device-module-refactors
            # branch) reads target_id via ``line0.split()[1]``, which crashes
            # with AttributeError on line0=None. The old controller read
            # NORAD directly from line1[2:7], which would be more robust —
            # flagged upstream. For now, giving line0 a two-token string
            # (``NORAD <id>``) makes split()[1] yield the NORAD number.
            norad_id = (
                burr_target.tle_line1[2:7].strip()
                if burr_target.tle_line1 and len(burr_target.tle_line1) >= 7
                else "unknown"
            )
            return SKTLETarget(
                tle=TLE(
                    line0=f"NORAD {norad_id}",
                    line1=burr_target.tle_line1,
                    line2=burr_target.tle_line2,
                )
            )
        case _:
            raise ValueError(
                f"Unsupported burr target type: {type(burr_target).__name__}"
            )


def to_sk_sidereal_target(
    burr_target: TargetUnion,
    initial_time: datetime | None = None,
    site: SiteMetadata | None = None,
) -> Target:
    """SK target for sidereal-tracked frames, freezing AltAz to ICRS.

    RADec and TLE targets pass straight through ``to_sk_target``. AltAz/Lunar
    targets are *frozen* to ICRS at ``initial_time`` (default now) using
    ``site``: sensorkit tracks an ``AltAzTarget`` as **fixed** (alt/az held
    constant), which is wrong for a sidereal collect — the mount would hold the
    horizon coordinate and let the sky drift through the field. Converting the
    alt/az pointing to the RA/Dec it maps to *at emit time* makes the mount lock
    onto that sky position and track it sidereally.

    Coverage and lunar both map in alt/az but want sidereal tracking, so they
    flow through here. Flats (tracking OFF) keep their alt/az target via
    ``to_sk_target`` and are never passed to this function.
    """
    match burr_target:
        case LunarTarget() | AltAzTarget():
            if site is None:
                raise ValueError(
                    "site is required to freeze an AltAz/Lunar target to ICRS "
                    "for sidereal tracking"
                )
            if initial_time is None:
                initial_time = datetime.now(tz=timezone.utc)
            return ICRSTarget(
                coords=_altaz_to_equatorial(
                    az_deg=burr_target.azimuth_deg,
                    alt_deg=burr_target.altitude_deg,
                    time=initial_time,
                    site=site,
                )
            )
        case _:
            # RADec → ICRSTarget, TLE → SKTLETarget: already sidereal-correct.
            return to_sk_target(burr_target)


def to_sk_ratetarget(
    burr_target: TargetUnion,
    rates: Rates,
    initial_time: datetime | None = None,
    site: SiteMetadata | None = None,
    frame: ReferenceFrame = ReferenceFrame.ICRF,
) -> RateTarget:
    """Build a sensorkit ``RateTarget`` from a burr target + concrete rates.

    Initial position comes from ``burr_target``; rates are converted from
    arcsec/s (burr's unit, stored in ``Rates``) to deg/s (sensorkit's
    ``Coordinates`` unit convention). The frame label on the returned
    ``RateTarget`` is set by ``frame``: ICRF is what pwi4 accepts, CIRF
    is what node_platform accepts. Picking the wrong one sends the
    target through sensorkit's BaseTarget.adapt() → to_ephemeris_target()
    path, which is unimplemented for RateTarget. AltAz/Lunar inputs are
    converted using the provided ``site`` and ``initial_time``.

    Parameters
    ----------
    burr_target : TargetUnion
        The burr target whose *position* to use as the rate-follow
        origin. RADec is passed through directly; AltAz/Lunar are
        converted to ICRS using ``site`` + ``initial_time``.
    rates : Rates
        RA/Dec rate vector in arcsec/s.
    initial_time : datetime, optional
        UTC time at which ``initial_coords`` is valid. Defaults to
        ``datetime.now(UTC)``.
    site : SiteMetadata, optional
        Observer location. Required for AltAz/Lunar conversion; ignored
        for RADec inputs. When omitted for AltAz, raises ValueError.

    TLE targets raise: their motion is intrinsic to the orbit and
    should not be combined with custom rates.
    """
    if initial_time is None:
        initial_time = datetime.now(tz=timezone.utc)

    rates_coords = Equatorial(
        ra=rates.right_ascension_rate_arcsec_s / 3600.0,
        dec=rates.declination_rate_arcsec_s / 3600.0,
    )

    match burr_target:
        case RADecTarget():
            initial_coords = Equatorial(
                ra=burr_target.right_ascension_deg,
                dec=burr_target.declination_deg,
            )
        case LunarTarget() | AltAzTarget():
            if site is None:
                raise ValueError(
                    "site is required to convert AltAz/Lunar target to ICRS "
                    "for RateTarget construction"
                )
            initial_coords = _altaz_to_equatorial(
                az_deg=burr_target.azimuth_deg,
                alt_deg=burr_target.altitude_deg,
                time=initial_time,
                site=site,
            )
        case TLETarget():
            raise ValueError(
                "cannot combine custom rates with a TLE target — use TLETarget alone"
            )
        case _:
            raise ValueError(
                f"Unsupported burr target type: {type(burr_target).__name__}"
            )

    return RateTarget(
        frame=frame,
        rates=rates_coords,
        initial_time=initial_time,
        initial_frame=frame,
        initial_coords=initial_coords,
    )


def _altaz_to_equatorial(
    az_deg: float,
    alt_deg: float,
    time: datetime,
    site: SiteMetadata,
) -> Equatorial:
    """Convert an (az, alt) position at a given time + site to ICRS RA/Dec."""
    location = EarthLocation(
        lat=site.latitude * u.deg,
        lon=site.longitude * u.deg,
        height=(site.altitude_km or 0) * u.km,
    )
    altaz_coord = SkyCoord(
        az=az_deg * u.deg,
        alt=alt_deg * u.deg,
        frame=AstropyAltAz(obstime=Time(time), location=location),
    )
    icrs_coord = altaz_coord.icrs
    return Equatorial(
        ra=float(icrs_coord.ra.deg),
        dec=float(icrs_coord.dec.deg),
    )
