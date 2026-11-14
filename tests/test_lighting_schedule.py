"""LightingSchedule.calculate_twilights must schedule the night its date falls in, not tonight."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from burr.models.site import LightingSchedule, SiteMetadata

HST = ZoneInfo("Pacific/Honolulu")
SITE = SiteMetadata(name="Test", latitude=20.89, longitude=-156.32, altitude_km=0.336)


@pytest.mark.parametrize(
    ("date", "midnight"),
    [
        # A local midnight selects the night centered on it.
        (datetime(2026, 1, 15, 0, 0, tzinfo=HST), datetime(2026, 1, 15, tzinfo=HST)),
        # Before local noon: the night that ended that morning.
        (datetime(2026, 1, 15, 8, 0, tzinfo=HST), datetime(2026, 1, 15, tzinfo=HST)),
        # From local noon on: the night starting that evening.
        (datetime(2026, 1, 15, 12, 0, tzinfo=HST), datetime(2026, 1, 16, tzinfo=HST)),
        (datetime(2026, 1, 15, 20, 0, tzinfo=HST), datetime(2026, 1, 16, tzinfo=HST)),
        # A UTC time is placed by its site-local time (06:00 UTC is 20:00 HST the day before).
        (datetime(2026, 1, 16, 6, 0, tzinfo=ZoneInfo("UTC")), datetime(2026, 1, 16, tzinfo=HST)),
        # A naive time is taken as site-local.
        (datetime(2026, 1, 15, 20, 0), datetime(2026, 1, 16, tzinfo=HST)),
    ],
)
def test_schedule_follows_the_given_date(date, midnight):
    schedule = LightingSchedule.calculate_twilights(SITE, date)

    assert schedule.local_midnight == midnight


def test_a_past_night_keeps_its_own_schedule():
    """A run resumed later recomputes its schedule from its observation date; that must not
    become tonight's, or every later night resumes the first run."""
    night = datetime(2025, 10, 29, tzinfo=HST)

    schedule = LightingSchedule.calculate_twilights(SITE, night)

    assert schedule.local_midnight.date() == night.date()
    assert schedule.schedule_start_utc < schedule.evening_civil_end < schedule.schedule_end_utc
