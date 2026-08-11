"""Alpha-5 catalog numbers must survive parsing verbatim, and short numerics zero-pad.

Regression for the calsats crash: Spacebook's full feed now carries Alpha-5 objects
(e.g. 'A0000' == 100000), which a bare int() could not parse.
"""

from burr.utils.spacetrack import TLE


def test_alpha5_designator_preserved():
    tle = TLE.from_tle_lines((
        "0 SAT A0000",
        "1 A0000U 26067CY  26222.88571527  .00004719  00000-0  21676-3 0  9994",
        "2 A0000  97.4613 180.8071 0008233 161.3775 198.7764 15.20848640 20016",
    ))
    assert tle.norad_id == "A0000"          # kept as-is, not decoded to 100000
    assert tle.line1.startswith("1 A0000")  # raw TLE untouched -> propagation/mount see A0000


def test_short_numeric_zero_padded():
    # canonical 5-char key so a sub-10000 id matches Space-Track's zero-padded form
    tle = TLE.from_tle_lines(("0 SAT 5", "1 5U", "2 5 rest"))
    assert tle.norad_id == "00005"


def test_five_digit_numeric_unchanged():
    tle = TLE.from_tle_lines(("0 ISS", "1 25544U", "2 25544 rest"))
    assert tle.norad_id == "25544"
