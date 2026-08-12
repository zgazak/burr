"""Landolt photometric-standards catalog loads clean.

Regression guard: the VizieR header block is sandwiched between two dashes
lines, and the loader used to start at the first one — ingesting the column
headers ("Name", the dashes, a blank line) as stars with NaN coordinates.
load_catalog() now drops any row without usable coordinates, so no header
junk can reach a consumer.
"""

import math
from pathlib import Path

from burr.core.constants import LANDOLT_STARS_PATH
from burr.task_source.photometry.catalog import load_catalog


def test_catalog_has_no_invalid_coordinate_rows():
    stars = load_catalog(Path(LANDOLT_STARS_PATH))

    assert len(stars) > 500, "catalog should hold the ~595 real Landolt standards"

    # Every star has usable coordinates — no NaN, nothing out of range.
    for s in stars:
        assert not math.isnan(s.ra_deg) and not math.isnan(s.dec_deg)
        assert 0 <= s.ra_deg <= 360
        assert -90 <= s.dec_deg <= 90

    # The specific header rows that used to leak in are gone.
    names = {s.name for s in stars}
    assert names.isdisjoint({"", "Name", "-----------"})
