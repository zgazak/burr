"""Spacebook TLE fetcher.

Spacebook (by COMSPOC) serves a free, unauthenticated TLE feed at
https://spacebook.com/api/entity/tle. The response is a stream of
alternating TLE line1/line2 pairs (no name lines).

This module mirrors the public surface of `burr.utils.spacetrack.fresh_tles`:
it downloads the catalog, writes it to disk in the same 3-line format that
`TLECatalog.from_file` expects (synthesizing a name line from the NORAD ID),
and returns the file path.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import requests
from loguru import logger

from burr.core.constants import RESOURCES_DIR
from burr.utils.spacetrack import TLECatalog


SPACEBOOK_URL = os.environ.get("SPACEBOOK_URL", "https://spacebook.com/api/entity/tle")
SPACEBOOK_TIMEOUT = int(os.environ.get("SPACEBOOK_TIMEOUT", "60"))
SPACEBOOK_DIR = RESOURCES_DIR / "cache" / "spacebook"


def _parse_two_line_stream(text: str) -> list[tuple[str, str, str]]:
    """Parse alternating line1/line2 pairs from Spacebook's response."""
    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out: list[tuple[str, str, str]] = []
    i = 0
    while i + 1 < len(raw_lines):
        line1, line2 = raw_lines[i], raw_lines[i + 1]
        if line1.startswith("1 ") and line2.startswith("2 "):
            norad_id = line1[2:7].strip()
            line0 = f"0 SAT {norad_id}"
            out.append((line0, line1, line2))
            i += 2
        else:
            i += 1
    return out


def fresh_tles() -> str:
    """Download the full Spacebook catalog and return the path to a 3-line TLE file.

    Mirrors `burr.utils.spacetrack.fresh_tles` so the calsats manager can pick a
    provider at runtime without further plumbing.
    """
    logger.info("Downloading fresh TLEs from Spacebook")

    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    os.makedirs(SPACEBOOK_DIR, exist_ok=True)
    tle_file = os.path.join(SPACEBOOK_DIR, f"{date}.tle")
    latest_file = os.path.join(SPACEBOOK_DIR, "latest.tle")

    if os.path.exists(tle_file):
        catalog = TLECatalog.from_file(tle_file, catalog_type="SPACEBOOK")
        if catalog.tles:
            logger.info(f"Already loaded Spacebook TLEs for {date}, {len(catalog.tles)} satellites in {tle_file}")
            catalog.to_file(latest_file)
            return tle_file

    try:
        resp = requests.get(SPACEBOOK_URL, timeout=SPACEBOOK_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        if os.path.exists(latest_file):
            catalog = TLECatalog.from_file(latest_file, catalog_type="SPACEBOOK")
            logger.warning(f"Spacebook fetch failed ({e}); falling back to {len(catalog.tles)} cached TLEs in {latest_file}")
            return latest_file
        raise RuntimeError(f"Spacebook fetch failed and no cached TLE file available: {e}") from e

    triples = _parse_two_line_stream(resp.text)
    if not triples:
        if os.path.exists(latest_file):
            logger.warning("Spacebook returned no parseable TLEs; falling back to cached file")
            return latest_file
        raise RuntimeError("Spacebook returned no parseable TLEs and no fallback file available")

    with open(tle_file, "w") as fp:
        for line0, line1, line2 in triples:
            fp.write(f"{line0}\n{line1}\n{line2}\n")

    catalog = TLECatalog.from_file(tle_file, catalog_type="SPACEBOOK")
    logger.info(f"Download complete: {len(catalog.tles)} satellites in {tle_file}")
    catalog.to_file(latest_file)
    return tle_file
