import logging
import os

from burr.core.constants import ASTROPY_CACHE_DIR, EPHEMERIS_DIR

logger = logging.getLogger(__name__)


def setup_astropy():
    """Configure Astropy's data and cache handling."""
    logger.info("Setting up Astropy...")

    # Set cache directory BEFORE importing astropy
    os.environ["XDG_CACHE_HOME"] = str(ASTROPY_CACHE_DIR)
    os.environ["ASTROPY_CACHE_DIR"] = str(ASTROPY_CACHE_DIR)

    # Now safe to import and configure astropy
    from astropy.utils import iers

    # Configure IERS
    iers.conf.auto_download = False
    iers.conf.iers_degraded_accuracy = "warn"

    # Load Burr's local finals2000A.all as the authoritative IERS table for
    # this session. We can't rely on iers_auto_url with auto_download=False —
    # astropy never consults the URL in that mode and falls back to a stale
    # bundled / cached IERS_A table, which trips the "predictive values more
    # than 30 days old" check on any observatory that hasn't been run
    # recently. ensure_ephemeris() refreshes this file on startup.
    iers_file = EPHEMERIS_DIR / "finals2000A.all"
    if iers_file.exists():
        iers.earth_orientation_table.set(iers.IERS_A.open(str(iers_file)))

    logger.info("Astropy setup completed")

