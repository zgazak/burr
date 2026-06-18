import logging
import urllib.request
import ssl
from functools import lru_cache
from pathlib import Path
from typing import Dict, Literal

from tqdm import tqdm

from burr.core.constants import EPHEMERIS_DIR

logger = logging.getLogger(__name__)

# Define valid ephemeris file names
EphemerisFileName = Literal["de430.bsp", "finals2000A.all"]

EPHEMERIS_FILE = "de432s.bsp"

REQUIRED_FILES: Dict[EphemerisFileName, str] = {
    EPHEMERIS_FILE: f"https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/{EPHEMERIS_FILE}",
    "finals2000A.all": "https://datacenter.iers.org/products/eop/rapid/standard/finals2000A.all",
}


def _download_file(url: str, destination: Path, desc: str) -> None:
    """Download a file with progress bar."""
    try:
        # Create an unverified SSL context
        ssl_context = ssl._create_unverified_context()
        
        # Use the context in the request
        response = urllib.request.urlopen(url, context=ssl_context)
        total_size = int(response.headers.get("content-length", 0))

        with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
            with open(destination, "wb") as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
    except Exception as e:
        # If download fails, ensure we don't leave a partial file
        if destination.exists():
            destination.unlink()
        raise RuntimeError(f"Failed to download file from {url}: {str(e)}")


@lru_cache
def get_ephemeris_file(filename: EphemerisFileName) -> Path:
    """Get path to an ephemeris file, ensuring it exists."""
    if filename not in REQUIRED_FILES:
        raise ValueError(f"Unknown ephemeris file: {filename}")

    dest = EPHEMERIS_DIR / filename
    if not dest.exists():
        logger.info(f"Downloading required file: {filename}")
        try:
            _download_file(REQUIRED_FILES[filename], dest, f"Downloading {filename}")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            raise

    return dest


def ensure_ephemeris() -> None:
    """Ensure all required ephemeris files exist."""
    logger.info("Ensuring ephemeris files exist...")
    for filename in REQUIRED_FILES:
        try:
            eph = get_ephemeris_file(filename)
            logger.info(f"Ephemeris file {filename} exists at {eph}")
        except Exception as e:
            logger.error(f"Failed to ensure ephemeris file {filename}: {e}")
            raise