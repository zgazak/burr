from pathlib import Path
import os
import sys

def _get_package_root():
    """Get the root directory of the burr package."""
    if getattr(sys, 'frozen', False):
        # If we're running in a PyInstaller bundle
        return Path(sys._MEIPASS)
    else:
        # Normal Python installation
        return Path(__file__).parent.parent

# Initialize with basic paths first
PACKAGE_ROOT = _get_package_root()
RESOURCES_DIR = PACKAGE_ROOT / "resources"

CONFIG_DIR = RESOURCES_DIR / "config"
DATA_DIR = RESOURCES_DIR / "data"
ASSETS_DIR = RESOURCES_DIR / "assets"
EPHEMERIS_DIR = RESOURCES_DIR / "ephemeris"
ASTROPY_CACHE_DIR = RESOURCES_DIR / "cache" / "astropy"
SPACETRACK_DIR = RESOURCES_DIR / "cache" / "spacetrack"
CACHE_DIR = RESOURCES_DIR / "cache"

# Config overrides
LOCAL_APP_CONFIG_OVERRIDE = CONFIG_DIR / "basic.yaml"
LOCAL_APP_LOCAL_ASTROMETRY_CONFIG_OVERRIDE = CONFIG_DIR / "local-localastrometry.yaml"
DEV_APP_CONFIG_OVERRIDE = CONFIG_DIR / "dev.yaml"
PROD_APP_CONFIG_OVERRIDE = CONFIG_DIR / "prod.yaml"

# App-specific paths
LOG_PATH = CACHE_DIR / "logs" / "app.log"
SENPAI_ASTROMETRY_CONFIG_PATH = RESOURCES_DIR / "astrometry" / "astrometry.cfg"
LANDOLT_STARS_PATH = RESOURCES_DIR / "data" / "photometric_standards_landolt2009table.txt"

TWILIGHT_ANGLES = {"Day": 0, "Civil": -6, "Nautical": -12, "Astronomical": -18}

# Create necessary directories
os.makedirs(ASTROPY_CACHE_DIR, exist_ok=True)
os.makedirs(EPHEMERIS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOG_PATH.parent, exist_ok=True)
os.makedirs(SPACETRACK_DIR, exist_ok=True)

SUPPORTED_CALSATS = [
    "24876",
    "26360",
    "26605",
    "27663",
    "27704",
    "28190",
    "28474",
    "28874",
    "29486",
    "29601",
    "32260",
    "32384",
    "32711",
    "35752",
    "36585",
    "37753",
    "38833",
    "39166",
    "39533",
    "39741",
    "40105",
    "40294",
    "40534",
    "40730",
    "41019",
    "41328",
    "43873",
    "44506",
    "45854",
    "46826",
    "48859",
    "37846",
    "37847",
    "38857",
    "38858",
    "40128",
    "40129",
    "40544",
    "40545",
    "40889",
    "40890",
    "41174",
    "41175",
    "41549",
    "41550",
    "41859",
    "41860",
    "41861",
    "41862",
    "43055",
    "43056",
    "43057",
    "43058",
    "43564",
    "43565",
    "43566",
    "43567",
    "49809",
    "49810",
]
