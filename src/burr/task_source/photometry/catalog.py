from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, NamedTuple
from dataclasses import dataclass
from statistics import mean, stdev


@dataclass
class CatalogStats:
    """Statistics for a catalog or subset of stars."""

    total_stars: int
    magnitude_range: Dict[str, float]  # min, max, mean, std
    color_indices: Dict[str, Dict[str, float]]  # color -> {min, max, mean, std}
    coordinate_coverage: Dict[str, tuple[float, float]]  # ra_range, dec_range


class Star(NamedTuple):
    """Representation of a standard star with all its properties."""

    name: str
    vmag: float
    b_v: float
    u_b: float
    v_r: float
    r_i: float
    nobs: int
    raj2000: str
    dej2000: str
    pmra: float
    pmde: float
    simbad_name: str
    ra_deg: float
    dec_deg: float


def load_catalog(path: Path) -> List[Star]:
    """
    Load a VizieR astronomical catalog file.

    This function reads VizieR catalog files with the standard format including
    extensive headers and fixed-width columns.

    Parameters
    ----------
    path : Path
        Path to the catalog file

    Returns
    -------
    List[Star]
        List of Star objects containing the catalog data
    """
    # Read the file and find where the data starts
    with open(path, "r") as f:
        lines = f.readlines()

    # Find the start of data (after the dashes line)
    data_start_idx = None
    column_info = {}

    for i, line in enumerate(lines):
        # Skip comment lines
        if line.startswith("#"):
            # Extract column information from header
            if "Details of Columns:" in line:
                continue
            elif line.strip().startswith("Name") and "(a11)" in line:
                # This is a column definition line
                parts = line.strip().split()
                if len(parts) >= 3:
                    col_name = parts[0]
                    col_format = parts[1].strip("()")
                    column_info[col_name] = col_format
            continue

        # Look for the dashes line that separates header from data
        if "-------" in line:
            data_start_idx = i + 1
            break

    if data_start_idx is None:
        raise ValueError("Could not find data start marker in catalog file")

    # Define column specifications based on the format
    colspecs = [
        (0, 11),  # Name
        (12, 18),  # Vmag
        (19, 25),  # B-V
        (26, 32),  # U-B
        (33, 39),  # V-R
        (40, 46),  # R-I
        (47, 50),  # Nobs
        (51, 63),  # RAJ2000
        (64, 76),  # DEJ2000
        (77, 85),  # pmRA
        (86, 94),  # pmDE
        (95, 115),  # SimbadName
    ]

    # Read the data portion
    data_lines = lines[data_start_idx:]
    catalog_data = []

    for line in data_lines:
        if line.strip() == "" or line.startswith("#"):
            continue

        # Extract values using column specifications
        name = line[0:11].strip()
        vmag = _parse_float(line[12:18])
        b_v = _parse_float(line[19:25])
        u_b = _parse_float(line[26:32])
        v_r = _parse_float(line[33:39])
        r_i = _parse_float(line[40:46])
        nobs = _parse_int(line[47:50])
        raj2000 = line[51:63].strip()
        dej2000 = line[64:76].strip()
        pmra = _parse_float(line[77:85])
        pmde = _parse_float(line[86:94])
        simbad_name = line[95:115].strip()

        # Convert coordinates to decimal degrees
        ra_deg = _ra_to_degrees(raj2000)
        dec_deg = _dec_to_degrees(dej2000)

        # Create Star object
        star = Star(
            name=name,
            vmag=vmag,
            b_v=b_v,
            u_b=u_b,
            v_r=v_r,
            r_i=r_i,
            nobs=nobs,
            raj2000=raj2000,
            dej2000=dej2000,
            pmra=pmra,
            pmde=pmde,
            simbad_name=simbad_name,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
        )
        catalog_data.append(star)

    return catalog_data


def _parse_float(value: str) -> float:
    """Parse a string to float, returning NaN for invalid values."""
    value = value.strip()
    if value == "" or value == "---":
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def _parse_int(value: str) -> int:
    """Parse a string to int, returning 0 for invalid values."""
    value = value.strip()
    if value == "" or value == "---":
        return 0
    try:
        return int(value)
    except ValueError:
        return 0


def _ra_to_degrees(ra_str: str) -> float:
    """Convert RA from 'hh mm ss.s' format to decimal degrees."""
    if not ra_str or ra_str == "":
        return np.nan
    try:
        parts = ra_str.split()
        if len(parts) >= 3:
            h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
            return (h + m / 60 + s / 3600) * 15  # Convert hours to degrees
    except (ValueError, IndexError):
        pass
    return np.nan


def _dec_to_degrees(dec_str: str) -> float:
    """Convert Dec from 'dd mm ss.s' format to decimal degrees."""
    if not dec_str or dec_str == "":
        return np.nan
    try:
        parts = dec_str.split()
        if len(parts) >= 3:
            d, m, s = float(parts[0]), float(parts[1]), float(parts[2])
            sign = -1 if d < 0 else 1
            return d + sign * (m / 60 + s / 3600)
    except (ValueError, IndexError):
        pass
    return np.nan


def filter_catalog(
    stars: List[Star],
    mag_range: Optional[tuple] = None,
    color_range: Optional[Dict[str, tuple]] = None,
    ra_range: Optional[tuple] = None,
    dec_range: Optional[tuple] = None,
) -> List[Star]:
    """
    Filter catalog data based on various criteria.

    Parameters
    ----------
    stars : List[Star]
        List of Star objects
    mag_range : tuple, optional
        (min_mag, max_mag) for V magnitude filtering
    color_range : dict, optional
        Dictionary with color index names as keys and (min, max) tuples as values
        e.g., {'B-V': (0.0, 1.5), 'V-R': (0.2, 1.0)}
    ra_range : tuple, optional
        (min_ra, max_ra) in decimal degrees
    dec_range : tuple, optional
        (min_dec, max_dec) in decimal degrees

    Returns
    -------
    List[Star]
        Filtered list of Star objects
    """
    filtered_stars = stars.copy()

    # Filter by magnitude
    if mag_range is not None:
        min_mag, max_mag = mag_range
        filtered_stars = [star for star in filtered_stars if min_mag <= star.vmag <= max_mag]

    # Filter by colors
    if color_range is not None:
        for color_name, (min_val, max_val) in color_range.items():
            if color_name == "B-V":
                filtered_stars = [star for star in filtered_stars if min_val <= star.b_v <= max_val]
            elif color_name == "U-B":
                filtered_stars = [star for star in filtered_stars if min_val <= star.u_b <= max_val]
            elif color_name == "V-R":
                filtered_stars = [star for star in filtered_stars if min_val <= star.v_r <= max_val]
            elif color_name == "R-I":
                filtered_stars = [star for star in filtered_stars if min_val <= star.r_i <= max_val]

    # Filter by RA
    if ra_range is not None:
        min_ra, max_ra = ra_range
        filtered_stars = [star for star in filtered_stars if min_ra <= star.ra_deg <= max_ra]

    # Filter by Dec
    if dec_range is not None:
        min_dec, max_dec = dec_range
        filtered_stars = [star for star in filtered_stars if min_dec <= star.dec_deg <= max_dec]

    return filtered_stars


def catalog_stats(stars: List[Star]) -> CatalogStats:
    """
    Calculate basic statistics for the catalog.

    Parameters
    ----------
    stars : List[Star]
        List of Star objects

    Returns
    -------
    CatalogStats
        Object containing catalog statistics
    """
    if not stars:
        raise ValueError("Cannot calculate statistics for empty catalog")

    # Helper function to calculate stats for a list of values
    def calc_stats(values: List[float]) -> Dict[str, float]:
        valid_values = [v for v in values if not np.isnan(v)]
        if not valid_values:
            return {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
        return {
            "min": min(valid_values),
            "max": max(valid_values),
            "mean": mean(valid_values),
            "std": stdev(valid_values) if len(valid_values) > 1 else 0.0,
        }

    # Calculate magnitude statistics
    vmags = [star.vmag for star in stars]
    magnitude_range = calc_stats(vmags)

    # Calculate color indices statistics
    color_indices = {
        "B-V": calc_stats([star.b_v for star in stars]),
        "U-B": calc_stats([star.u_b for star in stars]),
        "V-R": calc_stats([star.v_r for star in stars]),
        "R-I": calc_stats([star.r_i for star in stars]),
    }

    # Calculate coordinate coverage
    ra_values = [star.ra_deg for star in stars if not np.isnan(star.ra_deg)]
    dec_values = [star.dec_deg for star in stars if not np.isnan(star.dec_deg)]

    coordinate_coverage = {
        "ra_range": (min(ra_values), max(ra_values)) if ra_values else (np.nan, np.nan),
        "dec_range": (min(dec_values), max(dec_values)) if dec_values else (np.nan, np.nan),
    }

    return CatalogStats(
        total_stars=len(stars),
        magnitude_range=magnitude_range,
        color_indices=color_indices,
        coordinate_coverage=coordinate_coverage,
    )
