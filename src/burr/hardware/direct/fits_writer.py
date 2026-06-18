"""
FITS metadata and file writing extracted from runner.py.
"""




def generate_fits_filename(
    manager_type: str,
    task_id: str,
    collect_id: str,
    tracking_mode: str,
    frame_num: int,
    total_frames: int,
    exposure_num: int,
    total_exposures: int,
) -> str:
    """Generate a standardized FITS filename.

    Parameters
    ----------
    manager_type : str
        Name of the task source (e.g. 'photometry', 'coverage').
    task_id : str
        Task UUID.
    collect_id : str
        Collection UUID.
    tracking_mode : str
        Tracking mode value string.
    frame_num : int
        Frame number within the exposure.
    total_frames : int
        Total frames per exposure.
    exposure_num : int
        Exposure number in the set.
    total_exposures : int
        Total exposures in the set.

    Returns
    -------
    str
        Formatted FITS filename.
    """
    return (
        f"{manager_type}_"
        f"task{task_id}_"
        f"collect{collect_id}_"
        f"exp{exposure_num}of{total_exposures}_"
        f"frame{frame_num}of{total_frames}_"
        f"{tracking_mode}.fits"
    )


def build_frame_headers(
    task_id: str,
    collect_id: str,
    frame_num: int,
    total_frames: int,
    exposure_num: int,
    total_exposures: int,
    source_name: str,
) -> dict:
    """Build standard FITS header entries for a frame.

    Returns
    -------
    dict
        FITS header keyword -> (value, comment) pairs.
    """
    return {
        "TASKID": (task_id, "Task identifier (UUID)"),
        "COLLECTID": (collect_id, "Collection identifier (UUID)"),
        "FRAMENUM": (frame_num, "Frame number in sequence"),
        "NFRAMES": (total_frames, "Total frames in sequence"),
        "EXPNUM": (exposure_num, "Exposure number in set"),
        "NEXPS": (total_exposures, "Total exposures in set"),
        "MANAGER": (source_name, "Task source name"),
    }
