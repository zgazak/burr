"""Twilight flat readback and exposure control (SK mode).

SK's data graph writes each camera's frame as an empty primary HDU plus a
CompImageHDU, named by the controller's FileNameTemplate rather than burr's,
and only after the collect task has returned. These tests build frames in
that shape for several cameras and check the manager finds, measures and
counts them per camera, and that SK flat identity headers (IMAGETYP, UNITID,
FILTER) are validated and normalised.
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from burr.core.config import TwilightFlatsConfig
from burr.task_source.flats.manager import FlatFieldManager, load_image_hdu


def _write_sk_frame(
    path,
    level,
    *,
    unit="C0",
    filter_name="Lum",
    exptime=1.0,
    imagetyp="flat",
    date_obs=None,
    omit=(),
):
    """Write a frame the way SK's compress_fits + write_file do.

    ``unit``, ``filter_name`` and ``imagetyp`` set the UNITID, FILTER and
    IMAGETYP cards; passing ``None`` writes the card with an undefined value.
    Card names in ``omit`` are left out of the header entirely.
    """
    rng = np.random.default_rng(0)
    data = rng.normal(level, 50, size=(400, 600)).clip(0, 65520).astype(np.int32)
    cards = {
        "IMAGETYP": imagetyp,
        "EXPTIME": exptime,
        "UNITID": unit,
        "FILTER": filter_name,
        "DATE-OBS": (date_obs or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%S.%f"),
    }
    header = fits.Header()
    for key, value in cards.items():
        if key not in omit:
            header[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    hdus = [fits.PrimaryHDU(), fits.CompImageHDU(data=data, header=header)]
    fits.HDUList(hdus).writeto(path)


def _manager(tmp_path, *, local_midnight, **overrides):
    config = TwilightFlatsConfig(
        windows=["nautical_twilight"],
        n_frames=2,
        min_counts=40000,
        max_counts=50000,
        frame_settle_seconds=3.0,
        **overrides,
    )
    commands = []
    run = SimpleNamespace(
        config=SimpleNamespace(schedule=SimpleNamespace(twilight_flats=config)),
        executed_commands=commands,
        output_directory=tmp_path / "processed",
        lighting_schedule=SimpleNamespace(local_midnight=local_midnight),
        log_command=lambda command, result=None, metadata=None, error=None: commands.append(
            {"command": command, "result": result, "metadata": metadata}
        ),
    )
    return FlatFieldManager(run, sk_data_root=tmp_path / "raw")


def test_load_image_hdu_reads_compressed_hdu(tmp_path):
    path = tmp_path / "frame.fits"
    _write_sk_frame(path, 45000, unit="C0")
    data, header = load_image_hdu(path)
    assert data.shape == (400, 600)
    assert header["UNITID"] == "C0"


def test_readback_counts_each_camera_and_ignores_non_flats(tmp_path):
    dusk_midnight = datetime.now(UTC) + timedelta(hours=4)
    manager = _manager(tmp_path, local_midnight=dusk_midnight)
    request = manager.generate_request(task_id="t1")
    raw = tmp_path / "raw" / "Huntsman"

    # Names follow the controller template, not burr's *twilight_flats* one.
    _write_sk_frame(raw / "C0" / "uuid-a_0.fits", 45000, unit="C0")
    _write_sk_frame(raw / "C1" / "uuid-a_0.fits", 20000, unit="C1")  # too faint
    _write_sk_frame(raw / "C2" / "uuid-x_0.fits", 45000, unit="C2", imagetyp="light")  # otto frame
    manager.on_request_completed(request, duration_seconds=1.0)

    assert set(manager.cameras) == {"C0", "C1"}
    assert manager.cameras["C0"].good_frames == 1
    assert manager.cameras["C1"].good_frames == 0
    assert (tmp_path / "processed" / "data" / "flats" / "C0" / "flat_001.fits").exists()
    assert manager.good_frames == 0  # collection waits for the camera furthest behind
    assert not manager.is_collection_complete()

    # C1 at 20000 ADU wants ~2.25x; C0 is on target -> shared exposure in between, > 1s at dusk.
    assert manager.cameras["C1"].next_exposure > 2.0
    assert 1.0 < manager._shared_exposure() < manager.cameras["C1"].next_exposure


def test_sky_factor_shortens_at_dawn_and_lengthens_at_dusk(tmp_path):
    taken = datetime.now(UTC) - timedelta(seconds=180)  # one doubling time ago
    dusk = _manager(tmp_path, local_midnight=taken + timedelta(hours=3))
    dawn = _manager(tmp_path, local_midnight=taken - timedelta(hours=3))
    on_target = dusk.target_counts
    assert np.isclose(dusk._next_exposure_for(1.0, on_target, taken), 2.0, rtol=0.02)
    assert np.isclose(dawn._next_exposure_for(1.0, on_target, taken), 0.5, rtol=0.02)

    # Long gaps within one twilight clip to the limits instead of overflowing.
    long_ago = datetime.now(UTC) - timedelta(hours=2)
    late_dusk = _manager(tmp_path, local_midnight=datetime.now(UTC) + timedelta(hours=1))
    early_dawn = _manager(tmp_path, local_midnight=long_ago - timedelta(hours=1))
    assert late_dusk._next_exposure_for(1.0, on_target, long_ago) == late_dusk.maximum_exposure
    assert early_dawn._next_exposure_for(1.0, on_target, long_ago) == early_dawn.minimum_exposure

    # Last evening's frame seen at dawn restarts the sequence at the minimum.
    evening = datetime.now(UTC) - timedelta(hours=9)
    crossed = _manager(tmp_path, local_midnight=evening + timedelta(hours=4))
    assert crossed._next_exposure_for(20.0, 1000.0, evening) == crossed.minimum_exposure


def test_resume_restores_per_camera_counts(tmp_path):
    midnight = datetime.now(UTC) + timedelta(hours=4)
    first = _manager(tmp_path, local_midnight=midnight)
    request = first.generate_request(task_id="t1")
    _write_sk_frame(tmp_path / "raw" / "C0" / "a_0.fits", 45000, unit="C0")
    first.on_request_completed(request, duration_seconds=1.0)

    resumed = _manager(tmp_path, local_midnight=midnight)
    resumed.run.executed_commands.extend(first.run.executed_commands)
    resumed.cameras = resumed._get_camera_state_from_history()
    assert resumed.cameras["C0"].good_frames == 1


def test_saturation_autodetect_for_12bit_in_int32(tmp_path):
    manager = _manager(tmp_path, local_midnight=datetime.now(UTC))
    data = np.full((100, 100), 45000, dtype=np.int32)
    data[0, :10] = 65520  # 0.1% saturated at the ASI183's scaled full well
    pct, threshold = manager._calculate_saturation_percentage(data)
    assert threshold < 65520
    assert np.isclose(pct, 0.1)


def _image_header(path):
    """Header of the image HDU (HDU 1 in an SK-format file)."""
    return load_image_hdu(path)[1]


def _saved(manager):
    """flat_field_saved history entries."""
    commands = manager.run.executed_commands
    return [cmd for cmd in commands if cmd["command"] == "flat_field_saved"]


def _processed_fits(tmp_path):
    flats = tmp_path / "processed" / "data" / "flats"
    return sorted(flats.rglob("*.fits")) if flats.exists() else []


def test_sk_flat_metadata_is_normalised_and_preserved(tmp_path):
    manager = _manager(tmp_path, local_midnight=datetime.now(UTC) + timedelta(hours=4))
    source = tmp_path / "raw" / "Huntsman" / "C1" / "uuid-a_0.fits"
    _write_sk_frame(
        source, 45000, unit=" C1 ", filter_name=" Lum ", imagetyp=" flat ", exptime=1.5
    )
    source_bytes = source.read_bytes()

    assert manager.process_exposure(str(source), device_control=None) is True

    output = tmp_path / "processed" / "data" / "flats" / "C1" / "flat_001.fits"
    assert _processed_fits(tmp_path) == [output]
    header = _image_header(output)
    assert header["IMAGETYP"] == "FLAT"
    assert header["UNITID"] == "C1"
    assert header["FILTER"] == "Lum"
    assert header["EXPTIME"] == 1.5

    assert source.read_bytes() == source_bytes

    (entry,) = _saved(manager)
    assert entry["metadata"]["camera"] == "C1"
    assert entry["metadata"]["filter"] == "Lum"
    assert set(manager.cameras) == {"C1"}


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param({"omit": {"UNITID"}}, id="missing-unitid"),
        pytest.param({"unit": None}, id="unitid-none"),
        pytest.param({"omit": {"FILTER"}}, id="missing-filter"),
        pytest.param({"filter_name": None}, id="filter-none"),
        pytest.param({"imagetyp": "light"}, id="imagetyp-light"),
        pytest.param({"omit": {"IMAGETYP"}}, id="missing-imagetyp"),
    ],
)
def test_sk_flat_with_invalid_identity_headers_is_rejected(tmp_path, frame):
    manager = _manager(tmp_path, local_midnight=datetime.now(UTC) + timedelta(hours=4))
    source = tmp_path / "raw" / "Huntsman" / "C1" / "uuid-a_0.fits"
    _write_sk_frame(source, 45000, **frame)

    assert manager.process_exposure(str(source), device_control=None) is False
    assert manager.cameras == {}
    assert _processed_fits(tmp_path) == []
    assert _saved(manager) == []
