"""Helpers for the raw hitting C3D files from Driveline's OpenBiomechanics project.

The C3Ds are not stored in the openbiomechanics git repo - they ship as a zipped
asset on the ``dataset-v1`` release. ``download_c3d()`` pulls that asset down and
unpacks it into ``data/c3d`` (about 400 MB, 687 swings from 97 hitters).

Marker data is 360 Hz, in meters, in the lab frame described in the OBP hitting
README: +x runs from home plate toward the mound, +y toward the right handed
batter's box, +z up. Four force plates sit under the batter's box and are
sampled at 1080 Hz.
"""

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# One definition of which certificates to trust and how long to wait, shared
# with the metrics loaders.
from data_functions import TIMEOUT, _read_csv_url, ca_bundle

C3D_ZIP_URL = (
    "https://github.com/drivelineresearch/openbiomechanics/releases/download/"
    "dataset-v1/hitting_c3d.zip"
)
METADATA_URL = (
    "https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/main/"
    "baseball_hitting/data/metadata.csv"
)

DEFAULT_DATA_DIR = Path("data")
MPS_TO_MPH = 2.23694
LB_TO_N = 4.44822

# Marker1 sits just above the knob, Marker2/Marker3 straddle the end of the barrel.
BAT_KNOB = "Marker1"
BAT_BARREL = ("Marker2", "Marker3")

# Fraction of the knob-to-barrel span used as the sweet spot. 0.75 was picked by
# comparing peak speed at that point against Driveline's own bat_speed_mph_max_x:
# median error across matched swings is under 1 mph.
SWEET_SPOT_FRACTION = 0.75

# Force plates 1 and 2 are in front (toward the mound), 3 and 4 are in back.
FRONT_PLATES = (1, 2)
REAR_PLATES = (3, 4)

# Joint centers built by averaging markers. Anything missing from a file is
# dropped and the segment falls back to whichever marker is present.
VIRTUAL_POINTS = {
    "head_center": ("LFHD", "RFHD", "LBHD", "RBHD"),
    "pelvis_center": ("LASI", "RASI", "LPSI", "RPSI"),
    "l_hip": ("LASI", "LPSI"),
    "r_hip": ("RASI", "RPSI"),
    "l_elbow": ("LELB", "LMELB"),
    "r_elbow": ("RELB", "RMELB"),
    "l_wrist": ("LWRA", "LWRB"),
    "r_wrist": ("RWRA", "RWRB"),
    "l_knee": ("LKNE", "LMKNE"),
    "r_knee": ("RKNE", "RMKNE"),
    "l_ankle": ("LANK", "LMANK"),
    "r_ankle": ("RANK", "RMANK"),
    "hands": ("LWRA", "LWRB", "RWRA", "RWRB"),
}

# Each entry is one polyline through the marker/virtual-point names.
SEGMENTS = [
    ["LFHD", "RFHD", "RBHD", "LBHD", "LFHD"],
    ["head_center", "C7"],
    ["LSHO", "C7", "RSHO"],
    ["LSHO", "CLAV", "RSHO"],
    ["C7", "T10", "pelvis_center"],
    ["CLAV", "STRN", "pelvis_center"],
    ["LASI", "RASI", "RPSI", "LPSI", "LASI"],
    ["LSHO", "l_elbow", "l_wrist", "LFIN"],
    ["RSHO", "r_elbow", "r_wrist", "RFIN"],
    ["l_hip", "l_knee", "l_ankle", "LHEE", "LTOE"],
    ["r_hip", "r_knee", "r_ankle", "RHEE", "RTOE"],
]

_FILENAME = re.compile(
    r"^(?P<user>\d+)_(?P<session>\d+)_(?P<height_in>\d+)_(?P<mass_lb>\d+)_"
    r"(?P<side>[LR])_(?P<swing_no>\d+)_(?P<exit_velo>\d+)$"
)


# ---------------------------------------------------------------- downloading


def download_c3d(data_dir=DEFAULT_DATA_DIR, force=False):
    """Fetch and unpack the hitting C3D release asset. Returns the c3d directory."""
    data_dir = Path(data_dir)
    c3d_dir = data_dir / "c3d"
    if c3d_dir.exists() and any(c3d_dir.glob("*/*.c3d")) and not force:
        return c3d_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "hitting_c3d.zip"
    with requests.get(C3D_ZIP_URL, stream=True, verify=ca_bundle(), timeout=TIMEOUT) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_dir)
    zip_path.unlink()
    return c3d_dir


def load_metadata():
    """Session level metadata (age, playing level, bat spec, bat speed) from OBP."""
    return _read_csv_url(METADATA_URL)


# ------------------------------------------------------------------ indexing


def _parse_filename(path):
    """Pull hitter, session, height, weight, side, swing number and exit velo from a path.

    Returns ``None`` for anything that does not match the naming convention,
    which is how the static model files get skipped.
    """
    match = _FILENAME.match(Path(path).stem)
    if match is None:
        return None
    parts = match.groupdict()
    return {
        "path": str(path),
        "user": int(parts["user"]),
        "session": int(parts["session"]),
        "height_in": int(parts["height_in"]),
        "mass_lb": int(parts["mass_lb"]),
        "side": parts["side"],
        "swing_no": int(parts["swing_no"]),
        # Exit velocity is encoded with the decimal point stripped: 883 -> 88.3.
        "exit_velo_mph": int(parts["exit_velo"]) / 10,
    }


def _match_session_swing(c3d_evs, meta_evs, tol=0.051):
    """Map C3D rows onto metadata rows within a session.

    Both lists are in swing order and the metadata is a subset of what was
    captured, so a single forward pass on exit velocity lines them up.
    """
    pairs = {}
    i = j = 0
    while i < len(c3d_evs) and j < len(meta_evs):
        if abs(c3d_evs[i] - meta_evs[j]) <= tol:
            pairs[i] = j
            i += 1
            j += 1
        else:
            i += 1
    return pairs


def index_swings(c3d_dir=None, metadata=None):
    """One row per swing C3D, with the session_swing key joined on where possible.

    The static model files (``*_model.c3d``) are skipped. ``session_swing`` is
    recovered by lining up exit velocities within a session, which resolves the
    687 C3D swings against the 677 rows of published metadata.
    """
    c3d_dir = Path(c3d_dir) if c3d_dir is not None else DEFAULT_DATA_DIR / "c3d"
    rows = [
        parsed
        for path in sorted(c3d_dir.glob("*/*.c3d"))
        if (parsed := _parse_filename(path)) is not None
    ]
    if not rows:
        raise FileNotFoundError(
            f"no swing C3D files under {c3d_dir} - run download_c3d() first"
        )
    index = pd.DataFrame(rows).sort_values(["session", "swing_no"]).reset_index(drop=True)

    if metadata is None:
        metadata = load_metadata()
    metadata = metadata.copy()
    metadata["swing_idx"] = metadata["session_swing"].str.split("_").str[1].astype(int)
    metadata = metadata.sort_values(["session", "swing_idx"])

    index["session_swing"] = pd.NA
    for session, group in index.groupby("session"):
        meta_group = metadata[metadata["session"] == session]
        if meta_group.empty:
            continue
        pairs = _match_session_swing(
            list(group["exit_velo_mph"]),
            list(meta_group["exit_velo_mph_x"].fillna(-999)),
        )
        for i, j in pairs.items():
            index.loc[group.index[i], "session_swing"] = meta_group["session_swing"].iloc[j]

    extra = [
        "session_swing",
        "athlete_age",
        "highest_playing_level",
        "bat_weight_oz",
        "bat_length_in",
        "bat_speed_mph_max_x",
        "blast_bat_speed_mph_x",
    ]
    extra = [c for c in extra if c in metadata.columns]
    return index.merge(metadata[extra], on="session_swing", how="left")


# ------------------------------------------------------------------- loading


def _despike(points, tol=0.05):
    """Blank single frame marker jumps and fill them by linear interpolation.

    A handful of trials have frames where a bat or body marker teleports and
    comes straight back, which turns into a 500 mph bat if you differentiate it.
    Real motion is smooth at 360 Hz, so a point that sits more than `tol` meters
    off the midpoint of its two neighbors is a glitch, not a swing.
    """
    cleaned = points.astype(float).copy()
    n_frames = cleaned.shape[-1]
    if n_frames < 3:
        return cleaned

    midpoints = (cleaned[..., :-2] + cleaned[..., 2:]) / 2
    deviation = np.linalg.norm(cleaned[..., 1:-1] - midpoints, axis=1)
    bad = np.zeros((cleaned.shape[0], n_frames), dtype=bool)
    bad[:, 1:-1] = deviation > tol
    if not bad.any():
        return cleaned

    frames = np.arange(n_frames)
    for marker in np.flatnonzero(bad.any(axis=1)):
        good = ~bad[marker]
        if good.sum() < 2:
            continue
        for axis in range(3):
            cleaned[marker, axis, bad[marker]] = np.interp(
                frames[bad[marker]], frames[good], cleaned[marker, axis, good]
            )
    return cleaned


def _fix_bat(points, index, tol=0.01):
    """Repair stretches where a bat marker drops out or drifts.

    The bat is a rigid body, so the knob to barrel span is fixed for a given
    bat. Frames where that span moves off its own median are frames where the
    tracking lost a marker - usually as the bat leaves the capture volume after
    contact - and they get filled in from the frames on either side.
    """
    bat_markers = [i for label, i in index.items() if label.startswith("MARKER")]
    knob_i = index.get(BAT_KNOB.upper())
    barrel_i = [index[m.upper()] for m in BAT_BARREL if m.upper() in index]
    if knob_i is None or not barrel_i or len(bat_markers) < 2:
        return points, 0.0

    span = np.linalg.norm(points[barrel_i].mean(axis=0) - points[knob_i], axis=0)
    bad = np.abs(span - np.median(span)) > tol

    # Once tracking is lost for a stretch it rarely comes back cleanly, and the
    # stray frames on the far side of the gap read as a 300 mph bat. Everything
    # from the last long dropout onward is written off and the bat freezes.
    runs = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]))
    starts, ends = runs[::2], runs[1::2]
    long_runs = starts[(ends - starts) > 5]
    if long_runs.size:
        bad[long_runs[-1] :] = True

    if not bad.any() or bad.all():
        return points, float(bad.mean())

    frames = np.arange(points.shape[-1])
    for marker in bat_markers:
        for axis in range(3):
            points[marker, axis, bad] = np.interp(
                frames[bad], frames[~bad], points[marker, axis, ~bad]
            )
    return points, float(bad.mean())


def load_swing(path, metadata_row=None):
    """Read one swing C3D into a dict of arrays.

    Returns marker positions as ``(n_markers, 3, n_frames)`` in meters, the
    force plate channels, and whatever the filename tells us about the hitter.
    """
    import ezc3d

    c3d = ezc3d.c3d(str(path))
    labels = [label.strip() for label in c3d["parameters"]["POINT"]["LABELS"]["value"]]
    index = {label.upper(): i for i, label in enumerate(labels)}
    points, bat_repaired = _fix_bat(
        _despike(np.transpose(c3d["data"]["points"][:3], (1, 0, 2))), index
    )
    rate = float(c3d["parameters"]["POINT"]["RATE"]["value"][0])

    analogs = c3d["data"]["analogs"][0]
    analog_labels = [l.strip() for l in c3d["parameters"]["ANALOG"]["LABELS"]["value"]]
    analog_rate = float(c3d["parameters"]["ANALOG"]["RATE"]["value"][0])

    swing = _parse_filename(path) or {"path": str(path)}
    swing.update(
        {
            "labels": labels,
            "points": points,
            # A handful of sessions label the bat MARKER1 instead of Marker1, so
            # the lookup is keyed on the upper cased label.
            "index": index,
            "rate": rate,
            "n_frames": points.shape[-1],
            # Share of frames where the bat had to be rebuilt. A few percent is
            # the odd dropped frame; anything large means the trial lost the bat.
            "bat_repaired": bat_repaired,
            "analogs": analogs,
            "analog_labels": analog_labels,
            "analog_rate": analog_rate,
        }
    )
    if metadata_row is not None:
        swing["metadata"] = dict(metadata_row)
    return swing


def point(swing, name):
    """Resolve a marker label or virtual joint center to a ``(3, n_frames)`` array."""
    index = swing["index"]
    if name.upper() in index:
        return swing["points"][index[name.upper()]]
    members = [index[m.upper()] for m in VIRTUAL_POINTS.get(name, ()) if m.upper() in index]
    if not members:
        return None
    return swing["points"][members].mean(axis=0)


# ---------------------------------------------------------------- kinematics


def bat_points(swing):
    """Knob, sweet spot and barrel tip positions, each ``(3, n_frames)``."""
    knob = point(swing, BAT_KNOB)
    barrel = np.mean([point(swing, m) for m in BAT_BARREL], axis=0)
    sweet_spot = knob + SWEET_SPOT_FRACTION * (barrel - knob)
    return knob, sweet_spot, barrel


def bat_speed_mph(swing):
    """Sweet spot speed over the whole trial, in mph."""
    _, sweet_spot, _ = bat_points(swing)
    velocity = np.gradient(sweet_spot, axis=1) * swing["rate"]
    return np.linalg.norm(velocity, axis=0) * MPS_TO_MPH


def contact_frame(swing):
    """Frame of peak sweet spot speed, which lands on contact within a frame or two."""
    return int(np.argmax(bat_speed_mph(swing)))


def tracking_ok(swing, max_repaired=0.05, max_bat_mph=95.0):
    """Whether a trial's bat tracking held up well enough to animate.

    Twelve of the 687 swings lose the bat markers badly enough that the
    reconstructed barrel flies off; the fastest bat any hitter in this dataset
    actually swings is a shade under 80 mph, so anything past 95 is tracking
    noise rather than a swing.
    """
    return swing["bat_repaired"] <= max_repaired and bat_speed_mph(swing).max() <= max_bat_mph


def vertical_grf(swing):
    """Lead and rear vertical ground reaction force as a fraction of bodyweight.

    Returned on the marker clock so it lines up with the animation. The plates
    record downward load as negative, and which of the two plates in each pair
    the hitter actually stands on depends on their side, so the loaded one wins.
    """
    labels = swing["analog_labels"]
    analogs = swing["analogs"]

    def _plate_force(plate_numbers):
        best = None
        for number in plate_numbers:
            channel = f"Fz{number}"
            if channel not in labels:
                continue
            force = -analogs[labels.index(channel)]
            if best is None or np.nanmax(force) > np.nanmax(best):
                best = force
        return np.zeros(analogs.shape[-1]) if best is None else np.clip(best, 0, None)

    bodyweight_n = swing.get("mass_lb", 0) * LB_TO_N
    ratio = int(round(swing["analog_rate"] / swing["rate"]))
    n_frames = swing["n_frames"]

    def _to_marker_clock(force):
        trimmed = force[: n_frames * ratio]
        if trimmed.size < n_frames * ratio:
            trimmed = np.pad(trimmed, (0, n_frames * ratio - trimmed.size))
        per_frame = trimmed.reshape(n_frames, ratio).mean(axis=1)
        return per_frame / bodyweight_n if bodyweight_n else per_frame

    return _to_marker_clock(_plate_force(FRONT_PLATES)), _to_marker_clock(
        _plate_force(REAR_PLATES)
    )


def force_plate_corners(path):
    """Corner coordinates of the four plates, as ``(n_plates, 4, 3)`` in meters."""
    import ezc3d

    corners = ezc3d.c3d(str(path))["parameters"]["FORCE_PLATFORM"]["CORNERS"]["value"]
    return np.transpose(corners, (2, 1, 0))
