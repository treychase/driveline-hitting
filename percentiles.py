"""Percentile ranks for the biomechanics the exit velocity model runs on.

The POI table gives raw numbers, which say little on their own: a hitter turning
their torso at 950 deg/sec means nothing until you know that puts them in the
82nd percentile of this group. ``build_percentiles()`` ranks every model feature
across the swings the model was fit on and writes the result to
``data/biomech_percentiles.csv``, which is what the dashboard reads.

Percentiles are ranks inside this dataset, not against any wider population, and
they carry no judgement about direction. A high attack angle percentile means
steeper than most of the room, not better than most of the room.

    from percentiles import build_percentiles, save_percentiles

    save_percentiles(build_percentiles())
"""

from pathlib import Path

import numpy as np
import pandas as pd

from data_functions import load_hittrax, load_poi_metrics

DEFAULT_PATH = Path("data") / "biomech_percentiles.csv"

TARGET = "exit_velo_mph_x"

# The model's feature list, with the name and unit to show in the table. Order
# is the order the table renders in: swing mechanics first, then where the ball
# met the bat, then what the pitch was doing. Units follow Driveline's own data
# dictionary, which reports both hand speed and torso velocity in deg/sec.
FEATURES = {
    "hand_speed_mag_swing_max_velo_x": ("Peak hand speed", "deg/s"),
    "torso_angular_velocity_swing_max_x": ("Peak torso rotation speed", "deg/s"),
    "swing_efficiency": ("Swing efficiency", "ratio"),
    "bat_torso_angle_connection_x": ("Early connection", "deg"),
    "attack_angle_contact_x": ("Attack angle", "deg"),
    "poi_x": ("Contact point, across", "in"),
    "poi_y": ("Contact point, height", "in"),
    "pitch_angle": ("Pitch descent angle", "deg"),
}

FEATURE_COLUMNS = list(FEATURES)


def swing_efficiency(df):
    """Bat speed over hand speed, guarding the swings that logged zero hand speed."""
    return np.where(
        df["hand_speed_blast_bat_mph_max_x"] == 0,
        np.nan,
        df["blast_bat_speed_mph_x"] / df["hand_speed_blast_bat_mph_max_x"],
    )


def model_frame(raw_df=None):
    """The swings the model is fit on: POI joined to HitTrax, complete features only."""
    if raw_df is None:
        raw_df = load_poi_metrics().merge(load_hittrax(), on="session_swing", how="inner")
    raw_df = raw_df.copy()
    raw_df["swing_efficiency"] = swing_efficiency(raw_df)
    return raw_df.dropna(subset=FEATURE_COLUMNS + [TARGET])


def build_percentiles(raw_df=None):
    """Rank every model feature across the modelled swings.

    Returns one row per swing: the raw value of each feature alongside its
    percentile, plus exit velocity and its percentile for context.
    """
    model_df = model_frame(raw_df)

    out = pd.DataFrame({"session_swing": model_df["session_swing"].values})
    for column in FEATURE_COLUMNS + [TARGET]:
        values = model_df[column].astype(float).reset_index(drop=True)
        out[column] = values.round(4)
        out[f"{column}_pct"] = (values.rank(pct=True) * 100).round(1)
    return out


def save_percentiles(percentiles, path=DEFAULT_PATH):
    """Write the percentile table to CSV, creating the directory if it is missing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    percentiles.to_csv(path, index=False)
    return path


def load_percentiles(path=DEFAULT_PATH, rebuild=True):
    """Read the percentile table, building and saving it if it is not on disk yet."""
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    if not rebuild:
        raise FileNotFoundError(f"{path} not found - run build_percentiles() first")
    percentiles = build_percentiles()
    save_percentiles(percentiles, path)
    return percentiles


def swing_percentiles(percentiles, session_swing):
    """One tidy row per metric for a single swing, ready to render as a table.

    Columns are named for display: Metric, Value, Unit, Percentile. Returns an
    empty frame with those columns when the swing was never modelled.
    """
    columns = ["Metric", "Value", "Unit", "Percentile"]
    match = percentiles[percentiles["session_swing"] == session_swing]
    if match.empty:
        return pd.DataFrame(columns=columns)

    row = match.iloc[0]
    return pd.DataFrame(
        [
            {
                "Metric": label,
                "Value": float(row[column]),
                "Unit": unit,
                "Percentile": float(row[f"{column}_pct"]),
            }
            for column, (label, unit) in FEATURES.items()
        ],
        columns=columns,
    )


if __name__ == "__main__":
    table = build_percentiles()
    print(f"wrote {len(table)} swings to {save_percentiles(table)}")
