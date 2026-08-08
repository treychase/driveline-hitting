import os
from io import StringIO

import certifi
import pandas as pd
import requests

# A request with no timeout waits forever. Behind a firewall that drops packets
# rather than refusing them that means a page that never finishes its first
# paint, so every call here gets a deadline and raises instead.
TIMEOUT = (10, 60)

POI_METRICS_URL = "https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/main/baseball_hitting/data/poi/poi_metrics.csv"
HITTRAX_URL = "https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/main/baseball_hitting/data/poi/hittrax.csv"

def ca_bundle():
    """Which CA bundle to verify downloads against.

    Passing ``verify=`` to requests overrides the environment, which breaks the
    download for anyone sitting behind a proxy that signs its own certificates.
    Honour the standard variables first and fall back to certifi.
    """
    return os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE") or certifi.where()

def _read_csv_url(url):
    """Fetch a CSV over HTTPS and parse it, raising on anything that is not one.

    A proxy login page comes back as a 200 full of HTML, which pandas will
    happily turn into a one column frame of nonsense; checking the status and
    the content type turns that into an error at the point it happens.
    """
    response = requests.get(url, verify=ca_bundle(), timeout=TIMEOUT)
    response.raise_for_status()
    if "html" in response.headers.get("Content-Type", "").lower():
        raise ValueError(
            f"{url} returned HTML rather than CSV - something between here and GitHub is "
            "intercepting the request"
        )
    return pd.read_csv(StringIO(response.text))

def load_poi_metrics():
    """Download the hitting point-of-interest table from the openbiomechanics repo.

    One row per swing, keyed on ``session_swing``, with the biomechanical
    metrics Driveline publishes for each capture.
    """
    return _read_csv_url(POI_METRICS_URL)

def load_hittrax():
    """Download the HitTrax ball flight table, keyed on ``session_swing``.

    Exit velocity, launch angle, spray angle, carry and the point of impact for
    each tracked swing.
    """
    return _read_csv_url(HITTRAX_URL)

def show_missingness(df):
    """Non-null counts for the columns that have gaps, largest first.

    Columns with no missing values are dropped, so an empty result means the
    frame is complete.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    return missing

def create_keyword_dummies(df):
    """Add a 0/1 column per body part, phase, measurement and event keyword.

    Each flag records whether the swing has any non-null value among the columns
    whose names contain that keyword. Raises ``ValueError`` when none of the
    keywords match, which usually means the wrong frame was passed in.
    """
    body_parts = ["pelvis", "torso", "hip", "shoulder", "knee", "wrist", "bat", "hand", "upper_arm", "x_factor", "cog"]
    phases = ["load", "stride", "swing"]
    measurements = ["max", "min"]
    events = ["fm", "fp", "hs"]

    all_keywords = body_parts + phases + measurements + events
    result = df.copy()
    found_any = False

    for kw in all_keywords:
        matching_cols = [c for c in df.columns if kw.lower() in c.lower()]
        if matching_cols:
            result[kw] = df[matching_cols].notna().any(axis=1).astype(int)
            found_any = True

    if not found_any:
        raise ValueError("None of the keywords were found in the dataframe columns")

    return result

def keyword_summary(df, keywords):
    """Count how many columns contain each keyword, with one example each.

    Returns a frame of keyword, count and example_column, ready for
    ``plot_keyword_summary()``.
    """
    rows = []
    for kw in keywords:
        matches = [col for col in df.columns if kw.lower() in col.lower()]
        rows.append({"keyword": kw, "count": len(matches), "example_column": matches[0] if matches else ""})
    return pd.DataFrame(rows)

keywords = [
    "pelvis", "torso", "hip", "shoulder", "knee", "wrist", "bat", "hand_speed",
    "upper_arm", "x_factor", "cog",
    "fm", "fp", "hs", "load", "stride", "swing", "maxhss", "launchpos",
    "max", "min", "angular_velocity", "angle"
]

def plot_keyword_summary(summary_df):
    """Horizontal bar chart of column counts per keyword.

    Kept here for the import path used early in the notebook; the same chart
    lives in ``plot_functions.py``.
    """
    import matplotlib.pyplot as plt
    

    sorted_df = summary_df.sort_values("count", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(5, len(sorted_df) * 0.4)))
    bars = ax.barh(sorted_df["keyword"], sorted_df["count"], color="#2b6ca3")

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2, str(int(width)),
                va="center", fontsize=9)

    ax.set_xlabel("Column Count")
    ax.set_title("Column Counts by Keyword")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()