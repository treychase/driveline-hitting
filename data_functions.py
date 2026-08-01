import certifi
import pandas as pd
from io import StringIO
import requests

def load_poi_metrics():
    url = "https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/main/baseball_hitting/data/poi/poi_metrics.csv"
    response = requests.get(url, verify=certifi.where())
    return pd.read_csv(StringIO(response.text))

def load_hittrax():
    url = "https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/main/baseball_hitting/data/poi/hittrax.csv"
    response = requests.get(url, verify=certifi.where())
    return pd.read_csv(StringIO(response.text))

def show_missingness(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    return missing

def create_keyword_dummies(df):
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