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

def keyword_summary(data_dict, keywords):
    rows = []
    for kw in keywords:
        matches = data_dict[data_dict['column'].str.contains(kw, case=False, na=False)]
        count = len(matches)
        example_desc = matches['description'].dropna().iloc[0] if not matches['description'].dropna().empty else ""
        rows.append({"keyword": kw, "count": count, "example_description": example_desc})
    return pd.DataFrame(rows)

keywords = [
    "pelvis", "torso", "hip", "shoulder", "knee", "wrist", "bat", "hand_speed",
    "upper_arm", "x_factor", "cog",
    "fm", "fp", "hs", "load", "stride", "swing", "maxhss", "launchpos",
    "max", "min", "angular_velocity", "angle"
]

def plot_keyword_summary(summary_df):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.bar(summary_df['keyword'], summary_df['count'])
    plt.xlabel("Keyword")
    plt.ylabel("Column Count")
    plt.title("Column Counts by Keyword")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()