"""Streamlit front end for the swing dashboard.

    pip install -r requirements.txt
    streamlit run streamlit_app.py

Picking the hitter in the sidebar rather than inside the figure means only one
swing is ever loaded, so the page stays small and the first paint is quick. The
play controls and the frame slider still belong to Plotly and run in the
browser, so scrubbing through a swing does not round trip to the server.

The first run downloads the 400 MB C3D archive into ``data/c3d`` and fits the
exit velocity model. Both are cached, so it only happens once.
"""

import numpy as np
import pandas as pd
import streamlit as st

from c3d_functions import download_c3d, index_swings
from dashboard import prepare_swings, swing_dashboard
from data_functions import load_hittrax, load_poi_metrics

st.set_page_config(page_title="Driveline swing dashboard", page_icon="⚾", layout="wide")

FEATURES = [
    "pitch_angle",
    "bat_torso_angle_connection_x",
    "hand_speed_mag_swing_max_velo_x",
    "swing_efficiency",
    "torso_angular_velocity_swing_max_x",
    "attack_angle_contact_x",
    "poi_x",
    "poi_y",
]
TARGET = "exit_velo_mph_x"


@st.cache_resource(show_spinner="Fetching the C3D archive (400 MB, first run only)")
def get_c3d_dir():
    return download_c3d()


@st.cache_data(show_spinner="Indexing swings")
def get_index():
    return index_swings(get_c3d_dir())


@st.cache_data(show_spinner="Fitting the exit velocity model")
def get_predictions():
    """Out-of-fold exit velocity predictions, mirroring the notebook's model.

    Kept here rather than imported so the app stands on its own; the notebook
    walks through the same fit with the reasoning attached.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold, cross_val_predict

    raw = load_poi_metrics().merge(load_hittrax(), on="session_swing", how="inner")
    raw["swing_efficiency"] = np.where(
        raw["hand_speed_blast_bat_mph_max_x"] == 0,
        np.nan,
        raw["blast_bat_speed_mph_x"] / raw["hand_speed_blast_bat_mph_max_x"],
    )
    model_df = raw.dropna(subset=FEATURES + [TARGET])
    X, y = model_df[FEATURES], model_df[TARGET]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(RandomForestRegressor(random_state=42), X, y, cv=kf)
    return pd.DataFrame(
        {"session_swing": model_df["session_swing"].values, "actual": y.values, "predicted": oof}
    )


@st.cache_data(show_spinner="Reading motion capture")
def get_swing(_index, path):
    """Load and resample one swing. Cached on the file path, not the index."""
    prepared = prepare_swings(_index[_index["path"] == path])
    return prepared[0] if prepared else None


def hitter_label(row):
    level = f" · {row['highest_playing_level']}" if isinstance(
        row["highest_playing_level"], str
    ) else ""
    return f"Hitter {int(row['user']):03d} · {row['side']}HH{level}"


index = get_index()
predictions = get_predictions()
scored = set(predictions["session_swing"].dropna())

st.title("Driveline hitters, swing by swing")
st.caption(
    f"{len(index)} swings from {index['user'].nunique()} hitters, straight off the "
    "OpenBiomechanics C3D files. Markers at 360 Hz, force plates at 1080 Hz."
)

with st.sidebar:
    st.header("Pick a swing")

    only_scored = st.checkbox(
        "Only swings the model scored",
        value=True,
        help="A swing with no Blast bat speed reading has no swing efficiency, so it "
        "never made it into the training set and has nothing to plot on the model panel.",
    )
    pool = index[index["session_swing"].isin(scored)] if only_scored else index

    hitters = pool.drop_duplicates("user").sort_values("exit_velo_mph", ascending=False)
    hitter = st.selectbox(
        "Hitter",
        options=list(hitters["user"]),
        format_func=lambda u: hitter_label(hitters[hitters["user"] == u].iloc[0]),
    )

    swings = pool[pool["user"] == hitter].sort_values("exit_velo_mph", ascending=False)
    path = st.selectbox(
        "Swing",
        options=list(swings["path"]),
        format_func=lambda p: (
            f"{swings.loc[swings['path'] == p, 'exit_velo_mph'].iloc[0]:.1f} mph exit velo"
        ),
    )

    show_model = st.checkbox("Show the model panel", value=True)
    st.divider()
    st.caption(
        "Press play under the 3D view, or drag the frame slider. Both run in the "
        "browser, so scrubbing does not reload the page."
    )

swing = get_swing(index, path)

if swing is None:
    st.error(
        "This trial loses the bat markers partway through, so the reconstructed barrel "
        "is not trustworthy enough to animate. Pick another swing."
    )
    st.stop()

row = swing["row"]
prediction = predictions[predictions["session_swing"] == row["session_swing"]]

columns = st.columns(5)
columns[0].metric("Exit velocity", f"{row['exit_velo_mph']:.1f} mph")
columns[1].metric("Peak bat speed", f"{swing['speed'].max():.1f} mph")
columns[2].metric("Lead leg peak", f"{swing['lead_grf'].max():.0f}% BW")
feet, inches = divmod(int(row["height_in"]), 12)
columns[3].metric("Hitter", f"{feet}'{inches}\" · {int(row['mass_lb'])} lb")
if not prediction.empty:
    predicted = float(prediction["predicted"].iloc[0])
    columns[4].metric(
        "Model predicted",
        f"{predicted:.1f} mph",
        delta=f"{predicted - row['exit_velo_mph']:+.1f} vs actual",
        delta_color="off",
    )
else:
    columns[4].metric("Model predicted", "not scored")

fig = swing_dashboard(
    [swing],
    predictions=predictions if show_model else None,
    title=hitter_label(row),
    show_selector=False,
)
# No width argument: the figure sets no width of its own, so Plotly autosizes it
# to the container on every Streamlit version rather than only the recent ones.
st.plotly_chart(fig)

with st.expander("How this is put together"):
    st.markdown(
        """
These C3D files carry no labelled contact event, so contact is taken as the frame of peak
sweet spot speed. Peak speed measured that way correlates 0.96 with Driveline's own
`bat_speed_mph_max_x` across the 660 swings that join back to the published metrics, with a
median error near 1 mph. Every swing is resampled onto the same clock with zero at contact,
so hitters compare like for like.

The model panel plots a random forest's out-of-fold prediction of exit velocity against what
the ball actually did, one dot per training swing. The cloud is flatter than the parity line
because the model pulls everything toward 90 mph: it can find the hitters who move fast and
rotate hard, but nothing in the feature set records where on the barrel the ball hit.

Twelve of the 687 trials lose the bat markers badly enough to reconstruct a barrel travelling
at several hundred mph. Those are filtered out rather than animated.
        """
    )
