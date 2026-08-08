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

from pathlib import Path

import pandas as pd
import streamlit as st

from c3d_functions import DEFAULT_DATA_DIR, download_c3d, index_swings
from dashboard import BAT_COLOR, prepare_swings, swing_dashboard
from percentiles import FEATURE_COLUMNS, TARGET, load_percentiles, model_frame, swing_percentiles

st.set_page_config(page_title="Driveline swing dashboard", page_icon="⚾", layout="wide")


C3D_DIR = DEFAULT_DATA_DIR / "c3d"


def c3d_present(directory=C3D_DIR):
    """Whether the motion capture files have been unpacked yet."""
    return Path(directory).exists() and any(Path(directory).glob("*/*.c3d"))


@st.cache_resource(show_spinner="Fetching the C3D archive (400 MB, first run only)")
def get_c3d_dir():
    """Path to the unpacked C3D files, downloading the archive on first call."""
    return download_c3d()


@st.cache_data(show_spinner="Indexing swings")
def get_index():
    """One row per swing C3D, joined to the published metadata where it matches."""
    return index_swings(get_c3d_dir())


@st.cache_data(show_spinner="Fitting the exit velocity model")
def get_predictions():
    """Out-of-fold exit velocity predictions, mirroring the notebook's model.

    Kept here rather than imported so the app stands on its own; the notebook
    walks through the same fit with the reasoning attached.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold, cross_val_predict

    model_df = model_frame()
    X, y = model_df[FEATURE_COLUMNS], model_df[TARGET]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(RandomForestRegressor(random_state=42), X, y, cv=kf)
    return pd.DataFrame(
        {"session_swing": model_df["session_swing"].values, "actual": y.values, "predicted": oof}
    )


@st.cache_data(show_spinner="Ranking the biomechanics")
def get_percentiles():
    """The percentile table from percentiles.py, built on first run if missing."""
    return load_percentiles()


@st.cache_data(show_spinner="Reading motion capture")
def get_swing(_index, path):
    """Load and resample one swing. Cached on the file path, not the index."""
    prepared = prepare_swings(_index[_index["path"] == path])
    return prepared[0] if prepared else None


def hitter_label(row):
    """Hitter number, side and playing level, for the selector and the figure title."""
    level = f" · {row['highest_playing_level']}" if isinstance(
        row["highest_playing_level"], str
    ) else ""
    return f"Hitter {int(row['user']):03d} · {row['side']}HH{level}"


# The header goes up before anything touches the data, so a download or a failed
# load leaves a page that still looks like the app rather than a bare traceback.
st.title("Driveline hitters, swing by swing")
header = st.empty()

if not c3d_present():
    header.caption("Motion capture files not found yet.")
    with st.container(border=True):
        st.subheader("One-time setup")
        st.markdown(
            f"""
The animation reads Driveline's raw C3D motion capture, which is not stored in this repo. It
ships as an asset on the openbiomechanics `dataset-v1` release: about **400 MB**, 687 swings
from 97 hitters, unpacked into `{C3D_DIR}`.

Fetch it once and this page will not ask again. From a terminal, if you would rather watch the
progress there:

```bash
python -c "from c3d_functions import download_c3d; download_c3d()"
```
            """
        )
        if st.button("Download the C3D archive", type="primary"):
            try:
                with st.spinner("Downloading and unpacking, a few minutes on a decent connection"):
                    get_c3d_dir()
                st.rerun()
            except Exception as error:  # network, TLS, disk - all land here
                st.error(f"The download failed: {error}")
                st.caption(
                    "Behind a proxy that signs its own certificates, point REQUESTS_CA_BUNDLE at "
                    "your CA file before starting the app. Otherwise check the connection and "
                    "that there is 500 MB free."
                )
    st.stop()

try:
    index = get_index()
    predictions = get_predictions()
    percentiles = get_percentiles()
except Exception as error:
    header.empty()
    st.error(f"Could not load the swing data: {type(error).__name__}: {error}")
    st.caption(
        f"The published metrics tables are fetched from GitHub at startup, so this usually means "
        f"no connection. The motion capture itself is already unpacked in `{C3D_DIR}`."
    )
    st.stop()

scored = set(predictions["session_swing"].dropna())
header.caption(
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
columns[0].metric("Exit velocity", f"{row['exit_velo_mph']:.1f} mph", border=True)
columns[1].metric("Peak bat speed", f"{swing['speed'].max():.1f} mph", border=True)
columns[2].metric("Lead leg peak", f"{swing['lead_grf'].max():.0f}% BW", border=True)
feet, inches = divmod(int(row["height_in"]), 12)
columns[3].metric("Hitter", f"{feet}'{inches}\" · {int(row['mass_lb'])} lb", border=True)
if not prediction.empty:
    predicted = float(prediction["predicted"].iloc[0])
    miss = predicted - row["exit_velo_mph"]
    # The arrow reads as direction of the miss, not as good or bad, so the
    # colour stays neutral: up means the model over-called this swing.
    columns[4].metric(
        "Model predicted",
        f"{predicted:.1f} mph",
        delta=f"{miss:+.1f} vs actual",
        delta_color="off",
        border=True,
    )
else:
    columns[4].metric("Model predicted", "not scored", border=True)

with st.container(border=True):
    fig = swing_dashboard(
        [swing],
        predictions=predictions if show_model else None,
        title=hitter_label(row),
        show_selector=False,
    )
    # No width argument: the figure sets no width of its own, so Plotly autosizes
    # it to the container on every Streamlit version, not only the recent ones.
    st.plotly_chart(fig)

with st.container(border=True):
    st.subheader("Biomechanics percentiles")
    st.caption(
        "Where this swing ranks against the 581 swings the model was fit on. Rank inside "
        "this group, not against any wider population, and no metric here has a good end: "
        "a high attack angle percentile means steeper than most of the room, not better."
    )

    table = swing_percentiles(percentiles, row["session_swing"])
    if table.empty:
        st.info("This swing is not in the modelled set, so it has nothing to rank against.")
    else:
        st.dataframe(
            table,
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn(
                    "Metric", width="medium", help="The model's input features"
                ),
                "Value": st.column_config.NumberColumn(
                    "Measured", format="%.2f", help="This swing's raw reading"
                ),
                "Unit": st.column_config.TextColumn("Unit", width="small"),
                "Percentile": st.column_config.ProgressColumn(
                    "Percentile in this dataset",
                    format="%.0f",
                    min_value=0,
                    max_value=100,
                    width="large",
                    color=BAT_COLOR,
                ),
            },
        )

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
