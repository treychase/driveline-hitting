# API reference

Every module, constant and function in the project, generated from the source by
`docs/build_api_docs.py`. Rerun that script after changing a signature or a
docstring so this file stays honest.

## Contents

- [`data_functions.py`](#data_functionspy) — Loading the published metrics tables.
- [`plot_functions.py`](#plot_functionspy) — The matplotlib charts the notebook draws.
- [`c3d_functions.py`](#c3d_functionspy) — Reading, repairing and measuring the raw motion capture.
- [`percentiles.py`](#percentilespy) — Ranking the model's inputs across the dataset.
- [`dashboard.py`](#dashboardpy) — Building the animated Plotly figure.
- [`streamlit_app.py`](#streamlit_apppy) — The Streamlit front end.
- [`doctor.py`](#doctorpy) — Checking that a machine can run the app.

## `data_functions.py`

*Loading the published metrics tables.*

### Constants

| Name | Value |
| --- | --- |
| `TIMEOUT` | `(10, 60)` |
| `POI_METRICS_URL` | `...` |
| `HITTRAX_URL` | `...` |

### Functions

#### `ca_bundle()`

Which CA bundle to verify downloads against.

Passing ``verify=`` to requests overrides the environment, which breaks the
download for anyone sitting behind a proxy that signs its own certificates.
Honour the standard variables first and fall back to certifi.

#### `load_poi_metrics()`

Download the hitting point-of-interest table from the openbiomechanics repo.

One row per swing, keyed on ``session_swing``, with the biomechanical
metrics Driveline publishes for each capture.

#### `load_hittrax()`

Download the HitTrax ball flight table, keyed on ``session_swing``.

Exit velocity, launch angle, spray angle, carry and the point of impact for
each tracked swing.

#### `show_missingness(df)`

Non-null counts for the columns that have gaps, largest first.

Columns with no missing values are dropped, so an empty result means the
frame is complete.

#### `create_keyword_dummies(df)`

Add a 0/1 column per body part, phase, measurement and event keyword.

Each flag records whether the swing has any non-null value among the columns
whose names contain that keyword. Raises ``ValueError`` when none of the
keywords match, which usually means the wrong frame was passed in.

#### `keyword_summary(df, keywords)`

Count how many columns contain each keyword, with one example each.

Returns a frame of keyword, count and example_column, ready for
``plot_keyword_summary()``.

#### `plot_keyword_summary(summary_df)`

Horizontal bar chart of column counts per keyword.

Kept here for the import path used early in the notebook; the same chart
lives in ``plot_functions.py``.

### Internal helpers

| Function | What it does |
| --- | --- |
| `_read_csv_url(url)` | Fetch a CSV over HTTPS and parse it, raising on anything that is not one. |

## `plot_functions.py`

*The matplotlib charts the notebook draws.*

### Functions

#### `plot_histogram(data, bins=30)`

Histogram per column, with the mean marked and the tails called out.

Accepts a Series or a DataFrame and lays multiple columns out in a grid.
Bars whose centre falls more than two standard deviations from the mean are
filled orange, so outliers are visible without reading the axis.

#### `plot_correlation(data)`

Correlation heatmap over every numeric column, annotated with the values.

Sized from the column count, so a wide frame produces a large figure.

#### `plot_keyword_summary(summary_df)`

Horizontal bar chart of how many columns match each keyword.

Takes the frame ``keyword_summary()`` returns and sorts it ascending, so the
densest parts of the dataset land at the top.

#### `plot_feature_importance(model, feature_names, top_n=15, X=None, y=None, n_repeats=10, random_state=42)`

The fitted model's top_n features by importance, largest at the top.

Estimators exposing ``feature_importances_`` report it directly. Anything
that does not, a Gaussian process or a pipeline wrapping one, falls back to
permutation importance, which needs ``X`` and ``y``: each column is shuffled
in turn and the importance is how many mph of RMSE that costs. The two
scales are not comparable, so the axis label says which one is being drawn.

Both are measured on whatever data is passed in. On a model that fits its
training set closely that flatters every column at once, so read the order
rather than the magnitudes.

#### `plot_regression_diagnostics(model, X, y, preds=None)`

Predicted against actual, and residuals against predicted, side by side.

The dashed line on the left is parity. A residual panel that slopes rather
than sitting flat is the model regressing toward the mean.

Predictions come from the model's own fit unless ``preds`` is passed, which
is how to diagnose out-of-fold predictions instead. Worth doing for any
model that fits its training data closely, where the in-sample panels only
show how well it memorised.

## `c3d_functions.py`

*Reading, repairing and measuring the raw motion capture.*

Helpers for the raw hitting C3D files from Driveline's OpenBiomechanics project.

The C3Ds are not stored in the openbiomechanics git repo - they ship as a zipped
asset on the ``dataset-v1`` release. ``download_c3d()`` pulls that asset down and
unpacks it into ``data/c3d`` (about 400 MB, 687 swings from 97 hitters).

Marker data is 360 Hz, in meters, in the lab frame described in the OBP hitting
README: +x runs from home plate toward the mound, +y toward the right handed
batter's box, +z up. Four force plates sit under the batter's box and are
sampled at 1080 Hz.

### Constants

| Name | Value |
| --- | --- |
| `C3D_ZIP_URL` | `...` |
| `METADATA_URL` | `...` |
| `DEFAULT_DATA_DIR` | `Path('data')` |
| `MPS_TO_MPH` | `2.23694` |
| `LB_TO_N` | `4.44822` |
| `BAT_KNOB` | `'Marker1'` |
| `BAT_BARREL` | `('Marker2', 'Marker3')` |
| `SWEET_SPOT_FRACTION` | `0.75` |
| `FRONT_PLATES` | `(1, 2)` |
| `REAR_PLATES` | `(3, 4)` |
| `VIRTUAL_POINTS` | `...` |
| `SEGMENTS` | `...` |
| `_FILENAME` | `...` |

### Functions

#### `download_c3d(data_dir=DEFAULT_DATA_DIR, force=False)`

Fetch and unpack the hitting C3D release asset. Returns the c3d directory.

#### `load_metadata()`

Session level metadata (age, playing level, bat spec, bat speed) from OBP.

#### `index_swings(c3d_dir=None, metadata=None)`

One row per swing C3D, with the session_swing key joined on where possible.

The static model files (``*_model.c3d``) are skipped. ``session_swing`` is
recovered by lining up exit velocities within a session, which resolves the
687 C3D swings against the 677 rows of published metadata.

#### `load_swing(path, metadata_row=None)`

Read one swing C3D into a dict of arrays.

Returns marker positions as ``(n_markers, 3, n_frames)`` in meters, the
force plate channels, and whatever the filename tells us about the hitter.

#### `point(swing, name)`

Resolve a marker label or virtual joint center to a ``(3, n_frames)`` array.

#### `bat_points(swing)`

Knob, sweet spot and barrel tip positions, each ``(3, n_frames)``.

#### `bat_speed_mph(swing)`

Sweet spot speed over the whole trial, in mph.

#### `contact_frame(swing)`

Frame of peak sweet spot speed, which lands on contact within a frame or two.

#### `tracking_ok(swing, max_repaired=0.05, max_bat_mph=95.0)`

Whether a trial's bat tracking held up well enough to animate.

Twelve of the 687 swings lose the bat markers badly enough that the
reconstructed barrel flies off; the fastest bat any hitter in this dataset
actually swings is a shade under 80 mph, so anything past 95 is tracking
noise rather than a swing.

#### `vertical_grf(swing)`

Lead and rear vertical ground reaction force as a fraction of bodyweight.

Returned on the marker clock so it lines up with the animation. The plates
record downward load as negative, and which of the two plates in each pair
the hitter actually stands on depends on their side, so the loaded one wins.

#### `force_plate_corners(path)`

Corner coordinates of the four plates, as ``(n_plates, 4, 3)`` in meters.

### Internal helpers

| Function | What it does |
| --- | --- |
| `_parse_filename(path)` | Pull hitter, session, height, weight, side, swing number and exit velo from a path. |
| `_match_session_swing(c3d_evs, meta_evs, tol=0.051)` | Map C3D rows onto metadata rows within a session. |
| `_despike(points, tol=0.05)` | Blank single frame marker jumps and fill them by linear interpolation. |
| `_fix_bat(points, index, tol=0.01)` | Repair stretches where a bat marker drops out or drifts. |

## `percentiles.py`

*Ranking the model's inputs across the dataset.*

Percentile ranks for the biomechanics the exit velocity model runs on.

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

### Constants

| Name | Value |
| --- | --- |
| `DEFAULT_PATH` | `Path('data') / 'biomech_percentiles.csv'` |
| `TARGET` | `'exit_velo_mph_x'` |
| `FEATURES` | `...` |
| `FEATURE_COLUMNS` | `list(FEATURES)` |

### Functions

#### `swing_efficiency(df)`

Bat speed over hand speed, guarding the swings that logged zero hand speed.

#### `model_frame(raw_df=None)`

The swings the model is fit on: POI joined to HitTrax, complete features only.

#### `build_percentiles(raw_df=None)`

Rank every model feature across the modelled swings.

Returns one row per swing: the raw value of each feature alongside its
percentile, plus exit velocity and its percentile for context.

#### `save_percentiles(percentiles, path=DEFAULT_PATH)`

Write the percentile table to CSV, creating the directory if it is missing.

#### `load_percentiles(path=DEFAULT_PATH, rebuild=True)`

Read the percentile table, building and saving it if it is not on disk yet.

#### `swing_percentiles(percentiles, session_swing)`

One tidy row per metric for a single swing, ready to render as a table.

Columns are named for display: Metric, Value, Unit, Percentile. Returns an
empty frame with those columns when the swing was never modelled.

## `dashboard.py`

*Building the animated Plotly figure.*

Animated swing dashboard built from the raw hitting C3D files.

``swing_dashboard()`` returns a Plotly figure holding a 3D skeleton and bat that
play through the swing, plus a synced timeline of bat speed and vertical ground
reaction force. Every swing is resampled onto a shared clock with zero at
contact, so switching hitters in the dropdown compares like for like.

    from c3d_functions import download_c3d, index_swings
    from dashboard import pick_showcase, prepare_swings, swing_dashboard

    c3d_dir = download_c3d()
    index = index_swings(c3d_dir)
    swings = prepare_swings(pick_showcase(index, n=8))
    fig = swing_dashboard(swings)
    fig.show()

### Constants

| Name | Value |
| --- | --- |
| `PRE_CONTACT_S` | `0.55` |
| `POST_CONTACT_S` | `0.12` |
| `N_FRAMES` | `90` |
| `TRAIL_FRAMES` | `18` |
| `DL_BLACK` | `'#0e0f12'` |
| `DL_CHARCOAL` | `'#16181d'` |
| `DL_SLATE` | `'#23262e'` |
| `DL_GREY` | `'#6f7681'` |
| `DL_BONE` | `'#eceef1'` |
| `DL_RED` | `'#e23c2f'` |
| `DL_STEEL` | `'#7fa8c9'` |
| `DL_MOSS` | `'#5fbf9b'` |
| `DL_PLUM` | `'#a98cd0'` |
| `BACKGROUND` | `DL_BLACK` |
| `PANEL` | `DL_CHARCOAL` |
| `GRID_COLOR` | `DL_SLATE` |
| `TEXT_COLOR` | `DL_BONE` |
| `MUTED_COLOR` | `DL_GREY` |
| `BODY_COLOR` | `DL_STEEL` |
| `BAT_COLOR` | `DL_RED` |
| `TRAIL_COLOR` | `'#f2856f'` |
| `LEAD_COLOR` | `DL_MOSS` |
| `REAR_COLOR` | `DL_PLUM` |
| `PLATE_COLOR` | `'#2c3038'` |
| `PERCENTILE_LOW` | `DL_STEEL` |
| `PERCENTILE_HIGH` | `DL_RED` |

### Functions

#### `percentile_color(percentile, low=PERCENTILE_LOW, high=PERCENTILE_HIGH)`

Blend the two ends of the ramp, as a hex string.

The blend happens in linear light rather than on the sRGB bytes. Averaging
the bytes of two saturated colours dims whatever sits between them - blue
into red gives a muddy plum halfway - where mixing the light they stand for
holds the brightness up across the middle of the scale.

Direction, not judgement: 100 is the top of this group, which for attack
angle means the steepest swing in the room rather than the best one.

#### `pick_showcase(index, n=8, restrict_to=None)`

Pick n swings from distinct hitters spanning the exit velocity range.

Uses each hitter's hardest swing so the dropdown reads as a tour of the
dataset rather than eight cuts from the same guy, and keeps at least a
couple of lefties in the mix when they are available. Pass ``restrict_to``
a set of ``session_swing`` keys - the ones the model could score, say - to
choose only from swings that carry whatever else you want to show.

#### `prepare_swings(rows, n_frames=N_FRAMES, pre_s=PRE_CONTACT_S, post_s=POST_CONTACT_S)`

Load each swing and resample it onto a shared, contact-aligned clock.

#### `swing_dashboard(prepared, predictions=None, title='Driveline hitters, swing by swing', show_selector=True)`

Assemble the animated dashboard for a list of prepared swings.

Pass ``predictions`` as a frame of ``session_swing``, ``actual`` and
``predicted`` columns to add the model panel, which puts the swing being
animated inside the model's overall predicted against actual scatter.

``show_selector`` draws the in-figure hitter dropdown. Turn it off when the
surrounding app already has a picker, as the Streamlit front end does, and
hand this a single prepared swing instead.

#### `save_dashboard(fig, path='dashboards/hitter_swing_dashboard.html', inline_plotly=True)`

Write a standalone HTML copy that opens without a Python kernel.

### Internal helpers

| Function | What it does |
| --- | --- |
| `_to_linear(byte)` | One sRGB channel as linear light. |
| `_to_srgb(value)` | One linear light channel back to an sRGB byte. |
| `_sample(series, frames)` | Linearly interpolate a ``(3, n_frames)`` track at fractional frame indices. |
| `_body_polyline(swing, frames)` | Skeleton as one ``(n_sampled, n_points, 3)`` polyline, NaN separated. |
| `_swing_label(row)` | One line naming the hitter, their side, exit velo and level, for the dropdown. |
| `_subtitle(swing)` | The detail line under the title: build, bat speed, exit velo, bat, model miss. |
| `_plate_traces(plates)` | Static outlines of the four force plates, drawn on the turf at z = 0. |
| `_miss_arrow(prediction)` | Arrow from where the swing sat on the parity line to where the model put it. |
| `_time_cursor(t_ms)` | The playhead on the timeline. |
| `_attach_predictions(prepared, predictions)` | Hang each swing's model prediction off the prepared dict, where there is one. |
| `_model_traces(predictions)` | The predicted against actual cloud, drawn once and shared by every swing. |
| `_arrow_relayout(index, prediction)` | The arrow's per-swing properties, flattened for a dropdown relayout. |
| `_scene_range(swing)` | Bounding box around one hitter and the arc their bat travels through. |
| `_scene_ranges(swing)` | Per-swing axis ranges, in the shape the dropdown needs to relayout them. |
| `_layout(prepared, time_ms, title, model_range=None)` | The figure's layout: palette, scene camera and ranges, and the two 2D axes. |
| `_play_menu()` | Play, slow motion and pause buttons driving the frame animation. |
| `_swing_menu(fig, prepared, animated, shared, title, arrow_index=None)` | Dropdown swapping which swing is visible, retitling and reframing as it goes. |
| `_slider(time_ms)` | Frame slider labelled in milliseconds to contact. |

## `streamlit_app.py`

*The Streamlit front end.*

Streamlit front end for the swing dashboard.

pip install -r requirements.txt
    streamlit run streamlit_app.py

Picking the hitter in the sidebar rather than inside the figure means only one
swing is ever loaded, so the page stays small and the first paint is quick. The
play controls and the frame slider still belong to Plotly and run in the
browser, so scrubbing through a swing does not round trip to the server.

The first run downloads the 400 MB C3D archive into ``data/c3d`` and fits the
exit velocity model. Both are cached, so it only happens once.

Nothing heavy is imported at module scope. ``streamlit``, ``pandas``, ``plotly``
and this project's own modules together take several seconds to import on a cold
interpreter, and until the first widget is written the browser has an empty page
to show - which is what a blank screen usually is. The page header goes up
first, then the imports happen inside a spinner, and anything that fails on the
way renders as a message rather than as nothing at all.

### Constants

| Name | Value |
| --- | --- |
| `METRIC_BORDER` | `{'border': True} if _accepts(st.metric, 'border') else {}` |
| `CONTAINER_BORDER` | `{'border': True} if _accepts(st.container, 'border') else {}` |
| `C3D_DIR` | `DEFAULT_DATA_DIR / 'c3d'` |
| `PERCENTILE_CSS` | `...` |

### Functions

#### `rerun()`

Rerun the script on whichever name this Streamlit version uses.

#### `c3d_present(directory=C3D_DIR)`

Whether the motion capture files have been unpacked yet.

#### `get_c3d_dir()`

Path to the unpacked C3D files, downloading the archive on first call.

#### `get_index()`

One row per swing C3D, joined to the published metadata where it matches.

#### `get_predictions()`

Out-of-fold exit velocity predictions, mirroring the notebook's model.

The Gaussian process, which won the notebook's comparison at 6.16 RMSE
against 6.57 for a random forest and 6.71 for XGBoost. Kept here rather than
imported so the app stands on its own; the notebook walks through the same
fit with the reasoning attached, including why the kernel is a Matern rather
than the squared exponential.

#### `get_percentiles()`

The percentile table from percentiles.py, built on first run if missing.

#### `get_swing(_index, path)`

Load and resample one swing. Cached on the file path, not the index.

#### `hitter_label(row)`

Hitter number, side and playing level, for the selector and the figure title.

#### `percentile_table_html(table)`

The percentile table as HTML, with the bars drawn off a shared ramp.

Streamlit's ProgressColumn takes one colour for the whole column, so the
bars are built here instead: every one is a window onto the same blue to
red ramp across 0-100, which is what lets the colours be compared between
rows.

#### `environment_note()`

Versions and paths, for working out why someone else's copy misbehaves.

### Internal helpers

| Function | What it does |
| --- | --- |
| `_accepts(func, argument)` | Whether ``func`` takes a keyword argument by that name. |
| `_version()` | The running Streamlit version as a tuple of ints, best effort. |

## `doctor.py`

*Checking that a machine can run the app.*

Check that this machine can run the Streamlit app, and say what is missing.

python doctor.py

A blank page in the browser says nothing about which of the half dozen things
the app needs has gone wrong, so this walks them in order - interpreter,
packages, versions, the C3D archive, and whether GitHub is reachable - and
prints a line per check. Nothing here imports Streamlit's runtime or starts a
server, so it is safe to run while the app is up.

### Constants

| Name | Value |
| --- | --- |
| `REQUIREMENTS` | `...` |

### Functions

#### `version_tuple(text)`

Leading numeric components of a version string, as ints.

#### `check_packages()`

Import each requirement and compare its version against the floor.

#### `check_data()`

Whether the C3D archive and the percentile table are on disk.

#### `check_network()`

Whether the published metrics tables are reachable from here.

#### `main()`

_No docstring._
