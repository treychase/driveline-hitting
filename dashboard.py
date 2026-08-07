"""Animated swing dashboard built from the raw hitting C3D files.

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
"""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from c3d_functions import (
    SEGMENTS,
    bat_points,
    bat_speed_mph,
    contact_frame,
    force_plate_corners,
    load_swing,
    point,
    tracking_ok,
    vertical_grf,
)

# Everything interesting happens in the second before contact and the blink after.
PRE_CONTACT_S = 0.55
POST_CONTACT_S = 0.12
N_FRAMES = 90
TRAIL_FRAMES = 18

BACKGROUND = "#11141c"
PANEL = "#171b26"
BODY_COLOR = "#8fd3f4"
BAT_COLOR = "#ff9f43"
TRAIL_COLOR = "#ff6b6b"
LEAD_COLOR = "#4dd4ac"
REAR_COLOR = "#b98cff"
PLATE_COLOR = "#2b3244"
TEXT_COLOR = "#e6e9f0"
GRID_COLOR = "#2a3040"


def pick_showcase(index, n=8, restrict_to=None):
    """Pick n swings from distinct hitters spanning the exit velocity range.

    Uses each hitter's hardest swing so the dropdown reads as a tour of the
    dataset rather than eight cuts from the same guy, and keeps at least a
    couple of lefties in the mix when they are available. Pass ``restrict_to``
    a set of ``session_swing`` keys - the ones the model could score, say - to
    choose only from swings that carry whatever else you want to show.
    """
    if restrict_to is not None:
        index = index[index["session_swing"].isin(set(restrict_to))]
        if index.empty:
            raise ValueError("restrict_to matched none of the indexed swings")

    best = (
        index.sort_values("exit_velo_mph", ascending=False)
        .groupby("user", as_index=False)
        .first()
        .sort_values("exit_velo_mph")
        .reset_index(drop=True)
    )
    picks = best.iloc[np.linspace(0, len(best) - 1, min(n, len(best))).round().astype(int)]
    picks = picks.drop_duplicates(subset="user")

    lefties = best[best["side"] == "L"]
    if not lefties.empty and (picks["side"] == "L").sum() < 2:
        wanted = 2 - int((picks["side"] == "L").sum())
        add = lefties[~lefties["user"].isin(picks["user"])].tail(wanted)
        if not add.empty:
            drop = picks[picks["side"] == "R"].index[len(picks) - len(add) :]
            picks = picks.drop(index=drop)
            picks = pd.concat([picks, add])

    return picks.sort_values("exit_velo_mph", ascending=False).reset_index(drop=True)


def _sample(series, frames):
    """Linearly interpolate a ``(3, n_frames)`` track at fractional frame indices."""
    grid = np.arange(series.shape[-1])
    return np.stack([np.interp(frames, grid, axis) for axis in series])


def _body_polyline(swing, frames):
    """Skeleton as one ``(n_sampled, n_points, 3)`` polyline, NaN separated."""
    columns = []
    for segment in SEGMENTS:
        resolved = [point(swing, name) for name in segment]
        resolved = [p for p in resolved if p is not None]
        if len(resolved) < 2:
            continue
        for track in resolved:
            columns.append(_sample(track, frames).T)
        columns.append(np.full((len(frames), 3), np.nan))
    return np.stack(columns, axis=1)


def prepare_swings(rows, n_frames=N_FRAMES, pre_s=PRE_CONTACT_S, post_s=POST_CONTACT_S):
    """Load each swing and resample it onto a shared, contact-aligned clock."""
    time_s = np.linspace(-pre_s, post_s, n_frames)
    prepared = []

    for _, row in rows.iterrows():
        swing = load_swing(row["path"], metadata_row=row)
        if not tracking_ok(swing):
            warnings.warn(f"skipping {row['path']}: bat markers drop out mid trial")
            continue
        rate = swing["rate"]
        contact = contact_frame(swing)
        frames = np.clip(contact + time_s * rate, 0, swing["n_frames"] - 1)

        knob, sweet_spot, barrel = bat_points(swing)
        bat = np.stack(
            [_sample(knob, frames).T, _sample(sweet_spot, frames).T, _sample(barrel, frames).T],
            axis=1,
        )

        sweet_path = _sample(sweet_spot, frames).T
        trail = np.full((n_frames, TRAIL_FRAMES, 3), np.nan)
        for i in range(n_frames):
            window = sweet_path[max(0, i - TRAIL_FRAMES + 1) : i + 1]
            trail[i, TRAIL_FRAMES - len(window) :] = window

        lead_grf, rear_grf = vertical_grf(swing)
        grid = np.arange(swing["n_frames"])
        prepared.append(
            {
                "row": row,
                "label": _swing_label(row),
                "time_ms": time_s * 1000,
                # Single precision is well past what the capture resolves and
                # halves the size of the exported HTML.
                "body": _body_polyline(swing, frames).astype("float32"),
                "bat": bat.astype("float32"),
                "trail": trail.astype("float32"),
                "speed": np.interp(frames, grid, bat_speed_mph(swing)).astype("float32"),
                "lead_grf": (np.interp(frames, grid, lead_grf) * 100).astype("float32"),
                "rear_grf": (np.interp(frames, grid, rear_grf) * 100).astype("float32"),
                "plates": force_plate_corners(row["path"]).astype("float32"),
            }
        )
    return prepared


def _swing_label(row):
    level = row.get("highest_playing_level")
    level = f" · {level}" if isinstance(level, str) else ""
    return (
        f"Hitter {int(row['user']):03d} · {row['side']}HH · "
        f"{row['exit_velo_mph']:.1f} mph EV{level}"
    )


def _subtitle(swing):
    row = swing["row"]
    feet, inches = divmod(int(row["height_in"]), 12)
    bits = [
        f"{feet}'{inches}\" · {int(row['mass_lb'])} lb",
        f"peak bat speed {swing['speed'].max():.1f} mph",
        f"exit velo {row['exit_velo_mph']:.1f} mph",
    ]
    if row.get("bat_length_in") == row.get("bat_length_in") and row.get("bat_length_in"):
        bits.append(f"{row['bat_length_in']:.0f} in / {row['bat_weight_oz']:.0f} oz bat")

    if "prediction" in swing:
        prediction = swing["prediction"]
        if prediction:
            actual, predicted = prediction
            bits.append(f"model said {predicted:.1f} ({predicted - actual:+.1f})")
        else:
            bits.append("not scored by the model")
    return "  |  ".join(bits)


def _plate_traces(plates):
    """Static outlines of the four force plates, drawn on the turf at z = 0."""
    traces = []
    for corners in plates:
        loop = np.vstack([corners, corners[:1]])
        traces.append(
            go.Scatter3d(
                x=loop[:, 0],
                y=loop[:, 1],
                z=np.zeros(len(loop)),
                mode="lines",
                line=dict(color=PLATE_COLOR, width=4),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return traces


def _time_cursor(t_ms):
    """The playhead on the timeline.

    It lives in layout rather than as a trace on purpose. Animating a cartesian
    trace makes Plotly redraw the subplot and drop whatever sits on the
    secondary y axis, which would take the force curves with it.
    """
    return dict(
        type="line",
        xref="x",
        yref="y domain",
        x0=t_ms,
        x1=t_ms,
        y0=0,
        y1=1,
        line=dict(color=TEXT_COLOR, width=1),
    )


def _attach_predictions(prepared, predictions):
    """Hang each swing's model prediction off the prepared dict, where there is one."""
    lookup = {}
    if predictions is not None:
        lookup = {
            row["session_swing"]: (row["actual"], row["predicted"])
            for _, row in predictions.iterrows()
        }
    for swing in prepared:
        key = swing["row"].get("session_swing")
        swing["prediction"] = lookup.get(key)


def _model_traces(predictions):
    """The predicted against actual cloud, drawn once and shared by every swing."""
    lo = min(predictions["actual"].min(), predictions["predicted"].min()) - 2
    hi = max(predictions["actual"].max(), predictions["predicted"].max()) + 2
    return [
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(color="#485570", width=1, dash="dash"),
            hoverinfo="skip",
            showlegend=False,
        ),
        go.Scatter(
            x=predictions["actual"],
            y=predictions["predicted"],
            mode="markers",
            marker=dict(size=4.5, color=BODY_COLOR, opacity=0.32,
                        line=dict(color="rgba(0,0,0,0)", width=0)),
            name="All modelled swings",
            hovertemplate="actual %{x:.1f} · predicted %{y:.1f} mph<extra></extra>",
            showlegend=False,
        ),
    ], (lo, hi)


def swing_dashboard(prepared, predictions=None, title="Driveline hitters, swing by swing"):
    """Assemble the animated dashboard for a list of prepared swings.

    Pass ``predictions`` as a frame of ``session_swing``, ``actual`` and
    ``predicted`` columns to add the model panel, which puts the swing being
    animated inside the model's overall predicted against actual scatter.
    """
    if not prepared:
        raise ValueError("no swings to plot")

    _attach_predictions(prepared, predictions)
    has_model = predictions is not None and len(predictions) > 0

    fig = make_subplots(
        rows=2 if has_model else 1,
        cols=2,
        column_widths=[0.58, 0.42],
        row_heights=[0.54, 0.46] if has_model else None,
        vertical_spacing=0.16,
        specs=(
            [
                [{"type": "scene", "rowspan": 2}, {"type": "xy", "secondary_y": True}],
                [None, {"type": "xy"}],
            ]
            if has_model
            else [[{"type": "scene"}, {"type": "xy", "secondary_y": True}]]
        ),
        subplot_titles=(
            ("", "Bat speed and weight shift", "Exit velo: model vs actual")
            if has_model
            else ("", "Bat speed and weight shift")
        ),
    )

    model_range = None
    if has_model:
        traces, model_range = _model_traces(predictions)
        for trace in traces:
            fig.add_trace(trace, row=2, col=2)
    shared = list(range(len(fig.data)))

    time_ms = prepared[0]["time_ms"]
    animated = []  # trace indices updated on every frame, per swing
    for s, swing in enumerate(prepared):
        first = s == 0
        indices = {"start": len(fig.data)}

        for trace in _plate_traces(swing["plates"]):
            trace.visible = first
            fig.add_trace(trace, row=1, col=1)

        body = swing["body"][0]
        fig.add_trace(
            go.Scatter3d(
                x=body[:, 0],
                y=body[:, 1],
                z=body[:, 2],
                mode="lines+markers",
                line=dict(color=BODY_COLOR, width=6),
                marker=dict(size=2.5, color=BODY_COLOR),
                hoverinfo="skip",
                showlegend=False,
                visible=first,
            ),
            row=1,
            col=1,
        )
        indices["body"] = len(fig.data) - 1

        trail = swing["trail"][0]
        fig.add_trace(
            go.Scatter3d(
                x=trail[:, 0],
                y=trail[:, 1],
                z=trail[:, 2],
                mode="lines",
                line=dict(color=TRAIL_COLOR, width=5),
                hoverinfo="skip",
                showlegend=False,
                visible=first,
            ),
            row=1,
            col=1,
        )
        indices["trail"] = len(fig.data) - 1

        bat = swing["bat"][0]
        fig.add_trace(
            go.Scatter3d(
                x=bat[:, 0],
                y=bat[:, 1],
                z=bat[:, 2],
                mode="lines+markers",
                line=dict(color=BAT_COLOR, width=10),
                marker=dict(size=[4, 6, 4], color=BAT_COLOR),
                hoverinfo="skip",
                showlegend=False,
                visible=first,
            ),
            row=1,
            col=1,
        )
        indices["bat"] = len(fig.data) - 1

        fig.add_trace(
            go.Scatter(
                x=time_ms,
                y=swing["speed"],
                mode="lines",
                line=dict(color=BAT_COLOR, width=2.5),
                name="Bat speed (mph)",
                visible=first,
            ),
            row=1,
            col=2,
            secondary_y=False,
        )
        for key, color, name in (
            ("lead_grf", LEAD_COLOR, "Lead leg force (% BW)"),
            ("rear_grf", REAR_COLOR, "Rear leg force (% BW)"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=time_ms,
                    y=swing[key],
                    mode="lines",
                    line=dict(color=color, width=2, dash="dot"),
                    name=name,
                    visible=first,
                ),
                row=1,
                col=2,
                secondary_y=True,
            )

        if has_model:
            actual, predicted = swing["prediction"] or (None, None)
            fig.add_trace(
                go.Scatter(
                    x=[actual] if actual is not None else [],
                    y=[predicted] if predicted is not None else [],
                    mode="markers",
                    marker=dict(size=13, color=BAT_COLOR, symbol="circle",
                                line=dict(color=BACKGROUND, width=1.5)),
                    hovertemplate="this swing<br>actual %{x:.1f} · predicted %{y:.1f} mph"
                    "<extra></extra>",
                    showlegend=False,
                    visible=first,
                ),
                row=2,
                col=2,
            )

        indices["stop"] = len(fig.data)
        animated.append(indices)

    frames = []
    for k in range(len(time_ms)):
        data, traces = [], []
        for swing, indices in zip(prepared, animated):
            body, trail, bat = swing["body"][k], swing["trail"][k], swing["bat"][k]
            data += [
                go.Scatter3d(x=body[:, 0], y=body[:, 1], z=body[:, 2]),
                go.Scatter3d(x=trail[:, 0], y=trail[:, 1], z=trail[:, 2]),
                go.Scatter3d(x=bat[:, 0], y=bat[:, 1], z=bat[:, 2]),
            ]
            traces += [indices["body"], indices["trail"], indices["bat"]]
        frames.append(
            go.Frame(
                name=str(k),
                data=data,
                traces=traces,
                layout=dict(shapes=[_time_cursor(time_ms[k])]),
            )
        )
    fig.frames = frames

    fig.update_layout(
        _layout(prepared, time_ms, title, model_range),
        updatemenus=[
            _play_menu(),
            _swing_menu(fig, prepared, animated, shared, title),
        ],
        sliders=[_slider(time_ms)],
    )
    return fig


def _scene_range(swing):
    """Bounding box around one hitter and the arc their bat travels through."""
    points = np.concatenate([swing["body"].reshape(-1, 3), swing["bat"].reshape(-1, 3)])
    low = np.nanmin(points, axis=0) - 0.12
    high = np.nanmax(points, axis=0) + 0.12
    return low, high


def _scene_ranges(swing):
    """Per-swing axis ranges, in the shape the dropdown needs to relayout them."""
    low, high = _scene_range(swing)
    return {
        "scene.xaxis.range": [low[0], high[0]],
        "scene.yaxis.range": [low[1], high[1]],
        "scene.zaxis.range": [0, high[2]],
    }


def _layout(prepared, time_ms, title, model_range=None):
    low, high = _scene_range(prepared[0])
    axis = dict(
        backgroundcolor=PANEL,
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        color=TEXT_COLOR,
        showbackground=True,
    )
    layout = dict(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:13px;color:#9aa3b5'>"
            f"{_subtitle(prepared[0])}</span>",
            x=0.015,
            y=0.97,
            xanchor="left",
            yanchor="top",
        ),
        template="plotly_dark",
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT_COLOR, family="Inter, Helvetica, Arial, sans-serif"),
        height=760 if model_range else 700,
        margin=dict(l=55, r=10, t=110, b=90),
        legend=dict(orientation="h", y=-0.16, x=0.60, font=dict(size=11)),
        scene=dict(
            xaxis=dict(title="to mound (m)", range=[low[0], high[0]], **axis),
            yaxis=dict(title="to RHH box (m)", range=[low[1], high[1]], **axis),
            zaxis=dict(title="up (m)", range=[0, high[2]], **axis),
            aspectmode="data",
            # Looking in from the open side, a little above the hitter's hands.
            camera=dict(eye=dict(x=-1.3, y=1.2, z=0.6), up=dict(x=0, y=0, z=1)),
            domain=dict(x=[0.0, 0.55], y=[0.05, 1.0]),
        ),
        xaxis=dict(
            title="time to contact (ms)",
            range=[time_ms[0], time_ms[-1]],
            gridcolor=GRID_COLOR,
            zerolinecolor="#4a5468",
        ),
        yaxis=dict(title="bat speed (mph)", range=[0, 110], gridcolor=GRID_COLOR),
        yaxis2=dict(title="vertical force (% bodyweight)", range=[0, 320], showgrid=False),
        shapes=[_time_cursor(time_ms[0])],
    )

    if model_range:
        # Identical ranges on both axes so the parity line runs corner to corner.
        # Not scaleanchor'd: that would stretch the x range to match the panel's
        # aspect and leave the data floating in the middle of it.
        for name, label in (("xaxis2", "actual exit velo (mph)"), ("yaxis3", "predicted (mph)")):
            layout[name] = dict(
                title=label,
                range=list(model_range),
                gridcolor=GRID_COLOR,
                zeroline=False,
            )
    return layout


def _play_menu():
    def step(duration):
        return dict(frame=dict(duration=duration, redraw=True), mode="immediate",
                    transition=dict(duration=0), fromcurrent=True)

    return dict(
        type="buttons",
        direction="left",
        x=0.02,
        y=-0.02,
        xanchor="left",
        yanchor="top",
        pad=dict(t=10, r=10),
        bgcolor=PANEL,
        bordercolor=GRID_COLOR,
        font=dict(size=12),
        buttons=[
            dict(label="▶ Play", method="animate", args=[None, step(45)]),
            dict(label="🐢 Slow", method="animate", args=[None, step(140)]),
            dict(
                label="⏸ Pause",
                method="animate",
                args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
            ),
        ],
    )


def _swing_menu(fig, prepared, animated, shared, title):
    n_traces = len(fig.data)
    buttons = []
    for swing, indices in zip(prepared, animated):
        visible = [False] * n_traces
        for i in list(shared) + list(range(indices["start"], indices["stop"])):
            visible[i] = True
        layout = {
            "title.text": f"<b>{title}</b><br>"
            f"<span style='font-size:13px;color:#9aa3b5'>{_subtitle(swing)}</span>"
        }
        layout.update(_scene_ranges(swing))
        buttons.append(
            dict(label=swing["label"], method="update", args=[{"visible": visible}, layout])
        )
    return dict(
        type="dropdown",
        direction="down",
        x=1.0,
        y=1.16,
        xanchor="right",
        yanchor="top",
        bgcolor=PANEL,
        bordercolor=GRID_COLOR,
        font=dict(size=12),
        showactive=True,
        buttons=buttons,
    )


def _slider(time_ms):
    return dict(
        active=0,
        x=0.02,
        y=-0.06,
        len=0.55,
        pad=dict(t=48),
        currentvalue=dict(prefix="time to contact: ", suffix=" ms", font=dict(size=12)),
        bgcolor=GRID_COLOR,
        steps=[
            dict(
                # Every step needs a label: the slider's readout shows it.
                label=f"{t:.0f}",
                method="animate",
                args=[
                    [str(k)],
                    dict(mode="immediate", frame=dict(duration=0, redraw=True),
                         transition=dict(duration=0)),
                ],
            )
            for k, t in enumerate(time_ms)
        ],
    )


def save_dashboard(fig, path="dashboards/hitter_swing_dashboard.html", inline_plotly=True):
    """Write a standalone HTML copy that opens without a Python kernel."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        path,
        include_plotlyjs=True if inline_plotly else "cdn",
        full_html=True,
        auto_play=False,
    )
    return path
