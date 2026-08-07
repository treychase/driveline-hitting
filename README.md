# Driveline Hitting Analysis
An analysis of Driveline's open source biomechanics dataset.

The notebook works through the point-of-interest and HitTrax tables, builds a swing efficiency
feature, and fits Random Forest and XGBoost models for exit velocity. The last section drops down to
the raw C3D motion capture and animates the swings themselves.

## Animated swing dashboard

`dashboards/hitter_swing_dashboard.html` is a standalone page: pick a hitter, press play, and watch
the skeleton and bat move through the swing next to a timeline of bat speed and the vertical force
each leg puts into the ground. A third panel plots the model's predicted exit velocity against what
the ball actually did, with the loaded swing highlighted. Open the file directly in a browser, or
rebuild it with

```python
from c3d_functions import download_c3d, index_swings
from dashboard import pick_showcase, prepare_swings, swing_dashboard, save_dashboard

index = index_swings(download_c3d())
save_dashboard(swing_dashboard(prepare_swings(pick_showcase(index, n=8))))
```

Pass `predictions` (a frame of `session_swing`, `actual` and `predicted`) to `swing_dashboard()` for
the model panel, and the same keys to `pick_showcase(restrict_to=...)` so every hitter in the
dropdown has a prediction to show. The notebook builds those from out-of-fold forest predictions.

## Streamlit app

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Same three panels, with the hitter and swing chosen in the sidebar instead of inside the figure, so
only one swing is ever loaded and the page stays light. Any of the 687 swings is reachable, not just
the eight in the static file. The download, the swing index and the model fit are all cached, so the
wait is a first-run cost.

Under the figure is a percentile table: where the selected swing's biomechanics rank against the 581
swings the model was fit on.

## Percentile dataset

```bash
python percentiles.py
```

Writes `data/biomech_percentiles.csv`, one row per modelled swing holding each of the model's eight
input features alongside its percentile rank, plus exit velocity and its rank. That file is what
powers the table in the app; `load_percentiles()` builds it on first use if it is not there.

Ranks are inside this dataset, not against any wider population, and no metric has a good end. A high
attack angle percentile means steeper than most of the room, not better than most of the room.

## Colour

Every colour in the figure comes from the palette block at the top of `dashboard.py`, and
`.streamlit/config.toml` mirrors it for the page chrome. Changing the brand hexes means editing those
two places and nothing else.

## Documentation

[`docs/api.md`](docs/api.md) documents every module, constant and function. It is generated from the
source, so regenerate it after changing a signature or a docstring:

```bash
python docs/build_api_docs.py
```

`download_c3d()` fetches the 400 MB C3D archive from the openbiomechanics `dataset-v1` release into
`data/c3d`, which is gitignored. Pass any subset of `index_swings()` rows to `prepare_swings()` to
animate different hitters.

<img width="970" height="300" alt="image" src="https://github.com/user-attachments/assets/ed7583d1-dd82-4c35-ba22-f515ad087004" />
