# Driveline Hitting Analysis
An analysis of Driveline's open source biomechanics dataset.

The notebook works through the point-of-interest and HitTrax tables, builds a swing efficiency
feature, and fits Random Forest and XGBoost models for exit velocity. The last section drops down to
the raw C3D motion capture and animates the swings themselves.

## Animated swing dashboard

`dashboards/hitter_swing_dashboard.html` is a standalone page: pick a hitter, press play, and watch
the skeleton and bat move through the swing next to a timeline of bat speed and the vertical force
each leg puts into the ground. Open the file directly in a browser, or rebuild it with

```python
from c3d_functions import download_c3d, index_swings
from dashboard import pick_showcase, prepare_swings, swing_dashboard, save_dashboard

index = index_swings(download_c3d())
save_dashboard(swing_dashboard(prepare_swings(pick_showcase(index, n=8))))
```

`download_c3d()` fetches the 400 MB C3D archive from the openbiomechanics `dataset-v1` release into
`data/c3d`, which is gitignored. Pass any subset of `index_swings()` rows to `prepare_swings()` to
animate different hitters.

<img width="970" height="300" alt="image" src="https://github.com/user-attachments/assets/ed7583d1-dd82-4c35-ba22-f515ad087004" />
