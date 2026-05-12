# Figure and Table Plan

Paper-local figure copies live in `paper/figures/`. The generated source figures remain under `results/figures/`.

## Main Figures

1. `t15_source_sweep_weighted_per.png` + `t15_input_layer_calibration_recovery_source_middle_epochs5.png`
   - Message: fixed non-native source layers break the decoder; learned adapters recover only modestly.

2. `t15_previous_vs_geometry_per.png` + `t15_selected_source_timeline.png`
   - Message: previous-session recency is the strongest simple source baseline; geometry reveals older-state matches.

3. `t15_kshot_recency_geometry_override_weighted_per.png` + `t12_diagnostic_geometry_distance_vs_days.png`
   - Message: a simple geometry override is not enough; T12 supports the longitudinal geometry signal.

## Main Tables

1. T15 stress test and adaptation ladder.
   - Native-day, fixed early/middle/late, z-score, moment matching, diagonal affine K=20, input-layer K=20.

2. Beginning-of-day source selection.
   - Native-day, fixed middle, previous source, K-shot geometry source, best simple override for K=5/10/20.

## Optional/Supplementary Figures

- `t15_selected_source_lag_histogram.png`
- `t12_diagnostic_selected_source_timeline.png`
- `t12_diagnostic_selected_source_lag_histogram.png`
- `t15_recovery_vs_time_distance.png`
- `t15_near_far_recovery_by_covariance.png`
