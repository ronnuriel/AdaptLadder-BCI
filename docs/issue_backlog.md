# Issue Backlog

The GitHub connector was not granted issue-write permission in this environment.
These are the first issues to open once issue creation is available.

## 1. Implement decoder-facing PER probe for T15 validation split

Goal: Add a decoder-facing probe for the T15 validation split without running the Redis language model.

Acceptance:
- Loads the official GRU checkpoint.
- Runs the T15 `val` split.
- Saves trial-level greedy phoneme error rate results.
- Saves session-level summary results.
- Includes blank rate, mean confidence, and entropy columns.

Expected outputs:
- `results/tables/t15_decoder_probe_val.csv`
- `results/tables/t15_decoder_probe_session_summary.csv`

## 2. Add cross-day stress-test evaluation mode

Goal: Evaluate target sessions through a fixed source day-specific input layer to measure decoder degradation under drift.

Acceptance:
- Supports `--mode cross-day`.
- Supports `--source-session`.
- Evaluates target sessions through the selected source input layer.
- Reports PER degradation compared with `native-day` mode.

Expected comparison:
- Native-day: target session uses its own input layer.
- Cross-day: target session uses a fixed source input layer.

## 3. Evaluate adaptation ladder on cross-day stress test

Goal: Test lightweight input-space corrections before the fixed source input layer in the cross-day stress test.

Acceptance:
- Supports `none`.
- Supports `target_zscore`.
- Supports `moment_match_to_source`.
- Saves trial-level results.
- Saves session summary CSV.

Expected outputs:
- `results/tables/t15_adaptation_trial_results.csv`
- `results/tables/t15_adaptation_session_summary.csv`

## 4. Join drift metrics with adaptation gains

Goal: Connect distribution drift metrics to decoder-facing adaptation gains.

Acceptance:
- Creates `results/tables/t15_drift_vs_adaptation_gain.csv`.
- Includes drift metrics, PER by method, gains, and best method per session.
- Creates gain-vs-drift figures.

Expected figures:
- `results/figures/t15_gain_vs_mean_shift.png`
- `results/figures/t15_gain_vs_scale_shift.png`
- `results/figures/t15_gain_vs_days.png`
- `results/figures/t15_best_method_by_session.png`

## 5. Write first results section in paper/main.tex

Goal: Turn the first T15 sanity, drift, and decoder-facing results into the initial paper results section.

Acceptance:
- Adds dataset paragraph.
- Adds method paragraph for drift and decoder probe.
- Adds first table reference.
- Adds first two figure references.
- Keeps title stable until PER/adaptation results decide whether gating is central.
