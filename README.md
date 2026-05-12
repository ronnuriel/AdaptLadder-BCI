# AdaptLadder-BCI

Lightweight cross-session drift analysis and first-line adaptation experiments for
brain-to-text BCI.

## Working claim

Lightweight input adaptation is a strong first-line response to cross-session drift
in brain-to-text BCI, but its remaining errors show when stronger adaptation is
needed.

The project is deliberately not framed as a new decoder. The goal is to measure
how much simple correction helps before moving up an adaptation ladder:

1. No adaptation
2. Target-session normalization
3. Source-to-target moment matching / diagonal affine correction
4. Later: TTA, fine-tuning, or supervised recalibration

## Expected local layout

```text
AdaptLadder-BCI/
  data/
    raw/
      hdf5_data_final/
        t15.2023.08.13/
          data_train.hdf5
          data_val.hdf5
          data_test.hdf5
        ...
    external/
      t15_copyTaskData_description.csv
      btt-25-gru-pure-baseline-0-0898/
        checkpoint/              # ignored by git
  notebooks/
  paper/
  results/
    tables/                      # generated, ignored by git
    figures/                     # generated, ignored by git
  scripts/
  src/
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate T15 sanity table

```bash
python scripts/run_t15_sanity.py
```

Writes:

```text
results/tables/t15_dataset_summary.csv
```

### 3. Generate T15 drift map

```bash
python scripts/run_t15_drift_map.py
```

Writes:

```text
results/tables/t15_session_drift_metrics.csv
results/tables/t15_session_mean_pca.csv
results/figures/t15_drift_over_time.png
results/figures/t15_pca_sessions.png
results/figures/t15_mean_shift_heatmap.png
```

## Current status

- T15 local HDF5 data is present.
- Dataset sanity summary generation works.
- Train-split drift metrics and figures generation works.
- Decoder-facing greedy PER evaluation works with the official GRU checkpoint.
- Native-day validation reaches 9.06% phoneme-weighted PER.
- Cross-day stress with source `t15.2023.08.13` reaches 28.81% phoneme-weighted PER and harms 40/41 sessions.
- A three-source sweep gives 28.81% PER for early, 22.67% for middle, and 34.43% for late source layers; each harms 40/41 sessions.
- The first adaptation ladder run on the middle `t15.2023.11.26` source finds that target z-scoring and source moment matching do not recover the cross-day loss, suggesting mean/variance alignment alone is insufficient.
- A small supervised diagonal-affine calibration script is available for the next ladder step: learn only per-channel scale and bias from the first K labeled trials, then evaluate PER on the remaining validation trials.
- A CPU-feasible all-session affine run with 5 training epochs shows a modest but consistent recovery: diagonal affine improves weighted PER from 22.51% to 21.99% at K=5, 22.48% to 21.81% at K=10, and 21.51% to 20.21% at K=20.
- A CPU-feasible all-session full input-layer run with 5 epochs gives similar modest recovery: input-layer calibration improves weighted PER from 22.51% to 21.70% at K=5, 22.48% to 21.60% at K=10, and 21.51% to 20.27% at K=20.
- Recovery-geometry analysis suggests cross-day PER is strongly associated with temporal/covariance distance from the source, while input-layer recovery is weaker for more distant or covariance-shifted sessions.

## Decoder Experiments

Native-day:

```bash
python scripts/run_t15_decoder_probe.py \
  --model-path data/external/btt-25-gru-pure-baseline-0-0898 \
  --data-dir data/raw/hdf5_data_final \
  --csv-path data/external/t15_copyTaskData_description.csv \
  --eval-type val \
  --mode native-day \
  --gpu-number -1
```

Cross-day source sweep:

```bash
python scripts/run_t15_decoder_probe.py \
  --eval-type val \
  --mode cross-day \
  --source-session t15.2023.08.13 \
  --gpu-number -1 \
  --output-trials results/tables/t15_decoder_probe_cross_day_source_2023_08_13_val.csv \
  --output-summary results/tables/t15_decoder_probe_cross_day_source_2023_08_13_session_summary.csv

python scripts/run_t15_decoder_probe.py \
  --eval-type val \
  --mode cross-day \
  --source-session t15.2023.11.26 \
  --gpu-number -1 \
  --output-trials results/tables/t15_decoder_probe_cross_day_source_2023_11_26_val.csv \
  --output-summary results/tables/t15_decoder_probe_cross_day_source_2023_11_26_session_summary.csv

python scripts/run_t15_decoder_probe.py \
  --eval-type val \
  --mode cross-day \
  --source-session t15.2025.04.13 \
  --gpu-number -1 \
  --output-trials results/tables/t15_decoder_probe_cross_day_source_2025_04_13_val.csv \
  --output-summary results/tables/t15_decoder_probe_cross_day_source_2025_04_13_session_summary.csv

python scripts/plot_t15_source_sweep.py
```

Adaptation ladder, middle source:

```bash
python scripts/run_t15_adaptation_eval.py \
  --model-path data/external/btt-25-gru-pure-baseline-0-0898 \
  --data-dir data/raw/hdf5_data_final \
  --csv-path data/external/t15_copyTaskData_description.csv \
  --eval-type val \
  --source-session t15.2023.11.26 \
  --adaptations none target_zscore moment_match_to_source \
  --gpu-number -1 \
  --output-trials results/tables/t15_adaptation_trial_results_source_middle.csv \
  --output-summary results/tables/t15_adaptation_session_summary_source_middle.csv \
  --output-overall results/tables/t15_adaptation_overall_summary_source_middle.csv \
  --output-weighted-figure results/figures/t15_adaptation_ladder_weighted_per_source_middle.png \
  --output-delta-figure results/figures/t15_adaptation_ladder_delta_per_by_session_source_middle.png
```

Small-calibration diagonal affine, middle source:

```bash
python scripts/run_t15_affine_calibration_eval.py \
  --model-path data/external/btt-25-gru-pure-baseline-0-0898 \
  --data-dir data/raw/hdf5_data_final \
  --csv-path data/external/t15_copyTaskData_description.csv \
  --eval-type val \
  --source-session t15.2023.11.26 \
  --calibration-trials 5 10 20 \
  --methods native-day none moment_match_to_source diagonal_affine \
  --epochs 40 \
  --learning-rate 0.01 \
  --device cpu \
  --gpu-number -1 \
  --output-trials results/tables/t15_affine_calibration_trial_results_source_middle.csv \
  --output-summary results/tables/t15_affine_calibration_session_summary_source_middle.csv \
  --output-overall results/tables/t15_affine_calibration_overall_summary_source_middle.csv \
  --output-training results/tables/t15_affine_calibration_training_summary_source_middle.csv \
  --output-weighted-figure results/figures/t15_affine_calibration_weighted_per_source_middle.png \
  --output-recovery-figure results/figures/t15_affine_calibration_recovery_source_middle.png
```

The checked pilot used the same command with `--epochs 5` and `_epochs5`
output suffixes.

Small-calibration full input layer, middle source:

```bash
python scripts/run_t15_input_layer_calibration_eval.py \
  --model-path data/external/btt-25-gru-pure-baseline-0-0898 \
  --data-dir data/raw/hdf5_data_final \
  --csv-path data/external/t15_copyTaskData_description.csv \
  --eval-type val \
  --source-session t15.2023.11.26 \
  --calibration-trials 5 10 20 \
  --methods native-day none moment_match_to_source input_layer \
  --epochs 5 \
  --learning-rate 1e-4 \
  --l2-weight 1e-4 \
  --device cpu \
  --gpu-number -1 \
  --output-trials results/tables/t15_input_layer_calibration_trial_results_source_middle_epochs5.csv \
  --output-summary results/tables/t15_input_layer_calibration_session_summary_source_middle_epochs5.csv \
  --output-overall results/tables/t15_input_layer_calibration_overall_summary_source_middle_epochs5.csv \
  --output-training results/tables/t15_input_layer_calibration_training_summary_source_middle_epochs5.csv \
  --output-weighted-figure results/figures/t15_input_layer_calibration_weighted_per_source_middle_epochs5.png \
  --output-recovery-figure results/figures/t15_input_layer_calibration_recovery_source_middle_epochs5.png
```

Recovery geometry analysis:

```bash
python scripts/analyze_t15_recovery_geometry.py \
  --data-dir data/raw/hdf5_data_final \
  --stats-split train \
  --source-session t15.2023.11.26 \
  --max-frames 60000 \
  --n-components 20 \
  --cov-shrinkage 0.05
```

Writes:

```text
results/tables/t15_session_recovery_geometry_joined.csv
results/tables/t15_recovery_geometry_correlations.csv
results/figures/t15_recovery_vs_subspace_angle.png
results/figures/t15_recovery_vs_cross_day_per.png
results/figures/t15_recovery_vs_time_distance.png
```

## Paper draft

See `paper/main.tex`.
