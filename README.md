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

## Paper draft

See `paper/main.tex`.
