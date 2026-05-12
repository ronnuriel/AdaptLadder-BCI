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
- Decoder-facing WER/PER/CTC evaluation is the next important missing piece.

## Paper draft

See `paper/main.tex`.
