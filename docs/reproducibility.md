# Reproducibility Guide

This guide lists the commands used to reproduce the paper-facing T15 and T12
outputs. Generated files are written under `results/`, which is ignored by git.
Selected final figures are copied into `paper/figures/`.

## 1. T15 Sanity and Drift

```bash
python scripts/run_t15_sanity.py
python scripts/run_t15_drift_map.py
```

Main outputs:

```text
results/tables/t15_dataset_summary.csv
results/tables/t15_session_drift_metrics.csv
results/figures/t15_drift_over_time.png
results/figures/t15_pca_sessions.png
results/figures/t15_mean_shift_heatmap.png
```

## 2. Decoder-Facing PER Probe

Native-day baseline:

```bash
python scripts/run_t15_decoder_probe.py \
  --model-path data/external/btt-25-gru-pure-baseline-0-0898 \
  --data-dir data/raw/hdf5_data_final \
  --csv-path data/external/t15_copyTaskData_description.csv \
  --eval-type val \
  --mode native-day \
  --gpu-number -1
```

Fixed-source stress tests:

```bash
for SRC in t15.2023.08.13 t15.2023.11.26 t15.2025.04.13; do
  SAFE_SRC=$(echo "$SRC" | sed 's/t15\.//' | tr '.' '_')
  python scripts/run_t15_decoder_probe.py \
    --model-path data/external/btt-25-gru-pure-baseline-0-0898 \
    --data-dir data/raw/hdf5_data_final \
    --csv-path data/external/t15_copyTaskData_description.csv \
    --eval-type val \
    --mode cross-day \
    --source-session "$SRC" \
    --gpu-number -1 \
    --output-trials "results/tables/t15_decoder_probe_cross_day_source_${SAFE_SRC}_val.csv" \
    --output-summary "results/tables/t15_decoder_probe_cross_day_source_${SAFE_SRC}_session_summary.csv"
done

python scripts/plot_t15_source_sweep.py
```

## 3. Moment-Based Adaptation

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

## 4. Small Supervised Calibration

Diagonal affine:

```bash
python scripts/run_t15_affine_calibration_eval.py \
  --model-path data/external/btt-25-gru-pure-baseline-0-0898 \
  --data-dir data/raw/hdf5_data_final \
  --csv-path data/external/t15_copyTaskData_description.csv \
  --eval-type val \
  --source-session t15.2023.11.26 \
  --calibration-trials 5 10 20 \
  --methods native-day none moment_match_to_source diagonal_affine \
  --epochs 5 \
  --learning-rate 0.01 \
  --device cpu \
  --gpu-number -1 \
  --output-trials results/tables/t15_affine_calibration_trial_results_source_middle_epochs5.csv \
  --output-summary results/tables/t15_affine_calibration_session_summary_source_middle_epochs5.csv \
  --output-overall results/tables/t15_affine_calibration_overall_summary_source_middle_epochs5.csv \
  --output-training results/tables/t15_affine_calibration_training_summary_source_middle_epochs5.csv \
  --output-weighted-figure results/figures/t15_affine_calibration_weighted_per_source_middle_epochs5.png \
  --output-recovery-figure results/figures/t15_affine_calibration_recovery_source_middle_epochs5.png
```

Input-layer calibration:

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

## 5. Drift Geometry and Recovery

```bash
python scripts/analyze_t15_recovery_geometry.py \
  --data-dir data/raw/hdf5_data_final \
  --stats-split train \
  --source-session t15.2023.11.26 \
  --max-frames 60000 \
  --n-components 20 \
  --cov-shrinkage 0.05 \
  --n-bootstrap 1000 \
  --n-permutations 1000
```

## 6. Geometry Source Selection

Offline and past-only source selection:

```bash
python scripts/run_t15_geometry_source_selection_eval.py \
  --data-dir data/raw/hdf5_data_final \
  --eval-type val \
  --stats-split train \
  --selection-metric cov_relative_fro_shift_from_source \
  --source-candidate-mode all \
  --max-frames 60000 \
  --n-components 20 \
  --cov-shrinkage 0.05 \
  --gpu-number -1

python scripts/run_t15_geometry_source_selection_eval.py \
  --data-dir data/raw/hdf5_data_final \
  --eval-type val \
  --stats-split train \
  --selection-metric cov_relative_fro_shift_from_source \
  --source-candidate-mode past-only \
  --max-frames 60000 \
  --n-components 20 \
  --cov-shrinkage 0.05 \
  --gpu-number -1
```

Beginning-of-day K-shot source selection:

```bash
python scripts/run_t15_kshot_geometry_source_selection.py \
  --calibration-trials 5 10 20 \
  --source-candidate-mode past-only \
  --selection-metric cov_relative_fro_shift_from_source \
  --max-source-frames 60000 \
  --n-components 20 \
  --cov-shrinkage 0.05 \
  --gpu-number -1
```

Previous-session comparison and recency-aware override:

```bash
python scripts/analyze_t15_selected_sources.py \
  --model-path data/external/btt-25-gru-pure-baseline-0-0898 \
  --data-dir data/raw/hdf5_data_final \
  --csv-path data/external/t15_copyTaskData_description.csv \
  --eval-type val \
  --gpu-number -1

python scripts/analyze_t15_recency_geometry_override.py \
  --calibration-trials 5 10 20 \
  --alpha 0.5 0.6 0.7 0.8 0.9 1.0 \
  --max-source-frames 60000
```

## 7. Optional T12 Geometry-Only Check

Prepare the Dryad manifest:

```bash
python scripts/prepare_t12_dryad_manifest.py
```

Extract `diagnosticBlocks.tar.gz` under:

```text
data/raw/t12_diagnosticBlocks/
```

Run:

```bash
python scripts/run_t12_geometry_feasibility.py \
  --data-dir data/raw/t12_diagnosticBlocks \
  --feature-key spikePow \
  --source-candidate-mode past-only \
  --selection-metric cov_relative_fro_shift_from_source \
  --max-frames 60000 \
  --n-components 20 \
  --cov-shrinkage 0.05
```

This is a geometry-only support analysis. It does not reproduce the T12 decoder.

## 8. Paper

```bash
cd paper
tectonic main.tex
```
