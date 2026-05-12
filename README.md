# AdaptLadder-BCI

Decoder-facing cross-session drift analysis for brain-to-text BCI.

This repository supports the short paper:

**How Much Adaptation Is Enough? Drift Geometry and Minimal Calibration in Brain-to-Text BCI**

## Core Claim

The project does not propose a new decoder. It audits how much input adaptation is
needed when an existing brain-to-text decoder encounters cross-session drift.

The current conclusion is:

> Fixed non-native input layers strongly degrade T15 decoding. Simple
> moment-based correction is not enough. Small learned adapters recover only
> modestly. Most practical recovery comes from choosing a compatible past input
> layer, where previous-session recency is a strong default and drift geometry is
> a useful signal for possible overrides.

## Main Results

| Result | Finding |
| --- | --- |
| Native-day T15 decoding | 9.06% phoneme-weighted PER |
| Fixed non-native source layers | 22.67-34.43% PER; 40/41 sessions harmed |
| Target z-score / moment matching | No recovery; slightly worse than no correction |
| Diagonal affine, K=20 | 20.21% PER on remaining trials; about 10% gap recovery |
| Input-layer calibration, K=20 | 20.27% PER on remaining trials; about 9.7% gap recovery |
| K-shot geometry source selection, K=20 | 12.74% PER; about 71% fixed-source gap recovery |
| Previous-session source, K=20 | 12.21% PER; strongest simple source baseline |
| Simple geometry override | Tiny K=20 gain only; richer gate needed |
| T12 diagnostic blocks | Geometry-only support: covariance distance increases with temporal separation |

## Repository Layout

```text
paper/
  main.tex                  # paper draft
  main.pdf                  # compiled draft
  figures/                  # selected paper figures tracked in git
scripts/
  run_t15_decoder_probe.py
  run_t15_adaptation_eval.py
  run_t15_affine_calibration_eval.py
  run_t15_input_layer_calibration_eval.py
  run_t15_geometry_source_selection_eval.py
  run_t15_kshot_geometry_source_selection.py
  analyze_t15_selected_sources.py
  analyze_t15_recency_geometry_override.py
  run_t12_geometry_feasibility.py
src/
  decoder_eval.py
  drift_metrics.py
  t15_utils.py
data/external/
  t15_copyTaskData_description.csv
  t12_high_performance_speech/      # metadata only
third_party/
  nejm-brain-to-text                # official baseline submodule
```

Raw datasets, checkpoints, generated result tables, and generated result figures
are local-only and ignored by git.

## Reproducing

Install dependencies:

```bash
pip install -r requirements.txt
```

Expected local data layout:

```text
data/raw/hdf5_data_final/
data/external/btt-25-gru-pure-baseline-0-0898/
data/raw/t12_diagnosticBlocks/      # optional geometry-only support
```

The complete command sequence is in
[docs/reproducibility.md](docs/reproducibility.md).

To compile the paper:

```bash
cd paper
tectonic main.tex
```

## Scope

T15 is the main decoder-facing result. T12 is included only as a geometry-only
feasibility check; this repository does not claim T12 decoder or PER
replication.
