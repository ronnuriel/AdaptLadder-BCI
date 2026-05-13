from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_t15_decoder_probe import _as_plain_string, _set_use_amp
from src.decoder_eval import (
    add_official_model_training_to_path,
    greedy_ctc_decode,
    load_official_gru_decoder,
    resolve_device,
)


KEYS = ["calibration_trials", "session", "block", "trial"]
FEATURE_COLUMNS = [
    "confidence_delta",
    "entropy_delta",
    "blank_rate_delta",
    "pred_length_delta",
    "nonblank_rate_delta",
    "logit_margin_delta",
    "previous_mean_confidence",
    "geometry_mean_confidence",
    "previous_entropy",
    "geometry_entropy",
    "previous_blank_rate",
    "geometry_blank_rate",
]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    return probs / probs.sum(axis=-1, keepdims=True)


def logit_quality(logits: np.ndarray) -> dict[str, float]:
    probs = softmax(logits)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=-1)
    pred = np.argmax(probs, axis=-1)
    top2 = np.partition(probs, -2, axis=-1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    decoded = greedy_ctc_decode(logits)
    return {
        "blank_rate": float(np.mean(pred == 0)),
        "nonblank_rate": float(np.mean(pred != 0)),
        "mean_confidence": float(np.mean(np.max(probs, axis=-1))),
        "entropy": float(np.mean(entropy)),
        "logit_margin": float(np.mean(margin)),
        "pred_length": float(len(decoded)),
    }


def weighted_per(frame: pd.DataFrame, prefix: str) -> float:
    return float(frame[f"{prefix}_edit_distance"].sum() / frame[f"{prefix}_num_phonemes"].sum())


def weighted_per_from_choice(frame: pd.DataFrame, use_geometry: np.ndarray) -> float:
    edits = np.where(use_geometry, frame["geometry_edit_distance"], frame["previous_edit_distance"]).sum()
    phonemes = np.where(use_geometry, frame["geometry_num_phonemes"], frame["previous_num_phonemes"]).sum()
    return float(edits / phonemes)


def add_prefix(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep = KEYS + ["edit_distance", "num_phonemes", "PER", "blank_rate", "mean_confidence", "entropy"]
    available = [column for column in keep if column in frame.columns]
    renamed = {column: f"{prefix}_{column}" for column in available if column not in KEYS}
    return frame[available].rename(columns=renamed)


def build_eval_table(previous_trials_path: Path, geometry_trials_path: Path) -> pd.DataFrame:
    previous = add_prefix(pd.read_csv(previous_trials_path), "previous")
    geometry = add_prefix(pd.read_csv(geometry_trials_path), "geometry")
    table = previous.merge(geometry, on=KEYS, how="inner", validate="one_to_one")
    return table


def session_eval_table(eval_trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (k, session), frame in eval_trials.groupby(["calibration_trials", "session"]):
        previous_edits = frame["previous_edit_distance"].sum()
        previous_phonemes = frame["previous_num_phonemes"].sum()
        geometry_edits = frame["geometry_edit_distance"].sum()
        geometry_phonemes = frame["geometry_num_phonemes"].sum()
        rows.append(
            {
                "calibration_trials": int(k),
                "session": session,
                "previous_edit_distance": float(previous_edits),
                "previous_num_phonemes": float(previous_phonemes),
                "previous_weighted_PER": float(previous_edits / previous_phonemes),
                "geometry_edit_distance": float(geometry_edits),
                "geometry_num_phonemes": float(geometry_phonemes),
                "geometry_weighted_PER": float(geometry_edits / geometry_phonemes),
                "geometry_better_weighted": bool((geometry_edits / geometry_phonemes) < (previous_edits / previous_phonemes)),
            }
        )
    return pd.DataFrame(rows)


def fit_logistic(train: pd.DataFrame, feature_columns: list[str], l2: float) -> dict[str, object]:
    train = train[~train["selected_is_previous"].astype(bool)].copy()
    y = train["geometry_better_weighted"].astype(float).to_numpy()
    if len(train) < 4 or len(np.unique(y)) < 2:
        return {"kind": "constant", "p": float(y.mean()) if len(y) else 0.0}
    x = train[feature_columns].astype(float).replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0).to_numpy()
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    x = (x - mean) / std
    x = np.c_[np.ones(len(x)), x]
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    weights = np.where(y > 0, len(y) / (2.0 * pos), len(y) / (2.0 * neg))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        probs = expit(x @ beta)
        eps = 1e-8
        loss = -np.sum(weights * (y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps)))
        loss += 0.5 * l2 * np.sum(beta[1:] ** 2)
        grad = x.T @ (weights * (probs - y))
        grad[1:] += l2 * beta[1:]
        return float(loss), grad

    result = minimize(
        fun=lambda beta: objective(beta)[0],
        x0=np.zeros(x.shape[1]),
        jac=lambda beta: objective(beta)[1],
        method="BFGS",
        options={"maxiter": 500},
    )
    return {"kind": "logistic", "beta": result.x, "mean": mean, "std": std, "feature_columns": feature_columns}


def predict_proba(model: dict[str, object], frame: pd.DataFrame) -> np.ndarray:
    if model["kind"] == "constant":
        return np.full(len(frame), float(model["p"]))
    columns = list(model["feature_columns"])
    x = frame[columns].astype(float).replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0).to_numpy()
    x = (x - model["mean"]) / model["std"]
    x = np.c_[np.ones(len(x)), x]
    return expit(x @ model["beta"])


def choose_threshold(train: pd.DataFrame, probs: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.5, probs])
    best: tuple[float, int, float] | None = None
    for threshold in candidates:
        use_geometry = (probs >= threshold) & (~train["selected_is_previous"].to_numpy(dtype=bool))
        per = weighted_per_from_choice(train, use_geometry)
        candidate = (per, -int(use_geometry.sum()), float(threshold))
        if best is None or candidate < best:
            best = candidate
    return 1.0 if best is None else best[2]


def loso_confidence_gate(features: pd.DataFrame, l2: float) -> pd.DataFrame:
    rows = []
    for k, frame in features.groupby("calibration_trials"):
        frame = frame.reset_index(drop=True)
        for idx, test in frame.iterrows():
            train = frame.drop(idx).reset_index(drop=True)
            nonprevious_train = train[~train["selected_is_previous"].astype(bool)].copy()
            model = fit_logistic(nonprevious_train, FEATURE_COLUMNS, l2=l2)
            threshold = choose_threshold(nonprevious_train, predict_proba(model, nonprevious_train)) if not nonprevious_train.empty else 1.0
            prob = float(predict_proba(model, test.to_frame().T)[0])
            use_geometry = (not bool(test["selected_is_previous"])) and prob >= threshold
            rows.append(
                {
                    "calibration_trials": int(k),
                    "session": test["session"],
                    "policy": "logistic_confidence_loso",
                    "use_geometry": bool(use_geometry),
                    "prob_geometry_better": prob,
                    "threshold": threshold,
                }
            )
    return pd.DataFrame(rows)


def policy_decisions(features: pd.DataFrame, l2: float) -> pd.DataFrame:
    base = features[["calibration_trials", "session", "selected_is_previous", "geometry_better_weighted"]].copy()
    policies = {
        "previous": np.zeros(len(base), dtype=bool),
        "geometry": ~base["selected_is_previous"].to_numpy(dtype=bool),
        "higher_confidence": features["confidence_delta"].to_numpy() > 0,
        "lower_entropy": features["entropy_delta"].to_numpy() > 0,
        "lower_blank_rate": features["blank_rate_delta"].to_numpy() < 0,
        "higher_logit_margin": features["logit_margin_delta"].to_numpy() > 0,
    }
    votes = (
        policies["higher_confidence"].astype(int)
        + policies["lower_entropy"].astype(int)
        + policies["lower_blank_rate"].astype(int)
        + policies["higher_logit_margin"].astype(int)
    )
    policies["confidence_vote"] = votes >= 3
    rows = []
    for policy, use_geometry in policies.items():
        use_geometry = use_geometry & (~base["selected_is_previous"].to_numpy(dtype=bool))
        frame = base.copy()
        frame["policy"] = policy
        frame["use_geometry"] = use_geometry
        rows.append(frame)
    rows.append(loso_confidence_gate(features, l2=l2).merge(base, on=["calibration_trials", "session"], how="left"))
    return pd.concat(rows, ignore_index=True)


def summarize_policies(features: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (k, policy), frame in decisions.groupby(["calibration_trials", "policy"]):
        merged = frame.merge(
            features[
                [
                    "calibration_trials",
                    "session",
                    "previous_edit_distance",
                    "previous_num_phonemes",
                    "geometry_edit_distance",
                    "geometry_num_phonemes",
                ]
            ],
            on=["calibration_trials", "session"],
            how="left",
        )
        use_geometry = merged["use_geometry"].to_numpy(dtype=bool)
        rows.append(
            {
                "calibration_trials": int(k),
                "policy": policy,
                "weighted_PER": weighted_per_from_choice(merged, use_geometry),
                "num_sessions": int(len(merged)),
                "overrides_used": int(use_geometry.sum()),
                "correct_overrides": int((merged["use_geometry"] & merged["geometry_better_weighted"]).sum()),
            }
        )
    summary = pd.DataFrame(rows)
    previous = summary[summary["policy"] == "previous"][["calibration_trials", "weighted_PER"]].rename(
        columns={"weighted_PER": "previous_weighted_PER"}
    )
    summary = summary.merge(previous, on="calibration_trials", how="left")
    summary["gain_vs_previous"] = summary["previous_weighted_PER"] - summary["weighted_PER"]
    return summary.sort_values(["calibration_trials", "weighted_PER"]).reset_index(drop=True)


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    policies = [
        "previous",
        "geometry",
        "higher_confidence",
        "lower_entropy",
        "lower_blank_rate",
        "higher_logit_margin",
        "confidence_vote",
        "logistic_confidence_loso",
    ]
    colors = {
        "previous": "#e76f51",
        "geometry": "#457b9d",
        "higher_confidence": "#2a9d8f",
        "lower_entropy": "#8ab17d",
        "lower_blank_rate": "#f4a261",
        "higher_logit_margin": "#9d4edd",
        "confidence_vote": "#264653",
        "logistic_confidence_loso": "#6d597a",
    }
    present = [policy for policy in policies if policy in set(summary["policy"])]
    ks = sorted(summary["calibration_trials"].unique())
    x = np.arange(len(ks))
    width = min(0.09, 0.82 / max(len(present), 1))
    offsets = (np.arange(len(present)) - (len(present) - 1) / 2) * width
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for offset, policy in zip(offsets, present, strict=True):
        values = []
        for k in ks:
            row = summary[(summary["calibration_trials"] == k) & (summary["policy"] == policy)]
            values.append(float(row["weighted_PER"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + offset, values, width=width, label=policy.replace("_", " "), color=colors[policy])
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_ylabel("Phoneme-weighted PER")
    ax.set_title("Exploratory decoder-confidence gate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore unlabeled decoder-confidence gates for T15 previous vs geometry source selection.")
    parser.add_argument("--model-path", type=Path, default=Path("data/external/btt-25-gru-pure-baseline-0-0898"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/hdf5_data_final"))
    parser.add_argument("--csv-path", type=Path, default=Path("data/external/t15_copyTaskData_description.csv"))
    parser.add_argument("--eval-type", choices=["val"], default="val")
    parser.add_argument("--selection-path", type=Path, default=Path("results/tables/t15_kshot_previous_vs_geometry_session_comparison.csv"))
    parser.add_argument("--previous-trials", type=Path, default=Path("results/tables/t15_kshot_previous_source_trial_results.csv"))
    parser.add_argument("--geometry-trials", type=Path, default=Path("results/tables/t15_kshot_geometry_source_trial_results.csv"))
    parser.add_argument("--calibration-trials", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--gpu-number", type=int, default=-1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--l2", type=float, default=10.0)
    parser.add_argument("--output-confidence", type=Path, default=Path("results/tables/_explore_t15_decoder_confidence_gate_features.csv"))
    parser.add_argument("--output-decisions", type=Path, default=Path("results/tables/_explore_t15_decoder_confidence_gate_decisions.csv"))
    parser.add_argument("--output-summary", type=Path, default=Path("results/tables/_explore_t15_decoder_confidence_gate_summary.csv"))
    parser.add_argument("--output-figure", type=Path, default=Path("results/figures/_explore_t15_decoder_confidence_gate_per.png"))
    args = parser.parse_args()

    if args.device == "cuda" and args.gpu_number < 0:
        args.gpu_number = 0
    device = torch.device("cpu") if args.device == "cpu" else resolve_device(args.gpu_number)
    model, model_args = load_official_gru_decoder(ROOT, args.model_path, device)
    _set_use_amp(model_args, enabled=(device.type == "cuda" and bool(model_args["use_amp"]) and not args.disable_amp))
    add_official_model_training_to_path(ROOT)
    from evaluate_model_helpers import load_h5py_file, runSingleDecodingStep

    csv_df = pd.read_csv(args.csv_path)
    model_sessions = list(model_args["dataset"]["sessions"])
    selection = pd.read_csv(args.selection_path)
    selection = selection[selection["calibration_trials"].isin(args.calibration_trials)].copy()
    eval_trials = session_eval_table(build_eval_table(args.previous_trials, args.geometry_trials))
    selection = selection.merge(eval_trials, on=["calibration_trials", "session"], how="inner")

    loaded = {}
    dtype = torch.bfloat16 if device.type == "cuda" and not args.disable_amp else torch.float32
    rows = []
    total = int(selection["calibration_trials"].sum() * 2)
    with tqdm(total=total, desc="calibration confidence", unit="trial-source") as progress:
        for session, session_selection in selection.groupby("session"):
            data = load_h5py_file(str(args.data_dir / session / f"data_{args.eval_type}.hdf5"), csv_df)
            loaded[session] = data
            for _, row in session_selection.iterrows():
                k = int(row["calibration_trials"])
                sources = {
                    "previous": str(row["previous_source_session"]),
                    "geometry": str(row["geometry_source_session"]),
                }
                metrics_by_source: dict[str, list[dict[str, float]]] = {name: [] for name in sources}
                for source_name, source_session in sources.items():
                    input_layer = model_sessions.index(source_session)
                    for trial_idx in range(k):
                        neural_input = np.expand_dims(data["neural_features"][trial_idx], axis=0)
                        neural_input = torch.tensor(neural_input, device=device, dtype=dtype)
                        logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, device)[0]
                        metrics_by_source[source_name].append(logit_quality(logits))
                        progress.update(1)
                feature_row = {
                    "calibration_trials": k,
                    "session": session,
                    "previous_source_session": sources["previous"],
                    "geometry_source_session": sources["geometry"],
                    "selected_is_previous": bool(row["selected_is_previous"]),
                    "geometry_better_weighted": bool(row["geometry_better_weighted"]),
                    "previous_edit_distance": row["previous_edit_distance"],
                    "previous_num_phonemes": row["previous_num_phonemes"],
                    "geometry_edit_distance": row["geometry_edit_distance"],
                    "geometry_num_phonemes": row["geometry_num_phonemes"],
                }
                for source_name, metric_rows in metrics_by_source.items():
                    metric_frame = pd.DataFrame(metric_rows)
                    for column in metric_frame.columns:
                        feature_row[f"{source_name}_{column}"] = float(metric_frame[column].mean())
                feature_row["confidence_delta"] = feature_row["geometry_mean_confidence"] - feature_row["previous_mean_confidence"]
                feature_row["entropy_delta"] = feature_row["previous_entropy"] - feature_row["geometry_entropy"]
                feature_row["blank_rate_delta"] = feature_row["geometry_blank_rate"] - feature_row["previous_blank_rate"]
                feature_row["nonblank_rate_delta"] = feature_row["geometry_nonblank_rate"] - feature_row["previous_nonblank_rate"]
                feature_row["pred_length_delta"] = feature_row["geometry_pred_length"] - feature_row["previous_pred_length"]
                feature_row["logit_margin_delta"] = feature_row["geometry_logit_margin"] - feature_row["previous_logit_margin"]
                feature_row["sentence_label_first"] = _as_plain_string(data["sentence_label"][0]) if len(data["sentence_label"]) else ""
                rows.append(feature_row)

    features = pd.DataFrame(rows)
    decisions = policy_decisions(features, l2=args.l2)
    summary = summarize_policies(features, decisions)

    for path in [args.output_confidence, args.output_decisions, args.output_summary, args.output_figure]:
        path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_confidence, index=False)
    decisions.to_csv(args.output_decisions, index=False)
    summary.to_csv(args.output_summary, index=False)
    plot_summary(summary, args.output_figure)

    print(f"Loaded official GRU checkpoint on {device}.")
    print(f"Wrote {args.output_confidence}")
    print(f"Wrote {args.output_decisions}")
    print(f"Wrote {args.output_summary}")
    print(f"Wrote {args.output_figure}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
