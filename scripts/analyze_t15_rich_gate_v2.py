from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


BASE_FEATURES = [
    "geometry_previous_distance_ratio",
    "margin_abs",
    "margin_fraction",
    "geometry_metric_value",
    "previous_metric_value",
    "selected_lag_days",
    "selected_lag_sessions",
    "lag_diff_days",
    "lag_diff_sessions",
    "confidence_gain",
    "entropy_drop",
    "blank_rate_drop",
]

SUBSPACE_FEATURES = [
    "geometry_minus_previous_mean_principal_angle_deg",
    "geometry_previous_ratio_mean_principal_angle_deg",
    "geometry_minus_previous_subspace_chordal_distance",
    "geometry_previous_ratio_subspace_chordal_distance",
    "geometry_minus_previous_basis_procrustes_error",
    "geometry_previous_ratio_basis_procrustes_error",
]

MOMENT_FEATURES = [
    "geometry_minus_previous_mean_shift_from_source",
    "geometry_previous_ratio_mean_shift_from_source",
    "geometry_minus_previous_scale_shift_from_source",
    "geometry_previous_ratio_scale_shift_from_source",
    "geometry_minus_previous_cov_relative_fro_shift_from_source",
    "geometry_previous_ratio_cov_relative_fro_shift_from_source",
]


def weighted_per(frame: pd.DataFrame, prefix: str) -> float:
    return float(frame[f"{prefix}_edit_distance"].sum() / frame[f"{prefix}_num_phonemes"].sum())


def weighted_per_from_choice(frame: pd.DataFrame, use_geometry: np.ndarray) -> float:
    edit_distance = np.where(use_geometry, frame["geometry_edit_distance"], frame["previous_edit_distance"]).sum()
    num_phonemes = np.where(use_geometry, frame["geometry_num_phonemes"], frame["previous_num_phonemes"]).sum()
    return float(edit_distance / num_phonemes)


def clean_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    x = frame[feature_columns].astype(float).replace([np.inf, -np.inf], np.nan)
    return x.fillna(x.median(numeric_only=True)).fillna(0.0).to_numpy()


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-8] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def fit_logistic(train: pd.DataFrame, feature_columns: list[str], l2: float) -> dict[str, object]:
    non_previous = train[~train["selected_is_previous"].astype(bool)].copy()
    y = non_previous["geometry_better_weighted"].astype(float).to_numpy()
    if len(non_previous) < 4 or len(np.unique(y)) < 2:
        return {"kind": "constant", "p": float(y.mean()) if len(y) else 0.0, "feature_columns": feature_columns}

    x_raw = clean_matrix(non_previous, feature_columns)
    x, _ = standardize(x_raw, x_raw)
    x = np.c_[np.ones(len(x)), x]
    pos = max(y.sum(), 1.0)
    neg = max(len(y) - y.sum(), 1.0)
    weights = np.where(y > 0, len(y) / (2.0 * pos), len(y) / (2.0 * neg))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = x @ beta
        probs = expit(logits)
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
    train_mean = x_raw.mean(axis=0)
    train_std = x_raw.std(axis=0)
    train_std[train_std < 1e-8] = 1.0
    return {
        "kind": "logistic",
        "beta": result.x,
        "mean": train_mean,
        "std": train_std,
        "feature_columns": feature_columns,
    }


def predict_proba(model: dict[str, object], frame: pd.DataFrame) -> np.ndarray:
    if model["kind"] == "constant":
        return np.full(len(frame), float(model["p"]))
    feature_columns = list(model["feature_columns"])
    x = clean_matrix(frame, feature_columns)
    x = (x - model["mean"]) / model["std"]
    x = np.c_[np.ones(len(x)), x]
    return expit(x @ model["beta"])


def choose_threshold(train: pd.DataFrame, probs: np.ndarray, min_overrides: int) -> float:
    candidates = np.unique(np.r_[0.5, probs])
    best: tuple[float, int, float] | None = None
    for threshold in candidates:
        use_geometry = (probs >= threshold) & (~train["selected_is_previous"].to_numpy(dtype=bool))
        if int(use_geometry.sum()) < min_overrides:
            continue
        per = weighted_per_from_choice(train, use_geometry)
        candidate = (per, -int(use_geometry.sum()), float(threshold))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return 1.0
    return best[2]


def run_loso(table: pd.DataFrame, feature_columns: list[str], l2: float, min_overrides: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = []
    for k, frame in table.groupby("calibration_trials"):
        frame = frame.reset_index(drop=True)
        for idx, test_row in frame.iterrows():
            train = frame.drop(idx).reset_index(drop=True)
            model = fit_logistic(train, feature_columns, l2=l2)
            train_nonprev = train[~train["selected_is_previous"].astype(bool)]
            threshold = choose_threshold(train_nonprev, predict_proba(model, train_nonprev), min_overrides=min_overrides)
            prob = float(predict_proba(model, test_row.to_frame().T)[0])
            use_geometry = (not bool(test_row["selected_is_previous"])) and prob >= threshold
            decisions.append(
                {
                    "calibration_trials": int(k),
                    "session": test_row["session"],
                    "prob_geometry_better": prob,
                    "threshold": threshold,
                    "use_geometry": bool(use_geometry),
                    "geometry_better_weighted": bool(test_row["geometry_better_weighted"]),
                    "selected_is_previous": bool(test_row["selected_is_previous"]),
                    "previous_edit_distance": test_row["previous_edit_distance"],
                    "previous_num_phonemes": test_row["previous_num_phonemes"],
                    "geometry_edit_distance": test_row["geometry_edit_distance"],
                    "geometry_num_phonemes": test_row["geometry_num_phonemes"],
                }
            )
    decisions = pd.DataFrame(decisions)
    rows = []
    for k, frame in table.groupby("calibration_trials"):
        rows.append(
            {
                "calibration_trials": int(k),
                "policy": "previous",
                "weighted_PER": weighted_per(frame, "previous"),
                "overrides_used": 0,
                "correct_overrides": 0,
                "num_sessions": int(len(frame)),
            }
        )
        rows.append(
            {
                "calibration_trials": int(k),
                "policy": "geometry",
                "weighted_PER": weighted_per(frame, "geometry"),
                "overrides_used": int((~frame["selected_is_previous"].astype(bool)).sum()),
                "correct_overrides": int((~frame["selected_is_previous"].astype(bool) & frame["geometry_better_weighted"].astype(bool)).sum()),
                "num_sessions": int(len(frame)),
            }
        )
    for k, frame in decisions.groupby("calibration_trials"):
        use_geometry = frame["use_geometry"].to_numpy(dtype=bool)
        rows.append(
            {
                "calibration_trials": int(k),
                "policy": "rich_gate_v2",
                "weighted_PER": weighted_per_from_choice(frame, use_geometry),
                "overrides_used": int(use_geometry.sum()),
                "correct_overrides": int((frame["use_geometry"] & frame["geometry_better_weighted"]).sum()),
                "num_sessions": int(len(frame)),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["calibration_trials", "policy"])
    previous = summary[summary["policy"] == "previous"][["calibration_trials", "weighted_PER"]].rename(
        columns={"weighted_PER": "previous_weighted_PER"}
    )
    summary = summary.merge(previous, on="calibration_trials", how="left")
    summary["gain_vs_previous"] = summary["previous_weighted_PER"] - summary["weighted_PER"]
    return decisions, summary


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    order = ["previous", "geometry", "rich_gate_v2"]
    colors = {"previous": "#e76f51", "geometry": "#457b9d", "rich_gate_v2": "#2a9d8f"}
    ks = sorted(summary["calibration_trials"].unique())
    x = np.arange(len(ks))
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    offsets = np.linspace(-width, width, len(order))
    for offset, policy in zip(offsets, order, strict=True):
        values = []
        for k in ks:
            row = summary[(summary["calibration_trials"] == k) & (summary["policy"] == policy)]
            values.append(float(row["weighted_PER"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + offset, values, width=width, label=policy.replace("_", " "), color=colors[policy])
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_ylabel("Phoneme-weighted PER")
    ax.set_title("Exploratory rich gate v2")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory richer recency-aware gate with subspace features.")
    parser.add_argument("--input", type=Path, default=Path("results/tables/t15_drift_type_fingerprint_joined.csv"))
    parser.add_argument("--feature-set", choices=["base", "base_subspace", "all"], default="base_subspace")
    parser.add_argument("--l2", type=float, default=10.0)
    parser.add_argument("--min-overrides", type=int, default=1)
    parser.add_argument("--output-decisions", type=Path, default=Path("results/tables/_explore_t15_rich_gate_v2_decisions.csv"))
    parser.add_argument("--output-summary", type=Path, default=Path("results/tables/_explore_t15_rich_gate_v2_summary.csv"))
    parser.add_argument("--output-figure", type=Path, default=Path("results/figures/_explore_t15_rich_gate_v2_per.png"))
    args = parser.parse_args()

    if args.feature_set == "base":
        feature_columns = BASE_FEATURES
    elif args.feature_set == "base_subspace":
        feature_columns = BASE_FEATURES + SUBSPACE_FEATURES
    else:
        feature_columns = BASE_FEATURES + SUBSPACE_FEATURES + MOMENT_FEATURES

    table = pd.read_csv(args.input)
    missing = [feature for feature in feature_columns if feature not in table.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    decisions, summary = run_loso(table, feature_columns, l2=args.l2, min_overrides=args.min_overrides)
    for path in [args.output_decisions, args.output_summary, args.output_figure]:
        path.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(args.output_decisions, index=False)
    summary.to_csv(args.output_summary, index=False)
    plot_summary(summary, args.output_figure)
    print(f"features: {len(feature_columns)} {args.feature_set}")
    print(f"Wrote {args.output_decisions}")
    print(f"Wrote {args.output_summary}")
    print(f"Wrote {args.output_figure}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
