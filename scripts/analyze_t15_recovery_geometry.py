from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.t15_utils import session_date


def discover_eval_sessions(data_dir: Path, split: str) -> list[str]:
    return sorted(
        [path.parent.name for path in data_dir.glob(f"t15.*/data_{split}.hdf5")],
        key=session_date,
    )


def trial_lengths(file_path: Path) -> list[tuple[str, int]]:
    lengths = []
    with h5py.File(file_path, "r") as handle:
        for key in handle.keys():
            lengths.append((key, int(handle[key]["input_features"].shape[0])))
    return lengths


def sample_session_frames(file_path: Path, max_frames: int, seed: int) -> np.ndarray:
    lengths = trial_lengths(file_path)
    total_frames = sum(length for _key, length in lengths)
    if total_frames == 0:
        raise ValueError(f"No frames found in {file_path}")

    rng = np.random.default_rng(seed)
    if total_frames <= max_frames:
        selected = np.arange(total_frames)
    else:
        selected = np.sort(rng.choice(total_frames, size=max_frames, replace=False))

    frames = []
    selected_pos = 0
    offset = 0
    with h5py.File(file_path, "r") as handle:
        for key, length in lengths:
            stop = offset + length
            local = []
            while selected_pos < len(selected) and offset <= selected[selected_pos] < stop:
                local.append(int(selected[selected_pos] - offset))
                selected_pos += 1
            if local:
                frames.append(handle[key]["input_features"][local])
            offset = stop

    if not frames:
        raise ValueError(f"No sampled frames gathered from {file_path}")
    return np.concatenate(frames, axis=0).astype(np.float64, copy=False)


def covariance_stats(frames: np.ndarray, shrinkage: float) -> dict[str, np.ndarray | int]:
    mean = frames.mean(axis=0)
    centered = frames - mean
    cov = centered.T @ centered / max(len(frames) - 1, 1)
    if shrinkage > 0:
        diagonal = np.diag(np.diag(cov))
        cov = (1 - shrinkage) * cov + shrinkage * diagonal
    std = np.sqrt(np.maximum(np.diag(cov), 1e-8))
    return {
        "n_frames_sampled": int(frames.shape[0]),
        "mean": mean,
        "std": std,
        "cov": cov,
    }


def top_basis(cov: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    return eigvecs[:, :n_components], eigvals[:n_components]


def subspace_metrics(source_basis: np.ndarray, target_basis: np.ndarray) -> dict[str, float]:
    singular_values = np.linalg.svd(source_basis.T @ target_basis, compute_uv=False)
    singular_values = np.clip(singular_values, -1.0, 1.0)
    angles = np.arccos(singular_values)
    projection_distance = np.linalg.norm(source_basis @ source_basis.T - target_basis @ target_basis.T, ord="fro")

    u, _s, vt = np.linalg.svd(target_basis.T @ source_basis)
    rotation = u @ vt
    procrustes_error = np.linalg.norm(target_basis @ rotation - source_basis, ord="fro") / np.sqrt(source_basis.shape[1])

    return {
        "mean_principal_angle_deg": float(np.degrees(np.mean(angles))),
        "max_principal_angle_deg": float(np.degrees(np.max(angles))),
        "subspace_chordal_distance": float(np.sqrt(np.sum(np.sin(angles) ** 2))),
        "projection_fro_distance": float(projection_distance),
        "basis_procrustes_error": float(procrustes_error),
    }


def geometry_table(
    data_dir: Path,
    sessions: list[str],
    source_session: str,
    split: str,
    max_frames: int,
    seed: int,
    n_components: int,
    shrinkage: float,
) -> pd.DataFrame:
    stats = {}
    for idx, session in enumerate(sessions):
        file_path = data_dir / session / f"data_{split}.hdf5"
        frames = sample_session_frames(file_path, max_frames=max_frames, seed=seed + idx)
        stats[session] = covariance_stats(frames, shrinkage=shrinkage)
        basis, eigvals = top_basis(stats[session]["cov"], n_components=n_components)
        stats[session]["basis"] = basis
        stats[session]["eigvals"] = eigvals

    source = stats[source_session]
    source_date = session_date(source_session)
    rows = []
    for session in sessions:
        target = stats[session]
        mean = target["mean"]
        std = target["std"]
        cov = target["cov"]
        src_mean = source["mean"]
        src_std = source["std"]
        src_cov = source["cov"]

        cov_delta = cov - src_cov
        src_cov_norm = max(float(np.linalg.norm(src_cov, ord="fro")), 1e-8)
        row = {
            "session": session,
            "date": session_date(session).isoformat(),
            "source_session": source_session,
            "source_date": source_date.isoformat(),
            "days_from_source": int((session_date(session) - source_date).days),
            "abs_days_from_source": int(abs((session_date(session) - source_date).days)),
            "n_frames_sampled": int(target["n_frames_sampled"]),
            "mean_shift_from_source": float(np.linalg.norm(mean - src_mean)),
            "scale_shift_from_source": float(np.linalg.norm(std - src_std)),
            "diag_cov_shift_from_source": float(np.linalg.norm(np.square(std) - np.square(src_std))),
            "cov_fro_shift_from_source": float(np.linalg.norm(cov_delta, ord="fro")),
            "cov_relative_fro_shift_from_source": float(np.linalg.norm(cov_delta, ord="fro") / src_cov_norm),
            "coral_distance_from_source": float((np.linalg.norm(cov_delta, ord="fro") ** 2) / (4 * cov.shape[0] * cov.shape[0])),
        }
        row.update(subspace_metrics(source["basis"], target["basis"]))
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_session_method_frame(summary_path: Path, prefix: str) -> pd.DataFrame:
    summary = pd.read_csv(summary_path)
    keep_methods = [method for method in summary["method"].unique()]
    frames = []
    for method in keep_methods:
        method_frame = summary[summary["method"] == method][
            ["calibration_trials", "session", "n_trials", "mean_PER", "median_PER"]
        ].rename(
            columns={
                "n_trials": f"{prefix}_{method}_n_trials",
                "mean_PER": f"{prefix}_{method}_mean_PER",
                "median_PER": f"{prefix}_{method}_median_PER",
            }
        )
        frames.append(method_frame)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["calibration_trials", "session"], how="outer")
    return merged


def join_recovery_and_geometry(
    geometry: pd.DataFrame,
    native_summary_path: Path,
    cross_summary_path: Path,
    affine_summary_path: Path,
    input_layer_summary_path: Path,
) -> pd.DataFrame:
    native = pd.read_csv(native_summary_path)[["session", "mean_PER", "median_PER"]].rename(
        columns={"mean_PER": "native_full_mean_PER", "median_PER": "native_full_median_PER"}
    )
    cross = pd.read_csv(cross_summary_path)[["session", "mean_PER", "median_PER"]].rename(
        columns={"mean_PER": "cross_full_mean_PER", "median_PER": "cross_full_median_PER"}
    )
    full = geometry.merge(native, on="session", how="left").merge(cross, on="session", how="left")
    full["cross_full_delta_mean_PER"] = full["cross_full_mean_PER"] - full["native_full_mean_PER"]

    affine = prepare_session_method_frame(affine_summary_path, "affine")
    input_layer = prepare_session_method_frame(input_layer_summary_path, "input_layer")
    joined = affine.merge(input_layer, on=["calibration_trials", "session"], how="outer")
    joined = joined.merge(full, on="session", how="left")

    # Both calibration summaries use identical native/none subsets. Prefer affine names for shared baselines.
    native_col = "affine_native-day_mean_PER"
    none_col = "affine_none_mean_PER"
    if native_col in joined and none_col in joined:
        joined["calib_native_mean_PER"] = joined[native_col]
        joined["calib_none_mean_PER"] = joined[none_col]
        joined["calib_gap_mean_PER"] = joined["calib_none_mean_PER"] - joined["calib_native_mean_PER"]

    method_columns = {
        "moment_match": "affine_moment_match_to_source_mean_PER",
        "diagonal_affine": "affine_diagonal_affine_mean_PER",
        "input_layer": "input_layer_input_layer_mean_PER",
    }
    for method, col in method_columns.items():
        if col not in joined:
            continue
        joined[f"{method}_gain_mean_PER"] = joined["calib_none_mean_PER"] - joined[col]
        joined[f"{method}_recovery_fraction"] = joined[f"{method}_gain_mean_PER"] / joined["calib_gap_mean_PER"].replace(0, np.nan)
        joined[f"{method}_harmed"] = joined[col] > joined["calib_none_mean_PER"]
    return joined.sort_values(["calibration_trials", "date"]).reset_index(drop=True)


def _corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_1d(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    return _corr_1d(x_rank, y_rank)


def permutation_p_value(
    x: np.ndarray,
    y: np.ndarray,
    observed: float,
    rng: np.random.Generator,
    n_permutations: int,
    method: str,
) -> float:
    if n_permutations <= 0 or not np.isfinite(observed):
        return np.nan

    hits = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(y)
        if method == "spearman":
            value = _spearman_1d(x, shuffled)
        else:
            value = _corr_1d(x, shuffled)
        if np.isfinite(value) and abs(value) >= abs(observed):
            hits += 1
    return float((hits + 1) / (n_permutations + 1))


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
    method: str,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if n_bootstrap <= 0 or len(x) < 4:
        return np.nan, np.nan

    values = []
    n = len(x)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if method == "spearman":
            value = _spearman_1d(x[idx], y[idx])
        else:
            value = _corr_1d(x[idx], y[idx])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return np.nan, np.nan
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def correlation_table(
    joined: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
    n_permutations: int,
) -> pd.DataFrame:
    geometry_cols = [
        "abs_days_from_source",
        "mean_shift_from_source",
        "scale_shift_from_source",
        "diag_cov_shift_from_source",
        "cov_relative_fro_shift_from_source",
        "coral_distance_from_source",
        "mean_principal_angle_deg",
        "max_principal_angle_deg",
        "subspace_chordal_distance",
        "projection_fro_distance",
        "basis_procrustes_error",
    ]
    outcome_cols = [
        "cross_full_mean_PER",
        "cross_full_delta_mean_PER",
        "calib_none_mean_PER",
        "diagonal_affine_gain_mean_PER",
        "diagonal_affine_recovery_fraction",
        "input_layer_gain_mean_PER",
        "input_layer_recovery_fraction",
        "moment_match_gain_mean_PER",
        "moment_match_recovery_fraction",
    ]
    rows = []
    for k, frame in joined.groupby("calibration_trials", dropna=False):
        for metric in geometry_cols:
            for outcome in outcome_cols:
                if metric not in frame or outcome not in frame:
                    continue
                pair = frame[[metric, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(pair) < 3:
                    continue
                x = pair[metric].to_numpy(dtype=float)
                y = pair[outcome].to_numpy(dtype=float)
                pearson_r = _corr_1d(x, y)
                spearman_r = _spearman_1d(x, y)
                pearson_low, pearson_high = bootstrap_ci(x, y, rng, n_bootstrap, method="pearson")
                spearman_low, spearman_high = bootstrap_ci(x, y, rng, n_bootstrap, method="spearman")
                rows.append(
                    {
                        "calibration_trials": int(k) if pd.notna(k) else np.nan,
                        "geometry_metric": metric,
                        "outcome": outcome,
                        "n": int(len(pair)),
                        "pearson_r": pearson_r,
                        "pearson_p_permutation": permutation_p_value(
                            x, y, pearson_r, rng, n_permutations, method="pearson"
                        ),
                        "pearson_bootstrap_low": pearson_low,
                        "pearson_bootstrap_high": pearson_high,
                        "spearman_r": spearman_r,
                        "spearman_p_permutation": permutation_p_value(
                            x, y, spearman_r, rng, n_permutations, method="spearman"
                        ),
                        "spearman_bootstrap_low": spearman_low,
                        "spearman_bootstrap_high": spearman_high,
                    }
                )
    return pd.DataFrame(rows).sort_values(["calibration_trials", "outcome", "spearman_r"], ascending=[True, True, False])


def near_far_table(joined: pd.DataFrame) -> pd.DataFrame:
    split_metrics = [
        "cov_relative_fro_shift_from_source",
        "abs_days_from_source",
        "mean_principal_angle_deg",
    ]
    methods = {
        "moment_match": "moment_match",
        "diagonal_affine": "diagonal_affine",
        "input_layer": "input_layer",
    }
    rows = []
    for k, k_frame in joined.groupby("calibration_trials", dropna=False):
        for split_metric in split_metrics:
            if split_metric not in k_frame:
                continue
            threshold = float(k_frame[split_metric].replace([np.inf, -np.inf], np.nan).median())
            for method, prefix in methods.items():
                recovery_col = f"{prefix}_recovery_fraction"
                gain_col = f"{prefix}_gain_mean_PER"
                harmed_col = f"{prefix}_harmed"
                method_per_col = {
                    "moment_match": "affine_moment_match_to_source_mean_PER",
                    "diagonal_affine": "affine_diagonal_affine_mean_PER",
                    "input_layer": "input_layer_input_layer_mean_PER",
                }[method]
                needed = [
                    split_metric,
                    recovery_col,
                    gain_col,
                    "calib_none_mean_PER",
                    "calib_native_mean_PER",
                    method_per_col,
                ]
                if not all(col in k_frame for col in needed):
                    continue
                valid = k_frame[needed + ([harmed_col] if harmed_col in k_frame else [])].replace(
                    [np.inf, -np.inf], np.nan
                ).dropna(subset=[split_metric, recovery_col, gain_col, "calib_none_mean_PER", method_per_col])
                if valid.empty:
                    continue
                for group_name, group_frame in [
                    ("near", valid[valid[split_metric] <= threshold]),
                    ("far", valid[valid[split_metric] > threshold]),
                ]:
                    if group_frame.empty:
                        continue
                    rows.append(
                        {
                            "calibration_trials": int(k) if pd.notna(k) else np.nan,
                            "split_metric": split_metric,
                            "split_threshold": threshold,
                            "distance_group": group_name,
                            "method": method,
                            "n_sessions": int(len(group_frame)),
                            "mean_native_PER": float(group_frame["calib_native_mean_PER"].mean()),
                            "mean_cross_day_none_PER": float(group_frame["calib_none_mean_PER"].mean()),
                            "mean_method_PER": float(group_frame[method_per_col].mean()),
                            "mean_gain_PER": float(group_frame[gain_col].mean()),
                            "median_gain_PER": float(group_frame[gain_col].median()),
                            "mean_recovery_fraction": float(group_frame[recovery_col].mean()),
                            "median_recovery_fraction": float(group_frame[recovery_col].median()),
                            "sessions_improved_vs_none": int((group_frame[gain_col] > 0).sum()),
                            "sessions_harmed_vs_none": int((group_frame[gain_col] < 0).sum()),
                        }
                    )
    return pd.DataFrame(rows).sort_values(["split_metric", "calibration_trials", "method", "distance_group"])


def scatter_by_k(joined: pd.DataFrame, x_col: str, y_col: str, output_path: Path, title: str, ylabel: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for k, frame in joined.groupby("calibration_trials"):
        plot_frame = frame[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if plot_frame.empty:
            continue
        ax.scatter(plot_frame[x_col], plot_frame[y_col] * 100, label=f"K={int(k)}", alpha=0.78, s=38)
    ax.axhline(0, color="0.25", linewidth=1.0)
    ax.set_xlabel(x_col.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_near_far_recovery(near_far: pd.DataFrame, output_path: Path) -> None:
    plot_frame = near_far[
        (near_far["split_metric"] == "cov_relative_fro_shift_from_source")
        & (near_far["method"].isin(["diagonal_affine", "input_layer"]))
    ].copy()
    if plot_frame.empty:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ks = sorted(plot_frame["calibration_trials"].dropna().unique())
    methods = ["diagonal_affine", "input_layer"]
    groups = ["near", "far"]
    x = np.arange(len(ks), dtype=float)
    width = 0.18

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    offsets = {
        ("diagonal_affine", "near"): -1.5 * width,
        ("diagonal_affine", "far"): -0.5 * width,
        ("input_layer", "near"): 0.5 * width,
        ("input_layer", "far"): 1.5 * width,
    }
    colors = {
        ("diagonal_affine", "near"): "#6baed6",
        ("diagonal_affine", "far"): "#2171b5",
        ("input_layer", "near"): "#74c476",
        ("input_layer", "far"): "#238b45",
    }
    labels = {
        ("diagonal_affine", "near"): "Diagonal near",
        ("diagonal_affine", "far"): "Diagonal far",
        ("input_layer", "near"): "Input-layer near",
        ("input_layer", "far"): "Input-layer far",
    }
    for method in methods:
        for group in groups:
            values = []
            for k in ks:
                row = plot_frame[
                    (plot_frame["calibration_trials"] == k)
                    & (plot_frame["method"] == method)
                    & (plot_frame["distance_group"] == group)
                ]
                values.append(float(row["mean_recovery_fraction"].iloc[0] * 100) if not row.empty else np.nan)
            ax.bar(x + offsets[(method, group)], values, width=width, color=colors[(method, group)], label=labels[(method, group)])

    ax.axhline(0, color="0.25", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={int(k)}" for k in ks])
    ax.set_ylabel("Recovered native-day gap (%)")
    ax.set_title("Recovery is higher for covariance-near target sessions")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Join T15 drift geometry with adaptation recovery.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/hdf5_data_final"))
    parser.add_argument("--stats-split", default="train")
    parser.add_argument("--source-session", default="t15.2023.11.26")
    parser.add_argument("--max-frames", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--cov-shrinkage", type=float, default=0.05)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--native-summary", type=Path, default=Path("results/tables/t15_decoder_probe_session_summary.csv"))
    parser.add_argument(
        "--cross-summary",
        type=Path,
        default=Path("results/tables/t15_decoder_probe_cross_day_source_2023_11_26_session_summary.csv"),
    )
    parser.add_argument(
        "--affine-summary",
        type=Path,
        default=Path("results/tables/t15_affine_calibration_session_summary_source_middle_epochs5.csv"),
    )
    parser.add_argument(
        "--input-layer-summary",
        type=Path,
        default=Path("results/tables/t15_input_layer_calibration_session_summary_source_middle_epochs5.csv"),
    )
    parser.add_argument(
        "--output-joined",
        type=Path,
        default=Path("results/tables/t15_session_recovery_geometry_joined.csv"),
    )
    parser.add_argument(
        "--output-correlations",
        type=Path,
        default=Path("results/tables/t15_recovery_geometry_correlations.csv"),
    )
    parser.add_argument(
        "--output-near-far",
        type=Path,
        default=Path("results/tables/t15_recovery_geometry_near_far.csv"),
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    sessions = discover_eval_sessions(args.data_dir, split="val")
    if args.source_session not in sessions:
        raise ValueError(f"Source session {args.source_session!r} does not have a validation split.")

    geometry = geometry_table(
        data_dir=args.data_dir,
        sessions=sessions,
        source_session=args.source_session,
        split=args.stats_split,
        max_frames=args.max_frames,
        seed=args.seed,
        n_components=args.n_components,
        shrinkage=args.cov_shrinkage,
    )
    joined = join_recovery_and_geometry(
        geometry=geometry,
        native_summary_path=args.native_summary,
        cross_summary_path=args.cross_summary,
        affine_summary_path=args.affine_summary,
        input_layer_summary_path=args.input_layer_summary,
    )
    correlations = correlation_table(
        joined,
        rng=rng,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
    )
    near_far = near_far_table(joined)

    args.output_joined.parent.mkdir(parents=True, exist_ok=True)
    args.output_correlations.parent.mkdir(parents=True, exist_ok=True)
    args.output_near_far.parent.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(args.output_joined, index=False)
    correlations.to_csv(args.output_correlations, index=False)
    near_far.to_csv(args.output_near_far, index=False)

    scatter_by_k(
        joined,
        x_col="mean_principal_angle_deg",
        y_col="input_layer_recovery_fraction",
        output_path=args.figures_dir / "t15_recovery_vs_subspace_angle.png",
        title="Input-layer recovery vs neural subspace angle",
        ylabel="Input-layer recovered gap (%)",
    )
    scatter_by_k(
        joined,
        x_col="calib_none_mean_PER",
        y_col="input_layer_recovery_fraction",
        output_path=args.figures_dir / "t15_recovery_vs_cross_day_per.png",
        title="Input-layer recovery vs cross-day PER",
        ylabel="Input-layer recovered gap (%)",
    )
    scatter_by_k(
        joined,
        x_col="abs_days_from_source",
        y_col="input_layer_recovery_fraction",
        output_path=args.figures_dir / "t15_recovery_vs_time_distance.png",
        title="Input-layer recovery vs temporal distance",
        ylabel="Input-layer recovered gap (%)",
    )
    plot_near_far_recovery(
        near_far,
        output_path=args.figures_dir / "t15_near_far_recovery_by_covariance.png",
    )

    print(f"Wrote {args.output_joined}")
    print(f"Wrote {args.output_correlations}")
    print(f"Wrote {args.output_near_far}")
    print(f"Wrote {args.figures_dir / 't15_recovery_vs_subspace_angle.png'}")
    print(f"Wrote {args.figures_dir / 't15_recovery_vs_cross_day_per.png'}")
    print(f"Wrote {args.figures_dir / 't15_recovery_vs_time_distance.png'}")
    print(f"Wrote {args.figures_dir / 't15_near_far_recovery_by_covariance.png'}")
    focus = correlations[correlations["outcome"].isin(["input_layer_recovery_fraction", "diagonal_affine_recovery_fraction"])]
    print(focus.sort_values("spearman_r", key=lambda s: s.abs(), ascending=False).head(20).to_string(index=False))
    print()
    print(
        near_far[
            (near_far["split_metric"] == "cov_relative_fro_shift_from_source")
            & (near_far["method"].isin(["diagonal_affine", "input_layer"]))
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
