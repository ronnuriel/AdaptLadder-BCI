from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_t15_geometry_source_selection_eval import covariance_stats, subspace_metrics, top_basis


POSITIVE_NAME_HINTS = (
    "neural",
    "feature",
    "tx",
    "spike",
    "threshold",
    "power",
    "sbp",
    "binned",
    "rate",
)
NEGATIVE_NAME_HINTS = (
    "audio",
    "sound",
    "label",
    "sentence",
    "cue",
    "clock",
    "time",
    "text",
    "phone",
    "phoneme",
    "block",
)


@dataclass
class FeatureCandidate:
    file_path: Path
    key: str
    shape: tuple[int, ...]
    dtype: str
    score: float


def parse_session_date(path: Path) -> pd.Timestamp | None:
    text = path.stem
    match = re.search(r"(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})", text)
    if not match:
        return None
    return pd.Timestamp(year=int(match.group(1)), month=int(match.group(2)), day=int(match.group(3)))


def iter_mat_files(data_dir: Path) -> list[Path]:
    return sorted([*data_dir.rglob("*.mat"), *data_dir.rglob("*.h5"), *data_dir.rglob("*.hdf5")])


def _is_numeric_array(value) -> bool:
    return isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number)


def _walk_scipy_object(value, prefix: str) -> list[tuple[str, np.ndarray]]:
    arrays = []
    if _is_numeric_array(value):
        arrays.append((prefix, value))
    elif isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).startswith("__"):
                continue
            arrays.extend(_walk_scipy_object(child, f"{prefix}/{key}" if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        for idx, child in enumerate(value):
            arrays.extend(_walk_scipy_object(child, f"{prefix}[{idx}]"))
    elif isinstance(value, np.ndarray) and value.dtype == object:
        for idx, child in np.ndenumerate(value):
            arrays.extend(_walk_scipy_object(child, f"{prefix}{idx}"))
    return arrays


def numeric_arrays_from_file(file_path: Path) -> list[tuple[str, np.ndarray]]:
    try:
        data = loadmat(file_path, simplify_cells=True)
        arrays = []
        for key, value in data.items():
            if key.startswith("__"):
                continue
            arrays.extend(_walk_scipy_object(value, key))
        if arrays:
            return arrays
    except NotImplementedError:
        pass
    except ValueError:
        pass

    arrays = []
    try:
        with h5py.File(file_path, "r") as handle:
            def visitor(name: str, obj) -> None:
                if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number):
                    arrays.append((name, np.asarray(obj)))

            handle.visititems(visitor)
    except OSError:
        return arrays
    return arrays


def score_candidate(name: str, shape: tuple[int, ...], dtype: str) -> float:
    if len(shape) < 2:
        return -np.inf
    total = int(np.prod(shape))
    if total < 1000:
        return -np.inf
    lower = name.lower()
    score = 0.0
    score += sum(3.0 for hint in POSITIVE_NAME_HINTS if hint in lower)
    score -= sum(4.0 for hint in NEGATIVE_NAME_HINTS if hint in lower)
    score += min(np.log10(max(total, 1)), 8.0) / 4.0
    channel_like = [dim for dim in shape if 16 <= dim <= 1024]
    if channel_like:
        score += 2.0
    if any(dim in (128, 256, 512) for dim in shape):
        score += 1.5
    if np.issubdtype(np.dtype(dtype), np.floating):
        score += 0.5
    return score


def candidates_for_file(file_path: Path) -> list[FeatureCandidate]:
    candidates = []
    for key, array in numeric_arrays_from_file(file_path):
        candidates.append(
            FeatureCandidate(
                file_path=file_path,
                key=key,
                shape=tuple(int(dim) for dim in array.shape),
                dtype=str(array.dtype),
                score=score_candidate(key, tuple(int(dim) for dim in array.shape), str(array.dtype)),
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def choose_channel_axis(shape: tuple[int, ...], channel_axis: int | None) -> int:
    if channel_axis is not None:
        return channel_axis if channel_axis >= 0 else len(shape) + channel_axis
    preferred = [idx for idx, dim in enumerate(shape) if dim in (128, 256, 512)]
    if preferred:
        return preferred[-1]
    candidates = [idx for idx, dim in enumerate(shape) if 16 <= dim <= 1024]
    if candidates:
        return min(candidates, key=lambda idx: (abs(shape[idx] - 256), -idx))
    return len(shape) - 1


def array_to_frames(array: np.ndarray, channel_axis: int | None, max_frames: int, seed: int) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim < 2:
        raise ValueError(f"Need at least 2D array, got {array.shape}")
    axis = choose_channel_axis(tuple(array.shape), channel_axis)
    frames = np.moveaxis(array, axis, -1).reshape(-1, array.shape[axis]).astype(np.float64, copy=False)
    frames = frames[np.all(np.isfinite(frames), axis=1)]
    if len(frames) > max_frames:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(frames), size=max_frames, replace=False))
        frames = frames[keep]
    return frames


def load_feature_frames(
    file_path: Path,
    feature_key: str | None,
    channel_axis: int | None,
    max_frames: int,
    seed: int,
) -> tuple[str, tuple[int, ...], np.ndarray]:
    arrays = numeric_arrays_from_file(file_path)
    if feature_key:
        matches = [(key, array) for key, array in arrays if key == feature_key or key.endswith(feature_key)]
        if not matches:
            raise KeyError(f"Feature key {feature_key!r} not found in {file_path}")
        key, array = matches[0]
    else:
        candidates = candidates_for_file(file_path)
        if not candidates:
            raise ValueError(f"No usable numeric feature candidates found in {file_path}")
        candidate = candidates[0]
        key = candidate.key
        array = next(array for array_key, array in arrays if array_key == key)
    return key, tuple(int(dim) for dim in array.shape), array_to_frames(array, channel_axis, max_frames, seed)


def compute_stats(frames: np.ndarray, n_components: int, shrinkage: float) -> dict[str, np.ndarray | int]:
    stats = covariance_stats(frames, shrinkage=shrinkage)
    basis, eigvals = top_basis(stats["cov"], n_components=n_components)
    stats["basis"] = basis
    stats["eigvals"] = eigvals
    return stats


def pairwise_table(sessions: pd.DataFrame, stats: dict[str, dict[str, np.ndarray | int]], source_mode: str) -> pd.DataFrame:
    rows = []
    for target in sessions.to_dict(orient="records"):
        target_stats = stats[target["session"]]
        target_date = pd.Timestamp(target["date"])
        for source in sessions.to_dict(orient="records"):
            if source["session"] == target["session"]:
                continue
            source_date = pd.Timestamp(source["date"])
            if source_mode == "past-only" and source_date >= target_date:
                continue
            source_stats = stats[source["session"]]
            cov_delta = target_stats["cov"] - source_stats["cov"]
            source_cov_norm = max(float(np.linalg.norm(source_stats["cov"], ord="fro")), 1e-8)
            row = {
                "target_session": target["session"],
                "source_session": source["session"],
                "target_date": target_date.date().isoformat(),
                "source_date": source_date.date().isoformat(),
                "days_from_source": int((target_date - source_date).days),
                "abs_days_from_source": int(abs((target_date - source_date).days)),
                "mean_shift_from_source": float(np.linalg.norm(target_stats["mean"] - source_stats["mean"])),
                "scale_shift_from_source": float(np.linalg.norm(target_stats["std"] - source_stats["std"])),
                "diag_cov_shift_from_source": float(
                    np.linalg.norm(np.square(target_stats["std"]) - np.square(source_stats["std"]))
                ),
                "cov_relative_fro_shift_from_source": float(np.linalg.norm(cov_delta, ord="fro") / source_cov_norm),
                "coral_distance_from_source": float(
                    (np.linalg.norm(cov_delta, ord="fro") ** 2)
                    / (4 * target_stats["cov"].shape[0] * target_stats["cov"].shape[0])
                ),
            }
            row.update(subspace_metrics(source_stats["basis"], target_stats["basis"]))
            rows.append(row)
    return pd.DataFrame(rows)


def select_nearest(pairwise: pd.DataFrame, metric: str) -> pd.DataFrame:
    selected = (
        pairwise.sort_values(["target_session", metric, "abs_days_from_source", "source_session"])
        .groupby("target_session", as_index=False)
        .first()
    )
    selected["selection_metric"] = metric
    selected["selection_metric_value"] = selected[metric]
    return selected.sort_values("target_date").reset_index(drop=True)


def add_recency_metadata(selected: pd.DataFrame, pairwise: pd.DataFrame, sessions: pd.DataFrame, metric: str) -> pd.DataFrame:
    ordered = sessions.sort_values("date")["session"].tolist()
    rows = []
    for row in selected.to_dict(orient="records"):
        target = row["target_session"]
        target_idx = ordered.index(target)
        previous = ordered[target_idx - 1] if target_idx > 0 else None
        previous_row = pairwise[(pairwise["target_session"] == target) & (pairwise["source_session"] == previous)]
        metric_rank = (
            pairwise[pairwise["target_session"] == target]
            .sort_values([metric, "abs_days_from_source", "source_session"])
            .reset_index(drop=True)
        )
        previous_rank = np.nan
        if previous is not None and not previous_row.empty:
            previous_rank = int(metric_rank.index[metric_rank["source_session"] == previous][0] + 1)
        rows.append(
            {
                **row,
                "previous_session": previous or "",
                "previous_metric_value": float(previous_row[metric].iloc[0]) if not previous_row.empty else np.nan,
                "previous_abs_days": int(previous_row["abs_days_from_source"].iloc[0]) if not previous_row.empty else np.nan,
                "selected_is_previous": bool(previous is not None and row["source_session"] == previous),
                "selected_lag_sessions": int(target_idx - ordered.index(row["source_session"])),
                "previous_rank_by_geometry": previous_rank,
                "geometry_previous_distance_ratio": float(row[metric] / previous_row[metric].iloc[0])
                if not previous_row.empty and previous_row[metric].iloc[0] != 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_summary(pairwise: pd.DataFrame, selected: pd.DataFrame, metric: str) -> pd.DataFrame:
    rho, p_value = spearmanr(pairwise["abs_days_from_source"], pairwise[metric], nan_policy="omit")
    return pd.DataFrame(
        [
            {
                "num_sessions": int(selected["target_session"].nunique()),
                "num_pairwise_edges": int(len(pairwise)),
                "selection_metric": metric,
                "spearman_days_vs_metric": float(rho),
                "spearman_days_vs_metric_p": float(p_value),
                "selected_previous_sessions": int(selected["selected_is_previous"].sum()),
                "selected_previous_fraction": float(selected["selected_is_previous"].mean()),
                "median_selected_lag_days": float(selected["abs_days_from_source"].median()),
                "mean_selected_lag_days": float(selected["abs_days_from_source"].mean()),
                "median_selected_lag_sessions": float(selected["selected_lag_sessions"].median()),
                "mean_selected_lag_sessions": float(selected["selected_lag_sessions"].mean()),
                "median_previous_rank_by_geometry": float(selected["previous_rank_by_geometry"].median()),
                "mean_previous_rank_by_geometry": float(selected["previous_rank_by_geometry"].mean()),
                "median_geometry_previous_distance_ratio": float(selected["geometry_previous_distance_ratio"].median()),
                "mean_geometry_previous_distance_ratio": float(selected["geometry_previous_distance_ratio"].mean()),
            }
        ]
    )


def plot_distance_vs_days(pairwise: pd.DataFrame, metric: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.scatter(pairwise["abs_days_from_source"], pairwise[metric], s=18, alpha=0.55, color="#457b9d")
    ax.set_xlabel("Absolute days between sessions")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("T12 diagnostic geometry distance vs time")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_lag_histogram(selected: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    bins = np.arange(0, max(1, int(selected["abs_days_from_source"].max())) + 8, 7)
    ax.hist(selected["abs_days_from_source"], bins=bins, color="#7b2cbf", alpha=0.82)
    ax.set_xlabel("Selected source lag in days")
    ax.set_ylabel("Sessions")
    ax.set_title("T12 selected-source lag")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_timeline(selected: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    target_dates = pd.to_datetime(selected["target_date"])
    source_dates = pd.to_datetime(selected["source_date"])
    ax.scatter(target_dates, source_dates, s=34, alpha=0.78, color="#b23a48")
    all_dates = pd.to_datetime(pd.concat([selected["target_date"], selected["source_date"]], ignore_index=True))
    ax.plot([all_dates.min(), all_dates.max()], [all_dates.min(), all_dates.max()], color="0.25", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Target session date")
    ax.set_ylabel("Geometry-selected source date")
    ax.set_title("T12 geometry-selected source timeline")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Geometry-only feasibility validation for T12 diagnosticBlocks MAT files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/t12_diagnosticBlocks"))
    parser.add_argument("--feature-key", default=None, help="Optional exact/suffix MAT variable path to use as neural features.")
    parser.add_argument("--channel-axis", type=int, default=None)
    parser.add_argument("--list-variables", action="store_true", help="Only write candidate variables and exit.")
    parser.add_argument("--source-candidate-mode", choices=["all", "past-only"], default="past-only")
    parser.add_argument("--selection-metric", default="cov_relative_fro_shift_from_source")
    parser.add_argument("--max-frames", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--cov-shrinkage", type=float, default=0.05)
    parser.add_argument("--output-candidates", type=Path, default=Path("results/tables/t12_diagnostic_geometry_feature_candidates.csv"))
    parser.add_argument("--output-sessions", type=Path, default=Path("results/tables/t12_diagnostic_geometry_session_summary.csv"))
    parser.add_argument("--output-pairwise", type=Path, default=Path("results/tables/t12_diagnostic_geometry_pairwise.csv"))
    parser.add_argument("--output-selection", type=Path, default=Path("results/tables/t12_diagnostic_geometry_source_selection.csv"))
    parser.add_argument("--output-summary", type=Path, default=Path("results/tables/t12_diagnostic_geometry_recency_summary.csv"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    files = iter_mat_files(args.data_dir)
    if not files:
        raise FileNotFoundError(
            f"No .mat/.h5/.hdf5 files found under {args.data_dir}. "
            "Download/extract T12 diagnosticBlocks.tar.gz into data/raw/t12_diagnosticBlocks first."
        )

    candidates = []
    for file_path in files:
        date = parse_session_date(file_path)
        for candidate in candidates_for_file(file_path):
            candidates.append(
                {
                    "file": str(candidate.file_path),
                    "session": candidate.file_path.stem,
                    "date": date.date().isoformat() if date is not None else "",
                    "key": candidate.key,
                    "shape": "x".join(str(dim) for dim in candidate.shape),
                    "dtype": candidate.dtype,
                    "score": candidate.score,
                }
            )
    candidates_df = pd.DataFrame(candidates).sort_values(["file", "score"], ascending=[True, False])
    args.output_candidates.parent.mkdir(parents=True, exist_ok=True)
    candidates_df.to_csv(args.output_candidates, index=False)
    if args.list_variables:
        print(f"Wrote {args.output_candidates}")
        print(candidates_df.head(30).to_string(index=False))
        return

    session_rows = []
    stats = {}
    for idx, file_path in enumerate(files):
        date = parse_session_date(file_path)
        if date is None:
            continue
        key, original_shape, frames = load_feature_frames(
            file_path=file_path,
            feature_key=args.feature_key,
            channel_axis=args.channel_axis,
            max_frames=args.max_frames,
            seed=args.seed + idx,
        )
        session = file_path.stem
        n_components = min(args.n_components, frames.shape[1])
        stats[session] = compute_stats(frames, n_components=n_components, shrinkage=args.cov_shrinkage)
        session_rows.append(
            {
                "session": session,
                "date": date.date().isoformat(),
                "file": str(file_path),
                "feature_key": key,
                "original_shape": "x".join(str(dim) for dim in original_shape),
                "n_frames_sampled": int(frames.shape[0]),
                "feature_dim": int(frames.shape[1]),
            }
        )
    sessions = pd.DataFrame(session_rows).sort_values("date").reset_index(drop=True)
    if len(sessions) < 3:
        raise ValueError(f"Need at least 3 dated sessions for geometry validation, found {len(sessions)}.")

    pairwise = pairwise_table(sessions, stats, source_mode=args.source_candidate_mode)
    selected = select_nearest(pairwise, args.selection_metric)
    selected = add_recency_metadata(selected, pairwise, sessions, args.selection_metric)
    summary = build_summary(pairwise, selected, args.selection_metric)

    for path in [args.output_sessions, args.output_pairwise, args.output_selection, args.output_summary]:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    sessions.to_csv(args.output_sessions, index=False)
    pairwise.to_csv(args.output_pairwise, index=False)
    selected.to_csv(args.output_selection, index=False)
    summary.to_csv(args.output_summary, index=False)
    plot_distance_vs_days(pairwise, args.selection_metric, args.figures_dir / "t12_diagnostic_geometry_distance_vs_days.png")
    plot_lag_histogram(selected, args.figures_dir / "t12_diagnostic_selected_source_lag_histogram.png")
    plot_timeline(selected, args.figures_dir / "t12_diagnostic_selected_source_timeline.png")

    print(f"Wrote {args.output_candidates}")
    print(f"Wrote {args.output_sessions}")
    print(f"Wrote {args.output_pairwise}")
    print(f"Wrote {args.output_selection}")
    print(f"Wrote {args.output_summary}")
    print(f"Wrote {args.figures_dir / 't12_diagnostic_geometry_distance_vs_days.png'}")
    print(f"Wrote {args.figures_dir / 't12_diagnostic_selected_source_lag_histogram.png'}")
    print(f"Wrote {args.figures_dir / 't12_diagnostic_selected_source_timeline.png'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
