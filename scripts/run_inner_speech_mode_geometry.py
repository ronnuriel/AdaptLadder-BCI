from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_t15_geometry_source_selection_eval import covariance_stats


@dataclass
class GeometryGroup:
    participant: str
    session: str
    date: str
    behavior: str
    split: str
    n_trials: int
    n_frames: int
    feature_dim: int
    stats: dict[str, np.ndarray | int]


def iter_mat_files(data_dir: Path) -> list[Path]:
    return sorted(path for path in data_dir.rglob("*.mat") if not path.name.startswith("._"))


def parse_participant_session(path: Path) -> tuple[str, str, str]:
    stem = path.stem
    match = re.search(r"(t\d+)\.(20\d{2})\.(\d{2})\.(\d{2})", stem, flags=re.IGNORECASE)
    if not match:
        return stem.split("_")[0], stem, ""
    participant = match.group(1).lower()
    date = f"{match.group(2)}-{match.group(3)}-{match.group(4)}"
    return participant, stem, date


def normalize_cue(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        value = np.asarray(value).ravel()
        if len(value) == 1:
            return normalize_cue(value[0])
        return " ".join(normalize_cue(item) for item in value)
    return str(value).strip()


def behavior_from_cue(cue: str) -> str:
    lower = cue.lower()
    if "donothing" in lower or "do nothing" in lower:
        return "do_nothing"
    if "passivelistening" in lower or lower.endswith("$0"):
        return "listening"
    if "imaginedlistening" in lower:
        return "imagined_listening"
    if "listen" in lower:
        return "listening"
    if "attempted" in lower or "mimed" in lower:
        return "attempted"
    if "imagined" in lower:
        return "imagined"
    return "unknown"


def get_feature(data: dict, feature_key: str) -> np.ndarray:
    if feature_key not in data:
        keys = ", ".join(sorted(key for key in data if not key.startswith("__")))
        raise KeyError(f"Feature key {feature_key!r} not found. Available keys: {keys}")
    array = np.asarray(data[feature_key])
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix for {feature_key}, got {array.shape}")
    return array.astype(np.float64, copy=False)


def epoch_frames(features: np.ndarray, epochs: np.ndarray, trial_indices: np.ndarray) -> np.ndarray:
    pieces = []
    for trial_idx in trial_indices:
        start, end = [int(v) for v in epochs[int(trial_idx)]]
        start = max(start - 1, 0)
        end = min(end, len(features))
        if end > start:
            pieces.append(features[start:end])
    if not pieces:
        return np.empty((0, features.shape[1]), dtype=np.float64)
    frames = np.concatenate(pieces, axis=0)
    frames = frames[np.all(np.isfinite(frames), axis=1)]
    return frames


def sample_frames(frames: np.ndarray, max_frames: int, rng: np.random.Generator) -> np.ndarray:
    if len(frames) <= max_frames:
        return frames
    keep = np.sort(rng.choice(len(frames), size=max_frames, replace=False))
    return frames[keep]


def build_groups(
    file_path: Path,
    feature_key: str,
    epoch_name: str,
    max_frames_per_group: int,
    min_trials_per_split: int,
    shrinkage: float,
    seed: int,
) -> tuple[list[GeometryGroup], pd.DataFrame]:
    participant, session, date = parse_participant_session(file_path)
    data = loadmat(file_path, simplify_cells=True)
    features = get_feature(data, feature_key)
    epochs = np.asarray(data[epoch_name])
    trial_cues = np.asarray(data["trialCues"]).astype(int).ravel()
    cue_list = [normalize_cue(cue) for cue in np.asarray(data["cueList"]).ravel()]
    rng = np.random.default_rng(seed)

    trial_rows = []
    for trial_idx, cue_id in enumerate(trial_cues):
        cue = cue_list[cue_id - 1]
        trial_rows.append(
            {
                "participant": participant,
                "session": session,
                "date": date,
                "trial_index": trial_idx,
                "cue_id": int(cue_id),
                "cue": cue,
                "behavior": behavior_from_cue(cue),
            }
        )
    trials = pd.DataFrame(trial_rows)

    groups = []
    for behavior, behavior_trials in trials.groupby("behavior", sort=True):
        if behavior == "unknown":
            continue
        indices = behavior_trials["trial_index"].to_numpy(dtype=int)
        if len(indices) < 2 * min_trials_per_split:
            continue
        for split_name, split_indices in [("a", indices[::2]), ("b", indices[1::2])]:
            if len(split_indices) < min_trials_per_split:
                continue
            frames = epoch_frames(features, epochs, split_indices)
            if len(frames) == 0:
                continue
            frames = sample_frames(frames, max_frames=max_frames_per_group, rng=rng)
            stats = covariance_stats(frames, shrinkage=shrinkage)
            groups.append(
                GeometryGroup(
                    participant=participant,
                    session=session,
                    date=date,
                    behavior=behavior,
                    split=split_name,
                    n_trials=int(len(split_indices)),
                    n_frames=int(len(frames)),
                    feature_dim=int(frames.shape[1]),
                    stats=stats,
                )
            )
    return groups, trials


def distance_row(target: GeometryGroup, source: GeometryGroup) -> dict[str, object]:
    cov_delta = target.stats["cov"] - source.stats["cov"]
    source_cov_norm = max(float(np.linalg.norm(source.stats["cov"], ord="fro")), 1e-8)
    return {
        "target_participant": target.participant,
        "target_session": target.session,
        "target_date": target.date,
        "target_behavior": target.behavior,
        "target_split": target.split,
        "source_participant": source.participant,
        "source_session": source.session,
        "source_behavior": source.behavior,
        "source_split": source.split,
        "same_participant": target.participant == source.participant,
        "same_behavior": target.behavior == source.behavior,
        "mean_shift": float(np.linalg.norm(target.stats["mean"] - source.stats["mean"])),
        "scale_shift": float(np.linalg.norm(target.stats["std"] - source.stats["std"])),
        "cov_relative_fro": float(np.linalg.norm(cov_delta, ord="fro") / source_cov_norm),
        "coral_distance": float(
            (np.linalg.norm(cov_delta, ord="fro") ** 2)
            / (4 * target.stats["cov"].shape[0] * target.stats["cov"].shape[0])
        ),
    }


def pairwise_groups(groups: list[GeometryGroup]) -> pd.DataFrame:
    rows = []
    for target in groups:
        for source in groups:
            if target.participant != source.participant:
                continue
            if target.behavior == source.behavior and target.split == source.split:
                continue
            rows.append(distance_row(target, source))
    return pd.DataFrame(rows)


def nearest_behavior(pairwise: pd.DataFrame, metric: str) -> pd.DataFrame:
    nearest = (
        pairwise.sort_values(
            ["target_participant", "target_behavior", "target_split", metric, "source_behavior", "source_split"]
        )
        .groupby(["target_participant", "target_behavior", "target_split"], as_index=False)
        .first()
    )
    nearest["nearest_correct_behavior"] = nearest["same_behavior"]
    return nearest


def summarize(pairwise: pd.DataFrame, nearest: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    scopes = [("all", nearest, pairwise)]
    for participant in sorted(nearest["target_participant"].unique()):
        scopes.append(
            (
                participant,
                nearest[nearest["target_participant"] == participant],
                pairwise[pairwise["target_participant"] == participant],
            )
        )
    for scope, nearest_frame, pair_frame in scopes:
        same = pair_frame[pair_frame["same_behavior"]]
        different = pair_frame[~pair_frame["same_behavior"]]
        rows.append(
            {
                "scope": scope,
                "metric": metric,
                "n_targets": int(len(nearest_frame)),
                "n_pairwise": int(len(pair_frame)),
                "nearest_behavior_accuracy": float(nearest_frame["nearest_correct_behavior"].mean()),
                "chance_behavior_accuracy": float(1.0 / nearest_frame["target_behavior"].nunique()),
                "mean_same_behavior_distance": float(same[metric].mean()),
                "mean_different_behavior_distance": float(different[metric].mean()),
                "between_within_ratio": float(different[metric].mean() / same[metric].mean()),
                "median_same_behavior_distance": float(same[metric].median()),
                "median_different_behavior_distance": float(different[metric].median()),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = summary[summary["scope"] != "all"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].bar(frame["scope"], frame["nearest_behavior_accuracy"], color="#2a9d8f", label="nearest")
    axes[0].bar(frame["scope"], frame["chance_behavior_accuracy"], color="none", edgecolor="0.25", label="chance")
    axes[0].set_ylabel("Nearest-behavior accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].bar(frame["scope"], frame["between_within_ratio"], color="#457b9d")
    axes[1].axhline(1.0, color="0.25", linewidth=1)
    axes[1].set_ylabel("Between / within distance")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Inner-speech interleaved behavior geometry")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Geometry feasibility scan for inner-speech interleaved behavior modes.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/inner_speech/interleavedVerbalBehaviors"))
    parser.add_argument("--feature-key", default="binnedTX")
    parser.add_argument("--epoch", choices=["go", "delay"], default="go")
    parser.add_argument("--metric", default="cov_relative_fro")
    parser.add_argument("--max-frames-per-group", type=int, default=40000)
    parser.add_argument("--min-trials-per-split", type=int, default=10)
    parser.add_argument("--cov-shrinkage", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output-groups", type=Path, default=Path("results/tables/inner_interleaved_mode_geometry_groups.csv"))
    parser.add_argument("--output-trials", type=Path, default=Path("results/tables/inner_interleaved_mode_geometry_trials.csv"))
    parser.add_argument("--output-pairwise", type=Path, default=Path("results/tables/inner_interleaved_mode_geometry_pairwise.csv"))
    parser.add_argument("--output-nearest", type=Path, default=Path("results/tables/inner_interleaved_mode_geometry_nearest.csv"))
    parser.add_argument("--output-summary", type=Path, default=Path("results/tables/inner_interleaved_mode_geometry_summary.csv"))
    parser.add_argument("--output-figure", type=Path, default=Path("results/figures/inner_interleaved_mode_geometry_summary.png"))
    args = parser.parse_args()

    epoch_name = "goTrialEpochs" if args.epoch == "go" else "delayTrialEpochs"
    files = iter_mat_files(args.data_dir)
    if not files:
        raise FileNotFoundError(f"No MAT files found under {args.data_dir}")

    all_groups: list[GeometryGroup] = []
    all_trials = []
    for idx, file_path in enumerate(files):
        groups, trials = build_groups(
            file_path=file_path,
            feature_key=args.feature_key,
            epoch_name=epoch_name,
            max_frames_per_group=args.max_frames_per_group,
            min_trials_per_split=args.min_trials_per_split,
            shrinkage=args.cov_shrinkage,
            seed=args.seed + idx,
        )
        all_groups.extend(groups)
        all_trials.append(trials)

    if not all_groups:
        raise ValueError("No behavior groups were built. Try lowering --min-trials-per-split.")

    group_rows = [
        {
            "participant": group.participant,
            "session": group.session,
            "date": group.date,
            "behavior": group.behavior,
            "split": group.split,
            "n_trials": group.n_trials,
            "n_frames": group.n_frames,
            "feature_dim": group.feature_dim,
        }
        for group in all_groups
    ]
    groups_df = pd.DataFrame(group_rows).sort_values(["participant", "behavior", "split"])
    trials_df = pd.concat(all_trials, ignore_index=True)
    pairwise = pairwise_groups(all_groups)
    nearest = nearest_behavior(pairwise, args.metric)
    summary = summarize(pairwise, nearest, args.metric)

    for path in [
        args.output_groups,
        args.output_trials,
        args.output_pairwise,
        args.output_nearest,
        args.output_summary,
        args.output_figure,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
    groups_df.to_csv(args.output_groups, index=False)
    trials_df.to_csv(args.output_trials, index=False)
    pairwise.to_csv(args.output_pairwise, index=False)
    nearest.to_csv(args.output_nearest, index=False)
    summary.to_csv(args.output_summary, index=False)
    plot_summary(summary, args.output_figure)

    print(f"Wrote {args.output_groups}")
    print(f"Wrote {args.output_trials}")
    print(f"Wrote {args.output_pairwise}")
    print(f"Wrote {args.output_nearest}")
    print(f"Wrote {args.output_summary}")
    print(f"Wrote {args.output_figure}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
