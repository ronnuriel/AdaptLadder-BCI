from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .t15_utils import available_splits, discover_sessions, finite_std, iter_hdf5_trials, session_date


def summarize_t15_dataset(data_dir: Path, splits: tuple[str, ...] = ("train", "val", "test")) -> pd.DataFrame:
    rows = []
    for session in discover_sessions(data_dir):
        session_dir = data_dir / session
        split_counts: dict[str, int] = {}
        lengths: list[int] = []
        feature_dim = None
        has_seq_class_ids = False
        has_transcription = False
        has_sentence_label = False

        for split in splits:
            file_path = session_dir / f"data_{split}.hdf5"
            if not file_path.exists():
                split_counts[split] = 0
                continue

            split_count = 0
            for record, _features in iter_hdf5_trials(file_path):
                split_count += 1
                lengths.append(record.n_time_steps)
                feature_dim = record.feature_dim
                has_seq_class_ids = has_seq_class_ids or record.has_seq_class_ids
                has_transcription = has_transcription or record.has_transcription
                has_sentence_label = has_sentence_label or record.has_sentence_label
            split_counts[split] = split_count

        rows.append(
            {
                "session": session,
                "date": session_date(session).isoformat(),
                "num_trials": int(sum(split_counts.values())),
                "train_trials": int(split_counts.get("train", 0)),
                "val_trials": int(split_counts.get("val", 0)),
                "test_trials": int(split_counts.get("test", 0)),
                "avg_trial_len": float(np.mean(lengths)) if lengths else np.nan,
                "feature_dim": int(feature_dim) if feature_dim is not None else np.nan,
                "has_seq_class_ids": bool(has_seq_class_ids),
                "has_transcription": bool(has_transcription),
                "has_sentence_label": bool(has_sentence_label),
                "available_splits": ",".join(available_splits(session_dir)),
            }
        )

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def compute_session_stats(data_dir: Path, splits: tuple[str, ...] = ("train",)) -> dict[str, dict[str, np.ndarray | int | float | str]]:
    stats: dict[str, dict[str, np.ndarray | int | float | str]] = {}
    for session in discover_sessions(data_dir):
        n_frames = 0
        n_trials = 0
        sum_x = None
        sum_x2 = None

        for split in splits:
            file_path = data_dir / session / f"data_{split}.hdf5"
            if not file_path.exists():
                continue
            for _record, features in iter_hdf5_trials(file_path):
                x = np.asarray(features, dtype=np.float64)
                if sum_x is None:
                    sum_x = np.zeros(x.shape[1], dtype=np.float64)
                    sum_x2 = np.zeros(x.shape[1], dtype=np.float64)
                sum_x += x.sum(axis=0)
                sum_x2 += np.square(x).sum(axis=0)
                n_frames += x.shape[0]
                n_trials += 1

        if n_frames == 0 or sum_x is None or sum_x2 is None:
            continue

        mean = sum_x / n_frames
        std = finite_std(sum_x, sum_x2, n_frames)
        stats[session] = {
            "session": session,
            "date": session_date(session).isoformat(),
            "n_frames": int(n_frames),
            "n_trials": int(n_trials),
            "mean": mean,
            "std": std,
        }

    return stats


def drift_metric_table(stats: dict[str, dict[str, np.ndarray | int | float | str]]) -> pd.DataFrame:
    sessions = sorted(stats, key=lambda item: stats[item]["date"])
    if not sessions:
        return pd.DataFrame()

    source = sessions[0]
    source_date = session_date(source)
    prev = None
    rows = []
    for session in sessions:
        mean = stats[session]["mean"]
        std = stats[session]["std"]
        assert isinstance(mean, np.ndarray)
        assert isinstance(std, np.ndarray)

        src_mean = stats[source]["mean"]
        src_std = stats[source]["std"]
        assert isinstance(src_mean, np.ndarray)
        assert isinstance(src_std, np.ndarray)

        if prev is None:
            prev_mean = mean
            prev_std = std
            days_from_previous = 0
        else:
            prev_mean = stats[prev]["mean"]
            prev_std = stats[prev]["std"]
            assert isinstance(prev_mean, np.ndarray)
            assert isinstance(prev_std, np.ndarray)
            days_from_previous = (session_date(session) - session_date(prev)).days

        rows.append(
            {
                "session": session,
                "date": stats[session]["date"],
                "n_trials": stats[session]["n_trials"],
                "n_frames": stats[session]["n_frames"],
                "days_from_source": (session_date(session) - source_date).days,
                "days_from_previous": days_from_previous,
                "mean_shift_from_source": float(np.linalg.norm(mean - src_mean)),
                "scale_shift_from_source": float(np.linalg.norm(std - src_std)),
                "diag_cov_shift_from_source": float(np.linalg.norm(np.square(std) - np.square(src_std))),
                "mean_shift_from_previous": float(np.linalg.norm(mean - prev_mean)),
                "scale_shift_from_previous": float(np.linalg.norm(std - prev_std)),
                "diag_cov_shift_from_previous": float(np.linalg.norm(np.square(std) - np.square(prev_std))),
            }
        )
        prev = session

    return pd.DataFrame(rows)


def pairwise_mean_shift(stats: dict[str, dict[str, np.ndarray | int | float | str]]) -> tuple[list[str], np.ndarray]:
    sessions = sorted(stats, key=lambda item: stats[item]["date"])
    means = np.stack([stats[session]["mean"] for session in sessions]).astype(np.float64)
    diffs = means[:, None, :] - means[None, :, :]
    return sessions, np.linalg.norm(diffs, axis=-1)


def session_mean_pca(stats: dict[str, dict[str, np.ndarray | int | float | str]]) -> pd.DataFrame:
    sessions = sorted(stats, key=lambda item: stats[item]["date"])
    means = np.stack([stats[session]["mean"] for session in sessions]).astype(np.float64)
    centered = means - means.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:2].T
    return pd.DataFrame(
        {
            "session": sessions,
            "date": [stats[session]["date"] for session in sessions],
            "pc1": projected[:, 0],
            "pc2": projected[:, 1],
        }
    )
