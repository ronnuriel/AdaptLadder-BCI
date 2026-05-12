from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class TrialRecord:
    session: str
    split: str
    trial_id: str
    n_time_steps: int
    feature_dim: int
    has_seq_class_ids: bool
    has_transcription: bool
    has_sentence_label: bool


def session_date(session: str) -> date:
    date_part = session.split(".", 1)[1]
    return datetime.strptime(date_part, "%Y.%m.%d").date()


def discover_sessions(data_dir: Path) -> list[str]:
    return sorted(path.name for path in data_dir.glob("t15.*") if path.is_dir())


def available_splits(session_dir: Path) -> list[str]:
    splits = []
    for split in ("train", "val", "test"):
        if (session_dir / f"data_{split}.hdf5").exists():
            splits.append(split)
    return splits


def iter_hdf5_trials(file_path: Path):
    split = file_path.stem.removeprefix("data_")
    with h5py.File(file_path, "r") as handle:
        for trial_id in sorted(handle.keys()):
            group = handle[trial_id]
            features = group["input_features"]
            session = group.attrs.get("session", file_path.parent.name)
            if isinstance(session, bytes):
                session = session.decode("utf-8")
            yield TrialRecord(
                session=str(session),
                split=split,
                trial_id=trial_id,
                n_time_steps=int(group.attrs.get("n_time_steps", features.shape[0])),
                feature_dim=int(features.shape[1]),
                has_seq_class_ids="seq_class_ids" in group,
                has_transcription="transcription" in group,
                has_sentence_label="sentence_label" in group.attrs,
            ), features[:]


def finite_std(sum_x: np.ndarray, sum_x2: np.ndarray, n: int, eps: float = 1e-6) -> np.ndarray:
    mean = sum_x / max(n, 1)
    variance = sum_x2 / max(n, 1) - mean * mean
    return np.sqrt(np.maximum(variance, eps))
