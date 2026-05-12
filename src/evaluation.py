from __future__ import annotations

import numpy as np
import pandas as pd


def distribution_alignment_score(mean: np.ndarray, std: np.ndarray, reference_mean: np.ndarray, reference_std: np.ndarray) -> float:
    mean_term = np.linalg.norm(mean - reference_mean)
    scale_term = np.linalg.norm(std - reference_std)
    return float(mean_term + scale_term)


def summarize_method_scores(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (
        df.groupby("method", as_index=False)
        .agg(
            mean_proxy_score=("proxy_score", "mean"),
            median_proxy_score=("proxy_score", "median"),
            num_sessions=("session", "count"),
        )
        .sort_values("mean_proxy_score")
        .reset_index(drop=True)
    )
