from __future__ import annotations

import numpy as np


def target_zscore(x: np.ndarray, target_mean: np.ndarray, target_std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (x - target_mean) / np.maximum(target_std, eps)


def source_to_target_moment_match(
    x: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    source_mean: np.ndarray,
    source_std: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    z = target_zscore(x, target_mean, target_std, eps=eps)
    return z * source_std + source_mean


def diagonal_affine(x: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x * scale + bias
