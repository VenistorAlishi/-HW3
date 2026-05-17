from __future__ import annotations

import numpy as np


def find_best_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    score_fn,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    if grid is None:
        grid = np.linspace(0.2, 0.8, 61)

    n_labels = y_true.shape[1]
    thresholds = np.full(n_labels, 0.5, dtype=np.float32)
    best_global = score_fn(y_true, (y_proba >= thresholds).astype(np.int64))

    for label in range(n_labels):
        best_t = thresholds[label]
        best_score = best_global
        for t in grid:
            trial = thresholds.copy()
            trial[label] = float(t)
            pred = (y_proba >= trial).astype(np.int64)
            score = score_fn(y_true, pred)
            if score > best_score:
                best_score = score
                best_t = float(t)
        thresholds[label] = best_t
        best_global = best_score

    return thresholds, float(best_global)


def apply_thresholds(y_proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (y_proba >= thresholds).astype(np.int64)
