from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
except ImportError:  # pragma: no cover
    MultilabelStratifiedKFold = None


@dataclass
class CVResult:
    oof_proba: np.ndarray
    models: list
    fold_scores: list[float]


def build_base_classifier(seed: int = 322):
    base = LogisticRegression(
        C=2.0,
        max_iter=600,
        solver="liblinear",
        random_state=seed,
    )
    return OneVsRestClassifier(base)


def make_multilabel_splitter(n_splits: int = 5, seed: int = 322):
    if MultilabelStratifiedKFold is not None:
        return MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def fit_cv_models(
    x: np.ndarray,
    y: np.ndarray,
    estimator,
    splitter,
    score_fn,
) -> CVResult:
    n_samples, n_labels = y.shape
    oof_proba = np.zeros((n_samples, n_labels), dtype=np.float32)
    models = []
    fold_scores: list[float] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
        model = clone(estimator)
        model.fit(x[train_idx], y[train_idx])
        proba = model.predict_proba(x[valid_idx])
        oof_proba[valid_idx] = proba
        pred = (proba >= 0.5).astype(np.int64)
        score = score_fn(y[valid_idx], pred)
        fold_scores.append(float(score))
        print(f"Fold {fold_idx}: hamming_score={score:.6f}")
        models.append(model)

    return CVResult(oof_proba=oof_proba, models=models, fold_scores=fold_scores)


def fit_full_model(x: np.ndarray, y: np.ndarray, estimator):
    model = clone(estimator)
    model.fit(x, y)
    return model
