from __future__ import annotations

import ast
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


SEED = 322
TARGET_COLUMNS = ["label_0", "label_1", "label_2", "label_3", "label_4"]


@dataclass
class DataBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_target(value: str) -> np.ndarray:
    raw = ast.literal_eval(value)
    arr = np.asarray(raw, dtype=np.int64)
    if arr.shape != (5,):
        raise ValueError(f"Target must have 5 labels, got shape={arr.shape}.")
    if not np.isin(arr, [0, 1]).all():
        raise ValueError("Target values must be binary.")
    return arr


def load_competition_data(data_dir: str | Path = ".") -> DataBundle:
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv", sep="\t")
    test = pd.read_csv(data_dir / "test.csv", sep="\t")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
    return DataBundle(train=train, test=test, sample_submission=sample_submission)


def validate_schema(train: pd.DataFrame, test: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    expected_train = {"id", "source", "title", "text", "publication_date", "target"}
    expected_test = {"id", "source", "title", "text", "publication_date"}
    expected_submission = {"id", "target"}

    if set(train.columns) != expected_train:
        raise ValueError(f"Unexpected train columns: {train.columns.tolist()}")
    if set(test.columns) != expected_test:
        raise ValueError(f"Unexpected test columns: {test.columns.tolist()}")
    if set(sample_submission.columns) != expected_submission:
        raise ValueError(f"Unexpected sample_submission columns: {sample_submission.columns.tolist()}")


def add_target_columns(train: pd.DataFrame) -> pd.DataFrame:
    parsed = np.vstack(train["target"].map(parse_target).to_numpy())
    result = train.copy()
    for i, col in enumerate(TARGET_COLUMNS):
        result[col] = parsed[:, i]
    return result


def format_submission_targets(binary_matrix: np.ndarray) -> list[str]:
    rows = []
    for row in binary_matrix.astype(int):
        row_text = ",".join(str(int(v)) for v in row.tolist())
        rows.append(f"[{row_text}]")
    return rows


def hamming_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    mismatches = np.not_equal(y_true, y_pred).mean()
    return float(1.0 - mismatches)


def extract_targets(df: pd.DataFrame, target_cols: Iterable[str] = TARGET_COLUMNS) -> np.ndarray:
    return df[list(target_cols)].to_numpy(dtype=np.int64)
