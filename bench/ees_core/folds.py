"""Shared fold generation + OOF assembly for reliable, alignable validation.

Operators use these primitives so their out-of-fold (OOF) predictions cover
every train row exactly once, in original order — which makes exported
validation artifacts full-length and alignable across operators.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def choose_n_splits(y, *, stratified: bool, max_splits: int = 5, min_splits: int = 2) -> int:
    y = np.asarray(y)
    if stratified:
        _, counts = np.unique(y, return_counts=True)
        cap = int(counts.min()) if len(counts) else min_splits
    else:
        cap = len(y)
    return max(min_splits, min(max_splits, cap))


def fold_safe_stratify_labels(y, *, n_splits: int) -> np.ndarray | None:
    """Labels safe to hand StratifiedKFold(n_splits), or None to go unstratified.

    Classes with fewer than ``n_splits`` members (e.g. tps-dec-2021's singleton
    Cover_Type=5) make StratifiedKFold raise — and the historical fallback of
    dropping stratification entirely silently degraded every OTHER class's fold
    balance because of one stray row. Instead, rare classes are relabeled to the
    most frequent class FOR FOLD ASSIGNMENT ONLY (callers keep training on the
    true labels); returns None when fewer than two classes are populous enough
    for stratification to mean anything.
    """
    y = np.asarray(y)
    values, counts = np.unique(y, return_counts=True)
    sufficient = counts >= n_splits
    if int(sufficient.sum()) < 2:
        return None
    if bool(sufficient.all()):
        return y
    majority = values[int(np.argmax(counts))]
    rare_values = values[~sufficient]
    return np.where(np.isin(y, rare_values), majority, y)


def make_folds(y, *, n_splits: int, stratified: bool, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y)
    if stratified:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(splitter.split(np.zeros(len(y)), y))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(np.zeros(len(y))))


def assemble_oof(n_rows: int, n_cols: int, fold_predictions: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Place per-fold validation predictions into a full (n_rows, n_cols) OOF matrix.

    Asserts every row is filled exactly once — the coverage guarantee that makes
    the OOF full-length and id-alignable.
    """
    oof = np.full((n_rows, n_cols), np.nan, dtype=float)
    filled = np.zeros(n_rows, dtype=int)
    for va_idx, va_pred in fold_predictions:
        va_idx = np.asarray(va_idx)
        va_pred_arr = np.asarray(va_pred, dtype=float)

        # Validate block shape before reshaping
        expected_shape = (len(va_idx), n_cols) if n_cols > 1 else (len(va_idx),)
        if va_pred_arr.ndim == 1:
            # 1-D case: check it has exactly len(va_idx) elements (n_cols must be 1)
            if n_cols != 1 or len(va_pred_arr) != len(va_idx):
                raise ValueError(
                    f"assemble_oof: expected shape {expected_shape}, got shape {va_pred_arr.shape}"
                )
        else:
            # 2-D case: check exact shape match
            if va_pred_arr.shape != (len(va_idx), n_cols):
                raise ValueError(
                    f"assemble_oof: expected shape {expected_shape}, got shape {va_pred_arr.shape}"
                )

        block = va_pred_arr.reshape(len(va_idx), n_cols)
        oof[va_idx] = block
        filled[va_idx] += 1
    if not (filled == 1).all():
        raise ValueError(
            f"assemble_oof coverage error: {(filled == 0).sum()} rows unfilled, "
            f"{(filled > 1).sum()} rows filled more than once"
        )
    return oof
