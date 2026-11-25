"""Discrete-time length-of-stay model -> discharge-within-24h probability for any day of the stay.

Idea (hazard model): the classifier predicts P(LOS = k | admission features) for k = 1..14.
For a patient who has already stayed d days, P(discharge within the next 24h) = P(LOS = d+1) / P(LOS > d).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OrdinalEncoder
from . import config as C

DAYS = np.arange(1, C.MAX_LOS + 1)


def patient_split(df: pd.DataFrame, seed: int = C.SEED) -> pd.Series:
    """train / valid / test label per row; all encounters of one patient land in the same part."""
    groups = df["patient_nbr"].to_numpy()
    idx = np.arange(len(df))
    gss = GroupShuffleSplit(n_splits=1, test_size=C.SPLIT["test"], random_state=seed)
    trv, te = next(gss.split(idx, groups=groups))
    rel_valid = C.SPLIT["valid"] / (C.SPLIT["train"] + C.SPLIT["valid"])
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_valid, random_state=seed + 1)
    tr_i, va_i = next(gss2.split(trv, groups=groups[trv]))
    out = np.empty(len(df), dtype=object)
    out[trv[tr_i]], out[trv[va_i]], out[te] = "train", "valid", "test"
    return pd.Series(out, index=df.index, name="part")


class Encoder:
    """Ordinal-encode categoricals (unknown -> -1), pass numbers through."""
    def __init__(self, cat, num):
        self.cat, self.num = list(cat), list(num)
        self.enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    def fit(self, df):
        if self.cat:
            self.enc.fit(df[self.cat].astype(str))
        return self

    def transform(self, df):
        parts = []
        if self.cat:
            parts.append(self.enc.transform(df[self.cat].astype(str)))
        if self.num:
            parts.append(df[self.num].to_numpy(dtype=float))
        return np.hstack(parts)

    @property
    def cat_mask(self):
        return [True] * len(self.cat) + [False] * len(self.num)


def feature_lists(kind: str):
    cat = list(C.ADMISSION_FEATURES_CAT)
    num = list(C.ADMISSION_FEATURES_NUM)
    if kind in ("with_diagnosis", "leaky"):
        cat.append(C.DIAG_FEATURE)
    if kind == "leaky":
        num += C.LEAKY_NUM
    return cat, num


class LosModel:
    def __init__(self, kind: str = "with_diagnosis", params: dict | None = None):
        self.kind = kind
        self.params = dict(learning_rate=0.06, max_iter=250, max_leaf_nodes=15, min_samples_leaf=60, l2_regularization=1.0,
                           early_stopping=True, validation_fraction=0.1, n_iter_no_change=15, random_state=C.SEED)
        if params:
            self.params.update(params)

    def fit(self, df):
        cat, num = feature_lists(self.kind)
        self.enc = Encoder(cat, num).fit(df)
        X = self.enc.transform(df)
        self.clf = HistGradientBoostingClassifier(categorical_features=self.enc.cat_mask, **self.params)
        self.clf.fit(X, df["los"].to_numpy())
        # make the 14 columns always line up with day 1..14, even if a rare class were missing
        self.classes_ = self.clf.classes_
        return self

    def pmf(self, df) -> np.ndarray:
        """n x 14 matrix: P(LOS = day) for day 1..14."""
        p = self.clf.predict_proba(self.enc.transform(df))
        out = np.zeros((len(df), C.MAX_LOS))
        for j, k in enumerate(self.classes_):
            out[:, int(k) - 1] = p[:, j]
        return out / out.sum(axis=1, keepdims=True)


def hazard_matrix(pmf: np.ndarray) -> np.ndarray:
    """n x 14: hazard for day_done = 0..13 (vectorised)."""
    tail = np.cumsum(pmf[:, ::-1], axis=1)[:, ::-1]          # tail[:, d] = sum_{k>=d} p  (col index)
    return np.clip(pmf / np.maximum(tail, 1e-9), 0.0, 1.0)


def expected_los(pmf: np.ndarray) -> np.ndarray:
    return pmf @ DAYS


def los_quantile(pmf: np.ndarray, q: float) -> np.ndarray:
    cdf = np.cumsum(pmf, axis=1)
    return (cdf < q).sum(axis=1) + 1


def day_rows(df: pd.DataFrame, pmf: np.ndarray) -> pd.DataFrame:
    """Expand each encounter into one row per day it was in hospital (day_done = 0..los-1).
    Label = 1 on the last day (discharge within the next 24h)."""
    los = df["los"].to_numpy()
    reps = los
    idx = np.repeat(np.arange(len(df)), reps)
    day_done = np.concatenate([np.arange(l) for l in los])
    H = hazard_matrix(pmf)
    out = pd.DataFrame({
        "row": idx, "day_done": day_done,
        "label": (day_done == los[idx] - 1).astype(int),
        "p_model": H[idx, day_done],
    })
    return out


def binary_metrics(y, p, thr: float) -> dict:
    yhat = (p >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    return {"threshold": float(thr),
            "precision": float(precision_score(y, yhat, zero_division=0)),
            "recall": float(recall_score(y, yhat, zero_division=0)),
            "f1": float(f1_score(y, yhat, zero_division=0)),
            "accuracy": float((tp + tn) / max(len(y), 1)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def best_f1_threshold(y, p) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    f = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f))])


def ece(y, p, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = ids == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = ids == b
        if m.sum() >= 20:
            rows.append({"predicted": float(p[m].mean()), "actual": float(y[m].mean()), "rows": int(m.sum())})
    return pd.DataFrame(rows)
