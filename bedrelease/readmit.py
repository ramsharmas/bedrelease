"""30-day readmission risk at discharge (used as a 'safe release' check, not as a bed-planning input)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from . import config as C
from .model import Encoder

CAT = ["race", "gender", "age", "admission_type", "admission_source", "payer_code", "ward", "diag_group", "disposition",
       "A1Cresult", "insulin", "change", "diabetesMed"]
NUM = ["number_outpatient", "number_emergency", "number_inpatient", "prior_encounters", "prior_mean_los", "last_los",
       "los", "num_lab_procedures", "num_procedures", "num_medications", "number_diagnoses"]


class ReadmitModel:
    def __init__(self, params=None):
        self.params = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=2.0,
                          early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, class_weight="balanced",
                          random_state=C.SEED)
        if params:
            self.params.update(params)

    def fit(self, df):
        d = df[~df["died_or_hospice"]]
        self.enc = Encoder(CAT, NUM).fit(d)
        self.clf = HistGradientBoostingClassifier(categorical_features=self.enc.cat_mask, **self.params)
        self.clf.fit(self.enc.transform(d), d["readmit30"].to_numpy())
        return self

    def predict(self, df) -> np.ndarray:
        p = self.clf.predict_proba(self.enc.transform(df))[:, 1]
        return np.where(df["died_or_hospice"].to_numpy(), np.nan, p)
