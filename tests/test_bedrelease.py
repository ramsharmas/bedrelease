"""Tests for BedRelease Planner. Run:  python -m pytest -q   (after `python run_pipeline.py`)

Three groups: (1) planner math on hand-checkable cases, (2) data/leakage/split rules, (3) saved outputs recomputed from raw rows.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bedrelease import config as C
from bedrelease.model import (LosModel, binary_metrics, ece, feature_lists, hazard_matrix, patient_split)
from bedrelease.planner import (PlannerInputError, band, census_probs, interval, poisson_binomial_pmf, poisson_pmf,
                                safe_elective_admissions, sample_census)

OUT = ROOT / "outputs"
needs_outputs = pytest.mark.skipif(not (OUT / "metrics.json").exists(), reason="run python run_pipeline.py first")
PMF = [f"p_day{i}" for i in range(1, C.MAX_LOS + 1)]


# ------------------------------------------------------------------ planner math
def test_poisson_binomial_matches_hand_calc():
    pmf = poisson_binomial_pmf([0.5, 0.5])
    assert np.allclose(pmf, [0.25, 0.5, 0.25])
    assert np.allclose(poisson_binomial_pmf([]), [1.0])
    assert np.isclose(poisson_binomial_pmf([0.2, 0.7, 0.9]).sum(), 1.0)


def test_poisson_binomial_mean_is_sum_of_probabilities():
    p = np.random.default_rng(0).uniform(0, 1, 25)
    pmf = poisson_binomial_pmf(p)
    assert np.isclose((np.arange(len(pmf)) * pmf).sum(), p.sum())


def test_poisson_pmf_sums_to_one_and_mean():
    pm = poisson_pmf(3.0)
    assert np.isclose(pm.sum(), 1.0)
    assert np.isclose((np.arange(len(pm)) * pm).sum(), 3.0, atol=1e-3)
    assert np.allclose(poisson_pmf(0.0)[:2], [1.0, 0.0])


@pytest.mark.parametrize("bad", [[-0.1], [1.2], [np.nan]])
def test_invalid_probabilities_rejected(bad):
    with pytest.raises(PlannerInputError):
        poisson_binomial_pmf(bad)


def test_invalid_inputs_rejected():
    with pytest.raises(PlannerInputError):
        safe_elective_admissions(-1, [0.5], 3.0)
    with pytest.raises(PlannerInputError):
        safe_elective_admissions(2.5, [0.5], 3.0)
    with pytest.raises(PlannerInputError):
        safe_elective_admissions(2, [0.5], 3.0, confidence=0.2)
    with pytest.raises(PlannerInputError):
        safe_elective_admissions(2, [0.5], -1.0)


def test_safe_elective_known_cases():
    assert safe_elective_admissions(5, [], 0.0)["safe_elective"] == 5          # no patients, no ER: all 5 beds
    assert safe_elective_admissions(0, [], 0.0)["safe_elective"] == 0
    assert safe_elective_admissions(0, [1, 1, 1, 1], 0.0)["safe_elective"] == 4  # 4 certain discharges
    assert safe_elective_admissions(0, [0, 0, 0], 0.0)["safe_elective"] == 0


def test_safe_elective_monotonic():
    p = [0.3] * 12
    a = [safe_elective_admissions(f, p, 3.0)["safe_elective"] for f in range(0, 8)]
    assert a == sorted(a)                                                       # more free beds -> never fewer admissions
    b = [safe_elective_admissions(3, p, e)["safe_elective"] for e in [0, 2, 4, 8]]
    assert b == sorted(b, reverse=True)                                         # more ER -> never more admissions
    c = [safe_elective_admissions(3, p, 3.0, conf)["safe_elective"] for conf in [0.6, 0.8, 0.95]]
    assert c == sorted(c, reverse=True)                                         # more confidence -> never more admissions


def test_safe_elective_meets_stated_confidence_by_simulation():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.1, 0.6, 20)
    free, lam, conf = 2, 3.0, 0.90
    a = safe_elective_admissions(free, p, lam, conf)["safe_elective"]
    n = 200000
    d = (rng.random((n, len(p))) < p).sum(axis=1)
    e = rng.poisson(lam, n)
    ok = (free + d - e >= a).mean()
    assert ok >= conf - 0.01                                                    # simulated coverage >= stated confidence (MC tolerance)
    if a + 1 > 0:
        assert (free + d - e >= a + 1).mean() < conf + 0.01                     # and a+1 would not be safe


def test_doctor_override_raises_expected_discharges():
    p = np.array([0.2, 0.3, 0.1])
    base = safe_elective_admissions(2, p, 3.0)
    p2 = p.copy(); p2[0] = 1.0                                                  # same maths the dashboard uses for the tick box
    over = safe_elective_admissions(2, p2, 3.0)
    assert np.isclose(over["expected_discharges"] - base["expected_discharges"], 0.8)
    assert over["safe_elective"] >= base["safe_elective"]


def test_interval_and_band():
    lo, hi = interval(np.array([0.1, 0.2, 0.4, 0.2, 0.1]), 0.8)
    assert 0 <= lo <= hi <= 4
    assert band(C.GREEN_P) == "green" and band(C.YELLOW_P) == "yellow" and band(C.YELLOW_P - 0.01) == "red"


def _fake_scored(n=300, seed=0):
    rng = np.random.default_rng(seed)
    pm = rng.dirichlet(np.ones(C.MAX_LOS), n)
    df = pd.DataFrame(pm, columns=PMF)
    df["los"] = [int(rng.choice(np.arange(1, 15), p=r)) for r in pm]
    df["ward"] = rng.choice(["A", "B"], n)
    return df


def test_census_is_deterministic_and_valid():
    s = _fake_scored()
    c1, c2 = sample_census(s, 30, 7, "A"), sample_census(s, 30, 7, "A")
    assert c1.equals(c2)
    assert (c1["ward"] == "A").all() and len(c1) == 30
    assert ((c1["day_done"] >= 0) & (c1["day_done"] < c1["los"])).all()         # patient is always inside their own stay
    out = census_probs(c1, PMF)
    assert out["p24"].between(0, 1).all() and out["p48"].between(0, 1).all()
    assert (out["p48"] >= out["p24"] - 1e-9).all()                              # 48h chance can never be below 24h chance
    assert (out["expected_remaining_days"] >= 0).all()
    with pytest.raises(PlannerInputError):
        sample_census(s, 30, 7, "no such ward")


def test_last_possible_day_means_certain_discharge():
    pm = np.zeros((1, C.MAX_LOS)); pm[0, 2] = 1.0                               # always stays exactly 3 days
    cen = pd.DataFrame(pm, columns=PMF); cen["los"] = 3; cen["day_done"] = 2
    out = census_probs(cen, PMF)
    assert np.isclose(out["p24"].iloc[0], 1.0)


def test_hazard_matrix_known_values():
    pmf = np.zeros((1, C.MAX_LOS)); pmf[0, 0], pmf[0, 1], pmf[0, 2] = 0.5, 0.25, 0.25
    h = hazard_matrix(pmf)[0]
    assert np.allclose(h[:3], [0.5, 0.5, 1.0])                                  # P(LOS=2|LOS>=2)=0.25/0.5


# ------------------------------------------------------------------ data rules (use the real file)
@pytest.fixture(scope="module")
def clean():
    if not C.RAW_CSV.exists():
        pytest.skip("raw data missing")
    from bedrelease.data import load_clean
    return load_clean()


def test_raw_file_checksum_and_shape():
    if not C.RAW_CSV.exists():
        pytest.skip("raw data missing")
    from bedrelease.data import sha256_of
    assert sha256_of(C.RAW_CSV) == C.RAW_SHA256


def test_cleaning_invariants(clean):
    df = clean.df
    assert len(df) == C.EXPECTED_ROWS
    assert df["encounter_id"].is_unique
    assert df["los"].between(1, C.MAX_LOS).all()
    assert df["readmit30"].isin([0, 1]).all()
    assert (df.loc[df["died_or_hospice"], "disposition"].notna()).all()
    assert (df["prior_encounters"] >= 0).all()


def test_history_features_use_only_earlier_encounters(clean):
    df = clean.df.sort_values(["patient_nbr", "encounter_id"])
    first = df.groupby("patient_nbr").head(1)
    assert (first["prior_encounters"] == 0).all()                               # a patient's first encounter has no history
    two = df[df.groupby("patient_nbr")["patient_nbr"].transform("size") >= 2].groupby("patient_nbr").head(2)
    second = two.groupby("patient_nbr").tail(1).head(500)
    firstmap = two.groupby("patient_nbr").head(1).set_index("patient_nbr")["los"]
    assert np.allclose(second["last_los"].to_numpy(), firstmap.loc[second["patient_nbr"]].to_numpy())


def test_patient_split_has_no_patient_in_two_parts(clean):
    df = clean.df
    part = patient_split(df)
    g = pd.DataFrame({"p": df["patient_nbr"], "part": part}).groupby("p")["part"].nunique()
    assert (g == 1).all()
    shares = part.value_counts(normalize=True)
    assert abs(shares["train"] - 0.70) < 0.02 and abs(shares["test"] - 0.15) < 0.02
    assert (patient_split(df) == part).all()                                    # deterministic


def test_leaky_columns_not_in_production_features():
    for kind in ["admission_only", "with_diagnosis"]:
        cat, num = feature_lists(kind)
        assert not (set(num) & set(C.LEAKY_NUM)) and not (set(cat) & set(C.LEAKY_NUM)), kind
    assert set(C.LEAKY_NUM) <= set(feature_lists("leaky")[1])                   # ...but the leakage demo does include them
    assert "los" not in feature_lists("with_diagnosis")[0] + feature_lists("with_diagnosis")[1]


def test_los_model_pmf_rows_sum_to_one(clean):
    sub = clean.df.sample(4000, random_state=0)
    m = LosModel("with_diagnosis", dict(max_iter=20)).fit(sub)
    p = m.pmf(sub.head(50))
    assert p.shape == (50, C.MAX_LOS) and np.allclose(p.sum(axis=1), 1.0) and (p >= 0).all()


def test_binary_metrics_and_ece():
    y = np.array([1, 1, 0, 0, 1, 0]); p = np.array([.9, .6, .4, .2, .3, .1])
    m = binary_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 0, 1, 3)
    assert np.isclose(m["precision"], 1.0) and np.isclose(m["recall"], 2 / 3)
    assert ece(np.array([0, 1] * 500), np.full(1000, 0.5)) < 1e-9               # perfectly calibrated coin


# ------------------------------------------------------------------ saved outputs
@needs_outputs
def test_metrics_recompute_from_raw_rows():
    from sklearn.metrics import average_precision_score, roc_auc_score
    m = json.loads((OUT / "metrics.json").read_text())
    d = pd.read_csv(OUT / "test_day_rows.csv.gz")
    assert abs(roc_auc_score(d["label"], d["p_model"]) - m["discharge_24h"]["p_model"]["roc_auc"]) < 1e-4
    assert abs(roc_auc_score(d["label"], d["p_base"]) - m["discharge_24h"]["p_base"]["roc_auc"]) < 1e-4
    assert abs(average_precision_score(d["label"], d["p_model"]) - m["discharge_24h"]["p_model"]["pr_auc"]) < 1e-3
    assert len(d) == m["discharge_24h"]["rows"]
    op = m["discharge_24h"]["operating_point"]
    mm = binary_metrics(d["label"].to_numpy(), d["p_model"].to_numpy(), op["threshold"])
    assert abs(mm["precision"] - op["precision"]) < 2e-3 and abs(mm["recall"] - op["recall"]) < 2e-3


@needs_outputs
def test_threshold_was_chosen_on_validation_not_test():
    m = json.loads((OUT / "metrics.json").read_text())
    assert m["split"]["train"]["patients"] + m["split"]["valid"]["patients"] + m["split"]["test"]["patients"] == 71518
    assert 0.05 <= m["decision_threshold_discharge24h"] <= 0.95


@needs_outputs
def test_test_patients_never_in_training_ids():
    s = pd.read_csv(OUT / "test_scored_encounters.csv.gz")
    from bedrelease.data import load_clean
    df = load_clean().df
    part = patient_split(df)
    test_pat = set(df.loc[part == "test", "patient_nbr"])
    train_pat = set(df.loc[part == "train", "patient_nbr"])
    assert not (test_pat & train_pat)
    assert set(s["patient_nbr"]) <= test_pat and len(s) == (part == "test").sum()


@needs_outputs
def test_scored_pmf_valid_and_calibrated():
    s = pd.read_csv(OUT / "test_scored_encounters.csv.gz")
    assert np.allclose(s[PMF].sum(axis=1), 1.0, atol=2e-4)   # CSV stores 5 decimals
    m = json.loads((OUT / "metrics.json").read_text())
    assert m["discharge_24h"]["p_model"]["ece"] < 0.02                          # probabilities usable for adding up beds


@needs_outputs
def test_model_beats_baseline_and_leakage_is_flagged():
    m = json.loads((OUT / "metrics.json").read_text())
    assert m["discharge_24h"]["p_model"]["roc_auc"] > m["discharge_24h"]["p_base"]["roc_auc"]
    assert m["ablation"]["leaky"]["roc_auc"] > m["ablation"]["with_diagnosis"]["roc_auc"] + 0.05   # the leak is real, which is why it is excluded
    assert m["time_forward_check"]["roc_auc_model"] > m["time_forward_check"]["roc_auc_baseline"]
    wb = m["ward_backtest"]
    assert wb["model"]["plan_failed_pct"] <= 10.0                               # planning promise: 90% confidence => <=10% failures (+MC noise)
    assert wb["precision_at_k"]["model"] > wb["precision_at_k"]["random"]


@needs_outputs
def test_pipeline_is_deterministic_on_saved_run():
    m = json.loads((OUT / "metrics.json").read_text())
    meta = json.loads((OUT / "meta.json").read_text())
    assert meta["seed"] == C.SEED and meta["raw_sha256"] == C.RAW_SHA256
    assert m["test_encounters"] == m["split"]["test"]["encounters"]
