"""SYNTHETIC in-stay ward data (clearly simulated). Real UCI admissions give age / admission type / diagnosis / stay length;
we ADD simulated daily in-stay signals that real hospital records carry but the UCI file does not
(doctor's discharge plan, vitals stability, mobility, lab trend, IV drug). Purpose: show what accuracy becomes possible
once such columns exist. Numbers on this data describe the simulator, not a real hospital."""
import numpy as np, pandas as pd

SIGNALS = ["day_done", "frac_stay_done", "doctor_plan_tomorrow", "vitals_stable", "mobility_score", "crp_trend", "on_iv_drug",
           "age_num", "emergency", "diag_code"]


def make(enc: pd.DataFrame, seed: int = 7, copies: int = 2) -> pd.DataFrame:
    """One row per patient-day. enc needs columns los, age, admission_type, diag_group, patient_nbr."""
    rng = np.random.default_rng(seed)
    e = pd.concat([enc] * copies, ignore_index=True)
    e["copy"] = np.repeat(np.arange(copies), len(enc))
    r = e.loc[e.index.repeat(e["los"].to_numpy())].reset_index(drop=True)
    r["day_done"] = np.concatenate([np.arange(n) for n in e["los"]])
    left = r["los"] - r["day_done"] - 1                      # days remaining after today
    r["label"] = (left == 0).astype(int)                     # goes home within 24h
    n = len(r)
    # doctor's plan: usually right the day before, sometimes late or premature
    plan = np.where(left == 0, rng.random(n) < 0.88, np.where(left == 1, rng.random(n) < 0.12, rng.random(n) < 0.015))
    r["doctor_plan_tomorrow"] = plan.astype(int)
    r["vitals_stable"] = (rng.random(n) < np.clip(0.35 + 0.55 * np.exp(-left / 2.0), 0, 1)).astype(int)
    r["mobility_score"] = np.clip(np.round(1 + 3 * np.exp(-left / 2.5) + rng.normal(0, 0.7, n)), 0, 4)
    r["crp_trend"] = np.round(-1.2 * np.exp(-left / 3.0) + 0.6 * np.exp(-r["day_done"] / 2) + rng.normal(0, 0.5, n), 2)
    r["on_iv_drug"] = (rng.random(n) < np.clip(0.75 - 0.6 * np.exp(-left / 2.0), 0, 1)).astype(int)
    r["frac_stay_done"] = np.round(r["day_done"] / r["los"].clip(lower=1), 2) * 0  # placeholder, not known in practice -> zeroed
    amap = {"[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95}
    r["age_num"] = r["age"].map(amap).fillna(60)
    r["emergency"] = (r["admission_type"].astype(str).str.contains("Emergency")).astype(int)
    r["diag_code"] = r["diag_group"].astype("category").cat.codes
    return r
