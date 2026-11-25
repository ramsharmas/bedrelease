"""Decision layer: from per-patient discharge probabilities to bed-release numbers a ward in-charge can act on.

Everything here is plain arithmetic on probabilities, so it is unit-tested. Assumptions (ER arrivals, number of beds)
are inputs, never hidden constants.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C


class PlannerInputError(ValueError):
    pass


def poisson_binomial_pmf(p) -> np.ndarray:
    """Exact distribution of the number of successes for independent Bernoulli(p_i)."""
    p = np.asarray(p, dtype=float)
    if p.size and (np.isnan(p).any() or (p < 0).any() or (p > 1).any()):
        raise PlannerInputError("probabilities must be numbers between 0 and 1")
    pmf = np.array([1.0])
    for q in p:
        pmf = np.convolve(pmf, [1.0 - q, q])
    return pmf


def poisson_pmf(lam: float, kmax: int | None = None) -> np.ndarray:
    if lam < 0 or np.isnan(lam):
        raise PlannerInputError("expected ER arrivals cannot be negative")
    kmax = kmax or int(lam + 8 * np.sqrt(lam + 1) + 10)
    k = np.arange(kmax + 1)
    logp = -lam + k * np.log(max(lam, 1e-12)) - np.array([np.sum(np.log(np.arange(1, i + 1))) for i in k])
    pmf = np.exp(logp) if lam > 0 else np.eye(1, kmax + 1)[0]
    return pmf / pmf.sum()


def interval(pmf: np.ndarray, level: float = 0.80) -> tuple[int, int]:
    cdf = np.cumsum(pmf)
    lo = int(np.searchsorted(cdf, (1 - level) / 2))
    hi = int(np.searchsorted(cdf, 1 - (1 - level) / 2))
    return lo, min(hi, len(pmf) - 1)


def band(p: float) -> str:
    return "green" if p >= C.GREEN_P else ("yellow" if p >= C.YELLOW_P else "red")


def safe_elective_admissions(free_now: int, p_discharge, er_lambda: float, confidence: float = C.DEFAULT_CONFIDENCE) -> dict:
    """Largest number `a` of planned (elective) admissions such that
    P(free_now + discharges - ER_arrivals >= a) >= confidence.   (ER patients must always get a bed.)"""
    if free_now < 0 or int(free_now) != free_now:
        raise PlannerInputError("free beds now must be a whole number, 0 or more")
    if not 0.5 <= confidence < 1:
        raise PlannerInputError("confidence must be between 0.5 and 0.99")
    pd_ = poisson_binomial_pmf(p_discharge)                  # discharges D
    pe = poisson_pmf(er_lambda)                              # ER arrivals E
    # distribution of free_now + D - E on integer support
    d_vals = np.arange(len(pd_)) + int(free_now)
    e_vals = np.arange(len(pe))
    support = np.arange(d_vals.min() - e_vals.max(), d_vals.max() + 1)
    dist = np.zeros(len(support))
    for dv, pdv in zip(d_vals, pd_):
        idx = dv - e_vals - support[0]
        dist[idx] += pdv * pe
    surv = dist[::-1].cumsum()[::-1]                         # P(X >= x)
    ok = support[(surv >= confidence) & (support >= 0)]
    a = int(ok.max()) if ok.size else 0
    shortfall = float(dist[support < 0].sum())               # P(an ER patient finds no bed even with 0 elective)
    exp_free = float(free_now + (np.arange(len(pd_)) * pd_).sum())
    return {"safe_elective": a, "expected_discharges": float((np.arange(len(pd_)) * pd_).sum()),
            "expected_free_beds": exp_free, "discharge_interval_80": interval(pd_, 0.80),
            "p_er_shortfall_no_elective": shortfall, "confidence": confidence}


def sample_census(scored: pd.DataFrame, beds: int, seed: int, ward: str | None = None) -> pd.DataFrame:
    """A steady-state ward: encounters are drawn with probability proportional to LOS (long stays are more likely to be
    in a bed on any given day) and the day of stay is uniform over the stay. Patients are real encounters; the ward and
    the day are simulated because the file has no dates or census."""
    pool = scored if ward in (None, "All wards") else scored[scored["ward"] == ward]
    if len(pool) < 5:
        raise PlannerInputError("not enough patients in this ward group to build a census")
    rng = np.random.default_rng(seed)
    w = pool["los"].to_numpy(dtype=float)
    pick = rng.choice(len(pool), size=min(beds, len(pool)), replace=False, p=w / w.sum())
    cen = pool.iloc[pick].copy().reset_index(drop=True)
    cen["day_done"] = [int(rng.integers(0, l)) for l in cen["los"]]        # 0 = admitted today
    return cen


def census_probs(cen: pd.DataFrame, pmf_cols: list[str]) -> pd.DataFrame:
    pm = cen[pmf_cols].to_numpy()
    tail = np.cumsum(pm[:, ::-1], axis=1)[:, ::-1]
    n = len(cen)
    d = cen["day_done"].to_numpy()
    h = pm[np.arange(n), d] / np.maximum(tail[np.arange(n), d], 1e-9)
    h2 = np.where(d + 1 < C.MAX_LOS,
                  pm[np.arange(n), np.minimum(d + 1, C.MAX_LOS - 1)] / np.maximum(tail[np.arange(n), np.minimum(d + 1, C.MAX_LOS - 1)], 1e-9), 1.0)
    days = np.arange(1, C.MAX_LOS + 1)
    exp_rem = np.array([max(((pm[i, d[i]:] * days[d[i]:]).sum() / max(pm[i, d[i]:].sum(), 1e-9)) - d[i], 0.0) for i in range(n)])
    out = cen.copy()
    out["p24"] = np.clip(h, 0, 1)
    out["p48"] = np.clip(1 - (1 - h) * (1 - h2), 0, 1)
    out["expected_remaining_days"] = exp_rem
    out["band"] = [band(x) for x in out["p24"]]
    return out
