"""End-to-end run: verify data -> split by patient -> tune -> test -> backtests -> write outputs/."""
from __future__ import annotations
import json, platform, time
import joblib, numpy as np, pandas as pd, sklearn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve, precision_recall_curve
from . import config as C, data as D, model as M, planner as P
from .readmit import ReadmitModel

PMF_COLS = [f"p_day{k}" for k in range(1, C.MAX_LOS + 1)]
GRID = [dict(learning_rate=lr, max_leaf_nodes=n) for lr in (0.05, 0.08) for n in (8, 15, 31)]


def _log(msg):
    print(msg, flush=True)


def base_hazard(los: np.ndarray) -> np.ndarray:
    return np.array([(los == d + 1).sum() / max((los > d).sum(), 1) for d in range(C.MAX_LOS)])


def true_logloss(model: M.LosModel, df) -> float:
    pm = model.pmf(df)
    return float(-np.log(np.clip(pm[np.arange(len(df)), df["los"].to_numpy() - 1], 1e-9, 1)).mean())


def precision_target_threshold(y, p, target=0.90, min_flagged=30):
    """Lowest threshold whose precision >= target with at least min_flagged flags (chosen on validation only)."""
    for t in np.unique(np.round(np.quantile(p, np.linspace(0.50, 0.9995, 400)), 5)):
        m = p >= t
        if m.sum() >= min_flagged and y[m].mean() >= target:
            return float(t)
    return None


def confident_band(y, p, hi, target=0.91, min_cov=0.03):
    """Abstain band: call 'stays' below lo, 'leaves' at/above hi, abstain between. lo = largest cut whose validation accuracy of the
    combined confident calls stays >= target (chosen on validation only)."""
    best = None
    for lo in np.unique(np.round(np.quantile(p, np.linspace(0.01, 0.6, 300)), 5)):
        m = (p < lo) | (p >= hi)
        if m.mean() < min_cov:
            continue
        if ((p[m] >= hi).astype(int) == y[m]).mean() >= target:
            best = float(lo)
    return best


def confident_metrics(y, p, lo, hi) -> dict:
    m = (p < lo) | (p >= hi)
    yh = (p >= hi).astype(int)
    return {"lo": float(lo), "hi": float(hi), "coverage": float(m.mean()), "accuracy_on_confident_calls": float((yh[m] == y[m]).mean()),
            "n_calls": int(m.sum()), "n_abstain": int((~m).sum())}


def day_level_metrics(r: pd.DataFrame, thr: float, hp_thr=None, lo_thr=None) -> dict:
    y = r["label"].to_numpy()
    out = {"rows": int(len(r)), "positive_rate_pct": round(float(y.mean() * 100), 2)}
    for name in ("p_model", "p_base"):
        p = r[name].to_numpy()
        out[name] = {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
                     "brier": float(brier_score_loss(y, p)), "ece": M.ece(y, p)}
    out["operating_point"] = M.binary_metrics(y, r["p_model"].to_numpy(), thr)
    if hp_thr is not None:
        out["high_precision_point"] = M.binary_metrics(y, r["p_model"].to_numpy(), hp_thr)
        out["high_precision_point"]["target_precision"] = 0.90
    if hp_thr is not None and lo_thr is not None:
        out["confident_calls"] = confident_metrics(y, r["p_model"].to_numpy(), lo_thr, hp_thr)
    out["majority_class_accuracy"] = float(1 - y.mean())
    # alert budget: flag the 20% highest-probability patient-days
    cut = np.quantile(r["p_model"], 0.80)
    out["top20pct_alert"] = M.binary_metrics(y, r["p_model"].to_numpy(), float(cut))
    cutb = np.quantile(r["p_base"], 0.80)
    out["top20pct_alert_baseline"] = M.binary_metrics(y, r["p_base"].to_numpy(), float(cutb))
    return out


def permutation_importance(model: M.LosModel, df: pd.DataFrame, repeats: int = 3, n: int = 6000) -> pd.DataFrame:
    rng = np.random.default_rng(C.SEED)
    d = df.sample(min(n, len(df)), random_state=C.SEED).reset_index(drop=True)
    X = model.enc.transform(d)
    names = model.enc.cat + model.enc.num
    y = d["los"].to_numpy()

    def ll(Xm):
        p = model.clf.predict_proba(Xm)
        cols = {int(k): j for j, k in enumerate(model.classes_)}
        pr = np.array([p[i, cols[y[i]]] if y[i] in cols else 1e-9 for i in range(len(y))])
        return float(-np.log(np.clip(pr, 1e-9, 1)).mean())

    base = ll(X)
    rows = []
    for j, nm in enumerate(names):
        inc = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            inc.append(ll(Xp) - base)
        rows.append({"feature": nm, "logloss_increase": float(np.mean(inc))})
    return pd.DataFrame(rows).sort_values("logloss_increase", ascending=False).reset_index(drop=True)


def ward_backtest(scored: pd.DataFrame, hb: np.ndarray, n_days: int = 2000, beds: int = C.DEFAULT_WARD_BEDS,
                  free_now: int = C.DEFAULT_FREE_BEDS, er_lambda: float = C.DEFAULT_ER_ARRIVALS,
                  conf: float = C.DEFAULT_CONFIDENCE) -> dict:
    """Simulate many ward-days from real test encounters; compare tomorrow's predicted discharges with what really happened.
    Patients and their true discharge day are real; ward size, free beds and ER arrivals are assumptions (inputs)."""
    rng = np.random.default_rng(C.SEED + 7)
    naive_p = 1.0 / float(scored["los"].mean())               # naive: everyone leaves with probability 1 / average LOS
    rows = []
    for _ in range(n_days):
        cen = P.sample_census(scored, beds, seed=int(rng.integers(0, 2**31 - 1)))
        cp = P.census_probs(cen, PMF_COLS)
        actual = (cp["day_done"] + 1 == cp["los"]).to_numpy().astype(int)
        probs = {"model": cp["p24"].to_numpy(), "baseline_day": hb[cp["day_done"].to_numpy()],
                 "naive_constant": np.full(len(cp), naive_p)}
        er = int(rng.poisson(er_lambda))                       # the same ER arrivals for every planner
        d_act = int(actual.sum())
        k = max(1, int(round(probs["model"].sum())))
        row = {"actual": d_act, "K": k, "base_rate": float(actual.mean()),
               "prec_at_k_model": float(actual[np.argsort(-probs["model"], kind="stable")[:k]].mean()),
               "prec_at_k_baseline": float(actual[np.argsort(-(probs["baseline_day"] + 1e-6 * rng.random(len(cp))), kind="stable")[:k]].mean())}
        for name, p in probs.items():
            lo, hi = P.interval(P.poisson_binomial_pmf(p), 0.80)
            a = P.safe_elective_admissions(free_now, p, er_lambda, conf)["safe_elective"]
            row[f"{name}_mean"] = float(p.sum()); row[f"{name}_in80"] = int(lo <= d_act <= hi)
            row[f"{name}_safe"] = a; row[f"{name}_violation"] = int(a > 0 and free_now + d_act - er < a)
            row[f"{name}_empty"] = int(max(free_now + d_act - er - a, 0))      # beds left empty after the elective plan
        rows.append(row)
    r = pd.DataFrame(rows)
    out = {"ward_days": n_days, "beds": beds, "free_beds_now": free_now, "er_arrivals_assumed": er_lambda,
           "confidence": conf, "mean_actual_discharges": float(r["actual"].mean()),
           "precision_at_k": {"model": float(r["prec_at_k_model"].mean()), "baseline_day_hazard": float(r["prec_at_k_baseline"].mean()),
                              "random": float(r["base_rate"].mean())}}
    for name in ("model", "baseline_day", "naive_constant"):
        out[name] = {"mae_discharges": float((r[f"{name}_mean"] - r["actual"]).abs().mean()),
                     "bias": float((r[f"{name}_mean"] - r["actual"]).mean()),
                     "interval80_coverage_pct": float(r[f"{name}_in80"].mean() * 100),
                     "avg_safe_elective": float(r[f"{name}_safe"].mean()),
                     "plan_failed_pct": float(r[f"{name}_violation"].mean() * 100)}
    return out


def run(verbose: bool = True) -> dict:
    t0 = time.time()
    C.OUTPUTS.mkdir(exist_ok=True); C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean = D.load_clean()
    df = clean.df
    _log(f"[1/7] data: {len(df):,} encounters, {clean.report['patients']:,} patients, sha256={clean.report['sha256'][:12]}...")
    df["part"] = M.patient_split(df)
    tr, va, te = (df[df.part == k].reset_index(drop=True) for k in ("train", "valid", "test"))
    assert not (set(tr.patient_nbr) & set(te.patient_nbr)) and not (set(va.patient_nbr) & set(te.patient_nbr)) \
        and not (set(tr.patient_nbr) & set(va.patient_nbr)), "patients leaked across splits"
    split = {k: {"encounters": int(len(v)), "patients": int(v.patient_nbr.nunique())} for k, v in (("train", tr), ("valid", va), ("test", te))}

    # ---- tune on validation (true-LOS log-loss) ----
    tuning = []
    for g in GRID:
        m = M.LosModel("with_diagnosis", g).fit(tr)
        tuning.append({**g, "valid_logloss": true_logloss(m, va), "iters": int(m.clf.n_iter_)})
    best = min(tuning, key=lambda x: x["valid_logloss"])
    best_params = {k: best[k] for k in ("learning_rate", "max_leaf_nodes")}
    _log(f"[2/7] tuned {len(GRID)} settings on validation; best {best_params}")

    trva = pd.concat([tr, va], ignore_index=True)
    los_model = M.LosModel("with_diagnosis", best_params).fit(trva)
    hb = base_hazard(trva["los"].to_numpy())
    pm_te = los_model.pmf(te)
    r = M.day_rows(te, pm_te)
    r["p_base"] = hb[r["day_done"].to_numpy()]
    # decision threshold picked on validation only
    m_val = M.LosModel("with_diagnosis", best_params).fit(tr)
    rv = M.day_rows(va, m_val.pmf(va))
    thr = M.best_f1_threshold(rv["label"].to_numpy(), rv["p_model"].to_numpy())
    hp_thr = precision_target_threshold(rv["label"].to_numpy(), rv["p_model"].to_numpy())
    lo_thr = confident_band(rv["label"].to_numpy(), rv["p_model"].to_numpy(), hp_thr) if hp_thr is not None else None
    day_m = day_level_metrics(r, thr, hp_thr, lo_thr)
    # LOS metrics
    med = M.los_quantile(pm_te, 0.5); q10 = M.los_quantile(pm_te, 0.10); q90 = M.los_quantile(pm_te, 0.90)
    los_m = {"mae_median_model": float(np.abs(te.los - med).mean()), "mae_median_baseline": float(np.abs(te.los - trva.los.median()).mean()),
             "mean_los_actual": float(te.los.mean()), "mean_los_predicted": float(M.expected_los(pm_te).mean()),
             "p10_p90_coverage_pct": float(((te.los >= q10) & (te.los <= q90)).mean() * 100),
             "rmse_expected": float(np.sqrt(((te.los - M.expected_los(pm_te)) ** 2).mean()))}
    long_y = (te.los >= 7).astype(int).to_numpy(); long_p = pm_te[:, 6:].sum(axis=1)
    cut = float(np.quantile(long_p, 0.80))
    long_m = {"definition": "stay of 7+ days (flagged at admission)", "base_rate_pct": float(long_y.mean() * 100),
              "roc_auc": float(roc_auc_score(long_y, long_p)), "pr_auc": float(average_precision_score(long_y, long_p)),
              "top20pct_flag": M.binary_metrics(long_y, long_p, cut),
              "bed_days_share_of_long_pct": float(te.los[te.los >= 7].sum() / te.los.sum() * 100)}
    _log(f"[3/7] test: discharge-24h AUC model {day_m['p_model']['roc_auc']:.3f} vs baseline {day_m['p_base']['roc_auc']:.3f}")

    # ---- ablations / leakage / time-forward ----
    abl = {}
    for kind in ("admission_only", "with_diagnosis", "leaky"):
        mm = los_model if kind == "with_diagnosis" else M.LosModel(kind, best_params).fit(trva)
        pmx = mm.pmf(te) if kind != "with_diagnosis" else pm_te
        rx = M.day_rows(te, pmx)
        abl[kind] = {"roc_auc": float(roc_auc_score(rx["label"], rx["p_model"])),
                     "mae_median": float(np.abs(te.los - M.los_quantile(pmx, 0.5)).mean())}
    cut_id = df["encounter_id"].quantile([0.70, 0.85]).to_numpy()
    f_tr = df[df.encounter_id <= cut_id[0]]; f_te = df[df.encounter_id > cut_id[1]].reset_index(drop=True)
    mf = M.LosModel("with_diagnosis", best_params).fit(f_tr)
    rf = M.day_rows(f_te, mf.pmf(f_te)); hbf = base_hazard(f_tr["los"].to_numpy())
    fwd = {"train_encounters": int(len(f_tr)), "test_encounters": int(len(f_te)),
           "roc_auc_model": float(roc_auc_score(rf["label"], rf["p_model"])),
           "roc_auc_baseline": float(roc_auc_score(rf["label"], hbf[rf["day_done"].to_numpy()])),
           "note": "train on the earliest 70% of encounter ids, test on the latest 15% (encounter_id order is chronological per patient)"}
    _log(f"[4/7] ablations + time-forward done (leaky AUC {abl['leaky']['roc_auc']:.3f} -> excluded)")

    # ---- subgroup fairness-style table ----
    r["row"] = r["row"].astype(int)
    meta = te[["ward", "age", "gender", "admission_type"]].reset_index(drop=True)
    r = r.join(meta, on="row")
    sub = []
    for col in ("gender", "age", "admission_type", "ward"):
        for val, g in r.groupby(col):
            if len(g) >= 500 and g["label"].nunique() == 2:
                sub.append({"group": col, "value": str(val), "rows": int(len(g)),
                            "roc_auc_model": float(roc_auc_score(g["label"], g["p_model"])),
                            "roc_auc_baseline": float(roc_auc_score(g["label"], g["p_base"]))})
    sub = pd.DataFrame(sub)
    perday = []
    for d, g in r.groupby("day_done"):
        if d <= 7 and g["label"].nunique() == 2 and len(g) > 300:
            perday.append({"day_done": int(d), "rows": int(len(g)), "discharge_rate_pct": float(g["label"].mean() * 100),
                           "roc_auc_model": float(roc_auc_score(g["label"], g["p_model"])),
                           "roc_auc_baseline": float(roc_auc_score(g["label"], g["p_base"]))})
    perday = pd.DataFrame(perday)

    # ---- readmission model ----
    rm = ReadmitModel().fit(trva)
    va_m = ReadmitModel().fit(tr)
    pv = va_m.predict(va); ok_v = ~np.isnan(pv)
    thr_r = float(np.quantile(pv[ok_v], 0.80))               # flag the top 20% highest-risk discharges
    pr_te = rm.predict(te); ok = ~np.isnan(pr_te)
    yr = te["readmit30"].to_numpy()[ok]; pr = pr_te[ok]
    rule = (te["number_inpatient"].to_numpy()[ok] >= 2).astype(int)
    readm = {"cohort": "discharged alive and not to hospice", "test_rows": int(ok.sum()), "base_rate_pct": float(yr.mean() * 100),
             "roc_auc": float(roc_auc_score(yr, pr)), "pr_auc": float(average_precision_score(yr, pr)), "brier": float(brier_score_loss(yr, pr)),
             "threshold_top20pct_from_validation": thr_r,
             "flag_top20pct": M.binary_metrics(yr, pr, thr_r),
             "rule_baseline_2plus_prior_inpatient": M.binary_metrics(yr, rule, 0.5),
             "lift_top_decile": float(yr[pr >= np.quantile(pr, 0.9)].mean() / yr.mean()),
             "note": "class_weight balanced, so raw probabilities are over-estimates; use as a risk rank, not a percentage"}
    _log(f"[5/7] readmission AUC {readm['roc_auc']:.3f}")

    # ---- ward-day backtest on real test encounters ----
    scored = te.copy()
    for j, c in enumerate(PMF_COLS):
        scored[c] = pm_te[:, j]
    scored["readmit_risk"] = pr_te
    scored["readmit_flag"] = np.where(np.isnan(pr_te), 0, (pr_te >= thr_r).astype(int))
    bt = ward_backtest(scored, hb)
    _log(f"[6/7] ward backtest: MAE model {bt['model']['mae_discharges']:.2f} vs baseline {bt['baseline_day']['mae_discharges']:.2f}")

    imp = permutation_importance(los_model, te)

    # ---- BI tables (whole file) ----
    bi = {}
    df["long_stay"] = (df["los"] >= 7).astype(int)
    for col in ("admission_type", "ward", "age", "diag_group", "disposition"):
        g = df.groupby(col).agg(encounters=("los", "size"), mean_los=("los", "mean"), long_stay_pct=("long_stay", "mean"),
                                bed_days=("los", "sum"), readmit30_pct=("readmit30", "mean")).reset_index()
        g["long_stay_pct"] *= 100; g["readmit30_pct"] *= 100
        g["share_of_bed_days_pct"] = g["bed_days"] / g["bed_days"].sum() * 100
        g.to_csv(C.OUTPUTS / f"bi_by_{col}.csv", index=False)
    bi["long_stay_encounters_pct"] = float(df["long_stay"].mean() * 100)
    bi["long_stay_bed_days_pct"] = float(df.loc[df.long_stay == 1, "los"].sum() / df["los"].sum() * 100)
    pd.DataFrame({"day": np.arange(1, C.MAX_LOS + 1), "encounters": [(df.los == d).sum() for d in range(1, C.MAX_LOS + 1)],
                  "hazard": base_hazard(df["los"].to_numpy())}).to_csv(C.OUTPUTS / "bi_los_distribution.csv", index=False)

    # ---- write outputs ----
    cal = M.calibration_table(r["label"].to_numpy(), r["p_model"].to_numpy())
    cal.to_csv(C.OUTPUTS / "calibration_discharge24h.csv", index=False)
    r_out = r[["day_done", "label", "p_model", "p_base", "ward"]].copy()
    r_out["p_model"] = r_out["p_model"].round(5); r_out["p_base"] = r_out["p_base"].round(5)
    r_out.to_csv(C.OUTPUTS / "test_day_rows.csv.gz", index=False, compression="gzip")
    keep = ["encounter_id", "patient_nbr", "ward", "age", "gender", "admission_type", "admission_source", "diag_group", "disposition",
            "los", "number_inpatient", "readmit30", "readmit_risk", "readmit_flag"] + PMF_COLS
    s_out = scored[keep].copy()
    s_out[PMF_COLS] = s_out[PMF_COLS].round(5); s_out["readmit_risk"] = s_out["readmit_risk"].round(4)
    s_out.to_csv(C.OUTPUTS / "test_scored_encounters.csv.gz", index=False, compression="gzip")
    pd.DataFrame(tuning).to_csv(C.OUTPUTS / "tuning_results.csv", index=False)
    sub.to_csv(C.OUTPUTS / "subgroup_auc.csv", index=False); perday.to_csv(C.OUTPUTS / "per_day_auc.csv", index=False)
    imp.to_csv(C.OUTPUTS / "feature_importance.csv", index=False)
    pd.DataFrame({"day": np.arange(1, C.MAX_LOS + 1), "base_hazard": hb}).to_csv(C.OUTPUTS / "base_hazard.csv", index=False)
    joblib.dump({"los_model": los_model, "readmit_model": rm, "base_hazard": hb, "pmf_cols": PMF_COLS,
                 "readmit_threshold": thr_r, "discharge_threshold": thr}, C.OUTPUTS / "model.joblib", compress=3)
    metrics = {"split": split, "best_params": best_params, "decision_threshold_discharge24h": thr,
               "discharge_24h": day_m, "length_of_stay": los_m, "long_stay": long_m, "ablation": abl, "time_forward_check": fwd,
               "readmission": readm, "ward_backtest": bt, "bi": bi,
               "test_encounters": int(len(te))}
    (C.OUTPUTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (C.OUTPUTS / "data_quality_report.json").write_text(json.dumps(clean.report, indent=2))
    (C.OUTPUTS / "meta.json").write_text(json.dumps({
        "dataset": "UCI Diabetes 130-US hospitals for years 1999-2008 (101,766 encounters)", "raw_sha256": clean.report["sha256"],
        "mirror": C.MIRROR_URL, "seed": C.SEED, "python": platform.python_version(), "sklearn": sklearn.__version__,
        "pandas": pd.__version__, "numpy": np.__version__, "run_seconds": round(time.time() - t0, 1)}, indent=2))
    _log(f"[7/7] outputs written to {C.OUTPUTS} in {time.time() - t0:.0f}s")
    return metrics
