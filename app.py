"""BedRelease Planner dashboard - run with:  streamlit run app.py

Ward in-charge view: which patients are likely to go home in the next 24h, how many beds will be free tomorrow and how
many planned (elective) admissions can be accepted safely. Everything is read from outputs/ written by run_pipeline.py.
Patients are REAL encounters from the UCI Diabetes-130 hospitals file; ward size, free beds, ER arrivals and the daily
census are simulated assumptions (the file has no dates, bed ids or census).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from bedrelease import config as C
from bedrelease.planner import (PlannerInputError, census_probs, poisson_binomial_pmf, safe_elective_admissions,
                                sample_census)

st.set_page_config(page_title="BedRelease Planner", page_icon=":hospital:", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
[data-testid="stMetricValue"] {font-size: 2.3rem; font-weight: 700;}
[data-testid="stMetricLabel"] p {font-size: .78rem; letter-spacing: .06em; text-transform: uppercase; color:#52606D;}
h1 {margin-bottom: 0; padding-bottom: 0; font-size: 2rem !important;}
.tiny {font-size: .82rem; color:#52606D;}
.callout {border-left: 4px solid #0F766E; background:#F0FDFA; padding:.6rem .9rem; border-radius:4px; margin:.4rem 0;}
.warn {border-left: 4px solid #D97706; background:#FFFBEB; padding:.6rem .9rem; border-radius:4px; margin:.4rem 0;}
</style>
""", unsafe_allow_html=True)

TEAL, GREY, AMBER, RED = "#0F766E", "#94A3B8", "#D97706", "#DC2626"


@st.cache_data(show_spinner=False)
def load_outputs(stamp: float = 0.0):
    """`stamp` only exists to invalidate the cache when the pipeline re-writes outputs."""
    out = C.OUTPUTS
    needed = ["metrics.json", "meta.json", "data_quality_report.json", "test_scored_encounters.csv.gz",
              "test_day_rows.csv.gz", "calibration_discharge24h.csv", "per_day_auc.csv", "subgroup_auc.csv",
              "feature_importance.csv", "bi_by_ward.csv", "bi_by_age.csv", "bi_by_admission_type.csv",
              "bi_by_diag_group.csv", "bi_by_disposition.csv", "bi_los_distribution.csv", "model.joblib"]
    missing = [n for n in needed if not (out / n).exists()]
    if missing:
        return None, missing
    rd = lambda n, **k: pd.read_csv(out / n, **k)
    D = dict(metrics=json.loads((out / "metrics.json").read_text()), meta=json.loads((out / "meta.json").read_text()),
             dq=json.loads((out / "data_quality_report.json").read_text()),
             scored=rd("test_scored_encounters.csv.gz"), days=rd("test_day_rows.csv.gz"),
             calib=rd("calibration_discharge24h.csv"), perday=rd("per_day_auc.csv"), sub=rd("subgroup_auc.csv"),
             imp=rd("feature_importance.csv"), los_dist=rd("bi_los_distribution.csv"),
             bi={k: rd(f"bi_by_{k}.csv") for k in ["ward", "age", "admission_type", "diag_group", "disposition"]})
    return D, []


@st.cache_resource(show_spinner=False)
def load_model(stamp: float = 0.0):
    import joblib
    try:
        return joblib.load(C.OUTPUTS / "model.joblib")
    except Exception as e:                                   # e.g. a different scikit-learn version than the one that saved it
        st.error("Could not load outputs/model.joblib (" + type(e).__name__ + "). Run `python run_pipeline.py` (or run_pipeline.bat) "
                 "to retrain it with the packages installed on this computer.")
        st.stop()


_mf = C.OUTPUTS / "metrics.json"
D, missing = load_outputs(_mf.stat().st_mtime if _mf.exists() else 0.0)
if D is None:
    st.error("Model outputs not found: " + ", ".join(missing) + ".  Run `python run_pipeline.py` first.")
    st.stop()
M, scored, days = D["metrics"], D["scored"], D["days"]
PMF_COLS = [f"p_day{i}" for i in range(1, C.MAX_LOS + 1)]

# ------------------------------------------------------------------ language
lang = st.sidebar.radio("Language / Bhasha", ["English", "Hinglish"], key="lang", horizontal=True)
HI = lang == "Hinglish"
t = lambda en, hi: hi if HI else en
pct = lambda x, d=1: f"{x * 100:.{d}f}%"

PAGES = [t("Ward board", "Ward board (aaj ka plan)"), t("New admission", "Naya patient (kitne din rukega)"),
         t("Evaluation", "Evaluation (model kitna sahi)"), t("BI insights", "BI insights (bed-days kahan jaate hain)"),
         t("Data & limits", "Data aur limits"), t("90%+ mode (synthetic in-stay data)", "90%+ mode (synthetic in-stay data)")]
page = st.sidebar.radio(t("Page", "Page"), PAGES, key="page")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<span class='tiny'>" + t("Real patients: UCI Diabetes-130 hospitals (US, 1999-2008, diabetic inpatients). "
                              "Ward size, free beds, ER arrivals and the daily census are <b>simulated assumptions</b>.",
                              "Patients asli hain: UCI Diabetes-130 hospitals (US, 1999-2008, diabetic inpatients). "
                              "Ward size, free beds, ER arrivals aur daily census <b>assumption/simulation</b> hain.") + "</span>",
    unsafe_allow_html=True)

st.title("BedRelease Planner")
st.markdown(t("**Who can go home in the next 24 hours, how many beds will be free tomorrow, and how many planned "
              "admissions can I accept safely?**",
              "**Agle 24 ghante me kaun ghar ja sakta hai, kal kitne bed khali honge, aur kitne planned admission safely le sakte hain?**"))


# ================================================================== WARD BOARD
def ward_board():
    named = [w for w in scored["ward"].value_counts().index if w != "Not recorded" and (scored["ward"] == w).sum() >= 200]
    wards = ["All wards"] + named
    c = st.columns([2, 1, 1, 1, 1.3, 1])
    ward = c[0].selectbox(t("Ward (specialty group)", "Ward (specialty group)"), wards, key="ward",
                          index=wards.index("Internal Medicine") if "Internal Medicine" in wards else 0)
    beds = int(c[1].number_input(t("Beds in ward", "Ward me bed"), 5, 100, C.DEFAULT_WARD_BEDS, 1, key="beds"))
    free = int(c[2].number_input(t("Empty beds now", "Abhi khali bed"), 0, 50, C.DEFAULT_FREE_BEDS, 1, key="free"))
    er = float(c[3].number_input(t("Expected ER arrivals", "ER se aane wale (expected)"), 0.0, 30.0, C.DEFAULT_ER_ARRIVALS, 0.5, key="er"))
    conf = float(c[4].slider(t("Plan confidence", "Plan confidence"), 0.60, 0.99, C.DEFAULT_CONFIDENCE, 0.01, key="conf"))
    seed = int(c[5].number_input(t("Census #", "Census #"), 0, 9999, 7, 1, key="seed", help=t("Different number = different simulated day", "Alag number = alag simulated din")))

    try:
        cen = census_probs(sample_census(scored, beds, seed, ward), PMF_COLS)
    except PlannerInputError as e:
        st.error(str(e)); return
    cen = cen.sort_values("p24", ascending=False).reset_index(drop=True)     # most likely to leave first
    top = st.container()                                                      # KPIs are drawn here after the table is read
    yb, pm_all = days["label"].to_numpy(), days["p_model"].to_numpy()
    rate = lambda m: float(yb[m].mean()) * 100
    st.caption(t(f"Signal: GREEN = P >= {C.GREEN_P:.2f} ({rate(pm_all >= C.GREEN_P):.0f}% of such patient-days really left within 24h), "
                 f"YELLOW = {C.YELLOW_P:.2f}-{C.GREEN_P:.2f} ({rate((pm_all >= C.YELLOW_P) & (pm_all < C.GREEN_P)):.0f}%), "
                 f"RED = below {C.YELLOW_P:.2f} ({rate(pm_all < C.YELLOW_P):.0f}%) - measured on held-out test data. Average over all patient-days: {yb.mean() * 100:.0f}%.",
                 f"Signal: GREEN = P >= {C.GREEN_P:.2f} (aise patient-days me {rate(pm_all >= C.GREEN_P):.0f}% sach me 24h me gaye), "
                 f"YELLOW = {C.YELLOW_P:.2f}-{C.GREEN_P:.2f} ({rate((pm_all >= C.YELLOW_P) & (pm_all < C.GREEN_P)):.0f}%), "
                 f"RED = {C.YELLOW_P:.2f} se neeche ({rate(pm_all < C.YELLOW_P):.0f}%) - held-out test data par measure kiya. Sab patient-days ka average: {yb.mean() * 100:.0f}%."))
    cen["day_in_ward"] = cen["day_done"] + 1
    ekey = f"{ward}|{beds}|{seed}"
    base = pd.DataFrame({
        "patient": [f"P-{int(i) % 100000:05d}" for i in cen["encounter_id"]],
        t("Day in ward", "Ward me din"): cen["day_in_ward"],
        t("Age", "Umar"): cen["age"], t("Diagnosis", "Diagnosis"): cen["diag_group"],
        t("Doctor: ready today", "Doctor: aaj ready"): False})
    st.caption(t("Tick **Doctor: ready today** for patients the doctor has already cleared: their discharge probability becomes 100%.",
                 "Jin patients ko doctor ne clear kar diya unko **Doctor: ready today** tick karo: unki discharge probability 100% ho jayegi."))
    ready_col = t("Doctor: ready today", "Doctor: aaj ready")
    # compute probabilities first (without override) so the editor can show them
    pre = base.copy()
    pre["P(go home <24h)"] = cen["p24"].round(2)
    pre["P(<48h)"] = cen["p48"].round(2)
    pre[t("Expected days left", "Baaki din (expected)")] = cen["expected_remaining_days"].round(1)
    pre[t("Signal", "Signal")] = cen["band"].map({"green": "GREEN", "yellow": "YELLOW", "red": "RED"})
    _hp = M["discharge_24h"].get("high_precision_point")
    if _hp:
        pre[t("Signal", "Signal")] = np.where(cen["p24"].to_numpy() >= _hp["threshold"], "GREEN (high-confidence)", pre[t("Signal", "Signal")])
    pre[t("Follow-up before release?", "Release se pehle follow-up?")] = np.where(cen["readmit_flag"].fillna(0) == 1, "YES - high readmission risk", "")
    edited = st.data_editor(pre, hide_index=True, width='stretch', key=f"ed_{ekey}", height=min(38 * len(pre) + 40, 430),
                            disabled=[c_ for c_ in pre.columns if c_ != ready_col])
    over = edited[ready_col].to_numpy(dtype=bool)
    p24 = np.where(over, 1.0, cen["p24"].to_numpy())

    res = safe_elective_admissions(free, p24, er, conf)
    lo, hi = res["discharge_interval_80"]
    with top:
        k = st.columns(4)
        k[0].metric(t("Expected discharges (24h)", "Expected discharge (24h)"), f"{res['expected_discharges']:.1f}",
                    help=t(f"80% range: {lo} to {hi}", f"80% range: {lo} se {hi}"))
        k[1].metric(t("Expected free beds tomorrow", "Kal expected khali bed"), f"{res['expected_free_beds']:.1f}")
        k[2].metric(t(f"Safe planned admissions ({int(conf * 100)}%)", f"Safe planned admission ({int(conf * 100)}%)"), res["safe_elective"],
                    help=t("Largest number of elective admissions such that ER patients still get a bed with the chosen confidence.",
                           "Itne elective admission lo ki ER patient ko bed mile, chosen confidence ke saath."))
        k[3].metric(t("ER shortfall risk (0 planned)", "ER bed shortage risk (0 planned)"), pct(res["p_er_shortfall_no_elective"]))
        st.markdown(f"<div class='callout'>" + t(
            f"With {free} empty beds, {er:g} expected ER arrivals and {int(over.sum())} doctor-confirmed patients: expect "
            f"<b>{res['expected_discharges']:.1f}</b> discharges (80% range {lo}-{hi}); you can accept <b>{res['safe_elective']}</b> planned "
            f"admission(s) and still have a bed for ER with {int(conf * 100)}% confidence.",
            f"{free} khali bed, {er:g} expected ER aur {int(over.sum())} doctor-confirmed patient ke saath: <b>{res['expected_discharges']:.1f}</b> "
            f"discharge expected (80% range {lo}-{hi}); aap <b>{res['safe_elective']}</b> planned admission le sakte ho aur ER ke liye bed "
            f"{int(conf * 100)}% confidence se rahega.") + "</div>", unsafe_allow_html=True)


    pmf = poisson_binomial_pmf(p24)
    fig = go.Figure(go.Bar(x=np.arange(len(pmf)), y=pmf, marker_color=[TEAL if lo <= i <= hi else GREY for i in range(len(pmf))]))
    fig.update_layout(height=290, margin=dict(l=10, r=10, t=40, b=30), title=t("How many patients will be discharged in the next 24h? (teal = 80% range)",
                      "Agle 24h me kitne discharge honge? (teal = 80% range)"), xaxis_title=t("discharges", "discharge"), yaxis_title="probability")
    cols = st.columns([1.3, 1])
    cols[0].plotly_chart(fig, width='stretch')
    confs = np.arange(0.60, 0.995, 0.05)
    curve = [safe_elective_admissions(free, p24, er, float(x))["safe_elective"] for x in confs]
    f2 = go.Figure(go.Scatter(x=[int(round(x * 100)) for x in confs], y=curve, mode="lines+markers", line=dict(color=AMBER)))
    f2.update_layout(height=290, margin=dict(l=10, r=10, t=40, b=30), title=t("Safe planned admissions vs confidence", "Safe planned admission vs confidence"),
                     xaxis_title="confidence %", yaxis_title=t("admissions", "admission"))
    cols[1].plotly_chart(f2, width='stretch')

    with st.expander(t("Reveal what actually happened in the data (honesty check)", "Asli data me kya hua dikhao (honesty check)")):
        actual = ((cen["los"] - cen["day_done"]) == 1).to_numpy()
        st.write(t(f"Actually discharged within the next day: **{int(actual.sum())}** patients (model expected {res['expected_discharges']:.1f}; "
                   f"doctor overrides above are NOT counted in the expectation shown here).",
                   f"Asli me agle din discharge hue: **{int(actual.sum())}** patient (model ne {res['expected_discharges']:.1f} expect kiya tha)."))
        chk = pd.DataFrame({"patient": base["patient"], "P(go home <24h)": cen["p24"].round(2), "actually_went_home": np.where(actual, "yes", "no")})
        st.dataframe(chk, hide_index=True, width='stretch')
        st.caption(t("Single-day numbers are noisy; the Evaluation page shows the 2,000-day backtest.", "Ek din ke numbers noisy hote hain; Evaluation page pe 2,000-din ka backtest hai."))
    st.markdown("<div class='tiny'>" + C.ward_note + t(" Patients are real test-set encounters (never seen in training); the ward composition and day are simulated.",
                " Patients asli test-set encounters hain (training me nahi the); ward ka mix aur din simulated hai.") + "</div>", unsafe_allow_html=True)


# ================================================================== NEW ADMISSION
def new_admission():
    bundle = load_model(_mf.stat().st_mtime)
    lm = bundle["los_model"]
    cats = {c_: list(cat) for c_, cat in zip(lm.enc.cat, lm.enc.enc.categories_)}
    st.subheader(t("How long will this new patient stay?", "Ye naya patient kitne din rukega?"))
    c = st.columns(4)
    age = c[0].selectbox(t("Age band", "Umar"), cats["age"], index=cats["age"].index("[60-70)"), key="n_age")
    gender = c[1].selectbox(t("Gender", "Gender"), [g for g in cats["gender"] if g != "Unknown"], key="n_gender")
    race = c[2].selectbox("Race", cats["race"], index=cats["race"].index("Caucasian"), key="n_race")
    payer = c[3].selectbox(t("Payer", "Payer"), cats["payer_code"], index=cats["payer_code"].index("MC"), key="n_payer")
    c = st.columns(4)
    atype = c[0].selectbox(t("Admission type", "Admission type"), cats["admission_type"], index=cats["admission_type"].index("Emergency"), key="n_atype")
    asrc = c[1].selectbox(t("Admission source", "Admission source"), cats["admission_source"], index=0, key="n_asrc")
    ward = c[2].selectbox("Ward", cats["ward"], index=cats["ward"].index("Internal Medicine"), key="n_ward")
    diag = c[3].selectbox(t("Working diagnosis group", "Working diagnosis"), cats["diag_group"], index=cats["diag_group"].index("Circulatory"), key="n_diag")
    c = st.columns(6)
    n_out = c[0].number_input(t("Outpatient visits (last yr)", "Outpatient visits"), 0, 50, 0, key="n_out")
    n_er = c[1].number_input(t("ER visits (last yr)", "ER visits"), 0, 50, 0, key="n_er")
    n_in = c[2].number_input(t("Inpatient stays (last yr)", "Inpatient stays"), 0, 50, 0, key="n_in")
    prior = c[3].number_input(t("Earlier admissions here", "Pehle ke admission"), 0, 50, 0, key="n_prior")
    pm_los = c[4].number_input(t("Avg earlier LOS (days)", "Pehle ka avg LOS"), 0.0, 14.0, 0.0, 0.5, key="n_pmlos")
    last = c[5].number_input(t("Last stay LOS (days)", "Pichla LOS"), 0.0, 14.0, 0.0, 0.5, key="n_last")
    row = pd.DataFrame([dict(race=race, gender=gender, age=age, admission_type=atype, admission_source=asrc, payer_code=payer, ward=ward,
                             diag_group=diag, number_outpatient=n_out, number_emergency=n_er, number_inpatient=n_in,
                             prior_encounters=prior, prior_mean_los=pm_los, last_los=last)])
    pmf = lm.pmf(row)[0]
    cdf = np.cumsum(pmf)
    q = lambda x: int(np.searchsorted(cdf, x) + 1)
    exp_los, p7 = float((pmf * np.arange(1, C.MAX_LOS + 1)).sum()), float(pmf[6:].sum())
    k = st.columns(5)
    k[0].metric(t("Expected stay", "Expected stay"), f"{exp_los:.1f} d")
    k[1].metric(t("Typical (median)", "Median"), f"{q(0.5)} d")
    k[2].metric("P10 - P90", f"{q(0.1)} - {q(0.9)} d")
    k[3].metric(t("Long-stay (7+ d) chance", "Long stay (7+ din) chance"), pct(p7, 0))
    k[4].metric(t("Hospital average", "Hospital average"), f"{M['length_of_stay']['mean_los_actual']:.1f} d")
    f = go.Figure()
    f.add_bar(x=list(range(1, C.MAX_LOS + 1)), y=pmf, marker_color=[AMBER if d >= 7 else TEAL for d in range(1, C.MAX_LOS + 1)], name="P(LOS = day)")
    f.add_scatter(x=list(range(1, C.MAX_LOS + 1)), y=cdf, mode="lines", line=dict(color="#334155"), name=t("cumulative", "cumulative"), yaxis="y2")
    f.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=30), title=t("Predicted length-of-stay distribution (amber = 7+ days)", "Predicted stay distribution (amber = 7+ din)"),
                    xaxis_title=t("days in hospital", "hospital me din"), yaxis_title="probability",
                    yaxis2=dict(overlaying="y", side="right", range=[0, 1], title="cumulative"), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(f, width='stretch')
    st.markdown("<div class='warn'>" + t(
        "Reading guide: this is a probability fan, not a promise. Model skill is modest (ROC-AUC 0.64 for the 24h-discharge task vs 0.60 for a "
        "day-count baseline) - the value is in calibrated probabilities and bed planning, not in guessing one patient's exact day.",
        "Dhyan se padho: ye ek probability fan hai, guarantee nahi. Model ki skill modest hai (24h-discharge ROC-AUC 0.64 vs baseline 0.60) - "
        "kaam ka hissa calibrated probability aur bed planning hai, ek patient ka exact din guess karna nahi.") + "</div>", unsafe_allow_html=True)


# ================================================================== EVALUATION
def evaluation():
    d24, los, ls, rd = M["discharge_24h"], M["length_of_stay"], M["long_stay"], M["readmission"]
    st.subheader(t("Evaluation on held-out PATIENTS (never seen in training)", "Held-out PATIENTS par evaluation (training me kabhi nahi dekhe)"))
    k = st.columns(5)
    k[0].metric("ROC-AUC (24h)", f"{d24['p_model']['roc_auc']:.3f}", f"{d24['p_model']['roc_auc'] - d24['p_base']['roc_auc']:+.3f} vs baseline")
    k[1].metric("PR-AUC (24h)", f"{d24['p_model']['pr_auc']:.3f}", f"{d24['p_model']['pr_auc'] - d24['p_base']['pr_auc']:+.3f} vs baseline")
    op = d24["operating_point"]
    k[2].metric(t("Precision @ alert", "Precision @ alert"), pct(op["precision"], 1), help=f"threshold {op['threshold']:.2f}")
    k[3].metric(t("Recall @ alert", "Recall @ alert"), pct(op["recall"], 1))
    k[4].metric(t("Calibration error (ECE)", "Calibration error (ECE)"), f"{d24['p_model']['ece']:.3f}")
    st.caption(t(f"{M['split']['test']['encounters']:,} test encounters, {d24['rows']:,} patient-day rows ({d24['positive_rate_pct']:.1f}% are discharge-within-24h). "
                 f"Split by patient: train {M['split']['train']['encounters']:,} / valid {M['split']['valid']['encounters']:,} / test {M['split']['test']['encounters']:,}.",
                 f"{M['split']['test']['encounters']:,} test encounter, {d24['rows']:,} patient-day rows ({d24['positive_rate_pct']:.1f}% 24h me discharge). "
                 f"Patient-wise split: train {M['split']['train']['encounters']:,} / valid {M['split']['valid']['encounters']:,} / test {M['split']['test']['encounters']:,}."))
    tabs = st.tabs([t("Threshold explorer", "Threshold explorer"), t("ROC / PR", "ROC / PR"), t("Calibration", "Calibration"),
                    t("Leakage & time check", "Leakage aur time check"), t("Per-day & subgroups", "Per-day aur subgroups"),
                    t("Ward backtest", "Ward backtest"), t("Length of stay", "Length of stay"), t("Readmission", "Readmission")])
    y, pm_, pb_ = days["label"].to_numpy(), days["p_model"].to_numpy(), days["p_base"].to_numpy()

    with tabs[0]:
        thr = st.slider(t("Alert threshold: flag patient if P(go home <24h) >= ", "Alert threshold: patient flag karo agar P(24h me ghar) >= "),
                        0.05, 0.90, float(round(M["decision_threshold_discharge24h"], 2)), 0.01, key="thr")
        def conf(p):
            yh = p >= thr
            tp = int((yh & (y == 1)).sum()); fp = int((yh & (y == 0)).sum()); fn = int((~yh & (y == 1)).sum()); tn = int((~yh & (y == 0)).sum())
            pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
            return tp, fp, fn, tn, pr, rc, 2 * pr * rc / max(pr + rc, 1e-9)
        a, b = conf(pm_), conf(pb_)
        c = st.columns(2)
        for col, name, r in [(c[0], t("Model", "Model"), a), (c[1], t("Baseline (day-count only)", "Baseline (sirf din ginti)"), b)]:
            col.markdown(f"**{name}**")
            col.metric("Precision / Recall / F1", f"{r[4]:.3f} / {r[5]:.3f} / {r[6]:.3f}")
            col.dataframe(pd.DataFrame([[r[0], r[1]], [r[2], r[3]]], index=[t("went home <24h", "24h me gaya"), t("stayed", "ruka")],
                          columns=[t("flagged", "flagged"), t("not flagged", "not flagged")]), width='stretch')
        hp = d24.get("high_precision_point")
        if hp:
            st.success(t(
                f"High-precision mode (target 90%): threshold {hp['threshold']:.2f} was picked on VALIDATION data. On unseen test data it flagged {hp['tp'] + hp['fp']} patient-days "
                f"and {hp['precision'] * 100:.1f}% of them really left within 24h ({hp['tp']} right, {hp['fp']} wrong). Catch: it finds only {hp['recall'] * 100:.1f}% of all discharges. "
                f"Overall accuracy is {hp['accuracy'] * 100:.1f}% vs {d24['majority_class_accuracy'] * 100:.1f}% for always saying 'stays'.",
                f"High-precision mode (target 90%): threshold {hp['threshold']:.2f} VALIDATION data par chuna gaya. Unseen test par {hp['tp'] + hp['fp']} patient-days flag hue "
                f"aur {hp['precision'] * 100:.1f}% sach me 24h me gaye ({hp['tp']} sahi, {hp['fp']} galat). Catch: yeh sirf {hp['recall'] * 100:.1f}% discharges pakadta hai. "
                f"Overall accuracy {hp['accuracy'] * 100:.1f}% hai, jabki hamesha 'rukega' bolne par {d24['majority_class_accuracy'] * 100:.1f}%."))
        cc = d24.get("confident_calls")
        if cc:
            st.success(t(
                f"90%+ accuracy mode (confident calls only): say 'stays' if P < {cc['lo']:.2f}, 'goes home' if P >= {cc['hi']:.2f}, and ABSTAIN in between (cuts chosen on validation). "
                f"On unseen test data this covers {cc['coverage'] * 100:.1f}% of patient-days ({cc['n_calls']:,} calls) and is right {cc['accuracy_on_confident_calls'] * 100:.1f}% of the time. "
                f"On the other {100 - cc['coverage'] * 100:.0f}% the model says 'not sure'. Accuracy on ALL days stays ~77%.",
                f"90%+ accuracy mode (sirf confident calls): P < {cc['lo']:.2f} to 'rukega', P >= {cc['hi']:.2f} to 'ghar jayega', beech me ABSTAIN (cuts validation par chune). "
                f"Unseen test par yeh {cc['coverage'] * 100:.1f}% patient-days cover karta hai ({cc['n_calls']:,} calls) aur {cc['accuracy_on_confident_calls'] * 100:.1f}% sahi hai. "
                f"Baaki {100 - cc['coverage'] * 100:.0f}% par model 'pakka nahi' bolta hai. Saare din par accuracy ~77% hi rehti hai."))
        st.caption(t("Precision = of flagged patients, how many really left within 24h. Recall = of those who left, how many we flagged. Moving the slider trades one for the other.",
                     "Precision = flagged me se kitne sach me 24h me gaye. Recall = jo gaye unme se kitne flag hue. Slider se ek badhta hai to doosra ghatta hai."))
    with tabs[1]:
        fpr, tpr, _ = roc_curve(y, pm_); fpr_b, tpr_b, _ = roc_curve(y, pb_)
        pr_, rc_, _ = precision_recall_curve(y, pm_); pr_b, rc_b, _ = precision_recall_curve(y, pb_)
        c = st.columns(2)
        f = go.Figure(); f.add_scatter(x=fpr, y=tpr, name="model", line=dict(color=TEAL)); f.add_scatter(x=fpr_b, y=tpr_b, name="baseline", line=dict(color=GREY))
        f.add_scatter(x=[0, 1], y=[0, 1], name="chance", line=dict(color="#CBD5E1", dash="dash"))
        f.update_layout(height=360, title=f"ROC  (AUC {d24['p_model']['roc_auc']:.3f} vs {d24['p_base']['roc_auc']:.3f})", xaxis_title="false positive rate", yaxis_title="recall", margin=dict(l=10, r=10, t=40, b=30))
        c[0].plotly_chart(f, width='stretch')
        f = go.Figure(); f.add_scatter(x=rc_, y=pr_, name="model", line=dict(color=TEAL)); f.add_scatter(x=rc_b, y=pr_b, name="baseline", line=dict(color=GREY))
        f.add_hline(y=float(y.mean()), line=dict(color="#CBD5E1", dash="dash"), annotation_text="base rate")
        f.update_layout(height=360, title=f"Precision-Recall  (PR-AUC {d24['p_model']['pr_auc']:.3f} vs {d24['p_base']['pr_auc']:.3f})", xaxis_title="recall", yaxis_title="precision", margin=dict(l=10, r=10, t=40, b=30))
        c[1].plotly_chart(f, width='stretch')
        ta, tb = d24["top20pct_alert"], d24["top20pct_alert_baseline"]
        st.write(t(f"Flagging the top 20% of patient-days: model precision {ta['precision']:.3f} / recall {ta['recall']:.3f}; baseline {tb['precision']:.3f} / {tb['recall']:.3f}.",
                   f"Top 20% patient-days flag karne par: model precision {ta['precision']:.3f} / recall {ta['recall']:.3f}; baseline {tb['precision']:.3f} / {tb['recall']:.3f}."))
    with tabs[2]:
        cal = D["calib"]
        f = go.Figure(); f.add_scatter(x=cal["predicted"], y=cal["actual"], mode="lines+markers", name="model", line=dict(color=TEAL))
        f.add_scatter(x=[0, 0.8], y=[0, 0.8], name="perfect", line=dict(color="#CBD5E1", dash="dash"))
        f.update_layout(height=380, title=f"Calibration: when the model says X%, X% really leave  (ECE {d24['p_model']['ece']:.3f})", xaxis_title="predicted probability", yaxis_title="observed share", margin=dict(l=10, r=10, t=40, b=30))
        st.plotly_chart(f, width='stretch')
        st.caption(t("Calibration is the model's strongest point: the probabilities can be added up to plan beds.", "Calibration model ka sabse strong point hai: probabilities ko jodke bed plan kar sakte hain."))
    with tabs[3]:
        ab, tf = M["ablation"], M["time_forward_check"]
        df_ab = pd.DataFrame({"feature set": [t("Admission-time only", "Sirf admission-time"), t("+ working diagnosis (USED)", "+ working diagnosis (USE HUA)"), t("+ stay-total counters (LEAKY, not used)", "+ stay-total counters (LEAKY, use nahi)")],
                              "ROC-AUC": [ab["admission_only"]["roc_auc"], ab["with_diagnosis"]["roc_auc"], ab["leaky"]["roc_auc"]],
                              "MAE median (days)": [ab["admission_only"]["mae_median"], ab["with_diagnosis"]["mae_median"], ab["leaky"]["mae_median"]]})
        st.dataframe(df_ab.round(3), hide_index=True, width='stretch')
        st.markdown("<div class='warn'>" + t(
            "Lab, medication and procedure counters are totals for the whole stay, so they grow with length of stay. Adding them lifts AUC to 0.75 - that looks impressive but is leakage, so they are excluded.",
            "Lab, medication, procedure counters poore stay ke totals hain, isliye stay badhne par badhte hain. Inhe jodne par AUC 0.75 ho jata hai - impressive dikhta hai par leakage hai, isliye exclude kiye.") + "</div>", unsafe_allow_html=True)
        st.write(t(f"Time-forward check ({tf['note']}): ROC-AUC {tf['roc_auc_model']:.3f} vs baseline {tf['roc_auc_baseline']:.3f} (train {tf['train_encounters']:,} / test {tf['test_encounters']:,} encounters).",
                   f"Time-forward check: ROC-AUC {tf['roc_auc_model']:.3f} vs baseline {tf['roc_auc_baseline']:.3f} (train {tf['train_encounters']:,} / test {tf['test_encounters']:,})."))
    with tabs[4]:
        pdv = D["perday"]
        f = go.Figure(); f.add_bar(x=pdv["day_done"] + 1, y=pdv["roc_auc_model"], name="model", marker_color=TEAL); f.add_bar(x=pdv["day_done"] + 1, y=pdv["roc_auc_baseline"], name="baseline", marker_color=GREY)
        f.update_layout(barmode="group", height=320, title=t("ROC-AUC by day of stay (skill fades on long stays)", "Stay ke din ke hisab se ROC-AUC (lambe stay me skill kam)"), xaxis_title=t("day in ward", "ward me din"), yaxis=dict(range=[0.4, 0.75]), margin=dict(l=10, r=10, t=40, b=30))
        st.plotly_chart(f, width='stretch')
        st.dataframe(D["sub"].round(3), hide_index=True, width='stretch')
        imp = D["imp"].sort_values("logloss_increase")
        f = go.Figure(go.Bar(x=imp["logloss_increase"], y=imp["feature"], orientation="h", marker_color=TEAL))
        f.update_layout(height=380, title=t("Permutation importance (log-loss increase when shuffled)", "Permutation importance (shuffle karne par log-loss kitna badha)"), margin=dict(l=10, r=10, t=40, b=30))
        st.plotly_chart(f, width='stretch')
    with tabs[5]:
        wb = M["ward_backtest"]
        st.write(t(f"{wb['ward_days']:,} simulated ward-days: {wb['beds']} beds, {wb['free_beds_now']} empty, ER arrivals {wb['er_arrivals_assumed']:g}/day, confidence {int(wb['confidence'] * 100)}%. Patients are real test encounters; ward composition is simulated.",
                   f"{wb['ward_days']:,} simulated ward-days: {wb['beds']} bed, {wb['free_beds_now']} khali, ER {wb['er_arrivals_assumed']:g}/din, confidence {int(wb['confidence'] * 100)}%. Patients asli test encounters; ward mix simulated."))
        tbl = pd.DataFrame({t("Method", "Method"): ["Model", t("Day-count baseline", "Din-ginti baseline"), t("Naive constant", "Naive constant")],
                            t("MAE of daily discharges", "Daily discharge MAE"): [wb[k]["mae_discharges"] for k in ["model", "baseline_day", "naive_constant"]],
                            t("80% interval coverage %", "80% interval coverage %"): [wb[k]["interval80_coverage_pct"] for k in ["model", "baseline_day", "naive_constant"]],
                            t("Avg safe planned admissions", "Avg safe planned admission"): [wb[k]["avg_safe_elective"] for k in ["model", "baseline_day", "naive_constant"]],
                            t("Plan failed % (target <= 10)", "Plan fail % (target <= 10)"): [wb[k]["plan_failed_pct"] for k in ["model", "baseline_day", "naive_constant"]]})
        st.dataframe(tbl.round(2).astype(object).map(lambda v: f'{v:.2f}' if isinstance(v, float) else v), hide_index=True, width='stretch')
        pk = wb["precision_at_k"]
        st.write(t(f"Precision@K (K = actual number of discharges): model {pk['model']:.3f} vs baseline {pk['baseline_day_hazard']:.3f} vs random {pk['random']:.3f}.",
                   f"Precision@K: model {pk['model']:.3f} vs baseline {pk['baseline_day_hazard']:.3f} vs random {pk['random']:.3f}."))
        st.markdown("<div class='warn'>" + t("Honest read: ward totals improve only slightly (MAE 1.80 vs 1.83 vs 1.85, about +0.07 to +0.2 safe admissions a day). The clear win is ranking WHICH patients are likely to leave (precision@K 0.34 vs 0.29 vs 0.23 random).",
            "Sach: ward totals me sudhaar thoda hai (MAE 1.80 vs 1.83 vs 1.85, roz +0.07 se +0.2 safe admission). Asli jeet ye hai ki KAUN patient jaane wala hai uski ranking behtar hai (precision@K 0.34 vs 0.29 vs 0.23 random).") + "</div>", unsafe_allow_html=True)
    with tabs[6]:
        k = st.columns(4)
        k[0].metric("MAE (median, days)", f"{los['mae_median_model']:.2f}", f"{los['mae_median_model'] - los['mae_median_baseline']:+.2f} vs baseline", delta_color="inverse")
        k[1].metric(t("Mean predicted / actual", "Mean predicted / actual"), f"{los['mean_los_predicted']:.2f} / {los['mean_los_actual']:.2f}")
        k[2].metric(t("P10-P90 coverage", "P10-P90 coverage"), f"{los['p10_p90_coverage_pct']:.1f}%", help="P10-P90 is nominally an 80% band; stays are whole days so it covers a bit more")
        k[3].metric(t("Long-stay 7+d AUC", "Long-stay 7+d AUC"), f"{ls['roc_auc']:.3f}")
        t20 = ls["top20pct_flag"]
        st.write(t(f"Long-stay flag at admission (top 20%): precision {t20['precision']:.3f}, recall {t20['recall']:.3f}. Long stays are {M['bi']['long_stay_encounters_pct']:.1f}% of encounters but {M['bi']['long_stay_bed_days_pct']:.1f}% of bed-days.",
                   f"Admission par long-stay flag (top 20%): precision {t20['precision']:.3f}, recall {t20['recall']:.3f}. Long stay {M['bi']['long_stay_encounters_pct']:.1f}% encounters hain par {M['bi']['long_stay_bed_days_pct']:.1f}% bed-days."))
    with tabs[7]:
        fl, rb = rd["flag_top20pct"], rd["rule_baseline_2plus_prior_inpatient"]
        k = st.columns(4)
        k[0].metric("ROC-AUC", f"{rd['roc_auc']:.3f}"); k[1].metric("PR-AUC", f"{rd['pr_auc']:.3f}", help=f"base rate {rd['base_rate_pct']:.1f}%")
        k[2].metric(t("Top-20% flag precision / recall", "Top-20% flag precision / recall"), f"{fl['precision']:.3f} / {fl['recall']:.3f}")
        k[3].metric(t("Top-decile lift", "Top-decile lift"), f"{rd['lift_top_decile']:.2f}x")
        st.write(t(f"Simple rule '2+ prior inpatient stays': precision {rb['precision']:.3f} / recall {rb['recall']:.3f}. Cohort: {rd['cohort']} ({rd['test_rows']:,} test rows). {rd['note']}.",
                   f"Simple rule '2+ pichhle inpatient stay': precision {rb['precision']:.3f} / recall {rb['recall']:.3f}. Cohort: {rd['cohort']} ({rd['test_rows']:,} rows). {rd['note']}."))


# ================================================================== BI
def bi():
    st.subheader(t("Where do the bed-days go?", "Bed-days kahan ja rahe hain?"))
    st.markdown("<div class='callout'>" + t(
        f"Long stays (7+ days) are only <b>{M['bi']['long_stay_encounters_pct']:.1f}%</b> of admissions but use <b>{M['bi']['long_stay_bed_days_pct']:.1f}%</b> of all bed-days.",
        f"Long stay (7+ din) sirf <b>{M['bi']['long_stay_encounters_pct']:.1f}%</b> admission hain par <b>{M['bi']['long_stay_bed_days_pct']:.1f}%</b> bed-days kha jate hain.") + "</div>", unsafe_allow_html=True)
    ld = D["los_dist"]
    c = st.columns(2)
    f = go.Figure(go.Bar(x=ld["day"], y=ld["encounters"], marker_color=TEAL)); f.update_layout(height=300, title=t("Length of stay distribution (all encounters)", "Stay distribution (sab encounters)"), margin=dict(l=10, r=10, t=40, b=30))
    c[0].plotly_chart(f, width='stretch')
    f = go.Figure(go.Scatter(x=ld["day"], y=ld["hazard"], mode="lines+markers", line=dict(color=AMBER))); f.update_layout(height=300, title=t("Chance of going home on each day, given still in hospital", "Har din ghar jaane ka chance, jo abhi tak admit hain"), margin=dict(l=10, r=10, t=40, b=30))
    c[1].plotly_chart(f, width='stretch')
    choice = st.selectbox(t("Break down by", "Is hisab se dekho"), [("ward", t("Ward (specialty)", "Ward (specialty)")), ("age", t("Age band", "Umar")), ("admission_type", t("Admission type", "Admission type")),
                          ("diag_group", t("Diagnosis group", "Diagnosis group")), ("disposition", t("Discharge destination", "Discharge destination"))], format_func=lambda x: x[1], key="bi_by")[0]
    df = D["bi"][choice].copy()
    key = df.columns[0]
    df["share_of_encounters_pct"] = df["encounters"] / df["encounters"].sum() * 100
    f = go.Figure()
    f.add_bar(x=df[key].astype(str), y=df["share_of_encounters_pct"], name=t("% of admissions", "% admissions"), marker_color=GREY)
    f.add_bar(x=df[key].astype(str), y=df["share_of_bed_days_pct"], name=t("% of bed-days", "% bed-days"), marker_color=TEAL)
    f.update_layout(barmode="group", height=360, margin=dict(l=10, r=10, t=40, b=30), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(f, width='stretch')
    st.dataframe(df.round(2), hide_index=True, width='stretch')
    st.caption(t("'Not recorded' ward = the file has no admitting specialty for ~49% of encounters.", "'Not recorded' ward = file me ~49% encounters ka specialty nahi diya hua."))


# ================================================================== DATA & LIMITS
def data_limits():
    dq, meta = D["dq"], D["meta"]
    st.subheader(t("Data, assumptions and limits", "Data, assumptions aur limits"))
    st.markdown(t(f"""
**Real dataset (one):** {meta['dataset']}. Obtained from a public GitHub mirror (commit pinned, SHA-256 `{meta['raw_sha256'][:16]}...` verified). Licence was not verified. US hospitals, 1999-2008, diabetic patients only.

**Used in two tasks:** length-of-stay / discharge-in-24h, and 30-day readmission.
""", f"""
**Asli dataset (ek):** {meta['dataset']}. Public GitHub mirror se liya (commit pinned, SHA-256 `{meta['raw_sha256'][:16]}...` verify hua). Licence verify nahi kiya. US hospitals, 1999-2008, sirf diabetic patients.

**Do tasks me use hua:** length-of-stay / 24h-discharge, aur 30-din readmission.
"""))
    st.markdown("<div class='warn'><b>" + t("What is simulated / assumed", "Kya simulated / assumption hai") + "</b><br>" + t(
        "Ward size, empty beds now, ER arrivals, confidence level, and the daily census (patients are real, the day-of-stay and ward mix are sampled). "
        "The file has no dates, ward ids, bed ids or hospital census. Ward = admitting specialty group (a proxy; 49% not recorded).",
        "Ward size, abhi khali bed, ER arrivals, confidence, aur daily census (patients asli hain, stay ka din aur ward mix sample kiya gaya). "
        "File me date, ward id, bed id ya census nahi hai. Ward = admitting specialty group (proxy; 49% recorded nahi).") + "</div>", unsafe_allow_html=True)
    c = st.columns(4)
    c[0].metric(t("Encounters", "Encounters"), f"{dq['rows_after_cleaning']:,}"); c[1].metric(t("Patients", "Patients"), f"{dq['patients']:,}")
    c[2].metric("LOS range", f"{dq['los_min']}-{dq['los_max']} d"); c[3].metric(t("30-day readmission", "30-din readmission"), f"{dq['readmit30_rate_pct']:.1f}%")
    st.write(t("Missing share (%) of important columns:", "Important columns me missing (%):"))
    st.dataframe(pd.Series(dq["missing_share_pct"], name="missing %").to_frame(), width='stretch')
    st.markdown(t("""
**Limits you should say out loud**
- Model skill is modest (ROC-AUC 0.643 vs 0.595 baseline); ward totals improve only slightly.
- Diabetic inpatients in US hospitals, 1999-2008: not a Pune or Indian hospital. Retrain on local data before any real use.
- Readmission probabilities are class-balanced, so treat them as a risk rank, not a percentage.
- Decision-support prototype for demonstration; not a clinical device. A doctor always decides on discharge.
""", """
**Ye limits khud bolna**
- Model ki skill modest hai (ROC-AUC 0.643 vs 0.595 baseline); ward totals me sudhaar thoda hai.
- US hospitals, 1999-2008, diabetic inpatients: Pune ya Indian hospital nahi. Real use se pehle local data par retrain karo.
- Readmission probabilities class-balanced hain, isliye risk rank samjho, percentage nahi.
- Ye decision-support prototype hai, clinical device nahi. Discharge ka faisla hamesha doctor ka.
"""))
    with st.expander("Run details"):
        st.json(meta)


def synthetic_page():
    import joblib, json as _j
    from pathlib import Path as _P
    o = _P(__file__).parent / "outputs"
    sm = _j.load(open(o / "synthetic_metrics.json"))
    st.subheader(t("90%+ mode: model trained on SYNTHETIC in-stay data", "90%+ mode: SYNTHETIC in-stay data par trained model"))
    st.warning(t("SYNTHETIC: admissions are real (UCI), but daily signals (doctor's plan, vitals, mobility, CRP trend, IV drug) are SIMULATED. "
                 "This shows what accuracy becomes possible if a hospital records these columns. It is not a real-hospital result.",
                 "SYNTHETIC: admissions real (UCI) hain, par daily signals (doctor plan, vitals, mobility, CRP trend, IV drug) SIMULATED hain. "
                 "Yeh dikhata hai ki hospital yeh columns record kare to kitni accuracy possible hai. Real-hospital result nahi hai."))
    k = st.columns(5)
    k[0].metric("Accuracy", f"{sm['accuracy']*100:.1f}%", f"vs {sm['majority_class_accuracy']*100:.1f}% always-'stays'")
    k[1].metric("Precision", f"{sm['precision']*100:.1f}%"); k[2].metric("Recall", f"{sm['recall']*100:.1f}%")
    k[3].metric("F1", f"{sm['f1']:.3f}"); k[4].metric("ROC-AUC", f"{sm['roc_auc']:.3f}")
    st.caption(t(f"Patient-wise split, test {sm['rows']['test']:,} synthetic patient-days. Without the doctor's-plan signal: ROC-AUC {sm['ablation_without_doctor_plan_roc_auc']:.3f}, accuracy {sm['accuracy_without_doctor_plan']*100:.1f}%. Real data (no in-stay signals): ROC-AUC {M['discharge_24h']['p_model']['roc_auc']:.3f}, accuracy ~77%.",
                 f"Patient-wise split, test {sm['rows']['test']:,} synthetic patient-days. Doctor-plan signal ke bina: ROC-AUC {sm['ablation_without_doctor_plan_roc_auc']:.3f}, accuracy {sm['accuracy_without_doctor_plan']*100:.1f}%. Real data (in-stay signals nahi): ROC-AUC {M['discharge_24h']['p_model']['roc_auc']:.3f}, accuracy ~77%."))
    b = joblib.load(o / "synthetic_model.joblib")
    st.markdown(t("**Try the pre-trained model**", "**Pre-trained model try karo**"))
    c = st.columns(4)
    v = {"day_done": c[0].number_input(t("Days already stayed", "Kitne din ruk chuka"), 0, 13, 3),
         "doctor_plan_tomorrow": int(c[1].checkbox(t("Doctor plans discharge", "Doctor discharge plan"), True)),
         "vitals_stable": int(c[2].checkbox(t("Vitals stable", "Vitals stable"), True)),
         "mobility_score": c[3].slider(t("Mobility (0-4)", "Mobility (0-4)"), 0, 4, 3),
         "crp_trend": c[0].slider(t("CRP trend (negative = improving)", "CRP trend (minus = sudhaar)"), -3.0, 2.0, -0.8),
         "on_iv_drug": int(c[1].checkbox(t("On IV drug", "IV drug chal raha"), False)),
         "age_num": c[2].slider(t("Age", "Umar"), 5, 95, 65), "emergency": int(c[3].checkbox(t("Emergency admission", "Emergency admission"), True)), "diag_code": 0}
    pr = float(b["model"].predict_proba(pd.DataFrame([v])[b["features"]])[0, 1])
    st.metric(t("P(go home within 24h)", "P(24h me ghar)"), f"{pr*100:.1f}%", t("GOES HOME" if pr >= b["threshold"] else "STAYS", "GHAR JAYEGA" if pr >= b["threshold"] else "RUKEGA"))


{PAGES[0]: ward_board, PAGES[1]: new_admission, PAGES[2]: evaluation, PAGES[3]: bi, PAGES[4]: data_limits, PAGES[5]: synthetic_page}[page]()
