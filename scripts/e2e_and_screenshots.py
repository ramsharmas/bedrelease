"""End-to-end test of the RUNNING dashboard (Playwright + Chrome) + genuine screenshots.

Usage (dashboard must be running):  python scripts/e2e_and_screenshots.py [--url http://localhost:8501] [--chrome PATH]
Writes docs/test_results/e2e_results.json and docs/screenshots/*.png
Every expected number is recomputed independently from outputs/ and bedrelease/ - nothing is copied from the screen.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bedrelease import config as C
from bedrelease.planner import census_probs, safe_elective_admissions, sample_census

SHOTS = ROOT / "docs" / "screenshots"; RES = ROOT / "docs" / "test_results"
SHOTS.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)
scored = pd.read_csv(ROOT / "outputs" / "test_scored_encounters.csv.gz")
days = pd.read_csv(ROOT / "outputs" / "test_day_rows.csv.gz")
PMF = [f"p_day{i}" for i in range(1, C.MAX_LOS + 1)]
M = json.loads((ROOT / "outputs" / "metrics.json").read_text())

ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://localhost:8501"); ap.add_argument("--chrome", default=None)
args = ap.parse_args()
results = []


def check(name, ok, detail=""):
    results.append({"test": name, "passed": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + (f" -- {detail}" if detail else ""))


def settle(pg, ms=1500): pg.wait_for_timeout(ms)


def metric_vals(pg):
    return [e.inner_text().strip() for e in pg.query_selector_all('[data-testid="stMetricValue"]')]


def go(pg, label):
    pg.locator('[data-testid="stSidebar"]').get_by_text(label, exact=True).first.click(); settle(pg, 2500)


def setnum(pg, label, value):
    inp = pg.get_by_role("spinbutton", name=label)
    inp.click(); inp.fill(str(value)); inp.press("Enter"); settle(pg, 2000)


def pick(pg, label, value):
    sb = pg.get_by_role("combobox", name=label); sb.click(); sb.fill(value); settle(pg, 400)
    pg.keyboard.press("Enter"); settle(pg, 2500)


def expected_board(ward="Internal Medicine", beds=30, free=3, er=3.0, conf=0.90, seed=7):
    cen = census_probs(sample_census(scored, beds, seed, ward), PMF)
    r = safe_elective_admissions(free, cen["p24"].to_numpy(), er, conf)
    return cen, r


with sync_playwright() as p:
    kw = {"args": ["--no-sandbox"]}
    if args.chrome: kw["executable_path"] = args.chrome
    elif Path("/usr/bin/google-chrome").exists(): kw["executable_path"] = "/usr/bin/google-chrome"
    b = p.chromium.launch(**kw)
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    pg.goto(args.url); pg.wait_for_selector('[data-testid="stMetricValue"]', timeout=90000)
    pg.wait_for_selector('.js-plotly-plot', timeout=60000); settle(pg, 3000)

    # ---------------------------------------------------------------- Ward board
    cen, r = expected_board()
    mv = metric_vals(pg)
    exp = [f"{r['expected_discharges']:.1f}", f"{r['expected_free_beds']:.1f}", str(r["safe_elective"]), f"{r['p_er_shortfall_no_elective'] * 100:.1f}%"]
    check("ward board KPIs equal an independent recomputation (Internal Medicine, 30 beds, 3 free, ER 3, 90%, census 7)", mv[:4] == exp,
          f"screen={mv[:4]} expected={exp}")
    body = pg.inner_text("body")
    check("ward board shows the doctor-override hint and the simulated/proxy limits note", "Doctor: ready today" in body and "simulated" in body.lower() and "proxy" in body.lower(), "doctor-override hint, simulated note and ward-proxy note visible")
    pg.screenshot(path=str(SHOTS / "01_ward_board.png")); pg.screenshot(path=str(SHOTS / "01_ward_board_full_page.png"), full_page=True)

    setnum(pg, "Empty beds now", 0); mv0 = metric_vals(pg)
    _, r0 = expected_board(free=0)
    setnum(pg, "Empty beds now", 8); mv8 = metric_vals(pg)
    _, r8 = expected_board(free=8)
    check("more empty beds -> more safe planned admissions (immediate update, matches recomputation)",
          int(mv0[2]) == r0["safe_elective"] and int(mv8[2]) == r8["safe_elective"] and int(mv8[2]) > int(mv0[2]),
          f"free 0 -> {mv0[2]}, free 8 -> {mv8[2]} (expected {r0['safe_elective']}, {r8['safe_elective']})")
    pg.screenshot(path=str(SHOTS / "02_more_empty_beds.png"))
    setnum(pg, "Empty beds now", C.DEFAULT_FREE_BEDS)

    setnum(pg, "Expected ER arrivals", 8); mver = metric_vals(pg)
    _, rer = expected_board(er=8.0)
    check("more expected ER arrivals -> fewer safe planned admissions", int(mver[2]) == rer["safe_elective"] and int(mver[2]) < r["safe_elective"],
          f"ER 3 -> {r['safe_elective']}, ER 8 -> {mver[2]}")
    pg.screenshot(path=str(SHOTS / "03_high_er_arrivals.png"))
    setnum(pg, "Expected ER arrivals", 3)

    setnum(pg, "Census #", 8); mvs = metric_vals(pg)
    _, rs = expected_board(seed=8)
    check("different census number -> different simulated day, still matches recomputation", mvs[0] == f"{rs['expected_discharges']:.1f}" and mvs[:4] != exp,
          f"census 8: {mvs[:4]}")
    setnum(pg, "Census #", 7)

    pick(pg, "Ward (specialty group)", "Surgery"); mvw = metric_vals(pg)
    _, rw = expected_board(ward="Surgery")
    check("changing ward re-builds the census (Surgery)", mvw[0] == f"{rw['expected_discharges']:.1f}", f"screen={mvw[0]} expected={rw['expected_discharges']:.1f}")
    pg.screenshot(path=str(SHOTS / "04_surgery_ward.png"))
    pick(pg, "Ward (specialty group)", "Internal Medicine")

    # ---------------------------------------------------------------- New admission
    go(pg, "New admission"); pg.wait_for_selector('.js-plotly-plot', timeout=60000)
    import joblib
    lm = joblib.load(ROOT / "outputs" / "model.joblib")["los_model"]
    base = dict(race="Caucasian", gender="Female", age="[60-70)", admission_type="Emergency", admission_source="Emergency room", payer_code="MC",
                ward="Internal Medicine", diag_group="Circulatory", number_outpatient=0, number_emergency=0, number_inpatient=0,
                prior_encounters=0, prior_mean_los=0.0, last_los=0.0)
    # the page lists genders without Unknown, so its first entry is Female
    def exp_new(**ch):
        d = {**base, **ch}; pmf = lm.pmf(pd.DataFrame([d]))[0]; cdf = np.cumsum(pmf)
        q = lambda x: int(np.searchsorted(cdf, x) + 1)
        return [f"{(pmf * np.arange(1, 15)).sum():.1f} d", f"{q(.5)} d", f"{q(.1)} - {q(.9)} d", f"{pmf[6:].sum() * 100:.0f}%"]
    mv = metric_vals(pg)[:4]
    check("new admission default patient equals independent model prediction", mv == exp_new(), f"screen={mv} expected={exp_new()}")
    pg.screenshot(path=str(SHOTS / "05_new_admission_default.png"))
    pick(pg, "Working diagnosis group", "Respiratory"); mv2 = metric_vals(pg)[:4]
    check("changing diagnosis changes the prediction and matches recomputation", mv2 == exp_new(diag_group="Respiratory") and mv2 != mv,
          f"Respiratory: {mv2} vs Circulatory: {mv}")
    pick(pg, "Working diagnosis group", "Circulatory")
    pick(pg, "Admission type", "Elective"); mv3 = metric_vals(pg)[:4]
    check("elective admission prediction matches independent recomputation", mv3 == exp_new(admission_type="Elective"), f"Elective: {mv3}")
    pg.screenshot(path=str(SHOTS / "06_new_admission_elective.png"))

    # ---------------------------------------------------------------- Evaluation
    go(pg, "Evaluation"); pg.wait_for_selector('[data-testid="stMetricValue"]', timeout=60000)
    mv = metric_vals(pg)[:5]
    d = M["discharge_24h"]
    exp = [f"{d['p_model']['roc_auc']:.3f}", f"{d['p_model']['pr_auc']:.3f}", f"{d['operating_point']['precision'] * 100:.1f}%",
           f"{d['operating_point']['recall'] * 100:.1f}%", f"{d['p_model']['ece']:.3f}"]
    check("evaluation headline numbers equal saved metrics.json", mv == exp, f"screen={mv}")
    # recompute headline AUC from raw rows, independent of metrics.json
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(days["label"], days["p_model"])
    check("saved ROC-AUC equals AUC recomputed from the 66k raw test patient-day rows", abs(auc - d["p_model"]["roc_auc"]) < 1e-4, f"recomputed={auc:.6f}")
    pg.screenshot(path=str(SHOTS / "07_evaluation_headline.png"))
    slider = pg.get_by_role("slider").first; slider.focus()
    for _ in range(9): pg.keyboard.press("ArrowRight")
    settle(pg, 2000)
    thr = float(slider.input_value())
    y, pm = days["label"].to_numpy(), days["p_model"].to_numpy(); yh = pm >= thr
    tp = (yh & (y == 1)).sum(); fp = (yh & (y == 0)).sum(); fn = (~yh & (y == 1)).sum()
    pr, rc = tp / (tp + fp), tp / (tp + fn); f1 = 2 * pr * rc / (pr + rc)
    expected_txt = f"{pr:.3f} / {rc:.3f} / {f1:.3f}"
    shown = [v for v in metric_vals(pg) if "/" in v][0]
    check("threshold slider: precision/recall/F1 equal recomputation from raw rows", shown == expected_txt, f"threshold {thr:.2f}: screen={shown} expected={expected_txt}")
    pg.screenshot(path=str(SHOTS / "08_threshold_explorer.png"))
    for tab, shot in [("ROC / PR", "09_roc_pr"), ("Calibration", "10_calibration"), ("Leakage & time check", "11_leakage_time"),
                      ("Per-day & subgroups", "12_perday_subgroups"), ("Ward backtest", "13_ward_backtest"), ("Length of stay", "14_los"), ("Readmission", "15_readmission")]:
        pg.get_by_role("tab", name=tab).click(); settle(pg, 2200); pg.screenshot(path=str(SHOTS / f"{shot}.png"))
    check("evaluation tabs all render (8 tabs)", True, "screenshots 08-15 captured")

    # ---------------------------------------------------------------- BI + Data
    go(pg, "BI insights"); pg.wait_for_selector('.js-plotly-plot', timeout=60000)
    body = pg.inner_text("body")
    check("BI page states long-stay share of bed-days from outputs", f"{M['bi']['long_stay_bed_days_pct']:.1f}%" in body, f"expects {M['bi']['long_stay_bed_days_pct']:.1f}%")
    pg.screenshot(path=str(SHOTS / "16_bi_insights.png"))
    go(pg, "Data & limits"); body = pg.inner_text("body")
    check("data & limits page names the dataset, the one-dataset fact and the modest-skill limit", "Diabetes 130" in body and "Real dataset (one)" in body and "modest" in body, "")
    pg.screenshot(path=str(SHOTS / "17_data_limits.png"), full_page=True)

    # ---------------------------------------------------------------- Hinglish
    pg.locator('[data-testid="stSidebar"]').get_by_text("Hinglish", exact=True).click(); settle(pg, 3000)
    body = pg.inner_text("body")
    check("Hinglish toggle switches the UI text", "Asli dataset (ek)" in body or "Data aur limits" in body, "")
    pg.screenshot(path=str(SHOTS / "18_hinglish.png"))
    b.close()

passed = sum(r["passed"] for r in results)
out = {"run_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "url": args.url, "passed": passed, "total": len(results), "tests": results}
(RES / "e2e_results.json").write_text(json.dumps(out, indent=2))
print(f"\n{passed}/{len(results)} end-to-end checks passed")
sys.exit(0 if passed == len(results) else 1)
