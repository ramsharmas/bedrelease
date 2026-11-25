"""Writes docs/METRICS_SUMMARY.md straight from outputs/metrics.json (so the document can never drift from the numbers)."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
M = json.loads((ROOT / "outputs" / "metrics.json").read_text())
d, los, ls, rd, wb, ab, tf = M["discharge_24h"], M["length_of_stay"], M["long_stay"], M["readmission"], M["ward_backtest"], M["ablation"], M["time_forward_check"]
op, t20, t20b = d["operating_point"], d["top20pct_alert"], d["top20pct_alert_baseline"]
f3 = lambda x: f"{x:.3f}"
L = []
L += ["# Metrics summary (auto-generated from outputs/metrics.json)", "",
      f"Held-out TEST set, split by patient: {M['split']['test']['encounters']:,} encounters / {M['split']['test']['patients']:,} patients "
      f"(train {M['split']['train']['encounters']:,}, valid {M['split']['valid']['encounters']:,}). Tuning and thresholds use the validation part only.", "",
      "## 1. Discharge within the next 24 h (66k patient-day rows, positive rate %.1f%%)" % d["positive_rate_pct"], "",
      "| Metric | Model | Day-count baseline |", "|---|---|---|",
      f"| ROC-AUC | {f3(d['p_model']['roc_auc'])} | {f3(d['p_base']['roc_auc'])} |",
      f"| PR-AUC | {f3(d['p_model']['pr_auc'])} | {f3(d['p_base']['pr_auc'])} |",
      f"| Brier score (lower is better) | {f3(d['p_model']['brier'])} | {f3(d['p_base']['brier'])} |",
      f"| Calibration error (ECE) | {f3(d['p_model']['ece'])} | {f3(d['p_base']['ece'])} |", "",
      "| Alert rule | Precision | Recall | F1 |", "|---|---|---|---|",
      f"| Model, threshold {op['threshold']:.2f} (chosen on validation) | {f3(op['precision'])} | {f3(op['recall'])} | {f3(op['f1'])} |",
      f"| Model, flag top 20% | {f3(t20['precision'])} | {f3(t20['recall'])} | {f3(t20['f1'])} |",
      f"| Baseline, flag top 20% | {f3(t20b['precision'])} | {f3(t20b['recall'])} | {f3(t20b['f1'])} |", "",
      f"Confusion matrix at threshold {op['threshold']:.2f}: TP {op['tp']:,}, FP {op['fp']:,}, FN {op['fn']:,}, TN {op['tn']:,}.", "",
      "## 2. Length of stay", "",
      f"- MAE of median prediction: {los['mae_median_model']:.2f} days (baseline {los['mae_median_baseline']:.2f}).",
      f"- Mean predicted {los['mean_los_predicted']:.2f} vs actual {los['mean_los_actual']:.2f} days.",
      f"- P10-P90 band covers {los['p10_p90_coverage_pct']:.1f}% of actual stays (stays are whole days, so it covers more than the nominal 80%).",
      f"- Long stay (7+ days, {ls['base_rate_pct']:.1f}% of encounters): ROC-AUC {f3(ls['roc_auc'])}, PR-AUC {f3(ls['pr_auc'])}; top-20% flag precision {f3(ls['top20pct_flag']['precision'])}, recall {f3(ls['top20pct_flag']['recall'])}.",
      f"- Long stays use {M['bi']['long_stay_bed_days_pct']:.1f}% of all bed-days ({M['bi']['long_stay_encounters_pct']:.1f}% of encounters).", "",
      "## 3. Leakage and time checks", "",
      "| Feature set | ROC-AUC | MAE median (days) | Used? |", "|---|---|---|---|",
      f"| Admission-time only | {f3(ab['admission_only']['roc_auc'])} | {ab['admission_only']['mae_median']:.2f} | fallback |",
      f"| + working diagnosis group | {f3(ab['with_diagnosis']['roc_auc'])} | {ab['with_diagnosis']['mae_median']:.2f} | **yes** |",
      f"| + stay-total counters (labs, meds, procedures) | {f3(ab['leaky']['roc_auc'])} | {ab['leaky']['mae_median']:.2f} | **no - leakage** |", "",
      f"Time-forward check (train on earliest 70% of encounter ids, test on latest 15%): ROC-AUC {f3(tf['roc_auc_model'])} vs baseline {f3(tf['roc_auc_baseline'])}.", "",
      "## 4. Ward backtest (%s simulated ward-days; %d beds, %d empty, ER arrivals %g/day, confidence %d%%)" % (f"{wb['ward_days']:,}", wb["beds"], wb["free_beds_now"], wb["er_arrivals_assumed"], round(wb["confidence"] * 100)), "",
      "| Method | MAE daily discharges | 80% interval coverage % | Avg safe planned admissions | Plan failed % |", "|---|---|---|---|---|"]
for k, n in [("model", "Model"), ("baseline_day", "Day-count baseline"), ("naive_constant", "Naive constant")]:
    r = wb[k]; L.append(f"| {n} | {r['mae_discharges']:.2f} | {r['interval80_coverage_pct']:.1f} | {r['avg_safe_elective']:.2f} | {r['plan_failed_pct']:.2f} |")
pk = wb["precision_at_k"]
L += ["", f"Precision@K (K = true number of discharges that day): model {f3(pk['model'])}, baseline {f3(pk['baseline_day_hazard'])}, random {f3(pk['random'])}.", "",
      "**Honest read:** ward-total accuracy improves only slightly; the clear gain is in ranking which patients are likely to leave.", "",
      "## 5. 30-day readmission (at discharge; cohort: %s, %s rows, base rate %.1f%%)" % (rd["cohort"], f"{rd['test_rows']:,}", rd["base_rate_pct"]), "",
      f"ROC-AUC {f3(rd['roc_auc'])}, PR-AUC {f3(rd['pr_auc'])}, top-decile lift {rd['lift_top_decile']:.2f}x. Top-20% flag: precision {f3(rd['flag_top20pct']['precision'])}, recall {f3(rd['flag_top20pct']['recall'])}. "
      f"Simple rule '2+ prior inpatient stays': precision {f3(rd['rule_baseline_2plus_prior_inpatient']['precision'])}, recall {f3(rd['rule_baseline_2plus_prior_inpatient']['recall'])}.",
      f"{rd['note']}.", ""]
(ROOT / "docs" / "METRICS_SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
print("written docs/METRICS_SUMMARY.md")
