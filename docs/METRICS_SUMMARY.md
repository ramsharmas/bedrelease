# Metrics summary (auto-generated from outputs/metrics.json)

Held-out TEST set, split by patient: 15,108 encounters / 10,728 patients (train 71,326, valid 15,332). Tuning and thresholds use the validation part only.

## 1. Discharge within the next 24 h (66k patient-day rows, positive rate 22.8%)

| Metric | Model | Day-count baseline |
|---|---|---|
| ROC-AUC | 0.643 | 0.595 |
| PR-AUC | 0.346 | 0.297 |
| Brier score (lower is better) | 0.167 | 0.171 |
| Calibration error (ECE) | 0.003 | 0.002 |

| Alert rule | Precision | Recall | F1 |
|---|---|---|---|
| Model, threshold 0.21 (chosen on validation) | 0.288 | 0.751 | 0.416 |
| Model, flag top 20% | 0.349 | 0.306 | 0.326 |
| Baseline, flag top 20% | 0.292 | 0.342 | 0.315 |

Confusion matrix at threshold 0.21: TP 11,345, FP 28,101, FN 3,763, TN 23,072.

## 2. Length of stay

- MAE of median prediction: 2.13 days (baseline 2.25).
- Mean predicted 4.38 vs actual 4.39 days.
- P10-P90 band covers 89.8% of actual stays (stays are whole days, so it covers more than the nominal 80%).
- Long stay (7+ days, 20.6% of encounters): ROC-AUC 0.649, PR-AUC 0.342; top-20% flag precision 0.340, recall 0.330.
- Long stays use 43.5% of all bed-days (20.8% of encounters).

## 3. Leakage and time checks

| Feature set | ROC-AUC | MAE median (days) | Used? |
|---|---|---|---|
| Admission-time only | 0.636 | 2.15 | fallback |
| + working diagnosis group | 0.643 | 2.13 | **yes** |
| + stay-total counters (labs, meds, procedures) | 0.746 | 1.67 | **no - leakage** |

Time-forward check (train on earliest 70% of encounter ids, test on latest 15%): ROC-AUC 0.631 vs baseline 0.592.

## 4. Ward backtest (2,000 simulated ward-days; 30 beds, 3 empty, ER arrivals 3/day, confidence 90%)

| Method | MAE daily discharges | 80% interval coverage % | Avg safe planned admissions | Plan failed % |
|---|---|---|---|---|
| Model | 1.80 | 87.2 | 3.20 | 6.75 |
| Day-count baseline | 1.83 | 86.5 | 3.13 | 6.55 |
| Naive constant | 1.85 | 87.6 | 3.00 | 6.10 |

Precision@K (K = true number of discharges that day): model 0.343, baseline 0.292, random 0.231.

**Honest read:** ward-total accuracy improves only slightly; the clear gain is in ranking which patients are likely to leave.

## 5. 30-day readmission (at discharge; cohort: discharged alive and not to hospice, 14,766 rows, base rate 11.1%)

ROC-AUC 0.671, PR-AUC 0.206, top-decile lift 2.23x. Top-20% flag: precision 0.218, recall 0.344. Simple rule '2+ prior inpatient stays': precision 0.211, recall 0.260.
class_weight balanced, so raw probabilities are over-estimates; use as a risk rank, not a percentage.
