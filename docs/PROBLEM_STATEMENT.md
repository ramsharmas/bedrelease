# Problem statement and proposed solution

## Problem
Every morning a ward in-charge has to answer three questions with little more than experience and a whiteboard:
1. **Who is likely to go home today or tomorrow?**
2. **How many beds will be free tomorrow?**
3. **How many planned (elective) admissions can we accept without leaving an emergency patient waiting for a bed?**

If the in-charge is too cautious, beds stay empty and planned admissions get postponed. If too optimistic, emergency patients wait. Discharge timing is uncertain, and one rule of thumb ("average stay is 4 days") ignores that a patient already on day 7 and a patient admitted today are not alike.

Published work frames this as discharge prediction for patient flow: predict who will leave in the next 24 hours so teams can plan (see `RESEARCH.md`).

## Proposed solution: BedRelease Planner
A decision-support dashboard that turns a patient list into numbers a ward in-charge can act on.

| Need | Feature |
|---|---|
| Who may go home? | Per-patient probability of discharge in 24 h and 48 h, colour signal, expected days left, doctor-confirmed override |
| Beds tomorrow? | Expected free beds with an 80% range (exact Poisson-binomial distribution, not a single guess) |
| Planned admissions? | **Safe planned admissions** at a chosen confidence, accounting for expected ER arrivals |
| Safe release? | 30-day readmission risk flag: "follow-up before release" |
| New patient? | Full length-of-stay distribution, P10-P90, chance of a 7+ day stay |
| Management view (BI) | Where bed-days go: long stays are ~21% of admissions but ~44% of bed-days; by ward, age, diagnosis, discharge destination |
| Trust | Held-out-patient evaluation, calibration, leakage test, time check, threshold slider, ward backtest |

## Evidence (held-out test patients)
- Discharge in 24 h: ROC-AUC 0.643 vs 0.595 baseline; PR-AUC 0.346 vs 0.297; well calibrated (ECE 0.003).
- High-precision mode (threshold chosen on validation): 91.2% precision on test, but finds only 0.9% of discharges (148 flags).
- Confident-calls mode (answers only when sure, abstains otherwise): 90.5% accuracy on 16.7% of patient-days. Over all days accuracy is ~77%, same as always saying "stays".
- Ranking: precision@K 0.343 vs 0.292 baseline vs 0.231 random.
- Planning: 2,000 simulated ward-days, plan failed 6.75% of days at a 90% confidence target.
- Readmission: ROC-AUC 0.671, top-decile lift 2.2x.

## Limits (say them out loud)
One real dataset (UCI Diabetes-130; US, 1999-2008, diabetic inpatients). No dates, ward or bed ids: ward = admitting specialty group (49% not recorded); ward size, empty beds, ER arrivals and census are simulated. Ward-total accuracy improves only slightly (MAE 1.80 vs 1.83 vs 1.85). Retrain on local data before any real use; a doctor always decides on discharge.
