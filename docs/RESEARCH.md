# Research notes (only sources that were actually opened)

| Source | What it supports in this project |
|---|---|
| Wei J, Zhou J, Zhang Z, et al. *Predicting individual patient and hospital-level discharge using machine learning.* Communications Medicine 4:236 (2024). doi:10.1038/s43856-024-00673-x. https://pmc.ncbi.nlm.nih.gov/articles/PMC11574281 | Predicting discharge within the next 24 h from hospital records (Oxfordshire, UK, three years of data, planned and emergency admissions); gradient-boosted trees beat simple logistic regression; also forecasts at hospital level. Our task framing (24 h discharge, tree-based model, baseline comparison) follows this line of work. Their data is far richer than ours (UK EHR with in-stay signals); our numbers are not comparable. |
| Bishop JA, Javed HA, el-Bouri R, et al. *Improving patient flow during infectious disease outbreaks using machine learning for real-time prediction of patient readiness for discharge.* PLoS ONE 16(11):e0260476 (2021). https://pmc.ncbi.nlm.nih.gov/articles/PMC8610279 | Strategy of scoring every inpatient and **ranking them by likelihood of discharge in the next 24 h**; the top 20% are treated as discharge candidates. Our "top 20% alert" operating point and the precision@K metric follow this idea. |
| Strack B, DeShazo JP, Gennings C, et al. *Impact of HbA1c measurement on hospital readmission rates.* BioMed Research International 2014:781670. doi:10.1155/2014/781670. https://pmc.ncbi.nlm.nih.gov/articles/PMC3996476 | Origin of the dataset: Cerner Health Facts extract, 10 years (1999-2008), 130 US hospitals and delivery networks; ~70,000 diabetic inpatient encounters after the authors' filtering (the public file has 101,766 encounters). |
| Wei et al. JMIR 2025;27:e63765 (search result page only; abstract read, full paper not studied). https://www.jmir.org/2025/1/e63765 | Context only: short-term discharge counts can be predicted hours ahead in multi-site systems. Not used for any number here. |

## Findings from our own data checks (reproducible)
- **encounter_id order is chronological within a patient.** Share of encounters that have a later encounter by the same patient: readmitted=NO 0.9%, >30 days 60.8%, <30 days 71.6%. So "earlier encounters of the same patient" can be used as history features, and "later encounters" must never be used (they would leak the readmission label).
- **Stay-total counters leak.** Adding labs / medications / procedures counts lifts discharge-24h ROC-AUC from 0.643 to 0.746. They are totals for the whole stay, so they are excluded; the test `test_model_beats_baseline_and_leakage_is_flagged` keeps this visible.

## What we did not find
No second reachable real dataset with ward / bed occupancy, so none is claimed.
