# Data dictionary (columns the project uses)

| Column | Meaning | Used for |
|---|---|---|
| encounter_id, patient_nbr | Encounter and patient ids. Within a patient, encounter_id order is chronological (evidence in RESEARCH.md); the split is by patient | split, history features |
| time_in_hospital -> `los` | Length of stay, 1-14 days (target) | target |
| age | 10-year band, `[0-10)` ... `[90-100)` | feature |
| gender, race | As recorded; missing -> `Unknown` | feature |
| admission_type_id -> `admission_type` | Emergency, Urgent, Elective, Newborn, Trauma center, Not available | feature |
| admission_source_id -> `admission_source` | Emergency room, Physician/clinic referral, Transfer, Other | feature |
| payer_code | Insurance class (39.6% missing -> `Unknown`) | feature |
| medical_specialty -> `ward` | Admitting specialty grouped (Internal Medicine, Cardiology, Surgery, ...). 49% "Not recorded". **A proxy for ward** | feature, ward selector |
| diag_1 -> `diag_group` | ICD-9 primary diagnosis grouped (Circulatory, Diabetes, Respiratory, ...). Assumes a working diagnosis is known at admission (ablation reported) | feature |
| number_outpatient / number_emergency / number_inpatient | Visits in the year before the encounter | features |
| `prior_encounters`, `prior_mean_los`, `last_los` | Built from the same patient's EARLIER encounters only | features |
| discharge_disposition_id -> `disposition` | Where the patient went (home, facility, expired, hospice ...) | BI, readmission cohort |
| readmitted -> `readmit30` | 1 if readmitted within 30 days | readmission target |
| num_lab_procedures, num_procedures, num_medications, number_diagnoses | **Stay totals; grow with stay length** | **excluded** (leakage demo only: AUC 0.746) |
| weight, max_glu_serum, A1Cresult | 97% / 95% / 83% not recorded | not used for stay model |

`None` strings in A1Cresult / max_glu_serum are read as "not tested". Cleaning rules, checks and counts: `outputs/data_quality_report.json`.
