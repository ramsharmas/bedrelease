# Data sources

| Item | Value |
|---|---|
| Dataset | UCI "Diabetes 130-US hospitals for years 1999-2008" (`diabetic_data.csv`, `IDS_mapping.csv`) |
| Original study | Strack B, DeShazo JP, Gennings C, et al. *Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records.* BioMed Research International 2014, 781670. doi:10.1155/2014/781670 (opened and read; the paper is open access). Data = Cerner Health Facts extract, 130 US hospitals and delivery networks, 1999-2008. |
| Copy used | Public GitHub mirror `lugooe/Diabetes`, pinned commit `630243bb0ca527d7c4e059e4364215dfa2f27463` |
| Raw URL | https://raw.githubusercontent.com/lugooe/Diabetes/630243bb0ca527d7c4e059e4364215dfa2f27463/diabetic_data.csv |
| SHA-256 of `diabetic_data.csv` | `0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97` (checked by the pipeline and by a test) |
| Size | 101,766 encounters, 50 columns, 71,518 patients, length of stay 1-14 days |
| Why a mirror | UCI, Kaggle, OpenML and PhysioNet were not reachable from the build environment. The original is available from the UCI Machine Learning Repository. |
| Licence | **Not verified.** The authors' paper is open access (CC BY); check the UCI page for the dataset's own terms before redistributing. |

Re-download: `python run_pipeline.py --download` (hash-checked).

## What the data does NOT contain
No dates or times, no ward or bed ids, no hospital census, no emergency-arrival counts, only diabetic patients, only US hospitals 1999-2008. That is why ward size, empty beds, ER arrivals and the daily census in the dashboard are **assumptions**, and why "ward" means the admitting specialty group (49% of encounters have none recorded).

## Only one real dataset
A second ward/bed-occupancy dataset was searched for but no reachable mirror was found, so the project uses this one dataset for two tasks (length of stay / discharge timing, and 30-day readmission).
