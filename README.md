# BedRelease Planner

A local dashboard for a hospital ward in-charge: **who can go home in the next 24 hours, how many beds will be free tomorrow, and how many planned (elective) admissions can be accepted without leaving an emergency patient without a bed.**

Built for the BIML (Business Intelligence & Machine Learning) course project at ADYPU Pune. Everything runs on your own computer: no internet, no cloud, no account.

> **Be honest about what this is.** Real patient records (UCI "Diabetes 130-US hospitals", 101,766 encounters, 1999-2008, diabetic inpatients only), real held-out evaluation. But the data has no dates, wards, bed ids or hospital census, so **ward size, empty beds, ER arrivals and the daily census are simulated assumptions**. The model beats a simple baseline (ROC-AUC 0.643 vs 0.595) but the edge is modest. Prototype for demonstration, not a clinical device.

## Start in 3 steps

### Windows
1. Install Python 3.11, 3.12 or 3.13 from python.org (tick **Add python.exe to PATH**).
2. Double-click **`setup.bat`** (one time, a few minutes: makes `.venv` and installs packages).
3. Double-click **`run_dashboard.bat`**. Your browser opens at http://localhost:8501.

The results the dashboard reads (`outputs/`) are already included, so it starts at once.
To retrain everything from the raw data: double-click `run_pipeline.bat` (about 2-3 minutes). To run the 28 tests: `run_tests.bat`.

### Linux / macOS
```bash
bash setup_and_run.sh        # creates .venv, installs, starts the dashboard
# retrain:  . .venv/bin/activate && python run_pipeline.py
# tests:    . .venv/bin/activate && python -m pytest -q
```

If the "New admission" page says `model.joblib` could not be loaded, you have a different scikit-learn version than the one that saved it: run `run_pipeline.bat` once.

## What the dashboard shows
| Page | What you get |
|---|---|
| **Ward board** | A simulated ward of real test-set patients, sorted by chance of going home in 24 h (GREEN / YELLOW / RED), chance in 48 h, expected days left, a "follow-up before release" flag from the readmission model, a doctor-confirmed tick box, expected free beds tomorrow with an 80% range, **safe planned admissions** at your chosen confidence, and an "ER shortfall risk" number. A "reveal what actually happened" box keeps the demo honest. |
| **New admission** | Enter a patient's admission details; get the full length-of-stay distribution, median, P10-P90, and chance of a 7+ day stay. |
| **Evaluation** | ROC and precision-recall curves, a live **threshold slider** with confusion matrix / precision / recall / F1, calibration, leakage test, time-forward check, per-day and subgroup AUC, feature importance, ward backtest, length-of-stay and readmission metrics. |
| **BI insights** | Where the bed-days go: by ward, age, admission type, diagnosis, discharge destination. Long stays are ~21% of admissions but ~44% of bed-days. |
| **Data & limits** | Dataset facts, what is simulated, what you must say out loud. |

English / Hinglish toggle in the sidebar.

## Folder map
```
app.py                 the dashboard            run_pipeline.py   retrain + evaluate
bedrelease/            data.py model.py planner.py readmit.py pipeline.py config.py
data/raw/              diabetic_data.csv (+ IDs_mapping.csv)       -> see docs/DATA_SOURCES.md
outputs/               metrics.json, model.joblib, scored test encounters, BI tables ...
tests/                 28 tests (planner maths, leakage/split rules, metrics recomputed from raw rows)
scripts/               e2e_and_screenshots.py (browser test), make_metrics_summary.py
docs/                  PROBLEM_STATEMENT, TUTORIAL_EN, TUTORIAL_HI, METRICS_SUMMARY, DATA_SOURCES, DATA_DICTIONARY, RESEARCH, screenshots/, test_results/
```

## How it works (one paragraph)
A gradient-boosted classifier predicts the full length-of-stay distribution (1-14 days) from information known at admission. For a patient who has already stayed *d* days, P(go home in the next 24 h) = P(LOS = d+1) / P(LOS > d). Adding those probabilities over a ward gives the exact distribution of tomorrow's discharges (Poisson-binomial); combined with Poisson ER arrivals it gives the largest number of planned admissions that is safe at the chosen confidence. Evaluation is on patients never seen in training, with leakage and time checks. Details: `docs/TUTORIAL_EN.md`.

## Tested on
Python 3.13 on Linux: pipeline, 28 pytest tests, and 16 browser end-to-end checks (see `docs/test_results/`). **Not run on real Windows**; the `.bat` files follow standard `venv` steps.

Team: Yash Raj, Ram Susil, Kartik Jansari, Shraddha Chippa. Faculty guide: Dr. Pradip Salve.

## 90%+ mode (synthetic in-stay data)
Real UCI data has no in-stay signals, so accuracy over all patient-days is ~77% (ROC-AUC 0.643). The **90%+ mode** page adds
SIMULATED daily signals (doctor's discharge plan, vitals, mobility, CRP trend, IV drug; `bedrelease/synthetic.py`) to real admissions
and trains on them: **accuracy 95.8%, precision 90.9%, recall 90.5%, ROC-AUC 0.990** on a patient-wise synthetic test set.
These numbers describe the simulator, not a real hospital; they show which columns a hospital must record. Pre-trained models
(`outputs/model.joblib`, `outputs/synthetic_model.joblib`) are included, so the dashboard runs without training.
