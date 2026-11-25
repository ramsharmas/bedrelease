# BedRelease Planner - tutorial (English)

## 0. What you will have at the end
A dashboard on your own computer that answers: *who can go home in 24 hours, how many beds are free tomorrow, how many planned admissions are safe*, plus an evaluation page you can defend in a viva.

## 1. Setup (about 5 minutes)
**Windows:** install Python 3.11-3.13 (python.org, tick "Add to PATH") -> unzip the project -> double-click `setup.bat` -> double-click `run_dashboard.bat`. Browser opens at http://localhost:8501.
**Linux/macOS:** `bash setup_and_run.sh`.
Stop the dashboard with Ctrl+C in the black window. Start again with `run_dashboard.bat`.

Trouble? 
- "Python not found": reinstall Python and tick *Add to PATH*, then run `setup.bat` again.
- Port busy: `python -m streamlit run app.py --server.port 8502`.
- "model.joblib could not be loaded": run `run_pipeline.bat` once (retrains in ~2-3 min).
- Blank page: refresh the browser; wait 10 seconds after start.

## 2. Two-minute demo script (what to click and say)
1. **Ward board.** Say: "30 beds, Internal Medicine. These are real patients from a hospital dataset, sorted by chance of going home in 24 hours. GREEN patients really left within a day 35% of the time, RED only 14%, on data the model never saw."
2. Read the four numbers on top: expected discharges, expected free beds, **safe planned admissions**, ER shortfall risk. Change **Empty beds now** from 3 to 0 and to 8: the safe number moves instantly.
3. Raise **Expected ER arrivals** to 8: safe planned admissions drops to 0. Say: "The planner always protects emergency patients first."
4. Tick **Doctor: ready today** for one patient: their probability becomes 100% and expected discharges rise.
5. Open **Reveal what actually happened**: honest check against the real data. Say that one day is noisy; the backtest shows the average.
6. **New admission:** change diagnosis and admission type; show the stay distribution and the 7+ day chance.
7. **Evaluation:** show the threshold slider. Move it: precision goes up, recall goes down. Then the Leakage tab: "adding lab and medication counts gives AUC 0.75, but that is cheating because they are totals for the whole stay; we excluded them."
8. **BI insights:** "Long stays are about 21% of admissions but 44% of bed-days."

## 3. How to read the numbers
- **P(go home <24h):** model probability that the patient leaves within the next 24 hours. It is calibrated: when it says 30%, about 30% leave.
- **Signal:** GREEN >= 0.29, YELLOW 0.21-0.29, RED < 0.21 (from the validation set).
- **Expected discharges / free beds:** sum of the probabilities, with an 80% range from the exact Poisson-binomial distribution.
- **Safe planned admissions (confidence c):** largest number a such that P(free beds + discharges - ER arrivals >= a) >= c.
- **ER shortfall risk (0 planned):** chance that even with no planned admissions an emergency patient finds no bed.
- **Follow-up before release:** top-20% 30-day-readmission-risk flag (risk rank, not a percentage).

## 4. Metrics in plain words
- **ROC-AUC** (0.5 = coin, 1 = perfect): chance that a patient who left gets a higher score than one who stayed. Model 0.643, baseline 0.595.
- **Precision:** of patients flagged, how many really left within 24 h. **Recall:** of those who left, how many were flagged. The slider trades one for the other.
- **PR-AUC:** area under precision-recall curve; base rate is 22.8%, model 0.346.
- **Calibration / ECE:** do probabilities match reality? ECE 0.003 = very well.
- **MAE:** average miss in days. **Coverage:** share of real stays inside the P10-P90 band.
- **Leakage:** using information that is only known after the outcome. **Time-forward check:** train on earlier encounters, test on later ones.

## 5. How the model works
1. Clean data, build admission-time features and history of the same patient's earlier stays.
2. Split by **patient** (70/15/15) so the test set contains only unseen patients.
3. A gradient-boosted classifier predicts P(LOS = 1, 2, ..., 14 days). Settings chosen on the validation part.
4. For a patient on day d: P(leave next 24 h) = P(LOS = d+1) / P(LOS > d).
5. Add probabilities over the ward -> distribution of discharges; combine with Poisson ER arrivals -> safe planned admissions.
6. Backtest on 2,000 simulated ward-days built from real test encounters.

## 6. Viva questions you should expect
- *Is this real data?* Yes, 101,766 real encounters (UCI Diabetes-130, US hospitals, 1999-2008, diabetic patients). Ward size, empty beds, ER arrivals and the daily census are **simulated**, because the file has none.
- *Why is accuracy only 0.64?* Admission information explains only part of how long a patient stays; the dataset has no in-stay clinical updates. The value is calibrated probabilities and ranking, and we show the baseline.
- *Did you leak the answer?* We tested it: stay-total counters leak (AUC 0.746) and are excluded; split is by patient; time-forward check gives 0.631 vs 0.592.
- *Why does ward-total accuracy barely improve?* Individual errors average out on a ward; honest result (MAE 1.80 vs 1.83 vs 1.85). The clear gain is ranking who is likely to leave (precision@K 0.343 vs 0.292 vs 0.231 random).
- *Can a hospital use it?* Not as is: US diabetic data, no real bed data. Retrain on local data; a doctor always decides.
- *Why a hazard model?* One model covers every day of stay: the same probabilities answer "new patient" and "patient on day 6".

## 7. Retrain, test, extend
`run_pipeline.bat` retrains and rewrites `outputs/`. `run_tests.bat` runs 28 tests. To use your own hospital data, map your columns to those in `docs/DATA_DICTIONARY.md` (needs: patient id, encounter order, admission type/source, age, diagnosis group, length of stay) and change `RAW_CSV` in `bedrelease/config.py`.

## 8. 90%+ mode (synthetic in-stay data)
Open the page **90%+ mode** in the sidebar. It shows a pre-trained model (no training needed) built on real admissions plus
SIMULATED daily signals: doctor's discharge plan, vitals stable, mobility, CRP trend, IV drug.
Test results (patient-wise split): accuracy 95.8%, precision 90.9%, recall 90.5%, ROC-AUC 0.990.
Tick the boxes and move the sliders to see P(go home within 24h) change.
**Say in the viva:** "On real data without in-stay signals we get ~77% accuracy. With realistic (synthetic) in-stay signals we get 95.8%.
That tells the hospital which columns to record." Never present the 95.8% as a real-hospital result.
