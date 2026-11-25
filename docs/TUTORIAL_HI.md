# BedRelease Planner - tutorial (Hinglish, Roman script)

## 0. Aakhir me kya milega
Apne computer par ek dashboard jo batata hai: **agle 24 ghante me kaun ghar ja sakta hai, kal kitne bed khali honge, aur kitne planned admission safely le sakte hain.** Saath me ek Evaluation page jo viva me defend kar sako.

## 1. Setup (lagbhag 5 minute)
**Windows:** python.org se Python 3.11-3.13 install karo (**Add to PATH** tick zaroor) -> project ZIP unzip karo -> `setup.bat` double-click karo -> `run_dashboard.bat` double-click karo. Browser me http://localhost:8501 khulega.
**Linux/macOS:** `bash setup_and_run.sh`.
Band karne ke liye black window me Ctrl+C. Dobara chalane ke liye `run_dashboard.bat`.

Dikkat aaye to:
- "Python not found": Python dobara install karo, **Add to PATH** tick karo, phir `setup.bat`.
- Port busy: `python -m streamlit run app.py --server.port 8502`.
- "model.joblib could not be loaded": ek baar `run_pipeline.bat` chalao (2-3 minute me retrain).
- Blank page: browser refresh karo, start ke baad 10 second ruko.

## 2. Do minute ka demo script (kya click karna, kya bolna)
1. **Ward board.** Bolo: "30 bed, Internal Medicine. Ye asli patients hain ek hospital dataset ke. Inhe 24 ghante me ghar jaane ke chance se sort kiya hai. GREEN patients 35% baar sach me ek din me gaye, RED sirf 14%, aur ye data model ne kabhi dekha nahi tha."
2. Upar ke chaar numbers padho: expected discharges, expected free beds, **safe planned admissions**, ER shortfall risk. **Empty beds now** ko 3 se 0 aur phir 8 karo: safe number turant badalta hai.
3. **Expected ER arrivals** 8 karo: safe planned admissions 0 ho jata hai. Bolo: "Planner hamesha pehle emergency patient ko bachata hai."
4. Ek patient ke liye **Doctor: ready today** tick karo: uski probability 100% ho jati hai, expected discharges badh jate hain.
5. **Reveal what actually happened** kholo: asli data se honesty check. Bolo ki ek din ka number noisy hota hai; average backtest me dikhta hai.
6. **New admission:** diagnosis aur admission type badlo; stay distribution aur 7+ din ka chance dikhao.
7. **Evaluation:** threshold slider hilao. Precision badhta hai to recall ghatta hai. Phir Leakage tab: "lab aur medication count jodne par AUC 0.75 ho jata hai, par ye cheating hai kyunki wo poore stay ke totals hain; humne exclude kiye."
8. **BI insights:** "Long stay sirf ~21% admission hain par ~44% bed-days kha jate hain."

## 3. Numbers kaise padhein
- **P(go home <24h):** model ki probability ki patient agle 24 ghante me chala jayega. Calibrated hai: jab 30% bolta hai, lagbhag 30% sach me jate hain.
- **Signal:** GREEN >= 0.29, YELLOW 0.21-0.29, RED < 0.21 (validation set se).
- **Expected discharges / free beds:** probabilities ka sum, 80% range exact Poisson-binomial distribution se.
- **Safe planned admissions (confidence c):** sabse bada a jisme P(khali bed + discharge - ER arrivals >= a) >= c.
- **ER shortfall risk (0 planned):** planned admission zero hone par bhi emergency patient ko bed na milne ka chance.
- **Follow-up before release:** 30-din readmission risk ka top-20% flag (risk rank hai, percentage nahi).

## 4. Metrics aasan bhasha me
- **ROC-AUC** (0.5 = sikka uchhalna, 1 = perfect): jo patient gaya uska score ruke hue patient se zyada hone ka chance. Model 0.643, baseline 0.595.
- **Precision:** flag kiye gaye me se kitne sach me 24h me gaye. **Recall:** jo gaye unme se kitne flag hue. Slider se ek badhta hai to doosra ghatta hai.
- **PR-AUC:** precision-recall curve ke neeche ka area; base rate 22.8%, model 0.346.
- **Calibration / ECE:** probabilities asliyat se match karti hain ya nahi. ECE 0.003 = bahut achhi.
- **MAE:** average galti (din me). **Coverage:** asli stays ka kitna hissa P10-P90 band me aaya.
- **Leakage:** aisi information use karna jo outcome ke baad hi pata chalti hai. **Time-forward check:** purane encounters par train, naye par test.

## 5. Model kaise kaam karta hai
1. Data saaf karo, admission-time features aur usi patient ke pichhle stays ki history banao.
2. **Patient ke hisab se** split karo (70/15/15) taaki test set me sirf anjaane patients hon.
3. Gradient-boosted classifier P(LOS = 1, 2, ..., 14 din) predict karta hai. Settings validation part par chuni gayi.
4. Din d par patient ke liye: P(agle 24h me jayega) = P(LOS = d+1) / P(LOS > d).
5. Ward ki probabilities jodo -> discharge ka distribution; Poisson ER arrivals ke saath mila ke -> safe planned admissions.
6. Real test encounters se bane 2,000 simulated ward-days par backtest.

## 6. Viva me ye sawal aayenge
- *Ye real data hai?* Haan, 101,766 asli encounters (UCI Diabetes-130, US hospitals, 1999-2008, diabetic patients). Ward size, khali bed, ER arrivals aur daily census **simulated** hain kyunki file me nahi hain.
- *Accuracy sirf 0.64 kyun?* Admission ki information se stay ka sirf ek hissa samjha ja sakta hai; dataset me stay ke dauran clinical update nahi hain. Value calibrated probability aur ranking me hai, aur baseline bhi dikhaya hai.
- *Answer leak to nahi hua?* Test kiya: stay-total counters leak karte hain (AUC 0.746), isliye exclude; split patient-wise; time-forward check 0.631 vs 0.592.
- *Ward-total accuracy zyada kyun nahi badhi?* Ek ward me individual galtiyan average ho jati hain; honest result (MAE 1.80 vs 1.83 vs 1.85). Asli jeet ye hai ki kaun jaane wala hai uski ranking behtar hai (precision@K 0.343 vs 0.292 vs 0.231 random).
- *Hospital isse use kar sakta hai?* Aise nahi: US diabetic data, real bed data nahi. Local data par retrain karo; discharge ka faisla hamesha doctor ka.
- *Hazard model kyun?* Ek hi model har din ke liye kaam karta hai: wahi probabilities "naya patient" aur "din 6 ka patient" dono ka jawab deti hain.

## 7. Retrain, test, extend
`run_pipeline.bat` retrain karke `outputs/` dobara likhta hai. `run_tests.bat` 28 tests chalata hai. Apne hospital ka data use karna ho to columns `docs/DATA_DICTIONARY.md` ke hisab se map karo (chahiye: patient id, encounter order, admission type/source, age, diagnosis group, length of stay) aur `bedrelease/config.py` me `RAW_CSV` badlo.

## 8. 90%+ mode (synthetic in-stay data)
Sidebar me **90%+ mode** page kholo. Isme pre-trained model hai (training ki zarurat nahi), jo real admissions + SIMULATED daily
signals par bana hai: doctor ka discharge plan, vitals stable, mobility, CRP trend, IV drug.
Test results (patient-wise split): accuracy 95.8%, precision 90.9%, recall 90.5%, ROC-AUC 0.990.
Checkbox aur slider badlo, P(24h me ghar) badalta dikhega.
**Viva me bolo:** "Real data par in-stay signals nahi hain, isliye ~77% accuracy. Realistic (synthetic) in-stay signals ke saath 95.8%.
Isse hospital ko pata chalta hai kaunse columns record karne hain." 95.8% ko kabhi real-hospital result mat bolna.
