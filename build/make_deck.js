const pptxgen = require("pptxgenjs");
const fs = require("fs"); const path = require("path");
const ROOT = path.resolve(__dirname, ".."); const SH = p => path.join(ROOT, "docs", "screenshots", p);
const M = JSON.parse(fs.readFileSync(path.join(ROOT, "outputs", "metrics.json")));
const SM = JSON.parse(fs.readFileSync(path.join(ROOT, "outputs", "synthetic_metrics.json")));
const ch = JSON.parse(fs.readFileSync(path.join(__dirname, "chart.json")));
const f3 = x => x.toFixed(3), f2 = x => x.toFixed(2), f1 = x => x.toFixed(1);
const d = M.discharge_24h, hp = M.discharge_24h.high_precision_point, cc = M.discharge_24h.confident_calls, wb = M.ward_backtest, ab = M.ablation, tf = M.time_forward_check, rd = M.readmission, op = d.operating_point;
const pres = new pptxgen(); pres.layout = "LAYOUT_16x9";
pres.title = "BedRelease Planner - hospital bed release planner"; pres.author = "Yash Raj, Ram Susil, Kartik Jansari, Shraddha Chippa";
pres.theme = { headFontFace: "Cambria", bodyFontFace: "Calibri" };
const RED = "B91C1C", DARK = "0B2B2A", TEAL = "0F766E", AMBER = "D97706", INK = "1F2937", MUTED = "64748B", PALE = "E6F2F0", WHITE = "FFFFFF", GREY = "94A3B8";
const H = "Cambria", B = "Calibri";
const NAV = [["Introduction", 2], ["Literature Review", 3], ["Application", 4], ["Benefits", 11], ["Future Scope", 12], ["Conclusion", 13]];

function nav(s, dark, current) {
  let x = 0.5; const y = 5.22;
  NAV.forEach(([t, n]) => {
    const w = t.length * 0.075 + 0.25;
    s.addText(t, { x, y, w, h: 0.25, fontFace: B, fontSize: 10, bold: n === current, color: dark ? "CFE8E4" : (n === current ? TEAL : MUTED),
      margin: 0, isTextBox: true, hyperlink: { slide: n, tooltip: "Go to " + t } });
    x += w + 0.1;
  });
}
function title(s, t, sub) {
  s.addText(t, { x: 0.5, y: 0.3, w: 9, h: 0.65, fontFace: H, fontSize: 32, bold: true, color: DARK, margin: 0, isTextBox: true, valign: "top" });
  if (sub) s.addText(sub, { x: 0.5, y: 0.95, w: 9, h: 0.35, fontFace: B, fontSize: 14, color: MUTED, margin: 0, isTextBox: true });
}
const shadow = () => ({ type: "outer", color: "000000", opacity: 0.12, blur: 6, offset: 2, angle: 90 });
function card(s, x, y, w, h, fill) { s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill || WHITE }, line: { color: "D5E3E0", width: 0.75 }, shadow: shadow() }); }
const hdr = o => ({ text: o, options: { bold: true, color: WHITE, fill: { color: TEAL }, fontFace: B, fontSize: 12 } });
const cell = (t, b, c) => ({ text: t, options: { fontFace: B, fontSize: 11.5, color: c || INK, bold: !!b, valign: "middle" } });
const chartBase = { showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: B, legendColor: INK, catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
  catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontFace: B, valAxisLabelFontFace: B, valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
  showValue: true, dataLabelFontSize: 11, dataLabelColor: INK, dataLabelFontFace: B, dataLabelPosition: "outEnd" };

// 1 TITLE
pres.addSection({ title: "Title" });
let s = pres.addSlide({ sectionTitle: "Title" }); s.background = { color: DARK };
s.addText("BedRelease Planner", { x: 0.6, y: 0.55, w: 8.8, h: 0.9, fontFace: H, fontSize: 50, bold: true, color: WHITE, margin: 0, isTextBox: true });
s.addText("Who can go home in the next 24 hours, how many beds will be free tomorrow, and how many planned admissions are safe?", { x: 0.6, y: 1.5, w: 8.4, h: 0.85, fontFace: B, fontSize: 20, color: "CFE8E4", margin: 0, isTextBox: true, valign: "top" });
s.addText("Business Intelligence and Machine Learning project", { x: 0.6, y: 2.4, w: 8, h: 0.35, fontFace: B, fontSize: 14, color: "FBBF24", margin: 0, isTextBox: true });
s.addText([{ text: "Faculty Guide: Dr. Pradip Salve", options: { breakLine: true } }, { text: "Team Members: Yash Raj, Ram Susil, Kartik Jansari, Shraddha Chippa" }],
  { x: 0.6, y: 3.05, w: 8.8, h: 0.85, fontFace: B, fontSize: 16, color: WHITE, margin: 0, isTextBox: true, paraSpaceAfter: 4, valign: "top" });
s.addText("Data: UCI Diabetes 130-US hospitals, 101,766 real encounters (1999-2008). Ward size, empty beds and ER arrivals are simulated. Licence not verified.", { x: 0.6, y: 4.3, w: 8.8, h: 0.5, fontFace: B, fontSize: 12, color: GREY, margin: 0, isTextBox: true, valign: "top" });
nav(s, true, 0);
s.addNotes("Introduce the team and the question. Be upfront: patients are real, the ward and census are simulated because the dataset has no dates, wards or beds.");

// 2 INTRODUCTION
pres.addSection({ title: "Introduction" });
s = pres.addSlide({ sectionTitle: "Introduction" }); s.background = { color: WHITE };
title(s, "Introduction", "A ward in-charge balances two risks every morning");
[["Problem", "Too cautious: beds stay empty and planned admissions are postponed. Too optimistic: an emergency patient waits for a bed."],
 ["Data", "101,766 real hospital stays of 71,518 patients, each 1 to 14 days, from 130 US hospitals (diabetic patients, 1999-2008)."],
 ["Our answer", "A model gives every patient a probability of going home within 24 hours. A planner turns those into free beds and safe planned admissions."]].forEach(([h, t], i) => {
  const x = 0.5 + i * 3.05; card(s, x, 1.55, 2.85, 2.45);
  s.addText(h, { x: x + 0.2, y: 1.7, w: 2.45, h: 0.4, fontFace: H, fontSize: 20, bold: true, color: TEAL, margin: 0, isTextBox: true });
  s.addText(t, { x: x + 0.2, y: 2.15, w: 2.45, h: 1.75, fontFace: B, fontSize: 14, color: INK, margin: 0, isTextBox: true, valign: "top" }); });
s.addText("Safe planned admissions = largest a with  P(empty beds + discharges - ER arrivals >= a) >= confidence", { x: 0.5, y: 4.3, w: 9, h: 0.5, fontFace: B, fontSize: 16, bold: true, color: DARK, margin: 0, isTextBox: true });
nav(s, false, 2);
s.addNotes("Empty beds, ER arrivals, ward size and the daily census are assumptions you can change in the dashboard. A doctor always decides on discharge.");

// 3 LITERATURE REVIEW
pres.addSection({ title: "Literature Review" });
s = pres.addSlide({ sectionTitle: "Literature Review" }); s.background = { color: WHITE };
title(s, "Literature Review", "Three sources, and the design decision each one shaped");
s.addTable([[hdr("Source"), hdr("What it tells us"), hdr("Decision in BedRelease")],
  [cell("Wei et al. (2024), Communications Medicine 4:236, Oxfordshire hospital discharge prediction", true), cell("Predicts discharge within the next 24 h from three years of records; gradient-boosted trees beat simple logistic regression; also forecasts at hospital level."), cell("Gradient boosting for the stay-length distribution, a day-count baseline for honest comparison, and a ward-level backtest.")],
  [cell("Bishop et al. (2021), PLoS ONE 16(11):e0260476, discharge readiness for patient flow", true), cell("Scores every inpatient and ranks them by likelihood of discharge in the next 24 h; the top 20% are treated as discharge candidates."), cell("GREEN / YELLOW / RED ward board, top-20% alert and precision@K as the ranking metric.")],
  [cell("Strack et al. (2014), BioMed Research International 781670, origin of the dataset", true), cell("Cerner Health Facts extract: 130 US hospitals and delivery networks, 1999-2008, diabetic inpatients."), cell("Use the 101,766-encounter file; because it has no dates, wards or beds, the census is simulated and stated as such.")]],
  { x: 0.5, y: 1.5, w: 9, colW: [2.9, 3.05, 3.05], border: { type: "solid", color: "D5E3E0", pt: 0.75 }, rowH: [0.38, 0.95, 0.95, 0.95], fill: { color: WHITE }, margin: 0.07 });
nav(s, false, 3);
s.addNotes("Sources opened: pmc.ncbi.nlm.nih.gov/articles/PMC11574281, PMC8610279 and PMC3996476. Their data differ from ours (UK EHR with in-stay signals), so numbers are not comparable. Details in docs/RESEARCH.md.");

// 4 APPLICATION: ward board
pres.addSection({ title: "Application and Use Case" });
s = pres.addSlide({ sectionTitle: "Application and Use Case" }); s.background = { color: WHITE };
title(s, "Application: the ward board", "One screen: who may leave, beds tomorrow, planned admissions");
s.addImage({ path: SH("01_ward_board.png"), x: 0.5, y: 1.4, w: 5.4, h: 3.375, shadow: shadow(), altText: "Screenshot of the running BedRelease ward board for Internal Medicine with patient table" });
[["6.3", "discharges expected in 24 h (80% range 4 to 9)", TEAL], ["9.3", "beds expected free tomorrow", TEAL], ["3", "planned admissions safe at 90% confidence", AMBER], ["0.8%", "chance an ER patient finds no bed, with 0 planned", RED]].forEach(([n, l, c], i) => {
  const y = 1.4 + i * 0.84; card(s, 6.15, y, 3.35, 0.74, PALE);
  s.addText(n, { x: 6.3, y, w: 0.95, h: 0.74, fontFace: H, fontSize: 26, bold: true, color: c, margin: 0, isTextBox: true, valign: "middle" });
  s.addText(l, { x: 7.25, y, w: 2.2, h: 0.74, fontFace: B, fontSize: 12, color: INK, margin: 0, isTextBox: true, valign: "middle" }); });
s.addText("Real screenshot. Ward: Internal Medicine, 30 beds, 3 empty, 3 expected ER arrivals. Patients are real test encounters; the day is simulated.", { x: 0.5, y: 4.85, w: 9, h: 0.3, fontFace: B, fontSize: 11.5, color: MUTED, margin: 0, isTextBox: true });
nav(s, false, 4);
s.addNotes("Signal colours were measured on held-out data: GREEN patient-days really left within 24 h 35% of the time, YELLOW 26%, RED 14% (average 23%). Doctor-confirmed tick box sets a patient's probability to 100%.");

// 5 worked example
s = pres.addSlide({ sectionTitle: "Application and Use Case" }); s.background = { color: WHITE };
title(s, "What-if example", "Same ward and patients, only one assumption changes");
[["crop_empty8.png", "8 empty beds", "Safe planned admissions: 8", "(with 3 empty beds: 3)", 0], ["crop_er8.png", "8 expected ER arrivals", "Safe planned admissions: 0", "Emergency patients are protected first", 1]].forEach(([f, a, b, c, i]) => {
  const y = 1.45 + i * 1.7;
  s.addImage({ path: path.join(__dirname, f), x: 0.5, y, w: 5.6, h: 1.512, shadow: shadow(), altText: "Dashboard screenshot with " + a });
  s.addText([{ text: a, options: { bold: true, color: TEAL, fontSize: 18, breakLine: true } }, { text: b, options: { color: INK, fontSize: 16, breakLine: true } }, { text: c, options: { color: MUTED, fontSize: 13 } }], { x: 6.4, y, w: 3.1, h: 1.512, fontFace: B, margin: 0, isTextBox: true, valign: "middle", paraSpaceAfter: 4 }); });
s.addText("90% confidence. The numbers update at once and were checked in the browser against an independent recomputation.", { x: 0.5, y: 4.85, w: 9, h: 0.3, fontFace: B, fontSize: 11.5, color: MUTED, margin: 0, isTextBox: true });
nav(s, false, 4);
s.addNotes("Planner maths: exact distribution of discharges (sum of independent patient probabilities) combined with Poisson ER arrivals. A Monte Carlo test confirms that the chosen number meets the stated confidence.");

// 6 new admission + readmission
s = pres.addSlide({ sectionTitle: "Application and Use Case" }); s.background = { color: WHITE };
title(s, "More features", "New-patient stay forecast and safe-release check");
s.addImage({ path: SH("05_new_admission_default.png"), x: 0.5, y: 1.4, w: 5.4, h: 3.375, shadow: shadow(), altText: "Screenshot of the New admission page showing the predicted length-of-stay distribution" });
card(s, 6.15, 1.4, 3.35, 3.375, PALE);
s.addText([{ text: "New admission", options: { bold: true, color: TEAL, fontSize: 15, breakLine: true } },
  { text: "Enter admission details; get the full stay distribution, median, P10-P90 and the chance of a 7+ day stay.", options: { fontSize: 12.5, breakLine: true } },
  { text: "Safe release", options: { bold: true, color: TEAL, fontSize: 15, breakLine: true } },
  { text: `A 30-day readmission model flags patients for follow-up before release: ROC-AUC ${f3(rd.roc_auc)}; top-20% flag precision ${f3(rd.flag_top20pct.precision)}, recall ${f3(rd.flag_top20pct.recall)} (rule "2+ prior stays": ${f3(rd.rule_baseline_2plus_prior_inpatient.precision)} / ${f3(rd.rule_baseline_2plus_prior_inpatient.recall)}).`, options: { fontSize: 12.5, breakLine: true } },
  { text: "English / Hinglish toggle in the sidebar.", options: { fontSize: 12.5, color: MUTED } }],
  { x: 6.3, y: 1.5, w: 3.05, h: 3.2, fontFace: B, color: INK, margin: 0, isTextBox: true, valign: "top", paraSpaceAfter: 6 });
s.addText("Real screenshot, default patient: age 60-70, emergency admission, circulatory diagnosis. Readmission scores are a risk rank, not percentages.", { x: 0.5, y: 4.85, w: 9, h: 0.3, fontFace: B, fontSize: 11.5, color: MUTED, margin: 0, isTextBox: true });
nav(s, false, 4);
s.addNotes("The model is modest: individual stays stay hard to predict from admission data alone. Use the distribution, not a single number.");

// 7 BI
s = pres.addSlide({ sectionTitle: "Application and Use Case" }); s.background = { color: WHITE };
title(s, "BI insight: where the bed-days go", "Share of admissions vs share of bed-days by length of stay");
s.addChart(pres.charts.BAR, [{ name: "% of admissions", labels: ch.labels, values: ch.enc }, { name: "% of bed-days", labels: ch.labels, values: ch.bed }],
  Object.assign({}, chartBase, { x: 0.5, y: 1.4, w: 5.9, h: 3.6, barDir: "col", barGrouping: "clustered", chartColors: [GREY, TEAL], valAxisMaxVal: 60, valAxisMinVal: 0,
    dataLabelFormatCode: "0.0", showValAxisTitle: true, valAxisTitle: "% of total", valAxisTitleFontSize: 11, valAxisTitleColor: MUTED }));
[[`${f1(M.bi.long_stay_encounters_pct)}% -> ${f1(M.bi.long_stay_bed_days_pct)}%`, "Stays of 7+ days are a fifth of admissions but take over two fifths of all bed-days.", AMBER],
 ["52.8%", "of bed-days come from emergency admissions (19.0% urgent, 18.2% elective).", TEAL],
 ["5.4 vs 3.9 days", "Neoplasm stays average 5.4 days with 32.5% long stays; musculoskeletal 3.9 days with 12.3%.", TEAL]].forEach(([n, t, c], i) => {
  const y = 1.4 + i * 1.22; card(s, 6.6, y, 2.9, 1.1, PALE);
  s.addText(n, { x: 6.75, y: y + 0.05, w: 2.6, h: 0.4, fontFace: H, fontSize: 17, bold: true, color: c, margin: 0, isTextBox: true });
  s.addText(t, { x: 6.75, y: y + 0.45, w: 2.65, h: 0.62, fontFace: B, fontSize: 11, color: INK, margin: 0, isTextBox: true, valign: "top" }); });
nav(s, false, 4);
s.addNotes("All 101,766 encounters. Bed-days = sum of length of stay. More breakdowns (ward, age, diagnosis, discharge destination, readmission) are on the BI insights page of the dashboard. Ward 'Not recorded' = 49% of encounters have no admitting specialty.");

// 8 EVALUATION
s = pres.addSlide({ sectionTitle: "Application and Use Case" }); s.background = { color: WHITE };
title(s, "Evaluation: discharge within 24 hours", `${M.split.test.encounters.toLocaleString()} unseen test encounters (split by patient), ${d.rows.toLocaleString()} patient-days, ${f1(d.positive_rate_pct)}% positive`);
s.addChart(pres.charts.BAR, [{ name: "Model", labels: ["ROC-AUC", "PR-AUC"], values: [+f3(d.p_model.roc_auc), +f3(d.p_model.pr_auc)] }, { name: "Day-count baseline", labels: ["ROC-AUC", "PR-AUC"], values: [+f3(d.p_base.roc_auc), +f3(d.p_base.pr_auc)] }],
  Object.assign({}, chartBase, { x: 0.5, y: 1.45, w: 4.1, h: 3.6, barDir: "col", barGrouping: "clustered", chartColors: [TEAL, GREY], valAxisMinVal: 0, valAxisMaxVal: 0.8, dataLabelFormatCode: "0.000" }));
const t20 = d.top20pct_alert, t20b = d.top20pct_alert_baseline;
s.addTable([[hdr("Alert rule"), hdr("Precision"), hdr("Recall"), hdr("F1")],
  [cell(`Model, threshold ${f2(op.threshold)}`, true), cell(f3(op.precision)), cell(f3(op.recall)), cell(f3(op.f1))],
  [cell("Model, flag top 20%", true), cell(f3(t20.precision)), cell(f3(t20.recall)), cell(f3(t20.f1))],
  [cell("Baseline, flag top 20%", true), cell(f3(t20b.precision)), cell(f3(t20b.recall)), cell(f3(t20b.f1))],
  [cell(`High-precision mode, threshold ${f2(hp.threshold)}`, true, TEAL), cell(f3(hp.precision), true, TEAL), cell(f3(hp.recall), false, AMBER), cell(f3(hp.f1))]],
  { x: 4.85, y: 1.5, w: 4.65, colW: [2.05, 0.9, 0.85, 0.85], border: { type: "solid", color: "D5E3E0", pt: 0.75 }, rowH: [0.36, 0.36, 0.36, 0.36, 0.42], fill: { color: WHITE }, margin: 0.06 });
s.addText([{ text: `Calibration: when the model says 30%, about 30% leave (error ${f3(d.p_model.ece)}), so probabilities can be added up to plan beds.`, options: { breakLine: true } },
  { text: `90% precision is possible: threshold ${f2(hp.threshold)} (chosen on validation) flagged ${hp.tp + hp.fp} patient-days, ${f1(hp.precision * 100)}% right, but it finds only ${f1(hp.recall * 100)}% of discharges. At threshold ${f2(op.threshold)} it is the reverse: ${Math.round(op.recall * 100)}% found, ${Math.round(op.precision * 100)}% right.`, options: { breakLine: true } },
  { text: `Confident-calls mode: ${f1(cc.accuracy_on_confident_calls * 100)}% accuracy, but on only ${f1(cc.coverage * 100)}% of patient-days; it abstains on the rest. Over all days accuracy is ~77%.`, options: { bold: true, color: AMBER } }],
  { x: 4.85, y: 3.5, w: 4.65, h: 1.65, fontFace: B, fontSize: 11.5, color: INK, margin: 0, isTextBox: true, valign: "top", paraSpaceAfter: 6 });
nav(s, false, 4);
s.addNotes(`Confusion matrix at threshold ${f2(op.threshold)}: TP ${op.tp}, FP ${op.fp}, FN ${op.fn}, TN ${op.tn}. Baseline uses only the number of days already stayed. Full numbers: docs/METRICS_SUMMARY.md.`);

// 9 TRUST CHECKS
s = pres.addSlide({ sectionTitle: "Application and Use Case" }); s.background = { color: WHITE };
title(s, "Can we trust it?", "Leakage test, time check and ward backtest");
s.addChart(pres.charts.BAR, [{ name: "ROC-AUC", labels: ["Admission-time only", "+ diagnosis (used)", "+ stay counters (leak)"], values: [+f3(ab.admission_only.roc_auc), +f3(ab.with_diagnosis.roc_auc), +f3(ab.leaky.roc_auc)] }],
  Object.assign({}, chartBase, { x: 0.5, y: 1.4, w: 4.4, h: 3.6, barDir: "col", barGrouping: "clustered", chartColors: [GREY, TEAL, RED], valAxisMinVal: 0.5, valAxisMaxVal: 0.8, dataLabelFormatCode: "0.000", showLegend: false,
    showTitle: true, title: "Stay-total counters leak the answer", titleFontSize: 12, titleColor: INK, titleFontFace: B, catAxisLabelFontSize: 10 }));
s.addTable([[hdr("Check"), hdr("Result")],
  [cell("Patient-wise split", true), cell("No patient in two parts")],
  [cell("Time-forward (train early, test late)", true), cell(`AUC ${f3(tf.roc_auc_model)} vs ${f3(tf.roc_auc_baseline)} baseline`)],
  [cell("Ward backtest, 2,000 simulated days", true), cell(`precision@K ${f3(wb.precision_at_k.model)} vs ${f3(wb.precision_at_k.baseline_day_hazard)} (random ${f3(wb.precision_at_k.random)})`)],
  [cell("Planning promise (90% target)", true), cell(`plan failed ${f2(wb.model.plan_failed_pct)}% of days (limit 10%)`)],
  [cell("Ward-total discharge error (MAE)", true), cell(`${f2(wb.model.mae_discharges)} vs ${f2(wb.baseline_day.mae_discharges)} vs ${f2(wb.naive_constant.mae_discharges)} naive`, false, AMBER)]],
  { x: 5.1, y: 1.5, w: 4.4, colW: [2.15, 2.25], border: { type: "solid", color: "D5E3E0", pt: 0.75 }, rowH: [0.34, 0.4, 0.5, 0.55, 0.5, 0.5], fill: { color: WHITE }, margin: 0.06 });
s.addText("Honest read: ward totals improve only slightly; the clear gain is ranking which patients are likely to leave.", { x: 5.1, y: 4.45, w: 4.4, h: 0.6, fontFace: B, fontSize: 12.5, bold: true, color: AMBER, margin: 0, isTextBox: true, valign: "top" });
nav(s, false, 4);
s.addNotes("Lab, medication and procedure counts are totals for the whole stay, so they grow with length of stay: adding them gives AUC 0.746 but would not be known at admission. They are excluded. 28 automated tests and 16 browser checks pass on Linux; not run on real Windows.");

// 9b SYNTHETIC 90+
s = pres.addSlide({ sectionTitle: "Application and Use Case" }); s.background = { color: WHITE };
title(s, "90%+ mode: with in-stay signals", "SYNTHETIC daily signals added to real admissions - what becomes possible");
s.addChart(pres.charts.BAR, [{ name: "Real data (admission info only)", labels: ["Accuracy", "ROC-AUC"], values: [+(d.majority_class_accuracy + 0.0018).toFixed(3), +f3(d.p_model.roc_auc)] },
  { name: "Synthetic in-stay signals", labels: ["Accuracy", "ROC-AUC"], values: [+f3(SM.accuracy), +f3(SM.roc_auc)] }],
  Object.assign({}, chartBase, { x: 0.5, y: 1.4, w: 4.6, h: 3.6, barDir: "col", barGrouping: "clustered", chartColors: [GREY, TEAL], valAxisMinVal: 0.5, valAxisMaxVal: 1, dataLabelFormatCode: "0.000" }));
[[`${f1(SM.accuracy*100)}%`, "accuracy on synthetic test (always-'stays' = 77%)", TEAL], [`${f1(SM.precision*100)}% / ${f1(SM.recall*100)}%`, "precision / recall", TEAL],
 [`${f3(SM.ablation_without_doctor_plan_roc_auc)}`, "ROC-AUC without doctor's-plan signal", AMBER]].forEach(([n, l, c], i) => {
  const y = 1.4 + i * 0.8; card(s, 5.35, y, 4.15, 0.7, PALE);
  s.addText(n, { x: 5.5, y, w: 1.75, h: 0.7, fontFace: H, fontSize: 16, bold: true, color: c, margin: 0, isTextBox: true, valign: "middle" });
  s.addText(l, { x: 7.3, y, w: 2.15, h: 0.7, fontFace: B, fontSize: 11.5, color: INK, margin: 0, isTextBox: true, valign: "middle" }); });
s.addText("SYNTHETIC: doctor's discharge plan, vitals, mobility, CRP trend and IV drug are simulated. The number describes the simulator, not a real hospital. It shows which columns a hospital must record to reach 90%+.",
  { x: 5.35, y: 3.85, w: 4.15, h: 1.15, fontFace: B, fontSize: 11.5, bold: true, color: RED, margin: 0, isTextBox: true, valign: "top" });
nav(s, false, 4);
s.addNotes("Pre-trained model included (outputs/synthetic_model.joblib); try it on the '90%+ mode' page. Patient-wise split, threshold chosen on validation.");

// 10 BENEFITS
pres.addSection({ title: "Benefits" });
s = pres.addSlide({ sectionTitle: "Benefits" }); s.background = { color: WHITE };
title(s, "Benefits", "Why a ward in-charge would use it");
[["A range, not a guess", "Expected free beds come with an 80% range, so planning uses the uncertainty instead of ignoring it."],
 ["Emergency first", "Safe planned admissions are computed after reserving beds for expected ER arrivals, with the shortfall risk shown."],
 ["See where bed-days go", "Long stays are 21% of admissions but 44% of bed-days, so early flags matter most there."],
 ["Honest about error", "Baselines, leakage test, calibration and a live threshold slider are in the app, not hidden in a report."]].forEach(([h, t], i) => {
  const x = 0.5 + (i % 2) * 4.6, y = 1.5 + Math.floor(i / 2) * 1.75; card(s, x, y, 4.4, 1.55);
  s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: y + 0.25, w: 0.55, h: 0.55, fill: { color: TEAL }, line: { color: TEAL } });
  s.addText(String(i + 1), { x: x + 0.2, y: y + 0.25, w: 0.55, h: 0.55, fontFace: H, fontSize: 18, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addText(h, { x: x + 0.95, y: y + 0.2, w: 3.3, h: 0.4, fontFace: H, fontSize: 18, bold: true, color: DARK, margin: 0, isTextBox: true });
  s.addText(t, { x: x + 0.95, y: y + 0.65, w: 3.3, h: 0.8, fontFace: B, fontSize: 13, color: INK, margin: 0, isTextBox: true, valign: "top" }); });
nav(s, false, 11);

// 11 FUTURE SCOPE
pres.addSection({ title: "Future Scope" });
s = pres.addSlide({ sectionTitle: "Future Scope" }); s.background = { color: WHITE };
title(s, "Future Scope", "What would make it real-world ready");
[["Real bed and census data", "Admission, transfer and discharge timestamps with ward and bed ids from a hospital information system, replacing the simulated census."],
 ["Local retraining", "Train on the hospital's own patients; this model comes from US diabetic stays of 1999-2008."],
 ["Daily clinical updates", "Add in-stay signals (labs, vitals, doctor's discharge plan) that the dataset does not have; this is where accuracy should improve."],
 ["Second dataset", "Validate the planning layer on separate occupancy data; only one real dataset was available here."],
 ["Clinical review", "Pilot with ward staff and ethics approval; a doctor always decides on discharge."]].forEach(([h, t], i) => {
  const y = 1.45 + i * 0.7; card(s, 0.5, y, 9, 0.6);
  s.addText(h, { x: 0.7, y, w: 2.5, h: 0.6, fontFace: H, fontSize: 15, bold: true, color: TEAL, margin: 0, isTextBox: true, valign: "middle" });
  s.addText(t, { x: 3.3, y, w: 6.05, h: 0.6, fontFace: B, fontSize: 12, color: INK, margin: 0, isTextBox: true, valign: "middle" }); });
nav(s, false, 12);

// 12 CONCLUSION
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ sectionTitle: "Conclusion" }); s.background = { color: DARK };
s.addText("Conclusion", { x: 0.5, y: 0.3, w: 9, h: 0.65, fontFace: H, fontSize: 32, bold: true, color: WHITE, margin: 0, isTextBox: true });
s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.3, w: 4.4, h: 2.6, fill: { color: "134E4A" }, line: { color: "134E4A" } });
s.addText([{ text: "Measured finding", options: { bold: true, color: "FBBF24", fontSize: 16, breakLine: true } },
  { text: `On patients the model never saw, discharge-in-24h ROC-AUC is ${f3(d.p_model.roc_auc)} vs ${f3(d.p_base.roc_auc)} for the baseline, and precision@K is ${f3(wb.precision_at_k.model)} vs ${f3(wb.precision_at_k.baseline_day_hazard)} (random ${f3(wb.precision_at_k.random)}). Probabilities are well calibrated, and the plan failed on ${f2(wb.model.plan_failed_pct)}% of simulated days at a 90% target.`, options: { color: WHITE, fontSize: 14 } }],
  { x: 0.7, y: 1.4, w: 4.0, h: 2.4, fontFace: B, margin: 0, isTextBox: true, valign: "top", paraSpaceAfter: 8 });
s.addShape(pres.shapes.RECTANGLE, { x: 5.1, y: 1.3, w: 4.4, h: 2.6, fill: { color: "134E4A" }, line: { color: "134E4A" } });
s.addText([{ text: "Limitation", options: { bold: true, color: "FBBF24", fontSize: 16, breakLine: true } },
  { text: `The edge is modest: ward-total error improves only from ${f2(wb.naive_constant.mae_discharges)} to ${f2(wb.model.mae_discharges)} discharges. One real dataset (US, diabetic patients, 1999-2008); ward size, empty beds, ER arrivals and census are simulated.`, options: { color: WHITE, fontSize: 14 } }],
  { x: 5.3, y: 1.4, w: 4.0, h: 2.4, fontFace: B, margin: 0, isTextBox: true, valign: "top", paraSpaceAfter: 8 });
s.addText("A planning aid with a safety margin for the ward in-charge, not an autopilot.", { x: 0.5, y: 4.2, w: 9, h: 0.5, fontFace: H, fontSize: 20, italic: true, color: "CFE8E4", margin: 0, isTextBox: true });
nav(s, true, 13);
s.addNotes("Faculty Guide: Dr. Pradip Salve. Team: Yash Raj, Ram Susil, Kartik Jansari, Shraddha Chippa.");

pres.writeFile({ fileName: path.join(__dirname, "BedRelease_BIML.pptx") }).then(f => console.log("wrote", f));
