"""Does a different (small-data-friendly) model beat the main model? Direct binary day-level classifiers, same patient split."""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from bedrelease import data, model as M, config as C
c = data.load_clean(); df = c.df if hasattr(c, "df") else c[0]
df = df.assign(part=M.patient_split(df).values)
cat, num = M.feature_lists("with_diagnosis")
def rows(d):
    r = d.loc[d.index.repeat(d["los"].to_numpy())].copy()
    r["day_done"] = np.concatenate([np.arange(n) for n in d["los"]]); r["label"] = (r["day_done"] + 1 == r["los"]).astype(int)
    return r
tr, va, te = (rows(df[df.part == k]) for k in ("train", "valid", "test"))
enc = M.Encoder(cat, num).fit(tr)
X = lambda r: np.hstack([enc.transform(r), r[["day_done"]].to_numpy(float)])
Xtr, Xva, Xte = (np.nan_to_num(X(r), nan=-1.0) for r in (tr, va, te))
def conf(y, p, hi, lo):
    m = (p < lo) | (p >= hi); yh = (p >= hi).astype(int); return float(m.mean()), float((yh[m] == y[m]).mean()) if m.any() else None
res = {}
models = {"logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)),
          "random_forest": RandomForestClassifier(200, min_samples_leaf=50, n_jobs=-1, random_state=1),
          "extra_trees": ExtraTreesClassifier(200, min_samples_leaf=50, n_jobs=-1, random_state=1),
          "hist_gradient_boosting_direct": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=1)}
for n, m in models.items():
    m.fit(Xtr, tr.label); pv = m.predict_proba(Xva)[:, 1]; pt = m.predict_proba(Xte)[:, 1]
    yv, yt = va.label.to_numpy(), te.label.to_numpy()
    top = np.argsort(-pt)[: int(0.01 * len(pt))]
    q = np.quantile(pv, 0.9985); lo = np.quantile(pv, 0.15)
    cov, acc = conf(yt, pt, q, lo)
    res[n] = {"roc_auc": float(roc_auc_score(yt, pt)), "precision_top1pct": float(yt[top].mean()), "confident_coverage": cov, "confident_accuracy": acc}
    print(n, res[n], flush=True)
main = json.load(open("outputs/metrics.json"))["discharge_24h"]
res["main_los_hazard_model"] = {"roc_auc": main["p_model"]["roc_auc"], "confident_coverage": main["confident_calls"]["coverage"], "confident_accuracy": main["confident_calls"]["accuracy_on_confident_calls"]}
json.dump(res, open("outputs/model_comparison.json", "w"), indent=1)
