#!/usr/bin/env python3
"""
ctrl_analysis.py  --  run AFTER ctrl_features.py (CPU only, ~1-2 minutes).

Part A. Does optimized-RoC fitness carry signal beyond pool confidence?
   Per dataset (queries from all 20 iterations, features z-scored within the
   dataset), correctness of the Top-3 prediction is modelled by
       confound-only : conf_max, pool_agree, knn20_acc
       fitness-only  : RoC_Fitness
       full          : confound + RoC_Fitness
   Cross-validated AUC uses GroupKFold over Iteration (no split leakage).
   Also reports the logistic coefficient of fitness in the full model.

Part B. Plain-kNN / static-ensemble controls vs bADE-RoC-DES (Top-3):
   paired Wilcoxon over the 30 dataset means, Holm-corrected.
   NOTE: bADE numbers come from the saved Top-K run, the controls from a
   deterministic recomputation on the same splits/pools. The optimizer was
   unseeded, so treat differences below ~0.4 pp with caution.

Outputs (publication_results/controls/): fitness_confound_by_dataset.csv,
control_vs_badE.csv, controls_report.txt
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
OUT = "publication_results/controls"
PRED = "publication_results/topk_ablation/topk_ablation_query_predictions.csv"

try:
    import statsmodels.api as sm
except Exception:
    sm = None

# ------------------------------------------------------------------ load
pred = pd.read_csv(PRED, usecols=["Dataset", "Iteration", "Query_Index",
                                  "RoC_Size", "RoC_Fitness", "Top3_Correct"])
feat = pd.read_csv(f"{OUT}/query_features.csv")
df = pred.merge(feat, on=["Dataset", "Iteration", "Query_Index"], how="inner")
print(f"queries in predictions: {len(pred):,}  matched with features: {len(df):,}")
if len(df) < 0.99 * len(pred):
    raise SystemExit("Alignment problem: Query_Index/test order differs. Stop and check.")

CONF = ["conf_max", "pool_agree", "knn20_acc"]
lines = []


def z(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


def cv_auc(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    p = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            return np.nan
        m = LogisticRegression(max_iter=500).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, p)


# ------------------------------------------------------------------ Part A
rows = []
for ds, g in df.groupby("Dataset"):
    y = g["Top3_Correct"].values.astype(int)
    if y.min() == y.max():
        continue
    Z = pd.DataFrame({c: z(g[c]) for c in CONF + ["RoC_Fitness"]})
    grp = g["Iteration"].values
    auc_conf = cv_auc(Z[CONF].values, y, grp)
    auc_fit = cv_auc(Z[["RoC_Fitness"]].values, y, grp)
    auc_full = cv_auc(Z[CONF + ["RoC_Fitness"]].values, y, grp)
    coef = pval = np.nan
    if sm is not None:
        try:
            r = sm.Logit(y, sm.add_constant(Z[CONF + ["RoC_Fitness"]])).fit(disp=0)
            coef, pval = r.params["RoC_Fitness"], r.pvalues["RoC_Fitness"]
        except Exception:
            pass
    rows.append(dict(Dataset=ds, n=len(g), acc=y.mean(), AUC_confound=auc_conf,
                     AUC_fitness_only=auc_fit, AUC_full=auc_full,
                     dAUC=auc_full - auc_conf, fitness_coef_z=coef, fitness_p=pval))
A = pd.DataFrame(rows)
A.to_csv(f"{OUT}/fitness_confound_by_dataset.csv", index=False)

lines.append("PART A  fitness vs. correctness, controlling for pool confidence")
lines.append("-" * 70)
lines.append(f"datasets analysed: {len(A)}")
for c in ["AUC_fitness_only", "AUC_confound", "AUC_full", "dAUC"]:
    lines.append(f"mean {c:17s}: {A[c].mean():.4f}   median {A[c].median():.4f}")
pos = int((A["dAUC"] > 0).sum())
lines.append(f"datasets where adding fitness raises CV-AUC: {pos}/{len(A)} "
             f"(sign test p={binomtest(pos, len(A), 0.5).pvalue:.4g})")
try:
    w = wilcoxon(A["dAUC"].dropna())
    lines.append(f"Wilcoxon on dAUC over datasets: p={w.pvalue:.4g}")
except Exception:
    pass
if A["fitness_coef_z"].notna().any():
    pos_c = int((A["fitness_coef_z"] > 0).sum())
    sig = int(((A["fitness_p"] < 0.05) & (A["fitness_coef_z"] > 0)).sum())
    lines.append(f"fitness coef > 0 in {pos_c}/{len(A)} datasets; "
                 f"significant positive (p<0.05, uncorrected) in {sig}")
    lines.append(f"mean standardized fitness coef: {A['fitness_coef_z'].mean():.4f}")
lines.append("INTERPRETATION: if dAUC is ~0 and few coefficients are significant,")
lines.append("fitness is mostly a proxy for pool confidence/agreement.")

# ------------------------------------------------------------------ Part B
ctrl = pd.read_csv(f"{OUT}/control_runs.csv")
cm = ctrl.groupby("Dataset").mean(numeric_only=True).drop(columns="Iteration")
bade = pred.groupby("Dataset")["Top3_Correct"].mean().rename("bADE_RoC_Top3")
T = cm.join(bade).dropna()
res = []
for c in cm.columns:
    d = (T["bADE_RoC_Top3"] - T[c]) * 100
    try:
        p = wilcoxon(d).pvalue
    except Exception:
        p = np.nan
    res.append(dict(Control=c, Control_mean=T[c].mean() * 100,
                    bADE_mean=T["bADE_RoC_Top3"].mean() * 100,
                    bADE_minus_control_pp=d.mean(), wins=int((d > 0).sum()),
                    losses=int((d < 0).sum()), p_raw=p))
B = pd.DataFrame(res).sort_values("p_raw")
m = len(B)
holm = (B["p_raw"].values * (m - np.arange(m)))
B["p_holm"] = np.minimum(np.maximum.accumulate(holm), 1.0)
B.to_csv(f"{OUT}/control_vs_badE.csv", index=False)

lines.append("")
lines.append("PART B  bADE-RoC-DES (Top-3) vs. controls without optimization")
lines.append("-" * 70)
lines.append(B.round(4).to_string(index=False))
lines.append("INTERPRETATION: bADE_minus_control_pp > 0 with small p_holm means the")
lines.append("optimizer helps; ~0 means plain kNN does just as well.")

txt = "\n".join(lines)
open(f"{OUT}/controls_report.txt", "w").write(txt)
print(txt)
