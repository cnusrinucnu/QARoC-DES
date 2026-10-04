import os
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, spearmanr
from itertools import combinations

BASE = "publication_results/topk_ablation"
RUN_FILE = os.path.join(BASE, "topk_ablation_run_level.csv")
OUT = os.path.join(BASE, "statistical_analysis")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(RUN_FILE)

print("Rows:", len(df))
print("Datasets:", df["Dataset"].nunique())
print("Iterations:", df["Iteration"].nunique())
print("TopK:", sorted(df["TopK"].unique()))

metrics = ["Accuracy", "MacroF1", "BalancedAccuracy"]
ks = [1, 3, 5, 7]

# ------------------------------------------------------------
# 1. Dataset-wise means: proper independent experimental unit
# ------------------------------------------------------------
dataset_mean = (
    df.groupby(["Dataset", "TopK"])[metrics]
      .mean()
      .reset_index()
)

dataset_mean.to_csv(
    os.path.join(OUT, "topk_dataset_wise_means.csv"),
    index=False
)

# ------------------------------------------------------------
# 2. Overall descriptive statistics
# ------------------------------------------------------------
desc = (
    df.groupby("TopK")[metrics]
      .agg(["mean", "std", "median"])
)

desc.to_csv(
    os.path.join(OUT, "topk_overall_descriptive.csv")
)

print("\n=== OVERALL DESCRIPTIVE RESULTS ===")
print(desc.round(6))

# ------------------------------------------------------------
# 3. Friedman test across K=1,3,5,7
# ------------------------------------------------------------
friedman_rows = []

for metric in metrics:
    wide = dataset_mean.pivot(
        index="Dataset",
        columns="TopK",
        values=metric
    )[ks].dropna()

    stat, p = friedmanchisquare(
        wide[1],
        wide[3],
        wide[5],
        wide[7]
    )

    n = len(wide)
    kendall_w = stat / (n * (len(ks) - 1))

    friedman_rows.append({
        "Metric": metric,
        "N_Datasets": n,
        "Friedman_Chi2": stat,
        "p_value": p,
        "Kendall_W": kendall_w
    })

friedman_df = pd.DataFrame(friedman_rows)
friedman_df.to_csv(
    os.path.join(OUT, "friedman_topk.csv"),
    index=False
)

print("\n=== FRIEDMAN TEST ===")
print(friedman_df.round(8).to_string(index=False))

# ------------------------------------------------------------
# 4. Pairwise Wilcoxon tests
#    Primary comparison: K=3 against other K values
# ------------------------------------------------------------
pairs = [(3,1), (3,5), (3,7)]

wilcoxon_rows = []

for metric in metrics:
    wide = dataset_mean.pivot(
        index="Dataset",
        columns="TopK",
        values=metric
    )[ks].dropna()

    for k_a, k_b in pairs:

        x = wide[k_a].values
        y = wide[k_b].values
        diff = x - y

        stat, p = wilcoxon(
            x, y,
            alternative="two-sided",
            zero_method="wilcox"
        )

        # Rank-biserial correlation
        abs_diff = np.abs(diff)
        nonzero = abs_diff > 0

        if nonzero.sum() > 0:
            ranks = pd.Series(abs_diff[nonzero]).rank().values
            signed = np.sign(diff[nonzero])
            r_plus = ranks[signed > 0].sum()
            r_minus = ranks[signed < 0].sum()
            rbc = (r_plus - r_minus) / (r_plus + r_minus)
        else:
            rbc = 0.0

        wilcoxon_rows.append({
            "Metric": metric,
            "Comparison": f"K={k_a} vs K={k_b}",
            "N_Datasets": len(x),
            "Mean_K3": np.mean(x),
            "Mean_Other": np.mean(y),
            "Mean_Difference": np.mean(diff),
            "Median_Difference": np.median(diff),
            "Wilcoxon_W": stat,
            "p_raw": p,
            "Rank_Biserial": rbc
        })

wilcoxon_df = pd.DataFrame(wilcoxon_rows)

# ------------------------------------------------------------
# Holm correction within each metric
# ------------------------------------------------------------
def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)

    for rank, idx in enumerate(order):
        adjusted[idx] = min((m-rank) * pvals[idx], 1.0)

    # enforce monotonicity
    for i in range(1, m):
        prev = order[i-1]
        curr = order[i]
        adjusted[curr] = max(adjusted[curr], adjusted[prev])

    return adjusted

wilcoxon_df["p_holm"] = (
    wilcoxon_df.groupby("Metric")["p_raw"]
    .transform(holm_adjust)
)

wilcoxon_df["Significant_0.05"] = wilcoxon_df["p_holm"] < 0.05

wilcoxon_df.to_csv(
    os.path.join(OUT, "wilcoxon_top3_vs_otherK.csv"),
    index=False
)

print("\n=== WILCOXON: K=3 VS OTHER K ===")
print(
    wilcoxon_df[
        [
            "Metric", "Comparison", "Mean_Difference",
            "Median_Difference", "p_raw", "p_holm",
            "Rank_Biserial", "Significant_0.05"
        ]
    ].round(8).to_string(index=False)
)

# ------------------------------------------------------------
# 5. Full pairwise Wilcoxon matrix
# ------------------------------------------------------------
all_pairs = list(combinations(ks, 2))
all_rows = []

for metric in metrics:

    wide = dataset_mean.pivot(
        index="Dataset",
        columns="TopK",
        values=metric
    )[ks].dropna()

    raw_ps = []

    temp = []

    for ka, kb in all_pairs:

        x = wide[ka].values
        y = wide[kb].values

        stat, p = wilcoxon(
            x, y,
            alternative="two-sided",
            zero_method="wilcox"
        )

        diff = x-y
        nz = np.abs(diff) > 0

        if nz.sum():
            ranks = pd.Series(np.abs(diff[nz])).rank().values
            signs = np.sign(diff[nz])
            rp = ranks[signs > 0].sum()
            rm = ranks[signs < 0].sum()
            rbc = (rp-rm)/(rp+rm)
        else:
            rbc = 0

        raw_ps.append(p)

        temp.append({
            "Metric": metric,
            "Comparison": f"K={ka} vs K={kb}",
            "Mean_Difference": np.mean(diff),
            "Median_Difference": np.median(diff),
            "p_raw": p,
            "Rank_Biserial": rbc
        })

    adj = holm_adjust(raw_ps)

    for row, p_adj in zip(temp, adj):
        row["p_holm"] = p_adj
        row["Significant_0.05"] = p_adj < 0.05
        all_rows.append(row)

all_pairwise = pd.DataFrame(all_rows)

all_pairwise.to_csv(
    os.path.join(OUT, "all_pairwise_wilcoxon_topk.csv"),
    index=False
)

# ------------------------------------------------------------
# 6. Per-dataset comparison table
# ------------------------------------------------------------
for metric in metrics:
    w = dataset_mean.pivot(
        index="Dataset",
        columns="TopK",
        values=metric
    )[ks].copy()

    w.columns = [f"K{int(k)}" for k in w.columns]
    w["K3_minus_K1"] = w["K3"] - w["K1"]
    w["K3_minus_K5"] = w["K3"] - w["K5"]
    w["K3_minus_K7"] = w["K3"] - w["K7"]

    filename = metric.lower().replace("_", "")
    w.to_csv(
        os.path.join(
            OUT,
            f"dataset_comparison_{filename}.csv"
        )
    )

# ------------------------------------------------------------
# 7. Count datasets where K=3 has highest metric
#    Descriptive only — no inferential claim.
# ------------------------------------------------------------
counts = []

for metric in metrics:
    w = dataset_mean.pivot(
        index="Dataset",
        columns="TopK",
        values=metric
    )[ks]

    counts.append({
        "Metric": metric,
        "K1": int((w[1] == w.max(axis=1)).sum()),
        "K3": int((w[3] == w.max(axis=1)).sum()),
        "K5": int((w[5] == w.max(axis=1)).sum()),
        "K7": int((w[7] == w.max(axis=1)).sum())
    })

counts_df = pd.DataFrame(counts)
counts_df.to_csv(
    os.path.join(OUT, "dataset_max_counts.csv"),
    index=False
)

print("\n=== DATASET MAXIMUM COUNTS ===")
print(counts_df.to_string(index=False))

# ------------------------------------------------------------
# 8. LaTeX descriptive table
# ------------------------------------------------------------
latex_rows = []

for k in ks:
    sub = df[df.TopK == k]

    row = {
        "K": k
    }

    for metric in metrics:
        row[f"{metric}_mean"] = sub[metric].mean()
        row[f"{metric}_std"] = sub[metric].std()

    latex_rows.append(row)

latex_df = pd.DataFrame(latex_rows)

with open(
    os.path.join(OUT, "topk_descriptive_table.tex"),
    "w"
) as f:

    f.write(
r"""\begin{table}[t]
\centering
\caption{Effect of the number of selected classifiers ($K$) on bADE-RoC-DES performance.}
\label{tab:topk_ablation}
\begin{tabular}{c|ccc}
\hline
$K$ & Accuracy (\%) & Macro-F1 (\%) & Balanced Accuracy (\%)\\
\hline
"""
    )

    for _, r in latex_df.iterrows():
        f.write(
            f"{int(r['K'])} & "
            f"{100*r['Accuracy_mean']:.2f} $\\pm$ {100*r['Accuracy_std']:.2f} & "
            f"{100*r['MacroF1_mean']:.2f} $\\pm$ {100*r['MacroF1_std']:.2f} & "
            f"{100*r['BalancedAccuracy_mean']:.2f} $\\pm$ {100*r['BalancedAccuracy_std']:.2f} \\\\\n"
        )

    f.write(
r"""\hline
\end{tabular}
\end{table}
"""
    )

# ------------------------------------------------------------
# 9. Statistical LaTeX table
# ------------------------------------------------------------
with open(
    os.path.join(OUT, "topk_statistical_table.tex"),
    "w"
) as f:

    f.write(
r"""\begin{table}[t]
\centering
\caption{Paired statistical analysis of $K=3$ against alternative selection sizes. Tests are conducted on dataset-wise means across the 30 datasets, with Holm correction within each metric.}
\label{tab:topk_stats}
\begin{tabular}{l|c|r|r|r}
\hline
Metric & Comparison & $\Delta$ & $p_{\mathrm{Holm}}$ & RBC\\
\hline
"""
    )

    for _, r in wilcoxon_df.iterrows():

        metric_name = {
            "Accuracy": "Accuracy",
            "MacroF1": "Macro-F1",
            "BalancedAccuracy": "Bal. Acc."
        }[r["Metric"]]

        sig = "*" if r["Significant_0.05"] else ""

        f.write(
            f"{metric_name} & "
            f"{r['Comparison']} & "
            f"{100*r['Mean_Difference']:.2f} & "
            f"{r['p_holm']:.4f}{sig} & "
            f"{r['Rank_Biserial']:.3f} \\\\\n"
        )

    f.write(
r"""\hline
\multicolumn{5}{l}{\footnotesize * $p_{\mathrm{Holm}}<0.05$. RBC: rank-biserial correlation.}\\
\end{tabular}
\end{table}
"""
    )

# ------------------------------------------------------------
# 10. Final analysis report
# ------------------------------------------------------------
with open(
    os.path.join(OUT, "TOPK_ANALYSIS_REPORT.txt"),
    "w"
) as f:

    f.write("TOP-K ABLATION STATISTICAL ANALYSIS\n")
    f.write("="*60 + "\n\n")

    f.write("Experimental structure:\n")
    f.write(f"Datasets: {df.Dataset.nunique()}\n")
    f.write(f"Iterations: {df.Iteration.nunique()}\n")
    f.write(f"Run-level rows: {len(df)}\n")
    f.write("K values: 1, 3, 5, 7\n\n")

    f.write("Overall descriptive results:\n")
    f.write(desc.round(6).to_string())
    f.write("\n\n")

    f.write("Friedman tests:\n")
    f.write(friedman_df.round(8).to_string(index=False))
    f.write("\n\n")

    f.write("K=3 versus alternatives:\n")
    f.write(
        wilcoxon_df.round(8).to_string(index=False)
    )
    f.write("\n\n")

    f.write("Dataset maximum counts:\n")
    f.write(counts_df.to_string(index=False))
    f.write("\n")

print("\nAnalysis complete.")
print("Output directory:")
print(os.path.abspath(OUT))
print("\nFiles:")
for x in sorted(os.listdir(OUT)):
    print(" ", x)
