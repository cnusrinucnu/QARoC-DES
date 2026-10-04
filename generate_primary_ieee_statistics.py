from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.libqsturng import qsturng

BASE = Path(".")
OUT = BASE / "publication_results" / "IEEE_Transactions_Analysis"
TABLES = OUT / "tables"
STATS = OUT / "statistics"
FIGS = OUT / "figures"
for _d in (OUT, TABLES, STATS, FIGS):
    _d.mkdir(parents=True, exist_ok=True)

FINAL = BASE / "publication_results/topk_ablation/topk_ablation_run_level.csv"
BASELINE = BASE / "All_Methods_30_Datasets_Mean.csv"

final = pd.read_csv(FINAL)
final = final[final["TopK"] == 3].drop(columns=["TopK"]).reset_index(drop=True)
baseline = pd.read_csv(BASELINE).set_index("Dataset")

# ============================================================
# 1. FINAL bADE DATASET MEANS
# ============================================================
bade = (
    final.groupby("Dataset")
    .agg(
        Accuracy=("Accuracy", "mean"),
        MacroF1=("MacroF1", "mean"),
        BalancedAccuracy=("BalancedAccuracy", "mean"),
        RoC_Size=("Mean_RoC_Size", "mean"),
        RoC_Fitness=("Mean_RoC_Fitness", "mean"),
    )
)

# Convert fractions to percentages for presentation/comparison
bade["Accuracy"] *= 100.0
bade["MacroF1"] *= 100.0
bade["BalancedAccuracy"] *= 100.0

# ============================================================
# 2. PRACTICAL METHODS ONLY
# ============================================================
# Baseline contains:
# 11 practical methods + Oracle
# Exclude Oracle from the primary inferential analysis.

practical_baselines = [
    "DES_MHA",
    "KNORA-U",
    "KNORAE",
    "DESKNN",
    "OLA",
    "LCA",
    "MLA",
    "MCB",
    "KNOP",
    "META-DES",
    "SingleBest",
]

comparison = baseline[practical_baselines].copy()
comparison["bADE-RoC-DES"] = bade["Accuracy"]
# remove floating-point noise so that identical accuracies are exact ties
comparison = comparison.round(9)

methods = list(comparison.columns)

N = len(comparison)
K = len(methods)

# ============================================================
# 3. FRIEDMAN TEST
# ============================================================
stat, p = friedmanchisquare(
    *[comparison[m].values for m in methods]
)

# rank 1 = best
rank_matrix = np.zeros((N, K))

for i in range(N):
    rank_matrix[i] = rankdata(
        -comparison.iloc[i].values,
        method="average"
    )

average_ranks = rank_matrix.mean(axis=0)

# Kendall's W
kendall_w = stat / (N * (K - 1))

rank1 = (comparison.values >= comparison.values.max(axis=1, keepdims=True) - 1e-9).sum(axis=0)

rank_table = pd.DataFrame({
    "Method": methods,
    "Mean_Accuracy": comparison.mean().values,
    "SD_Across_Datasets": comparison.std(ddof=1).values,
    "Average_Rank": average_ranks,
    "Rank_1_Count": rank1,
}).sort_values("Average_Rank")

rank_table.to_csv(
    TABLES / "Primary_Friedman_Ranks_12_Practical_Methods.csv",
    index=False
)

# ============================================================
# 4. FRIEDMAN SUMMARY
# ============================================================
friedman = pd.DataFrame([{
    "Datasets": N,
    "Practical_Methods": K,
    "Friedman_ChiSquare": stat,
    "Friedman_p": p,
    "Kendall_W": kendall_w,
}])

friedman.to_csv(
    STATS / "Primary_Friedman_Test.csv",
    index=False
)

# ============================================================
# 5. CRITICAL DIFFERENCE
# ============================================================
# Nemenyi CD:
# CD = q_alpha * sqrt(K(K+1)/(6N))
#
# statsmodels qsturng uses the studentized range distribution.
# For alpha=.05, q for the Nemenyi comparison is qsturng(.95,K,inf).

alpha = 0.05

q_alpha = float(qsturng(1 - alpha, K, np.inf)) / np.sqrt(2)   # Nemenyi q: Demsar (2006)
cd = q_alpha * np.sqrt(K * (K + 1) / (6 * N))

cd_table = pd.DataFrame([{
    "N_Datasets": N,
    "N_Methods": K,
    "Alpha": alpha,
    "q_alpha": q_alpha,
    "Critical_Difference": cd,
}])

cd_table.to_csv(
    STATS / "Nemenyi_Critical_Difference.csv",
    index=False
)

# ============================================================
# 6. PAIRWISE WILCOXON: bADE VS EACH PRACTICAL BASELINE
# ============================================================
target = "bADE-RoC-DES"

rows = []

for baseline_method in practical_baselines:

    a = comparison[target].values
    b = comparison[baseline_method].values

    diff = a - b

    W, raw_p = wilcoxon(
        a,
        b,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto"
    )

    nonzero = diff[diff != 0]

    if len(nonzero):
        abs_diff = np.abs(nonzero)
        abs_ranks = rankdata(abs_diff, method="average")

        positive_rank_sum = abs_ranks[nonzero > 0].sum()
        negative_rank_sum = abs_ranks[nonzero < 0].sum()

        rank_biserial = (
            positive_rank_sum - negative_rank_sum
        ) / (
            positive_rank_sum + negative_rank_sum
        )
    else:
        rank_biserial = 0.0

    rows.append({
        "Baseline": baseline_method,
        "Mean_Difference_pp": np.mean(diff),
        "Median_Difference_pp": np.median(diff),
        "Wilcoxon_W": W,
        "Raw_p": raw_p,
        "Wins": int(np.sum(diff > 0)),
        "Ties": int(np.sum(diff == 0)),
        "Losses": int(np.sum(diff < 0)),
        "Rank_Biserial": rank_biserial,
    })

pairwise = pd.DataFrame(rows)

reject, holm_p, _, _ = multipletests(
    pairwise["Raw_p"],
    method="holm"
)

pairwise["Holm_Adjusted_p"] = holm_p
pairwise["Significant_Holm_0.05"] = reject

pairwise = pairwise.sort_values(
    "Holm_Adjusted_p"
)

pairwise.to_csv(
    TABLES / "Primary_Wilcoxon_Holm_bADE_vs_11_Baselines.csv",
    index=False
)

# ============================================================
# 7. BOOTSTRAP CI FOR PAIRED DIFFERENCES
# ============================================================
rng = np.random.default_rng(20260925)

boot_rows = []

for baseline_method in practical_baselines:

    diff = (
        comparison[target].values
        - comparison[baseline_method].values
    )

    boot_means = np.empty(20000)

    for i in range(len(boot_means)):
        sample = rng.choice(
            diff,
            size=N,
            replace=True
        )
        boot_means[i] = sample.mean()

    lo, hi = np.percentile(
        boot_means,
        [2.5, 97.5]
    )

    boot_rows.append({
        "Baseline": baseline_method,
        "Observed_Mean_Difference_pp": diff.mean(),
        "Bootstrap_CI_Lower": lo,
        "Bootstrap_CI_Upper": hi,
    })

bootstrap = pd.DataFrame(boot_rows)

bootstrap.to_csv(
    STATS / "Primary_Bootstrap_CI_bADE_vs_11_Baselines.csv",
    index=False
)

# ============================================================
# 8. WIN/TIE/LOSS MATRIX
# ============================================================
wtl = pairwise[
    ["Baseline", "Wins", "Ties", "Losses"]
].copy()

wtl.to_csv(
    TABLES / "Primary_Win_Tie_Loss.csv",
    index=False
)

# ============================================================
# 9. CD-DIAGRAM DATA
# ============================================================
cd_data = rank_table.copy()
cd_data["Average_Rank"] = cd_data["Average_Rank"].astype(float)

cd_data.to_csv(
    TABLES / "Critical_Difference_Diagram_Data.csv",
    index=False
)

# ============================================================
# 10. LATEX TABLES
# ============================================================
def save_latex(df, filename, caption, label):
    tex = df.to_latex(
        index=False,
        escape=True,
        float_format=lambda x: f"{x:.4f}",
        caption=caption,
        label=label
    )
    (TABLES / filename).write_text(tex)

save_latex(
    rank_table,
    "Primary_Friedman_Ranks_12_Practical_Methods.tex",
    "Friedman average ranks across the 30 benchmark datasets, excluding Oracle from the primary inferential comparison.",
    "tab:primary-friedman"
)

save_latex(
    pairwise,
    "Primary_Wilcoxon_Holm_bADE_vs_11_Baselines.tex",
    "Paired Wilcoxon signed-rank tests comparing bADE-RoC-DES with the practical baseline methods.",
    "tab:primary-wilcoxon"
)

save_latex(
    wtl,
    "Primary_Win_Tie_Loss.tex",
    "Win, tie, and loss counts of bADE-RoC-DES against the practical baseline methods.",
    "tab:win-tie-loss"
)

# ============================================================
# 11. TEXT REPORT
# ============================================================
report = []

report.append("=" * 100)
report.append("PRIMARY IEEE TRANSACTIONS STATISTICAL ANALYSIS")
report.append("bADE-RoC-DES vs. 11 Practical Baselines")
report.append("Oracle excluded from primary inferential comparison")
report.append("=" * 100)
report.append("")
report.append(f"Datasets: {N}")
report.append(f"Methods: {K}")
report.append("")
report.append("FRIEDMAN TEST")
report.append(f"Chi-square       : {stat:.8f}")
report.append(f"p-value          : {p:.8e}")
report.append(f"Kendall's W      : {kendall_w:.8f}")
report.append("")
report.append("NEMENYI CRITICAL DIFFERENCE")
report.append(f"q(alpha=0.05)     : {q_alpha:.8f}")
report.append(f"Critical Difference: {cd:.8f}")
report.append("")
report.append("AVERAGE RANKS")
report.append(rank_table.to_string(index=False))
report.append("")
report.append("PAIRWISE WILCOXON + HOLM")
report.append(pairwise.to_string(index=False))
report.append("")
report.append("BOOTSTRAP 95% CI")
report.append(bootstrap.to_string(index=False))
report.append("")
report.append("WIN/TIE/LOSS")
report.append(wtl.to_string(index=False))

(STATS / "Primary_Statistical_Report.txt").write_text(
    "\n".join(report)
)

# ============================================================
# 12. CONSOLE OUTPUT
# ============================================================
print("=" * 100)
print("PRIMARY IEEE TRANSACTIONS STATISTICAL ANALYSIS COMPLETE")
print("=" * 100)

print(f"Datasets              : {N}")
print(f"Practical methods     : {K}")
print("")
print(f"Friedman chi-square   : {stat:.8f}")
print(f"Friedman p-value      : {p:.8e}")
print(f"Kendall W             : {kendall_w:.8f}")
print("")
print(f"Nemenyi q(0.05)       : {q_alpha:.8f}")
print(f"Critical Difference   : {cd:.8f}")
print("")
print("PRIMARY AVERAGE RANKS")
print(rank_table.to_string(index=False))
print("")
print("bADE-RoC-DES PAIRWISE TESTS")
print(pairwise.to_string(index=False))
print("")
print("Output:")
print(STATS)
print(TABLES)
