import os
import numpy as np
import pandas as pd

from scipy.stats import (
    spearmanr,
    mannwhitneyu
)

BASE = "publication_results/topk_ablation"

INPUT = os.path.join(
    BASE,
    "topk_ablation_query_predictions.csv"
)

OUT = os.path.join(
    BASE,
    "query_difficulty_analysis"
)

os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(INPUT)

print("=" * 90)
print("QUERY DIFFICULTY / RoC-SIZE ANALYSIS")
print("=" * 90)

print("Rows:", len(df))
print("Datasets:", df["Dataset"].nunique())
print(
    "Dataset × iteration:",
    df.groupby(["Dataset", "Iteration"]).ngroups
)

# ============================================================
# 1. OVERALL CORRELATION
# ============================================================

corr_rows = []

for k in [1, 3, 5, 7]:

    r_size, p_size = spearmanr(
        df["RoC_Size"],
        df[f"Top{k}_Correct"]
    )

    r_fit, p_fit = spearmanr(
        df["RoC_Fitness"],
        df[f"Top{k}_Correct"]
    )

    corr_rows.append({
        "TopK": k,
        "Spearman_RoC_Size_Correct": r_size,
        "P_RoC_Size_Correct": p_size,
        "Spearman_RoC_Fitness_Correct": r_fit,
        "P_RoC_Fitness_Correct": p_fit
    })

corr = pd.DataFrame(corr_rows)

print("\n--- Spearman correlations ---")
print(corr.to_string(index=False))

corr.to_csv(
    os.path.join(
        OUT,
        "roc_correctness_correlations.csv"
    ),
    index=False
)

# ============================================================
# 2. CORRECT VS INCORRECT
# ============================================================

summary_rows = []

for k in [1, 3, 5, 7]:

    correct = df.loc[
        df[f"Top{k}_Correct"] == 1
    ]

    incorrect = df.loc[
        df[f"Top{k}_Correct"] == 0
    ]

    # RoC size
    size_test = mannwhitneyu(
        correct["RoC_Size"],
        incorrect["RoC_Size"],
        alternative="two-sided"
    )

    # RoC fitness
    fit_test = mannwhitneyu(
        correct["RoC_Fitness"],
        incorrect["RoC_Fitness"],
        alternative="two-sided"
    )

    summary_rows.append({
        "TopK": k,

        "Correct_N": len(correct),
        "Incorrect_N": len(incorrect),

        "Correct_Mean_RoC": correct["RoC_Size"].mean(),
        "Correct_Median_RoC": correct["RoC_Size"].median(),
        "Correct_Q1_RoC": correct["RoC_Size"].quantile(.25),
        "Correct_Q3_RoC": correct["RoC_Size"].quantile(.75),

        "Incorrect_Mean_RoC": incorrect["RoC_Size"].mean(),
        "Incorrect_Median_RoC": incorrect["RoC_Size"].median(),
        "Incorrect_Q1_RoC": incorrect["RoC_Size"].quantile(.25),
        "Incorrect_Q3_RoC": incorrect["RoC_Size"].quantile(.75),

        "Correct_Mean_Fitness": correct["RoC_Fitness"].mean(),
        "Incorrect_Mean_Fitness": incorrect["RoC_Fitness"].mean(),

        "MannWhitney_RoC_U": size_test.statistic,
        "MannWhitney_RoC_p": size_test.pvalue,

        "MannWhitney_Fitness_U": fit_test.statistic,
        "MannWhitney_Fitness_p": fit_test.pvalue
    })

correctness = pd.DataFrame(summary_rows)

print("\n--- Correct vs Incorrect ---")
print(correctness.to_string(index=False))

correctness.to_csv(
    os.path.join(
        OUT,
        "correct_vs_incorrect_roc.csv"
    ),
    index=False
)

# ============================================================
# 3. RoC SIZE BINS
# ============================================================

bins = [
    0,
    4,
    7,
    10,
    15,
    20
]

labels = [
    "1-4",
    "5-7",
    "8-10",
    "11-15",
    "16-20"
]

df["RoC_Size_Bin"] = pd.cut(
    df["RoC_Size"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

bin_rows = []

for bin_name, g in df.groupby(
    "RoC_Size_Bin",
    observed=False
):

    row = {
        "RoC_Size_Bin": str(bin_name),
        "N_Queries": len(g),
        "Mean_RoC_Size": g["RoC_Size"].mean(),
        "Mean_RoC_Fitness": g["RoC_Fitness"].mean()
    }

    for k in [1, 3, 5, 7]:

        row[f"Top{k}_Accuracy"] = \
            g[f"Top{k}_Correct"].mean()

    bin_rows.append(row)

bins_df = pd.DataFrame(bin_rows)

print("\n--- Accuracy by RoC-size bin ---")
print(bins_df.to_string(index=False))

bins_df.to_csv(
    os.path.join(
        OUT,
        "accuracy_by_roc_size_bin.csv"
    ),
    index=False
)

# ============================================================
# 4. DATASET-LEVEL CORRELATION
# ============================================================

dataset_rows = []

for dataset, g in df.groupby("Dataset"):

    row = {
        "Dataset": dataset,
        "N_Queries": len(g),
        "Mean_RoC_Size": g["RoC_Size"].mean(),
        "Median_RoC_Size": g["RoC_Size"].median(),
        "Mean_RoC_Fitness": g["RoC_Fitness"].mean()
    }

    for k in [1, 3, 5, 7]:
        row[f"Top{k}_Accuracy"] = \
            g[f"Top{k}_Correct"].mean()

    dataset_rows.append(row)

dataset_df = pd.DataFrame(dataset_rows)

dataset_corr = []

for k in [1, 3, 5, 7]:

    r_size, p_size = spearmanr(
        dataset_df["Mean_RoC_Size"],
        dataset_df[f"Top{k}_Accuracy"]
    )

    r_fit, p_fit = spearmanr(
        dataset_df["Mean_RoC_Fitness"],
        dataset_df[f"Top{k}_Accuracy"]
    )

    dataset_corr.append({
        "TopK": k,
        "Dataset_Spearman_RoC_Size_Accuracy": r_size,
        "Dataset_P_RoC_Size_Accuracy": p_size,
        "Dataset_Spearman_RoC_Fitness_Accuracy": r_fit,
        "Dataset_P_RoC_Fitness_Accuracy": p_fit
    })

dataset_corr = pd.DataFrame(dataset_corr)

print("\n--- Dataset-level correlations ---")
print(dataset_corr.to_string(index=False))

dataset_df.to_csv(
    os.path.join(
        OUT,
        "dataset_query_difficulty_summary.csv"
    ),
    index=False
)

dataset_corr.to_csv(
    os.path.join(
        OUT,
        "dataset_level_correlations.csv"
    ),
    index=False
)

# ============================================================
# 5. ROС SIZE DISTRIBUTION
# ============================================================

roc_distribution = (
    df["RoC_Size"]
    .value_counts()
    .sort_index()
    .rename_axis("RoC_Size")
    .reset_index(name="N_Queries")
)

roc_distribution["Percentage"] = (
    100 *
    roc_distribution["N_Queries"] /
    len(df)
)

print("\n--- RoC-size distribution ---")
print(roc_distribution.to_string(index=False))

roc_distribution.to_csv(
    os.path.join(
        OUT,
        "roc_size_distribution.csv"
    ),
    index=False
)

print("\n" + "=" * 90)
print("FILES SAVED")
print("=" * 90)

for f in [
    "roc_correctness_correlations.csv",
    "correct_vs_incorrect_roc.csv",
    "accuracy_by_roc_size_bin.csv",
    "dataset_query_difficulty_summary.csv",
    "dataset_level_correlations.csv",
    "roc_size_distribution.csv"
]:
    print(
        os.path.join(OUT, f)
    )
