import os
import numpy as np
import pandas as pd

from scipy.stats import spearmanr

BASE = "publication_results/topk_ablation"
INPUT = os.path.join(
    BASE,
    "topk_ablation_query_predictions.csv"
)

OUT = os.path.join(
    BASE,
    "query_difficulty_analysis"
)

df = pd.read_csv(INPUT)

# Global fitness quintiles
df["Fitness_Quintile"] = pd.qcut(
    df["RoC_Fitness"],
    q=5,
    labels=["Q1-Low", "Q2", "Q3", "Q4", "Q5-High"],
    duplicates="drop"
)

rows = []

for q, g in df.groupby(
    "Fitness_Quintile",
    observed=False
):
    row = {
        "Fitness_Quintile": str(q),
        "N_Queries": len(g),
        "Mean_Fitness": g["RoC_Fitness"].mean(),
        "Mean_RoC_Size": g["RoC_Size"].mean(),
    }

    for k in [1, 3, 5, 7]:
        row[f"Top{k}_Accuracy"] = g[
            f"Top{k}_Correct"
        ].mean()

    rows.append(row)

qdf = pd.DataFrame(rows)

print("=" * 80)
print("ACCURACY BY RoC FITNESS QUINTILE")
print("=" * 80)
print(qdf.to_string(index=False))

qdf.to_csv(
    os.path.join(
        OUT,
        "accuracy_by_fitness_quintile.csv"
    ),
    index=False
)

# Dataset-level robustness:
# calculate within-dataset fitness/correctness Spearman
rows = []

for dataset, g in df.groupby("Dataset"):

    for k in [1, 3, 5, 7]:

        rho, p = spearmanr(
            g["RoC_Fitness"],
            g[f"Top{k}_Correct"]
        )

        rows.append({
            "Dataset": dataset,
            "TopK": k,
            "N_Queries": len(g),
            "Spearman_Fitness_Correct": rho,
            "P_Value": p
        })

within = pd.DataFrame(rows)

within.to_csv(
    os.path.join(
        OUT,
        "within_dataset_fitness_correctness.csv"
    ),
    index=False
)

print("\n" + "=" * 80)
print("WITHIN-DATASET FITNESS/CORRECTNESS")
print("=" * 80)

for k in [1, 3, 5, 7]:

    x = within[
        within["TopK"] == k
    ]

    print(
        f"\nTop-{k}:"
    )
    print(
        "Median rho:",
        x["Spearman_Fitness_Correct"].median()
    )
    print(
        "Mean rho:",
        x["Spearman_Fitness_Correct"].mean()
    )
    print(
        "Positive rho:",
        (x["Spearman_Fitness_Correct"] > 0).sum(),
        "/",
        len(x)
    )

print("\nSaved:")
print(
    os.path.join(
        OUT,
        "accuracy_by_fitness_quintile.csv"
    )
)
print(
    os.path.join(
        OUT,
        "within_dataset_fitness_correctness.csv"
    )
)
