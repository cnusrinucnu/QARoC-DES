from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(".")
OUT = BASE / "publication_results" / "IEEE_Transactions_Analysis"
TABLES = OUT / "tables"
FIGS = OUT / "figures"

rank_file = TABLES / "Primary_Friedman_Ranks_12_Practical_Methods.csv"
cd_file = OUT / "statistics" / "Nemenyi_Critical_Difference.csv"

ranks = pd.read_csv(rank_file)
cd = pd.read_csv(cd_file).iloc[0]["Critical_Difference"]

# Smaller rank is better
ranks = ranks.sort_values("Average_Rank").reset_index(drop=True)

methods = ranks["Method"].tolist()
values = ranks["Average_Rank"].to_numpy()

fig, ax = plt.subplots(figsize=(11, 4.8))

# Rank axis
xmin = 1
xmax = max(12, np.ceil(values.max() + 1))

ax.set_xlim(xmin - 0.15, xmax + 0.15)
ax.set_ylim(-1.25, 1.8)

ax.set_xlabel("Average Rank (lower is better)")
ax.set_yticks([])

# Main rank line
ax.plot(
    [xmin, xmax],
    [0, 0],
    linewidth=1.2
)

for x in range(int(xmin), int(xmax) + 1):
    ax.plot([x, x], [-0.06, 0.06], linewidth=1)
    ax.text(
        x,
        -0.16,
        str(x),
        ha="center",
        va="top",
        fontsize=9
    )

# Method labels
upper = [0, 1, 2, 3, 4, 5]
lower = [6, 7, 8, 9, 10, 11]

for idx in upper:
    r = values[idx]
    ax.plot([r, r], [0.05, 0.43], linewidth=0.9)
    ax.text(
        r,
        0.50,
        methods[idx],
        ha="center",
        va="bottom",
        fontsize=8.5,
        rotation=35
    )

for idx in lower:
    r = values[idx]
    ax.plot([r, r], [-0.05, -0.43], linewidth=0.9)
    ax.text(
        r,
        -0.50,
        methods[idx],
        ha="center",
        va="top",
        fontsize=8.5,
        rotation=-35
    )

# CD bar at top
cd_y = 1.25
cd_start = 1.0
cd_end = cd_start + cd

ax.plot(
    [cd_start, cd_end],
    [cd_y, cd_y],
    linewidth=2
)
ax.plot(
    [cd_start, cd_start],
    [cd_y - 0.08, cd_y + 0.08],
    linewidth=1
)
ax.plot(
    [cd_end, cd_end],
    [cd_y - 0.08, cd_y + 0.08],
    linewidth=1
)

ax.text(
    (cd_start + cd_end) / 2,
    cd_y + 0.12,
    f"Critical Difference = {cd:.3f}",
    ha="center",
    va="bottom",
    fontsize=9
)

ax.set_title(
    "Critical Difference Diagram: 12 Practical Methods",
    pad=25
)

ax.grid(False)

fig.tight_layout()

fig.savefig(
    FIGS / "Fig13_Critical_Difference_Diagram.pdf",
    bbox_inches="tight"
)
fig.savefig(
    FIGS / "Fig13_Critical_Difference_Diagram.png",
    bbox_inches="tight",
    dpi=600
)

plt.close(fig)

print("=" * 80)
print("CRITICAL DIFFERENCE DIAGRAM GENERATED")
print("=" * 80)
print(f"Critical Difference : {cd:.6f}")
print(f"Output              : {FIGS}")
print("")
print("Average ranks:")
print(ranks[["Method", "Average_Rank"]].to_string(index=False))
