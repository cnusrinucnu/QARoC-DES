import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path

OUT = Path("publication_results/paper_figures"); OUT.mkdir(parents=True, exist_ok=True)
tk   = pd.read_csv("publication_results/topk_ablation/topk_ablation_run_level.csv")
ctrl = pd.read_csv("publication_results/controls/control_runs.csv")
base = pd.read_csv("All_Methods_30_Datasets_Mean.csv").set_index("Dataset")
base = base.rename(columns={"DES_MHA": "DES-bADE"})

t3 = tk[tk.TopK == 3]
q = t3.groupby("Dataset").Accuracy.mean() * 100            # QARoC-DES, dataset means
c = ctrl.groupby("Dataset").mean(numeric_only=True) * 100   # controls, dataset means

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Fig. 2: mean accuracy over the 30 datasets
practical = {m: base[m].mean() for m in base.columns if m != "Oracle"}
practical["QARoC-DES"] = q.mean()
controls = {"kNN-7 + Top-3": c.KNN7_Top3.mean(), "kNN-10 + Top-3": c.KNN10_Top3.mean(),
            "kNN-20 + Top-3": c.KNN20_Top3.mean(), "Static majority": c.Static_Majority.mean()}
rows = sorted([(k, v, "p") for k, v in practical.items()] +
              [(k, v, "c") for k, v in controls.items()], key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(6, 5))
for i, (name, val, kind) in enumerate(rows):
    col = "tab:red" if name == "QARoC-DES" else ("tab:green" if kind == "c" else "tab:blue")
    ax.plot(val, i, "D" if kind == "c" else "o", color=col)
    ax.text(val + 0.05, i, f"{val:.2f}", va="center", fontsize=8)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel("Mean accuracy over 30 datasets (%)"); ax.set_xlim(77, 81.4)
save(fig, "Fig2_mean_accuracy")

# Fig. 4: dataset-level accuracy differences
rng = np.random.default_rng(0)
d1, d2 = q - base["DES-bADE"], q - c.KNN7_Top3
fig, ax = plt.subplots(figsize=(5, 4))
for x, d in enumerate((d1, d2)):
    ax.scatter(x + rng.uniform(-0.15, 0.15, len(d)), d, s=12, color="tab:blue")
    ax.hlines(d.mean(), x - 0.3, x + 0.3, color="red")
    ax.text(x + 0.32, d.mean(), f"mean {d.mean():+.2f}", color="red", fontsize=8, va="center")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xticks([0, 1]); ax.set_xticklabels(["QARoC-DES\nvs. DES-bADE", "QARoC-DES\nvs. kNN-7 + Top-3"])
ax.set_ylabel("Accuracy difference (pp)")
save(fig, "Fig4_dataset_differences")

# Fig. 5: metrics vs. K_E
fig, ax = plt.subplots(figsize=(5, 3.5))
for metric, lab in (("Accuracy", "Accuracy"), ("MacroF1", "Macro-F1"), ("BalancedAccuracy", "Balanced accuracy")):
    s = tk.groupby(["Dataset", "TopK"])[metric].mean().groupby("TopK").mean() * 100
    ax.plot(s.index, s.values, marker="o", label=lab)
ax.set_xticks([1, 3, 5, 7]); ax.set_xlabel("Number of selected classifiers, $K_E$")
ax.set_ylabel("Mean performance (%)"); ax.legend()
save(fig, "Fig5_topk_curve")

# Fig. 6: region characteristics
ds = t3.groupby("Dataset")[["Accuracy", "Mean_RoC_Size", "Mean_RoC_Fitness"]].mean()
fig, axs = plt.subplots(2, 2, figsize=(8, 6))
axs[0, 0].hist(t3.Mean_RoC_Size, bins=25); axs[0, 0].axvline(t3.Mean_RoC_Size.mean(), color="k")
axs[0, 0].set_title("(a) RoC size"); axs[0, 0].set_xlabel("Mean RoC size per run"); axs[0, 0].set_ylabel("Runs")
axs[0, 1].hist(t3.Mean_RoC_Fitness, bins=25); axs[0, 1].axvline(t3.Mean_RoC_Fitness.mean(), color="k")
axs[0, 1].set_title("(b) RoC fitness"); axs[0, 1].set_xlabel("Mean RoC fitness per run")
for ax, col, lab, ttl in ((axs[1, 0], "Mean_RoC_Size", "Dataset mean RoC size", "(c) Size vs. accuracy"),
                          (axs[1, 1], "Mean_RoC_Fitness", "Dataset mean RoC fitness", "(d) Fitness vs. accuracy")):
    r, p = spearmanr(ds[col], ds.Accuracy)
    ax.scatter(ds[col], ds.Accuracy * 100, s=14)
    ax.set_title(f"{ttl}\nrho={r:.2f}, p={p:.2f}", fontsize=9); ax.set_xlabel(lab); ax.set_ylabel("Mean accuracy (%)")
fig.tight_layout()
save(fig, "Fig6_roc_characteristics")
print("figures written to", OUT)
