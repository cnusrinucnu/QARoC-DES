import pickle
from pathlib import Path
import numpy as np, pandas as pd

base = pd.read_csv("All_Methods_30_Datasets_Mean.csv").set_index("Dataset")
rows = []
for ds in base.index:
    for m in base.columns:
        p = Path(f"Experiment/Results/{ds}/{m}_{ds}_result.p")
        if not p.exists():
            rows.append((ds, m, np.nan, base.loc[ds, m], np.nan, "missing"))
            continue
        r = pickle.load(open(p, "rb"))
        v = float(np.mean(r))
        rows.append((ds, m, v, base.loc[ds, m], len(r), "ok"))
df = pd.DataFrame(rows, columns=["Dataset", "Method", "Rebuilt_mean", "CSV_mean", "N_iter", "Status"])
df["AbsDiff"] = (df.Rebuilt_mean - df.CSV_mean).abs()
df.to_csv("baseline_rebuild_check.csv", index=False)
print("missing files   :", int((df.Status == "missing").sum()))
print("iterations != 20:", int((df.N_iter.dropna() != 20).sum()))
print("max |diff| (pp) :", df.AbsDiff.max())
bad = df[df.AbsDiff > 1e-6]
print("mismatches > 1e-6:", len(bad))
print(bad.head(20).to_string(index=False))
