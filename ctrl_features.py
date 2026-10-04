#!/usr/bin/env python3
"""
ctrl_features.py  --  run from the repository root (needs the Experiment/ data and the DES-bADE environment).

No optimizer is run. For every dataset/iteration it reloads the SAME saved
pool and the SAME split the main experiments used, and computes:

  (1) per-query confound features (independent of the ADE search):
        conf_max   : max of the pool-mean posterior
        pool_agree : fraction of classifiers voting for the pool-majority class
        knn20_acc  : mean classifier accuracy on the 20 nearest DSEL samples
  (2) controls, per run:
        Static_Majority   : majority vote of all pool classifiers
        KNN{7,10,20}_Top3 : Top-3 local-accuracy selection on a plain kNN
                            region (same competence rule + majority vote as
                            bADE-RoC-DES, but NO optimization)

Outputs (publication_results/controls/):
    query_features.csv, control_runs.csv
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, ".")
import helpers  # noqa: E402

DATASETS = [
    "Adult", "Banknote", "Blood", "Breast", "Cardiotocography", "Faults",
    "German", "Glass", "Haberman", "Heart", "ILPD", "Ionosphere", "Iris",
    "Laryngeal3", "Liver", "Magic", "Mammographic", "Monk2", "Pima",
    "Sonar", "Statlog", "Steel", "Thyroid", "Transfusion", "Vehicle",
    "Vertebral", "Voice3", "WDVG1", "Weaning", "Wine",
]
N_ITR = 20
K_LIST = [7, 10, 20]
TOP = 3
N_WORKERS = 30
OUT = "publication_results/controls"


def vote(preds_sel, n_classes):
    """preds_sel: (r, Q) labels -> majority label (ties -> smallest label)."""
    r, Q = preds_sel.shape
    counts = np.zeros((n_classes, Q), dtype=np.int32)
    cols = np.arange(Q)
    for i in range(r):
        np.add.at(counts, (preds_sel[i], cols), 1)
    return counts.argmax(axis=0), counts


def run_dataset(dataset):
    t0 = time.time()
    pools = helpers.load_pool(dataset)
    q_rows, r_rows = [], []

    for itr in range(N_ITR):
        d = np.load(f"./Experiment/Datasets/{dataset}/{dataset}{itr}.npz")
        X_dsel, y_dsel = d["X_DSEL"], d["y_DSEL"].astype(int)
        X_test, y_test = d["X_test"], d["y_test"].astype(int)
        pool = list(pools[itr])
        M = len(pool)

        dsel_pred = np.stack([c.predict(X_dsel) for c in pool]).astype(int)
        test_pred = np.stack([c.predict(X_test) for c in pool]).astype(int)
        test_prob = np.stack([c.predict_proba(X_test) for c in pool])  # (M,Q,C)
        n_classes = int(max(y_dsel.max(), y_test.max(),
                            dsel_pred.max(), test_pred.max())) + 1
        dsel_corr = (dsel_pred == y_dsel[None, :]).astype(np.float32)  # (M,N)

        Q = X_test.shape[0]
        cols = np.arange(Q)
        kmax = max(K_LIST)
        nn = NearestNeighbors(n_neighbors=kmax).fit(X_dsel)
        idx = nn.kneighbors(X_test, return_distance=False)  # (Q,kmax)

        # ---- confound features
        _, counts_all = vote(test_pred, n_classes)
        pool_agree = counts_all.max(axis=0) / M
        conf_max = test_prob.mean(axis=0).max(axis=1)
        knn20_acc = dsel_corr[:, idx[:, :20]].mean(axis=(0, 2))

        q_rows.append(pd.DataFrame({
            "Dataset": dataset, "Iteration": itr, "Query_Index": cols,
            "conf_max": conf_max, "pool_agree": pool_agree,
            "knn20_acc": knn20_acc,
        }))

        # ---- controls
        row = {"Dataset": dataset, "Iteration": itr}
        static_pred = counts_all.argmax(axis=0)
        row["Static_Majority"] = float((static_pred == y_test).mean())
        for K in K_LIST:
            comp = dsel_corr[:, idx[:, :K]].mean(axis=2)          # (M,Q)
            top = np.argsort(-comp, axis=0, kind="stable")[:TOP]  # (TOP,Q)
            sel = test_pred[top, cols[None, :]]                   # (TOP,Q)
            pred, _ = vote(sel, n_classes)
            row[f"KNN{K}_Top{TOP}"] = float((pred == y_test).mean())
        r_rows.append(row)

    print(f"[{dataset}] done in {time.time() - t0:.0f}s", flush=True)
    return pd.concat(q_rows), pd.DataFrame(r_rows)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with Pool(min(N_WORKERS, len(DATASETS))) as p:
        res = p.map(run_dataset, DATASETS, chunksize=1)
    qf = pd.concat([r[0] for r in res], ignore_index=True)
    cr = pd.concat([r[1] for r in res], ignore_index=True)
    qf.to_csv(f"{OUT}/query_features.csv", index=False)
    cr.to_csv(f"{OUT}/control_runs.csv", index=False)
    print(cr.groupby("Dataset").mean(numeric_only=True).round(4))
    print("\nMean over datasets:")
    print((cr.groupby("Dataset").mean(numeric_only=True).mean() * 100).round(3))
