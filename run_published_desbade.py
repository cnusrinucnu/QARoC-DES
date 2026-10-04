import os
import sys
import time
from multiprocessing import Pool

N_WORKERS = 32

# Prevent each worker from spawning additional BLAS/OpenMP threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import pickle
import json
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

# ------------------------------------------------------------------
# Published DES-bADE implementation
# ------------------------------------------------------------------
BASELINE = os.path.abspath("published_desbade_baseline")
sys.path.insert(0, BASELINE)

from des_mha import DES_MHA


# ------------------------------------------------------------------
# Multiprocessing workers for published DES-bADE prediction
# ------------------------------------------------------------------
_WORKER_MODEL = None
_WORKER_X_TEST = None


def _init_prediction_worker(model, X_test):
    """Initialize one independent fitted DES-bADE model per worker."""
    global _WORKER_MODEL, _WORKER_X_TEST
    _WORKER_MODEL = model
    _WORKER_X_TEST = X_test


def _predict_one_query(q):
    """Predict one test query using the worker-local fitted model."""
    global _WORKER_MODEL, _WORKER_X_TEST

    tq = time.time()

    pred = _WORKER_MODEL.predict(
        _WORKER_X_TEST[q:q+1]
    )

    elapsed_q = time.time() - tq

    pred_value = int(
        np.asarray(pred).reshape(-1)[0]
    )

    return q, pred_value, elapsed_q


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
EXP = os.path.abspath("Experiment")
OUTDIR = os.path.join("publication_results", "published_desbade_baseline")
os.makedirs(OUTDIR, exist_ok=True)

# Use the exact published 30-dataset configuration
import importlib.util

_cfg_path = os.path.join(BASELINE, "config.py")
_spec = importlib.util.spec_from_file_location("published_config", _cfg_path)
_published_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_published_config)

DATASETS = list(_published_config.datasets)
N_ITR = 20
K = 20

RUN_FILE = os.path.join(
    OUTDIR, "published_desbade_run_level.csv"
)

QUERY_FILE = os.path.join(
    OUTDIR, "published_desbade_query_level.csv"
)

STATUS_FILE = os.path.join(
    OUTDIR, "published_desbade_status.json"
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def load_existing_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def append_row(path, row):
    df = pd.DataFrame([row])

    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, mode="w", header=True, index=False)


def save_status(dataset, itr, status, **extra):
    data = {
        "dataset": dataset,
        "iteration": itr,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    data.update(extra)

    with open(STATUS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def completed_runs():
    if not os.path.exists(RUN_FILE):
        return set()

    df = pd.read_csv(RUN_FILE)

    if len(df) == 0:
        return set()

    return set(
        zip(
            df["dataset"].astype(str),
            df["iteration"].astype(int)
        )
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
completed = completed_runs()

print("=" * 80)
print("PUBLISHED DES-bADE BASELINE")
print("=" * 80)
print("Output directory:", OUTDIR)
print("Datasets:", DATASETS)
print("Iterations:", N_ITR)
print("K:", K)
print("Already completed:", len(completed))
print("=" * 80)


for dataset in DATASETS:

    pool_path = os.path.join(
        EXP,
        "Pools",
        dataset,
        f"{dataset}_pools.p"
    )

    if not os.path.exists(pool_path):
        print("ERROR: pool not found:", pool_path)
        continue

    print()
    print("#" * 80)
    print("DATASET:", dataset)
    print("#" * 80)

    # --------------------------------------------------------------
    # Load pool once
    # --------------------------------------------------------------
    print("Loading pool:", pool_path)

    t0 = time.time()

    with open(pool_path, "rb") as f:
        pools = pickle.load(f)

    print(
        "Pool loaded:",
        len(pools),
        "iterations |",
        round(time.time() - t0, 2),
        "sec"
    )

    for itr in range(N_ITR):

        key = (dataset, itr)

        if key in completed:
            print(
                f"[SKIP] {dataset} iteration {itr} "
                "(already completed)"
            )
            continue

        print()
        print("-" * 80)
        print(f"START: {dataset} | iteration {itr}")
        print("-" * 80)

        start_run = time.time()

        try:

            # ------------------------------------------------------
            # Load classifier pool for this iteration
            # ------------------------------------------------------
            pool = pools[itr]

            print("Classifier pool:", len(pool))

            # ------------------------------------------------------
            # Load existing split
            # ------------------------------------------------------
            data_path = os.path.join(
                EXP,
                "Datasets",
                dataset,
                f"{dataset}{itr}.npz"
            )

            data = np.load(data_path, allow_pickle=True)

            X_DSEL = data["X_DSEL"]
            y_DSEL = data["y_DSEL"]

            X_test = data["X_test"]
            y_test = data["y_test"]

            print("DSEL:", X_DSEL.shape)
            print("Test:", X_test.shape)

            # ------------------------------------------------------
            # Initialize published DES-bADE
            # ------------------------------------------------------
            t_fit = time.time()

            model = DES_MHA(
                pool,
                k=K
            )

            model.fit(
                X_DSEL,
                y_DSEL
            )

            fit_time = time.time() - t_fit

            print(
                "Model fit:",
                round(fit_time, 3),
                "sec"
            )

            # ------------------------------------------------------
            # Parallel query prediction using 32 workers
            # ------------------------------------------------------
            predictions = [None] * len(X_test)
            query_records = []

            t_predict = time.time()

            print(
                f"  Parallel prediction: {len(X_test)} queries "
                f"using {N_WORKERS} workers"
            )

            with Pool(
                processes=N_WORKERS,
                initializer=_init_prediction_worker,
                initargs=(model, X_test)
            ) as worker_pool:

                results = worker_pool.map(
                    _predict_one_query,
                    range(len(X_test))
                )

            # Preserve original query order
            results.sort(key=lambda x: x[0])

            for q, pred_value, elapsed_q in results:

                predictions[q] = pred_value

                # --------------------------------------------------
                # Obtain optimized RoC information if available
                # --------------------------------------------------
                roc_size = np.nan
                roc_fitness = np.nan

                try:
                    # Reproduce the published query preparation
                    model.compute_metrics(
                        X_test[q:q+1]
                    )

                    if hasattr(
                        model,
                        "selected_classifiers_indices"
                    ):
                        selected = (
                            model.selected_classifiers_indices
                        )

                        if selected is not None:
                            try:
                                roc_size = len(selected)
                            except Exception:
                                pass

                except Exception:
                    pass

                query_records.append({
                    "dataset": dataset,
                    "iteration": itr,
                    "query": q,
                    "true_label": int(y_test[q]),
                    "prediction": pred_value,
                    "correct": int(pred_value == y_test[q]),
                    "query_time_sec": elapsed_q,
                    "roc_size": roc_size,
                    "roc_fitness": roc_fitness,
                })

            prediction_time = time.time() - t_predict

            print(
                f"  Parallel prediction completed in "
                f"{prediction_time:.2f} sec"
            )


            y_pred = np.asarray(predictions)

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            macro_f1 = f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0
            )

            balanced_acc = balanced_accuracy_score(
                y_test,
                y_pred
            )

            total_time = time.time() - start_run

            # ------------------------------------------------------
            # Run-level result
            # ------------------------------------------------------
            run_row = {
                "dataset": dataset,
                "iteration": itr,
                "n_dsel": len(X_DSEL),
                "n_test": len(X_test),
                "n_classifiers": len(pool),
                "k": K,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "balanced_accuracy": balanced_acc,
                "fit_time_sec": fit_time,
                "prediction_time_sec": prediction_time,
                "total_time_sec": total_time,
                "mean_query_time_sec":
                    prediction_time / len(X_test),
            }

            append_row(
                RUN_FILE,
                run_row
            )

            # ------------------------------------------------------
            # Query-level results
            # ------------------------------------------------------
            for qr in query_records:
                append_row(
                    QUERY_FILE,
                    qr
                )

            completed.add(key)

            save_status(
                dataset,
                itr,
                "completed",
                accuracy=float(accuracy),
                macro_f1=float(macro_f1),
                balanced_accuracy=float(balanced_acc),
            )

            print()
            print(
                f"[DONE] {dataset} | iteration {itr}"
            )
            print(
                f"Accuracy       : {accuracy:.6f}"
            )
            print(
                f"Macro-F1       : {macro_f1:.6f}"
            )
            print(
                f"Balanced Acc.  : {balanced_acc:.6f}"
            )
            print(
                f"Mean query time: "
                f"{prediction_time / len(X_test):.4f} sec"
            )
            print(
                f"Total time     : {total_time:.2f} sec"
            )

        except Exception as e:

            print()
            print(
                f"[ERROR] {dataset} | iteration {itr}"
            )
            print(
                repr(e)
            )

            traceback.print_exc()

            save_status(
                dataset,
                itr,
                "failed",
                error=repr(e)
            )

            # Continue with next iteration
            continue


print()
print("=" * 80)
print("PUBLISHED DES-bADE RUN FINISHED")
print("=" * 80)
print("Run-level results:", RUN_FILE)
print("Query-level results:", QUERY_FILE)
print("=" * 80)
