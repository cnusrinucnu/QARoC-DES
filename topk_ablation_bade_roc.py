import os
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.special import erf
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)

from mealpy import Problem, FloatVar
from ADE import ADE

import helpers


# ============================================================
# CONFIGURATION
# ============================================================

DATASETS = [
    "Adult", "Banknote", "Blood", "Breast",
    "Cardiotocography", "Faults", "German", "Glass",
    "Haberman", "Heart", "ILPD", "Ionosphere",
    "Iris", "Laryngeal3", "Liver", "Magic",
    "Mammographic", "Monk2", "Pima", "Sonar",
    "Statlog", "Steel", "Thyroid", "Transfusion",
    "Vehicle", "Vertebral", "Voice3", "WDVG1",
    "Weaning", "Wine"
]

KNN_SIZE = 20
N_ITR = 20
N_WORKERS = 32

# Controlled Top-K values.
TOP_K_VALUES = [1, 3, 5, 7]

# Same frozen bADE configuration.
W_DISTANCE = 1.0 / 6.0
W_CONFIDENCE = 1.0 / 3.0
W_PROFILE = 1.0 / 2.0
W_DIVERSITY = 0.0

ADE_EPOCH = 300
ADE_POP = 50
ADE_EARLY_STOP = 300

OUT_DIR = Path(
    "publication_results/topk_ablation"
)

RUN_LEVEL_FILE = (
    OUT_DIR / "topk_ablation_run_level.csv"
)

QUERY_LEVEL_FILE = (
    OUT_DIR / "topk_ablation_query_level.csv"
)

SUMMARY_FILE = (
    OUT_DIR / "topk_ablation_dataset_summary.csv"
)


# ============================================================
# DETERMINISTIC RoC PROBLEM
# ============================================================

class TopKRoCProblem(Problem):

    def __init__(
        self,
        neighbor_distances,
        neighbor_conf,
        neighbor_profiles,
        query_conf,
        query_profile,
        **kwargs
    ):

        self.neighbor_distances = np.asarray(
            neighbor_distances,
            dtype=float
        )

        self.neighbor_conf = np.asarray(
            neighbor_conf,
            dtype=float
        )

        self.neighbor_profiles = np.asarray(
            neighbor_profiles
        )

        self.query_conf = np.asarray(
            query_conf,
            dtype=float
        )

        self.query_profile = np.asarray(
            query_profile
        )

        self.w1 = W_DISTANCE
        self.w2 = W_CONFIDENCE
        self.w3 = W_PROFILE
        self.w4 = W_DIVERSITY

        super().__init__(**kwargs)

    def amend_position(self, solution):

        x = np.asarray(
            solution,
            dtype=float
        )

        # Exact deterministic V-shaped transfer
        tf = np.abs(
            erf(
                (np.sqrt(np.pi) / 2.0) * x
            )
        )

        k = int(
            np.round(
                np.sum(tf)
            )
        )

        k = max(
            1,
            min(
                len(tf),
                k
            )
        )

        selected = np.argsort(
            -tf,
            kind="stable"
        )[:k]

        mask = np.zeros(
            len(tf),
            dtype=int
        )

        mask[selected] = 1

        return mask

    def obj_func(self, solution):

        mask = self.amend_position(
            solution
        )

        roc = np.where(mask == 1)[0]

        # -----------------------------
        # Distance
        # -----------------------------
        avg_dist = (
            1.0
            -
            np.mean(
                self.neighbor_distances[roc]
            )
        )

        # -----------------------------
        # Confidence similarity
        # -----------------------------
        confidence_values = []

        for i in roc:

            similarity = np.sum(
                self.query_conf
                *
                self.neighbor_conf[:, i, :],
                axis=1
            )

            confidence_values.append(
                np.mean(similarity)
            )

        avg_conf = np.mean(
            confidence_values
        )

        # -----------------------------
        # Output-profile similarity
        # -----------------------------
        profile_values = []

        for i in roc:

            profile_values.append(
                np.sum(
                    self.query_profile
                    ==
                    self.neighbor_profiles[i]
                )
            )

        avg_profile = np.mean(
            profile_values
        )

        n_classifiers = (
            self.neighbor_profiles.shape[1]
        )

        norm_avg_dist = avg_dist

        norm_avg_conf = (
            avg_conf /
            n_classifiers
        )

        norm_avg_profile = (
            avg_profile /
            n_classifiers
        )

        # Diversity deliberately excluded.
        norm_avg_div = 0.0

        fitness = (
            self.w1 * norm_avg_dist
            +
            self.w2 * norm_avg_conf
            +
            self.w3 * norm_avg_profile
            +
            self.w4 * norm_avg_div
        )

        return float(fitness)


# ============================================================
# MODEL
# ============================================================

class TopKBADEModel:

    def __init__(
        self,
        pool_classifiers
    ):

        self.pool_classifiers = (
            pool_classifiers
        )

        self.nn = NearestNeighbors(
            n_neighbors=KNN_SIZE
        )

    def fit(
        self,
        X_DSEL,
        y_DSEL
    ):

        self.X_DSEL = X_DSEL
        self.y_DSEL = y_DSEL

        self.nn.fit(
            X_DSEL
        )

        # Shape:
        # samples × classifiers
        self.dsel_predictions = np.asarray([
            clf.predict(X_DSEL)
            for clf in self.pool_classifiers
        ]).T

        # Shape:
        # classifiers × samples × classes
        self.dsel_probabilities = np.stack([
            clf.predict_proba(X_DSEL)
            for clf in self.pool_classifiers
        ])

        return self

    def optimize_one_query(
        self,
        query
    ):

        query = np.asarray(
            query
        ).reshape(1, -1)

        # ----------------------------------------------------
        # Candidate 20-NN region
        # ----------------------------------------------------

        distances, indices = (
            self.nn.kneighbors(
                query,
                n_neighbors=KNN_SIZE
            )
        )

        distances = distances[0]
        indices = indices[0]

        # ----------------------------------------------------
        # Local classifier predictions
        # ----------------------------------------------------

        local_predictions = (
            self.dsel_predictions[
                indices
            ]
        )

        local_probabilities = np.stack([
            self.dsel_probabilities[
                clf_idx,
                indices,
                :
            ]
            for clf_idx in range(
                len(self.pool_classifiers)
            )
        ])

        query_probabilities = np.stack([
            clf.predict_proba(query)[0]
            for clf in self.pool_classifiers
        ])

        query_predictions = np.asarray([
            clf.predict(query)[0]
            for clf in self.pool_classifiers
        ])

        neighbor_profiles = (
            local_predictions
        )

        query_profile = (
            query_predictions
        )

        # ----------------------------------------------------
        # bADE RoC optimization
        # ----------------------------------------------------

        problem = TopKRoCProblem(
            neighbor_distances=distances,
            neighbor_conf=local_probabilities,
            neighbor_profiles=neighbor_profiles,
            query_conf=query_probabilities,
            query_profile=query_profile,

            bounds=FloatVar(
                [0] * KNN_SIZE,
                [1] * KNN_SIZE
            ),

            name="TopK_Ablation_bADE_RoC",

            minmax="max",

            log_to=None
        )

        model = ADE(
            epoch=ADE_EPOCH,
            pop_size=ADE_POP
        )

        g_best = model.solve(
            problem,
            termination={
                "max_early_stop":
                ADE_EARLY_STOP
            }
        )

        # ----------------------------------------------------
        # Deterministic chromosome → RoC
        # ----------------------------------------------------

        mask = problem.amend_position(
            g_best.solution
        )

        roc = np.where(
            mask == 1
        )[0]

        fitness = float(
            g_best.target.fitness
        )

        # ----------------------------------------------------
        # Local competence
        # ----------------------------------------------------

        local_y = (
            self.y_DSEL[
                indices
            ]
        )

        roc_correct = (
            local_predictions[roc]
            ==
            local_y[roc, None]
        )

        competence = np.mean(
            roc_correct,
            axis=0
        )

        return {
            "roc": roc,
            "fitness": fitness,
            "competence": competence,
            "query_predictions":
                query_predictions,
            "query_indices":
                indices,
        }


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(
    dataset,
    itr
):

    filename = (
        f"./Experiment/Datasets/"
        f"{dataset}/{dataset}{itr}.npz"
    )

    data = np.load(
        filename
    )

    return (
        data["X_DSEL"],
        data["y_DSEL"],
        data["X_test"],
        data["y_test"]
    )


# ============================================================
# MAJORITY VOTE
# ============================================================

def majority_vote(
    predictions
):

    predictions = np.asarray(
        predictions
    )

    values, counts = np.unique(
        predictions,
        return_counts=True
    )

    return values[
        np.argmax(counts)
    ]


# ============================================================
# PARALLEL QUERY WORKER
# ============================================================

_WORKER_MODEL = None
_WORKER_DATASET = None
_WORKER_ITR = None


def _init_query_worker(
    pool,
    X_DSEL,
    y_DSEL,
    dataset,
    itr
):
    global _WORKER_MODEL
    global _WORKER_DATASET
    global _WORKER_ITR

    _WORKER_MODEL = TopKBADEModel(
        pool_classifiers=pool
    )

    _WORKER_MODEL.fit(
        X_DSEL,
        y_DSEL
    )

    _WORKER_DATASET = dataset
    _WORKER_ITR = itr


def _process_query(args):
    q_idx, query = args

    q_start = time.time()

    result = _WORKER_MODEL.optimize_one_query(
        query
    )

    roc = result["roc"]
    fitness = result["fitness"]
    competence = result["competence"]
    query_predictions = result["query_predictions"]

    ranking = np.argsort(
        -competence,
        kind="stable"
    )

    predictions = {}

    for k in TOP_K_VALUES:

        selected = ranking[:k]

        predictions[k] = majority_vote(
            query_predictions[selected]
        )

    q_time = time.time() - q_start

    record = {
        "Dataset": _WORKER_DATASET,
        "Iteration": _WORKER_ITR,
        "Query_Index": q_idx,
        "RoC_Size": len(roc),
        "RoC_Fitness": fitness,
        "Top1_Selected": int(ranking[0]),
        "Top3_Selected": ",".join(
            map(str, ranking[:3])
        ),
        "Top5_Selected": ",".join(
            map(str, ranking[:5])
        ),
        "Top7_Selected": ",".join(
            map(str, ranking[:7])
        ),
        "Optimization_Time_Sec": q_time
    }

    return predictions, record


# ============================================================
# ONE DATASET / ONE ITERATION
# ============================================================

def run_iteration(
    dataset,
    itr
):

    print(
        f"\nDataset={dataset} | "
        f"Iteration={itr}"
    )

    (
        X_DSEL,
        y_DSEL,
        X_test,
        y_test
    ) = load_dataset(
        dataset,
        itr
    )

    pool = helpers.load_pool(
        dataset
    )[itr]

    start_time = time.time()

    # One model initialization per worker process.
    ctx = mp.get_context("fork")

    predictions = {
        k: [None] * len(X_test)
        for k in TOP_K_VALUES
    }

    query_records = [None] * len(X_test)

    tasks = list(
        enumerate(X_test)
    )

    with ctx.Pool(
        processes=N_WORKERS,
        initializer=_init_query_worker,
        initargs=(
            pool,
            X_DSEL,
            y_DSEL,
            dataset,
            itr
        )
    ) as executor:

        for completed_q, (pred_dict, record) in enumerate(
            executor.imap(
                _process_query,
                tasks,
                chunksize=1
            ),
            start=1
        ):

            q_idx = record["Query_Index"]

            query_records[q_idx] = record

            for k in TOP_K_VALUES:
                predictions[k][q_idx] = pred_dict[k]

            if (
                completed_q % 25 == 0
                or
                completed_q == len(X_test)
            ):
                print(
                    f"  queries: "
                    f"{completed_q}/{len(X_test)}"
                )

    elapsed = time.time() - start_time

    results = []

    for k in TOP_K_VALUES:

        pred = np.asarray(
            predictions[k]
        )

        results.append({
            "Dataset": dataset,
            "Iteration": itr,
            "TopK": k,
            "Accuracy":
                accuracy_score(
                    y_test,
                    pred
                ),
            "MacroF1":
                f1_score(
                    y_test,
                    pred,
                    average="macro"
                ),
            "BalancedAccuracy":
                balanced_accuracy_score(
                    y_test,
                    pred
                ),
            "Mean_RoC_Size":
                np.mean([
                    r["RoC_Size"]
                    for r in query_records
                ]),
            "Mean_RoC_Fitness":
                np.mean([
                    r["RoC_Fitness"]
                    for r in query_records
                ]),
            "Mean_Query_Time_Sec":
                np.mean([
                    r["Optimization_Time_Sec"]
                    for r in query_records
                ]),
            "Total_Iteration_Time_Sec":
                elapsed
        })

    return results, query_records


# ============================================================
# SAVE HELPERS
# ============================================================

def append_csv(
    rows,
    filename
):

    if not rows:
        return

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.DataFrame(
        rows
    )

    if filename.exists():

        df.to_csv(
            filename,
            mode="a",
            header=False,
            index=False
        )

    else:

        df.to_csv(
            filename,
            index=False
        )


# ============================================================
# MAIN
# ============================================================

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 90)
    print("bADE-RoC TOP-K CONTROLLED ABLATION")
    print("=" * 90)
    print(
        f"Datasets      : {len(DATASETS)}"
    )
    print(
        f"Iterations    : {N_ITR}"
    )
    print(
        f"Top-K values  : {TOP_K_VALUES}"
    )
    print(
        f"Workers       : {N_WORKERS}"
    )
    print(
        "One bADE optimization per query; "
        "same RoC reused for K=1,3,5,7."
    )
    print(
        "Resume mode   : ENABLED"
    )
    print(
        f"Output        : {OUT_DIR}"
    )
    print("=" * 90)

    # --------------------------------------------------------
    # Determine already completed dataset/iteration pairs.
    # --------------------------------------------------------

    completed_pairs = set()

    if RUN_LEVEL_FILE.exists():

        existing = pd.read_csv(
            RUN_LEVEL_FILE
        )

        if not existing.empty:

            for _, row in existing.iterrows():

                completed_pairs.add(
                    (
                        str(row["Dataset"]),
                        int(row["Iteration"])
                    )
                )

        print(
            f"\nExisting completed "
            f"dataset/iteration pairs: "
            f"{len(completed_pairs)}"
        )

    total = (
        len(DATASETS)
        *
        N_ITR
    )

    completed_count = len(
        completed_pairs
    )

    print(
        f"Remaining dataset/iteration pairs: "
        f"{total - completed_count}"
    )

    # --------------------------------------------------------
    # Run only missing dataset/iteration pairs.
    # --------------------------------------------------------

    for dataset in DATASETS:

        for itr in range(N_ITR):

            pair = (
                dataset,
                itr
            )

            if pair in completed_pairs:

                print(
                    f"\nSKIP: "
                    f"{dataset} iteration {itr} "
                    f"(already completed)"
                )

                continue

            completed_count += 1

            print(
                f"\n[{completed_count}/{total}] "
                f"{dataset} iteration {itr}"
            )

            results, queries = run_iteration(
                dataset,
                itr
            )

            # Save immediately after each completed
            # dataset/iteration pair.
            append_csv(
                results,
                RUN_LEVEL_FILE
            )

            append_csv(
                queries,
                QUERY_LEVEL_FILE
            )

            completed_pairs.add(
                pair
            )

            print(
                "Completed:",
                dataset,
                itr
            )

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    if RUN_LEVEL_FILE.exists():

        df = pd.read_csv(
            RUN_LEVEL_FILE
        )

        summary = (
            df
            .groupby(
                "TopK",
                as_index=False
            )
            .agg({
                "Accuracy": [
                    "mean",
                    "std"
                ],
                "MacroF1": [
                    "mean",
                    "std"
                ],
                "BalancedAccuracy": [
                    "mean",
                    "std"
                ],
                "Mean_RoC_Size":
                    "mean",
                "Mean_RoC_Fitness":
                    "mean",
                "Mean_Query_Time_Sec":
                    "mean",
                "Total_Iteration_Time_Sec":
                    "mean"
            })
        )

        summary.columns = [
            "TopK",
            "Accuracy_Mean",
            "Accuracy_SD",
            "MacroF1_Mean",
            "MacroF1_SD",
            "BalancedAccuracy_Mean",
            "BalancedAccuracy_SD",
            "Mean_RoC_Size",
            "Mean_RoC_Fitness",
            "Mean_Query_Time_Sec",
            "Mean_Iteration_Time_Sec"
        ]

        summary.to_csv(
            SUMMARY_FILE,
            index=False
        )

        print(
            "\nDataset summary saved to:"
        )

        print(
            SUMMARY_FILE
        )

        print(
            "\nTOP-K SUMMARY"
        )

        print(
            summary.to_string(
                index=False
            )
        )

    print(
        "\n" + "=" * 90
    )

    print(
        "TOP-K ABLATION COMPLETE"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()
