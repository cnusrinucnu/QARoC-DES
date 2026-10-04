import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

import helpers
from asydes import ASyDES


DATASET = "Laryngeal3"
KNN_SIZE = 20
K = 3
N_ITR = 20


def majority_vote(x):
    values, counts = np.unique(
        x,
        return_counts=True
    )
    return values[np.argmax(counts)]


results = []

print("=" * 80)
print("STEP 13B — FROZEN K=3 ACROSS 20 LARYNGEAL3 ITERATIONS")
print("=" * 80)

for itr in range(N_ITR):

    print(f"\nIteration {itr:02d}")

    data = np.load(
        f"./Experiment/Datasets/Laryngeal3/Laryngeal3{itr}.npz"
    )

    X_DSEL = data["X_DSEL"]
    y_DSEL = data["y_DSEL"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    pool = helpers.load_pool(
        DATASET
    )[itr]

    model = ASyDES(
        pool_classifiers=pool,
        k_neighbors=KNN_SIZE,
        ensemble_size=K
    )

    model.fit(
        X_DSEL,
        y_DSEL
    )

    dsel_predictions = np.asarray(
        model.dsel_predictions
    )

    dsel_probabilities = np.asarray(
        model.dsel_probabilities
    )

    knn = NearestNeighbors(
        n_neighbors=KNN_SIZE
    )

    knn.fit(X_DSEL)

    pred_knn = []
    pred_bade = []

    roc_sizes = []

    for query in X_test:

        query = np.asarray(
            query
        ).reshape(1, -1)

        distances, indices = (
            knn.kneighbors(query)
        )

        distances = distances[0]
        indices = indices[0]

        local_predictions = (
            dsel_predictions[indices]
        )

        local_labels = (
            y_DSEL[indices]
        )

        correct = (
            local_predictions
            ==
            local_labels[:, None]
        )

        query_predictions = np.asarray([
            clf.predict(query)[0]
            for clf in pool
        ])

        query_probabilities = np.asarray([
            clf.predict_proba(query)[0]
            for clf in pool
        ])

        neighbor_conf = (
            dsel_probabilities[
                :,
                indices,
                :
            ]
        )

        roc, fitness = model.optimize_roc(
            distances=distances,
            neighbor_conf=neighbor_conf,
            neighbor_profiles=local_predictions,
            query_conf=query_probabilities,
            query_profile=query_predictions
        )

        roc = np.asarray(
            roc,
            dtype=int
        )

        if len(roc) == 0:
            roc = np.array([0])

        roc_sizes.append(
            len(roc)
        )

        # ----------------------------------------------------------
        # KNN + Top-3 competence
        # ----------------------------------------------------------

        knn_comp = np.mean(
            correct,
            axis=0
        )

        selected_knn = np.argsort(
            -knn_comp,
            kind="stable"
        )[:K]

        pred_knn.append(
            majority_vote(
                query_predictions[
                    selected_knn
                ]
            )
        )

        # ----------------------------------------------------------
        # bADE-RoC + Top-3 competence
        # ----------------------------------------------------------

        roc_correct = correct[
            roc
        ]

        roc_comp = np.mean(
            roc_correct,
            axis=0
        )

        selected_roc = np.argsort(
            -roc_comp,
            kind="stable"
        )[:K]

        pred_bade.append(
            majority_vote(
                query_predictions[
                    selected_roc
                ]
            )
        )

    pred_knn = np.asarray(
        pred_knn
    )

    pred_bade = np.asarray(
        pred_bade
    )

    knn_acc = accuracy_score(
        y_test,
        pred_knn
    )

    knn_f1 = f1_score(
        y_test,
        pred_knn,
        average="macro"
    )

    knn_bal = balanced_accuracy_score(
        y_test,
        pred_knn
    )

    bade_acc = accuracy_score(
        y_test,
        pred_bade
    )

    bade_f1 = f1_score(
        y_test,
        pred_bade,
        average="macro"
    )

    bade_bal = balanced_accuracy_score(
        y_test,
        pred_bade
    )

    results.append([
        itr,
        knn_acc,
        knn_f1,
        knn_bal,
        bade_acc,
        bade_f1,
        bade_bal,
        np.mean(roc_sizes)
    ])

    print(
        f"KNN  : "
        f"Acc={knn_acc:.4f} "
        f"F1={knn_f1:.4f} "
        f"Bal={knn_bal:.4f}"
    )

    print(
        f"bADE : "
        f"Acc={bade_acc:.4f} "
        f"F1={bade_f1:.4f} "
        f"Bal={bade_bal:.4f} "
        f"RoC={np.mean(roc_sizes):.2f}"
    )


results = np.asarray(
    results
)

# --------------------------------------------------------------
# Summary
# --------------------------------------------------------------

knn_acc = results[:, 1]
knn_f1 = results[:, 2]
knn_bal = results[:, 3]

bade_acc = results[:, 4]
bade_f1 = results[:, 5]
bade_bal = results[:, 6]

print("\n" + "=" * 80)
print("20-ITERATION SUMMARY")
print("=" * 80)

print(
    f"KNN  Accuracy : "
    f"{np.mean(knn_acc):.4f} ± {np.std(knn_acc, ddof=1):.4f}"
)

print(
    f"bADE Accuracy : "
    f"{np.mean(bade_acc):.4f} ± {np.std(bade_acc, ddof=1):.4f}"
)

print()

print(
    f"KNN  Macro-F1 : "
    f"{np.mean(knn_f1):.4f} ± {np.std(knn_f1, ddof=1):.4f}"
)

print(
    f"bADE Macro-F1 : "
    f"{np.mean(bade_f1):.4f} ± {np.std(bade_f1, ddof=1):.4f}"
)

print()

print(
    f"KNN  BalAcc   : "
    f"{np.mean(knn_bal):.4f} ± {np.std(knn_bal, ddof=1):.4f}"
)

print(
    f"bADE BalAcc   : "
    f"{np.mean(bade_bal):.4f} ± {np.std(bade_bal, ddof=1):.4f}"
)

print()

print(
    f"Accuracy difference : "
    f"{np.mean(bade_acc-knn_acc):+.4f}"
)

print(
    f"Macro-F1 difference : "
    f"{np.mean(bade_f1-knn_f1):+.4f}"
)

print(
    f"BalAcc difference   : "
    f"{np.mean(bade_bal-knn_bal):+.4f}"
)

print(
    f"Mean RoC size      : "
    f"{np.mean(results[:,7]):.4f}"
)

print(
    f"RoC size SD        : "
    f"{np.std(results[:,7], ddof=1):.4f}"
)

out_dir = (
    "publication_results/ablation/Laryngeal3"
)

os.makedirs(
    out_dir,
    exist_ok=True
)

np.savetxt(
    os.path.join(
        out_dir,
        "step13b_frozen_k3_20iterations.csv"
    ),
    results,
    delimiter=",",
    header=(
        "Iteration,"
        "KNN_Accuracy,KNN_MacroF1,KNN_BalAcc,"
        "bADE_Accuracy,bADE_MacroF1,bADE_BalAcc,"
        "Mean_RoC_Size"
    ),
    comments=""
)

print("\nSaved results to:")
print(out_dir)

print("=" * 80)
print("STEP 13B COMPLETE")
print("=" * 80)
