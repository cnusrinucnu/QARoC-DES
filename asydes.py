import os
import time
import pickle
import numpy as np
from scipy.special import erf

from multiprocessing import Pool

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score

from scipy.spatial.distance import pdist

from mealpy import Problem
from mealpy import FloatVar

from ADE import ADE


# ============================================================================
# ASyDES
# Adaptive Synergy-aware Dynamic Ensemble Selection
# ============================================================================

class ASyDESRoCProblem(Problem):

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

        self.k = (
            self.neighbor_profiles.shape[0]
        )

        super().__init__(
            **kwargs
        )

    # ========================================================================
    # Deterministic continuous-to-binary decoding
    # ========================================================================

    def amend_position(
        self,
        solution
    ):

        x = np.asarray(
            solution,
            dtype=float
        )

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
            -tf
        )[:k]

        mask = np.zeros(
            len(tf),
            dtype=int
        )

        mask[selected] = 1

        return mask

    # ========================================================================
    # Exact deterministic RoC objective
    # ========================================================================

    def obj_func(
        self,
        solution
    ):

        competence_region = (
            self.amend_position(
                solution
            )
        )

        # --------------------------------------------------------------------
        # Distance
        # --------------------------------------------------------------------

        avg_dist = (
            1.0
            -
            np.mean(
                self.neighbor_distances[
                    competence_region
                ]
            )
        )

        # --------------------------------------------------------------------
        # Confidence similarity
        # --------------------------------------------------------------------

        confidence_values = []

        for i in competence_region:

            similarity = np.sum(
                self.query_conf
                *
                self.neighbor_conf[
                    :,
                    i,
                    :
                ],
                axis=1
            )

            confidence_values.append(
                np.mean(
                    similarity
                )
            )

        avg_conf = np.mean(
            confidence_values
        )

        # --------------------------------------------------------------------
        # Output-profile similarity
        #
        # IMPORTANT:
        # The prototype uses SUM of matching classifier outputs.
        # It is normalized later by n_classifiers.
        # --------------------------------------------------------------------

        profile_values = []

        for i in competence_region:

            profile_values.append(
                np.sum(
                    self.query_profile
                    ==
                    self.neighbor_profiles[
                        i
                    ]
                )
            )

        avg_profile = np.mean(
            profile_values
        )

        # --------------------------------------------------------------------
        # Diversity
        #
        # IMPORTANT:
        # The prototype uses Euclidean pairwise distance.
        # --------------------------------------------------------------------

        selected_profiles = (
            self.neighbor_profiles[
                competence_region
            ]
        )

        if len(
            selected_profiles
        ) > 1:

            from sklearn.metrics import pairwise_distances

            avg_div = np.mean(
                pairwise_distances(
                    selected_profiles,
                    metric="euclidean"
                )
            )

        else:

            avg_div = 0.0

        n_classifiers = (
            selected_profiles.shape[1]
        )

        # --------------------------------------------------------------------
        # Normalization
        # --------------------------------------------------------------------

        norm_avg_dist = avg_dist

        norm_avg_conf = (
            avg_conf
            /
            n_classifiers
        )

        norm_avg_profile = (
            avg_profile
            /
            n_classifiers
        )

        norm_avg_div = (
            avg_div
            /
            np.sqrt(
                n_classifiers
            )
        )

        # --------------------------------------------------------------------
        # Objective weights
        # --------------------------------------------------------------------

        w1 = 0.1
        w2 = 0.2
        w3 = 0.3
        w4 = 0.4

        fitness = (
            w1 * norm_avg_dist
            +
            w2 * norm_avg_conf
            +
            w3 * norm_avg_profile
            +
            w4 * norm_avg_div
        )

        return float(
            fitness
        )

class ASyDES:

    def __init__(
        self,
        pool_classifiers,
        k_neighbors=20,
        ensemble_size=10,
        ade_epoch=300,
        ade_pop_size=50,
        ade_early_stop=300
    ):

        self.pool_classifiers = list(
            pool_classifiers
        )

        self.n_classifiers = len(
            self.pool_classifiers
        )

        self.k_neighbors = (
            k_neighbors
        )

        self.ensemble_size = (
            ensemble_size
        )

        self.ade_epoch = (
            ade_epoch
        )

        self.ade_pop_size = (
            ade_pop_size
        )

        self.ade_early_stop = (
            ade_early_stop
        )

        self.nn = NearestNeighbors(
            n_neighbors=k_neighbors
        )

        self.X_DSEL = None
        self.y_DSEL = None

        self.dsel_predictions = None
        self.dsel_probabilities = None

    # =========================================================================
    # FIT
    # =========================================================================

    def fit(
        self,
        X_DSEL,
        y_DSEL
    ):

        self.X_DSEL = np.asarray(
            X_DSEL
        )

        self.y_DSEL = np.asarray(
            y_DSEL
        )

        self.nn.fit(
            self.X_DSEL
        )

        print(
            "Computing DSEL predictions..."
        )

        self.dsel_predictions = np.array([
            clf.predict(
                self.X_DSEL
            )
            for clf in self.pool_classifiers
        ]).T

        self.dsel_probabilities = np.array([
            clf.predict_proba(
                self.X_DSEL
            )
            for clf in self.pool_classifiers
        ])

        return self

    # =========================================================================
    # bADE RoC
    # =========================================================================

    def optimize_roc(
        self,
        distances,
        neighbor_conf,
        neighbor_profiles,
        query_conf,
        query_profile
    ):

        problem = ASyDESRoCProblem(

            neighbor_distances=distances,

            neighbor_conf=neighbor_conf,

            neighbor_profiles=neighbor_profiles,

            query_conf=query_conf,

            query_profile=query_profile,

            bounds=FloatVar(
                [0] * self.k_neighbors,
                [1] * self.k_neighbors
            ),

            name="ASyDES_RoC",

            minmax="max",

            log_to=None
        )

        model = ADE(
            epoch=self.ade_epoch,
            pop_size=self.ade_pop_size
        )

        g_best = model.solve(
            problem,
            termination={
                "max_early_stop":
                self.ade_early_stop
            }
        )

        mask = problem.amend_position(
            g_best.solution
        )

        roc = np.where(
            mask == 1
        )[0]

        return (
            roc,
            float(
                g_best.target.fitness
            )
        )

    # =========================================================================
    # GREEDY MARGINAL SYNERGY
    # =========================================================================

    @staticmethod
    def greedy_synergy(
        roc_correct,
        ensemble_size
    ):

        n_neighbors, n_classifiers = (
            roc_correct.shape
        )

        ensemble_size = min(
            ensemble_size,
            n_classifiers
        )

        selected = []

        remaining = set(
            range(n_classifiers)
        )

        coverage = np.zeros(
            n_neighbors,
            dtype=bool
        )

        gains = []

        for _ in range(
            ensemble_size
        ):

            best_classifier = None
            best_gain = -np.inf

            for clf_idx in remaining:

                new_coverage = (
                    coverage
                    |
                    roc_correct[
                        :,
                        clf_idx
                    ]
                )

                gain = (
                    np.mean(
                        new_coverage
                    )
                    -
                    np.mean(
                        coverage
                    )
                )

                if (
                    gain > best_gain
                ):

                    best_gain = gain

                    best_classifier = (
                        clf_idx
                    )

            if best_classifier is None:
                break

            selected.append(
                best_classifier
            )

            gains.append(
                best_gain
            )

            coverage = (
                coverage
                |
                roc_correct[
                    :,
                    best_classifier
                ]
            )

            remaining.remove(
                best_classifier
            )

        return (
            np.asarray(
                selected,
                dtype=int
            ),
            np.asarray(
                gains,
                dtype=float
            )
        )

    # =========================================================================
    # PREDICT ONE QUERY
    # =========================================================================

    def predict_one(
        self,
        X
    ):

        query = np.asarray(
            X
        ).reshape(
            1,
            -1
        )

        # --------------------------------------------------------------
        # Neighbors
        # --------------------------------------------------------------

        distances, indices = (
            self.nn.kneighbors(
                query
            )
        )

        distances = distances[0]
        indices = indices[0]

        # --------------------------------------------------------------
        # Local predictions
        # --------------------------------------------------------------

        local_predictions = (
            self.dsel_predictions[
                indices
            ]
        )

        local_labels = (
            self.y_DSEL[
                indices
            ]
        )

        correct = (
            local_predictions
            ==
            local_labels[:, None]
        )

        # --------------------------------------------------------------
        # Query predictions / probabilities
        # --------------------------------------------------------------

        query_predictions = np.array([
            clf.predict(
                query
            )[0]
            for clf in self.pool_classifiers
        ])

        query_probabilities = np.array([
            clf.predict_proba(
                query
            )[0]
            for clf in self.pool_classifiers
        ])

        # --------------------------------------------------------------
        # Neighbor probabilities
        #
        # classifier × neighbor × class
        # --------------------------------------------------------------

        neighbor_conf = (
            self.dsel_probabilities[
                :,
                indices,
                :
            ]
        )

        # --------------------------------------------------------------
        # Neighbor profiles
        #
        # neighbor × classifier
        # --------------------------------------------------------------

        neighbor_profiles = (
            local_predictions
        )

        # --------------------------------------------------------------
        # Query profile
        # --------------------------------------------------------------

        query_profile = (
            query_predictions
        )

        # --------------------------------------------------------------
        # bADE competence region
        # --------------------------------------------------------------

        roc, fitness = (
            self.optimize_roc(

                distances=distances,

                neighbor_conf=neighbor_conf,

                neighbor_profiles=neighbor_profiles,

                query_conf=query_probabilities,

                query_profile=query_profile
            )
        )

        # --------------------------------------------------------------
        # Restrict competence to optimized RoC
        # --------------------------------------------------------------

        roc_correct = (
            correct[
                roc
            ]
        )

        # --------------------------------------------------------------
        # Greedy synergy
        # --------------------------------------------------------------

        selected, gains = (
            self.greedy_synergy(
                roc_correct,
                self.ensemble_size
            )
        )

        # --------------------------------------------------------------
        # Majority voting
        # --------------------------------------------------------------

        if len(selected) == 0:

            # Safety fallback
            best_classifier = int(
                np.argmax(
                    np.mean(
                        roc_correct,
                        axis=0
                    )
                )
            )

            selected = np.array([
                best_classifier
            ])

        selected_predictions = (
            query_predictions[
                selected
            ]
        )

        classes, counts = np.unique(
            selected_predictions,
            return_counts=True
        )

        prediction = classes[
            np.argmax(
                counts
            )
        ]

        return {
            "prediction": prediction,
            "selected": selected,
            "gains": gains,
            "roc": roc,
            "roc_fitness": fitness
        }

    # =========================================================================
    # PREDICT
    # =========================================================================

    def predict(
        self,
        X
    ):

        X = np.asarray(
            X
        )

        if X.ndim == 1:
            X = X.reshape(
                1,
                -1
            )

        predictions = []

        details = []

        for i, x in enumerate(X):

            result = self.predict_one(
                x
            )

            predictions.append(
                result["prediction"]
            )

            details.append(
                result
            )

        self.last_details = details

        return np.asarray(
            predictions
        )

    # =========================================================================
    # SCORE
    # =========================================================================

    def score(
        self,
        X_test,
        y_test
    ):

        y_pred = self.predict(
            X_test
        )

        return accuracy_score(
            y_test,
            y_pred
        )
