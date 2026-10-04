from imports import *
from ADE import ADE
from TransferFunctions import TransferFunctions 
np.random.seed(12345)

class MyProblem(Problem):
    def __init__(self, data, **kwargs):
        self.tf = TransferFunctions()
        self.X_test_conf = data["X_test_conf"]
        self.X_test_profile = data["X_test_profile"]
        self.neighbor_indices = data["neighbor_indices"]
        self.neighbor_distances = data["neighbor_distances"]
        self.neighbor_profiles = data["neighbor_profiles"]
        self.neighbor_conf = data["neighbor_conf"]
        super().__init__(**kwargs)

    def amend_position(self, solution):
        tf_solution = self.tf.vstf_01(solution)
        mask = np.random.uniform(0, 1, len(tf_solution)) < tf_solution
        if np.sum(mask) == 0:  # Ensure at least one classifier is selected
            mask[np.random.randint(0, len(mask))] = 1
        return mask.astype(int)

    def obj_func(self, solution):
        mask = self.amend_position(solution)
        competence_region = np.where(mask == 1)[0]

        # Compute metrics for the competence region
        avg_dist = 1 - np.mean(self.neighbor_distances[0][competence_region])
        avg_conf = np.mean([np.sum(self.X_test_conf[0] * self.neighbor_conf[0][i])for i in competence_region])
        avg_profile = np.mean([np.sum(self.X_test_profile[0] == self.neighbor_profiles[0][i]) for i in competence_region])
        selected_profiles = self.neighbor_profiles[0][competence_region]
        avg_div = np.mean(pairwise_distances(selected_profiles, metric='euclidean'))
        
        # Static normalization
        n_classifiers = selected_profiles.shape[1]
        norm_avg_dist = avg_dist  # Already in range [0, 1]
        norm_avg_conf = avg_conf / n_classifiers
        norm_avg_profile = avg_profile / n_classifiers
        norm_avg_div = avg_div / np.sqrt(n_classifiers)  # Normalize based on feature dimension
        
        # Weights for the fitness components
        w1, w2, w3, w4 = 0.1, 0.2, 0.3, 0.4
        fitness = w1 * norm_avg_dist + w2 * norm_avg_conf + w3 * norm_avg_profile + w4 * norm_avg_div
        #fitness = w2 * avg_conf + w3 * avg_profile 

        return fitness
    
class DES_MHA:
    def __init__(self, pool_classifiers, k=20):
        self.pool_classifiers = pool_classifiers
        self.n_classifiers = len(self.pool_classifiers)
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=self.k)
        self.desknn = DESKNN(self.pool_classifiers, k=self.k)

        self.X_DSEL = None
        self.y_DSEL = None
        self.predictions = None
        self.probabilities = None
        self.KNNs_indices = None
        self.KNNs_distances = None
        self.KNNs_labels = None
        self.KNNs_predictions = None
        self.KNNs_probabilities = None
        self.selected_classifiers_indices = None

    def fit(self, X_DSEL, y_DSEL):
        self.X_DSEL = X_DSEL
        self.y_DSEL = y_DSEL
        self.nn.fit(X_DSEL)
        self.desknn.fit(X_DSEL, y_DSEL)

        self.probabilities = np.array([clf.predict_proba(self.X_DSEL) for clf in self.pool_classifiers])
        self.probabilities = np.transpose(self.probabilities, (1, 0, 2))  # Reshape to (n_samples, n_classifiers, n_classes)
        self.predictions = np.array([clf.predict(self.X_DSEL) for clf in self.pool_classifiers])
        self.predictions = np.transpose(self.predictions)

    def compute_metrics(self, X_test):
        # Find the k-nearest neighbors for the test sample
        neighbor_distances, neighbor_indices = self.nn.kneighbors(X_test)
        neighbor_labels = self.y_DSEL[neighbor_indices]
        neighbor_profiles = self.predictions[neighbor_indices]
        neighbor_conf = self.probabilities[neighbor_indices]

        # **Method 1: Test-Sample Based **
        # Uncomment this block for Test-Sample Based method
        # -------------------------------------------------
        X_test_conf = np.array([clf.predict_proba(X_test) for clf in self.pool_classifiers])
        X_test_conf = np.transpose(X_test_conf, (1, 0, 2))  # Reshape to (1, n_classifiers, n_classes)
        X_test_profile = np.array([clf.predict(X_test) for clf in self.pool_classifiers])
        # -------------------------------------------------
        # **Method 2: Neighbor-Averaged **
        # Uncomment this block for Neighbor-Averaged method
        # -------------------------------------------------
        #X_test_conf = np.mean(neighbor_conf, axis=1)  # Averaged confidence
        #X_test_profile = np.mean(neighbor_profiles, axis=1)  # Averaged profile
        # -------------------------------------------------
        return {
            "X_test_conf": X_test_conf,
            "X_test_profile": X_test_profile,
            "neighbor_indices": neighbor_indices,
            "neighbor_distances": neighbor_distances,
            "neighbor_labels": neighbor_labels,
            "neighbor_profiles": neighbor_profiles,
            "neighbor_conf": neighbor_conf,
        }

    def amend_position(self, solution):
        mask = np.random.uniform(0, 1, len(solution)) < solution
        if np.sum(mask) == 0:  # Ensure at least one classifier is selected
            mask[np.random.randint(0, len(mask))] = 1
        return mask.astype(int)

    def estimate_competence(self, data):
        problem = MyProblem(
            bounds=FloatVar([0] * self.k, [1] * self.k),
            data = data,
            name="DES",
            minmax = "max",
            log_to = None # None,"console"
        )
        term_dict = {"max_early_stop": 300 } # after 30 epochs, if the global best doesn't improve then we stop the program
        model = ADE(epoch=300, pop_size=50)  # Metaheuristic optimization
        g_best = model.solve(problem, termination=term_dict)
        #g_best = model.solve(problem)

        mask = self.amend_position(g_best.solution)
        competence_region = np.where(mask == 1)[0]
        return competence_region

    def predict(self, X_test):
        #t1 = time.time()
        X_test = X_test if X_test.ndim > 1 else X_test.reshape(1, -1)
        data = self.compute_metrics(X_test)
        competence_region = self.estimate_competence(data)
        competence_region = data["neighbor_indices"][:,competence_region]
        competences, diversity = self.desknn.estimate_competence(
            competence_region=competence_region,
            predictions=self.predictions.reshape(-1, self.n_classifiers))
        selected_classifiers = self.desknn.select(competences, diversity)
        final_classifiers = [self.pool_classifiers[i] for i in selected_classifiers[0]]
        final_predictions = [clf.predict(X_test)[0] for clf in final_classifiers]
        majority_votes = mode(final_predictions)[0]
        #print(f"Predicting time = {time.time()-t1}")
        return majority_votes

    def score(self, X_test, y_test):
        with Pool(processes=32) as pool:
            predictions = pool.map(self.predict, X_test)
        predictions = [int(p) for p in predictions]
        # Calculate accuracy
        acc = accuracy_score(y_test, predictions)
        print(f'accuracy = {acc}')
        
        return acc, predictions
