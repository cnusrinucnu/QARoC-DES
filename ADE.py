from mealpy import Optimizer
import numpy as np

class ADE(Optimizer):
    """
    This is an example how to build new optimizer
    """
    def __init__(self, epoch=10000, pop_size=100, NPopmin=4, pCR=0.75, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.NPopmin = self.validator.check_int("NPopmin", NPopmin, [4, int(self.pop_size/2)])
        self.pCR = self.validator.check_float("pCR", pCR, (0, 1.0))
        self.lb, self.ub, self.n_dims  = None, None, None
        self.pop1_size, self.pop2_size = None, None

        self.sort_flag = True
        # Determine to sort the problem or not in each epoch
        ## if True, the problem always sorted with fitness value increase
        ## if False, the problem is not sorted

    def initialize_variables(self):
      pass
    def initialization(self):
        """
        Override this method if needed. But the first 2 lines of code is required.
        """
        ### Required code
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)

        ### Your additional code can be implemented here
        self.lb = self.problem.lb
        self.ub = self.problem.ub
        self.n_dims = self.problem.n_dims

        # ADE Parameters
        self.NPopinit = self.pop_size -2
        self.pop1_size = self.NPopinit
        self.pop2_size = self.pop_size - self.pop1_size
        self.beta_min = 0.2
        self.beta_max = 0.8

        #initial_pop = self._chaotic_initialization()
        #self.pop = [self.generate_agent(initial_pop[idx]) for idx in range(0, self.pop_size)]
        self.population = [agent for agent in self.pop if agent.target is not None] # Add this line

        self.g_best = self.get_sorted_population(self.population, minmax="min")[0]
        self.pop1 = self.population[:self.pop1_size]
        self.pop2 = self.population[self.pop1_size:]

    def evolve(self, epoch):
            self.pop1_size = round((((self.NPopmin - self.NPopinit) / self.epoch) * epoch) + self.NPopinit)
            if self.pop1_size < len(self.pop1):
                self.pop2_size = self.pop_size - self.pop1_size  # Adjust pop2_size based on pop1_size
                self.pop1, self.g_best = self.update_global_best_agent(self.pop1)
                differ = len(self.pop1) - self.pop1_size                # Number of members to adjust
                # Add top `differ` members of pop1 to pop2
                top_members = self.pop1[:differ]
                self.pop2.extend(top_members)
                # Remove the bottom `differ` members from pop1
                self.pop1 = self.pop1[:-differ]

            self._update_pop1()
            self._update_pop2()

    def _update_pop1(self):
        for i in range(self.pop1_size):
            x1 = self.pop1[i].solution
            A = np.random.permutation(self.pop1_size)
            A = np.delete(A, np.where(A == i))
            a, b, c = A[:3]

            # Scale Factor (beta1)
            if np.linalg.norm(self.pop1[b].solution - self.pop1[c].solution) > (np.linalg.norm(self.ub - self.lb) / 10):
                beta1 = np.random.uniform(self.beta_min, self.beta_max, self.n_dims)
            else:
                beta1 = np.random.uniform(self.beta_min, self.beta_max * (1 + np.random.rand()), self.n_dims)

            y1 = self.pop1[a].solution + beta1 * (self.pop1[b].solution - self.pop1[c].solution)
            y1 = np.clip(y1, self.lb, self.ub)
            z1 = self._crossover(x1, y1)

            # Evaluate and update best solution
            new_agent = self.generate_agent(z1)
            if self.compare_target(new_agent.target , self.pop1[i].target , minmax="min"):
                self.pop1[i] = new_agent

            # Update global best solution
            self.pop1, self.g_best = self.update_global_best_agent(self.pop1)

    def _update_pop2(self):
        if self.pop2_size >= 8:  # 2 * NPopmin
            self.pop2[:self.NPopmin] = self.pop1[:self.NPopmin]

        for j in range(self.pop2_size):
            x2 = self.pop2[j].solution
            res_of_top2 = (np.random.rand(self.n_dims) * (x2 - self.pop2[0].solution) +
                           np.random.rand(self.n_dims) * (x2 - self.pop2[1].solution))
            beta2 = np.random.uniform(self.beta_min, self.beta_max, self.n_dims)
            y2 = beta2 * res_of_top2
            y2 = np.clip(y2, self.lb, self.ub)
            z2 = self._crossover(x2, y2)

            new_agent2 = self.generate_agent(z2)
            self.g_best = self.get_better_agent(self.g_best, new_agent2, minmax = "min", reverse= False)
            self.pop2.append(new_agent2)

        # Sort pop2 and retain only the best solutions
        self.pop2 = self.get_sorted_and_trimmed_population(self.pop2, self.pop2_size, minmax = "min")

    def _logistic_map(self, x: np.ndarray, mu: float = 4) -> np.ndarray:
        x = np.clip(x, 0, 1)
        return mu * x * (1 - x)

    def _chaotic_initialization(self):
        pop = []
        num_iterations = 10
        lb = np.array(self.lb)
        ub = np.array(self.ub)
        x = np.random.rand(self.n_dims)  # Initialize within [0, 1]

        for i in range(self.pop_size):
            for _ in range(num_iterations):  # Apply logistic map multiple times for chaos
                x = self._logistic_map(x)
            # Rescale chaotic values to [lb, ub]
            chaotic_position = lb + x * (ub - lb)
            pop.append(chaotic_position)
        return pop

    def _crossover(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        z = np.zeros_like(x)
        j0 = np.random.randint(0, len(x))
        for jj in range(len(x)):
            if jj == j0 or np.random.rand() <= self.pCR:
                z[jj] = y[jj]
            else:
                z[jj] = x[jj]
        return z