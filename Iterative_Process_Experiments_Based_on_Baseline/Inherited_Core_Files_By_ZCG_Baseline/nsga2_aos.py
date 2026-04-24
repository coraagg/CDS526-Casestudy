# -*- coding: utf-8 -*-
"""
NSGA-II with Adaptive Operator Selection (AOS) for constrained multi-objective optimization.
Author: ZHANG Chenguo (Student ID: 5577723)
Date: March 2026
Description: This file implements the NSGA-II algorithm integrated with Q-learning based
             adaptive operator selection. It defines the main algorithm class NSGA2_AOS.
"""

import numpy as np
from operators import sbx_polynomial_mutation, de_operator, uniform_crossover_gaussian_mutation
from q_selector import QLearningSelector

class NSGA2_AOS:
    def __init__(self, problem, pop_size=100, max_gen=200, n_offsprings=100):
        self.problem = problem
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.n_offsprings = n_offsprings
        self.gen = 0
        self.pop = None
        self.F = None
        self.CV = None
        self.igd_history = []
        self.selector = QLearningSelector(n_actions=3)

    def initialize(self):
        # Randomly initialize the population
        self.pop = np.random.uniform(self.problem.xl, self.problem.xu,
                                     (self.pop_size, self.problem.n_var))
        self.F, self.CV = self.problem.evaluate(self.pop)
        self.igd_history.append(self.compute_igd(self.F))

    def compute_igd(self, F):
        # Compute Inverted Generational Distance (IGD) to the true Pareto front
        try:
            pf = self.problem.pareto_front()
            min_dist = []
            for f in F:
                dist = np.linalg.norm(pf - f, axis=1)
                min_dist.append(np.min(dist))
            return np.mean(min_dist)
        except:
            return np.inf

    def non_dominated_sort(self, F, CV):
        """Return a list of Pareto fronts and crowding distances."""
        n = F.shape[0]
        feasible = (CV <= 0).flatten()

        # Compute dominance counts and list of individuals dominated by each
        dominate_counts = np.zeros(n, dtype=int)
        dominated_by = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    if feasible[i] and not feasible[j]:
                        dominate_counts[j] += 1
                        dominated_by[i].append(j)
                    elif feasible[i] == feasible[j]:
                        dominate_counts[j] += 1
                        dominated_by[i].append(j)
                # i is dominated by j
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    if feasible[j] and not feasible[i]:
                        dominate_counts[i] += 1
                        dominated_by[j].append(i)
                    elif feasible[i] == feasible[j]:
                        dominate_counts[i] += 1
                        dominated_by[j].append(i)

        # Build fronts
        fronts = []
        remaining = set(range(n))
        while remaining:
            front = [i for i in remaining if dominate_counts[i] == 0]
            fronts.append(front)
            for i in front:
                for j in dominated_by[i]:
                    dominate_counts[j] -= 1
                remaining.remove(i)

        # Compute crowding distance
        crowding = np.zeros(n)
        for front in fronts:
            if len(front) <= 2:
                crowding[front] = np.inf
                continue
            f_front = F[front]
            m = f_front.shape[1]
            for obj in range(m):
                idx = np.argsort(f_front[:, obj])
                crowding[front[idx[0]]] = np.inf
                crowding[front[idx[-1]]] = np.inf
                fmin = f_front[idx[0], obj]
                fmax = f_front[idx[-1], obj]
                if fmax - fmin > 0:
                    for k in range(1, len(front)-1):
                        crowding[front[idx[k]]] += (f_front[idx[k+1], obj] - f_front[idx[k-1], obj]) / (fmax - fmin)
        return fronts, crowding

    def select_parents(self, pop, F, CV, n_parents):
        """Tournament selection: feasible first, then dominance, then crowding distance."""
        parents = []
        for _ in range(n_parents):
            idx1, idx2 = np.random.choice(len(pop), 2, replace=False)
            if CV[idx1] <= 0 and CV[idx2] <= 0:
                # Both feasible: compare dominance
                if np.all(F[idx1] <= F[idx2]) and np.any(F[idx1] < F[idx2]):
                    best = idx1
                elif np.all(F[idx2] <= F[idx1]) and np.any(F[idx2] < F[idx1]):
                    best = idx2
                else:
                    best = idx1 if np.random.rand() < 0.5 else idx2
            elif CV[idx1] <= 0:
                best = idx1
            elif CV[idx2] <= 0:
                best = idx2
            else:
                # Both infeasible: choose the one with smaller constraint violation
                best = idx1 if CV[idx1] < CV[idx2] else idx2
            parents.append(pop[best])
        return np.array(parents)

    def reproduce(self, parents, action):
        """Generate offspring using the selected operator."""
        n_offspring = len(parents)
        offspring = []
        # Ensure even number of parents
        for i in range(0, n_offspring, 2):
            p1 = parents[i]
            p2 = parents[i+1] if i+1 < n_offspring else parents[0]
            if action == 0:
                c1, c2 = sbx_polynomial_mutation(p1, p2, self.problem)
            elif action == 1:
                c1 = de_operator(p1, p2, self.problem)
                c2 = de_operator(p2, p1, self.problem)
            else:
                c1 = uniform_crossover_gaussian_mutation(p1, p2, self.problem)
                c2 = uniform_crossover_gaussian_mutation(p2, p1, self.problem)
            offspring.append(c1)
            offspring.append(c2)
        return np.array(offspring[:self.n_offsprings])

    def run_generation(self):
        # Get current state
        state = self.selector.get_state(self, self.problem)
        action = self.selector.select_action(state)

        # Parent selection
        parents = self.select_parents(self.pop, self.F, self.CV, self.n_offsprings)
        # Offspring generation
        offspring = self.reproduce(parents, action)
        F_off, CV_off = self.problem.evaluate(offspring)

        # Combine populations
        combined_pop = np.vstack([self.pop, offspring])
        combined_F = np.vstack([self.F, F_off])
        combined_CV = np.vstack([self.CV, CV_off])

        # Non-dominated sorting and crowding distance
        fronts, crowding = self.non_dominated_sort(combined_F, combined_CV)

        # Environmental selection
        new_pop = []
        new_F = []
        new_CV = []
        remaining = self.pop_size
        for front in fronts:
            if len(front) <= remaining:
                new_pop.extend(combined_pop[front])
                new_F.extend(combined_F[front])
                new_CV.extend(combined_CV[front])
                remaining -= len(front)
            else:
                # Select best individuals from this front based on crowding distance
                front_crowding = crowding[front]
                sorted_idx = np.argsort(front_crowding)[::-1]
                selected = [front[i] for i in sorted_idx[:remaining]]
                new_pop.extend(combined_pop[selected])
                new_F.extend(combined_F[selected])
                new_CV.extend(combined_CV[selected])
                break

        self.pop = np.array(new_pop)
        self.F = np.array(new_F)
        self.CV = np.array(new_CV)

        # Compute IGD and reward for Q-learning
        current_igd = self.compute_igd(self.F)
        if len(self.igd_history) == 0:
            reward = 0
        else:
            reward = self.igd_history[-1] - current_igd
        self.igd_history.append(current_igd)

        # Update Q-learning
        next_state = self.selector.get_state(self, self.problem)
        self.selector.update(state, action, reward, next_state)

        self.gen += 1
        if self.gen % 50 == 0:
            nds = len(self.non_dominated_sort(self.F, self.CV)[0][0])  # Size of the first front
            print(f"Gen {self.gen}, IGD: {current_igd:.4f}, Nondominated: {nds}")

    def run(self):
        self.initialize()
        for _ in range(self.max_gen):
            self.run_generation()