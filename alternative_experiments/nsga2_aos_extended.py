# -*- coding: utf-8 -*-
"""
Extended NSGA-II with configurable operator selection modes and proper multi-constraint handling.
Author: CHEUNG Hong Yuk (Student ID: 5506205)
Date: April 2026
Description: Inherits from NSGA2_AOS and adds:
             - mode parameter ('aos', 'fixed', 'random')
             - crowding distance for diversity estimation
             - correct multi-constraint non-dominated sorting and parent selection
"""

import numpy as np
from Inherited_Core_Files_By_ZCG_Baseline.nsga2_aos import NSGA2_AOS
from Inherited_Core_Files_By_ZCG_Baseline.q_selector import QLearningSelector
from q_selector_extended import QLearningSelectorExtended
from operators_de_updated import sbx_polynomial_mutation, de_operator, uniform_crossover_gaussian_mutation


class NSGA2_AOS_Extended(NSGA2_AOS):
    def __init__(self, problem, pop_size=100, max_gen=200, n_offsprings=100, mode='aos', use_crowding=True):
        super().__init__(problem, pop_size, max_gen, n_offsprings)
        self.mode = mode
        self.operator_history = []
        if mode == 'aos':
            if use_crowding:
                self.selector = QLearningSelectorExtended(n_actions=3)
            else:
                self.selector = QLearningSelector(n_actions=3)
        else:
            self.selector = None
        self.crowding = None

    # Non-dominated sorting
    def non_dominated_sort(self, F, CV):
        """
        Perform non-dominated sorting with constraint handling.
        CV: (n_individuals, n_constraints) where <=0 means feasible.
        Returns: fronts (list of lists), crowding (array)
        """
        n = F.shape[0]
        # Feasibility: all constraints <= 0
        feasible = np.all(CV <= 0, axis=1) if CV.ndim == 2 else (CV <= 0).flatten()

        # Precompute max constraint violation for each individual
        if CV.ndim == 2:
            cv_max = np.max(CV, axis=1)
        else:
            cv_max = CV.flatten()

        # Initialize dominance counters
        dominate_counts = np.zeros(n, dtype=int)
        dominated_by = [[] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Constraint-dominance rules
                if feasible[i] and not feasible[j]:
                    # i feasible, j infeasible -> i dominates j
                    dominate_counts[j] += 1
                    dominated_by[i].append(j)
                elif not feasible[i] and feasible[j]:
                    # i infeasible, j feasible -> j dominates i (handled when i=j)
                    continue
                elif feasible[i] and feasible[j]:
                    # Both feasible: compare objectives
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominate_counts[j] += 1
                        dominated_by[i].append(j)
                else:
                    # Both infeasible: compare max violation
                    if cv_max[i] < cv_max[j]:
                        dominate_counts[j] += 1
                        dominated_by[i].append(j)
                    # If cv_max[i] > cv_max[j], j will dominate i in its loop

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

        # Crowding distance (same as original, works on any feasible set)
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

    # Tournament selection
    def select_parents(self, pop, F, CV, n_parents):
        n = len(pop)
        # Feasibility and max violation
        if CV.ndim == 2:
            feasible = np.all(CV <= 0, axis=1)
            cv_max = np.max(CV, axis=1)
        else:
            feasible = (CV <= 0).flatten()
            cv_max = CV.flatten()

        parents = []
        for _ in range(n_parents):
            idx1, idx2 = np.random.choice(n, 2, replace=False)
            if feasible[idx1] and feasible[idx2]:
                # Both feasible: compare dominance
                if np.all(F[idx1] <= F[idx2]) and np.any(F[idx1] < F[idx2]):
                    best = idx1
                elif np.all(F[idx2] <= F[idx1]) and np.any(F[idx2] < F[idx1]):
                    best = idx2
                else:
                    best = idx1 if np.random.rand() < 0.5 else idx2
            elif feasible[idx1]:
                best = idx1
            elif feasible[idx2]:
                best = idx2
            else:
                # Both infeasible: choose smaller max violation
                best = idx1 if cv_max[idx1] < cv_max[idx2] else idx2
            parents.append(pop[best])
        return np.array(parents)

    # Rewrite reproduce with DE/rand/1
    def reproduce(self, parents, action):
        """Generate offspring using the selected operator."""
        n_offspring = self.n_offsprings
        offspring = []

        if action == 0:
            # Operator 0: SBX + PM
            for i in range(0, n_offspring, 2):
                p1 = parents[i]
                p2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
                c1, c2 = sbx_polynomial_mutation(p1, p2, self.problem)
                offspring.append(c1)
                offspring.append(c2)
        elif action == 1:
            # Operator 1：Standard DE/rand/1
            for _ in range(n_offspring):
                # Select the index of three parents randomly
                idx = np.random.choice(len(parents), 3, replace=False)
                target, r1, r2 = parents[idx[0]], parents[idx[1]], parents[idx[2]]
                c = de_operator(target, r1, r2, self.problem)
                offspring.append(c)
        else:
            # Operator 2：Uniform crossover + Gaussian mutation
            for i in range(0, n_offspring, 2):
                p1 = parents[i]
                p2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
                c1 = uniform_crossover_gaussian_mutation(p1, p2, self.problem)
                c2 = uniform_crossover_gaussian_mutation(p2, p1, self.problem)
                offspring.append(c1)
                offspring.append(c2)

        return np.array(offspring[:n_offspring])

    # Rewrite run_generation
    def run_generation(self):
        # 1. Select operator
        if self.mode == 'aos':
            state = self.selector.get_state(self, self.problem)
            action = self.selector.select_action(state)
        elif self.mode == 'fixed':
            action = 0
            state = None
        elif self.mode == 'random':
            action = np.random.randint(3)
            state = None
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        self.operator_history.append(action)

        # 2. Parent selection (using our new select_parents)
        parents = self.select_parents(self.pop, self.F, self.CV, self.n_offsprings)

        # 3. Reproduction
        offspring = self.reproduce(parents, action)
        F_off, CV_off = self.problem.evaluate(offspring)

        # 4. Merge and environmental selection
        combined_pop = np.vstack([self.pop, offspring])
        combined_F = np.vstack([self.F, F_off])
        combined_CV = np.vstack([self.CV, CV_off])

        fronts, crowding = self.non_dominated_sort(combined_F, combined_CV)
        self.crowding = crowding

        new_pop, new_F, new_CV = [], [], []
        remaining = self.pop_size
        for front in fronts:
            if len(front) <= remaining:
                new_pop.extend(combined_pop[front])
                new_F.extend(combined_F[front])
                new_CV.extend(combined_CV[front])
                remaining -= len(front)
            else:
                # Select best individuals from this front by crowding distance
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

        # 5. IGD and reward
        current_igd = self.compute_igd(self.F)
        if len(self.igd_history) == 0:
            reward = 0
        else:
            reward = self.igd_history[-1] - current_igd
        self.igd_history.append(current_igd)

        # 6. Q-learning update
        if self.mode == 'aos':
            next_state = self.selector.get_state(self, self.problem)
            self.selector.update(state, action, reward, next_state)

        self.gen += 1
        if self.gen % 50 == 0:
            nds = len(self.non_dominated_sort(self.F, self.CV)[0][0])
            print(f"Gen {self.gen}, IGD: {current_igd:.4f}, Nondominated: {nds}")

    # Run
    def run(self):
        self.initialize()
        self.operator_history = []
        for _ in range(self.max_gen):
            self.run_generation()
