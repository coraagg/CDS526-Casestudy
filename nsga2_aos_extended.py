# -*- coding: utf-8 -*-
"""
Extended NSGA-II with configurable operator selection modes.
Author: CHEUNG Hong Yuk (Student ID: 5506205)
Date: April 2026
Description: This file inherits from NSGA2_AOS and adds three operator selection modes:
             'aos' (adaptive Q-learning), 'fixed' (always SBX+PM), and 'random' (uniform random).
             It also replaces the Euclidean distance diversity metric with crowding distance,
             and fixes multi-constraint handling for problems like CTP1 and CTP2.
"""
import numpy as np
from nsga2_aos import NSGA2_AOS
from q_selector import QLearningSelector
from q_selector_extended import QLearningSelectorExtended

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

    # New select_parents: Adds support for multiple constraints
    def select_parents(self, pop, F, CV, n_parents):
        """
        Championship selection, supporting multiple constraints
        (combining multiple constraint violations into the maximum violation amount)
        """
        # Convert the constraint violations into scalars
        # (each individual takes the maximum value of all constraints)
        if CV.ndim == 2:
            cv_scalar = np.max(CV, axis=1)
        else:
            cv_scalar = CV.flatten()

        parents = []
        for _ in range(n_parents):
            idx1, idx2 = np.random.choice(len(pop), 2, replace=False)
            feasible1 = cv_scalar[idx1] <= 0
            feasible2 = cv_scalar[idx2] <= 0

            if feasible1 and feasible2:
                # All feasible solutions: comparison of dominance relations
                if np.all(F[idx1] <= F[idx2]) and np.any(F[idx1] < F[idx2]):
                    best = idx1
                elif np.all(F[idx2] <= F[idx1]) and np.any(F[idx2] < F[idx1]):
                    best = idx2
                else:
                    best = idx1 if np.random.rand() < 0.5 else idx2
            elif feasible1:
                best = idx1
            elif feasible2:
                best = idx2
            else:
                # All are infeasible solutions: the selection constraints violate the small value
                best = idx1 if cv_scalar[idx1] < cv_scalar[idx2] else idx2
            parents.append(pop[best])
        return np.array(parents)

    # New run_generation: depend on the original run_generation
    def run_generation(self):
        # 1. Obtain the state and actions based on the pattern
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

        # 2. Parental selection
        parents = self.select_parents(self.pop, self.F, self.CV, self.n_offsprings)

        # 3. Reproduce
        offspring = self.reproduce(parents, action)
        F_off, CV_off = self.problem.evaluate(offspring)
        if CV_off.ndim == 2 and CV_off.shape[1] > 1:
            CV_off = np.max(CV_off, axis=1).reshape(-1, 1)

        # 4. Population merger and environmental selection
        combined_pop = np.vstack([self.pop, offspring])
        combined_F = np.vstack([self.F, F_off])
        combined_CV = np.vstack([self.CV, CV_off])

        # Combine multiple constraints into a single column (select the maximum violation amount)
        if combined_CV.shape[1] > 1:
            combined_CV_scalar = np.max(combined_CV, axis=1).reshape(-1, 1)
        else:
            combined_CV_scalar = combined_CV

        fronts, crowding = self.non_dominated_sort(combined_F, combined_CV_scalar)
        self.crowding = crowding

        new_pop, new_F, new_CV = [], [], []
        remaining = self.pop_size
        for front in fronts:
            if len(front) <= remaining:
                new_pop.extend(combined_pop[front])
                new_F.extend(combined_F[front])
                new_CV.extend(combined_CV_scalar[front])
                remaining -= len(front)
            else:
                front_crowding = crowding[front]
                sorted_idx = np.argsort(front_crowding)[::-1]
                selected = [front[i] for i in sorted_idx[:remaining]]
                new_pop.extend(combined_pop[selected])
                new_F.extend(combined_F[selected])
                new_CV.extend(combined_CV_scalar[selected])
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

    # Run (reset operator_history)
    def run(self):
        self.initialize()
        # Compress the initial CV into a single column
        if self.CV.ndim == 2 and self.CV.shape[1] > 1:
            self.CV = np.max(self.CV, axis=1).reshape(-1, 1)
        self.operator_history = []
        for _ in range(self.max_gen):
            self.run_generation()