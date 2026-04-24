# -*- coding: utf-8 -*-
"""
Q-learning selector for adaptive operator selection.
Author: ZHANG Chenguo (Student ID: 5577723)
Date: March 2026
Description: Implements the Q-learning agent that observes the population state and
             selects among three variation operators.
"""

import numpy as np

class QLearningSelector:
    def __init__(self, n_actions=3, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_actions = n_actions
        self.alpha = alpha          # Learning rate
        self.gamma = gamma          # Discount factor
        self.epsilon = epsilon      # Exploration rate
        self.Q = {}                 # Q-table (state -> action values)

    def _discretize_state(self, state):
        """Discretize continuous state features into bins."""
        bins = [10, 10, 10, 10]   # 10 bins per dimension
        return tuple(int(s * b) for s, b in zip(state, bins))

    def get_state(self, algorithm, problem):
        """Extract normalized state features from the algorithm."""
        gen_norm = algorithm.gen / algorithm.max_gen

        # Population diversity: average pairwise Euclidean distance
        pop = algorithm.pop
        if len(pop) > 1:
            distances = []
            for i in range(len(pop)):
                for j in range(i+1, len(pop)):
                    d = np.linalg.norm(pop[i] - pop[j])
                    distances.append(d)
            diversity = np.mean(distances)
            max_dist = np.linalg.norm(problem.xu - problem.xl)
            diversity_norm = diversity / max_dist if max_dist > 0 else 0
        else:
            diversity_norm = 0

        # Average constraint violation
        if hasattr(algorithm, 'CV') and algorithm.CV is not None:
            cv_vals = algorithm.CV.flatten()
            cv_mean = np.mean(cv_vals)
            cv_norm = min(cv_mean / 10.0, 1.0)
        else:
            cv_norm = 0

        # IGD improvement (normalized to [0,1])
        if len(algorithm.igd_history) >= 2:
            improvement = (algorithm.igd_history[-2] - algorithm.igd_history[-1]) / (algorithm.igd_history[-2] + 1e-8)
            igd_improve_norm = min(max(improvement, -1), 1) / 2 + 0.5
        else:
            igd_improve_norm = 0.5

        return [gen_norm, diversity_norm, cv_norm, igd_improve_norm]

    def select_action(self, state):
        """Select an operator using epsilon-greedy policy."""
        s = self._discretize_state(state)
        if s not in self.Q:
            self.Q[s] = np.zeros(self.n_actions)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[s])

    def update(self, state, action, reward, next_state):
        """Update Q-table using standard Q-learning rule."""
        s = self._discretize_state(state)
        ns = self._discretize_state(next_state)
        if s not in self.Q:
            self.Q[s] = np.zeros(self.n_actions)
        if ns not in self.Q:
            self.Q[ns] = np.zeros(self.n_actions)
        best_next = np.max(self.Q[ns])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[s][action]
        self.Q[s][action] += self.alpha * td_error