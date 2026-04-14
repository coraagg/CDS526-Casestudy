# -*- coding: utf-8 -*-
"""
NSGA-II with Adaptive Operator Selection (AOS) for constrained multi-objective optimization.
Author: ZHANG Chenguo (Student ID: 5577723)
Date: April 2026
Description: NSGA-II integrated with Q-learning based adaptive operator selection.
             All operators and Q-selector are defined inside this file.
"""

import numpy as np

# =============================================================================
# Operators (SBX + polynomial mutation, DE, uniform crossover + Gaussian mutation)
# =============================================================================

def sbx_polynomial_mutation(p1, p2, problem, eta_c=20, eta_m=20, prob_m=None):
    xl = problem.xl
    xu = problem.xu
    n_var = problem.n_var
    if prob_m is None:
        prob_m = 1.0 / n_var

    # SBX crossover
    u = np.random.rand(n_var)
    beta = np.zeros(n_var)
    beta[u <= 0.5] = (2 * u[u <= 0.5]) ** (1.0 / (eta_c + 1))
    beta[u > 0.5] = (2.0 * (1 - u[u > 0.5])) ** (-1.0 / (eta_c + 1))
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)

    # Polynomial mutation for c1
    for i in range(n_var):
        if np.random.rand() < prob_m:
            r = np.random.rand()
            if r < 0.5:
                delta = (2 * r) ** (1.0 / (eta_m + 1)) - 1
            else:
                delta = 1 - (2 * (1 - r)) ** (1.0 / (eta_m + 1))
            c1[i] += delta * (xu[i] - xl[i])
            c1[i] = np.clip(c1[i], xl[i], xu[i])
    # Polynomial mutation for c2
    for i in range(n_var):
        if np.random.rand() < prob_m:
            r = np.random.rand()
            if r < 0.5:
                delta = (2 * r) ** (1.0 / (eta_m + 1)) - 1
            else:
                delta = 1 - (2 * (1 - r)) ** (1.0 / (eta_m + 1))
            c2[i] += delta * (xu[i] - xl[i])
            c2[i] = np.clip(c2[i], xl[i], xu[i])
    return c1, c2

def de_operator(parent, other, problem, F=0.5, CR=0.9):
    xl = problem.xl
    xu = problem.xu
    n_var = problem.n_var
    r1 = parent
    r2 = other
    r3 = np.random.uniform(xl, xu)
    v = r1 + F * (r2 - r3)
    u = np.where(np.random.rand(n_var) <= CR, v, parent)
    u = np.clip(u, xl, xu)
    return u

def uniform_crossover_gaussian_mutation(p1, p2, problem, sigma=0.1):
    xl = problem.xl
    xu = problem.xu
    n_var = problem.n_var
    mask = np.random.rand(n_var) < 0.5
    child = np.where(mask, p1, p2)
    for i in range(n_var):
        if np.random.rand() < 1.0 / n_var:
            delta = np.random.normal(0, sigma * (xu[i] - xl[i]))
            child[i] += delta
            child[i] = np.clip(child[i], xl[i], xu[i])
    return child


# =============================================================================
# Q-Learning Selector
# =============================================================================

class QLearningSelector:
    def __init__(self, n_actions=3, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_state(self, algo, problem):
        feasible = (algo.CV <= 0).sum() / len(algo.CV) if len(algo.CV) > 0 else 0.0
        f_level = min(int(feasible * 5), 4)

        if len(algo.igd_history) >= 5:
            recent = algo.igd_history[-5:]
            improvement = (recent[0] - recent[-1]) / (recent[0] + 1e-8)
        else:
            improvement = 0.0
        if improvement < -0.1:
            i_level = 0
        elif improvement < 0.0:
            i_level = 1
        elif improvement < 0.1:
            i_level = 2
        elif improvement < 0.2:
            i_level = 3
        else:
            i_level = 4
        return (f_level, i_level)

    def select_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.n_actions)
        old_q = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_q = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
        self.q_table[state][action] = new_q


# =============================================================================
# Main NSGA-II with AOS
# =============================================================================

class NSGA2_AOS:
    def __init__(self, problem, pop_size=100, max_gen=200, n_offsprings=100):
        self.problem = problem
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.n_offsprings = n_offsprings
        self.gen = 0
        self.pop = None
        self.F = None
        self.CV = None          # scalar total constraint violation
        self.igd_history = []
        self.selector = QLearningSelector(n_actions=3)

    def initialize(self):
        self.pop = np.random.uniform(self.problem.xl, self.problem.xu,
                                     (self.pop_size, self.problem.n_var))
        F, G = self.problem.evaluate(self.pop)
        self.F = F
        self.CV = np.sum(np.maximum(0, G), axis=1)
        self.igd_history.append(self.compute_igd(self.F, self.CV))

    def compute_igd(self, F, CV):
        """
        Compute Inverted Generational Distance using only feasible solutions.
        IGD = (1/|PF|) * sum_{p in PF} min_{f in F_feas} distance(p, f)
        """
        pf = self.problem.pareto_front()
        if pf is None:
            return 1e9
        # Only feasible solutions
        feasible_mask = CV <= 0
        if not np.any(feasible_mask):
            return 1e9   # no feasible solution
        F_feas = F[feasible_mask]
        total = 0.0
        for p in pf:
            dist = np.linalg.norm(F_feas - p, axis=1)
            total += np.min(dist)
        return total / len(pf)

    def constrained_dominates(self, i, j, F, CV):
        feasible_i = CV[i] <= 0
        feasible_j = CV[j] <= 0
        if feasible_i and not feasible_j:
            return True
        if not feasible_i and feasible_j:
            return False
        if feasible_i and feasible_j:
            return np.all(F[i] <= F[j]) and np.any(F[i] < F[j])
        else:
            return CV[i] < CV[j]

    def non_dominated_sort(self, F, CV):
        n = F.shape[0]
        dominate_count = np.zeros(n, dtype=int)
        dominated_list = [[] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self.constrained_dominates(i, j, F, CV):
                    dominated_list[i].append(j)
                elif self.constrained_dominates(j, i, F, CV):
                    dominate_count[i] += 1

        fronts = []
        rank = np.full(n, -1, dtype=int)
        current_front = [i for i in range(n) if dominate_count[i] == 0]
        front_idx = 0
        while current_front:
            fronts.append(current_front)
            for i in current_front:
                rank[i] = front_idx
            next_front = []
            for i in current_front:
                for j in dominated_list[i]:
                    dominate_count[j] -= 1
                    if dominate_count[j] == 0:
                        next_front.append(j)
            current_front = next_front
            front_idx += 1

        crowding = np.zeros(n)
        for front in fronts:
            if len(front) <= 2:
                crowding[front] = np.inf
                continue
            f_vals = F[front]
            m = f_vals.shape[1]
            for obj in range(m):
                idx = np.argsort(f_vals[:, obj])
                crowding[front[idx[0]]] = np.inf
                crowding[front[idx[-1]]] = np.inf
                fmin = f_vals[idx[0], obj]
                fmax = f_vals[idx[-1], obj]
                if fmax > fmin:
                    for k in range(1, len(front)-1):
                        crowding[front[idx[k]]] += (f_vals[idx[k+1], obj] - f_vals[idx[k-1], obj]) / (fmax - fmin)
        return fronts, rank, crowding

    def select_parents(self, pop, F, CV, rank, crowding, n_parents):
        parents = []
        n = len(pop)
        for _ in range(n_parents):
            i, j = np.random.choice(n, 2, replace=False)
            if rank[i] < rank[j]:
                best = i
            elif rank[i] > rank[j]:
                best = j
            else:
                best = i if crowding[i] > crowding[j] else j
            parents.append(pop[best])
        return np.array(parents)

    def reproduce(self, parents, action):
        n_offspring = len(parents)
        offspring = []
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
        state = self.selector.get_state(self, self.problem)
        action = self.selector.select_action(state)

        _, rank, crowding = self.non_dominated_sort(self.F, self.CV)
        parents = self.select_parents(self.pop, self.F, self.CV, rank, crowding, self.n_offsprings)
        offspring = self.reproduce(parents, action)

        F_off, G_off = self.problem.evaluate(offspring)
        CV_off = np.sum(np.maximum(0, G_off), axis=1)

        combined_pop = np.vstack([self.pop, offspring])
        combined_F = np.vstack([self.F, F_off])
        combined_CV = np.hstack([self.CV, CV_off])

        fronts, rank_comb, crowding_comb = self.non_dominated_sort(combined_F, combined_CV)

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
                front_crowding = crowding_comb[front]
                sorted_idx = np.argsort(front_crowding)[::-1]
                selected = [front[i] for i in sorted_idx[:remaining]]
                new_pop.extend(combined_pop[selected])
                new_F.extend(combined_F[selected])
                new_CV.extend(combined_CV[selected])
                break

        self.pop = np.array(new_pop)
        self.F = np.array(new_F)
        self.CV = np.array(new_CV)

        current_igd = self.compute_igd(self.F, self.CV)
        if len(self.igd_history) == 0:
            reward = 0.0
        else:
            reward = self.igd_history[-1] - current_igd
        self.igd_history.append(current_igd)

        next_state = self.selector.get_state(self, self.problem)
        self.selector.update(state, action, reward, next_state)

        self.gen += 1
        if self.gen % 50 == 0:
            fronts, _, _ = self.non_dominated_sort(self.F, self.CV)
            nds = len(fronts[0]) if fronts else 0
            print(f"Gen {self.gen}, IGD: {current_igd:.4f}, Nondominated: {nds}")

    def run(self):
        self.initialize()
        for _ in range(self.max_gen):
            self.run_generation()
