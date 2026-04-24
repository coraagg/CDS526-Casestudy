# -*- coding: utf-8 -*-
"""
Variation operators for adaptive operator selection.
Author: ZHANG Chenguo (Student ID: 5577723)
Date: March 2026
Description: Implements three operators:
             - Operator 0: Simulated binary crossover (SBX) + polynomial mutation
             - Operator 1: Differential evolution (DE/rand/1) mutation
             - Operator 2: Uniform crossover + Gaussian mutation
All operators respect variable bounds defined by the problem.
"""

import numpy as np

def sbx_polynomial_mutation(parent1, parent2, problem, eta_c=20, eta_m=20, prob_m=0.1):
    """Simulated Binary Crossover (SBX) + Polynomial Mutation."""
    n_var = problem.n_var
    child1 = parent1.copy()
    child2 = parent2.copy()
    if np.random.rand() <= 0.9:  # Crossover probability
        # Simplified SBX (uniform distribution)
        u = np.random.rand(n_var)
        beta = np.where(u <= 0.5, (2*u)**(1/(eta_c+1)), (1/(2*(1-u)))**(1/(eta_c+1)))
        child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
        child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
    if np.random.rand() <= prob_m:   # Mutation probability
        # Polynomial mutation
        r = np.random.rand(n_var)
        delta = np.where(r < 0.5, (2*r)**(1/(eta_m+1)) - 1, 1 - (2*(1-r))**(1/(eta_m+1)))
        child1 += delta * (problem.xu - problem.xl)
        child2 += delta * (problem.xu - problem.xl)
    child1 = np.clip(child1, problem.xl, problem.xu)
    child2 = np.clip(child2, problem.xl, problem.xu)
    return child1, child2

def de_operator(parent1, parent2, problem, F=0.5):
    """Differential Evolution mutation (DE/rand/1)."""
    mutant = parent1 + F * (parent2 - parent1)
    return np.clip(mutant, problem.xl, problem.xu)

def uniform_crossover_gaussian_mutation(parent1, parent2, problem, prob_cross=0.9, sigma=0.1):
    """Uniform crossover + Gaussian mutation."""
    child = np.where(np.random.rand(problem.n_var) < prob_cross, parent1, parent2)
    child += sigma * (problem.xu - problem.xl) * np.random.randn(problem.n_var)
    return np.clip(child, problem.xl, problem.xu)