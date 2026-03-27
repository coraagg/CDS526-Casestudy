# -*- coding: utf-8 -*-
"""
CF1 benchmark problem from CEC2009 constrained multi-objective competition.
Author: ZHANG Chenguo (Student ID: 5577723)
Date: March 2026
Description: Defines the CF1 problem with two objectives and one inequality constraint.
             The evaluate() method returns objectives and constraint violations.
             The pareto_front() method returns a sampled true Pareto front for IGD calculation.
"""

import numpy as np

class CF1:
    """
    CF1 problem from CEC2009 constrained multi-objective benchmark.
    Reference: "Constrained Multi-Objective Optimization: Test Problem
    Constructions and Performance Evaluations" (Zhang et al., 2009).
    """
    def __init__(self, n_var=10):
        self.n_var = n_var
        self.n_obj = 2
        self.n_constr = 1
        self.xl = np.zeros(n_var)
        self.xu = np.ones(n_var)

    def evaluate(self, X):
        n = X.shape[0]
        F = np.zeros((n, self.n_obj))
        G = np.zeros((n, self.n_constr))

        for i in range(n):
            x = X[i]
            # Indices for odd and even positions (1-indexed)
            J1 = [j for j in range(1, self.n_var) if j % 2 == 1]   # odd indices
            J2 = [j for j in range(1, self.n_var) if j % 2 == 0]   # even indices

            # Compute g1 and g2
            g1 = 0.0
            for j in J1:
                yj = x[j] - np.sin(6 * np.pi * x[0] + j * np.pi / len(J1))
                g1 += yj**2
            g1 = 2.0 / len(J1) * g1

            g2 = 0.0
            for j in J2:
                yj = x[j] - np.sin(6 * np.pi * x[0] + j * np.pi / len(J2))
                g2 += yj**2
            g2 = 2.0 / len(J2) * g2

            f1 = x[0] + g1
            f2 = 1 - x[0]**2 + g2

            F[i] = [f1, f2]
            # Constraint: sin(3π x1)^2 - (f1 + f2 - 1) <= 0
            G[i] = (np.sin(3 * np.pi * x[0]))**2 - (f1 + f2 - 1)

        return F, G

    def pareto_front(self):
        """Return a sampled true Pareto front (for IGD calculation)."""
        f1 = np.linspace(0, 1, 100)
        f2 = 1 - np.sqrt(f1)   # Approximate true PF of CF1
        return np.column_stack([f1, f2])
