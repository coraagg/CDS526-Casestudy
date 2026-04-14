# -*- coding: utf-8 -*-
"""
Benchmark problems for constrained multi-objective optimization.
Author: ZHANG Chenguo (Student ID: 5577723)
Date: April 2026
Description: Provides CF1 (custom implementation, standard constraint),
             and wrappers for CTP and MW series from pymoo.
"""

import numpy as np
from pymoo.problems import get_problem as pymoo_get_problem

# ----------------------------------------------------------------------
# CF1 problem (custom implementation, CEC2009 standard)
# ----------------------------------------------------------------------
class CF1:
    """
    CF1 problem from CEC2009 constrained multi-objective benchmark.
    Constraint (standard): sin(3π x0) - (f1+f2-1) <= 0
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

        J1 = [j for j in range(1, self.n_var) if j % 2 == 1]   # odd indices (1‑based)
        J2 = [j for j in range(1, self.n_var) if j % 2 == 0]   # even indices

        for i in range(n):
            x = X[i]
            g1 = 0.0
            for j in J1:
                yj = x[j] - np.sin(6 * np.pi * x[0] + (j - 1) * np.pi / len(J1))
                g1 += yj ** 2
            g1 = 2.0 / len(J1) * g1

            g2 = 0.0
            for j in J2:
                yj = x[j] - np.sin(6 * np.pi * x[0] + (j - 1) * np.pi / len(J2))
                g2 += yj ** 2
            g2 = 2.0 / len(J2) * g2

            f1 = x[0] + g1
            f2 = 1 - x[0] ** 2 + g2
            F[i] = [f1, f2]
            # Standard constraint (no square)
            G[i] = np.sin(3 * np.pi * x[0]) - (f1 + f2 - 1)

        return F, G

    def pareto_front(self, n_points=500):
        """
        Return the feasible part of the true Pareto front.
        The feasible PF satisfies: sin(3π x0) - (f1+f2-1) <= 0,
        where f1 = x0, f2 = 1 - x0^2.
        """
        x0 = np.linspace(0, 1, n_points)
        f1 = x0
        f2 = 1 - x0 ** 2
        # Constraint value: sin(3π x0) - (x0 + (1-x0^2) - 1) = sin(3π x0) - (x0 - x0^2)
        constraint = np.sin(3 * np.pi * x0) - (x0 - x0 ** 2)
        feasible = constraint <= 1e-6   # small tolerance
        f1_feas = f1[feasible]
        f2_feas = f2[feasible]
        # Ensure unique and sorted by f1 (increasing)
        pf = np.column_stack([f1_feas, f2_feas])
        _, unique_idx = np.unique(f1_feas, return_index=True)
        pf = pf[unique_idx]
        return pf


# ----------------------------------------------------------------------
# Wrapper for pymoo built‑in problems (CTP, MW, and any other)
# ----------------------------------------------------------------------
class PymooProblemWrapper:
    """
    Adapts a pymoo problem object to the same evaluate() interface as CF1.
    """
    def __init__(self, pymoo_problem):
        self.problem = pymoo_problem
        self.n_var = pymoo_problem.n_var
        self.n_obj = pymoo_problem.n_obj
        self.n_constr = pymoo_problem.n_constr
        self.xl = pymoo_problem.xl
        self.xu = pymoo_problem.xu

    def evaluate(self, X):
        out = {}
        self.problem._evaluate(X, out)
        F = out["F"]
        G = out.get("G", np.zeros((X.shape[0], 0)))
        return F, G

    def pareto_front(self):
        try:
            return self.problem.pareto_front()
        except:
            return None


# ----------------------------------------------------------------------
# Factory function
# ----------------------------------------------------------------------
def get_problem(name, n_var=None):
    name_lower = name.lower()
    if name_lower == 'cf1':
        n = n_var if n_var is not None else 10
        return CF1(n_var=n)
    elif name_lower.startswith('ctp'):
        try:
            prob = pymoo_get_problem(name_lower)
            return PymooProblemWrapper(prob)
        except:
            raise ValueError(f"Unknown CTP problem: {name}")
    elif name_lower.startswith('mw'):
        try:
            prob = pymoo_get_problem(name_lower)
            return PymooProblemWrapper(prob)
        except:
            raise ValueError(f"Unknown MW problem: {name}")
    else:
        raise ValueError(f"Unsupported problem: {name}")
