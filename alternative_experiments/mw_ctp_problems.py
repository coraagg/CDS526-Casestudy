# -*- coding: utf-8 -*-
"""
Wrapper classes for MW1, CTP1, and CTP2 problems from pymoo.
Author: CHEUNG Hong Yuk (Student ID: 5506205)
Date: April 2026
Description: Provides a unified interface (evaluate, pareto_front, n_var, n_obj, n_constr, xl, xu)
             for three constrained multi-objective benchmark problems.
             - MW1: CEC2020 problem, scalable variables (default 15), Pareto front provided by pymoo.
             - CTP1 and CTP2: classic constrained problems, 2 variables each.
               For these, the true Pareto front is approximated as f2 = 1 - f1 (linear) due to the
               unavailability of exact fronts in pymoo.
             The evaluate() method returns objectives (F) and constraint violations (G) as numpy arrays.
"""

import numpy as np
from pymoo.problems.multi import MW1, CTP1, CTP2

class MW1_Wrapper:
    """
    Wrapper for MW1 problem from CEC2020.
    Default number of variables: 15 (can be changed via n_var parameter)
    """
    def __init__(self, n_var=15):
        self.problem = MW1(n_var=n_var)
        self.n_var = self.problem.n_var
        self.n_obj = self.problem.n_obj
        self.n_constr = self.problem.n_constr
        self.xl = self.problem.xl
        self.xu = self.problem.xu

    def evaluate(self, X):
        """
        Evaluate a population X.
        X: (n_individuals, n_var) array
        Returns:
            F: (n_individuals, n_obj) objectives
            G: (n_individuals, n_constr) constraint violations (<=0 means feasible)
        """
        out = self.problem.evaluate(X)
        # out is a Result object with attributes F and G
        return out[0], out[1]

    def pareto_front(self):
        """Return the true Pareto front of MW1 (provided by pymoo)."""
        return self.problem.pareto_front()


class CTP1_Wrapper:
    """
    Wrapper for CTP1 problem (classic constrained test problem).
    Number of variables is fixed to 2.
    """
    def __init__(self):
        self.problem = CTP1()
        self.n_var = self.problem.n_var   # 2
        self.n_obj = self.problem.n_obj   # 2
        self.n_constr = self.problem.n_constr  # 2 (usually)
        self.xl = self.problem.xl
        self.xu = self.problem.xu

    def evaluate(self, X):
        out = self.problem.evaluate(X)
        if isinstance(out, tuple):
            return out[0], out[1]
        else:
            return out.F, out.G

    # def pareto_front(self):
    #     """Return the true Pareto front of CTP1 (provided by pymoo)."""
    #     return self.problem.pareto_front()

    def pareto_front(self, n_points=100):
        # Return the frontier: f2 = 1 - f1, f1 in [0,1]
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - f1
        return np.column_stack([f1, f2])

class CTP2_Wrapper:
    def __init__(self):
        self.problem = CTP2()
        self.n_var = self.problem.n_var
        self.n_obj = self.problem.n_obj
        self.n_constr = self.problem.n_constr
        self.xl = self.problem.xl
        self.xu = self.problem.xu

    def evaluate(self, X):
        out = self.problem.evaluate(X)
        if isinstance(out, tuple):
            return out[0], out[1]
        else:
            return out.F, out.G

    # def pareto_front(self):
    #     return self.problem.pareto_front()

    def pareto_front(self, n_points=100):
        # Approximate frontier: f2 = 1 - f1, f1 in [0,1]
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - f1
        return np.column_stack([f1, f2])