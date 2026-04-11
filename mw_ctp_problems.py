# -*- coding: utf-8 -*-
"""
Wrapper classes for pymoo's CTP1 and CTP2 problems (and MW1, but unused).
Author: CHEUNG Hong Yuk (Student ID: 5506205)
Date: April 2026
Description: Provides a unified interface (evaluate(), pareto_front(), n_var, etc.)
             for CTP1 and CTP2, making them compatible with the NSGA2_AOS_Extended framework.
             MW1 is included but was abandoned due to its IGD value range (7→5) being
             different with other benchmarks.
"""
import numpy as np
from pymoo.problems.multi import MW1, CTP1

# Unused
class MW1_Wrapper:
    """
    Wrapper for MW1 problem from CEC2020.
    Default number of variables: 15
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
        # Out is a result object with attributes F and G
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

    def pareto_front(self):
        """Return the true Pareto front of CTP1 (provided by pymoo)."""
        return self.problem.pareto_front()

class CTP2_Wrapper:
    def __init__(self):
        from pymoo.problems.multi import CTP2
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

    def pareto_front(self):
        return self.problem.pareto_front()
