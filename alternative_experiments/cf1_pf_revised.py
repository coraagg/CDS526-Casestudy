# -*- coding: utf-8 -*-
"""
CF1 benchmark problem from CEC2009 constrained multi-objective competition (official original version).
Author: CHEUNG Hong Yuk (Student ID: 5506205)
Date: April 2026
Description: Implements the CF1 problem as defined in:
             Zhang et al. (2009) "Multiobjective optimization test instances for the CEC 2009 special session".
             - Two objectives, one inequality constraint.
             - Decision variables: 10 (default), each in [0,1].
             - The true Pareto front is given by f1 + f2 = 1, with f1 ∈ [0,1].
             - The evaluate() method returns objectives F and constraint violations G.
             - The pareto_front() method returns a sampled true Pareto front.
"""

import numpy as np

class CF1:
    """
    CF1 problem from CEC2009 constrained multi-objective benchmark (OFFICIAL ORIGINAL VERSION)
    Reference: Zhang et al. (2009) "Multiobjective optimization test instances for the CEC 2009 special session and competition"
    CEC09 contest standard parameters: N=10, a=1, n_var=10
    """

    def __init__(self, n_var=10, N=10, a=1.0):
        self.n_var = n_var
        self.n_obj = 2
        self.n_constr = 1
        self.xl = np.zeros(n_var)
        self.xu = np.ones(n_var)
        self.N = N  # The standard values in the literature: 10
        self.a = a  # The standard values in the literature: 1.0

    def evaluate(self, X):
        n = X.shape[0]
        F = np.zeros((n, self.n_obj))
        G = np.zeros((n, self.n_constr))

        for i in range(n):
            x = X[i]
            x1 = x[0]

            # J1/J2 definition：1-based 2~n → 0-based 1~n-1
            J1 = [j for j in range(1, self.n_var) if (j + 1) % 2 == 1]
            J2 = [j for j in range(1, self.n_var) if (j + 1) % 2 == 0]

            # Calculate g1（J1）
            g1 = 0.0
            for j in J1:
                exponent = 0.5 * (1.0 + 3.0 * (j - 1) / (self.n_var - 2))
                yj = x[j] - np.power(x1, exponent)
                g1 += yj ** 2
            g1 = 2.0 / len(J1) * g1

            # Calculate g2（J2）
            g2 = 0.0
            for j in J2:
                exponent = 0.5 * (1.0 + 3.0 * (j - 1) / (self.n_var - 2))
                yj = x[j] - np.power(x1, exponent)
                g2 += yj ** 2
            g2 = 2.0 / len(J2) * g2

            # Objective functions
            f1 = x1 + g1
            f2 = 1 - x1 + g2

            F[i] = [f1, f2]

            G[i] = f1 + f2 - self.a * np.abs(np.sin(self.N * np.pi * (f1 - f2 + 1))) - 1

        return F, G

    def pareto_front(self, n_points=100):
        """
        The true Pareto frontier
        formulation: f1 + f2 = 1, f1 ∈ [0, 1]
        """
        i = np.arange(0, 2 * self.N + 1)
        f1 = i / (2.0 * self.N)
        f2 = 1.0 - f1
        return np.column_stack([f1, f2])
