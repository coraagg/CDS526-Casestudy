# -*- coding: utf-8 -*-
"""
Extended Q-learning selector using crowding distance for diversity estimation.
Author: CHEUNG Hong Yuk (Student ID: 5506205)
Date: April 2026
Description: Inherits from QLearningSelector and overrides the get_state() method.
             Uses crowding distance (from NSGA-II) as the population diversity feature,
             which is more suitable for high-dimensional decision spaces than Euclidean distance.
"""
import numpy as np
from q_selector import QLearningSelector

class QLearningSelectorExtended(QLearningSelector):
    def get_state(self, algorithm, problem):
        gen_norm = algorithm.gen / algorithm.max_gen

        # Diversity: Using crowding distance for calculation
        if hasattr(algorithm, 'crowding') and algorithm.crowding is not None:
            crowding = algorithm.crowding
            max_cd = np.max(crowding)
            # Avoid division by zero or NaN
            if max_cd > 1e-8:
                diversity_norm = np.mean(crowding) / max_cd
            else:
                diversity_norm = 0.0
            # Limit the scope and handle possible NaN values
            if np.isnan(diversity_norm) or np.isinf(diversity_norm):
                diversity_norm = 0.0
            else:
                diversity_norm = min(max(diversity_norm, 0.0), 1.0)
        else:
            # If the congestion degree is not feasible, then use the Euclidean distance (the same as the original).
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

        # Calculate the average violation of constraints
        if hasattr(algorithm, 'CV') and algorithm.CV is not None:
            cv_vals = algorithm.CV.flatten()
            cv_mean = np.mean(cv_vals)
            cv_norm = min(cv_mean / 10.0, 1.0)
            if np.isnan(cv_norm) or np.isinf(cv_norm):
                cv_norm = 0.0
        else:
            cv_norm = 0.0

        # Improve IGD
        if len(algorithm.igd_history) >= 2:
            improvement = (algorithm.igd_history[-2] - algorithm.igd_history[-1]) / (algorithm.igd_history[-2] + 1e-8)
            igd_improve_norm = min(max(improvement, -1), 1) / 2 + 0.5
            if np.isnan(igd_improve_norm) or np.isinf(igd_improve_norm):
                igd_improve_norm = 0.5
        else:
            igd_improve_norm = 0.5

        state = [gen_norm, diversity_norm, cv_norm, igd_improve_norm]
        # Ensure that all values are finite numbers
        state = [0.0 if (np.isnan(x) or np.isinf(x)) else x for x in state]
        return state