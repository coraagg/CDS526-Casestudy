# -*- coding: utf-8 -*-
"""
Experiment runner for NSGA-II with adaptive operator selection.
Author: ZHANG Chenguo (Student ID: 5577723)
Date: April 2026
Description: Runs multiple independent trials (default 30) on a list of benchmark problems
             (CF1, CTP1-8, MW1-14). Results are saved as pickle files in the 'results/' directory.
"""

import os
import pickle
import warnings
import numpy as np
import random
from problem import get_problem
from nsga2_aos import NSGA2_AOS

def run_single_run(problem_name, run_id, pop_size=100, max_gen=200):
    """
    Run a single experiment on a given problem.
    """
    # Set seed for reproducibility (each run gets a different seed)
    seed = 42 + run_id
    np.random.seed(seed)
    random.seed(seed)

    problem = get_problem(problem_name)
    if problem.pareto_front() is None:
        warnings.warn(f"Problem {problem_name} has no true Pareto front. IGD will be set to a large constant.")
    algo = NSGA2_AOS(problem, pop_size=pop_size, max_gen=max_gen)
    algo.run()
    result = {
        'run': run_id,
        'problem': problem_name,
        'igd_history': algo.igd_history,
        'final_pop': algo.pop,
        'final_F': algo.F,
        'final_CV': algo.CV
    }
    return result

def run_experiments(problem_list, n_runs=30, max_gen=200):
    """
    Run multiple experiments for a list of problems.
    """
    os.makedirs('results', exist_ok=True)
    all_results = []

    for problem in problem_list:
        print(f"\n=== Running {problem} ===")
        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}")
            res = run_single_run(problem, run, max_gen=max_gen)
            all_results.append(res)

            with open(f'results/{problem}_run_{run}.pkl', 'wb') as f:
                pickle.dump(res, f)

        problem_results = [r for r in all_results if r['problem'] == problem]
        with open(f'results/{problem}_all.pkl', 'wb') as f:
            pickle.dump(problem_results, f)

    with open('results/all_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    print("\nAll experiments finished. Results saved in 'results/' directory.")

if __name__ == '__main__':
    # Set global seeds for reproducibility of the entire script
    np.random.seed(42)
    random.seed(42)

    cf_problems = ['cf1']
    ctp_problems = [f'ctp{i}' for i in range(1, 9)]
    mw_problems = [f'mw{i}' for i in range(1, 15)]

    problem_list = cf_problems + ctp_problems + mw_problems
    # Optional subset for quick test:
    # problem_list = ['cf1', 'ctp1', 'mw1']

    run_experiments(problem_list, n_runs=30, max_gen=200)
