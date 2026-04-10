# -*- coding: utf-8 -*-
"""
Experiment runner for CTP1 and CTP2 problems.
Author: CHEUNG Hong Yuk (Student ID: 5506205)
Date: April 2026
Description: Runs 30 independent trials for each problem (CTP1, CTP2) and each mode
             (aos, fixed, random) with 500 generations, fixed random seed (42),
             population size 100, and crowding distance enabled. Results are saved
             as pickle files in the 'results_CTP1&CTP2/' directory.
"""
import pickle
import os
import numpy as np
import random
from mw_ctp_problems import CTP1_Wrapper, CTP2_Wrapper
from nsga2_aos_extended import NSGA2_AOS_Extended

def run_single(problem_class, mode, run_id, max_gen=500, pop_size=100, use_crowding=True):
    prob = problem_class()
    algo = NSGA2_AOS_Extended(
        prob,
        pop_size=pop_size,
        max_gen=max_gen,
        n_offsprings=pop_size,
        mode=mode,
        use_crowding=use_crowding
    )
    algo.run()
    return {
        'run': run_id,
        'mode': mode,
        'problem': problem_class.__name__,
        'igd_history': algo.igd_history,
        'operator_history': algo.operator_history,
        'final_F': algo.F,
        'final_CV': algo.CV,
    }

def main():
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)

    # Run CTP1 and CTP2
    problems = [CTP1_Wrapper, CTP2_Wrapper]
    modes = ['aos', 'fixed', 'random']
    n_runs = 30
    max_gen = 500
    pop_size = 100
    use_crowding = True
    out_dir = 'results'
    os.makedirs(out_dir, exist_ok=True)

    for prob_cls in problems:
        prob_name = prob_cls.__name__
        for mode in modes:
            print(f"\nRunning {prob_name} - {mode}")
            all_results = []
            for run in range(n_runs):
                print(f"  Run {run+1}/{n_runs}")
                res = run_single(prob_cls, mode, run, max_gen=max_gen, pop_size=pop_size, use_crowding=use_crowding)
                all_results.append(res)
                with open(f"{out_dir}/{prob_name}_{mode}_run{run}.pkl", 'wb') as f:
                    pickle.dump(res, f)
            with open(f"{out_dir}/{prob_name}_{mode}_all.pkl", 'wb') as f:
                pickle.dump(all_results, f)

    print("\n CTP1 and CTP2 experiments completed. Results saved in 'results/' directory.")

if __name__ == '__main__':
    main()