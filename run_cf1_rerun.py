# -*- coding: utf-8 -*-
"""
Re-run CF1 experiments with unified settings (500 generations, fixed seed, crowding distance).
Author: CHEUNG Hong Yuk (Student ID: 5506205)
Date: April 2026
Description: Runs 30 independent trials for CF1 for each mode (aos, fixed, random)
             using the same parameter configuration as CTP1/CTP2, ensuring fair comparison.
             Overwrites previous CF1 results (originally 200 generations, no fixed seed).
"""
import pickle
import os
import numpy as np
import random
from problem import CF1
from nsga2_aos_extended import NSGA2_AOS_Extended

def run_single(mode, run_id, max_gen=500, pop_size=100, use_crowding=True):
    prob = CF1(n_var=10)
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
        'problem': 'CF1',
        'igd_history': algo.igd_history,
        'operator_history': algo.operator_history,
        'final_F': algo.F,
        'final_CV': algo.CV,
    }

def main():
    # Fix the random seed
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)

    modes = ['aos', 'fixed', 'random']
    n_runs = 30
    max_gen = 500
    pop_size = 100
    use_crowding = True
    out_dir = 'results'
    os.makedirs(out_dir, exist_ok=True)

    for mode in modes:
        print(f"\nRunning CF1 - {mode}")
        all_results = []
        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}")
            res = run_single(mode, run, max_gen=max_gen, pop_size=pop_size, use_crowding=use_crowding)
            all_results.append(res)
            # Save the running result
            with open(f"{out_dir}/CF1_{mode}_run{run}.pkl", 'wb') as f:
                pickle.dump(res, f)
        # Save all the running results of this mode
        with open(f"{out_dir}/CF1_{mode}_all.pkl", 'wb') as f:
            pickle.dump(all_results, f)

    print("\n CF1 experiments completed. Results saved in 'results/' directory.")

if __name__ == '__main__':
    main()