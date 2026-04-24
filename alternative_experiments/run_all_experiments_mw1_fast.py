# -*- coding: utf-8 -*-
"""
Parallel experiment runner for all benchmark problems (CF1, CTP1, CTP2, MW1).
Author: CHEUNG Hong Yuk (Student ID: 5506205)
Date: April 2026
Description: This script runs all experiments in parallel using multiprocessing.
             It supports:
             - Four problems: CF1, CTP1, CTP2, MW1
             - Three operator selection modes: aos, fixed, random
             - 30 independent runs, 500 generations, population 100, fixed base seed 42, with incremental seeds per run (42, 43, …, 71)
             - Crowding distance for diversity (use_crowding=True)
             - Checkpointing: skips already completed runs
             - Progress bar (tqdm) for each mode
             Results are saved in separate subdirectories: results/CF1/, results/CTP/, results/MW/.
"""

import sys
import pickle
import os
import numpy as np
import random
import multiprocessing as mp
from tqdm import tqdm

# Import all problems
from cf1_pf_revised import CF1
from mw_ctp_problems import CTP1_Wrapper, CTP2_Wrapper, MW1_Wrapper
from nsga2_aos_extended import NSGA2_AOS_Extended


def run_single(args):
    """
    Single experiment function (multi-process)
    args: (problem_class, mode, run_id, max_gen, pop_size, use_crowding, config, base_seed)
    """
    problem_class, mode, run_id, max_gen, pop_size, use_crowding, config, base_seed = args

    # Each run uses an independent random seed
    run_seed = base_seed + run_id
    np.random.seed(run_seed)
    random.seed(run_seed)

    # Check if it has been run before
    out_dir = config['out_dir']
    single_file = f"{out_dir}/{problem_class.__name__}_{mode}_run{run_id}.pkl"
    if os.path.exists(single_file):
        with open(single_file, 'rb') as f:
            res = pickle.load(f)
        return (run_id, res, "skipped")

    try:
        # Run the experiment
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

        # Combination result
        res = {
            'run': run_id,
            'mode': mode,
            'problem': problem_class.__name__,
            'config': config,
            'seed': run_seed,
            'igd_history': algo.igd_history,
            'operator_history': algo.operator_history if hasattr(algo, 'operator_history') else None,
            'final_F': algo.F,
            'final_CV': algo.CV,
        }

        # Save the results of a single experiment
        with open(single_file, 'wb') as f:
            pickle.dump(res, f)

        return (run_id, res, "success")

    except Exception as e:
        error_file = f"{out_dir}/{problem_class.__name__}_{mode}_run{run_id}_error.txt"
        with open(error_file, 'w') as f:
            f.write(str(e))
        return (run_id, None, f"error: {str(e)}")


def run_problem(problem_class, problem_display_name, out_dir, n_runs=30, max_gen=500, pop_size=100, config=None):
    """Run all modes for a single issue and support multi-process parallelism."""
    os.makedirs(out_dir, exist_ok=True)
    modes = ['aos', 'fixed', 'random']

    config['out_dir'] = out_dir

    # Set the number of processes: Use the number of CPU cores minus 1
    num_processes = max(1, mp.cpu_count() - 1)
    print(f"\n{'=' * 60}")
    print(f"Running {problem_display_name}")
    print(f"Using {num_processes} parallel processes")
    print(f"{'=' * 60}")

    for mode in modes:
        print(f"\nProcessing {problem_display_name} - {mode}")

        # Set all the parameters for the run
        args_list = [
            (problem_class, mode, run, max_gen, pop_size, True, config, config['seed'])
            for run in range(n_runs)
        ]

        # Multi-process execution
        all_results = [None] * n_runs
        with mp.Pool(processes=num_processes) as pool:
            # Display progress
            with tqdm(total=n_runs, desc=f"{problem_display_name} {mode}", unit="run") as pbar:
                for run_id, res, status in pool.imap_unordered(run_single, args_list):
                    all_results[run_id] = res
                    if status == "skipped":
                        pbar.set_postfix({"status": "skipped"})
                    elif status == "success":
                        final_igd = res['igd_history'][-1]
                        pbar.set_postfix({"final_igd": f"{final_igd:.4f}"})
                    else:
                        pbar.set_postfix({"status": status})
                    pbar.update(1)

        # Filter out the failed results
        valid_results = [res for res in all_results if res is not None]

        # Save the summary results
        if valid_results:
            summary_file = f"{out_dir}/{problem_class.__name__}_{mode}_all.pkl"
            with open(summary_file, 'wb') as f:
                pickle.dump(valid_results, f)
            print(f"Saved {len(valid_results)}/{n_runs} valid results to {summary_file}")
        else:
            print(f"Warning: No valid results for {problem_display_name} {mode}")


def main():
    # Experimental setup
    SEED = 42
    N_RUNS = 30
    MAX_GEN = 500
    POP_SIZE = 100

    # Save configuration information
    config = {
        'seed': SEED,
        'n_runs': N_RUNS,
        'max_gen': MAX_GEN,
        'pop_size': POP_SIZE,
        'date': '2026-04'
    }

    # Set the global random seed (only for the main process)
    np.random.seed(SEED)
    random.seed(SEED)

    # Run all experiments
    print("\n" + "=" * 60)
    print("Starting Parallel Experiments")
    print(f"Config: {config}")
    print(f"CPU cores available: {mp.cpu_count()}")
    print("=" * 60)

    # 1. CF1 (CEC2009)
    run_problem(CF1, "CF1", "results/CF1", n_runs=N_RUNS, max_gen=MAX_GEN, pop_size=POP_SIZE, config=config)

    # 2. CTP1 (Typical)
    run_problem(CTP1_Wrapper, "CTP1", "results/CTP", n_runs=N_RUNS, max_gen=MAX_GEN, pop_size=POP_SIZE, config=config)

    # 3. CTP2 (Typical)
    run_problem(CTP2_Wrapper, "CTP2", "results/CTP", n_runs=N_RUNS, max_gen=MAX_GEN, pop_size=POP_SIZE, config=config)

    # 4. MW1 (CEC2020)
    run_problem(MW1_Wrapper, "MW1", "results/MW", n_runs=N_RUNS, max_gen=MAX_GEN, pop_size=POP_SIZE, config=config)

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("Results saved in:")
    print("  - results/CF1/")
    print("  - results/CTP/")
    print("  - results/MW/")
    print("=" * 60)


if __name__ == '__main__':
    mp.freeze_support()
    main()