from problem import CF1
from nsga2_aos import NSGA2_AOS
import pickle
import os

def run_single_run(problem_class, run_id, pop_size=100, max_gen=200):
    problem = problem_class(n_var=10)
    algo = NSGA2_AOS(problem, pop_size=pop_size, max_gen=max_gen)
    algo.run()
    return {
        'run': run_id,
        'igd_history': algo.igd_history,
        'final_pop': algo.pop,
        'final_F': algo.F,
        'final_CV': algo.CV
    }

def run_experiments(n_runs=30, max_gen=200):
    os.makedirs('results', exist_ok=True)
    all_results = []
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}")
        res = run_single_run(CF1, run, max_gen=max_gen)
        all_results.append(res)
        with open(f'results/run_{run}.pkl', 'wb') as f:
            pickle.dump(res, f)
    with open('results/all_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print("Experiments finished.")

if __name__ == '__main__':
    run_experiments(n_runs=30, max_gen=200)