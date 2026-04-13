# -*- coding: utf-8 -*-
"""
Visualization tools for experimental results.
Author: ZHANG Chenguo (Student ID: 5577723)
Date: April 2026
Description: Loads results from all_results.pkl and generates three types of plots:
             - Convergence curves of IGD over generations (one per problem)
             - Boxplot of final IGD values across runs (all problems together)
             - Pareto front comparison (obtained vs true, if available)
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from problem import get_problem

def load_results(filepath='results/all_results.pkl'):
    """Load all experiment results from pickle file."""
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    return results

def plot_convergence_per_problem(results):
    """For each problem, plot the convergence curves of all runs."""
    problems = sorted(set(r['problem'] for r in results))
    for prob in problems:
        plt.figure(figsize=(10, 6))
        prob_results = [r for r in results if r['problem'] == prob]
        for res in prob_results:
            plt.plot(res['igd_history'], alpha=0.5, label=f"Run {res['run']+1}" if res['run'] < 10 else "")
        plt.xlabel('Generation')
        plt.ylabel('IGD')
        plt.title(f'Convergence Curves - {prob.upper()}')
        plt.grid(True)
        plt.legend(loc='upper right', ncol=2)
        plt.savefig(f'results/convergence_{prob}.png')
        plt.close()   # close to avoid displaying all interactively
        print(f"Saved convergence plot for {prob}")

def plot_boxplot_all_problems(results):
    """Boxplot of final IGD values for all problems."""
    problems = sorted(set(r['problem'] for r in results))
    data = []
    labels = []
    for prob in problems:
        final_igd = [r['igd_history'][-1] for r in results if r['problem'] == prob]
        data.append(final_igd)
        labels.append(prob.upper())
    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=labels)
    plt.ylabel('Final IGD')
    plt.title('Boxplot of Final IGD over 30 Runs per Problem')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/boxplot_all_problems.png')
    plt.show()

def plot_pareto_for_problem(problem_name, results):
    """Plot Pareto front comparison for a single problem."""
    prob_results = [r for r in results if r['problem'] == problem_name]
    if not prob_results:
        print(f"No results found for {problem_name}")
        return
    # Use the last run for visualization (or you could pick the best)
    final_res = prob_results[-1]
    final_F = final_res['final_F']
    # Get problem instance to access true front
    try:
        problem = get_problem(problem_name)
        pf = problem.pareto_front()
    except:
        pf = None
    plt.figure(figsize=(8, 6))
    plt.scatter(final_F[:, 0], final_F[:, 1], c='b', marker='o', s=20, alpha=0.6, label='Obtained Solutions')
    if pf is not None:
        plt.plot(pf[:, 0], pf[:, 1], 'r-', linewidth=2, label='True Pareto Front')
    else:
        plt.title(f'No true Pareto front available for {problem_name}')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title(f'Pareto Front Comparison ({problem_name.upper()})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/pareto_{problem_name}.png')
    plt.show()

def plot_all_pareto(results):
    """Generate Pareto plots for all problems that have a true front."""
    problems = sorted(set(r['problem'] for r in results))
    for prob in problems:
        # Only attempt if problem has a pareto_front method (CF1, CTP, MW all do, but may return None)
        try:
            prob_obj = get_problem(prob)
            if prob_obj.pareto_front() is not None:
                plot_pareto_for_problem(prob, results)
            else:
                print(f"Skipping Pareto plot for {prob} (no true front available)")
        except:
            print(f"Could not generate Pareto plot for {prob}")

if __name__ == '__main__':
    # Load results
    results = load_results()
    print(f"Loaded {len(results)} experiment records.")

    # 1. Convergence curves per problem
    plot_convergence_per_problem(results)

    # 2. Boxplot across all problems
    plot_boxplot_all_problems(results)

    # 3. Pareto front plots for each problem (where true front exists)
    plot_all_pareto(results)

    print("All figures saved to results/ directory.")
