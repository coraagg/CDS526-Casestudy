import pickle
import numpy as np
import matplotlib.pyplot as plt
from problem import CF1

def load_results(filepath='results/all_results.pkl'):
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    return results

def plot_convergence(results):
    plt.figure(figsize=(10, 6))
    for i, res in enumerate(results):
        plt.plot(res['igd_history'], alpha=0.5, label=f'Run {i+1}' if i < 10 else '')
    plt.xlabel('Generation')
    plt.ylabel('IGD')
    plt.title('Convergence Curves of 30 Runs')
    plt.grid(True)
    plt.legend(loc='upper right', ncol=2)
    plt.savefig('results/convergence_curves.png')
    plt.show()

def plot_boxplot(results):
    final_igd = [res['igd_history'][-1] for res in results]
    plt.figure(figsize=(8, 6))
    plt.boxplot(final_igd)
    plt.ylabel('Final IGD')
    plt.title('Boxplot of Final IGD over 30 Runs')
    plt.grid(True)
    plt.savefig('results/boxplot.png')
    plt.show()

def plot_pareto(results):
    # Use the last run for Pareto front visualization
    last_res = results[-1]
    final_F = last_res['final_F']
    problem = CF1()
    pf = problem.pareto_front()
    plt.figure(figsize=(8, 6))
    plt.scatter(final_F[:, 0], final_F[:, 1], c='b', marker='o', s=20, alpha=0.6, label='Obtained Solutions')
    plt.plot(pf[:, 0], pf[:, 1], 'r-', linewidth=2, label='True Pareto Front')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title('Pareto Front Comparison (Last Run)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/pareto_front.png')
    plt.show()

if __name__ == '__main__':
    results = load_results()
    plot_convergence(results)
    plot_boxplot(results)
    plot_pareto(results)
    print("All figures saved to results/ directory.")