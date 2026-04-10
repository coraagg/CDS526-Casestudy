# -*- coding: utf-8 -*-
"""
Visualization script for all three problems (CF1, CTP1, CTP2).
Author: CHEUNG Hong Yuk (Student ID: 5506205)
Date: April 2026
Description: Generates convergence curves (mean IGD ± std), boxplots of final IGD,
             and operator selection frequency stacked area plots (for AOS mode).
             All figures are saved to the 'figures/' directory.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def load_histories_from_folder(folder, prob_name, modes):
    """Load all running igd_history and operator_history for each mode"""
    histories = {mode: {'igd': [], 'op': []} for mode in modes}
    for mode in modes:
        filepath = os.path.join(folder, f'{prob_name}_{mode}_all.pkl')
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        for run in data:
            histories[mode]['igd'].append(run['igd_history'])
            if 'operator_history' in run:
                histories[mode]['op'].append(run['operator_history'])
    return histories


def plot_convergence(histories, prob_name, output_dir):
    """Draw the average IGD convergence curve (with shaded standard deviation)"""
    plt.figure(figsize=(10, 6))
    colors = {'aos': 'blue', 'fixed': 'green', 'random': 'orange'}
    for mode in ['aos', 'fixed', 'random']:
        igd_array = np.array(histories[mode]['igd'])
        mean_igd = np.mean(igd_array, axis=0)
        std_igd = np.std(igd_array, axis=0)
        gens = np.arange(len(mean_igd))
        plt.plot(gens, mean_igd, label=f'{mode.upper()}', color=colors[mode], linewidth=2)
        plt.fill_between(gens, mean_igd - std_igd, mean_igd + std_igd, alpha=0.2, color=colors[mode])
    plt.xlabel('Generation')
    plt.ylabel('IGD')
    plt.title(f'Convergence Curves on {prob_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prob_name}_convergence.png'))
    plt.close()


def plot_boxplot(histories, prob_name, output_dir):
    """Draw the box plot of the final IGD"""
    final_igd = {mode: [run[-1] for run in histories[mode]['igd']] for mode in ['aos', 'fixed', 'random']}
    data = [final_igd['aos'], final_igd['fixed'], final_igd['random']]
    plt.figure(figsize=(8, 6))
    bp = plt.boxplot(data, labels=['AOS', 'Fixed', 'Random'], patch_artist=True,
                     boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red'))
    plt.ylabel('Final IGD')
    plt.title(f'Final IGD Distribution on {prob_name}')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prob_name}_boxplot.png'))
    plt.close()


def plot_operator_frequency(histories, prob_name, output_dir):
    """Draw the frequency stacking area chart of the operator selection (only for AOS mode)"""
    if not histories['aos']['op']:
        print(f"Warning: No operator history for {prob_name}, skip operator frequency plot.")
        return
    op_array = np.array(histories['aos']['op'])  # shape: (n_runs, n_gens)
    n_gens = op_array.shape[1]
    # Count the number of times each operator is selected in each generation
    op_counts = np.zeros((n_gens, 3))
    for run_ops in op_array:
        for gen, op in enumerate(run_ops):
            op_counts[gen, op] += 1
    op_freq = op_counts / len(op_array)  # Calculate frequency
    plt.figure(figsize=(10, 6))
    gens = np.arange(n_gens)
    plt.stackplot(gens, op_freq[:, 0], op_freq[:, 1], op_freq[:, 2],
                  labels=['SBX+PM', 'DE', 'Uniform+Gauss'], alpha=0.8,
                  colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.xlabel('Generation')
    plt.ylabel('Selection Frequency')
    plt.title(f'Operator Selection Frequency on {prob_name} (AOS)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prob_name}_op_frequency.png'))
    plt.close()


def main():
    configs = [
        ('CF1', 'results', 'CF1'),
        ('CTP1', 'results_CTP1&CTP2', 'CTP1_Wrapper'),
        ('CTP2', 'results_CTP1&CTP2', 'CTP2_Wrapper')
    ]
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)

    for display_name, folder, prob_file in configs:
        print(f"Processing {display_name}...")
        histories = load_histories_from_folder(folder, prob_file, ['aos', 'fixed', 'random'])
        plot_convergence(histories, display_name, output_dir)
        plot_boxplot(histories, display_name, output_dir)
        plot_operator_frequency(histories, display_name, output_dir)
    print(f"All figures saved to '{output_dir}/'")


if __name__ == '__main__':
    main()