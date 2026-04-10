# -*- coding: utf-8 -*-
"""
Comprehensive statistical analysis for CF1, CTP1, and CTP2 results.
Author: CHEUNG Hong Yuk (Student ID: 5506205)
Date: April 2026
Description: Loads experimental results from different folders (results/ for CF1,
             results_CTP1&CTP2/ for CTP1/2). Computes mean and standard deviation of final IGD,
             performs one-sided Wilcoxon rank-sum tests (AOS vs fixed, AOS vs random),
             and exports a CSV summary table.
"""
import pickle
import numpy as np
import os
from scipy.stats import mannwhitneyu
from pymoo.indicators.igd import IGD


from problem import CF1
from mw_ctp_problems import CTP1_Wrapper, CTP2_Wrapper


def load_results_from_folder(folder, prob_name, modes):
    """Loading the results of the problems"""
    results = {mode: [] for mode in modes}
    for mode in modes:
        filepath = os.path.join(folder, f'{prob_name}_{mode}_all.pkl')
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            results[mode] = data
    return results


def compute_igd_for_problem(prob_name, final_F):
    if prob_name == 'CF1':
        pf = CF1().pareto_front()
    elif prob_name == 'CTP1_Wrapper':
        pf = CTP1_Wrapper().pareto_front()
    elif prob_name == 'CTP2_Wrapper':
        pf = CTP2_Wrapper().pareto_front()
    else:
        raise ValueError(f"Unknown problem: {prob_name}")
    igd = IGD(pf)
    return igd(final_F)


def extract_final_igd(results, prob_name):
    """Extract the final IGD for all experiments"""
    igd_dict = {}
    for mode, data in results.items():
        igd_list = []
        for run in data:
            igd_val = compute_igd_for_problem(prob_name, run['final_F'])
            igd_list.append(igd_val)
        igd_dict[mode] = igd_list
    return igd_dict


def main():
    modes = ['aos', 'fixed', 'random']

    # Loading CF1 results (from results/)
    cf1_results = load_results_from_folder('results', 'CF1', modes)
    cf1_igd = extract_final_igd(cf1_results, 'CF1')

    # Loading CTP1 results (from results_CTP1&CTP2/)
    ctp1_results = load_results_from_folder('results_CTP1&CTP2', 'CTP1_Wrapper', modes)
    ctp1_igd = extract_final_igd(ctp1_results, 'CTP1_Wrapper')

    # Loading CTP2 results (from results_CTP1&CTP2/)
    ctp2_results = load_results_from_folder('results_CTP1&CTP2', 'CTP2_Wrapper', modes)
    ctp2_igd = extract_final_igd(ctp2_results, 'CTP2_Wrapper')

    # Print the statistical table
    print("=" * 60)
    print("Final IGD (mean ± std)")
    print("=" * 60)
    for name, igd_dict in [('CF1', cf1_igd), ('CTP1', ctp1_igd), ('CTP2', ctp2_igd)]:
        print(f"\n{name}:")
        for mode in modes:
            vals = igd_dict[mode]
            print(f"  {mode:6s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Wilcoxon test
    print("\n" + "=" * 60)
    print("Wilcoxon rank-sum test (one-sided: AOS better than baseline)")
    print("=" * 60)
    for name, igd_dict in [('CF1', cf1_igd), ('CTP1', ctp1_igd), ('CTP2', ctp2_igd)]:
        aos = igd_dict['aos']
        fixed = igd_dict['fixed']
        random = igd_dict['random']
        _, p_fixed = mannwhitneyu(aos, fixed, alternative='less')
        _, p_random = mannwhitneyu(aos, random, alternative='less')
        print(f"\n{name}:")
        print(f"  AOS vs fixed : p = {p_fixed:.4f} {'(significant)' if p_fixed < 0.05 else '(not)'}")
        print(f"  AOS vs random: p = {p_random:.4f} {'(significant)' if p_random < 0.05 else '(not)'}")

    # Save CSV
    import csv
    with open('all_igd_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Problem', 'Mode', 'Mean_IGD', 'Std_IGD', 'p_vs_fixed', 'p_vs_random'])
        for name, igd_dict in [('CF1', cf1_igd), ('CTP1', ctp1_igd), ('CTP2', ctp2_igd)]:
            aos = igd_dict['aos']
            fixed = igd_dict['fixed']
            random = igd_dict['random']
            _, p_fixed = mannwhitneyu(aos, fixed, alternative='less')
            _, p_random = mannwhitneyu(aos, random, alternative='less')
            writer.writerow(
                [name, 'aos', f"{np.mean(aos):.4f}", f"{np.std(aos):.4f}", f"{p_fixed:.4f}", f"{p_random:.4f}"])
            writer.writerow([name, 'fixed', f"{np.mean(fixed):.4f}", f"{np.std(fixed):.4f}", "", ""])
            writer.writerow([name, 'random', f"{np.mean(random):.4f}", f"{np.std(random):.4f}", "", ""])
    print("\nResults saved to 'all_igd_results.csv'")


if __name__ == '__main__':
    main()