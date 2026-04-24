# -*- coding: utf-8 -*-
"""
Statistical analysis script for experimental results.
Author: CHEUNG Hong Yuk (Student ID: 5506205)
Date: April 2026
Description: Loads experiment results from pickle files (supports CF1, CTP1, CTP2, MW1),
             computes mean and standard deviation of final IGD for each algorithm (AOS, fixed, random),
             performs Wilcoxon signed-rank test between AOS and baselines,
             and exports an optimized statistics table (CSV) and prints formatted results.
             Decimal places and significance markers (***, **, *) are configurable.
"""

import pickle
import os
import csv
import numpy as np
from scipy.stats import wilcoxon

# setting
PROBLEMS = ["CF1", "CTP1", "CTP2", "MW1"]
MODES = ["aos", "fixed", "random"]
LABELS = ["AOS", "Fixed", "Random"]
DECIMAL_PLACES = 4

# load data
def load_problem(problem):
    if problem == "MW1":
        folder = "results/MW"
    elif problem in ["CTP1", "CTP2"]:
        folder = "results/CTP"
    else:
        folder = f"results/{problem}"

    data = {}
    for mode in MODES:
        if problem == "CF1":
            fname = f"{problem}_{mode}_all.pkl"
        else:
            fname = f"{problem}_Wrapper_{mode}_all.pkl"
        path = os.path.join(folder, fname)
        with open(path, "rb") as f:
            data[mode] = pickle.load(f)
    return data

# calculate IGD
def get_igd_stats(runs):
    igd_list = [r["igd_history"][-1] for r in runs]
    mean = round(np.mean(igd_list), DECIMAL_PLACES)
    std = round(np.std(igd_list), DECIMAL_PLACES)
    return mean, std, igd_list

# Wilcoxon test
def wilcoxon_test(x, y):
    try:
        stat, p = wilcoxon(x, y)
        return round(p, DECIMAL_PLACES)  # p值保留4位小数
    except:
        return round(1.0, DECIMAL_PLACES)

# Significance marking (retain p-values, additional annotation)
def get_significance(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

# save as CSV
def export_optimized_csv():
    rows = []
    header = [
        "Problem", "Algorithm", "IGD_Mean", "IGD_Std",
        "p_AOS_vs_Fixed", "Sig_AOS_vs_Fixed",
        "p_AOS_vs_Random", "Sig_AOS_vs_Random"
    ]
    rows.append(header)

    for prob in PROBLEMS:
        data = load_problem(prob)
        # calculate the statistics of AOS
        aos_mean, aos_std, aos_igd = get_igd_stats(data["aos"])
        # calculate the statistics of Fixed and Random
        fix_mean, fix_std, fix_igd = get_igd_stats(data["fixed"])
        rnd_mean, rnd_std, rnd_igd = get_igd_stats(data["random"])
        # test
        p_fix = wilcoxon_test(aos_igd, fix_igd)
        p_rnd = wilcoxon_test(aos_igd, rnd_igd)
        # sign for significance
        sig_fix = get_significance(p_fix)
        sig_rnd = get_significance(p_rnd)

        rows.append([prob, "AOS", aos_mean, aos_std, p_fix, sig_fix, p_rnd, sig_rnd])
        rows.append([prob, "Fixed", fix_mean, fix_std, "", "", "", ""])
        rows.append([prob, "Random", rnd_mean, rnd_std, "", "", "", ""])

    with open("optimized_statistics_table.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

# print
def print_optimized_statistics():
    print("\n" + "=" * 110)
    print(f" OPTIMIZED STATISTICAL RESULTS (Decimal: {DECIMAL_PLACES} places)")
    print("=" * 110)
    print(f"{'Prob':<6} {'Alg':<8} {'IGD(Mean±Std)':<20} {'AOS vs Fixed(p)':<16} {'AOS vs Random(p)':<16}")
    print("-" * 110)

    for prob in PROBLEMS:
        data = load_problem(prob)
        aos_mean, aos_std, aos_igd = get_igd_stats(data["aos"])
        fix_mean, fix_std, fix_igd = get_igd_stats(data["fixed"])
        rnd_mean, rnd_std, rnd_igd = get_igd_stats(data["random"])
        p_fix = wilcoxon_test(aos_igd, fix_igd)
        p_rnd = wilcoxon_test(aos_igd, rnd_igd)
        sig_fix = get_significance(p_fix)
        sig_rnd = get_significance(p_rnd)

        print(f"{prob:<6} AOS     {aos_mean:.{DECIMAL_PLACES}f}±{aos_std:.{DECIMAL_PLACES}f}    {p_fix:.{DECIMAL_PLACES}f} {sig_fix:<3}    {p_rnd:.{DECIMAL_PLACES}f} {sig_rnd:<3}")
        print(f"{'':<6} Fixed   {fix_mean:.{DECIMAL_PLACES}f}±{fix_std:.{DECIMAL_PLACES}f}")
        print(f"{'':<6} Random  {rnd_mean:.{DECIMAL_PLACES}f}±{rnd_std:.{DECIMAL_PLACES}f}")
        print("-" * 110)

# run
if __name__ == "__main__":
    print(f"Generating optimized table (Decimal: {DECIMAL_PLACES} places)...")
    print_optimized_statistics()
    export_optimized_csv()
    print("\nOptimized CSV saved: optimized_statistics_table.csv")