# -*- coding: utf-8 -*-
"""
Visualization script for experimental results.
Author: CHEUNG Hong Yuk (Student ID: 5506205)
Date: April 2026
Description: Loads experiment results for CF1, CTP1, CTP2, MW1,
             and generates four types of figures:
             - Convergence curves (IGD over generations) for all three modes
             - Boxplots of final IGD across 30 runs
             - Operator selection cumulative ratio plot (AOS mode only; skipped for MW1)
             - Pareto front scatter plot (AOS mode only, with approximated/true PF)
             All figures are saved to the 'output_images/' directory.
"""

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

if not os.path.exists("output_images"):
    os.makedirs("output_images")

# Setting
PROBLEMS = ["CF1", "CTP1", "CTP2", "MW1"]
MODES = ["aos", "fixed", "random"]
LABELS = ["AOS", "Fixed", "Random"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


# Load data
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


# 1. Convergence curve
def plot_convergence_single(problem, data):
    plt.figure(figsize=(10, 5))
    for i, mode in enumerate(MODES):
        igd_mat = np.array([run["igd_history"] for run in data[mode]])
        mean = igd_mat.mean(axis=0)
        std = igd_mat.std(axis=0)
        plt.plot(mean, label=LABELS[i], color=COLORS[i], linewidth=2)
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2, color=COLORS[i])

    plt.xlabel("Generation")
    plt.ylabel("IGD")
    plt.title(f"{problem} Convergence Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"output_images/{problem}_convergence.png", dpi=300)
    plt.close()


# 2. Box plot
def plot_boxplot_single(problem, data):
    plt.figure(figsize=(7, 5))
    values = [[run["igd_history"][-1] for run in data[m]] for m in MODES]

    bp = plt.boxplot(
        values,
        labels=LABELS,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2)
    )

    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)

    plt.title(f"{problem} IGD Boxplot (30 runs)")
    plt.ylabel("Final IGD")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"output_images/{problem}_boxplot.png", dpi=300)
    plt.close()


# 3. Operator frequency graph(AOS)
def plot_operator_single(problem, data):
    if problem == "MW1":
        return
    run0 = data["aos"][0]
    op_hist = run0["operator_history"]
    gens = len(op_hist)
    cnt = np.zeros((gens, 3))
    for g, op in enumerate(op_hist):
        cnt[g, op] = 1
    ratio = np.cumsum(cnt, axis=0) / np.arange(1, gens + 1)[:, None]

    plt.figure(figsize=(10, 4))
    plt.plot(ratio[:, 0], label="SBX", linewidth=2)
    plt.plot(ratio[:, 1], label="DE", linewidth=2)
    plt.plot(ratio[:, 2], label="PCD", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Cumulative Selection Ratio")
    plt.title(f"{problem} AOS Operator Selection")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"output_images/{problem}_operator.png", dpi=300)
    plt.close()


# 4. Pareto frontier scatter plot(AOS)
def plot_pf_single(problem, data):
    if problem == "CF1":
        from cf1_pf_revised import CF1
        prob = CF1()
    elif problem == "CTP1":
        from mw_ctp_problems import CTP1_Wrapper
        prob = CTP1_Wrapper()
    elif problem == "CTP2":
        from mw_ctp_problems import CTP2_Wrapper
        prob = CTP2_Wrapper()
    elif problem == "MW1":
        from mw_ctp_problems import MW1_Wrapper
        prob = MW1_Wrapper()
    else:
        return

    pf = prob.pareto_front()

    plt.figure(figsize=(7, 5))
    plt.plot(pf[:, 0], pf[:, 1], 'r--', label="True PF", linewidth=2)

    F = data["aos"][0]["final_F"]
    plt.scatter(F[:, 0], F[:, 1], s=12, alpha=0.8, color="#1f77b4", label="AOS")

    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title(f"{problem} Final Pareto Front (AOS Only)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"output_images/{problem}_pareto_front.png", dpi=300)
    plt.close()


# Main
if __name__ == "__main__":
    print("Generating all figures...")

    for prob in PROBLEMS:
        print(f"Processing {prob}...")
        data = load_problem(prob)
        plot_convergence_single(prob, data)
        plot_boxplot_single(prob, data)
        plot_operator_single(prob, data)
        plot_pf_single(prob, data)

    print("\nALL 16 IMAGES SAVED TO output_images/")