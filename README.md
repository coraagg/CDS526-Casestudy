# CDS526-Casestudy
# 🧠 Adaptive Operator Selection for Constrained Multi‑objective Optimization (AOS-NSGA-II)

**A Q‑learning based operator selection mechanism integrated into NSGA-II for solving constrained multi‑objective optimization problems.**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-orange.svg)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-green.svg)](https://matplotlib.org)
[![PyMoo](https://img.shields.io/badge/PyMoo-0.6+-red.svg)](https://pymoo.org)
[![SciPy](https://img.shields.io/badge/SciPy-1.10+-purple.svg)](https://scipy.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Colab](https://img.shields.io/badge/Run%20in-Colab-F9AB00.svg)](https://colab.research.google.com/github/coraagg/CDS526-Casestudy/blob/main/CF1_CTP_Runs.ipynb)

---

## 📌 Project Overview

Constrained multi‑objective optimization problems (CMOPs) require balancing **convergence**, **diversity**, and **feasibility** under constraints. This project implements an **Adaptive Operator Selection (AOS)** mechanism integrated into **NSGA-II** to dynamically choose the most suitable variation operator during the search process. A **Q‑learning agent** observes the current population state (feasible ratio and recent IGD improvement) and selects among three operators:

- **Operator 0**: SBX crossover + polynomial mutation (local exploitation)
- **Operator 1**: DE/rand/1 mutation (global exploration)
- **Operator 2**: Uniform crossover + Gaussian mutation (diversity enhancement)

The agent receives the **improvement in Inverted Generational Distance (IGD)** as reward, enabling online adaptation without problem‑specific tuning.

We compare the proposed **AOS-NSGA-II** against a **Fixed-NSGA-II** baseline that always uses Operator 0. Experiments are performed on three benchmark problems: **CF1** (CEC2009), **CTP1** and **CTP8** (classic CTP series). Each problem is run 30 independent times with 200 generations per run.

**Key Results** (mean ± std over 30 runs):
- ✅ AOS-NSGA-II achieves stable convergence and produces solution sets close to the true Pareto fronts.
- ✅ On **CTP1**, AOS significantly outperforms the fixed operator (**p < 0.001**), with IGD = 0.026 ± 0.005 vs 0.050 ± 0.015.
- ✅ Feasible solution ratio is **1.0** for all problems and both methods.
- ✅ On CF1 and CTP8, performance is comparable (no statistically significant difference).

---

## 📋 Table of Contents
- [Repository Structure](#-repository-structure)
- [Key Results](#-key-results)
- [Setup and Installation](#-setup-and-installation-reproducibility)
- [How to Run](#-how-to-run--reproduce-results)
- [Visualization](#-visualization)
- [Reproducing the Full Study (Colab)](#-reproducing-the-full-study-colab)
- [Technologies Used](#-technologies-used)
- [Team Members](#-team-members)

---

## 📁 Repository Structure

```text


