# CDS526-Casestudy
# 🧠 Adaptive Operator Selection for Constrained Multi‑objective Optimization (AOS-NSGA-II)

**A Q‑learning based operator selection mechanism integrated into NSGA-II for solving constrained multi‑objective optimization problems.**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-orange.svg)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-green.svg)](https://matplotlib.org)
[![PyMoo](https://img.shields.io/badge/PyMoo-0.6+-red.svg)](https://pymoo.org)
[![SciPy](https://img.shields.io/badge/SciPy-1.10+-purple.svg)](https://scipy.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Colab](https://img.shields.io/badge/Run%20in-Colab-F9AB00.svg)](https://colab.research.google.com/github/coraagg/CDS526-Casestudy/blob/main/notebook/CF1_CTP_Runs.ipynb)

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

---

## 📁 Repository Structure

```text
CDS526-Casestudy/
├── notebook/                    # Jupyter/Colab notebooks
│   └── CF1_CTP_Runs.ipynb       # Main notebook for running experiments & analysis
├── src/                         # Source code
│   ├── nsga2_aos.py             # Main algorithm (NSGA-II + Q-learning)
│   ├── operators.py             # Three variation operators
│   ├── problem.py               # CF1 problem definition (with feasible Pareto front)
│   ├── q_selector.py            # Q-learning agent
│   ├── run_experiments.py       # Script to run AOS experiments (30 runs, 200 gens)
│   └── visualize.py             # Script to generate result plots
├── results/                     # Output directory 
│   ├── convergence_CF1.png
|   ├── convergence_CTP1.png
|   ├── convergence_CTP8.png
|   ├── boxplot_all_problems.png
|   ├── pareto_CF1.png
|   ├── pareto_CTP1.png
|   └── pareto_CTP8.png
├── requirements.txt 
├── README.md
└── .gitignore
```


---

## 📊 Key Results

| Problem | Method          | IGD (mean ± std) | Feasible Ratio | Wilcoxon p‑value |
|---------|----------------|------------------|----------------|------------------|
| CF1     | AOS-NSGA-II     | 0.258 ± 0.168    | 1.000           | 0.869            |
| CF1     | Fixed-NSGA-II   | 0.196 ± 0.073    | 1.000           | –                |
| CTP1    | AOS-NSGA-II     | 0.026 ± 0.005    | 1.000           | 0.000***         |
| CTP1    | Fixed-NSGA-II   | 0.050 ± 0.015    | 1.000           | –                |
| CTP8    | AOS-NSGA-II     | 0.050 ± 0.102    | 1.000           | 0.995            |
| CTP8    | Fixed-NSGA-II   | 0.050 ± 0.102    | 1.000           | –                |

> **Observations**: AOS is particularly beneficial for problems where operator selection matters (CTP1). On CF1 and CTP8, performance is comparable to the fixed operator.

**Convergence curves, boxplot, and Pareto front comparisons** are automatically generated in the `results/` folder.

---

## ⚙️ Setup and Installation (Reproducibility)

### Prerequisites
- Python 3.8 or higher
- Git

### Step‑by‑step

**1. Clone the repository**
```
git clone https://github.com/coraagg/CDS526-Casestudy.git
cd CDS526-Casestudy
```
**2. Create virtual environment (recommended)**
```
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```
**3. Install dependencies**
```
pip install -r requirements.txt
```

---

## ▶️ How to Run & Reproduce Results
### Run AOS Experiments (30 runs, 200 generations each)
```
python src/run_experiments.py
```
By default, the script runs CF1, CTP1, CTP8 and saves results as .pkl files in the results/ directory.
### Run Fixed‑Operator Baseline (for comparison)
The fixed‑operator version is implemented in the Colab notebook (see below). To run it locally, you can modify run_experiments.py to use the NSGA2_Fixed class defined in the notebook.
### Run a Single Problem with Custom Parameters
Modify src/run_experiments.py:
```
problem_list = ['cf1']   # or ['ctp1'], ['ctp8']
run_experiments(problem_list, n_runs=30, max_gen=200)
```

---

## 🖼️ Visualization
The following plots will be saved in results/:
- convergence_CF1.png – IGD convergence curves for CF1
- convergence_CTP1.png – IGD convergence curves for CTP1
- convergence_CTP8.png – IGD convergence curves for CTP8
- boxplot_all_problems.png – Boxplot of final IGD values
- pareto_CF1.png, pareto_CTP1.png, pareto_CTP8.png – Pareto front comparisons
Note: The true Pareto fronts are obtained from pymoo (for CTP series) or computed analytically for CF1 (feasible part only).

---

## 🚀 Reproducing the Full Study (Colab)

We provide an **end‑to‑end Colab notebook** that reproduces all experiments, including AOS and fixed‑operator runs, statistical tests, and result visualisation.

- **Open in Colab**: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/coraagg/CDS526-Casestudy/blob/main/notebook/CF1_CTP_Runs.ipynb)

The notebook is organised into the following cells:

1. **Mount Google Drive** – to access saved results and store new outputs.
2. **Install dependencies** – `pymoo`, `matplotlib`.
3. **Clone the repository** – fetches the latest code.
4. **Modify save path** – redirects `run_experiments.py` output to Google Drive.
5. **Run AOS experiments (example)** – runs AOS on `ctp8` (30 runs, 200 generations) to demonstrate the process.  
   *Full AOS results for CF1, CTP1 and CTP8 are already pre‑loaded from Drive in step 9.*
6. **Generate figures** – uses existing AOS results to create convergence curves, boxplot and Pareto front plots (saved as PNG).
7. **Define fixed‑operator NSGA‑II** – a variant of `NSGA2_AOS` that always uses Operator 0 (SBX+polynomial mutation).
8. **Run fixed‑operator experiments** – runs the fixed operator on CF1, CTP1 and CTP8 (30 runs each, 200 generations). Results are saved locally (or to Drive if modified).
9. **Load results and compute statistics** – loads AOS results (from Drive) and fixed‑operator results (from local `results_fixed/`), calculates final IGD and feasible ratios, performs Wilcoxon signed‑rank tests, and generates a LaTeX comparison table.

All figures and the final LaTeX table are saved to the `results/` directory (or Google Drive). The notebook can be run from start to finish to fully reproduce the experimental results reported in the paper.

---

## 🛠️ Technologies Used
- Python 3.8+ – Core programming language
- NumPy – Numerical computations
- Matplotlib – Visualization
- PyMoo – Benchmark problem definitions (CTP series) and Pareto front retrieval
- SciPy – Wilcoxon signed‑rank test
- Google Colab – Cloud execution environment

---

**Thank you for checking out our project!**
For any questions or suggestions, please open an issue or contact the team.




















