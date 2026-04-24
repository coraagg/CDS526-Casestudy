# AOS-NSGA-II for Constrained Multi‑objective Optimization

This repository implements an **Adaptive Operator Selection (AOS)** mechanism integrated into **NSGA‑II** for solving constrained multi‑objective optimization problems. A Q‑learning agent dynamically selects among three variation operators (SBX+polynomial mutation, DE mutation, uniform crossover+Gaussian mutation) based on the current population state, using the improvement in Inverted Generational Distance (IGD) as reward.

## Repository Structure

The code is organized into two main parts:

- **`member_ZHANG Chenguo/`** – Basic experimental framework provided by team member ZHANG Chenguo (basic NSGA‑II + Q‑learning, single‑constraint handling, Euclidean distance diversity).
- **`my_contributions/`** – Extensions and improvements made by CHEUNG Hong Yuk, including:
  - Multi‑constraint handling (correct non‑dominated sorting and parent selection for multiple constraints)
  - Configurable operator selection modes: `aos`, `fixed`, `random`
  - Crowding distance for population diversity (replaces Euclidean distance)
  - Support for additional benchmark problems: **CF1** (revised official version), **CTP1**, **CTP2**, **MW1**
  - Parallel experiment runner (multiprocessing) with checkpointing and progress bars
  - Analysis and visualization scripts

Because the original code was refactored and only a few core modules (e.g., `operators.py`, `q_selector.py`) were reused after modification, **I have not included the entire original repository**.

```
alternative_experiments
├── Inherited_Core_Files_By_ZCG_Baseline/ # Original code (unchanged)
│   ├── __init__.py		# Makes the directory a Python package
│   ├── nsga2_aos.py
│   ├── operators.py
│   ├── problem.py	# Original CF1 (approximated PF)
│   └── q_selector.py
│
├── cf1_pf_revised.py # Official CF1 with correct PF (f1+f2=1)
├── mw_ctp_problems.py # Wrappers for CTP1, CTP2, MW1 with custom PF
├── nsga2_aos_extended.py # Extended NSGA‑II with mode, crowding, multi-constraint
├── q_selector_extended.py # Q‑learning using crowding distance
├── operators_de_updated.py # DE/rand/1 operator (standard)
├── run_all_experiments_mw1_fast.py # Parallel experiment runner (all problems)
├── analysis_all.py # Statistical analysis (mean, std, Wilcoxon)
├── visualize_results.py # Generate convergence, boxplot, operator, PF figures
│
├── output_images/ # Automatically generated figures
├── optimized_statistics_table.csv # Statistical test of experimental results
└── README.md # This file
```

- The folder `Inherited_Core_Files_By_ZCG` contains the **original, unmodified** code written by team member ZHANG Chenguo.  
- All other `.py` files in the root directory are my work, extending the original framework with multi-constraint handling, crowding distance, parallel execution, and additional benchmark problems.  

## Dependencies

The code requires Python 3.8 or higher and the following Python packages:

- `numpy` – numerical operations
- `matplotlib` – plotting and visualization
- `scipy` – statistical tests (Wilcoxon signed-rank test)
- `pymoo` – multi‑objective optimization library (provides CTP1, CTP2, MW1 problems)
- `tqdm` – progress bars for parallel experiments

## How to Run

All experiments (CF1, CTP1, CTP2, MW1) are executed by the parallel runner.

### Option 1: Run from IDE
Open `run_all_experiments_mw1_fast.py` in your Python IDE (e.g., VS Code, PyCharm).  
Click the Run button.

### Option 2: Run from command line
Open a terminal in the project root directory and execute:
```bash
python run_all_experiments_mw1_fast.py
```

The script will:
- Run 30 independent trials for each problem and each mode (`aos`, `fixed`, `random`)
- Use 500 generations, population size 100, fixed base seed 42 (incremented per run: 42, 43, …, 71)
- Use crowding distance for diversity (`use_crowding=True`)
- Skip already completed runs (checkpointing)
- Show progress bars using `tqdm`

Results are saved in:
- `results/CF1/`
- `results/CTP/`
- `results/MW/`

### Statistical summary
```bash
python analysis_all.py
```
This prints a formatted table to the console and exports `optimized_statistics_table.csv` containing:
- Mean and standard deviation of final IGD for each algorithm
- Wilcoxon signed‑rank test p‑values (AOS vs Fixed, AOS vs Random)
- Significance markers (`***`, `**`, `*`)

### Generate figures
```bash
python visualization_all.py
```
All figures are saved in `output_images/` and include:
- Convergence curves (IGD over generations)
- Boxplots of final IGD across 30 runs
- Operator selection cumulative ratio (AOS mode only)
- Pareto front scatter plots (AOS mode only)

## Results

The experimental results show that:
- On **CF1**, AOS significantly outperforms the fixed operator (p < 0.001) and performs similarly to random selection.
- On **CTP1**, AOS achieves the lowest mean IGD and shows a marginally significant improvement over the fixed operator (p = 0.0523).
- On **CTP2**, all strategies perform similarly, but AOS exhibits low variance, confirming robustness.

For detailed numbers and plots, please refer to the generated CSV and image files.

## Acknowledgments
- Basic experimental framework by team member ZHANG Chenguo.
- Problem definitions based on CEC2009, CEC2020, and pymoo library.



# Experimental Reflections & Takeaways
Although this experiment was not submitted as the final submission, its results and the associated engineering practices yielded several important insights for this study.

MW series problems have a higher dimension of decision variables and more complex nonlinear constraints. Among them, MW1 also has an unconnected feasible region topology. The convergence difficulty of the algorithm on this problem is significantly higher than that of CF1, CTP1, and CTP2. The 500 generations of running iterations may still not be sufficient for Q-learning to fully explore the state-action space and converge to the optimal strategy.

The diversity metric does not align well with the topology of the problem's feasible region, and thus fails to capture meaningful state features. As a result, the adaptive operator selection mechanism fails to reliably choose the most appropriate operator.

This also highlights the importance of carefully selecting benchmark problems. Their properties must be sufficiently compatible with the designed algorithm to ensure fair and meaningful evaluation. Different problem classes may require distinct search strategies, which can prevent accurate evaluation of the proposed operators' actual performance.

The multi-process parallel experimental framework significantly reduces the total runtime of large-scale repeated experiments, demonstrating clear advantages in computational efficiency. This framework can be reused in future, more comprehensive benchmark studies to further reduce computational overhead.
