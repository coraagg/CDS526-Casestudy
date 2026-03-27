# CDS526-Casestudy
# AOS-NSGA-II for Constrained Multi‑objective Optimization

This repository implements an **Adaptive Operator Selection (AOS)** mechanism integrated into **NSGA-II** for solving constrained multi‑objective optimization problems. A Q‑learning agent dynamically selects among three variation operators (SBX+polynomial mutation, DE mutation, uniform crossover+Gaussian mutation) based on the current population state, using Inverted Generational Distance (IGD) improvement as the reward.

The method is evaluated on the **CF1** benchmark problem from the CEC2009 constrained multi‑objective competition.

## Dependencies

- Python 3.8 or higher
- `numpy`
- `matplotlib`
- `pymoo` (>=0.6.0)
- `scipy` (optional, for statistical tests)

Install all required packages with:

```bash
pip install -r requirements.txt



# How to Run

## Run Experiments

To run 30 independent trials on the CF1 problem (200 generations each):

```bash
python run_experiments.py


```markdown
# Code Structure

- `nsga2_aos.py` – Main algorithm (NSGA-II + Q-learning)
- `operators.py` – Three variation operators
- `problem.py` – CF1 problem definition
- `q_selector.py` – Q-learning agent
- `run_experiments.py` – Script to run 30 independent trials
- `visualize.py` – Script to generate result plots









## Dependenc

