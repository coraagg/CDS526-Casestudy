# CDS526-Casestudy
# AOS-NSGA-II for Constrained Multi‑objective Optimization

This repository contains the implementation of an **Adaptive Operator Selection (AOS)** mechanism integrated into **NSGA-II** for solving constrained multi‑objective optimization problems (CMOPs). The AOS uses Q‑learning to dynamically choose the most suitable variation operator during evolution. Experiments are conducted on standard CMOP benchmarks (CF, CTP, MW series) using the `pymoo` library.
 

## Dependencies

- Python 3.8 or higher  
- `numpy`  
- `matplotlib`  
- `pymoo` (version 0.6.0 or later)  
- `scipy` (for statistical tests)

Install all dependencies with:

```bash
pip install -r requirements.txt
