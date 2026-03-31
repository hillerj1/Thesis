# Senior thesis: finite-difference quantum eigenvalue problems

This project implements 2nd- and 4th-order finite-difference discretizations for 1D quantum eigenvalue problems (time-independent Schrödinger equation) and produces figures for the thesis (ISW + QSHO benchmarks).

## Project overview

We solve eigenvalue problems of the form

$$-\frac{1}{2}\psi''(x) + V(x)\psi(x) = E\psi(x),\qquad \psi(x_{\rm left})=\psi(x_{\rm right})=0,$$

by discretizing the domain on a uniform interior grid and assembling a sparse Hamiltonian matrix.

## Repository structure

### Core modules
- `finite_difference_quantum.py`: grid utilities (`spatial`, `proj`) and sparse finite-difference Hamiltonian assembly via `hamiltonian(V, x_left, x_right, N, order=2|4)`.
- `eigenvalue_analysis.py`: analytical reference formulas, numerical eigenvalue solvers, error analysis, and convergence study helpers for ISW and QSHO.

### Analysis notebooks
- `analysis_framework.ipynb`: comprehensive notebook covering convergence, timing, spectra, wavefunctions, and error studies.
- `fd4_ISW_clean.ipynb`: 4th-order ISW analysis (clean).
- `fd4_QSHO_clean.ipynb`: 4th-order QSHO analysis (clean).
- `fd2_QSHO.ipynb`: 2nd-order QSHO analysis.
- `isw_convergence_simple.ipynb`: simplified ISW convergence workflow.
- `eigenvalue_analysis_demo.ipynb`: walkthrough of the `eigenvalue_analysis` module API.

### Figure generation scripts (recommended for thesis figures)
- `scripts/isw_convergence.ipynb`: generates `images/isw_convergence.png` — 2nd vs 4th order log-log error plot (extended to N=6000).
- `scripts/qsho_convergence.ipynb`: generates `images/qsho_convergence.png` — same structure as ISW; includes `x_inf` domain cutoff.
- `scripts/isw_eigenvalue_error_comparison.ipynb`: generates `images/isw_numerical_vs_analytical_eigenvalues.png` — numerical vs analytical + relative error panel.
- `scripts/generate_isw_convergence.py`: CLI script for ISW convergence plots (`--N-min`, `--N-max`, `--N-step`, `--L`).

### Outputs
- `images/`: generated plots and schematics used in the thesis.

## Numerical methods

- **2nd order**: standard 3-point stencil → tridiagonal sparse matrix.
- **4th order**: 5-point interior stencil with boundary closures → pentadiagonal sparse matrix. The boundary rows can make the assembled operator *slightly non-symmetric*; `eigs` (general ARPACK) with shift-invert is more robust than `eigsh` in those cases.

## Setup

```bash
pip install numpy scipy matplotlib jupyter
```

## Quick start

```python
from finite_difference_quantum import hamiltonian
from scipy.sparse.linalg import eigsh
import numpy as np

# Infinite square well on [0, 1]
V = lambda x: 0.0
H = hamiltonian(V, 0.0, 1.0, N=500, order=4)
eigenvalues, eigenvectors = eigsh(H, k=5, which='SM')
print(np.sort(eigenvalues))
# Analytical: n²π²/2 → [4.935, 19.739, 44.413, 78.957, 123.370]
```
