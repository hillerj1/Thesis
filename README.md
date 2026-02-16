# Senior thesis: finite-difference quantum eigenvalue problems

This project implements 2nd- and 4th-order finite-difference discretizations for 1D quantum eigenvalue problems (time-independent Schrödinger equation) and produces figures for the thesis (ISW + QSHO benchmarks).

## Project overview

We solve eigenvalue problems of the form

$$-\frac{1}{2}\psi''(x) + V(x)\psi(x) = E\psi(x),\qquad \psi(x_{\rm left})=\psi(x_{\rm right})=0,$$

by discretizing the domain on a uniform interior grid and assembling a sparse Hamiltonian matrix.

## Repository structure

### Core modules
- `finite_difference_quantum.py`: grid utilities and sparse finite-difference Hamiltonian assembly (`order=2` or `order=4`).
- `eigenvalue_analysis.py`: analytical reference formulas and numerical wrappers for eigenvalue computations (used by some plotting scripts).

### Analysis and figure generation
- `analysis_framework.ipynb`: larger “kitchen sink” notebook (convergence, timing, spectra, wavefunctions, error studies).
- `generate_figure.py`: prompt-based local figure generator (routes a caption-like prompt to a matplotlib figure and writes a PNG to `images/`).

### Standalone notebooks (recommended for thesis figures)
- `scripts/isw_convergence.ipynb`: generates `images/isw_convergence.png` (60 points, extended to \(N=6000\)).
- `scripts/qsho_convergence.ipynb`: generates `images/qsho_convergence.png` (same structure as ISW; includes `x_inf` domain cutoff).
- `scripts/isw_eigenvalue_error_comparison.ipynb`: generates `images/isw_numerical_vs_analytical_eigenvalues.png` (numerical vs analytical + relative error panel).

### Outputs
- `images/`: generated plots and schematics used in the thesis.
- `thesis_paper/`: LaTeX thesis (`thesis_paper.tex`) and bibliography (`references.bib`).

## Numerical methods (notes)

- **2nd order** uses the standard 3-point stencil and produces a tridiagonal sparse matrix.
- **4th order** uses a 5-point interior stencil plus boundary closures; the boundary rows can make the assembled operator *slightly non-symmetric*. In those cases, `eigs` (general ARPACK) with shift-invert is more robust than `eigsh` (Hermitian ARPACK).

## Setup

### Python dependencies

```bash
pip install numpy scipy matplotlib jupyter
```

## Thesis paper

The LaTeX thesis is in `thesis_paper/`:
- `thesis_paper/thesis_paper.tex`
- `thesis_paper/references.bib`