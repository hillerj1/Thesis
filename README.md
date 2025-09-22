# Josh’s Thesis: 4th-Order Finite Difference Quantum Eigen-problems
 
This repo is my senior thesis project for Union College Physics. The main idea is to build a **better numerical solver** for quantum systems by using a **4th-order finite difference method** instead of the usual 2nd-order. That way, we can get more accurate eigenvalues and eigenfunctions of the Schrödinger equation without blowing up the grid size.

---

## What’s here:

- `fd1d_QSHO.ipynb` → Notebook for the **quantum simple harmonic oscillator** (QSHO).  
- `hamiltonian` (2nd order) and `hamiltonian_4th` (4th order) functions → build sparse Hamiltonian matrices.  
- Convergence studies → check how errors scale with grid spacing (`dx`). Expect slope ≈ 2 for 2nd order, ≈ 4 for 4th order.  
- Timing tests → see how much work ARPACK has to do for each method.  
- Plots → eigenfunctions, spectra, comparisons between potentials.  

---

## How to run

1. Set up JupyterLab or VSCode with Python 3.  
2. Install the basics:
   ```bash
   pip install numpy scipy matplotlib
