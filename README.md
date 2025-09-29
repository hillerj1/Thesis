# Josh's Thesis: 4th-Order Finite Difference Quantum Eigen-problems

This repository contains my senior thesis project for Union College Physics. The main objective is to develop a **more accurate numerical solver** for quantum systems by implementing a **4th-order finite difference method** as an alternative to the standard 2nd-order approach. This enables us to obtain higher precision eigenvalues and eigenfunctions of the Schrödinger equation without significantly increasing computational complexity.

## 🎯 Project Overview

The project focuses on solving the time-independent Schrödinger equation:

$$-\frac{1}{2}\nabla^2\psi + V(x)\psi = E\psi$$

Using finite difference methods to discretize the kinetic energy operator $-\frac{1}{2}\nabla^2$ and solve the resulting eigenvalue problem using sparse matrix techniques.

## 📁 Repository Structure

### Core Files
- **`fd4_QSHO_clean.ipynb`** → Main analysis notebook for the Quantum Simple Harmonic Oscillator (QSHO) with 4th-order finite differences
- **`fd2_QSHO.ipynb`** → Earlier implementation with 2nd-order finite differences for comparison
- **`schrodinger_fd1d.py`** → Standalone Python module containing all finite difference functions

### Supporting Files
- **`thesis_paper.sty`** → LaTeX style file for thesis document formatting
- **`__pycache__/`** → Python bytecode cache directory

## 🔧 Technical Implementation

### Finite Difference Methods

#### 2nd-Order Method
- Standard three-point stencil for second derivatives
- Error scales as $\mathcal{O}(\Delta x^2)$
- Tridiagonal matrix structure

#### 4th-Order Method  
- Five-point stencil for improved accuracy
- Error scales as $\mathcal{O}(\Delta x^4)$
- Pentadiagonal matrix structure
- Includes boundary condition corrections

### Mathematical Formulation

The finite difference approximation for the second derivative operator can be expressed as:

**2nd-Order Stencil:**
$$\frac{d^2\psi}{dx^2} \approx \frac{\psi_{i+1} - 2\psi_i + \psi_{i-1}}{(\Delta x)^2}$$

**4th-Order Stencil:**
$$\frac{d^2\psi}{dx^2} \approx \frac{-\psi_{i+2} + 16\psi_{i+1} - 30\psi_i + 16\psi_{i-1} - \psi_{i-2}}{12(\Delta x)^2}$$

### Key Functions

```python
# Matrix assembly
sparseMatrixMaker(A, B, C, grid)      # 2nd-order
sparse_Matrix_Maker4(A, B, C, grid)   # 4th-order

# Hamiltonian construction
hamiltonian(V, x_left, x_right, N)    # 2nd-order
hamiltonian_4th(V, x_left, x_right, N) # 4th-order
```

### Analysis Features

- **Convergence Studies**: Error scaling analysis with log-log plots
- **Timing Comparisons**: Performance benchmarks between methods
- **Matrix Verification**: Validation against analytical solutions
- **Eigenvalue Analysis**: Ground state and excited state calculations

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- JupyterLab or VSCode with Python extension

### Required Packages
```bash
pip install numpy scipy matplotlib
```