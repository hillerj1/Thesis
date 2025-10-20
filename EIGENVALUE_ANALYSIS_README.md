# Eigenvalue Analysis Module

## Overview

`eigenvalue_analysis.py` provides functions for analyzing finite difference methods applied to quantum eigenvalue problems. All eigenvalues are computed in a single call to avoid redundant iterations.

## Main Functions

### Analytical Solutions
- `analytical_ISW_eigenvalues(n_eigenvalues, L=1.0)` - Exact ISW eigenvalues
- `analytical_QSHO_eigenvalues(n_eigenvalues, omega=1.0)` - Exact QSHO eigenvalues

### Numerical Solutions
- `numerical_ISW_eigenvalues(n_eigenvalues, n_grid_points, order=4, L=1.0)`
- `numerical_QSHO_eigenvalues(n_eigenvalues, n_grid_points, order=4, omega=1.0, x_max=None)`

**Key**: All n_eigenvalues computed in a single `eigsh()` call for efficiency.

### Analysis
- `compute_eigenvalue_discrepancies(numerical, analytical)` - Returns absolute and relative errors
- `analyze_ISW_accuracy(...)` - Complete ISW analysis workflow
- `analyze_QSHO_accuracy(...)` - Complete QSHO analysis workflow
- `convergence_study_ISW(...)` - Multi-grid convergence study
- `convergence_study_QSHO(...)` - Multi-grid convergence study

## Quick Start

```python
import eigenvalue_analysis as eva

# Analyze first 5 ISW eigenvalues
results = eva.analyze_ISW_accuracy(n_eigenvalues=5, n_grid_points=201, order=4)
print(f"Absolute errors: {results['absolute_errors']}")

# Convergence study
grid_points = [51, 101, 201, 401, 801]
conv = eva.convergence_study_ISW(n_eigenvalues=1, grid_points_list=grid_points, order=4)
```

## Domain Settings
- **ISW**: Fixed domain [0, 1]
- **QSHO**: Auto-determined based on highest eigenvalue (can be manually set with `x_max`)
