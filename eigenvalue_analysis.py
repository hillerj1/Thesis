"""
Eigenvalue Analysis Module for Quantum Systems

Provides functions for analyzing finite difference methods applied to quantum
mechanical eigenvalue problems. All eigenvalues are computed in a single call
to avoid redundant iterative computations.
"""

import numpy as np
from scipy.sparse.linalg import eigsh
import finite_difference_quantum as fdq


# =============================================================================
# Analytical Eigenvalue Functions
# =============================================================================

def analytical_ISW_eigenvalues(n_eigenvalues, L=1.0):
    """
    Analytical eigenvalues for Infinite Square Well.
    E_n = n^2 * pi^2 / (2 * L^2) for n = 1, 2, 3, ...
    """
    n = np.arange(1, n_eigenvalues + 1)
    return n**2 * np.pi**2 / (2 * L**2)


def analytical_QSHO_eigenvalues(n_eigenvalues, omega=1.0):
    """
    Analytical eigenvalues for Quantum Simple Harmonic Oscillator.
    E_n = (n + 1/2) * omega for n = 0, 1, 2, ...
    """
    n = np.arange(n_eigenvalues)
    return (n + 0.5) * omega


# =============================================================================
# Numerical Eigenvalue Functions
# =============================================================================

def numerical_ISW_eigenvalues(n_eigenvalues, n_grid_points, order=4, L=1.0):
    """
    Numerical eigenvalues for ISW using finite differences.
    Computes all n_eigenvalues in a single call for efficiency.
    """
    
    def V_ISW(x):
        return np.zeros_like(x)
    
    x_left, x_right = 0.0, L
    H = fdq.hamiltonian(V_ISW, x_left, x_right, n_grid_points, order=order)
    
    # Adaptive parameters based on both matrix size (N) and number of eigenvalues (k)
    base_maxiter = 20000  # Increased base
    base_tol = 1e-9  # Slightly relaxed default tolerance
    
    # Scale maxiter based on N
    if n_grid_points > 5000:
        maxiter_multiplier = 10.0
        tol_val = 1e-7
    elif n_grid_points > 2000:
        maxiter_multiplier = 6.0
        tol_val = 1e-8
    else:
        maxiter_multiplier = 1.0
        tol_val = base_tol
    
    # Scale maxiter based on k (number of eigenvalues)
    if n_eigenvalues > 5:
        k_multiplier = 2.0 + 0.2 * (n_eigenvalues - 5)  # Extra 20% per eigenvalue above 5
    elif n_eigenvalues > 1:
        k_multiplier = 1.0 + 0.1 * (n_eigenvalues - 1)  # Extra 10% per eigenvalue above 1
    else:
        k_multiplier = 1.0
    
    # 2nd order may be less well-conditioned, so give it more iterations
    if order == 2:
        order_multiplier = 2.0  # Increased from 1.5
    else:
        order_multiplier = 1.0
    
    maxiter_val = int(base_maxiter * maxiter_multiplier * k_multiplier * order_multiplier)
    
    # Increase ncv (number of Arnoldi vectors) for better convergence
    ncv_val = min(max(2*n_eigenvalues + 1, n_eigenvalues + 2), n_grid_points, 50)
    
    try:
        eigenvalues, _ = eigsh(H, k=n_eigenvalues, which='SM', tol=tol_val, 
                              maxiter=maxiter_val, ncv=ncv_val)
    except Exception as e:
        # Fallback: try with relaxed tolerance
        try:
            eigenvalues, _ = eigsh(H, k=n_eigenvalues, which='SM', tol=tol_val*10, 
                                  maxiter=maxiter_val*2, ncv=ncv_val)
        except Exception as e2:
            # Last resort: try with even more relaxed parameters
            eigenvalues, _ = eigsh(H, k=n_eigenvalues, which='SM', tol=1e-6, 
                                  maxiter=maxiter_val*3, ncv=min(ncv_val*2, n_grid_points))
    
    return np.sort(eigenvalues)


def numerical_QSHO_eigenvalues(n_eigenvalues, n_grid_points, order=4, omega=1.0, x_max=None):
    """
    Numerical eigenvalues for QSHO using finite differences.
    Computes all n_eigenvalues in a single call for efficiency.
    x_max is auto-determined if not specified.
    """

    if x_max is None:
        E_max = (n_eigenvalues - 1 + 0.5) * omega
        x_classical = np.sqrt(2 * E_max / omega)
        x_max = x_classical + 8.0
    
    def V_QSHO(x):
        return 0.5 * omega * x**2
    
    x_left, x_right = -x_max, x_max
    H = fdq.hamiltonian(V_QSHO, x_left, x_right, n_grid_points, order=order)
    
    # Adaptive parameters based on both matrix size (N) and number of eigenvalues (k)
    base_maxiter = 20000  # Increased base
    base_tol = 1e-9  # Slightly relaxed default tolerance
    
    # Scale maxiter based on N
    if n_grid_points > 5000:
        maxiter_multiplier = 10.0
        tol_val = 1e-7
    elif n_grid_points > 2000:
        maxiter_multiplier = 6.0
        tol_val = 1e-8
    else:
        maxiter_multiplier = 1.0
        tol_val = base_tol
    
    # Scale maxiter based on k (number of eigenvalues)
    if n_eigenvalues > 5:
        k_multiplier = 2.0 + 0.2 * (n_eigenvalues - 5)  # Extra 20% per eigenvalue above 5
    elif n_eigenvalues > 1:
        k_multiplier = 1.0 + 0.1 * (n_eigenvalues - 1)  # Extra 10% per eigenvalue above 1
    else:
        k_multiplier = 1.0
    
    # 2nd order may be less well-conditioned, so give it more iterations
    if order == 2:
        order_multiplier = 2.0  # Increased from 1.5
    else:
        order_multiplier = 1.0
    
    maxiter_val = int(base_maxiter * maxiter_multiplier * k_multiplier * order_multiplier)
    
    # Increase ncv (number of Arnoldi vectors) for better convergence
    ncv_val = min(max(2*n_eigenvalues + 1, n_eigenvalues + 2), n_grid_points, 50)
    
    try:
        eigenvalues, _ = eigsh(H, k=n_eigenvalues, which='SM', tol=tol_val, 
                              maxiter=maxiter_val, ncv=ncv_val)
    except Exception as e:
        # Fallback: try with relaxed tolerance
        try:
            eigenvalues, _ = eigsh(H, k=n_eigenvalues, which='SM', tol=tol_val*10, 
                                  maxiter=maxiter_val*2, ncv=ncv_val)
        except Exception as e2:
            # Last resort: try with even more relaxed parameters
            eigenvalues, _ = eigsh(H, k=n_eigenvalues, which='SM', tol=1e-6, 
                                  maxiter=maxiter_val*3, ncv=min(ncv_val*2, n_grid_points))
    
    return np.sort(eigenvalues)


# =============================================================================
# Discrepancy Analysis Functions
# =============================================================================

def compute_eigenvalue_discrepancies(numerical_eigenvalues, analytical_eigenvalues):
    """
    Compute absolute and relative errors between numerical and analytical eigenvalues.
    """
    numerical = np.asarray(numerical_eigenvalues)
    analytical = np.asarray(analytical_eigenvalues)
    
    absolute_errors = np.abs(numerical - analytical)
    relative_errors = np.abs((numerical - analytical) / analytical)
    
    return absolute_errors, relative_errors


def analyze_ISW_accuracy(n_eigenvalues, n_grid_points, order=4, L=1.0):
    """
    Complete accuracy analysis for ISW.
    Returns dict with analytical, numerical, errors, and grid info.
    """
    analytical = analytical_ISW_eigenvalues(n_eigenvalues, L=L)
    numerical = numerical_ISW_eigenvalues(n_eigenvalues, n_grid_points, order=order, L=L)
    abs_errors, rel_errors = compute_eigenvalue_discrepancies(numerical, analytical)
    dx = L / (n_grid_points + 1)
    
    return {
        'analytical': analytical,
        'numerical': numerical,
        'absolute_errors': abs_errors,
        'relative_errors': rel_errors,
        'grid_spacing': dx,
        'n_grid_points': n_grid_points,
        'order': order
    }


def analyze_QSHO_accuracy(n_eigenvalues, n_grid_points, order=4, omega=1.0, x_max=None):
    """
    Complete accuracy analysis for QSHO.
    Returns dict with analytical, numerical, errors, and grid info.
    """
    analytical = analytical_QSHO_eigenvalues(n_eigenvalues, omega=omega)
    numerical = numerical_QSHO_eigenvalues(n_eigenvalues, n_grid_points, 
                                          order=order, omega=omega, x_max=x_max)
    abs_errors, rel_errors = compute_eigenvalue_discrepancies(numerical, analytical)
    
    if x_max is None:
        E_max = (n_eigenvalues - 1 + 0.5) * omega
        x_classical = np.sqrt(2 * E_max / omega)
        x_max = x_classical + 8.0
    
    dx = (2 * x_max) / (n_grid_points + 1)
    
    return {
        'analytical': analytical,
        'numerical': numerical,
        'absolute_errors': abs_errors,
        'relative_errors': rel_errors,
        'grid_spacing': dx,
        'n_grid_points': n_grid_points,
        'order': order,
        'domain': [-x_max, x_max]
    }


# =============================================================================
# Convergence Analysis Functions
# =============================================================================

def convergence_study_ISW(n_eigenvalues, grid_points_list, order=4, L=1.0):
    """
    Study convergence for ISW across different grid resolutions.
    Returns dict with grid_spacings, errors (2D arrays), and analytical values.
    """
    analytical = analytical_ISW_eigenvalues(n_eigenvalues, L=L)
    n_grids = len(grid_points_list)
    grid_spacings = np.zeros(n_grids)
    absolute_errors = np.zeros((n_grids, n_eigenvalues))
    relative_errors = np.zeros((n_grids, n_eigenvalues))
    
    for i, n_grid in enumerate(grid_points_list):
        numerical = numerical_ISW_eigenvalues(n_eigenvalues, n_grid, order=order, L=L)
        abs_err, rel_err = compute_eigenvalue_discrepancies(numerical, analytical)
        grid_spacings[i] = L / (n_grid + 1)
        absolute_errors[i, :] = abs_err
        relative_errors[i, :] = rel_err
    
    return {
        'grid_spacings': grid_spacings,
        'absolute_errors': absolute_errors,
        'relative_errors': relative_errors,
        'analytical': analytical,
        'grid_points': np.array(grid_points_list),
        'order': order
    }


def convergence_study_QSHO(n_eigenvalues, grid_points_list, order=4, omega=1.0, x_max=None):
    """
    Study convergence for QSHO across different grid resolutions.
    Returns dict with grid_spacings, errors (2D arrays), and analytical values.
    """
    analytical = analytical_QSHO_eigenvalues(n_eigenvalues, omega=omega)
    
    if x_max is None:
        E_max = (n_eigenvalues - 1 + 0.5) * omega
        x_classical = np.sqrt(2 * E_max / omega)
        x_max = x_classical + 8.0
    
    n_grids = len(grid_points_list)
    grid_spacings = np.zeros(n_grids)
    absolute_errors = np.zeros((n_grids, n_eigenvalues))
    relative_errors = np.zeros((n_grids, n_eigenvalues))
    
    for i, n_grid in enumerate(grid_points_list):
        numerical = numerical_QSHO_eigenvalues(n_eigenvalues, n_grid, 
                                              order=order, omega=omega, x_max=x_max)
        abs_err, rel_err = compute_eigenvalue_discrepancies(numerical, analytical)
        grid_spacings[i] = (2 * x_max) / (n_grid + 1)
        absolute_errors[i, :] = abs_err
        relative_errors[i, :] = rel_err
    
    return {
        'grid_spacings': grid_spacings,
        'absolute_errors': absolute_errors,
        'relative_errors': relative_errors,
        'analytical': analytical,
        'grid_points': np.array(grid_points_list),
        'order': order,
        'domain': [-x_max, x_max]
    }
