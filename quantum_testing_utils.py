import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite, factorial
from scipy.integrate import trapezoid
import time

# =========================
# Analytical Solutions
# =========================

def analytical_QSHO_eigenvalues(n_max, omega=1.0):
    """Analytical eigenvalues for Quantum Simple Harmonic Oscillator.
    
    Parameters:
    -----------
    n_max : int
        Maximum quantum number to compute
    omega : float, optional
        Angular frequency (default: 1.0)
        
    Returns:
    --------
    eigenvalues : ndarray
        Array of eigenvalues E_n = (n + 0.5) * omega
    """
    n = np.arange(n_max)
    return (n + 0.5) * omega

def analytical_QSHO_eigenfunctions(x, n, omega=1.0):
    """Analytical eigenfunctions for QSHO.
    
    Parameters:
    -----------
    x : array_like
        Spatial coordinates
    n : int
        Quantum number
    omega : float, optional
        Angular frequency (default: 1.0)
        
    Returns:
    --------
    psi : ndarray
        Normalized eigenfunction ψ_n(x)
    """
    x = np.asarray(x)
    alpha = np.sqrt(omega)
    xi = alpha * x
    
    H_n = hermite(n)
    N_n = (alpha / (np.pi * 2**n * factorial(n)))**0.25
    psi = N_n * H_n(xi) * np.exp(-xi**2 / 2)
    
    return psi

def analytical_ISW_eigenvalues(n_max, L=1.0):
    """Analytical eigenvalues for Infinite Square Well.
    
    Parameters:
    -----------
    n_max : int
        Maximum quantum number to compute
    L : float, optional
        Well width (default: 1.0)
        
    Returns:
    --------
    eigenvalues : ndarray
        Array of eigenvalues E_n = n²π²/(2L²)
    """
    n = np.arange(1, n_max + 1)
    return n**2 * np.pi**2 / (2 * L**2)

def analytical_ISW_eigenfunctions(x, n, L=1.0):
    """Analytical eigenfunctions for ISW.
    
    Parameters:
    -----------
    x : array_like
        Spatial coordinates
    n : int
        Quantum number (starts from 1)
    L : float, optional
        Well width (default: 1.0)
        
    Returns:
    --------
    psi : ndarray
        Normalized eigenfunction ψ_n(x)
    """
    x = np.asarray(x)
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

# =========================
# Error Analysis Functions
# =========================

def convergence_rate(dx_values, error_values):
    """Calculate convergence rate from log-log plot slope.
    
    Parameters:
    -----------
    dx_values : array_like
        Grid spacing values
    error_values : array_like
        Corresponding error values
        
    Returns:
    --------
    rate : float
        Convergence rate (slope of log-log plot)
    """
    log_dx = np.log(dx_values)
    log_error = np.log(error_values)
    
    # Linear fit to log-log data
    coeffs = np.polyfit(log_dx, log_error, 1)
    rate = coeffs[0]
    
    return rate

# =========================
# Plotting Utilities
# =========================

def plot_convergence(dx_values, error_values, title="Convergence Analysis", 
                    xlabel="Grid spacing dx", ylabel="Error", 
                    reference_slopes=None, labels=None):
    """Create log-log convergence plot.
    
    Parameters:
    -----------
    dx_values : array_like
        Grid spacing values
    error_values : array_like
        Error values (can be 2D for multiple error types)
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    reference_slopes : list, optional
        Reference slopes to plot as dashed lines
    labels : list, optional
        Labels for different error types
    """
    plt.figure(figsize=(10, 7))
    
    dx_values = np.asarray(dx_values)
    error_values = np.asarray(error_values)
    
    if error_values.ndim == 1:
        error_values = error_values.reshape(-1, 1)
    
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*']
    
    for i in range(error_values.shape[1]):
        label = labels[i] if labels else f"Error {i+1}"
        plt.loglog(dx_values, error_values[:, i], 
                  color=colors[i % len(colors)], 
                  marker=markers[i % len(markers)],
                  linewidth=2, markersize=6, label=label)
    
    # Add reference slopes
    if reference_slopes:
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        
        for i, slope in enumerate(reference_slopes):
            # Create reference line
            x_ref = np.array([x_min, x_max])
            y_ref = y_min * (x_ref / x_min)**slope
            plt.loglog(x_ref, y_ref, '--', alpha=0.7, 
                      color=colors[i % len(colors)],
                      label=f'Slope = {slope}')
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# =========================
# Performance Analysis
# =========================

def timing_analysis(hamiltonian_func, N_values, *args, **kwargs):
    """Analyze timing performance of hamiltonian construction.
    
    Parameters:
    -----------
    hamiltonian_func : callable
        Function to create hamiltonian
    N_values : array_like
        Grid sizes to test
    *args, **kwargs
        Arguments to pass to hamiltonian function
        
    Returns:
    --------
    times : ndarray
        Timing results for each N
    """
    times = []
    
    for N in N_values:
        start_time = time.time()
        H = hamiltonian_func(*args, N=N, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return np.array(times)

def plot_timing_comparison(N_values, times_2nd, times_4th, title="Timing Comparison"):
    """Plot timing comparison between 2nd and 4th order methods.
    
    Parameters:
    -----------
    N_values : array_like
        Grid sizes
    times_2nd : array_like
        Timing for 2nd order method
    times_4th : array_like
        Timing for 4th order method
    title : str, optional
        Plot title
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(N_values, times_2nd, 'b-o', label='2nd Order', linewidth=2, markersize=4)
    plt.plot(N_values, times_4th, 'r-s', label='4th Order', linewidth=2, markersize=4)
    
    plt.xlabel('Grid Size N', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()