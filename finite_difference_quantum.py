import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

# =========================
# Grid utilities
# =========================
def spatial(x_left, x_right, N):
    """
    Generate interior uniform grid points for finite difference discretization.
    
    Parameters:
    -----------
    x_left : float
        Left boundary of the domain
    x_right : float
        Right boundary of the domain
    N : int
        Number of interior grid points
    
    Returns:
    --------
    grid : ndarray
        Array of N interior grid points uniformly spaced on [x_left, x_right]
    """
    return np.linspace(x_left, x_right, N + 2)[1:-1]


def proj(f, grid):
    """
    Evaluate scalar function f at all grid points and return a NumPy array.
    
    Parameters:
    -----------
    f : callable
        Scalar function to be evaluated
    grid : array_like
        Grid points where the function should be evaluated
    
    Returns:
    --------
    values : ndarray
        Array of function values at grid points, with dtype=complex
    """
    return np.array([f(point) for point in grid], dtype=complex)

# =========================
# Matrix utilities
# =========================

def _get_coefficients(order):
    """
    Get coefficient arrays for finite difference stencils.
    
    Parameters:
    -----------
    order : int
        Finite difference order: 2 or 4
    
    Returns:
    --------
    coeffs_d2 : ndarray
        Coefficients for second derivative term
    coeffs_d1 : ndarray  
        Coefficients for first derivative term
    coeffs_d0 : ndarray
        Coefficients for zeroth derivative term
    """
    if order == 2:
        coeffs_d2 = np.array([-2.0, 1.0, 1.0])
        coeffs_d1 = np.array([-1.0, 0.0, 1.0])
        coeffs_d0 = np.array([0.0, 1.0, 0.0])
    elif order == 4:
        coeffs_d2 = np.array([-1.0/12, 16.0/12, -30.0/12, 16.0/12, -1.0/12])
        coeffs_d1 = np.array([1.0/12, -8.0/12, 0.0, 8.0/12, -1.0/12])
        coeffs_d0 = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    
    return coeffs_d2, coeffs_d1, coeffs_d0

def _diagonal_unified(A, B, C, grid, k, order=2):
    """
    Unified function for matrix diagonals with offset k.
    
    Parameters:
    -----------
    A : callable
        Coefficient function for second derivative term (A * u'')
    B : callable or None
        Coefficient function for first derivative term (B * u')
    C : callable or None
        Coefficient function for zeroth derivative term (C * u)
    grid : array_like
        Spatial grid points where the operator is evaluated
    k : int
        Diagonal offset: -2, -1, 0, 1, or 2
    order : int, optional
        Finite difference order: 2 or 4 (default: 2)
    
    Returns:
    --------
    rows : ndarray
        Row indices for sparse matrix construction
    cols : ndarray
        Column indices for sparse matrix construction
    data : ndarray
        Matrix values for sparse matrix construction
    """
    h = grid[1] - grid[0]
    N = len(grid)
    
    a = max(0, -k)
    b = min(N, N - k)
    
    if b <= a:
        return np.array([]), np.array([]), np.array([])
    
    rows = np.arange(a, b)
    cols = rows + k
    x = grid[a:b]
    
    a_vals = proj(A, x)
    b_vals = proj(B, x) if B is not None else np.zeros_like(a_vals)
    c_vals = proj(C, x) if C is not None else np.zeros_like(a_vals)
    
    coeffs_d2, coeffs_d1, coeffs_d0 = _get_coefficients(order)
    
    idx = k + order//2
    data = (coeffs_d2[idx] * a_vals / (h**2) + 
            coeffs_d1[idx] * b_vals / h + 
            coeffs_d0[idx] * c_vals)
    
    return rows, cols, data


def _boundary_corrections(A, B, C, grid, order=4):
    """
    Generate boundary corrections for finite differences.
    
    Parameters:
    -----------
    A : callable
        Coefficient function for second derivative term (A * u'')
    B : callable or None
        Coefficient function for first derivative term (B * u')
    C : callable or None
        Coefficient function for zeroth derivative term (C * u)
    grid : array_like
        Spatial grid points where the operator is evaluated
    order : int, optional
        Finite difference order: 2 or 4 (default: 4)
    
    Returns:
    --------
    rows : ndarray
        Row indices for sparse matrix construction
    cols : ndarray
        Column indices for sparse matrix construction
    data : ndarray
        Correction values for sparse matrix construction
    """
    N = len(grid)
    
    rows = []
    cols = []
    data = []
    
    if order == 2:
        return np.array(rows), np.array(cols), np.array(data)
    
    if N >= 4 and order == 4:
        h = grid[1] - grid[0]
        coeffs_d2, coeffs_d1, coeffs_d0 = _get_coefficients(order)
        
        a = proj(A, grid)
        b_vals = proj(B, grid) if B is not None else np.zeros(N)
        c_vals = proj(C, grid) if C is not None else np.zeros(N)
        
        M_1_minus_1 = (coeffs_d2[0] * a[0] / (h**2) + 
                      coeffs_d1[0] * b_vals[0] / h + 
                      coeffs_d0[0] * c_vals[0])
        
        M_N_plus_1 = (coeffs_d2[4] * a[N-1] / (h**2) + 
                     coeffs_d1[4] * b_vals[N-1] / h + 
                     coeffs_d0[4] * c_vals[N-1])
        
        rows.extend([0, 0, 0])
        cols.extend([0, 1, 2])
        data.extend([
            -6.0 * M_1_minus_1,
            4.0 * M_1_minus_1,
            -1.0 * M_1_minus_1
        ])

        rows.extend([N-1, N-1, N-1])
        cols.extend([N-3, N-2, N-1])
        data.extend([
            -1.0 * M_N_plus_1,
            4.0 * M_N_plus_1,
            -6.0 * M_N_plus_1
        ])
    
    return np.array(rows), np.array(cols), np.array(data)


# =========================
# Matrix builders
# =========================
def sparseMatrixMaker(A, B, C, grid):
    """
    Assemble the sparse tridiagonal operator for A u'' + B u' + C u using 2nd-order finite differences.
    
    Parameters:
    -----------
    A : callable
        Coefficient function for second derivative term (A * u'')
    B : callable or None
        Coefficient function for first derivative term (B * u')
    C : callable or None
        Coefficient function for zeroth derivative term (C * u)
    grid : array_like
        Spatial grid points where the operator is evaluated
    
    Returns:
    --------
    matrix : scipy.sparse.coo_matrix
        Sparse matrix in COO format representing the finite difference operator
    """
    return sparse_Matrix_Maker(A, B, C, grid, order=2)


def sparse_Matrix_Maker(A, B, C, grid, order=2):
    """
    Assemble the sparse operator for A u'' + B u' + C u using finite differences with boundary corrections.
    
    Parameters:
    -----------
    A : callable
        Coefficient function for second derivative term (A * u'')
    B : callable or None
        Coefficient function for first derivative term (B * u')
    C : callable or None
        Coefficient function for zeroth derivative term (C * u)
    grid : array_like
        Spatial grid points where the operator is evaluated
    order : int, optional
        Finite difference order: 2 or 4 (default: 2)
    
    Returns:
    --------
    matrix : scipy.sparse.coo_matrix
        Sparse matrix in COO format representing the finite difference operator
    """
    rows = np.array([])
    cols = np.array([])
    data = np.array([])
    
    for k in range(-order//2, order//2 + 1):
        r, c, d = _diagonal_unified(A, B, C, grid, k, order=order)
        rows = np.concatenate([rows, r])
        cols = np.concatenate([cols, c])
        data = np.concatenate([data, d])
    
    rBC, cBC, dBC = _boundary_corrections(A, B, C, grid, order=order)
    rows = np.concatenate([rows, rBC])
    cols = np.concatenate([cols, cBC])
    data = np.concatenate([data, dBC])

    N = len(grid)
    return sparse.coo_matrix((data, (rows, cols)), shape=(N, N))


def sparse_Matrix_Maker4(A, B, C, grid):
    """
    Assemble the sparse pentadiagonal operator for A u'' + B u' + C u using 4th-order finite differences with boundary corrections.
    
    Parameters:
    -----------
    A : callable
        Coefficient function for second derivative term (A * u'')
    B : callable or None
        Coefficient function for first derivative term (B * u')
    C : callable or None
        Coefficient function for zeroth derivative term (C * u)
    grid : array_like
        Spatial grid points where the operator is evaluated
    
    Returns:
    --------
    matrix : scipy.sparse.coo_matrix
        Sparse matrix in COO format representing the finite difference operator
    """
    return sparse_Matrix_Maker(A, B, C, grid, order=4)


# =========================
# Hamiltonian functions
# =========================
def hamiltonian(V, x_left, x_right, N, order=2):
    """
    Create quantum mechanical hamiltonian operator H = -½∇² + V(x).
    
    Parameters:
    -----------
    V : callable
        Potential energy function V(x)
    x_left : float
        Left boundary of the domain
    x_right : float
        Right boundary of the domain
    N : int
        Number of interior grid points
    order : int, optional
        Finite difference order: 2 or 4 (default: 2)
    
    Returns:
    --------
    H : scipy.sparse.csr_matrix
        Sparse matrix representing the hamiltonian operator in CSR format
    """
    N = int(N)
    grid = spatial(x_left, x_right, N)
    
    def A(x): return -0.5
    def B(x): return 0.0
    def C(x): return V(x)
    
    return sparse_Matrix_Maker(A, B, C, grid, order=order).tocsr()


def hamiltonian_4th(V, x_left, x_right, N):
    """
    Create 4th-order quantum mechanical hamiltonian operator H = -½∇² + V(x) with boundary corrections.
    
    Parameters:
    -----------
    V : callable
        Potential energy function V(x)
    x_left : float
        Left boundary of the domain
    x_right : float
        Right boundary of the domain
    N : int
        Number of interior grid points
    
    Returns:
    --------
    H : scipy.sparse.csr_matrix
        Sparse matrix representing the hamiltonian operator in CSR format
    """
    return hamiltonian(V, x_left, x_right, N, order=4)

