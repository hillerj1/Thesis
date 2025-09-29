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
    
    if k == 0:
        rows = np.arange(N)
        cols = rows
        x = grid
    elif k > 0:
        rows = np.arange(N - k)
        cols = rows + k
        x = grid[:-k] if k < N else grid[:0]
    else:
        rows = np.arange(-k, N)
        cols = rows + k
        x = grid[-k:] if -k < N else grid[:0]
    
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])
    
    a = proj(A, x)
    b = proj(B, x) if B is not None else np.zeros_like(a)
    c = proj(C, x) if C is not None else np.zeros_like(a)
    
    if order == 2:
        if k == 0:
            data = c - (2.0 * a) / (h**2)
        elif k == 1:
            data = (a / (h**2)) + (b / (2.0 * h))
        elif k == -1:
            data = (a / (h**2)) - (b / (2.0 * h))
        else:
            data = np.zeros_like(a)
    
    elif order == 4:
        if k == 0:
            data = c - (30.0 * a) / (12 * h**2)
        elif k == 1:
            data = (16.0 * a) / (12 * h**2) + (8.0 * b) / (12 * h)
        elif k == -1:
            data = (16.0 * a) / (12 * h**2) - (8.0 * b) / (12 * h)
        elif k == 2:
            data = (-1.0 * a) / (12 * h**2) + (1.0 * b) / (12 * h)
        elif k == -2:
            data = (-1.0 * a) / (12 * h**2) - (1.0 * b) / (12 * h)
    
    return rows, cols, data


def _boundary_corrections_4th(A, grid):
    """
    Generate boundary corrections for 4th order finite differences.
    
    Parameters:
    -----------
    A : callable
        Coefficient function for second derivative term (A * u'')
    grid : array_like
        Spatial grid points where the operator is evaluated
    
    Returns:
    --------
    rows : ndarray
        Row indices for sparse matrix construction
    cols : ndarray
        Column indices for sparse matrix construction
    data : ndarray
        Correction values for sparse matrix construction
    """
    h = grid[1] - grid[0]
    N = len(grid)
    
    rows = []
    cols = []
    data = []
    
    a = proj(A, grid)
    
    if N >= 4:
        correction_factor = 1.0 / (12 * h**2)
        
        # Row 0 corrections: add to positions (0,0), (0,1), (0,2), (0,3)
        rows.extend([0, 0, 0, 0])
        cols.extend([0, 1, 2, 3])
        data.extend([4.0 * a[0] * correction_factor, -6.0 * a[0] * correction_factor, 4.0 * a[0] * correction_factor, -1.0 * a[0] * correction_factor])
        
        # Row 1 corrections: add to positions (1,0), (1,1), (1,2), (1,3)
        rows.extend([1, 1, 1, 1])
        cols.extend([0, 1, 2, 3])
        data.extend([4.0 * a[1] * correction_factor, -6.0 * a[1] * correction_factor, 4.0 * a[1] * correction_factor, -1.0 * a[1] * correction_factor])
        
        # Row N-2 corrections: add to positions (N-2,N-4), (N-2,N-3), (N-2,N-2), (N-2,N-1)
        rows.extend([N-2, N-2, N-2, N-2])
        cols.extend([N-4, N-3, N-2, N-1])
        data.extend([-1.0 * a[N-2] * correction_factor, 4.0 * a[N-2] * correction_factor, -6.0 * a[N-2] * correction_factor, 4.0 * a[N-2] * correction_factor])
        
        # Row N-1 corrections: add to positions (N-1,N-4), (N-1,N-3), (N-1,N-2), (N-1,N-1)
        rows.extend([N-1, N-1, N-1, N-1])
        cols.extend([N-4, N-3, N-2, N-1])
        data.extend([-1.0 * a[N-1] * correction_factor, 4.0 * a[N-1] * correction_factor, -6.0 * a[N-1] * correction_factor, 4.0 * a[N-1] * correction_factor])
    
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
    rD, cD, dD = _diagonal_unified(A, B, C, grid, 0, order=2)
    rU, cU, dU = _diagonal_unified(A, B, C, grid, 1, order=2)
    rL, cL, dL = _diagonal_unified(A, B, C, grid, -1, order=2)

    rows = np.concatenate([rD, rU, rL])
    cols = np.concatenate([cD, cU, cL])
    data = np.concatenate([dD, dU, dL])

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
    rD, cD, dD = _diagonal_unified(A, B, C, grid, 0, order=4)
    rU, cU, dU = _diagonal_unified(A, B, C, grid, 1, order=4)
    rL, cL, dL = _diagonal_unified(A, B, C, grid, -1, order=4)
    rU2, cU2, dU2 = _diagonal_unified(A, B, C, grid, 2, order=4)
    rL2, cL2, dL2 = _diagonal_unified(A, B, C, grid, -2, order=4)
    rBC, cBC, dBC = _boundary_corrections_4th(A, grid)

    rows = np.concatenate([rD, rU, rL, rU2, rL2, rBC])
    cols = np.concatenate([cD, cU, cL, cU2, cL2, cBC])
    data = np.concatenate([dD, dU, dL, dU2, dL2, dBC])

    N = len(grid)
    return sparse.coo_matrix((data, (rows, cols)), shape=(N, N))


# =========================
# Hamiltonian functions
# =========================
def hamiltonian(V, x_left, x_right, N):
    """
    Create 2nd-order quantum mechanical hamiltonian operator H = -½∇² + V(x).
    
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
    N = int(N)
    grid = spatial(x_left, x_right, N)
    
    def A(x): return -0.5  # kinetic energy coefficient
    def B(x): return 0.0
    def C(x): return V(x)  # potential energy
    
    return sparseMatrixMaker(A, B, C, grid).tocsr()


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
    N = int(N)
    grid = spatial(x_left, x_right, N)
    
    def A(x): return -0.5
    def B(x): return 0.0
    def C(x): return V(x)
    
    return sparse_Matrix_Maker4(A, B, C, grid).tocsr()

