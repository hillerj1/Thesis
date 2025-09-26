import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

# =========================
# Grid utilities
# =========================
def spatial(x_left, x_right, N):
    """
    Return the interior uniform grid of length N on [x_left, x_right]
    (endpoints dropped).
    """
    return np.linspace(x_left, x_right, N + 2)[1:-1]


def proj(f, grid):
    """
    Evaluate scalar function f at all points in grid and return a NumPy array.
    """
    return np.array([f(point) for point in grid], dtype=complex)


# =========================
# Private assembly functions (2nd order)
# =========================
def _diag(A, C, grid):
    x = grid
    del_x = x[1] - x[0]
    N = len(x)
    
    rows = np.arange(N)
    cols = rows
    
    a = proj(A, x)
    c = proj(C, x)
    data = c - (2.0 * a) / (del_x**2)
    
    return rows, cols, data


def _upper(A, B, grid):
    x = grid[:-1]  # all except last point
    del_x = grid[1] - grid[0]
    N = len(grid)
    
    rows = np.arange(N-1)
    cols = rows + 1
    
    a = proj(A, x)
    b = proj(B, x)
    data = (a / (del_x**2)) + (b / (2.0 * del_x))
    
    return rows, cols, data


def _lower(A, B, grid):
    x = grid[1:]  # all except first point
    del_x = grid[1] - grid[0]
    N = len(grid)
    
    rows = np.arange(1, N)
    cols = rows - 1
    
    a = proj(A, x)
    b = proj(B, x)
    data = (a / (del_x**2)) - (b / (2.0 * del_x))
    
    return rows, cols, data


# =========================
# Private assembly functions (4th order)
# =========================
def _diag_4th(A, B, C, grid):
    x = grid  # all points
    h = grid[1] - grid[0]
    N = len(grid)
    
    rows = np.arange(N)
    cols = rows
    
    a = proj(A, x)
    b = proj(B, x)
    c = proj(C, x)
    data = c - (30.0 * a) / (12 * h**2)
    
    return rows, cols, data


def _upper_4th(A, B, grid):
    x = grid[:-1]  # all except last point
    h = grid[1] - grid[0]
    N = len(grid)
    
    rows = np.arange(N-1)
    cols = rows + 1
    
    a = proj(A, x)
    b = proj(B, x)
    data = (16.0 * a) / (12 * h**2) + (8.0 * b) / (12 * h)
    
    return rows, cols, data


def _lower_4th(A, B, grid):
    x = grid[1:]  # all except first point
    h = grid[1] - grid[0]
    N = len(grid)
    
    rows = np.arange(1, N)
    cols = rows - 1
    
    a = proj(A, x)
    b = proj(B, x)
    data = (16.0 * a) / (12 * h**2) - (8.0 * b) / (12 * h)
    
    return rows, cols, data


def _upper2_4th(A, B, grid):
    x = grid[:-2]  # all except last two points
    h = grid[1] - grid[0]
    N = len(grid)
    
    rows = np.arange(N-2)
    cols = rows + 2
    
    a = proj(A, x)
    b = proj(B, x)
    data = (-1.0 * a) / (12 * h**2) + (1.0 * b) / (12 * h)
    
    return rows, cols, data


def _lower2_4th(A, B, grid):
    x = grid[2:]  # all except first two points
    h = grid[1] - grid[0]
    N = len(grid)
    
    rows = np.arange(2, N)
    cols = rows - 2
    
    a = proj(A, x)
    b = proj(B, x)
    data = (-1.0 * a) / (12 * h**2) - (1.0 * b) / (12 * h)
    
    return rows, cols, data


def _triplets_4th_Au_xx_plus_Bu_x_plus_Cu(A, B, C, grid):
    rD, cD, dD = _diag_4th(A, B, C, grid)
    rU, cU, dU = _upper_4th(A, B, grid)
    rL, cL, dL = _lower_4th(A, B, grid)
    rU2, cU2, dU2 = _upper2_4th(A, B, grid)
    rL2, cL2, dL2 = _lower2_4th(A, B, grid)

    rows = np.concatenate([rD, rU, rL, rU2, rL2])
    cols = np.concatenate([cD, cU, cL, cU2, cL2])
    data = np.concatenate([dD, dU, dL, dU2, dL2])

    return rows, cols, data


# =========================
# Public matrix builders
# =========================
def sparseMatrixMaker(A, B, C, grid):
    """
    Assemble the sparse tridiagonal operator for A u'' + B u' + C u
    using 2nd-order finite differences. Returns a COO matrix.
    """
    rD, cD, dD = _diag(A, C, grid)
    rU, cU, dU = _upper(A, B, grid)
    rL, cL, dL = _lower(A, B, grid)

    rows = np.concatenate([rD, rU, rL])
    cols = np.concatenate([cD, cU, cL])
    data = np.concatenate([dD, dU, dL])

    N = len(grid)
    return sparse.coo_matrix((data, (rows, cols)), shape=(N, N))


def sparse_Matrix_Maker4(A, B, C, grid):
    """
    Assemble the sparse operator for A u'' + B u' + C u using 4th-order finite differences.
    Returns a COO matrix.
    """
    r, c, d = _triplets_4th_Au_xx_plus_Bu_x_plus_Cu(A, B, C, grid)
    N = len(grid)
    return sparse.coo_matrix((d, (r, c)), shape=(N, N))


# =========================
# Public hamiltonian functions
# =========================
def hamiltonian(V, x_left, x_right, N):
    """
    Create 2nd-order hamiltonian for domain [x_left, x_right].
    
    Parameters:
    - V: potential function
    - x_left, x_right: domain boundaries
    - N: number of interior grid points
    
    Returns: CSR sparse matrix
    """
    N = int(N)
    grid = spatial(x_left, x_right, N)
    
    def A(x): return -0.5  # kinetic energy coefficient
    def B(x): return 0.0
    def C(x): return V(x)  # potential energy
    
    return sparseMatrixMaker(A, B, C, grid).tocsr()


def hamiltonian_4th(V, x_left, x_right, N):
    """
    Create 4th-order hamiltonian for domain [x_left, x_right].
    
    Parameters:
    - V: potential function
    - x_left, x_right: domain boundaries
    - N: number of interior grid points
    
    Returns: CSR sparse matrix
    """
    N = int(N)
    grid = spatial(x_left, x_right, N)
    
    def A(x): return -0.5
    def B(x): return 0.0
    def C(x): return V(x)
    
    return sparse_Matrix_Maker4(A, B, C, grid).tocsr()

