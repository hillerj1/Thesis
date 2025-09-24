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
# 4th-order finite difference weights
# =========================
def _fd_weights_second_derivative(offsets):
    offs = np.asarray(offsets, dtype=complex)
    m = len(offs)
    A = np.zeros((m, m), dtype=complex)
    b = np.zeros(m, dtype=complex)
    b[2] = 2.0
    for k in range(m):
        A[k, :] = offs**k
    return np.linalg.solve(A, b)

# 4th-order stencil offsets
_OFF_CENT = np.array([-2,-1, 0, 1, 2], dtype=int)
_OFF_L0   = np.array([ 0, 1, 2, 3, 4], dtype=int)   
_OFF_L1   = np.array([-1, 0, 1, 2, 3], dtype=int)
_OFF_R1   = np.array([-3,-2,-1, 0, 1], dtype=int)   
_OFF_R0   = np.array([-4,-3,-2,-1, 0], dtype=int)   

_W_CENT = _fd_weights_second_derivative(_OFF_CENT)
_W_L0   = _fd_weights_second_derivative(_OFF_L0)
_W_L1   = _fd_weights_second_derivative(_OFF_L1)
_W_R1   = _fd_weights_second_derivative(_OFF_R1)
_W_R0   = _fd_weights_second_derivative(_OFF_R0)


def _triplets_4th_Au_xx_plus_Cu(A, C, grid):
    x = np.asarray(grid, dtype=complex)
    h = x[1] - x[0]
    N = x.size

    a = np.array([A(xi) for xi in x], dtype=complex)
    c = np.array([C(xi) for xi in x], dtype=complex)

    rows, cols, data = [], [], []

    def add_row(i, offsets, W):
        for off, w in zip(offsets, W/(h*h)):
            j = i + off
            if 0 <= j < N:
                rows.append(i); cols.append(j); data.append(a[i]*w)
        rows.append(i); cols.append(i); data.append(c[i])
        
    # handle boundaries and interior differently
    add_row(0, _OFF_L0, _W_L0)
    add_row(1, _OFF_L1, _W_L1)
    
    for i in range(2, N-2):
        add_row(i, _OFF_CENT, _W_CENT)
    
    add_row(N-2, _OFF_R1, _W_R1)
    add_row(N-1, _OFF_R0, _W_R0)

    return np.array(rows), np.array(cols), np.array(data)


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


def sparse_Matrix_Maker4(A, C, grid):
    """
    Assemble the sparse operator for A u'' + C u using 4th-order finite differences.
    Returns a COO matrix.
    """
    r, c, d = _triplets_4th_Au_xx_plus_Cu(A, C, grid)
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
    def C(x): return V(x)
    
    return sparse_Matrix_Maker4(A, C, grid).tocsr()

