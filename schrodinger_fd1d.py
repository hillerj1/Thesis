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
    return np.array([f(grid[i]) for i in range(len(grid))], dtype=float)


# =========================
# Assembly (triplets)
# =========================
def diag(A, C, grid):
    """
    Triplets (rows, cols, data) for the main diagonal of
    A(x) u'' + C(x) u using central differences on a uniform grid.
    """
    x = np.asarray(grid, dtype=float)
    del_x = x[1] - x[0]

    N = x.size
    rows = np.arange(N)
    cols = rows

    a = np.array([A(xi) for xi in x], dtype=float)
    c = np.array([C(xi) for xi in x], dtype=float)
    data = c - (2.0 * a) / (del_x**2)

    return rows, cols, data


def upper(A, B, grid):
    x = np.asarray(grid, dtype=float)
    del_x = x[1] - x[0]
    N = len(x)

    rows = np.array([i for i in range(0, N - 1)])
    cols = np.array([i + 1 for i in range(0, N - 1)])

    a = np.array([A(x[i]) for i in range(0, N - 1)], dtype=float)
    b = np.array([B(x[i]) for i in range(0, N - 1)], dtype=float)

    data = (a / (del_x**2)) + (b / (2.0 * del_x))
    return rows, cols, data


def lower(A, B, grid):
    x = np.asarray(grid, dtype=float)
    del_x = x[1] - x[0]
    N = len(x)

    rows = np.array([i for i in range(1, N)])
    cols = np.array([i - 1 for i in range(1, N)])

    a = np.array([A(x[i]) for i in range(1, N)], dtype=float)
    b = np.array([B(x[i]) for i in range(1, N)], dtype=float)

    data = (a / (del_x**2)) - (b / (2.0 * del_x))
    return rows, cols, data


# =========================
# Matrix builder (2nd Order + 4th Order)
# =========================
def sparseMatrixMaker(A, B, C, grid):
    """
    Assemble the sparse tridiagonal operator for A u'' + B u' + C u
    by concatenating triplets from diag/upper/lower. Returns a COO matrix.
    """
    rD, cD, dD = diag(A, C, grid)
    rU, cU, dU = upper(A, B, grid)
    rL, cL, dL = lower(A, B, grid)

    rows = np.concatenate([rD, rU, rL])
    cols = np.concatenate([cD, cU, cL])
    data = np.concatenate([dD, dU, dL])

    N = len(grid)
    return sparse.coo_matrix((data, (rows, cols)), shape=(N, N))

# grid array for 4th order
def triplets_4th_Au_xx_plus_Cu(A, C, grid):
    x = np.asarray(grid, dtype=float)
    h = x[1] - x[0]
    N = x.size

    a = np.array([A(xi) for xi in x], dtype=float)
    c = np.array([C(xi) for xi in x], dtype=float)

    rows, cols, data = [], [], []

def sparse_Matrix_Maker4(A, C, grid):
    r, c, d = triplets_4th_Au_xx_plus_Cu(A, C, grid)
    N = len(grid)
    return sparse.coo_matrix((d, (r, c)), shape=(N, N))

# =========================
# Hamiltonian wrapper (2nd Order + 4th Order)
# =========================
def hamiltonian(V, x_left, x_right, N):
    N = int(N)
    grid = spatial(x_left, x_right, N)   # <-- uses your existing utility

    def A(x): return -0.5      # -(1/2) * u''
    def B(x): return 0.0       # no first-derivative term here
    def C(x): return V(x)      # + V(x) * u

    return sparseMatrixMaker(A, B, C, grid).tocsr()

def hamiltonian_4th(V, x_left, x_right, N):
    N = int(N)
    grid = spatial(x_left, x_right, N)

    def A(x): return -0.5
    def C(x): return V(x)

    return sparse_Matrix_Maker4(A, C, grid).tocsr()

