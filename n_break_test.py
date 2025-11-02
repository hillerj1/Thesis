import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy.linalg import eig
import datetime

def create_2nd_order_matrix(N, A, B, C, x_left, x_right):
    Delta = (x_right - x_left) / (N + 1)
    grid = np.linspace(x_left + Delta, x_right - Delta, N)
    A_vals = A(grid)
    B_vals = B(grid)
    C_vals = C(grid)
    diagonal = C_vals - 2 * A_vals / Delta**2
    upper = A_vals[:-1] / Delta**2 + B_vals[:-1] / (2 * Delta)
    lower = A_vals[1:] / Delta**2 - B_vals[1:] / (2 * Delta)
    return sp.diags([lower, diagonal, upper], [-1, 0, 1], format='csr')


def create_4th_order_matrix(N, A, B, C, x_left, x_right):
    if N < 4:
        raise ValueError(f"N must be >= 4 for 4th order, got N={N}")
    
    Delta = (x_right - x_left) / (N + 1)
    grid = np.linspace(x_left + Delta, x_right - Delta, N)
    A_vals = A(grid)
    B_vals = B(grid)
    C_vals = C(grid)
    main_diag = C_vals - 30 * A_vals / (12 * Delta**2)
    upper1 = 16 * A_vals[:-1] / (12 * Delta**2) + 8 * B_vals[:-1] / (12 * Delta)
    lower1 = 16 * A_vals[1:] / (12 * Delta**2) - 8 * B_vals[1:] / (12 * Delta)
    upper2 = -A_vals[:-2] / (12 * Delta**2) - B_vals[:-2] / (12 * Delta)
    lower2 = -A_vals[2:] / (12 * Delta**2) + B_vals[2:] / (12 * Delta)
    
    data = []
    offsets = []
    for offset, vals in [(-2, lower2), (-1, lower1), (0, main_diag), (1, upper1), (2, upper2)]:
        if len(vals) > 0:
            data.append(vals)
            offsets.append(offset)
    
    return sp.diags(data, offsets, format='csr', shape=(N, N))


def test_method(order, x_left, x_right):
    def A(x):
        return -0.5 * np.ones_like(x)
    def B(x):
        return np.zeros_like(x)
    def C(x):
        return np.zeros_like(x)
    
    log_file = f"n_break_test_order{order}.txt"
    n = 10
    max_n_reached = 10
    
    with open(log_file, 'w') as f:
        f.write(f"Testing {order}nd order finite difference method\n")
        f.write(f"Started: {datetime.datetime.now()}\n")
        f.write(f"=" * 60 + "\n\n")
    
    while True:
        try:
            if order == 4 and n < 4:
                n *= 2
                continue
            
            if order == 2:
                M = create_2nd_order_matrix(n, A, B, C, x_left, x_right)
            else:
                M = create_4th_order_matrix(n, A, B, C, x_left, x_right)
            
            print(f"Order {order}: N={n}")
            with open(log_file, 'a') as f:
                f.write(f"N={n}: Matrix created ({M.nnz} nonzeros)\n")
            
            if n <= 4:
                eigenvalues, _ = eig(M.toarray())
                eigenvalues = np.sort(eigenvalues.real)[:1]
            else:
                try:
                    eigenvalues, _ = eigsh(M, k=1, which='SA', tol=1e-8, maxiter=50000)
                except ArpackNoConvergence:
                    try:
                        eigenvalues, _ = eigsh(M, k=1, which='SA', tol=1e-6, maxiter=100000)
                    except ArpackNoConvergence:
                        eigenvalues, _ = eigsh(M, k=1, sigma=0, which='LM', maxiter=100000)
            
            max_n_reached = n
            with open(log_file, 'a') as f:
                f.write(f"N={n}: Success, E0={eigenvalues[0]:.6f}\n\n")
            
            n *= 2
            
        except MemoryError as e:
            print(f"Order {order}: Memory error at N={n}")
            with open(log_file, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"STOPPED at N={n} due to MemoryError\n")
                f.write(f"Last successful N: {max_n_reached}\n")
                f.write(f"Error: {str(e)}\n")
            break
            
        except KeyboardInterrupt:
            print(f"\nOrder {order}: Interrupted at N={n}")
            with open(log_file, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"STOPPED at N={n} due to KeyboardInterrupt\n")
                f.write(f"Last successful N: {max_n_reached}\n")
            break
            
        except Exception as e:
            error_type = type(e).__name__
            print(f"Order {order}: Error at N={n}: {error_type}")
            
            if isinstance(e, (ValueError, np.linalg.LinAlgError, ArpackNoConvergence)):
                with open(log_file, 'a') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"STOPPED at N={n} due to {error_type}\n")
                    f.write(f"Last successful N: {max_n_reached}\n")
                    f.write(f"Error: {str(e)}\n")
                break
            else:
                with open(log_file, 'a') as f:
                    f.write(f"N={n}: Error - {error_type}: {str(e)}\n")
                    f.write(f"Continuing...\n\n")
                n *= 2
                continue
    
    with open(log_file, 'a') as f:
        f.write(f"\nFinished: {datetime.datetime.now()}\n")
    
    print(f"Order {order}: Max N = {max_n_reached}, saved to {log_file}")


if __name__ == "__main__":
    x_left, x_right = 0.0, 1.0
    
    print("N_break test: doubling N until failure")
    print(f"Results saved to n_break_test_order*.txt\n")
    
    test_method(2, x_left, x_right)
    test_method(4, x_left, x_right)
    
    print("\nDone")
