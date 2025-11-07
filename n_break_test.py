import numpy as np
import sys
from finite_difference_quantum import sparseMatrixMaker, sparse_Matrix_Maker4, spatial

def A(x):
    return np.random.rand()
def B(x):
    return np.random.rand()
def C(x):
    return np.random.rand()

x_left, x_right = 0.0, 1.0

def test_order(order):
    print(f"\n{'='*50}")
    print(f"Starting {order}nd order test")
    print(f"{'='*50}")
    n = 10
    last_n_written = None
    max_iterations = 100  
    
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        try:
            print(f"  Attempting n={n}...", end=" ", flush=True)
            grid = spatial(x_left, x_right, n)
            
            if order == 2:
                M = sparseMatrixMaker(A, B, C, grid)
            else:
                if n < 4:
                    print(f"skipped (need n >= 4)")
                    n *= 2
                    continue
                M = sparse_Matrix_Maker4(A, B, C, grid)
            
            with open(f"n_break_order{order}.txt", "a") as f:
                f.write(f"{n}\n")
            
            print(f"Success!")
            last_n_written = n
            n *= 2
        except MemoryError as e:
            print(f"\n{'='*50}")
            print(f"{order}nd order FINISHED (Memory Error)!")
            print(f"Last successful n: {last_n_written}")
            print(f"Failed at n: {n}")
            print(f"Error: {str(e)}")
            print(f"Results saved to: n_break_order{order}.txt")
            print(f"{'='*50}\n")
            return
    
    print(f"\n{'='*50}")
    print(f"{order}nd order FINISHED (reached max iterations)!")
    print(f"Last successful n: {last_n_written}")
    print(f"Results saved to: n_break_order{order}.txt")
    print(f"{'='*50}\n")

# Run tests
for order in [4, 2]:
    test_order(order)

print("\n" + "="*50)
print("ALL TESTS COMPLETE!")
print("="*50)
