import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    try:
        # Step 1: Convert input to numpy array
        A = np.asarray(matrix, dtype=float)

        # Step 2: Check if matrix is 2D
        if A.ndim != 2:
            return None

        # Step 3: Check if matrix is square
        rows, cols = A.shape
        if rows != cols:
            return None

        # Step 4: Handle empty matrix
        if rows == 0:
            return None

        # Step 5: Calculate eigenvalues
        eigen_vals = np.linalg.eigvals(A)

        # Step 6: Sort eigenvalues (real first, then imaginary)
        sorted_eigen_vals = eigen_vals[np.lexsort((eigen_vals.imag, eigen_vals.real))]

        return sorted_eigen_vals

    except:
        return None