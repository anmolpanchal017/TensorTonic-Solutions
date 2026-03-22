import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    try:
        # 1. Convert to numpy array and ensure it's 2D
        matrix = np.array(matrix, dtype=float)
        if matrix.ndim != 2:
            return None
        
        # 2. Compute Norm based on type
        if norm_type == 'l2':
            # ||x||2 = sqrt(sum(xi^2))
            norm = np.sqrt(np.sum(matrix**2, axis=axis, keepdims=True))
        elif norm_type == 'l1':
            # ||x||1 = sum(|xi|)
            norm = np.sum(np.abs(matrix), axis=axis, keepdims=True)
        elif norm_type == 'max':
            # ||x||inf = max(|xi|)
            norm = np.max(np.abs(matrix), axis=axis, keepdims=True)
        else:
            return None # Invalid norm type

        # 3. Handle Zero Vectors to avoid division by zero
        # If the norm is 0, we leave the values as 0 (divide by 1)
        norm = np.where(norm == 0, 1, norm)

        # 4. Perform the division (Broadcasting handles the shape)
        return matrix / norm

    except Exception:
        # Return None for any unexpected invalid inputs (e.g., non-numeric)
        return None