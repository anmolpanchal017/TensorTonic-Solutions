import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    # Write code here
    # 1 step
    x = np.array(x)
    y = np.array(y)

    #2 step
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("input must be 1D")

    # 3 step
    if x.shape[0] != y.shape[0]:
        raise ValueError("vector must have same size")

    # final step
    result = np.dot(x, y)

    return float(result)
    pass