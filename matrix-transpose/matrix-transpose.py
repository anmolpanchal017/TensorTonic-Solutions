import numpy as np

def matrix_transpose(A):
    row = len(A)
    cols = len(A[0])

    T = []

    for j in range(cols):
        new_row = []
        for i in range(row):
            new_row.append(A[i][j])
        T.append(new_row)

    return np.array(T)