import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Convert to numpy array
    X = np.array(X, dtype=float)

    # Validate input
    if X.ndim != 2 or X.shape[0] < 2:
        return None

    N = X.shape[0]

    # Step 1: Compute mean
    mu = np.mean(X, axis=0)

    # Step 2: Center the data
    X_centered = X - mu

    # Step 3: Compute covariance matrix
    cov_matrix = (X_centered.T @ X_centered) / (N - 1)

    return cov_matrix