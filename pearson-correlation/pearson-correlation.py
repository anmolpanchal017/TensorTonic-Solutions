import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
   
    X = np.array(X, dtype=float)
    
    # Validate input
    if X.ndim != 2 or X.shape[0] < 2:
        return None
    
    N = X.shape[0]
    
    # Step 1: Center the data (subtract mean of each column)
    X_centered = X - np.mean(X, axis=0)
    
    # Step 2: Compute covariance matrix
    cov = (X_centered.T @ X_centered) / (N - 1)
    
    # Step 3: Compute standard deviations
    std = np.sqrt(np.diag(cov))
    
    # Step 4: Outer product of std deviations
    denom = np.outer(std, std)
    
    # Step 5: Compute correlation matrix
    corr = cov / denom
    
    # Step 6: Handle zero variance (avoid division by zero)
    corr[denom == 0] = np.nan
    
    return corr