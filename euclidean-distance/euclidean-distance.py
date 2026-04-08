def euclidean_distance(x, y):
    x = np.array(x)
    y = np.array(y)
    
    if x.shape != y.shape:
        raise ValueError("Vectors must have same dimensions")
    
    return float(np.sqrt(np.sum((x - y) ** 2)))