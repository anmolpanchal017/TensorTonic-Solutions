import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    
    # dot product
    dot_product = np.dot(a, b)
    
    # norms
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # handle zero vector case
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)