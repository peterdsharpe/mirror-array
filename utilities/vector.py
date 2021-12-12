import numpy as np

def normalize(v):
    """Normalizes vector v."""
    return v / np.linalg.norm(v)

def project(v, n):
    """Projects vector v onto the plane normal to n."""
    n = normalize(n)
    return v - np.dot(v, n) * n