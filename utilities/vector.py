import numpy as np
import numba


def normalize(v):
    """Normalizes vector v."""
    return v / np.expand_dims(np.linalg.norm(v, axis=-1), axis=-1)


def project(v, n):
    """Projects vector v onto the plane normal to n."""
    n = normalize(n)
    return v - np.dot(v, n) * n


@numba.jit(nopython=True)
def normalize_1D_jit(v):
    norm = np.sqrt(
        v[0] ** 2 +
        v[1] ** 2 +
        v[2] ** 2
    )
    return v / norm


@numba.jit(nopython=True)
def normalize_2D_jit(v):
    norms = np.sqrt(
        v[:, 0] ** 2 +
        v[:, 1] ** 2 +
        v[:, 2] ** 2
    )
    norms = np.expand_dims(norms, -1)
    return v / norms


@numba.jit(nopython=True)
def dot_jit(a, b):
    return (
            a[0] * b[0] +
            a[1] * b[1] +
            a[2] * b[2]
    )


@numba.jit(nopython=True)
def dist_jit(a, b):
    return (
            (a[0] - b[0]) ** 2 +
            (a[1] - b[1]) ** 2 +
            (a[2] - b[2]) ** 2
    ) ** 0.5
