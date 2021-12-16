import numpy as np
import numba


@numba.njit
def normalize_1D_jit(v):
    norm = np.sqrt(
        v[0] ** 2 +
        v[1] ** 2 +
        v[2] ** 2
    )
    return v / norm


@numba.njit
def normalize_2D_jit(v):
    norms = np.sqrt(
        v[:, 0] ** 2 +
        v[:, 1] ** 2 +
        v[:, 2] ** 2
    )
    norms = np.expand_dims(norms, -1)
    return v / norms


@numba.njit
def dot_jit(a, b):
    return (
            a[0] * b[0] +
            a[1] * b[1] +
            a[2] * b[2]
    )


@numba.njit
def dist_jit(a, b):
    return (
                   (a[0] - b[0]) ** 2 +
                   (a[1] - b[1]) ** 2 +
                   (a[2] - b[2]) ** 2
           ) ** 0.5
