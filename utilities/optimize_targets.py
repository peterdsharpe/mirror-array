from utilities.vector import normalize
import numpy as np
from scipy import spatial
import copy
import numba


def loss(
        mirrors_3,
        targets_3,
):
    mirror_to_target = normalize(targets_3 - mirrors_3)

    cosine = np.einsum(
        "ik,jk->ij",
        mirror_to_target,
        mirror_to_target,
    )

    pairwise_distances = spatial.distance.squareform(
        spatial.distance.pdist(mirror_to_target)
    )

    distance_factor = 1 / ((pairwise_distances / 1) ** 2 + 1)

    return np.mean(
        distance_factor * (1 - cosine),
    )


def optimize_by_permuting(
        mirrors_3,
        targets_3,
        N=1000,
):
    order = np.arange(len(mirrors_3))

    best_order = copy.copy(order)
    best_loss = loss(mirrors_3, targets_3[order, :])

    for i in range(N):
        np.random.shuffle(order)
        this_loss = loss(mirrors_3, targets_3[order, :])
        if this_loss < best_loss:
            best_order = order
            best_loss = this_loss
            print(this_loss)

    return best_order
