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
        spatial.distance.pdist(targets_3)
    )

    distance_factor = 1 / ((pairwise_distances / 1) ** 2 + 1)

    return np.sum(
        distance_factor * (1 - cosine),
    )

def optimize_none(
        mirrors_3,
        targets_3,
):
    return np.arange(len(mirrors_3))

def optimize_naive(
        mirrors_3,
        targets_3,
):
    order = np.arange(len(mirrors_3))

    best_order = copy.copy(order)
    best_loss = loss(mirrors_3, targets_3[order, :])

    try:
        while True:
            np.random.shuffle(order)
            this_loss = loss(mirrors_3, targets_3[order, :])
            if this_loss < best_loss:
                best_order = order
                best_loss = this_loss
                print(this_loss)
    except KeyboardInterrupt:
        pass

    return best_order


def optimize_anneal(
        mirrors_3,
        targets_3,
):
    N = len(mirrors_3)
    order = np.arange(N)

    ### Compute original loss
    pairwise_distances_sqr = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            ti = targets_3[i, :]
            tj = targets_3[j, :]
            pairwise_distances_sqr[i, j] = (
                    (ti[0] - tj[0]) ** 2 +
                    (ti[1] - tj[1]) ** 2 +
                    (ti[2] - tj[2]) ** 2
            )
    mirror_to_target = targets_3 - mirrors_3
    mirror_to_target = mirror_to_target / np.expand_dims(np.linalg.norm(mirror_to_target, axis=-1), axis=-1)

    cosine = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            ri = mirror_to_target[i]
            rj = mirror_to_target[j]
            cosine[i, j] = (
                    ri[0] * rj[0] +
                    ri[1] * rj[1] +
                    ri[2] * rj[2]
            )

    losses = (
            (1 / pairwise_distances_sqr + 1) *
            (1 - cosine)
    )
    loss = np.sum(losses)

    best_order = copy.copy(order)
    for i in range(10 ** 3):
        ### Pick two indices to swap
        swap_1 = np.random.randint(0, N - 1)
        swap_2 = np.random.randint(0, N - 1)
        while swap_2 == swap_1:
            swap_2 = np.random.randint(0, N - 1)
        indices = np.array([swap_1, swap_2])

        ### Swap two indices in the order
        temp = order[swap_2]
        order[swap_2] = order[swap_1]
        order[swap_1] = temp

        ### Calculate the loss
        pairwise_distances_sqr = pairwise_distances_sqr[order.reshape(-1, 1), order]

        cosine = cosine[order.reshape(-1, 1), order]

        losses = (
                (1 / (pairwise_distances_sqr + 1)) *
                (1 - cosine)
        )
        print(np.sum(losses))

    return best_order


if __name__ == '__main__':

    mirrors_3 = np.array([[0., 0., 0.],
                          [0., -1.73205081, 0.],
                          [0., -0.8660254, 1.5],
                          [0., 0.8660254, 1.5],
                          [0., 1.73205081, 0.],
                          [0., 0.8660254, -1.5],
                          [0., -0.8660254, -1.5]])

    targets_3 = np.array([[0., -10., -20.],
                          [7.81831482, -6.23489802, -20.],
                          [9.74927912, 2.22520934, -20.],
                          [4.33883739, 9.00968868, -20.],
                          [-4.33883739, 9.00968868, -20.],
                          [-9.74927912, 2.22520934, -20.],
                          [-7.81831482, -6.23489802, -20.]])

    best_order = optimize_naive(mirrors_3, targets_3)
