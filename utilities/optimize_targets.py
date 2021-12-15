from utilities.vector import normalize
import numpy as np
from scipy import spatial
import copy


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


def optimize_none(N):
    return np.arange(N)


def optimize_naive(
        mirrors_3,
        targets_3,
        n_iter: int,
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


def optimize_bartlett(
        mirrors,
        mirrors_p,
        targets_p,
):
    assert targets_p.shape[1] == 2

    mean_mirror = np.mean(mirrors_p, axis=0)
    mean_target = np.mean(targets_p, axis=0)

    mirrors_p = mirrors_p - mean_mirror
    targets_p = targets_p - mean_target

    id = np.arange(len(mirrors_p))

    ring_numbers = np.array([m.ring_number for m in mirrors])

    # Compute statistics
    mirror_azimuth = np.arctan2(mirrors_p[:, 1], mirrors_p[:, 0]) * 180 / np.pi
    target_azimuth = np.arctan2(targets_p[:, 1], -targets_p[:, 0]) * 180 / np.pi

    target_radius = np.linalg.norm(targets_p, axis=1)

    # Assign targets
    remaining_mirrors_sorted_by_ring = np.argsort(ring_numbers)
    remaining_targets_sorted_by_radius = np.argsort(target_radius)
    mirror_order = []
    target_order = []

    for ring in np.sort(np.unique(ring_numbers)):
        n_in_ring = np.sum(ring_numbers == ring)

        # Targets
        targets_in_ring = remaining_targets_sorted_by_radius[:n_in_ring]
        remaining_targets_sorted_by_radius = remaining_targets_sorted_by_radius[n_in_ring:]

        targets_in_ring_sorted_by_azimuth = targets_in_ring[np.argsort(target_azimuth[targets_in_ring])]

        target_order.extend(list(targets_in_ring_sorted_by_azimuth))

        # Mirrors
        mirrors_in_ring = np.argwhere(ring_numbers == ring)[:,0]
        mirrors_in_ring_sorted_by_azimuth = mirrors_in_ring[np.argsort(mirror_azimuth[mirrors_in_ring])]

        mirror_order.extend(list(mirrors_in_ring_sorted_by_azimuth))

    mirror_order = np.array(mirror_order)
    target_order = np.array(target_order)

    # order = mirror_order[target_order]
    order = np.arange(len(mirror_order))
    order[mirror_order] = target_order

    return order


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
