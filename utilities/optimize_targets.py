from utilities.vector import *
import numpy as np
import numba

distance_radius = 4


@numba.jit(nopython=True)
def optimize_anneal(
        mirrors_3,
        targets_3,
        n_iter=int(3e6),
):
    ### Setup
    N = len(mirrors_3)
    current_mirror_order = np.arange(N)
    np.random.shuffle(current_mirror_order)
    # current_mirror_order[0] == the id of the mirror that the first target should match with

    ### Setup loss calculation, and perform the first one as a baseline.
    target_distance_factor = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dist = dist_jit(
                targets_3[i, :],
                targets_3[j, :]
            )
            target_distance_factor[i, j] = 1 / (dist ** 2 + distance_radius)  # power law
            # target_distance_factor[i, j] = np.exp(-dist / distance_radius)
            # target_distance_factor[i, j] = 1 if dist < distance_radius else 1e-3
            # target_distance_factor[i, j] = 1 if dist < distance_radius else 1e-3

    current_ray_directions = normalize_2D_jit(mirrors_3[current_mirror_order, :] - targets_3)
    current_misalignment_factor = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            cos = dot_jit(
                current_ray_directions[i, :],
                current_ray_directions[j, :]
            )
            current_misalignment_factor[i, j] = 1 - cos

    current_losses = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            current_losses[i, j] = target_distance_factor[i, j] * current_misalignment_factor[i, j]
    current_loss = np.sum(current_losses)

    ### Initialize iteration
    best_mirror_order = np.copy(current_mirror_order)
    best_loss = current_loss

    temperature_start = current_loss / N
    temperature_end = current_loss / N / 1e9
    temperature = temperature_start
    alpha = (temperature_end/temperature_start) ** (1 / n_iter)

    ### Iterate
    for iteration in range(n_iter):
        ### Pick two mirrors to swap
        # swap1 and swap2 say that target[swap1] and target[swap2] switch mirrors.
        swap1 = np.random.randint(0, N - 1)
        swap2 = np.random.randint(0, N - 1)
        while swap2 == swap1:
            swap2 = np.random.randint(0, N - 1)

        ### Figure out how much loss these mirrors contribute in the status quo
        swap1_loss_status_quo = 0
        swap2_loss_status_quo = 0
        for i in range(N):
            if i < swap1:
                swap1_loss_status_quo += current_losses[i, swap1]
            else:
                swap1_loss_status_quo += current_losses[swap1, i]
            if i < swap2:
                swap2_loss_status_quo += current_losses[i, swap2]
            else:
                swap2_loss_status_quo += current_losses[swap2, i]
        swap_loss_status_quo = swap1_loss_status_quo + swap2_loss_status_quo

        ### Figure out how much loss these mirrors would contribute if you swapped them
        swap1_ray_new = normalize_1D_jit(mirrors_3[current_mirror_order[swap2], :] - targets_3[swap1, :])
        swap1_losses_new = np.zeros((N))
        for i in range(N):
            if i == swap1:
                continue
            ### Compute misalignment factor
            cos = dot_jit(
                current_ray_directions[i, :],
                swap1_ray_new
            )
            new_misalignment_factor = 1 - cos

            ### Compute loss
            swap1_losses_new[i] = target_distance_factor[swap1, i] * new_misalignment_factor
        swap1_loss_new = np.sum(swap1_losses_new)

        swap2_ray_new = normalize_1D_jit(mirrors_3[current_mirror_order[swap1], :] - targets_3[swap2, :])
        swap2_losses_new = np.zeros((N))
        for i in range(N):
            if i == swap2:
                continue
            ### Compute misalignment factor
            cos = dot_jit(
                current_ray_directions[i, :],
                swap2_ray_new
            )
            new_misalignment_factor = 1 - cos

            ### Compute loss
            swap2_losses_new[i] = target_distance_factor[swap2, i] * new_misalignment_factor
        swap2_loss_new = np.sum(swap2_losses_new)

        swap_loss_new = swap1_loss_new + swap2_loss_new
        delta_loss = swap_loss_new - swap_loss_status_quo

        ### Determine whether to execute the swap
        if delta_loss < 0:
            execute_swap = True
        else:
            execute_swap = np.random.rand() < np.exp(-delta_loss / temperature)
        temperature *= alpha

        # Execute the swap
        if execute_swap:

            reordering = np.arange(N)
            reordering[swap1] = swap2
            reordering[swap2] = swap1

            current_mirror_order = current_mirror_order[reordering]
            current_ray_directions[swap1] = swap1_ray_new
            current_ray_directions[swap2] = swap2_ray_new

            for i in range(N):
                if i < swap1:
                    current_losses[i, swap1] = swap1_losses_new[i]
                else:
                    current_losses[swap1, i] = swap1_losses_new[i]
                if i < swap2:
                    current_losses[i, swap2] = swap2_losses_new[i]
                else:
                    current_losses[swap2, i] = swap2_losses_new[i]

            current_loss = 0
            for i in range(N):
                for j in range(i + 1, N):
                    current_loss += current_losses[i, j]

            if current_loss < best_loss:
                best_mirror_order = np.copy(current_mirror_order)
                best_loss = current_loss
                print(iteration, current_loss, delta_loss, temperature)

    print(temperature_end)

    return best_mirror_order


# @numba.jit(nopython=True, parallel=True)
# def get_losses(
#         mirrors_3,
#         targets_3,
# ):
#     N = len(mirrors_3)
#     distance_factor = np.zeros((N, N))
#     for i in range(N):
#         for j in range(i, N):
#             ti = targets_3[i, :]
#             tj = targets_3[j, :]
#             dist_sqr = (
#                     (ti[0] - tj[0]) ** 2 +
#                     (ti[1] - tj[1]) ** 2 +
#                     (ti[2] - tj[2]) ** 2
#             )
#             distance_factor[i, j] = 1 / (dist_sqr + distance_radius)
#
#     ray_direction = normalize_jit(targets_3 - mirrors_3)
#
#     misalignment_factor = np.zeros((N, N))
#     for i in range(N):
#         for j in range(i, N):
#             ri = ray_direction[i]
#             rj = ray_direction[j]
#             cos = (
#                     ri[0] * rj[0] +
#                     ri[1] * rj[1] +
#                     ri[2] * rj[2]
#             )
#             misalignment_factor[i, j] = 1 - cos
#
#     losses = np.zeros((N, N))
#     for i in range(N):
#         for j in range(i, N):
#             losses[i, j] = distance_factor[i, j] * misalignment_factor[i, j]
#
#     return losses
#
#
# def optimize_none(N):
#     return np.arange(N)
#
#
# def optimize_naive(
#         mirrors_3,
#         targets_3,
#         n_iter=1000,
# ):
#     order = np.arange(len(mirrors_3))
#
#     best_order = copy.copy(order)
#     best_loss = loss(mirrors_3, targets_3[order, :])
#
#     try:
#         for i in range(n_iter):
#             np.random.shuffle(order)
#             this_loss = loss(mirrors_3, targets_3[order, :])
#             if this_loss < best_loss:
#                 best_order = order
#                 best_loss = this_loss
#                 print(this_loss)
#     except KeyboardInterrupt:
#         pass
#
#     return best_order
#
#
# def optimize_bartlett(
#         mirror_faces,
#         mirrors_p,
#         targets_p,
#         partition_by="ring",
# ):
#     assert targets_p.shape[1] == 2
#
#     mean_mirror = np.mean(mirrors_p, axis=0)
#     mean_target = np.mean(targets_p, axis=0)
#
#     mirrors_p = mirrors_p - mean_mirror
#     targets_p = targets_p - mean_target
#
#     if partition_by == "ring":
#         mirror_radius = np.array([m.ring_number for m in mirror_faces])
#     elif partition_by == "radius":
#         n_partitions = 7
#         mirror_radius = np.linalg.norm(targets_p, axis=1)
#         mirror_radius = mirror_radius / np.max(mirror_radius)
#         mirror_radius = np.round(mirror_radius * n_partitions).astype(int)
#     else:
#         raise ValueError()
#
#     # Compute statistics
#     mirror_azimuth = np.arctan2(mirrors_p[:, 1], mirrors_p[:, 0]) * 180 / np.pi
#     target_azimuth = np.arctan2(targets_p[:, 1], -targets_p[:, 0]) * 180 / np.pi
#
#     target_radius = np.linalg.norm(targets_p, axis=1)
#
#     # Assign targets
#     remaining_targets_sorted_by_radius = np.argsort(target_radius)
#     mirror_order = []
#     target_order = []
#
#     for ring in np.sort(np.unique(mirror_radius)):
#         n_in_ring = np.sum(mirror_radius == ring)
#
#         # Targets
#         targets_in_ring = remaining_targets_sorted_by_radius[:n_in_ring]
#         remaining_targets_sorted_by_radius = remaining_targets_sorted_by_radius[n_in_ring:]
#
#         targets_in_ring_sorted_by_azimuth = targets_in_ring[np.argsort(target_azimuth[targets_in_ring])]
#
#         target_order.extend(list(targets_in_ring_sorted_by_azimuth))
#
#         # Mirrors
#         mirrors_in_ring = np.argwhere(mirror_radius == ring)[:, 0]
#         mirrors_in_ring_sorted_by_azimuth = mirrors_in_ring[np.argsort(mirror_azimuth[mirrors_in_ring])]
#
#         mirror_order.extend(list(mirrors_in_ring_sorted_by_azimuth))
#
#     mirror_order = np.array(mirror_order)
#     target_order = np.array(target_order)
#
#     order = np.arange(len(mirror_order))
#     order[mirror_order] = target_order
#
#     return order

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
