from utilities.vector_jit import *
import numpy as np
import numba

distance_radius = 8


@numba.njit(fastmath=True)
def optimize_anneal(
        mirrors_3,
        targets_3,
        temp_start_rel=1,
        temp_end_rel=1e-7,
        n_iter=5e6,
        verbose=False,
):
    n_iter = int(n_iter)

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
            # target_distance_factor[i, j] = 1 / (dist + distance_radius)  # power law
            # target_distance_factor[i, j] = 1 / (dist ** 2 + distance_radius)  # power law
            target_distance_factor[i, j] = np.exp(-dist / distance_radius)  # Exponential
            # target_distance_factor[i, j] = 1 if dist < distance_radius else 1e-3  # Hard wall
            # target_distance_factor[i, j] = np.exp(-(dist / distance_radius) ** 2)  # Gaussian

    current_ray_directions = normalize_2D_jit(mirrors_3[current_mirror_order, :] - targets_3)

    def misalignment_function(ray1, ray2):
        return 1 - dot_jit(ray1, ray2)  # Cosine distance
        # return dist_jit(ray1, ray2)  # Euclidian distance

    current_misalignment_factor = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            current_misalignment_factor[i, j] = misalignment_function(
                current_ray_directions[i, :],
                current_ray_directions[j, :],
            )

    current_losses = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            current_losses[i, j] = target_distance_factor[i, j] * current_misalignment_factor[i, j]
    current_loss = np.sum(current_losses)

    ### Initialize iteration
    best_mirror_order = np.copy(current_mirror_order)
    best_loss = current_loss

    temperature_start = current_loss / N * temp_start_rel
    temperature_end = current_loss / N * temp_end_rel
    temperature = temperature_start
    alpha = (temperature_end / temperature_start) ** (1 / n_iter)

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
            new_misalignment_factor = misalignment_function(
                current_ray_directions[i, :],
                swap1_ray_new
            )

            ### Compute loss
            swap1_losses_new[i] = target_distance_factor[swap1, i] * new_misalignment_factor
        swap1_loss_new = np.sum(swap1_losses_new)

        swap2_ray_new = normalize_1D_jit(mirrors_3[current_mirror_order[swap1], :] - targets_3[swap2, :])
        swap2_losses_new = np.zeros((N))
        for i in range(N):
            if i == swap2:
                continue
            ### Compute misalignment factor
            new_misalignment_factor = misalignment_function(
                current_ray_directions[i, :],
                swap2_ray_new
            )

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
                if verbose:
                    print(iteration, current_loss, delta_loss, temperature)

        if iteration * 20 % n_iter == 0:
            print(f"Iteration {iteration} of {n_iter}...")

    print("Final loss:", best_loss)

    return best_mirror_order
