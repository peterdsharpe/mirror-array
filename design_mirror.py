import aerosandbox.numpy as np
import pyvista as pv
from utilities.vector import normalize, distance
from utilities.coordinate_math import Plane, angle_axis_from_vectors
from utilities.reflection_math import compute_orientations
from utilities.plotter import make_plotter
from typing import List, Dict
from text_to_points.text_to_points import get_points_from_string
import copy
import multiprocessing as mp
import functools

"""
Notes:

Suffixes:
* _p = pixel planar coordinates
* _3 = 3D coordinates

Units in inches.

"""
if __name__ == '__main__':

    ### Inputs
    debug = True
    # Source properties
    source_location = 15 * 12 * np.array([
        np.cosd(10), 0, np.sind(10)
    ])

    # Mirror properties
    n_rings = 7  # How many triangle-edges from the center of the big mirror to the outside?

    N = 6 * n_rings ** 2  # The total number of triangles there will be

    # Target properties
    target_message = "marta\nyou are\nmy light"
    target_scale = 8  # How many units tall should each letter be on the target plane?

    target_plane = Plane(
        origin_3=np.array([36, 0, -41.5]),
        normal_3=np.array([0, 0, 1]),
        x_hat_3=np.array([0, -1, 0])
    )
    actual_source_location = np.copy(source_location)
    # actual_source_location = 15 * 12 * normalize(source_location)
    actual_focal_plane = Plane(
        origin_3=np.array([36, 0, -41.5]) * 2,
        normal_3=np.array([0, 0, 1]),
    )

    # Optimizer parameters
    use_cached_solution = True
    temp_start_rel = 1
    temp_end_rel = 1e-8
    n_iter = 1e8  # Note: Asymptotic runtime of ~220,000 iterations/second on my machine, 100,000 on SuperCloud.
    verbose = False

    ### Setup
    print("Generating targets...")
    targets_p = target_scale * get_points_from_string(
        s=target_message,
        n_points=N,
        kerning=1.2
    )
    print("Targets generated.")
    targets_3 = target_plane.to_3D(targets_p)
    center_of_targets = target_plane.to_3D(np.array([[0, 0]]))[0, :]
    center_of_mirrors = np.array([0, 0, 0])
    mirror_to_source = normalize(source_location - center_of_mirrors)
    mirror_to_target = normalize(center_of_targets - center_of_mirrors)
    mirror_plane_normal = normalize(mirror_to_source + mirror_to_target)

    ### Compute locations of mirrors
    mirror_plane = Plane(
        origin_3=center_of_mirrors,
        normal_3=mirror_plane_normal,
        x_hat_3=np.array([0, 1, 0])
    )

    from utilities.mesh_hexagon import mesh_hexagon

    base_faces: List[pv.PolyData] = mesh_hexagon(
        n_rings=n_rings,
    )

    from single_mirror_template import (
        tesselation_base,
        tesselation_height,
        mirror as mirror_template,
        bevel as bevel_template,
        base as base_template
    )
    from utilities.barycentric import bary_to_cart

    for base_face in base_faces:
        base_face.points *= np.array([tesselation_base, tesselation_height, 1]).reshape((1, -1))

    ### Make mirror faces and bevels
    mirror_faces = []
    bevel_faces = []

    from utilities.misc import get_index_of_unique

    for base_face in base_faces:
        top_vertex_index = get_index_of_unique([
            distance(base_face.points[1, :], base_face.points[2, :]),
            distance(base_face.points[2, :], base_face.points[0, :]),
            distance(base_face.points[0, :], base_face.points[1, :]),
        ])
        top_vertex = base_face.points[top_vertex_index, :]
        left_vertex = base_face.points[(top_vertex_index + 1) % 3, :]
        right_vertex = base_face.points[(top_vertex_index + 2) % 3, :]
        this_vertices = [
            top_vertex,
            left_vertex,
            right_vertex,
        ]
        this_mirror = pv.PolyData(
            bary_to_cart(
                mirror_template.points_bary,
                vertices_cart=this_vertices
            ),
            faces=mirror_template.faces
        )
        if this_mirror.face_normals[0, 2] < 0:
            this_mirror.flip_normals()

        this_bevel = pv.PolyData(
            bary_to_cart(
                bevel_template.points_bary,
                vertices_cart=this_vertices
            ),
            faces=bevel_template.faces
        )
        if this_bevel.face_normals[0, 2] < 0:
            this_bevel.flip_normals()

        # put everything on the mirrors plane
        base_face.points = mirror_plane.to_3D(base_face.points[:, :2])
        this_mirror.points = mirror_plane.to_3D(this_mirror.points[:, :2])
        this_bevel.points = mirror_plane.to_3D(this_bevel.points[:, :2])

        mirror_faces.append(this_mirror)
        bevel_faces.append(this_bevel)

    mirrors_3 = np.stack(  # The locations of the centers of the mirrors.
        [
            base.center_of_mass()
            for base in base_faces
        ],
        axis=0
    )  # shape: (tri_id, axis_id)

    ### Assign targets

    if use_cached_solution:
        print("Using cached mirror-target matching.")
        best_mirror_reordering = np.loadtxt("cache/mirror_order.txt", dtype=int)
    else:
        print("Optimizing mirror-target matching...")
        from utilities.optimize_targets import *
        import time

        start = time.perf_counter()

        kwargs = dict(
                mirrors_3=mirrors_3,
                targets_3=targets_3,
                temp_start_rel=temp_start_rel,
                temp_end_rel=temp_end_rel,
                n_iter=n_iter,
                verbose=verbose,
        )

        # best_mirror_reordering, _ = optimize_targets(**kwargs)
        print(f"Parallelizing on {mp.cpu_count()} CPUs...")
        with mp.Pool(mp.cpu_count()) as p:
            results = [
                p.apply_async(
                    functools.partial(optimize_targets, **kwargs)
                )
                for _ in range(mp.cpu_count())
            ]
            results = [res.get() for res in results]
            orders = [res[0] for res in results]
            losses = [res[1] for res in results]
        best_cpu = np.argmin(losses)
        best_mirror_reordering = orders[best_cpu]

        end = time.perf_counter()
        assert len(best_mirror_reordering) == len(np.unique(best_mirror_reordering))
        print(f"Optimized in {end - start} seconds.")
        np.savetxt("cache/mirror_order.txt", best_mirror_reordering, fmt='%i')

    base_faces = np.array(base_faces)[best_mirror_reordering]
    mirror_faces = np.array(mirror_faces)[best_mirror_reordering]
    bevel_faces = np.array(bevel_faces)[best_mirror_reordering]
    mirrors_3 = mirrors_3[best_mirror_reordering]

    ### Compute orientations
    mirror_normals_3 = compute_orientations(
        source_locations=source_location,
        mirror_locations=mirrors_3,
        target_locations=targets_3
    )

    ### Make actual mesh
    # Rotate the mirrors and their bevels to the necessary orientations
    for i in range(N):
        starting_normal = mirror_faces[i].face_normals[0, :]
        angle, axis = angle_axis_from_vectors(
            starting_normal,
            mirror_normals_3[i, :]
        )
        rotation_center = base_faces[i].center_of_mass()
        mirror_faces[i].rotate_vector(
            vector=axis,
            angle=180 / np.pi * angle,
            point=rotation_center,
            inplace=True
        )
        bevel_faces[i].rotate_vector(
            vector=axis,
            angle=180 / np.pi * angle,
            point=rotation_center,
            inplace=True
        )
        assert np.allclose(
            mirror_faces[i].face_normals[0, :],
            mirror_normals_3[i, :],
            atol=1e-4,
            rtol=1e-4,
        )

    mirrors = np.array([])

    # Determine how far back the base needs to be
    all_points = np.concatenate(
        [m.points for m in mirror_faces] + [b.points for b in bevel_faces],
        axis=0
    )
    base_depth = np.min([
        np.dot(point - mirror_plane.origin_3, mirror_plane.normal_3)
        for point in all_points
    ])
    base_depth_vector = base_depth * mirror_plane.normal_3

    # Push the base faces back that far
    for i in range(N):
        base_faces[i].points += np.reshape(base_depth_vector, (1, 3))

    ### Extrude all faces into solids
    mirror_solids = []
    bevel_solids = []
    base_solids = []

    from single_mirror_template import bevel_height

    for i in range(N):
        mirror_solid: pv.PolyData = mirror_faces[i].extrude(
            vector=1.5 * base_depth_vector,
            capping=True
        )

        bevel_solid_down: pv.PolyData = bevel_faces[i].extrude(
            vector=1.5 * base_depth_vector,
            capping=True
        )

        bevel_solid_up: pv.PolyData = bevel_faces[i].extrude(
            vector=bevel_height * mirror_plane.normal_3,
            capping=True
        )

        base_solid: pv.PolyData = base_faces[i].extrude(
            vector=1.5 * base_depth_vector,
            capping=True
        )

        mirror_solids.append(mirror_solid)
        bevel_solids.append(pv.PolyData().merge([bevel_solid_down, bevel_solid_up]))
        base_solids.append(base_solid)

    # Merge everything
    model = pv.PolyData().merge(
        mirror_solids + bevel_solids + base_solids
    )
    print("Generating, partitioning, and writing print files...")
    model_mm = copy.deepcopy(model)

    angle, axis = angle_axis_from_vectors(
        mirror_plane.normal_3,
        [0, 0, 1]
    )
    model_mm.rotate_vector(axis, angle * 180 / np.pi, point=mirror_plane.origin_3, inplace=True)
    model_mm.scale(25.4, inplace=True)

    model_mm.save("to_print/print.stl")
    # model_mm.plot(show_grid=True)
    print("Written.")

    ### Draw everything
    p = make_plotter("Mirror for Marta")
    p.add_light(pv.Light(
        position=actual_source_location, focal_point=mirror_plane.origin_3
    ))

    p.add_mesh(model)

    for face in mirror_faces:
        p.add_mesh(face,
                   color=np.array([242, 222, 197]) / 255,
                   pbr=True,
                   # metallic=1,
                   # roughness=0,
                   )

    p.add_points(  # Draw the intended targets
        targets_3,
        color=0 * np.ones(3),
        point_size=5,
        opacity=0.5
    )

    # Draw optics and actual targets
    for i in range(N):
        ### Compute *actual* optics
        actual_normal = mirror_normals_3[i, :]
        actual_source_to_mirror = normalize(mirrors_3[i, :] - actual_source_location)
        actual_outgoing_ray_direction = normalize(
            actual_source_to_mirror - 2 * np.dot(actual_source_to_mirror, actual_normal) * actual_normal)
        actual_target = actual_focal_plane.intersection_with_line_3(
            line_origin_3=mirrors_3[i, :],
            line_direction_3=actual_outgoing_ray_direction,
        )

        ### Draw line to target
        p.add_mesh(
            pv.Spline(
                points=np.array([
                    mirrors_3[i, :],
                    actual_target
                ]),
                n_points=2
            ),
            opacity=0.3
        )
        p.add_points(
            actual_target,
            color=1 * np.ones(3),
            point_size=5,
            opacity=1,
        )

    # p.add_floor("-z")
    p.camera_position = 'xy'
    p.camera.roll = 90
    p.show()
