import aerosandbox.numpy as np
import pyvista as pv
from utilities.vector import normalize
from utilities.coordinate_math import Plane, angle_axis_from_vectors
from utilities.reflection_math import compute_orientations
from typing import List
from text_to_points.text_to_points import get_points_from_string
import copy

"""
Notes:

Suffixes:
* _p = pixel planar coordinates
* _3 = 3D coordinates

Units in inches.

"""

### Inputs
debug = True
# Source properties
source_location = 15 * 12 * np.array([
    np.cosd(10), 0, np.sind(10)
])

# Mirror properties
n_rings = 7  # How many triangle-edges from the center of the big mirror to the outside?
mirror_base_length = 24.65 / 25.4  # Note: assumed the mirrors are isoceles triangles, where "base" is the odd side length out.
mirror_side_lengths = 27.38 / 25.4
bevel_width = 0.8 / 25.4
gap_width = 1.5 / 25.4  # What's the gap between adjacent triangles?
bevel_height = 1.5 / 25.4

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
n_iter = 4.3e9  # Note: Asymptotic runtime of ~220,000 iterations/second on my machine.
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

### Compute mirror triangles
mirror_height_length = (
                               mirror_side_lengths ** 2 -
                               (mirror_base_length / 2) ** 2
                       ) ** 0.5

### Compute locations of mirrors
mirror_plane = Plane(
    origin_3=center_of_mirrors,
    normal_3=mirror_plane_normal,
    x_hat_3=np.array([0, 1, 0])
)

from utilities.mesh_hexagon import mesh_hexagon

mirror_faces: List[pv.PolyData] = mesh_hexagon(
    n_rings=n_rings,
    plane=mirror_plane,
    source_location=source_location
)

mirror_faces = np.array(mirror_faces)
mirrors_3 = np.stack(  # The locations of the centers of the mirrors.
    [
        mirror.center_of_mass()
        for mirror in mirror_faces
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
    best_mirror_reordering = optimize_anneal(
        mirrors_3=mirrors_3,
        targets_3=targets_3,
        temp_start_rel=temp_start_rel,
        temp_end_rel=temp_end_rel,
        n_iter=n_iter,
        verbose=verbose
    )
    end = time.perf_counter()
    assert len(best_mirror_reordering) == len(np.unique(best_mirror_reordering))
    print(f"Optimized in {end - start} seconds.")
    np.savetxt("cache/mirror_order.txt", best_mirror_reordering, fmt='%i')

mirror_faces = mirror_faces[best_mirror_reordering]
mirrors_3 = mirrors_3[best_mirror_reordering]

### Compute orientations
mirror_normals_3 = compute_orientations(
    source_locations=source_location,
    mirror_locations=mirrors_3,
    target_locations=targets_3
)


### Make actual mesh
# Rotate the mirrors to the necessary orientations
def rotate_mirror_to_normal(mirror, normal_3):
    starting_normal = mirror.face_normals[0, :]
    angle, axis = angle_axis_from_vectors(
        starting_normal,
        normal_3,
    )
    mirror.rotate_vector(vector=axis, angle=180 / np.pi * angle, point=mirror.center_of_mass())
    assert np.allclose(mirror.face_normals[0, :], normal_3, atol=1e-4, rtol=1e-4)
    return mirror


mirror_faces = [
    rotate_mirror_to_normal(mirror, normal)
    for mirror, normal in zip(mirror_faces, mirror_normals_3)
]


# Add in bevels
def make_bevel(
        mirror,
        far_corner_id=0,
):
    corners = np.roll(mirror.points, far_corner_id, axis=0)
    bevel = pv.PolyData(
        [
            corners[1, :],
            corners[2, :],
            corners[2, :] + bevel_width * normalize(corners[0, :] - corners[2, :]),
            corners[1, :] + bevel_width * normalize(corners[0, :] - corners[1, :]),
        ],
        faces=[
            4, 0, 1, 2, 3,
        ]
    )
    bevel = bevel.extrude(vector=bevel_height * mirror_plane.normal_3, capping=True)
    bevel.points_to_double()
    return bevel


bevels = []
for far_corner_id in [1, 2]:
    for mirror in mirror_faces:
        bevels.append(make_bevel(mirror, far_corner_id))

# Extrude the mirrors
mirror_corners_3 = np.concatenate(
    [
        face.points
        for face in mirror_faces
    ], axis=0
)  # shape: (corner_id, axis_id)
depth = np.dot(
    mirror_corners_3 - mirror_plane.origin_3.reshape((1, 3)),
    mirror_plane.normal_3
).max()

mirror_solids = [
    mirror.extrude(vector=-3 * depth * mirror_plane.normal_3, capping=True)
    for mirror in mirror_faces
]
for m in mirror_solids:
    m.points_to_double()

# Make the base
base = pv.Polygon(
    center=[0, 0, 0],
    normal=[0, 0, 1],
    radius=1,
    n_sides=6
)
base.points = np.array([
    [0, 1, 0],
    [rt3 / 2, 0.5, 0],
    [rt3 / 2, -0.5, 0],
    [0, -1, 0],
    [-rt3 / 2, -0.5, 0],
    [-rt3 / 2, 0.5, 0]
])
base.rotate_z(30)
base.scale(n_rings * (side_length + gap_width * rt3) + gap_width * 2 / rt3)
base.points = mirror_plane.to_3D(
    base.points[:, :2]
)

base.translate(-depth * mirror_plane.normal_3)
base = base.extrude(vector=-3 * depth * mirror_plane.normal_3, capping=True)
base.points_to_double()

# Merge everything
things = [base] + mirror_solids + bevels
things = [
    thing.triangulate()
    for thing in things
]

model = pv.PolyData().merge(things)

### Export print
print("Generating, partitioning, and writing print files...")
model_mm = copy.deepcopy(model)

angle, axis = angle_axis_from_vectors(
    mirror_plane.normal_3,
    [0, 0, 1]
)
model_mm.rotate_vector(axis, angle * 180 / np.pi, point=mirror_plane.origin_3)
model_mm.scale(25.4)

model_mm.save("to_print/print.stl")
# model_mm.plot(show_grid=True)
print("Written.")

### Draw everything
plotter = pv.Plotter()
plotter.add_light(pv.Light(
    position=actual_source_location, focal_point=mirror_plane.origin_3
))

plotter.add_mesh(model, pbr=False)

for face in mirror_faces:
    plotter.add_mesh(face,
                     color=np.array([242, 222, 197]) / 255,
                     pbr=True,
                     # metallic=1,
                     # roughness=0,
                     )

plotter.add_points(  # Draw the intended targets
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
    # plotter.add_lines(
    #     lines=np.array([
    #         mirrors_3[i, :],
    #         actual_target
    #     ]),
    #     color=0.2 * np.ones(3),
    #     width=0.5,
    # )
    plotter.add_mesh(
        pv.Spline(
            points=np.array([
                mirrors_3[i, :],
                actual_target
            ]),
            n_points=2
        ),
        opacity=0.3
    )
    plotter.add_points(
        actual_target,
        color=1 * np.ones(3),
        point_size=5,
        opacity=1,
    )

# plotter.add_mesh(
#     pv.Sphere(radius=1, center=source_location),
#     color="yellow"
# )

plotter.add_floor("-z")

plotter.add_axes()
plotter.show_grid()
plotter.title = "Mirror for Marta"
plotter.camera_position = 'xy'
plotter.camera.roll = 90
plotter.show()
