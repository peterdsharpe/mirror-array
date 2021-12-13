import numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
# import hexy as hx
import pyvista as pv
import stl
from utilities.vector import normalize
from utilities.coordinate_math import Plane, angle_axis_from_vectors
from utilities.reflection_math import compute_orientations
from scipy import spatial
import copy

"""
Notes:

Suffixes:
* _p = pixel planar coordinates
* _3 = 3D coordinates

Units in inches.

"""

### Inputs
# Source properties
source_location = np.array([
    1e6, 0, 0
])

# Mirror properties
size = 5  # How many triangle-edges from the center of the big mirror to the outside?
mirror_side_length = 1  # What's the side length of the triangles?
bevel_width = 1.5 / 25.4
side_length = mirror_side_length + bevel_width
spacing = 2 / 25.4  # What's the gap between adjacent triangles?
bevel_height = 1.5 / 25.4

mirror_plane = Plane(
    origin_3=np.array([0, 0, 0]),
    normal_3=np.array([1, 0, -0.3]),
    x_hat_3=np.array([0, 1, 0])
)

N = 6 * size ** 2 # The total number of triangles there will be

# Target properties
t = np.linspace(0, 2 * np.pi, N, endpoint=False)
targets_p = 10 * np.stack((
    np.cos(t),
    np.sin(t),
), axis=-1)

target_plane = Plane(
    origin_3=np.array([20, 0, -20]),
    normal_3=np.array([0, 0, 1]),
    x_hat_3=np.array([0, -1, 0])
)
focal_plane = Plane(
    origin_3=np.array([20, 0, -20]),
    normal_3=target_plane.normal_3,
    x_hat_3=target_plane.x_hat_3,
)

### Compute locations of mirrors
mirrors = []

rt3 = np.sqrt(3)
for ring_number in np.arange(size) + 1:
    inside_radius = (ring_number - 1) * rt3 / 2
    for side in np.arange(6):
        for stride in np.arange(1 - ring_number, ring_number)[::-1]:
            tri = pv.Triangle([
                [0, 0.5 * rt3, 0],
                [0.5, 0, 0],
                [-0.5, 0, 0],
            ])

            if (ring_number + stride) % 2 == 1:
                tri.flip_y()

            tri.translate([0, inside_radius, 0])

            tri.translate([stride / 2, 0, 0])

            tri.rotate_z(60 * side)

            tri.scale(side_length + spacing * rt3)

            center = tri.center_of_mass()
            tri.translate(-center)
            tri.scale(side_length / (side_length + spacing * rt3))
            tri.translate(center)

            tri.points = mirror_plane.to_3D(tri.points[:, :2])

            if np.dot(tri.face_normals[0, :], source_location - tri.center_of_mass()) < 0:
                tri.flip_normals()

            mirrors.append(tri)

assert N == len(mirrors)

mirrors_3 = np.stack(
    [
        mirror.center_of_mass()
        for mirror in mirrors
    ],
    axis=0
)  # shape: (tri_id, axis_id)

### Assign targets
targets_3 = target_plane.to_3D(targets_p)

from utilities.optimize_targets import *

best_order = optimize_none(
    mirrors_3, targets_3
)

targets_3 = targets_3[best_order]

### Compute orientations
mirror_to_source = normalize(source_location - mirrors_3)
mirror_to_target = normalize(targets_3 - mirrors_3)

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
    assert np.allclose(mirror.face_normals[0, :], normal_3, atol=1e-5, rtol=1e-5)
    return mirror


mirrors = [
    rotate_mirror_to_normal(mirror, normal)
    for mirror, normal in zip(mirrors, mirror_normals_3)
]
mirror_faces = copy.deepcopy(mirrors)

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
    return bevel


mirror_corners_3 = np.stack(
    [
        np.stack([
            mirror.points[point_id, :]
            for mirror in mirrors
        ], axis=0)
        for point_id in range(3)
    ], axis=0
)  # shape: (point_id, tri_id, axis_id)

bevels = [
             make_bevel(mirror, far_corner_id=1)
             for mirror in mirrors
         ] + [
             make_bevel(mirror, far_corner_id=2)
             for mirror in mirrors
         ]

# Extrude the mirrors
mirrors = [
    mirror.extrude(vector=-1 * mirror_plane.normal_3 * side_length, capping=True)
    for mirror in mirrors
]

# Make the base
base = pv.Polygon(
    center=[0, 0, 0],
    normal=[0, 0, 1],
    radius=size * (side_length + spacing * rt3) + spacing * 2 / rt3,
    n_sides=6
)
base.rotate_z(30)
base.points = mirror_plane.to_3D(
    base.points[:, :2]
)

base.translate(-0.4 * side_length * mirror_plane.normal_3)
base = base.extrude(vector=-1 * mirror_plane.normal_3 * side_length, capping=True)

### Draw everything
plotter = pv.Plotter(lighting="three lights")

for mirror in mirrors:  # Draw the mirrors
    plotter.add_mesh(mirror)

for face in mirror_faces:
    plotter.add_mesh(face, color=np.array([242, 222, 197]) / 255)

plotter.add_mesh(base)

for bevel in bevels:
    plotter.add_mesh(bevel)

plotter.add_points(  # Draw the intended targets
    targets_3, color=(0.2, 0.2, 0.2)
)

# Draw optics and actual targets
for i in range(N):
    ### Compute *actual* optics
    actual_normal = mirror_normals_3[i, :]
    actual_source_to_mirror = normalize(mirrors_3[i, :] - source_location)
    actual_outgoing_ray_direction = normalize(
        actual_source_to_mirror - 2 * np.dot(actual_source_to_mirror, actual_normal) * actual_normal)
    actual_target = focal_plane.intersection_with_line_3(
        line_origin_3=mirrors_3[i, :],
        line_direction_3=actual_outgoing_ray_direction,
    )

    ### Draw line to target
    plotter.add_lines(
        lines=np.array([
            mirrors_3[i, :],
            actual_target
        ]),
        color=(0.2, 0.2, 0.2),
        width=1
    )
    plotter.add_points(points=actual_target)

# plotter.add_mesh(
#     pv.Sphere(radius=1, center=source_location),
#     color="yellow"
# )

plotter.add_axes()
# plotter.show_grid()
plotter.show()
