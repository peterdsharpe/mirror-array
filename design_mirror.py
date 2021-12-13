import numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import hexy as hx
import pyvista as pv
import stl
from utilities.vector import normalize
from utilities.coordinate_math import Plane, angle_axis_from_vectors
from utilities.reflection_math import compute_orientations
from scipy import spatial

"""
Notes:

Suffixes:
* _h = hexagonal (cube) planar coordinates
* _p = pixel planar coordinates
* _3 = 3D coordinates

Units in inches.

"""

### Inputs
source_location = np.array([
    1e6, 0, 0
])

mirrors_h = hx.get_spiral(
    center=[0, 0, 0],
    radius_end=5
)
radius_p = 1
N = mirrors_h.shape[0]

t = np.linspace(0, 2 * np.pi, N, endpoint=False)
targets_p = 10 * np.stack((
    np.cos(t),
    np.sin(t),
), axis=-1)


mirror_plane = Plane(
    origin_3=np.array([0, 0, 0]),
    normal_3=np.array([1, 0, 0]),
    x_hat_3=np.array([0, 1, 0])
)
target_plane = Plane(
    origin_3=np.array([25, 0, -20]),
    normal_3=np.array([0, 0, 1]),
    x_hat_3=np.array([0, -1, 0])
)
focal_plane = Plane(
    origin_3=np.array([25, 0, -20]),
    normal_3=target_plane.normal_3,
    x_hat_3=target_plane.x_hat_3,
)

### Compute locations of centers and corners in pixel coordinates
mirrors_p = hx.cube_to_pixel(mirrors_h, radius=radius_p)
mirrors_3 = mirror_plane.to_3D(mirrors_p)

h = mirrors_h.reshape((1, -1, 3))
dirs = hx.ALL_DIRECTIONS.reshape((6, 1, 3))

mirror_corners_h = (
                           (h) +
                           (h + dirs) +
                           (h + np.roll(dirs, 1, axis=0))
                   ) / 3

mirror_corners_p = hx.cube_to_pixel(mirror_corners_h.reshape((-1, 3)), radius=radius_p).reshape((6, -1, 2))

### Plot mirror plane
fig, ax = plt.subplots()
plt.scatter(mirrors_p[:, 0], mirrors_p[:, 1], c="k")
for i in range(N):
    plt.fill(
        mirror_corners_p[:, i, 0],
        mirror_corners_p[:, i, 1],
        alpha=0.4
    )
p.equal()
p.set_ticks(1, 1, 1, 1)
p.show_plot()

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

### Draw mirror
plotter = pv.Plotter()
plotter.add_points(
    targets_3,
    color=(0.2, 0.2, 0.2)
)
for i in range(N):
    ### Draw mirror
    poly = pv.Polygon(
        center=(0, 0, 0),
        radius=radius_p,
        normal=mirror_plane.normal_3,
        n_sides=6
    )
    angle, axis = angle_axis_from_vectors(
        mirror_plane.normal_3,
        mirror_normals_3[i, :],
    )
    poly.rotate_vector(vector=axis, angle=180 / np.pi * angle)
    poly.rotate_vector(vector=[0,1,0], angle=5)
    poly.translate(mirrors_3[i, :])
    plotter.add_mesh(poly)

    ### Compute *actual* optics
    actual_normal = poly.point_normals[0,:]
    actual_source_to_mirror = normalize(mirrors_3[i, :] - source_location)
    actual_outgoing_ray_direction = normalize(actual_source_to_mirror - 2 * np.dot(actual_source_to_mirror, actual_normal) * actual_normal)
    actual_target = focal_plane.intersection_with_line_3(
        line_origin_3=mirrors_3[i,:],
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
