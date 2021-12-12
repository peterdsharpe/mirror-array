import numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import hexy as hx
import pyvista as pv
import stl
from utilities.coordinate_math import Plane

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
    10, 0, 0
])

hex_centers_h = hx.get_spiral(
    center=[0, 0, 0],
    radius_end=7
)
# hex_centers_h = np.array([
#     [0, 0, 0]
# ])
hex_radius = 1

mirror_plane = Plane(
    origin=np.array([0, 0, 0]),
    normal=np.array([1, 0, 0]),
    x_hat_p=np.array([0, 1, 0])
)
target_plane = Plane(
    origin=np.array([0, 0, -30]),
    normal=np.array([0, 0, 1]),
    x_hat_p=np.array([0, -1, 0])
)

### Auxiliary
N = hex_centers_h.shape[0]

### Compute locations of centers and corners in pixel coordinates
h = hex_centers_h.reshape((1, -1, 3))
dirs = hx.ALL_DIRECTIONS.reshape((6, 1, 3))

hex_corners_h = (
                        (h) +
                        (h + dirs) +
                        (h + np.roll(dirs, 1, axis=0))
                ) / 3

hex_centers_p = hx.cube_to_pixel(hex_centers_h, radius=hex_radius)
hex_corners_p = hx.cube_to_pixel(hex_corners_h.reshape((-1, 3)), radius=hex_radius).reshape((6, -1, 2))

# fig, ax = plt.subplots()
# plt.scatter(hex_centers_p[:, 0], hex_centers_p[:, 1], c="k")
# for i in range(N):
#     plt.fill(
#         hex_corners_p[:, i, 0],
#         hex_corners_p[:, i, 1],
#         alpha=0.4
#     )
# p.equal()
# p.set_ticks(1, 1, 1, 1)
# p.show_plot()

### Assign targets
t = np.linspace(0, 2 * np.pi, N, endpoint=False)
targets_p = 10 * np.stack((
    np.cos(t),
    np.sin(t),
))

targets_
