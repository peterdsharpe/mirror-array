from utilities.barycentric import bary_to_cart, cart_to_bary
import aerosandbox.numpy as np
import pyvista as pv
import copy

mirror_base = b = 24.65 / 25.4  # Note: assumed the mirrors are isoceles triangles, where "base" is the odd side length out.
mirror_side = s = 27.38 / 25.4
bevel_width = bev = 1 / 25.4
gap_width = gap = 1 / 25.4  # What's the gap between adjacent triangles?

mirror_height = h = (
                            s ** 2 -
                            (b / 2) ** 2
                    ) ** 0.5

mirror = pv.Triangle([
    [b / 2, h, 0],
    [0, 0, 0],
    [b, 0, 0],
])
bevel = pv.PolyData(
    [
        [0, 0, 0],
        [b / 2, h, 0],
        [b, 0, 0],
        [b + bev * s/h, 0, 0],
        [b / 2, h + bev * s / (b / 2), 0],
        [bev * -s / h, 0, 0]
    ],
    faces=[4, 0, 1, 4, 5, 4, 1, 2, 3, 4]
)
base = pv.Triangle([
    [ # Top
        b/2,
        h + (bev + gap) * s/ (b/2),
        0
    ],
    [ # Left
        (bev + gap) * -s / h - gap * (b/2) / h,
        -gap,
        0
    ],
    [ # Right
        b + (bev + gap) * s/h + gap * (b/2) / h,
        -gap,
        0
    ],
])

base.points_bary = cart_to_bary(
    points_cart=base.points,
    vertices_cart=base.points
)
bevel.points_bary = cart_to_bary(
    points_cart=bevel.points,
    vertices_cart=base.points,
)
mirror.points_bary = cart_to_bary(
    points_cart=mirror.points,
    vertices_cart=base.points
)


if __name__ == '__main__':
    bevel.translate([0, 0, 1e-3], inplace=True)
    mirror.translate([0, 0, 2e-3], inplace=True)

    # bevel = bevel.extrude(vector=[0, 0, 1], capping=True)
    # mirror = mirror.extrude(vector=[0, 0, 1], capping=True)

    p = pv.Plotter()


    def draw(mesh, color=None):
        p.add_mesh(
            mesh,
            color=color,
            show_edges=True
        )


    draw(base, 'gray')
    draw(bevel, 'r')
    draw(mirror, 'w')

    p.show_grid()
    p.add_axes()
    p.camera_position = 'xy'
    p.show()
