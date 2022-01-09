from utilities.coordinate_math import Plane
import aerosandbox.numpy as np
import pyvista as pv
rt3 = np.sqrt(3)

def mesh_hexagon(
        n_rings: int = 2,
        plane: Plane = None,
        source_location = np.array([np.Inf, 0, 0])
):
    if plane is None:
        plane = Plane()

    mirror_faces = []

    for ring in np.arange(n_rings) + 1:
        inside_radius = (ring - 1) * rt3 / 2
        for side in np.arange(6):
            for stride in np.arange(1 - ring, ring)[::-1]:
                tri = pv.Triangle([
                    [0, 0, 0],
                    [0.5, rt3 / 2, 0],
                    [-0.5, rt3 / 2, 0],
                ])

                if (ring + stride) % 2 == 0:
                    tri.flip_y()

                tri.translate([0, inside_radius, 0])

                tri.translate([stride / 2, 0, 0])

                tri.rotate_z(60 * side)

                tri.points = plane.to_3D(tri.points[:, :2])

                if np.dot(tri.face_normals[0, :], source_location - tri.center_of_mass()) < 0:
                    tri.flip_normals()

                tri.ring = ring
                tri.side = side
                tri.stride = stride

                mirror_faces.append(tri)

    assert len(mirror_faces) == 6 * n_rings ** 2

    return mirror_faces

if __name__ == '__main__':
    mf = mesh_hexagon(
        n_rings=7,
    )
    plotter = pv.Plotter()
    for m in mf:
        plotter.add_mesh(m, show_edges=True)
    plotter.add_axes()
    plotter.show_grid()
    plotter.camera_position='xy'
    plotter.show()
