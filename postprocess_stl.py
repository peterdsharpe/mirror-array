import pyvista as pv
from utilities.plotter import make_plotter
from design_mirror import (
    mirror_solids_for_printing,
    bevel_solids_for_printing,
    base_solids_for_printing,
    model_for_printing,
    N
)
import copy

# Merge all individual tiles
complete_tiles = []
for i in range(N):
    print(i)
    complete_tile = copy.deepcopy(base_solids_for_printing[i])
    complete_tile.boolean_union(
        mirror_solids_for_printing[i],
        tolerance=1e-3,
    )
    complete_tiles.append(complete_tile)

# mesh = pv.PolyData("to_print/print.stl")

sphere_a = pv.Sphere()
sphere_b = pv.Sphere(center=(1, 0, 0))

p = make_plotter("Postprocessing")
p.add_mesh(
    sphere_a.boolean_union(sphere_b),
    show_edges=True
)
# p.add_mesh(
#     mesh,
#     show_edges=True
# )
p.show()
