import pyvista as pv

def make_plotter(title: str = None):
    p = pv.Plotter()
    p.add_axes()
    p.show_grid()
    if title is not None:
        p.title=title
    return p