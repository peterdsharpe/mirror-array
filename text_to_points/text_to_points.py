import os, sys
from pathlib import Path
import svgpathtools
import svgpathtools as spt
from xml.dom import minidom
import numpy as np

font_to_use = 'multicolore'

alphabet_folder = Path("./alphabet_svgs")

with open(alphabet_folder / "alphabet_characters.txt", "r") as f:
    alphabet_characters = f.read()

font_file_path = alphabet_folder / (font_to_use + ".svg")
with open(font_file_path, "r") as f:
    svg_paths = minidom.parse(f).getElementsByTagName('path')

letter_paths = {
    char: spt.parse_path(path.attributes['d'].value)
    for char, path in zip(alphabet_characters, svg_paths)
}

position_text = [s.attributes['transform'].value for s in svg_paths]
positions = np.array([
    float(s[s.index('(') + 1:s.index(',')])
    for s in position_text
])
spacings = np.diff(positions)
spacings = np.append(spacings, spacings[-1])

letter_spacings = {
    char: spacing
    for char, spacing in zip(
        alphabet_characters,
        spacings
    )
}
letter_spacings[" "] = 100

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p


def get_points_from_letter(l: str = "A", n_points=50):
    path = letter_paths[l]
    lengths_nondim = np.linspace(0, 1, n_points) * path.length()
    t = np.array([path.ilength(l) for l in lengths_nondim])
    points_complex = np.array([path.point(ti) for ti in t])
    # Scale
    points_complex += 1
    points_complex /= 187.5
    # TODO

    x = np.real(points_complex)
    y = np.imag(points_complex) * -1

    points = np.stack([x, y], axis=1)
    return points


fig, ax = plt.subplots()
l = "A"
n_points = 21
c = get_points_from_letter(l, n_points)
plt.plot(
    c[:, 0],
    c[:, 1],
    ".",
    markersize=10,
)
p.equal()
p.show_plot()


def get_points_from_string(s: str = "Testing", n_points=200):
    letter_path_lengths = np.array([
        letter_paths[l].length() if l != " " else 0
        for l in s
    ])
    total_path_lengths = np.sum(letter_path_lengths)

    # Compute how many points each letter gets
    letter_n_points_ideal = n_points * letter_path_lengths / total_path_lengths
    letter_n_points = np.floor(letter_n_points_ideal).astype(int)
    for i in range(n_points - np.sum(letter_n_points)):
        letter_n_points[np.argmax(letter_n_points_ideal - letter_n_points)] += 1

    # Pick points for each letter
    points = []

    x_origin = 0

    for i, l in enumerate(s):

        if not l == " ":
            p = get_points_from_letter(l, n_points=letter_n_points[i])
            p[:, 0] += x_origin
            points.append(p)

        x_origin += letter_spacings[l] / 187.5

    points = np.concatenate(points, axis=0)

    return points

# fig, ax = plt.subplots()
# s= "Sei la mia luce"
# n_points = 200
# c = get_points_from_string(s, n_points)
# plt.plot(
#     c[:, 0],
#     c[:, 1],
#     ".",
#     markersize=3,
# )
# p.equal()
# p.show_plot()
