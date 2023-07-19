from pathlib import Path
import svgpathtools as spt
from xml.dom import minidom
import numpy as np

# font_to_use = 'script'
font_to_use = 'multicolore'

alphabet_folder = Path(__file__).parent / "alphabet_svgs"

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
representative_bbox = letter_paths["O"].bbox()
letter_height = representative_bbox[3] - representative_bbox[2]
letter_spacings[" "] = letter_spacings["l"]

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p


def get_points_from_string(s: str = "Testing", n_points=200, line_spacing=1.5, kerning: float = 1):
    # Clean up string
    s = "\n".join([line.strip() for line in s.split("\n")])  # strip spaces off each line
    admissible_characters = alphabet_characters + "\n" + " "
    for l in s:
        if l not in admissible_characters:
            s = s.replace(l, "")
            import warnings
            warnings.warn(f"Removed inadmissible character '{l}'.", stacklevel=2)

    # Calculate line lengths
    def get_line_length(s_line):
        assert "\n" not in s_line
        length = 0
        for l in s_line:
            length += letter_spacings[l] * kerning
        return length

    # Compute all subpaths of letters, and where they should be
    subpaths = []
    sp_offsets = []

    for lineno, line in enumerate(s.split("\n")):
        line_length = get_line_length(line)
        line_origin = -line_length / 2 + line_spacing * letter_height * lineno * 1j
        current_offset_on_line = 0
        for l in line:
            if not l == " ":
                sp = letter_paths[l].continuous_subpaths()
                subpaths.extend(sp)
                sp_offsets.extend(len(sp) * [line_origin + current_offset_on_line])

            current_offset_on_line += letter_spacings[l] * kerning

    # Compute how many points each subpath gets
    sp_lengths = np.array([
        sp.length() for sp in subpaths
    ])
    total_length = np.sum(sp_lengths)
    sp_n_points_ideal = n_points * sp_lengths / total_length
    sp_n_points = np.floor(sp_n_points_ideal).astype(int)
    for i in range(n_points - np.sum(sp_n_points)):
        sp_n_points[np.argmax(sp_n_points_ideal - sp_n_points)] += 1

    # Pick points for each subpath
    points = []

    for sp, length, offset, sp_n in zip(subpaths, sp_lengths, sp_offsets, sp_n_points):

        lengths_nondim = np.linspace(0, 1, sp_n, endpoint=not sp.isclosed()) * length
        t = np.array([sp.ilength(l) for l in lengths_nondim])
        points_complex = np.array([sp.point(ti) for ti in t])
        # Scale
        points_complex += offset
        points_complex /= letter_height

        x = np.real(points_complex)
        y = np.imag(points_complex) * -1

        sp_points = np.stack([x, y], axis=1)
        points.append(sp_points)

    points = np.concatenate(points, axis=0)
    xlim = (points[:, 0].min(), points[:, 0].max())
    ylim = (points[:, 1].min(), points[:, 1].max())
    points -= np.array([
        (xlim[0] + xlim[1]) / 2,
        (ylim[0] + ylim[1]) / 2,
    ])

    return points


if __name__ == '__main__':
    fig, ax = plt.subplots()
    s = "M"
    n_points = 96
    c = get_points_from_string(s, n_points, kerning=1.5)
    plt.plot(
        c[:, 0],
        c[:, 1],
        ".",
        markersize=10,
        alpha=0.5
    )
    p.equal()
    p.show_plot()
