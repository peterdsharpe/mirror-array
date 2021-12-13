import numpy as np
from utilities.vector import normalize

def compute_orientations(
        source_locations,
        mirror_locations,
        target_locations,
):
    mirror_to_source = normalize(source_locations - mirror_locations)
    mirror_to_target = normalize(target_locations - mirror_locations)

    mirror_normals = normalize(mirror_to_source + mirror_to_target)

    return mirror_normals
