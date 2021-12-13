import numpy as np
from utilities.vector import *


class Plane:
    def __init__(self,
                 origin_3=np.array([0, 0, 0]),
                 normal_3=np.array([0, 0, 1]),
                 x_hat_3=np.array([1, 0, 0]),
                 ):
        self.origin_3 = origin_3
        self.normal_3 = normalize(normal_3)
        self.x_hat_3 = normalize(project(x_hat_3, self.normal_3))
        self.y_hat_3 = np.cross(self.normal_3, self.x_hat_3)

    def to_3D(self, v_p):
        return self.origin_3 + v_p[:, [0]] * self.x_hat_3 + v_p[:, [1]] * self.y_hat_3

    def to_p(self, v_3):
        v_3 = v_3 - self.origin_3
        x = np.dot(v_3, self.x_hat_3)
        y = np.dot(v_3, self.y_hat_3)
        return np.stack((x, y))

    def intersection_with_line_3(self,
                                 line_origin_3=np.array([0, 0, 0]),
                                 line_direction_3=np.array([0, 0, 1])
                                 ):
        line_direction_3 = normalize(line_direction_3)
        d = np.dot(self.origin_3 - line_origin_3, self.normal_3) / np.dot(line_direction_3, self.normal_3)
        return line_origin_3 + line_direction_3 * d


def angle_axis_from_vectors(a, b):
    """
    Computes the angle and axis required to rotate vector a such that is is parallel to vector b.

    Angle is returned in radians. Axis is normalized.

    """
    norm_a = normalize(a)
    norm_b = normalize(b)
    cos = np.dot(a, b)

    angle = np.arccos(cos)
    axis = np.cross(a, b)

    return angle, axis
