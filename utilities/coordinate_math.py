import numpy as np
from vector import *


class Plane:
    def __init__(self,
                 origin=np.array([0, 0, 0]),
                 normal=np.array([0, 0, 1]),
                 x_hat_p=np.array([1, 0, 0]),
                 ):
        self.origin_3 = origin
        self.normal_3 = normalize(normal)
        self.x_hat_p = normalize(project(x_hat_p, self.normal_3))
        self.y_hat_p = np.cross(self.normal_3, self.x_hat_p)

    def to_3D(self, v_p):
        return self.origin_3 + v_p[0] * self.x_hat_p + v_p[1] * self.y_hat_p

    def to_p(self, v_3):
        v_3 = v_3 - self.origin_3
        x = np.dot(v_3, self.x_hat_p)
        y = np.dot(v_3, self.y_hat_p)
        return np.stack((x, y))
