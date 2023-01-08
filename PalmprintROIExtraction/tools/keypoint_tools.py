from math import cos, sin
import numpy as np


def rotate_point(point1, point2, theta):
    xx = (point1[0] - point2[0]) * cos(-theta) - (point1[1] - point2[1]) * sin(-theta) + point2[0]
    yy = (point1[0] - point2[0]) * sin(-theta) + (point1[1] - point2[1]) * cos(-theta) + point2[1]
    return xx, yy


def rotate_points(points, center, theta):
    rotated_points = np.zeros_like(points)
    rotated_points[:, 0] = (points[:, 0] - center[0]) * cos(-theta) - (points[:, 1] - center[1]) * sin(-theta) + center[0]
    rotated_points[:, 1] = (points[:, 1] - center[1]) * sin(-theta) - (points[:, 1] - center[1]) * cos(-theta) + center[1]
    return rotated_points
