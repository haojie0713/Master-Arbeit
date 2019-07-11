import math
import numpy as np
from config import *


def grid_points():
    """This function returns grid points indices with shape (-1, 2) for the FoldingNet decoder"""
    n_grid = np.sqrt(OUTPUT_POINT_SIZE)
    offset = boundingBoxSize/n_grid
    x = np.linspace(-boundingBoxSize+offset, boundingBoxSize-offset, n_grid)
    y = np.linspace(-boundingBoxSize+offset, boundingBoxSize-offset, n_grid)
    index = np.array(np.meshgrid(x, y))
    index = np.reshape(index, (2, -1)).T
    return index


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def sample(points):
    validIndicies = np.logical_and(np.logical_and(np.abs(points[:, 0]) < boundingBoxSize, np.abs(points[:, 1]) < boundingBoxSize), np.abs(points[:, 2]) < boundingBoxSize)
    points = points[validIndicies, :]

    if len(points) == 0:
        points = np.zeros((2048, 3), dtype=np.float32)
    while len(points) < INPUT_POINT_SIZE:
        points = np.repeat(points, 2, axis=0)

    randInidices = np.arange(len(points))
    np.random.shuffle(randInidices)
    points_sampled = points[randInidices[:INPUT_POINT_SIZE], :]
    return points_sampled
