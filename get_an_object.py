import os
import numpy as np
import json
from random import choice
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


def get_an_object():
    dir = '/home/haojie/Desktop/shapenet_offline_processing/'
    list_object_offsets = os.listdir(dir)
    offset = choice(list_object_offsets)
    dir_offset = dir + offset + '/'
    list_objects = os.listdir(dir_offset)
    filename = dir_offset+choice(list_objects)
    npy_filename = re.sub('.json', '.npy', filename)
    points = np.load(npy_filename)
    with open(re.sub('.npy', '.json', npy_filename)) as f:
        dict = json.load(f)
        center = np.array(dict["centroid"], dtype=np.float32)
        max_dis = np.array(dict["max_dis"], dtype=np.float32)
    return points, center, max_dis


def get_an_object_extra():
    file = '/home/haojie/Desktop/shapenet_offline_processing/02773838/39.json'
    npy_filename = re.sub('.json', '.npy', file)
    points = np.load(npy_filename)
    with open(re.sub('.npy', '.json', npy_filename)) as f:
        dict = json.load(f)
        center = np.array(dict["centroid"], dtype=np.float32)
        max_dis = np.array(dict["max_dis"], dtype=np.float32)
    return points, center, max_dis


def get_an_object_extr():
    dir = '/home/haojie/Desktop/shapenet_offline_processing/'
    list_object_offsets = os.listdir(dir)
    # offset = choice(list_object_offsets)
    # print(offset)
    for i in list_object_offsets:
        offset = i
        print(i)
        offset = '02773838'
        dir_offset = dir + offset + '/'
        list_objects = os.listdir(dir_offset)
        ss = choice(list_objects)
        print(ss)
        ss = '39.json'
        filename = dir_offset+ss
        npy_filename = re.sub('.json', '.npy', filename)
        points = np.load(npy_filename)
        with open(re.sub('.npy', '.json', npy_filename)) as f:
            dict = json.load(f)
            center = np.array(dict["centroid"], dtype=np.float32)
        #     max_dis = np.array(dict["max_dis"], dtype=np.float32)
        points -= center
        points[:, 2] = 0
        theta = np.arctan(0.13)
        R_x = np.array(
            [[1, 0, 0], [0, math.cos(theta), -math.sin(theta)], [0, math.sin(theta), math.cos(theta)]])
        points = np.dot(points, R_x.T)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
        ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
        ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
        randInidices = np.arange(len(points))
        np.random.shuffle(randInidices)
        points = points[randInidices[:1000], :]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='o', s=15, alpha=1)
        plt.show()

# #
# get_an_object_extr()