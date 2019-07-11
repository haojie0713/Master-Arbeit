import os
import numpy as np
import json
from random import choice
import re


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
