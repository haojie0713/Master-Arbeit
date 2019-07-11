#!/usr/bin/python3

"""ShapeNet Offline Processing Step2:

        This script extracts the normalization information of each .pcd file and save each object to .npy file"""

import numpy as np
import json
import os


class Point():
    def __init__(self, x, y, z):
        self.x = np.array(x, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)
        self.z = np.array(z, dtype=np.float32)

    def toarray(self):
        return np.array([self.x, self.y, self.z])


dir = '/home/haojie/Desktop/shapenet_offline_processing/'
list_object_offsets = os.listdir(dir)
for offset in list_object_offsets:
    dir_offset_folder = dir+offset+'/'
    list_objects = os.listdir(dir_offset_folder)
    for object in list_objects:
        with open(dir_offset_folder+object) as f1:
            lines = f1.readlines()
            n_lines = len(lines) - 11
            points = np.empty((n_lines, 3), dtype=np.float32)
            for i, line in enumerate(lines[11:len(lines)]):
                strs = line.split(' ')
                points[i, :] = Point(strs[0], strs[1], strs[2].strip()).toarray()
        os.remove(dir_offset_folder + object)
        if points.shape[0] < 6000:
            continue
        randIndices = np.arange(points.shape[0])
        np.random.shuffle(randIndices)
        points_sampled = points[randIndices[:6000]]
        np.save(dir_offset_folder+object.strip('.pcd')+'.npy', points_sampled)
        centroid = np.mean(points, axis=0)
        euc_dis = np.linalg.norm(points-centroid, axis=1)
        max_euc_dis = np.max(euc_dis)
        dic = {'centroid': centroid.tolist(), 'max_dis': max_euc_dis.tolist()}
        with open(dir_offset_folder+object.strip('.pcd')+'.json', 'w') as f2:
            json.dump(dic, f2)




