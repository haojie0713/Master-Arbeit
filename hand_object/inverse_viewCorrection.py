import numpy as np
import matplotlib.pyplot as plt
import os
from six.moves import xrange
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import time
from sklearn.cluster import KMeans


u0 = 315.944855
v0 = 245.287079
f_x = 475.065948
f_y = 475.065857

cropSizePlus = 185.625  # 135*1.375
cropDepthPlus = 235  # 135+100

dir_pose = '/home/haojie/Desktop/hand_object/pose.npy'
dir_rotMat = '/home/haojie/Desktop/hand_object/rotMat/'
dir_center3Drot = '/home/haojie/Desktop/hand_object/center3Drot/'

f = open('/home/haojie/Desktop/hand_object/pose_out.txt', 'w')
pose = np.load(dir_pose)  # mm
for i in range(1, 2966):
    joints = pose[i-1].reshape(21, 3)
    name = 'image_D{:08d}.npy'.format(i)
    rotMat = np.load(dir_rotMat + name)
    center3Drot = np.load(dir_center3Drot + name)
    rotMat_inverse = np.linalg.inv(rotMat)
    joints = np.matmul(joints + center3Drot, rotMat_inverse)
    joints = joints.ravel()

    print_name = 'frame\images'+'\image_D{:08d}.png\t  '.format(i)
    f.write(print_name)
    for x in joints:
        f.write('{:.4f}\t  '.format(x))
    f.write('\n')
f.close()
