import tensorflow as tf
import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


pose = np.load('/home/haojie/Desktop/EgoDexter/data/Rotunda/pose.npy')
dir_joints = '/home/haojie/Desktop/EgoDexter/data/Rotunda/joints/'
dir_points = '/home/haojie/Desktop/EgoDexter/data/Rotunda/points/'
dir_center3d = '/home/haojie/Desktop/EgoDexter/data/Rotunda/center3d/'
dir_rotmat = '/home/haojie/Desktop/EgoDexter/data/Rotunda/rotmat/'

label = np.load('/home/haojie/Desktop/EgoDexter/data/Rotunda/label.npy')
indices = np.where(np.sum(label, axis=1)!=0)[0]

error = 0
counter = 0
for i in indices:
    gt_pose = np.load(dir_joints+str(i)+'.npy')

    label_this = label[i].reshape(-1, 3)
    pose_this = (pose[counter].reshape(-1,3)[[8,11,14,17,20]])*label_this
    pose_this = pose_this[np.where(np.sum(pose_this, axis=1)!=0)]

    n_joints = pose_this.shape[0]
    error_this = np.sum(np.sqrt(np.sum((pose_this-gt_pose)**2, axis=1)))/n_joints
    error += error_this

    ##########visualization########################################################
    points = np.load(dir_points + str(i) + '.npy')
    randInidices = np.arange(len(points))
    np.random.shuffle(randInidices)
    points = points[randInidices[:1000], :]
    joints = pose[counter].reshape(-1, 3)
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    ax2 = Axes3D(fig)
    ax2.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    ax2.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax2.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax2.view_init(-90, -90)
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', s=1, alpha=1)

    for i in range(6):
        k = np.mod(i + 1, 6)
        ax2.plot([joints[i, 0], joints[k, 0]], [joints[i, 1], joints[k, 1]], [joints[i, 2], joints[k, 2]], color='y', marker='*')
    for j in range(1, 6):
        q = 3 * (j + 1)
        ax2.plot([joints[j, 0], joints[q, 0]], [joints[j, 1], joints[q, 1]], [joints[j, 2], joints[q, 2]], color='y', marker='*')
        for m in range(2):
            ax2.plot([joints[q + m, 0], joints[q + m + 1, 0]], [joints[q + m, 1], joints[q + m + 1, 1]], [joints[q + m, 2], joints[q + m + 1, 2]], color='y', marker='*')
    plt.show()
    ##############################################################################
    counter += 1

print(error/251)




