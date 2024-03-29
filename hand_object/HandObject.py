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

dir_image = '/home/haojie/Desktop/hand_object/images/'
dir_BBox = '/home/haojie/Desktop/hand_object/BoundingBox.txt'
dir_points = '/home/haojie/Desktop/hand_object/points/'
dir_rotMat = '/home/haojie/Desktop/hand_object/rotMat/'
dir_center3Drot = '/home/haojie/Desktop/hand_object/center3Drot/'


def rot_x(rot_angle):
    cosAngle = np.cos(rot_angle)
    sinAngle = np.sin(rot_angle)
    rotMat = np.asarray([[1.0, 0.0, 0.0],
                       [0.0, cosAngle, -sinAngle],
                       [0.0, sinAngle, cosAngle]], dtype=float)
    return rotMat


def rot_y(rot_angle):
    cosAngle = np.cos(rot_angle)
    sinAngle = np.sin(rot_angle)
    rotMat = np.asarray([[cosAngle, 0.0, sinAngle],
                       [0.0, 1.0, 0.0],
                       [-sinAngle, 0.0, cosAngle]], dtype=float)
    return rotMat


def rot_z(rot_angle):
    cosAngle = np.cos(rot_angle)
    sinAngle = np.sin(rot_angle)
    rotMat = np.asarray([[cosAngle, -sinAngle, 0.0],
                      [sinAngle, cosAngle, 0.0],
                      [0.0, 0.0, 1.0]], dtype=float)
    return rotMat


def viewCorrection(center3D):

    aroundYAngle = np.arctan((center3D[0])/center3D[2])
    center3DRotated = np.matmul(center3D, np.transpose(rot_y(-aroundYAngle)))
    aroundXAngle = np.arctan((center3DRotated[1])/center3DRotated[2])

    viewRotation = np.matmul(rot_x(aroundXAngle), rot_y(-aroundYAngle))
    # cloud = np.matmul(cloud, np.transpose(viewRotation))
    # joint = np.matmul(joint, np.transpose(viewRotation))
    return np.transpose(viewRotation)


counter = 0
with open(dir_BBox) as f:
    for line in f.readlines():
        line = line.split()
        name = line[0]
        pose = np.array(line[1:], dtype=np.float).reshape(2, 2)
        img = np.array(Image.open(dir_image+name))

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.imshow(img, cmap='gray')
        # tl_u = pose[0, 0]
        # tl_v = pose[0, 1]
        # br_u = pose[1, 0]
        # br_v = pose[1, 1]
        # ax.plot([tl_u, tl_u], [tl_v, br_v])
        # ax.plot([tl_u, br_u], [tl_v, tl_v])
        # plt.show()

        tl_u = np.floor(pose[0, 0])
        tl_v = np.floor(pose[0, 1])
        br_u = np.floor(pose[1, 0])
        br_v = np.floor(pose[1, 1])
        img_cropped = img[int(tl_v):int(br_v), int(tl_u):int(br_u)]

        # get 3D center (center of bounding box + centerDepth --> transformed to 3D camera cooordinates)
        img_cropped[np.where(img_cropped == 0)] = np.max(img_cropped) + 1
        depths = img_cropped.reshape(-1, 1)
        label = KMeans(n_clusters=2, init=np.reshape([np.max(img_cropped) + 1, np.min(img_cropped)], [-1, 1]), n_init=1).fit_predict(depths)
        centerDepth = np.mean(depths[label.astype(bool)])

        center3D = np.zeros(3)
        center3D[0] = (0.5 * (tl_u + br_u) + 0.5 - u0) * centerDepth / f_x
        center3D[1] = (0.5 * (tl_v + br_v) + 0.5 - v0) * centerDepth / f_y
        center3D[2] = centerDepth

        cropStart = np.zeros(2)
        cropStart[0] = (center3D[0] - cropSizePlus * 1.41) * f_x / center3D[2] + u0
        cropStart[1] = (center3D[1] - cropSizePlus * 1.41) * f_y / center3D[2] + v0

        cropEnd = np.zeros(2)
        cropEnd[0] = (center3D[0] + cropSizePlus * 1.41) * f_x / center3D[2] + u0
        cropEnd[1] = (center3D[1] + cropSizePlus * 1.41) * f_y / center3D[2] + v0

        # view correction
        rotMat = viewCorrection(center3D)
        center3Drot = np.matmul(center3D.reshape(1, -1), rotMat)

        # padding and cropping image
        padSize = 900
        img_pad = np.pad(img, ((padSize, padSize), (padSize, padSize)), 'constant')
        us = int(cropStart[0] + padSize)
        ue = int(cropEnd[0] + padSize)
        vs = int(cropStart[1] + padSize)
        ve = int(cropEnd[1] + padSize)
        image = img_pad[vs:ve, us:ue].astype(np.float)
        image[np.where(image==0)] = np.inf

        # project to 3D camera coordinates
        a, b = np.meshgrid(np.arange(np.floor(cropStart[0]), np.floor(cropEnd[0])), np.arange(np.floor(cropStart[1]), np.floor(cropEnd[1])))
        u = a.ravel(1)
        v = b.ravel(1)
        d = image.ravel(1)
        points = np.empty((3, image.size))
        points[0, :] = (u + 0.5 - u0) * d / f_x
        points[1, :] = (v + 0.5 - v0) * d / f_y
        points[2, :] = d
        points = points.T

        # keep points in bounding box
        points = points[points[:, 2] < center3D[2] + cropDepthPlus * 1.4]
        points = points[points[:, 2] > center3D[2] - cropDepthPlus * 1.4]

        points = np.matmul(points, rotMat) - center3Drot
        validIndicies = np.logical_and(np.logical_and(np.abs(points[:, 0]) < 140, np.abs(points[:, 1]) < 140), np.abs(points[:, 2]) < 140)
        points = points[validIndicies, :]

        # sample points
        if len(points) < 5000:
            continue
        while len(points) < 6000:
            points = np.repeat(points, 2, axis=0)
        randInidices = np.arange(len(points))
        np.random.shuffle(randInidices)
        final_points = points[randInidices[:2000], :]  # mm

        fig2 = plt.figure()
        pose = np.load('/home/haojie/Desktop/hand_object/pose.npy')
        joints = pose[counter].reshape(-1, 3)
        ax2 = fig2.add_subplot(111)
        ax2 = Axes3D(fig2)
        ax2.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
        ax2.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
        ax2.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
        ax2.view_init(-90, -90)
        ax2.scatter(final_points[:, 0], final_points[:, 1], final_points[:, 2], c='b', marker='o', s=1, alpha=1)

        for i in range(6):
            k = np.mod(i + 1, 6)
            ax2.plot([joints[i, 0], joints[k, 0]], [joints[i, 1], joints[k, 1]], [joints[i, 2], joints[k, 2]], color='y',
                    marker='*')
        for j in xrange(1, 6):
            q = 3 * (j + 1)
            ax2.plot([joints[j, 0], joints[q, 0]], [joints[j, 1], joints[q, 1]], [joints[j, 2], joints[q, 2]], color='y',
                    marker='*')
            for m in range(2):
                ax2.plot([joints[q + m, 0], joints[q + m + 1, 0]], [joints[q + m, 1], joints[q + m + 1, 1]],
                        [joints[q + m, 2], joints[q + m + 1, 2]], color='y', marker='*')

        plt.show()
        counter += 1
        # np.save(dir_rotMat + name[:-4] + '.npy', rotMat)
        # np.save(dir_center3Drot + name[:-4] + '.npy', center3Drot)
        # np.save(dir_points + name[:-4] + '.npy', final_points)

