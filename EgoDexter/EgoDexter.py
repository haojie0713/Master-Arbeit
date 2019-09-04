import numpy as np
import os
import glob
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dir_desk = '/home/haojie/Desktop/EgoDexter/data/Desk/depth/'
dir_desk_pose = '/home/haojie/Desktop/EgoDexter/data/Desk/annotation.txt_3D.txt'
write_desk_pose = '/home/haojie/Desktop/EgoDexter/data/Desk/pose.npy'
write_desk_label = '/home/haojie/Desktop/EgoDexter/data/Desk/label.npy'

dir_fruits = '/home/haojie/Desktop/EgoDexter/data/Fruits/depth/'
dir_fruits_pose = '/home/haojie/Desktop/EgoDexter/data/Fruits/annotation.txt_3D.txt'
write_fruits_pose = '/home/haojie/Desktop/EgoDexter/data/Fruits/pose.npy'
write_fruits_label = '/home/haojie/Desktop/EgoDexter/data/Fruits/label.npy'

dir_kitchen = '/home/haojie/Desktop/EgoDexter/data/Kitchen/depth/'
dir_kitchen_pose = '/home/haojie/Desktop/EgoDexter/data/Kitchen/annotation.txt_3D.txt'
write_kitchen_pose = '/home/haojie/Desktop/EgoDexter/data/Kitchen/pose.npy'
write_kitchen_label = '/home/haojie/Desktop/EgoDexter/data/Kitchen/label.npy'

dir_rotunda = '/home/haojie/Desktop/EgoDexter/data/Rotunda/depth/'
dir_rotunda_pose = '/home/haojie/Desktop/EgoDexter/data/Rotunda/annotation.txt_3D.txt'
write_rotunda_pose = '/home/haojie/Desktop/EgoDexter/data/Rotunda/pose.npy'
write_rotunda_label = '/home/haojie/Desktop/EgoDexter/data/Rotunda/label.npy'

cam_extr = np.eye(4)
cam_intr = np.array([[475.62, 0, 311.125],
                     [0, 475.62, 245.965],
                     [0, 0, 1]])
u0 = 311.125
v0 = 245.965
f_x = 475.62
f_y = 475.62

reorder = np.array([0, 1, 5, 9, 13, 17, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20])

cropSizePlus = 185.625  # 135*1.375
cropDepthPlus = 235  # 135+100

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


with open(dir_rotunda_pose) as f:
    lines = f.readlines()
    all_pose = np.zeros([len(lines), 15])
    k = 0
    for line in lines:
        line = line.split()
        pose = np.zeros(15)
        for i in range(15):
            pose[i] = line[i][:-1]

        all_pose[k] = pose
        k += 1
label = all_pose.astype(bool)*1.0
# np.save(write_rotunda_label, label)

for i in range(len(all_pose)):
    gt_pose = all_pose[i].reshape(-1, 3)
    if np.any(gt_pose):
        frame = dir_rotunda+'image_{:05d}_depth.png'.format(i)

        img = np.array(Image.open(frame))
        skeleton = gt_pose[np.where(np.sum(gt_pose, axis=1)!=0)]
        skel_hom = np.concatenate([skeleton, np.ones([skeleton.shape[0], 1])], 1)
        skel_camcoords = cam_extr.dot(skel_hom.transpose()).transpose()[:, :3].astype(np.float32)  # camera coordinates(N,3)
        skel_hom2d = np.array(cam_intr).dot(skel_camcoords.transpose()).transpose()
        skel_proj = (skel_hom2d / skel_hom2d[:, 2:])[:, :2]  # pixel coordinates

        tl_u = np.maximum(np.floor(np.min(skel_proj[:, 0])-50), 0)
        tl_v = np.maximum(np.floor(np.min(skel_proj[:, 1])-50), 0)
        br_u = np.minimum(np.floor(np.max(skel_proj[:, 0])+50), 639)
        br_v = np.minimum(np.floor(np.max(skel_proj[:, 1])+50), 479)
        img_cropped = img[int(tl_v):int(br_v), int(tl_u):int(br_u)]

        img_cropped[np.where(img_cropped == 0)] = np.max(img_cropped) + 1
        depths = img_cropped.reshape(-1, 1)
        label = KMeans(n_clusters=2, init=np.reshape([np.max(img_cropped) + 1, np.min(img_cropped)], [-1, 1]),
                       n_init=1).fit_predict(depths)
        centerDepth = np.median(depths[label.astype(bool)])

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
        joints = np.matmul(skel_camcoords, rotMat) - center3Drot

        padSize = 500
        img_pad = np.pad(img, ((padSize, padSize), (padSize, padSize)), 'constant')
        us = int(np.floor(cropStart[0]) + padSize)
        ue = int(np.floor(cropEnd[0]) + padSize)
        vs = int(np.floor(cropStart[1]) + padSize)
        ve = int(np.floor(cropEnd[1]) + padSize)

        image = img_pad[vs:ve, us:ue].astype(np.float)
        image[np.where(image == 0)] = np.inf

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

        validIndicies = np.logical_and(
            np.logical_and(np.abs(points[:, 0]) < 140, np.abs(points[:, 1]) < 140),
            np.abs(points[:, 2]) < 140)
        points = points[validIndicies, :]

        # if len(points) < 5000:
        #     print(i)
        #     print(len(points))
        #     continue
        while len(points) < 6000:
            points = np.repeat(points, 2, axis=0)
        randInidices = np.arange(len(points))
        np.random.shuffle(randInidices)
        final_points = points[randInidices[:1000], :] * np.array([-1, 1, 1])  # transform to right hand (mm)

        final_joints = joints * 0.001 * np.array([-1, 1, 1])  # transform to right hand (mm->m)

        # np.save(dir_rotunda[:-6]+'points/'+str(i)+'.npy', final_points)
        # np.save(dir_rotunda[:-6] + 'joints/' + str(i) + '.npy', final_joints)
        # np.save(dir_rotunda[:-6] + 'rotmat/' + str(i) + '.npy', rotMat)
        # np.save(dir_rotunda[:-6] + 'center3d/' + str(i) + '.npy', center3Drot)


        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
        ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
        ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})

        ax.view_init(-90, -90)

        ax.scatter(final_points[:, 0], final_points[:, 1], final_points[:, 2])
        plt.show()






