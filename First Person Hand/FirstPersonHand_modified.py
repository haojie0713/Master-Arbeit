import numpy as np
import matplotlib.pyplot as plt
import os
from six.moves import xrange
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import time
from sklearn.cluster import KMeans

boundingBoxSize = 140

dir_points = '/home/haojie/Desktop/FirstPersonHand/points/'
dir_joints = '/home/haojie/Desktop/FirstPersonHand/joints/'
dir_video = '/home/haojie/Desktop/FirstPersonHand/video/'
dir_pose = '/home/haojie/Desktop/FirstPersonHand/pose/'

cam_extr = np.eye(4)
cam_intr = np.array([[475.065948, 0, 315.944855],
                     [0, 475.065857, 245.287079],
                     [0, 0, 1]])
u0 = 315.944855
v0 = 245.287079
f_x = 475.065948
f_y = 475.065857

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


list_subject = os.listdir(dir_video)


counter = 0
for subject in list_subject:
    dir_video_subject = dir_video+subject+'/'
    dir_pose_subject = dir_pose+subject+'/'
    list_action = os.listdir(dir_video_subject)
    for action in list_action:
        dir_video_action = dir_video_subject+action+'/'
        dir_pose_action = dir_pose_subject+action+'/'
        list_seq = os.listdir(dir_video_action)
        for seq in list_seq:
            dir_video_seq = dir_video_action+seq+'/depth/'
            dir_pose_seq = dir_pose_action+seq+'/skeleton.txt'
            skeleton = np.loadtxt(dir_pose_seq)
            skeleton = skeleton[:, 1:].reshape(skeleton.shape[0], 21, -1)
            num_frames = len(os.listdir(dir_video_seq))
            for i in range(num_frames):
                frame = dir_video_seq + 'depth_{:04d}.png'.format(i)
                img = np.array(Image.open(frame))

                skel_hom = np.concatenate([skeleton[i], np.ones([skeleton[i].shape[0], 1])], 1)
                skel_camcoords = cam_extr.dot(skel_hom.transpose()).transpose()[:, :3].astype(np.float32)  # camera coordinates(N,3)

                skel_hom2d = np.array(cam_intr).dot(skel_camcoords.transpose()).transpose()
                skel_proj = (skel_hom2d / skel_hom2d[:, 2:])[:, :2]  # pixel coordinates

                # discard sample that has more than 2 joints out of the 2D image
                invalidIndices = np.logical_or(np.logical_or(np.logical_or(skel_proj[:, 0] > 639, skel_proj[:, 1] > 479), skel_proj[:, 0] < 0), skel_proj[:, 1] < 0)
                if np.sum(invalidIndices) > 2:
                    continue

                # cropping
                tl_u = np.maximum(np.floor(np.min(skel_proj[:, 0])), 0)
                tl_v = np.maximum(np.floor(np.min(skel_proj[:, 1])), 0)
                br_u = np.minimum(np.floor(np.max(skel_proj[:, 0])), 639)
                br_v = np.minimum(np.floor(np.max(skel_proj[:, 1])), 479)
                img_cropped = img[int(tl_v):int(br_v), int(tl_u):int(br_u)]

                # get 3D center (center of bounding box + centerDepth --> transformed to 3D camera cooordinates)
                img_cropped[np.where(img == 0)] = np.max(img_cropped)+1
                depths = img_cropped.reshape(-1, 1)
                label = KMeans(n_clusters=2, init=np.reshape([np.max(img_cropped)+1, np.min(img_cropped)], [-1,1]), n_init=100).fit_predict(depths)
                centerDepth = np.mean(depths[label.astype(bool)])
                if np.abs(centerDepth-skel_camcoords[3, 2]) > 100:
                    centerDepth = skel_camcoords[3, 2]

                center3D = np.zeros(3)
                center3D[0] = (0.5 * (tl_u + br_u) + 0.5 - u0) * centerDepth / f_x
                center3D[1] = (0.5 * (tl_v + br_v) + 0.5 - v0) * centerDepth / f_y
                center3D[2] = centerDepth

                cropStart = np.zeros(2)
                cropStart[0] = (center3D[0] - cropSizePlus*1.41) * f_x / center3D[2] + u0
                cropStart[1] = (center3D[1] - cropSizePlus*1.41) * f_y / center3D[2] + v0

                cropEnd = np.zeros(2)
                cropEnd[0] = (center3D[0] + cropSizePlus*1.41) * f_x / center3D[2] + u0
                cropEnd[1] = (center3D[1] + cropSizePlus*1.41) * f_y / center3D[2] + v0

                # view correction
                rotMat = viewCorrection(center3D)
                # cloud = np.matmul(cloud, np.transpose(viewRotation))
                center3Drot = np.matmul(center3D.reshape(1, -1), rotMat)
                joints = np.matmul(skel_camcoords, rotMat) - center3Drot

                # padding and cropping image
                padSize = 500
                img_pad = np.pad(img, ((padSize, padSize),(padSize, padSize)), 'constant')
                us = int(cropStart[0]+padSize)
                ue = int(cropEnd[0]+padSize)
                vs = int(cropStart[1]+padSize)
                ve = int(cropEnd[1]+padSize)
                image = img_pad[vs:ve, us:ue]

                # project to 3D camera coordinates
                a, b = np.meshgrid(np.arange(us, ue), np.arange(vs, ve))
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

                # sample points
                if len(points) < 5000:
                    continue
                while len(points) < 6000:
                    points = np.repeat(points, 2, axis=0)
                randInidices = np.arange(len(points))
                np.random.shuffle(randInidices)
                final_points = points[randInidices[:6000], :]
                final_points = final_points  # mm

                final_joints = joints * 0.001  # mm->m
                np.save(dir_points + str(counter) + '.npy', final_points)
                np.save(dir_joints + str(counter) + '.npy', final_joints)
                counter += 1