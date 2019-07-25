import numpy as np
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dir_f_noobject = '/home/haojie/Desktop/SynthHand/SynthHands_Release/female_noobject/'  # 46646
dir_f_object = '/home/haojie/Desktop/SynthHand/SynthHands_Release/female_object/'  # 46830
dir_m_noobject = '/home/haojie/Desktop/SynthHand/SynthHands_Release/male_noobject/'  # 46830
dir_m_object = '/home/haojie/Desktop/SynthHand/SynthHands_Release/male_object/'  # 45328
dir = [dir_f_noobject, dir_f_object, dir_m_noobject, dir_m_object]

dir_f_noobject_points = '/home/haojie/Desktop/SynthHand/SynthHands_Release/processed_result/female_noobject/points/'
dir_f_object_points = '/home/haojie/Desktop/SynthHand/SynthHands_Release/processed_result/female_object/points/'
dir_m_noobject_points = '/home/haojie/Desktop/SynthHand/SynthHands_Release/processed_result/male_noobject/points/'
dir_m_object_points = '/home/haojie/Desktop/SynthHand/SynthHands_Release/processed_result/male_object/points/'
dir_points = [dir_f_noobject_points, dir_f_object_points, dir_m_noobject_points, dir_m_object_points]

dir_f_noobject_joints = '/home/haojie/Desktop/SynthHand/SynthHands_Release/processed_result/female_noobject/joints/'
dir_f_object_joints = '/home/haojie/Desktop/SynthHand/SynthHands_Release/processed_result/female_object/joints/'
dir_m_noobject_joints = '/home/haojie/Desktop/SynthHand/SynthHands_Release/processed_result/male_noobject/joints/'
dir_m_object_joints = '/home/haojie/Desktop/SynthHand/SynthHands_Release/processed_result/male_object/joints/'
dir_joints = [dir_f_noobject_joints, dir_f_object_joints, dir_m_noobject_joints, dir_m_object_joints]

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


for i in range(4):
    counter = 0
    path = dir[i]
    list_seq = os.listdir(path)
    for seq in list_seq:
        dir_cam = path + seq + '/'
        list_cam = os.listdir(dir_cam)
        for cam in list_cam:
            dir_number = dir_cam + cam + '/'
            list_number = os.listdir(dir_number)
            for number in list_number:
                dir_frame = dir_number + number + '/'
                list_frame = glob.glob(dir_frame + '*.txt')
                for frame in list_frame:
                    # load data
                    with open(frame) as f:
                        skeleton = np.array(f.readline().split(','), dtype=np.float).reshape(-1, 3)
                    skeleton = skeleton[reorder]

                    img = np.array(Image.open(frame[:-13] + 'depth.png'))

                    skel_hom = np.concatenate([skeleton, np.ones([skeleton.shape[0], 1])], 1)
                    skel_camcoords = cam_extr.dot(skel_hom.transpose()).transpose()[:, :3].astype(
                        np.float32)  # camera coordinates(N,3)
                    skel_hom2d = np.array(cam_intr).dot(skel_camcoords.transpose()).transpose()
                    skel_proj = (skel_hom2d / skel_hom2d[:, 2:])[:, :2]  # pixel coordinates

                    # fig = plt.figure()
                    # ax = fig.add_subplot(111)
                    # ax.imshow(img, cmap='gray')
                    # plt.show()

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
                    img_cropped[np.where(img_cropped == 0)] = np.max(img_cropped) + 1
                    depths = img_cropped.reshape(-1, 1)
                    label = KMeans(n_clusters=2, init=np.reshape([np.max(img_cropped) + 1, np.min(img_cropped)], [-1, 1]), n_init=1).fit_predict(depths)
                    centerDepth = np.mean(depths[label.astype(bool)])
                    if np.abs(centerDepth - skel_camcoords[3, 2]) > 100:
                        centerDepth = skel_camcoords[3, 2]

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

                    # padding and cropping image
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

                    # sample points
                    if len(points) < 5000:
                        continue
                    while len(points) < 6000:
                        points = np.repeat(points, 2, axis=0)
                    randInidices = np.arange(len(points))
                    np.random.shuffle(randInidices)
                    final_points = points[randInidices[:6000], :] * np.array([-1, 1, 1])  # transform to right hand (mm)

                    final_joints = joints * 0.001 * np.array([-1, 1, 1])  # transform to right hand (mm->m)
                    np.save(dir_points[i] + str(counter) + '.npy', final_points)
                    np.save(dir_joints[i] + str(counter) + '.npy', final_joints)
                    counter += 1


            

