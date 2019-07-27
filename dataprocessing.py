from utilities import *
from get_an_object import *
from random import choice
import os

# dir_object_points = '/home/haojie/Desktop/MyCode/Visualization Figures/object points/'
# dir_clean_hand_points = '/home/haojie/Desktop/MyCode/Visualization Figures/clean hand points/'
# dir_occluded_hand_points = '/home/haojie/Desktop/MyCode/Visualization Figures/occluded hand points/'
# dir_depth_image = '/home/haojie/Desktop/MyCode/Visualization Figures/depth image/'

f_x = 475.06
f_y = 475.06
offset = 40


def preprocessPoint_training(points, joints, scales, label):
    points = points * 0.01
    joints = joints * 10.0
    joints = joints.reshape(21, 3)

    # rand translation
    randTrans = np.float32(np.maximum(np.minimum(np.random.normal(0.0, 10.0, (3,)), 25.0), -25.0)/1000)
    randTrans[2] = np.float32(np.maximum(np.minimum(np.random.normal(0.0, 15.0, (1,)), 27.0), -27.0)/1000)
    joints = joints + randTrans*10.0
    points = points + randTrans*10.0

    # rotation around camera view axis (z-axis)
    randAngle = (-math.pi+2*math.pi*np.random.rand(1))
    (points[:, 0], points[:, 1]) = rotate((0, 0), (points[:, 0], points[:, 1]), randAngle)
    (joints[:, 0], joints[:, 1]) = rotate((0, 0), (joints[:, 0], joints[:, 1]), randAngle)

    # rand scaling
    globalRandScale = np.float32(np.maximum(np.minimum(np.random.normal(1.0, 0.075), 1.25), 0.75)) # 0.75 -> 0.6 0.0075 -> 0.5
    randScale = globalRandScale * np.float32(np.maximum(np.minimum(np.random.normal(1.0, 0.0, (3,)), 1.1), 0.9))
    points = points*randScale
    joints = joints*randScale
    joints = joints.reshape(63)
    scales = scales*globalRandScale

    if label == 1:   # clean hand
        points_clean = sample(points)
        temp = np.random.rand()
        if temp < 0.3:   # do not add object
            points_occluded = sample(points)
            object_points = np.zeros([INPUT_POINT_SIZE, 3])
            depth_image = np.zeros([80, 80])
        else:   # add object
            points_augmented, object_points, depth_image = augmentation(points, joints)
            points_occluded = sample(points_augmented)
            object_points = sample(object_points)
    else:     # occluded hand without augmentation
        points_clean = np.random.rand(INPUT_POINT_SIZE, 3)  # randomly generate clean hand
        points_occluded = sample(points)
        object_points = np.zeros([INPUT_POINT_SIZE, 3])
        depth_image = np.zeros([80, 80])

    # # add objects to hand point clouds
    # points_clean = sample(points)
    # # points_augmented = np.zeros(np.shape(points))
    # temp = np.random.rand()
    # if temp < 0.4:
    #     points_occluded = sample(points)
    #     object_points = np.zeros([INPUT_POINT_SIZE, 3])
    #     depth_image = np.zeros([80, 80])
    # elif temp > 0.6:
    #     points_augmented, object_points, depth_image = augmentation_extra(points, joints)
    #     points_occluded = sample(points_augmented)
    #     object_points = sample(object_points)
    # else:
    #     points_augmented, object_points, depth_image = augmentation(points, joints)
    #     points_occluded = sample(points_augmented)
    #     object_points = sample(object_points)
    return np.float32(points_occluded), np.float32(points_clean), np.float32(joints), np.float32(scales), np.float32(object_points), np.float32(depth_image)


def preprocessPoint_validation(points, joints, scales, label):
    points = points * 0.01 # mm-> dm
    joints = joints * 10.0 # m -> dm

    if label == 1:   # clean hand
        points_clean = sample(points)
        temp = np.random.rand()
        if temp < 0.3:   # do not add object
            points_occluded = sample(points)
            object_points = np.zeros([INPUT_POINT_SIZE, 3])
            depth_image = np.zeros([80, 80])
        else:   # add object
            points_augmented, object_points, depth_image = augmentation(points, joints)
            points_occluded = sample(points_augmented)
            object_points = sample(object_points)
    else:     # occluded hand without augmentation
        points_clean = np.random.rand(INPUT_POINT_SIZE, 3)  # randomly generate clean hand
        points_occluded = sample(points)
        object_points = np.zeros([INPUT_POINT_SIZE, 3])
        depth_image = np.zeros([80, 80])

    # # add objects to hand point clouds
    # points_clean = sample(points)
    # if np.random.rand() < 0.4:
    #     points_occluded = sample(points)
    #     object_points = np.zeros([INPUT_POINT_SIZE, 3])
    #     depth_image = np.zeros([80, 80])
    # else:
    #     points_augmented, object_points, depth_image = augmentation(points, joints)
    #     points_occluded = sample(points_augmented)
    #     object_points = sample(object_points)

    return np.float32(points_occluded), np.float32(points_clean), np.float32(joints), np.float32(scales), np.float32(object_points), np.float32(depth_image)


def preprocessPoint_test(points, joints):
    points = points * 0.01  # mm-> dm
    joints = joints * 10.0  # m -> dm
    points_sampled = sample(points)
    return np.float32(points_sampled), np.float32(joints)


def augmentation(points, joints):
    """This function return the 3D Occluded Point Cloud"""
    # get an object randomly
    object_points, object_center, object_max_dis = get_an_object()
    object_points -= object_center

    # rotation & scaling
    theta = -math.pi + 2 * math.pi * np.random.rand(3)
    R_x = np.array([[1, 0, 0], [0, math.cos(theta[0]), -math.sin(theta[0])], [0, math.sin(theta[0]), math.cos(theta[0])]])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])], [0, 1, 0], [-math.sin(theta[1]), 0, math.cos(theta[1])]])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0], [math.sin(theta[2]), math.cos(theta[2]), 0], [0, 0, 1]])
    scale = np.float32(np.maximum(np.minimum(np.random.normal(0.8, 0.3), 1.3), 0.5)) / object_max_dis
    R_scaled = np.dot(np.dot(R_z, R_y), R_x) * scale
    object_points = np.dot(object_points, np.transpose(R_scaled))

    # translation(location)
    # joints_center = np.mean(joints.reshape(21, 3), axis=0)
    joints_center = joints.reshape(21, 3)[np.random.randint(0, 21)]
    d_xyz = np.random.normal(joints_center, 0.3)
    d_xyz = np.float32(np.maximum(np.minimum(d_xyz, boundingBoxSize), -boundingBoxSize))
    object_points += d_xyz

    # combine
    occluded_pcd = np.concatenate((object_points, points))

    if np.random.rand() < 0:      ########################## 0.4 -> 0
        # get an object randomly
        object_points, object_center, object_max_dis = get_an_object()
        object_points -= object_center

        # rotation & scaling
        theta = -math.pi + 2 * math.pi * np.random.rand(3)
        R_x = np.array(
            [[1, 0, 0], [0, math.cos(theta[0]), -math.sin(theta[0])], [0, math.sin(theta[0]), math.cos(theta[0])]])
        R_y = np.array(
            [[math.cos(theta[1]), 0, math.sin(theta[1])], [0, 1, 0], [-math.sin(theta[1]), 0, math.cos(theta[1])]])
        R_z = np.array(
            [[math.cos(theta[2]), -math.sin(theta[2]), 0], [math.sin(theta[2]), math.cos(theta[2]), 0], [0, 0, 1]])
        scale = np.float32(np.maximum(np.minimum(np.random.normal(0.4, 0.2), 0.6), 0.2)) / object_max_dis
        R_scaled = np.dot(np.dot(R_z, R_y), R_x) * scale
        object_points_1 = np.dot(object_points, np.transpose(R_scaled))

        # translation(location)
        joints_center = joints.reshape(21, 3)[choice([8, 11, 14, 17, 20])]
        d_xyz = np.random.normal(joints_center, 0.3)
        d_xyz = np.float32(np.maximum(np.minimum(d_xyz, boundingBoxSize), -boundingBoxSize))
        object_points_1 += d_xyz

        # combine
        occluded_pcd = np.concatenate((object_points_1, occluded_pcd))

    # projection to depth image
    occluded_pcd += np.array([0, 0, 19])
    depth_image = np.ones((80, 80)) * np.inf

    occluded_pcd[:, 0] = occluded_pcd[:, 0] * f_x / occluded_pcd[:, 2] + offset
    occluded_pcd[:, 1] = occluded_pcd[:, 1] * f_y / occluded_pcd[:, 2] + offset
    occluded_pcd[:, 0] = np.floor(occluded_pcd[:, 0])
    occluded_pcd[:, 1] = np.floor(occluded_pcd[:, 1])
    valid_indices = list(set(np.where(occluded_pcd[:, 0] >= 0.0)[0]) & set(np.where(occluded_pcd[:, 0] <= 79.0)[0])
                         & set(np.where(occluded_pcd[:, 1] >= 0.0)[0]) & set(np.where(occluded_pcd[:, 1] <= 79.0)[0]))
    occluded_pcd = occluded_pcd[valid_indices]
    for i in range(occluded_pcd.shape[0]):
        x_index = int(occluded_pcd[i, 0])
        y_index = int(occluded_pcd[i, 1])
        if occluded_pcd[i, 2] < depth_image[x_index, y_index]:
            depth_image[x_index, y_index] = occluded_pcd[i, 2]

    # projection to point cloud
    a, b = np.meshgrid(np.arange(0, 80), np.arange(0, 80))
    u = a.ravel(1)
    v = b.ravel(1)
    d = depth_image.ravel(1)
    points_augmented = np.empty((3, 6400))
    points_augmented[0, :] = (u + 0.5 - offset) * d / f_x
    points_augmented[1, :] = (v + 0.5 - offset) * d / f_y
    points_augmented[2, :] = d
    points_augmented = points_augmented.T
    # for u in range(80):
    #     for v in range(80):
    #         if depth_image[u, v] != np.inf:
    #             x = ((u + 0.5 - offset) * depth_image[u, v]) / f_x
    #             y = ((v + 0.5 - offset) * depth_image[u, v]) / f_y
    #             z = depth_image[u, v]
    #             points_augmented = np.concatenate((points_augmented, np.array([[x, y, z]])))
    validIndicies = np.logical_and(np.logical_and(np.abs(points_augmented[:, 0]) < boundingBoxSize, np.abs(points_augmented[:, 1]) < boundingBoxSize), np.abs(points_augmented[:, 2]) < boundingBoxSize)
    points_augmented = points_augmented[validIndicies, :]  # remove np.inf
    points_augmented -= np.array([0, 0, 19])
    return points_augmented, object_points, depth_image


def augmentation_extra(points, joints):
    """This function return the 3D Occluded Point Cloud"""
    # get an object randomly
    object_points, object_center, object_max_dis = get_an_object_extra()
    object_points -= object_center
    object_points[:, 2] = 0

    # rotation & scaling
    theta = np.zeros(3)
    theta[2] = -math.pi + 2 * math.pi * np.random.rand(1)
    theta[0] = (np.random.random() - 0.5) * 2 * 0.4
    theta[1] = (np.random.random() - 0.5) * 2 * 0.4
    R_x = np.array(
        [[1, 0, 0], [0, math.cos(theta[0]), -math.sin(theta[0])], [0, math.sin(theta[0]), math.cos(theta[0])]])
    R_y = np.array(
        [[math.cos(theta[1]), 0, math.sin(theta[1])], [0, 1, 0], [-math.sin(theta[1]), 0, math.cos(theta[1])]])
    R_z = np.array(
        [[math.cos(theta[2]), -math.sin(theta[2]), 0], [math.sin(theta[2]), math.cos(theta[2]), 0], [0, 0, 1]])
    scale = np.float32(np.maximum(np.minimum(np.random.normal(1.25, 0.4), 1.7), 0.8)) / object_max_dis
    R_scaled = np.dot(np.dot(R_z, R_y), R_x) * scale
    object_points = np.dot(object_points, np.transpose(R_scaled))

    # translation(location)
    # joints_center = np.mean(joints.reshape(21, 3), axis=0)
    joints = joints.reshape(21, 3)
    joints_center = np.squeeze(joints[np.where(joints[:, 2]==np.max(joints[:, 2]))])
    d_xyz = np.random.normal(joints_center, 0.08)
    d_xyz[2] += 0.08 # np.random.normal(joints_center, 0.25)
    d_xyz = np.float32(np.maximum(np.minimum(d_xyz, boundingBoxSize), -boundingBoxSize))
    object_points += d_xyz

    # combine
    occluded_pcd = np.concatenate((object_points, points))

    if np.random.rand() < 0.4:
        # get an object randomly
        object_points, object_center, object_max_dis = get_an_object()
        object_points -= object_center

        # rotation & scaling
        theta = -math.pi + 2 * math.pi * np.random.rand(3)
        R_x = np.array(
            [[1, 0, 0], [0, math.cos(theta[0]), -math.sin(theta[0])], [0, math.sin(theta[0]), math.cos(theta[0])]])
        R_y = np.array(
            [[math.cos(theta[1]), 0, math.sin(theta[1])], [0, 1, 0], [-math.sin(theta[1]), 0, math.cos(theta[1])]])
        R_z = np.array(
            [[math.cos(theta[2]), -math.sin(theta[2]), 0], [math.sin(theta[2]), math.cos(theta[2]), 0], [0, 0, 1]])
        scale = np.float32(np.maximum(np.minimum(np.random.normal(0.4, 0.2), 0.6), 0.2)) / object_max_dis
        R_scaled = np.dot(np.dot(R_z, R_y), R_x) * scale
        object_points_1 = np.dot(object_points, np.transpose(R_scaled))

        # translation(location)
        joints_center = joints.reshape(21, 3)[choice([12, 13, 16, 19, 15, 18, 4, 5])]
        d_xyz_1 = np.random.normal(joints_center, 0.3)
        d_xyz_1[2] = np.minimum(d_xyz_1[2], d_xyz[2])
        d_xyz_1 = np.float32(np.maximum(np.minimum(d_xyz_1, boundingBoxSize), -boundingBoxSize))
        object_points_1 += d_xyz_1

        # combine
        occluded_pcd = np.concatenate((object_points_1, occluded_pcd))

    # projection to depth image
    occluded_pcd += np.array([0, 0, 19])
    depth_image = np.ones((80, 80)) * np.inf

    occluded_pcd[:, 0] = occluded_pcd[:, 0] * f_x / occluded_pcd[:, 2] + offset
    occluded_pcd[:, 1] = occluded_pcd[:, 1] * f_y / occluded_pcd[:, 2] + offset
    occluded_pcd[:, 0] = np.floor(occluded_pcd[:, 0])
    occluded_pcd[:, 1] = np.floor(occluded_pcd[:, 1])
    valid_indices = list(set(np.where(occluded_pcd[:, 0] >= 0.0)[0]) & set(np.where(occluded_pcd[:, 0] <= 79.0)[0])
                         & set(np.where(occluded_pcd[:, 1] >= 0.0)[0]) & set(np.where(occluded_pcd[:, 1] <= 79.0)[0]))
    occluded_pcd = occluded_pcd[valid_indices]
    for i in range(occluded_pcd.shape[0]):
        x_index = int(occluded_pcd[i, 0])
        y_index = int(occluded_pcd[i, 1])
        if occluded_pcd[i, 2] < depth_image[x_index, y_index]:
            depth_image[x_index, y_index] = occluded_pcd[i, 2]

    # projection to point cloud
    points_augmented = np.empty((0, 3))
    for u in range(80):
        for v in range(80):
            if depth_image[u, v] != np.inf:
                x = ((u + 0.5 - offset) * depth_image[u, v]) / f_x
                y = ((v + 0.5 - offset) * depth_image[u, v]) / f_y
                z = depth_image[u, v]
                points_augmented = np.concatenate((points_augmented, np.array([[x, y, z]])))
    points_augmented -= np.array([0, 0, 19])
    return points_augmented, object_points, depth_image
