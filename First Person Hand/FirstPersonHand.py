import numpy as np
import matplotlib.pyplot as plt
import os
from six.moves import xrange
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import time
boundingBoxSize = 140

dir_points = '/home/haojie/Desktop/FirstPersonHand/points_1/'
dir_joints = '/home/haojie/Desktop/FirstPersonHand/joints_1/'


cam_extr = np.array(
    [[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
     [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
     [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
     [0, 0, 0, 1]])
cam_extr = np.eye(4)
cam_intr = np.array([[475.065948, 0, 315.944855],
                     [0, 475.065857, 245.287079],
                     [0, 0, 1]])
u0 = 315.944855
v0 = 245.287079
f_x = 475.065948
f_y = 475.065857

reorder_idx = np.array([0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20])

dir_video = '/home/haojie/Desktop/FirstPersonHand/video/'
dir_pose = '/home/haojie/Desktop/FirstPersonHand/pose/'


list_subject = os.listdir(dir_video)


def visualize_joints_2d(ax, joints, joint_idxs=True, links=None, alpha=1):
    """Draw 2d skeleton on matplotlib axis"""
    if links is None:
        links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                 (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    ax.scatter(x, y, 1, 'r')

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            plt.annotate(str(row_idx), (row[0], row[1]))
    _draw2djoints(ax, joints, links, alpha=alpha)


def _draw2djoints(ax, annots, links, alpha=1):
    """Draw segments, one color per link"""
    colors = ['r', 'm', 'b', 'c', 'g']

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw2dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=colors[finger_idx],
                alpha=alpha)


def _draw2dseg(ax, annot, idx1, idx2, c='r', alpha=1):
    """Draw segment of given color"""
    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]],
        c=c,
        alpha=alpha)


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


def viewCorrection(center3D, cloud, joint):

    aroundYAngle = np.arctan((center3D[0])/center3D[2])
    center3DRotated = np.matmul(center3D, np.transpose(rot_y(-aroundYAngle)))
    aroundXAngle = np.arctan((center3DRotated[1])/center3DRotated[2])

    viewRotation = np.matmul(rot_x(aroundXAngle), rot_y(-aroundYAngle))
    cloud = np.matmul(cloud, np.transpose(viewRotation))
    joint = np.matmul(joint, np.transpose(viewRotation))

    return viewRotation, cloud, joint


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
                # fig = plt.figure()
                # ax = fig.add_subplot(111)
                # ax = Axes3D(fig)
                # ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
                # ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
                # ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
                #
                # ax.view_init(-90, -90)

                frame = dir_video_seq+'depth_{:04d}.png'.format(i)
                # print(frame)
                img = Image.open(frame)
                img = np.array(img)
                # ax.imshow(img, cmap='gray') # 111111111111111111111111111

                # get joints ##################################
                skel_hom = np.concatenate([skeleton[i], np.ones([skeleton[i].shape[0], 1])], 1)
                skel_camcoords = cam_extr.dot(skel_hom.transpose()).transpose()[:, :3].astype(np.float32)

                skel_hom2d = np.array(cam_intr).dot(skel_camcoords.transpose()).transpose()
                skel_proj = (skel_hom2d / skel_hom2d[:, 2:])[:, :2]
                # print(skel_proj)
                invalidIndices = np.logical_or(np.logical_or(np.logical_or(skel_proj[:, 0]>640, skel_proj[:, 1]>480), skel_proj[:, 0]<0), skel_proj[:, 1]<0)
                # print(np.sum(invalidIndices))
                if np.sum(invalidIndices)>2:
                    #plt.close(fig)
                    continue
                tl_u = int(np.maximum(np.min(np.floor(skel_proj[:, 0]))-50, 0))
                tl_v = int(np.maximum(np.min(np.floor(skel_proj[:, 1]))-50, 0))
                br_u = int(np.minimum(np.max(np.floor(skel_proj[:, 0]))+50, 640))
                br_v = int(np.minimum(np.max(np.floor(skel_proj[:, 1]))+50, 480))

                img = img[tl_v:br_v, tl_u:br_u]
                pixel = np.empty((3, img.size))
                pixel[2, :] = img.ravel(1)
                # print(img)
                # ax.imshow(img, cmap='gray')
                a, b = np.meshgrid(np.arange(tl_u, br_u), np.arange(tl_v, br_v))
                pixel[0, :] = a.ravel(1)
                pixel[1, :] = b.ravel(1)
                # visualize_joints_2d(ax, skel_proj[reorder_idx], joint_idxs=False) # 3333333333333333333333333333
                # ax.plot([tl_u, br_u], [tl_v, tl_v]) # 22222222222222222222222222222222222
                # ax.plot([tl_u, br_u], [tl_v, tl_v])
                # ax.plot([tl_u, tl_u], [tl_v, br_v])
                # ax.plot([tl_u, tl_u], [tl_v, br_v])

                # starttime = time.time()
                # get points ##################################
                points = np.empty((3, img.size))
                points[0, :] = (pixel[0, :]+0.5-u0)*pixel[2, :]/f_x
                points[1, :] = (pixel[1, :]+0.5-v0)*pixel[2, :]/f_y
                points[2, :] = pixel[2, :]
                points = points.T
                # print(time.time() - starttime)
                # print(offset)
                # print(skel_camcoords)
                # print(points[:100, :])

                _, points, skel_camcoords = viewCorrection(np.mean(skel_camcoords, axis=0), points, skel_camcoords)

                offset = skel_camcoords[3]
                points -= offset
                skel_camcoords -= offset

                validIndicies = np.logical_and(np.logical_and(np.abs(points[:, 0]) < boundingBoxSize, np.abs(points[:, 1]) < boundingBoxSize), np.abs(points[:, 2]) < boundingBoxSize)
                points = points[validIndicies, :]

                if len(points) < 5000:
                    continue
                while len(points) < 6000:
                    points = np.repeat(points, 2, axis=0)

                randInidices = np.arange(len(points))
                np.random.shuffle(randInidices)
                final_points = points[randInidices[:6000], :]

                ###############################################
                # ax.scatter(final_points[:, 0], final_points[:, 1], final_points[:, 2], c='b', marker='o', s=15, alpha=1)
                # for i in range(6):
                #     k = np.mod(i + 1, 6)
                #     ax.plot([skel_camcoords[i, 0], skel_camcoords[k, 0]], [skel_camcoords[i, 1], skel_camcoords[k, 1]], [skel_camcoords[i, 2], skel_camcoords[k, 2]],
                #             color='r', marker='*')
                # for j in xrange(1, 6):
                #     q = 3 * (j + 1)
                #     ax.plot([skel_camcoords[j, 0], skel_camcoords[q, 0]], [skel_camcoords[j, 1], skel_camcoords[q, 1]], [skel_camcoords[j, 2], skel_camcoords[q, 2]],
                #             color='r', marker='*')
                #     for m in range(2):
                #         ax.plot([skel_camcoords[q + m, 0], skel_camcoords[q + m + 1, 0]], [skel_camcoords[q + m, 1], skel_camcoords[q + m + 1, 1]],
                #                 [skel_camcoords[q + m, 2], skel_camcoords[q + m + 1, 2]], color='r', marker='*')
                # plt.show()
                # print(',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,')

                # save all results
                # final_points = final_points.reshape([1, -1])  # mm
                # skel_camcoords = skel_camcoords.reshape([1, -1]) * 0.001  # mm->m
                final_points = final_points  # mm
                skel_camcoords = skel_camcoords * 0.001  # mm->m
                np.save(dir_points+str(counter)+'.npy', final_points)
                np.save(dir_joints+str(counter)+'.npy', skel_camcoords)
                counter += 1

