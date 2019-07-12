import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from six.moves import xrange


# dir_object_points = '/home/haojie/Desktop/MyCode/Visualization Figures/object points/'
# dir_clean_hand_points = '/home/haojie/Desktop/MyCode/Visualization Figures/clean hand points/'
# dir_occluded_hand_points = '/home/haojie/Desktop/MyCode/Visualization Figures/occluded hand points/'
# dir_depth_image = '/home/haojie/Desktop/MyCode/Visualization Figures/depth image/'
# dir_joints = '/home/haojie/Desktop/MyCode/Visualization Figures/joints/'
# dir_points_out = '/home/haojie/Desktop/MyCode/Visualization Figures/points out/'
# dir_pose_out = '/home/haojie/Desktop/MyCode/Visualization Figures/pose out/'
# dir_scores = '/home/haojie/Desktop/MyCode/Visualization Figures/scores/'

dir_occluded_hand_points = '/home/haojie/Desktop/MyCode/Visualization Figures/FPH_with_object/occluded hand points/'
dir_joints = '/home/haojie/Desktop/MyCode/Visualization Figures/FPH_with_object/joints/'
dir_points_out = '/home/haojie/Desktop/MyCode/Visualization Figures/FPH_with_object/points out/'
dir_pose_out = '/home/haojie/Desktop/MyCode/Visualization Figures/FPH_with_object/pose out/'
dir_scores = '/home/haojie/Desktop/MyCode/Visualization Figures/FPH_with_object/scores/'

dir_object_points = '/home/haojie/Desktop/object points/'
dir_clean_hand_points = '/home/haojie/Desktop/clean hand points/'
# dir_occluded_hand_points = '/home/haojie/Desktop/occluded hand points/'
dir_depth_image = '/home/haojie/Desktop/depth image/'
# dir_joints = '/home/haojie/Desktop/joints/'
# dir_points_out = '/home/haojie/Desktop/points out/'
# dir_pose_out = '/home/haojie/Desktop/pose out/'
# dir_scores = '/home/haojie/Desktop/scores/'
cm2 = plt.cm.plasma # object
cm1 = plt.cm.winter # clean hand
cm4 = plt.cm.viridis
cm3 = plt.cm.summer # output


def object(file):
    points = np.load(dir_object_points+file)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(-90, -90)
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    # ax.grid(False)
    ax.set_axis_off()

    ax.scatter(points[:, 0], points[:, 1], c=-points[:, 2], marker='o', s=15, alpha=1, cmap=cm2)
    # plt.show()


def clean_hand(file):
    points = np.load(dir_clean_hand_points+file)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(-90, -90)
    ax.set_axis_off()
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})

    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='o', s=15, alpha=1)
    ax.scatter(points[:, 0], points[:, 1], c=-points[:, 2], marker='o', s=15, alpha=1, cmap=cm1)
    # plt.show()


def clean_hand_and_joints(file):
    points = np.load(dir_clean_hand_points+file)
    joints = np.load(dir_joints+file)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.view_init(-90, -90)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', alpha=0.5)
    for i in range(6):
        k = np.mod(i+1, 6)
        ax.plot([joints[i, 0], joints[k, 0]], [joints[i, 1], joints[k, 1]], [joints[i, 2], joints[k, 2]], color='r', marker='*')
    for j in xrange(1, 6):
        q = 3*(j+1)
        ax.plot([joints[j, 0], joints[q, 0]], [joints[j, 1], joints[q, 1]], [joints[j, 2], joints[q, 2]], color='r', marker='*')
        for m in range(2):
            ax.plot([joints[q+m, 0], joints[q+m+1, 0]], [joints[q+m, 1], joints[q+m+1, 1]], [joints[q+m, 2], joints[q+m+1, 2]], color='r', marker='*')
    # plt.show()


def clean_hand_and_object(file):
    clean_hand = np.load(dir_clean_hand_points+file)
    object = np.load(dir_object_points+file)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(-90, -90)
    ax.set_axis_off()
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    # ax.scatter(clean_hand[:, 0], clean_hand[:, 1], clean_hand[:, 2], marker='o', s=15, alpha=1)
    # ax.scatter(object[:, 0], object[:, 1], object[:, 2], marker='o', s=15, alpha=1)

    ax.scatter(clean_hand[:, 0], clean_hand[:, 1], c=-clean_hand[:, 2], marker='o', s=15, alpha=1, cmap=cm1)
    ax.scatter(object[:, 0], object[:, 1], c=-object[:, 2], marker='o', s=15, alpha=1, cmap=cm2)
    # plt.show()


def occluded_hand(file):
    points = np.load(dir_occluded_hand_points+file)
    joints = np.load(dir_joints + file)
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.set_axis_off()
    ax.view_init(-90, -90)  # view from the camera angle
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})

    # ax.scatter(points[:, 0], points[:, 1], c=-points[:, 2], marker='o', alpha=1, s=15, cmap=cm1)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='o', alpha=1, s=15)

    for i in range(6):
        k = np.mod(i + 1, 6)
        ax.plot([joints[i, 0], joints[k, 0]], [joints[i, 1], joints[k, 1]], [joints[i, 2], joints[k, 2]], color='y', marker='*')
    for j in xrange(1, 6):
        q = 3 * (j + 1)
        ax.plot([joints[j, 0], joints[q, 0]], [joints[j, 1], joints[q, 1]], [joints[j, 2], joints[q, 2]], color='y', marker='*')
        for m in range(2):
            ax.plot([joints[q + m, 0], joints[q + m + 1, 0]], [joints[q + m, 1], joints[q + m + 1, 1]], [joints[q + m, 2], joints[q + m + 1, 2]], color='y', marker='*')

    # plt.show()


def occluded_hand_depth_and_joints(file):
    points = np.load(dir_occluded_hand_points + file)
    joints = np.load(dir_joints + file)
    scores = np.load(dir_scores+file)
    print(len(scores))
    print(np.shape(scores))
    scores = np.sum(scores, axis=1)

    median = np.median(scores)-2
    plt.figure()
    plt.axis('off')

    temp = np.where(scores>median)
    plt.scatter(points[temp, 0], -points[temp, 1], c='b', s=15, alpha=0.5, marker='o')
    temp = np.where(scores<=median)
    plt.scatter(points[temp, 0], -points[temp, 1], c='r', s=15, alpha=0.5, marker='o')

    # for i in range(6):
    #     k = np.mod(i + 1, 6)
    #     plt.plot([joints[i, 0], joints[k, 0]], [-joints[i, 1], -joints[k, 1]], color='g', marker='*')
    # for j in xrange(1, 6):
    #     q = 3 * (j + 1)
    #     plt.plot([joints[j, 0], joints[q, 0]], [-joints[j, 1], -joints[q, 1]], color='g', marker='*')
    #     for m in range(2):
    #         plt.plot([joints[q + m, 0], joints[q + m + 1, 0]], [-joints[q + m, 1], -joints[q + m + 1, 1]], color='g', marker='*')


def joints_and_pose_out(file):
    joints = np.load(dir_joints+file)
    pose_out = np.load(dir_pose_out+file)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(-90, -90)
    # ax.set_axis_off()
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})

    for i in range(6):
        k = np.mod(i + 1, 6)
        ax.plot([joints[i, 0], joints[k, 0]], [joints[i, 1], joints[k, 1]], [joints[i, 2], joints[k, 2]], color='y', marker='*')
    for j in xrange(1, 6):
        q = 3 * (j + 1)
        ax.plot([joints[j, 0], joints[q, 0]], [joints[j, 1], joints[q, 1]], [joints[j, 2], joints[q, 2]], color='y', marker='*')
        for m in range(2):
            ax.plot([joints[q + m, 0], joints[q + m + 1, 0]], [joints[q + m, 1], joints[q + m + 1, 1]], [joints[q + m, 2], joints[q + m + 1, 2]], color='y', marker='*')

    for i in range(6):
        k = np.mod(i + 1, 6)
        ax.plot([pose_out[i, 0], pose_out[k, 0]], [pose_out[i, 1], pose_out[k, 1]], [pose_out[i, 2], pose_out[k, 2]], color='g', marker='*')
    for j in xrange(1, 6):
        q = 3 * (j + 1)
        ax.plot([pose_out[j, 0], pose_out[q, 0]], [pose_out[j, 1], pose_out[q, 1]], [pose_out[j, 2], pose_out[q, 2]], color='g', marker='*')
        for m in range(2):
            ax.plot([pose_out[q + m, 0], pose_out[q + m + 1, 0]], [pose_out[q + m, 1], pose_out[q + m + 1, 1]], [pose_out[q + m, 2], pose_out[q + m + 1, 2]], color='g', marker='*')


def points_out(file):
    points = np.load(dir_points_out+file)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(-90, -90)
    ax.set_axis_off()

    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})

    ax.scatter(points[:, 0], points[:, 1], c=-points[:, 2], marker='o', s=15, alpha=1, cmap=cm3)
    # plt.show()


def depth_image(file):
    depth_map = np.load(dir_depth_image+file)
    plt.figure()
    plt.axis('off')

    temp = depth_map[np.where(depth_map!=np.inf)]
    depth_max = np.max(temp)
    depth_min = np.min(temp)
    depth_map_nor = (depth_map-depth_min)/(depth_max-depth_min)

    # cmap = colors.ListedColormap(['#FFFFFF', '#9FF113', '#5FBB44', '#F5F329', '#E50B32'], 'indexed')
    # fig
    fig = plt.imshow(-depth_map_nor, cmap='gray')
    # cb = plt.colorbar(fig, aspect=10)
    # cb.set_label('Normed Depth Value', fontsize=15)


file = '50.npy'  # 11 20 21 30 31
# 28(shouqiang), 61(bi), 26(lifangti), 64(erji), 31(feiji)
# clean_hand_and_joints(file)
# object(file)
# clean_hand(file)
# clean_hand_and_object(file)
occluded_hand(file)
occluded_hand_depth_and_joints(file)
# depth_image(file)
# points_out(file)
joints_and_pose_out(file)
plt.show()
