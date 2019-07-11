"""This script is for online visualization of the training process."""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
fig = plt.figure(figsize=(18, 6), dpi=105)


def display_points(points_list, step, fig):
    plt.ion()
    plt.clf()
    fig.suptitle('Step %d'%step)
    ax = []
    for i in range(5):
        points = points_list[i]
        if i == 0:
            label = "points_occluded"
        elif i == 1:
            label = "points_clean"
        elif i == 2:
            label = "joints"
        elif i == 3:
            label = "points_out"
        else:
            label = "joints_out"
        if i < 3:
            ax.append(fig.add_subplot(131+i, projection='3d'))
            ax[i].scatter(points[:, 0], points[:, 1], points[:, 2], c='r', label=label)
            ax[i].set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
            ax[i].set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
            ax[i].set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
            ax[i].view_init(-90, 0)
            ax[i].legend()
        else:
            ax[i-2].scatter(points[:, 0], points[:, 1], points[:, 2], c='g', label=label)
            ax[i-2].legend()
    plt.pause(0.1)
    plt.show()
    plt.ioff()


def visualize_result(sess, points_occluded, points_clean, joints, points_out, pose_out, handle, traning_handle, training_var, batch_size, step):
    points_occluded_this, points_clean_this, joints_this, points_out_this, pose_out_this = sess.run([points_occluded, points_clean, joints, points_out, pose_out], feed_dict={handle: traning_handle, training_var: False})

    index = np.random.randint(0, batch_size) # randomly choose a sample

    points_occluded_this = points_occluded_this[index]
    points_clean_this = points_clean_this[index]
    joints_this = np.reshape(joints_this[index], [-1, 3])
    points_out_this = points_out_this[index]
    pose_out_this = np.reshape(pose_out_this[index], [-1, 3])
    # print(points_occluded_this)
    # print(points_clean_this)
    # print(joints_this)
    # print(points_out_this)
    # print(pose_out_this)
    print('********Step: %d'%step)
    display_points([points_occluded_this, points_clean_this, joints_this, points_out_this, pose_out_this], step, fig)
