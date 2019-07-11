"""This script is for offline visualization of training result"""
import numpy as np
from config import *


# FPH -> First Person Hand

dir_object_points = '/home/haojie/Desktop/MyCode/Visualization Figures/object points/'
dir_clean_hand_points = '/home/haojie/Desktop/MyCode/Visualization Figures/clean hand points/'
dir_occluded_hand_points = '/home/haojie/Desktop/MyCode/Visualization Figures/occluded hand points/'
dir_depth_image = '/home/haojie/Desktop/MyCode/Visualization Figures/depth image/'
dir_joints = '/home/haojie/Desktop/MyCode/Visualization Figures/joints/'
dir_points_out = '/home/haojie/Desktop/MyCode/Visualization Figures/points out/'
dir_pose_out = '/home/haojie/Desktop/MyCode/Visualization Figures/pose out/'
dir_scores = '/home/haojie/Desktop/MyCode/Visualization Figures/scores/'

dir_occluded_hand_points_FPH_all = '/home/haojie/Desktop/MyCode/Visualization Figures/FPH_with_object/occluded hand points/'
dir_joints_FPH_all = '/home/haojie/Desktop/MyCode/Visualization Figures/FPH_with_object/joints/'
dir_points_out_FPH_all = '/home/haojie/Desktop/MyCode/Visualization Figures/FPH_with_object/points out/'
dir_pose_out_FPH_all = '/home/haojie/Desktop/MyCode/Visualization Figures/FPH_with_object/pose out/'
dir_scores_FPH_all = '/home/haojie/Desktop/MyCode/Visualization Figures/FPH_with_object/scores/'

dir_occluded_hand_points_FPH_with_object = '/home/haojie/Desktop/MyCode/Visualization Figures/FPH_with_object/occluded hand points/'
dir_joints_FPH_with_object = '/home/haojie/Desktop/MyCode/Visualization Figures/FPH_with_object/joints/'
dir_points_out_FPH_with_object = '/home/haojie/Desktop/MyCode/Visualization Figures/FPH_with_object/points out/'
dir_pose_out_FPH_with_object = '/home/haojie/Desktop/MyCode/Visualization Figures/FPH_with_object/pose out/'


def offline_visualization(sess, points_occluded, points_clean, joints, object, depth_image, points_out, pose_out, scores, handle, validation_handle, training_var, batch_size, visual_steps):
    for i in range(visual_steps):
        points_occluded_this, points_clean_this, joints_this, object_this, depth_image_this, points_out_this, pose_out_this, scores_this = sess.run([points_occluded, points_clean, joints, object, depth_image, points_out, pose_out, scores], feed_dict={handle:validation_handle, training_var:False})
        scores_this = np.squeeze(scores_this)
        for j in range(batch_size):
            if not np.all(object_this[j]==0):
                points_occluded_temp = points_occluded_this[j]
                points_clean_temp = points_clean_this[j]
                joints_temp = np.reshape(joints_this[j], [-1, 3])
                object_temp = object_this[j]
                depth_image_temp = depth_image_this[j]
                points_out_temp = points_out_this[j]
                pose_out_temp = np.reshape(pose_out_this[j], [-1, 3])
                scores_temp = scores_this[j]

                np.save(dir_occluded_hand_points+str(i)+str(j)+'.npy', points_occluded_temp)
                np.save(dir_clean_hand_points+str(i)+str(j)+'.npy', points_clean_temp)
                np.save(dir_joints+str(i)+str(j)+'.npy', joints_temp)
                np.save(dir_object_points+str(i)+str(j)+'.npy', object_temp)
                np.save(dir_depth_image+str(i)+str(j)+'.npy', depth_image_temp)
                np.save(dir_points_out+str(i)+str(j)+'.npy', points_out_temp)
                np.save(dir_pose_out+str(i)+str(j)+'.npy', pose_out_temp)
                np.save(dir_scores+str(i)+str(j)+'.npy', scores_temp)


def offline_visualization_FirstPersonHand(sess, points_occluded, joints, points_out, pose_out, scores, handle, test_handle, training_var, batch_size, visual_steps):
    for i in range(visual_steps):
        points_occluded_this, joints_this, points_out_this, pose_out_this, scores_this = sess.run([points_occluded, joints, points_out, pose_out, scores], feed_dict={handle:test_handle, training_var:False})
        scores_this = np.squeeze(scores_this)
        for j in range(batch_size):
            points_occluded_temp = points_occluded_this[j]
            joints_temp = np.reshape(joints_this[j], [-1, 3])
            points_out_temp = points_out_this[j]
            pose_out_temp = np.reshape(pose_out_this[j], [-1, 3])
            scores_temp = scores_this[j]

            np.save(dir_occluded_hand_points_FPH_all+str(i)+str(j)+'.npy', points_occluded_temp)
            np.save(dir_joints_FPH_all+str(i)+str(j)+'.npy', joints_temp)
            np.save(dir_points_out_FPH_all+str(i)+str(j)+'.npy', points_out_temp)
            np.save(dir_pose_out_FPH_all+str(i)+str(j)+'.npy', pose_out_temp)
            np.save(dir_scores_FPH_all + str(i) + str(j) + '.npy', scores_temp)
