import time
import numpy as np


def inference_validation(sess, loss_kl, loss_point_reconstruction, euc_joint_distance, loss_all, handle, validation_handle, training_var, validation_steps):
    start_time = time.time()
    print('validation_steps:  %d'%validation_steps)
    loss_kl_sum = 0
    loss_point_reconstruction_sum = 0
    loss_pose_estimation_sum = 0
    loss_all_sum = 0
    for _ in range(validation_steps):
        loss_kl_this, loss_point_reconstruction_this, loss_pose_estimation_this, loss_all_this = sess.run([loss_kl, loss_point_reconstruction, euc_joint_distance, loss_all], feed_dict={handle:validation_handle, training_var:False})

        loss_kl_sum += loss_kl_this
        loss_point_reconstruction_sum += loss_point_reconstruction_this
        loss_pose_estimation_sum += loss_pose_estimation_this
        loss_all_sum += loss_all_this

    loss_kl_avg = loss_kl_sum/validation_steps
    loss_point_reconstruction_avg = loss_point_reconstruction_sum/validation_steps
    loss_pose_estimation_avg = loss_pose_estimation_sum/validation_steps
    loss_all_avg = loss_all_sum/validation_steps

    duration = time.time() - start_time
    print('Average KL Loss per batch:  %.5f'%loss_kl_avg)
    print('Average Point Reconstruction Loss:  %.5f'%loss_point_reconstruction_avg)
    print('Average Pose Estimation euclidean distance:  %.5f'%loss_pose_estimation_avg)
    print('Average Total Loss:  %.5f'%loss_all_avg)
    print('Time for Evaluation:  %.2f'%duration)


def inference_test(sess, euc_joint_distance, handle, test_handle, training_var, test_steps):
    start_time = time.time()
    print('test_steps:  %d'%test_steps)

    loss_pose_estimation_sum = 0
    for _ in range(test_steps):
        loss_pose_estimation_this = sess.run(euc_joint_distance, feed_dict={handle: test_handle, training_var: False})

        loss_pose_estimation_sum += loss_pose_estimation_this

    loss_pose_estimation_avg = loss_pose_estimation_sum/test_steps

    duration = time.time() - start_time
    print('Average Pose Estimation euclidean distance:  %.5f'%loss_pose_estimation_avg)
    print('Time for Evaluation:  %.2f'%duration)


def inference_validation_batch(sess, loss_kl, loss_point_reconstruction, loss_pose_estimation, loss_all, handle, validation_handle, training_var, validation_steps):
    start_time = time.time()
    loss_kl_this, loss_point_reconstruction_this, loss_pose_estimation_this, loss_all_this = sess.run([loss_kl, loss_point_reconstruction, loss_pose_estimation, loss_all], feed_dict={handle: validation_handle, training_var: False})
    duration = time.time() - start_time
    print('KL Loss for current batch:  %.5f' % loss_kl_this)
    print('Point Reconstruction Loss for current batch:  %.5f' % loss_point_reconstruction_this)
    print('Pose Estimation Loss for current batch:  %.5f' % loss_pose_estimation_this)
    print('Total Loss for current batch:  %.5f' % loss_all_this)
    print('Time for Evaluation:  %.2f' % duration)


def inference_test_result(sess, points_out, pose_out, loss_kl, loss_point_reconstruction, loss_pose_estimation, loss_all, handle, test_handle, training_var, test_steps):
    start_time = time.time()

    F_points = open("points_out.txt", "wb")
    F_pose = open("pose_out.txt", "wb")
    # F_kl = open("loss_kl.txt", "wb")
    # F_point_reconstruction = open("loss_point_reconstruction.txt", "wb")
    # F_pose_estimation = open("loss_pose_estimation.txt", "wb")
    # F_all = open("loss_all.txt", "wb")

    for i in range(test_steps):
        if i%100 == 0:
            print(time.time() - start_time)
            print(i)
            start_time = time.time()
        points_out_this, pose_out_this, loss_kl_this, loss_point_reconstruction_this, loss_pose_estimation_this, loss_all_this = sess.run([points_out, pose_out, loss_kl, loss_point_reconstruction, loss_pose_estimation, loss_all], feed_dict={handle: test_handle, training_var: False})

        np.savetxt(F_points, points_out_this, delimiter="\t", newline="\n", fmt="%f")
        np.savetxt(F_pose, pose_out_this, delimiter="\t", newline="\n", fmt="%f")
        # np.savetxt(F_kl, loss_kl_this, delimiter="\t", newline="\n", fmt="%f")
        # np.savetxt(F_point_reconstruction, loss_point_reconstruction_this, delimiter="\t", newline="\n", fmt="%f")
        # np.savetxt(F_pose_estimation, loss_pose_estimation_this, delimiter="\t", newline="\n", fmt="%f")
        # np.savetxt(F_all, loss_all_this, delimiter="\t", newline="\n", fmt="%f")

    F_points.close()
    F_pose.close()
    # F_kl.close()
    # F_point_reconstruction.close()
    # F_pose_estimation.close()
    # F_all.close()
