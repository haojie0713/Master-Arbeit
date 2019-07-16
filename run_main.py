import argparse
import sys
import matplotlib.pyplot as plt
from six.moves import xrange

from dataset import *
from VAE import *
from loss import *
from evaluation import *
from visualization import *
from visualization_offline import *


def parse_args():
    desc = "Tensorflow implementation of 'Augmented Autoencoder'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate.')

    parser.add_argument('--batch_size', type=int, default=10, help='Batch size.')

    parser.add_argument('--start_step', type=int, default=3989001, help='Training step to start with.')

    parser.add_argument('--max_step', type=int, default=180000000000, help='Number of steps to run trainer.')

    parser.add_argument('--data_dir', type=str, default='/home/haojie/Desktop/MyCode/dataset/', help='Directory where the dataset lies.')

    parser.add_argument('--train_size', type=int, default=949984, help='Training data size (number of images).')

    parser.add_argument('--validation_size', type=int, default=6400, help='Validation data size (number of images).')

    parser.add_argument('--test_size', type=int, default=1000, help='Validation data size (number of images).')  #101641

    parser.add_argument('--log_dir', type=str, default='/home/haojie/Desktop/MyCode/log', help='Directory to put the log data.')

    parser.add_argument('--log_frequency', type=int, default=1000, help='Frequency (steps) with which data is logged')

    parser.add_argument('--run_mode', type=str, default='training', help='Mode in which network is run: visualization, training, cont_training, inference, training_preweights, inference_result')

    FlAGS_, unparsed_ = parser.parse_known_args()
    return FlAGS_, unparsed_


def main(_):
    training_filenames, validation_filenames, test_filenames = dataset_filenames(FLAGS.data_dir)

    # *_* ############# DEFINE THE GRAPH ##############
    handle = tf.placeholder(tf.string, shape=[])
    training_var = tf.placeholder(tf.bool)

    points_occluded, points_clean, joints, object, depth_image, training_iterator, validation_iterator, test_iterator = \
        create_datasets_boxnet(training_filenames, validation_filenames, test_filenames, handle, FLAGS.batch_size, 8, 3000)

    latent_mean, latent_stddev, scores = encoder_rPEL(points_occluded, training_var) # !!!!!!!!!!!!!!!points_occluded => points_clean

    z = re_parameterization(latent_mean, latent_stddev, training_var)

    points_out = decoder_FdNt(z, FLAGS.batch_size, training_var)
    pose_out = decoder_pose(z, training_var)

    loss_kl = KL_divergence(latent_mean, latent_stddev, FLAGS.batch_size)
    loss_point_reconstruction1 = loss_net_point_reconstruction1(points_out, points_clean, FLAGS.batch_size) # EM loss
    loss_point_reconstruction2 = loss_net_point_reconstruction2(points_out, points_clean, FLAGS.batch_size) # Chamfer loss
    loss_point_reconstruction = loss_point_reconstruction1+loss_point_reconstruction2
    tf.summary.scalar('loss point reconstruction', loss_point_reconstruction)
    loss_pose_estimation = loss_net_pose_estimation(pose_out, joints, FLAGS.batch_size)
    euc_joint_distance = pose_loss_for_test_and_validation(pose_out, joints, FLAGS.batch_size) # just for the function inference_validation
    loss_all = 0.001*loss_kl+loss_pose_estimation+loss_point_reconstruction

    tf.summary.scalar('loss all', loss_all)
    op_train = train_net(loss_all, FLAGS.learning_rate)
    # *_* ##############################################

    # Create a session for running operations on the Graph.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.68)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # Create a saver for writing training checkpoints (The 2 most recent are kept).
    var_list = tf.global_variables()
    saver = tf.train.Saver(var_list, max_to_keep=2)

    # Build the summary Tensor based on the TF collection of Summaries.
    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.log_dir + '/eval')

    # Number of steps per epoch
    validation_steps = int(math.ceil(FLAGS.validation_size / FLAGS.batch_size))
    test_steps = int(math.ceil(FLAGS.test_size / FLAGS.batch_size))
    n_training_batch = int(math.ceil(FLAGS.train_size / FLAGS.batch_size))

    # Create the needed string handles to feed the handle placeholder
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())
    test_handle = sess.run(test_iterator.string_handle())

    # Initialize the global variables.
    init = tf.global_variables_initializer()

    # Initialize initializable iterators
    sess.run(validation_iterator.initializer)
    sess.run(test_iterator.initializer)

    if FLAGS.run_mode == 'inference':
        # restore
        print('Model is being restored...')
        checkpoint_file = '/home/haojie/Desktop/MyCode/log/model.ckpt-4003000'
        saver.restore(sess, checkpoint_file)

        # print('**INFERENCE ==> VALIDATION**')
        # inference_validation(sess, loss_kl, loss_point_reconstruction, euc_joint_distance, loss_all, handle, validation_handle, training_var, validation_steps)
        inference_test(sess, euc_joint_distance, handle, test_handle, training_var, test_steps)
        # print('**INFERENCE ==> VALIDATION BATCH**')
        # inference_validation_batch(sess, loss_kl, loss_point_reconstruction, euc_joint_distance, loss_all, handle, validation_handle, training_var, validation_steps)

    elif FLAGS.run_mode == 'inference_result':
        # restore
        print('Model is being restored...')
        checkpoint_file = '/home/haojie/Desktop/MyCode/log'
        saver.restore(sess, checkpoint_file)

        # print('**INFERENCE ==> TEST**')
        # inference_test_result(sess, points_out, pose_out, loss_kl, loss_point_reconstruction, loss_pose_estimation, loss_all, handle, test_handle, training_var, test_steps)
    elif FLAGS.run_mode == 'visualization':
        # restore
        print('Model is being restored...')
        checkpoint_file = '/home/haojie/Desktop/MyCode/log/model.ckpt-4003000'
        saver.restore(sess, checkpoint_file)

        # offline_visualization(sess, points_occluded, points_clean, joints, object, depth_image, points_out, pose_out, scores, handle, validation_handle, training_var, FLAGS.batch_size, 10)
        offline_visualization_FirstPersonHand(sess, points_occluded, joints, points_out, pose_out, scores, handle, test_handle, training_var, FLAGS.batch_size, 100)
        print('Vidualization data are completely generated!')
    else:
        if FLAGS.run_mode == 'training':
            sess.run(init)
        elif FLAGS.run_mode == 'cont_training':
            sess.run(init)

            # restore
            print('Model is being restored...')
            # checkpoint_file = '/home/haojie/Desktop/MyCode/log/model.ckpt-1359000'
            # vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            # vars_filter = tf.contrib.framework.filter_variables(vars, exclude_patterns=['FdNt'])
            # saver1 = tf.train.Saver(var_list=vars_filter, max_to_keep=2)
            # saver1.restore(sess, checkpoint_file)

            checkpoint_file = '/home/haojie/Desktop/MyCode/log/model.ckpt-3989000'
            saver.restore(sess, checkpoint_file)

            # print('**INFERENCE ==> VALIDATION**')
            # inference_validation(sess, loss_kl, loss_point_reconstruction, loss_pose_estimation, loss_all, handle, validation_handle, training_var, validation_steps)

        elif FLAGS.run_mode == 'training_preweight':
            sess.run(init)

            # restore
            print('Model is being restored...')
            checkpoint_file = '/home/haojie/Desktop/MyCode/log/model.ckpt-2693000'
            #################################################################################
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            vars_preweights = tf.contrib.framework.filter_variables(vars, exclude_patterns=['rPEL/FC_layer/outputlayer', 'FC_pose/hiddenlayer0', 'FdNt/FdNt0_0', 'FdNt/FdNt1_0'])
            saver_preweight = tf.train.Saver(var_list=vars_preweights, max_to_keep=2)
            saver_preweight.restore(sess, checkpoint_file)

        #####################################################################
        print('#**#**# START TRAINING #**#**#')
        start_time = time.time()

        for step in xrange(FLAGS.start_step, FLAGS.max_step):
            # print('***TRAINING STEP: %d***' % step)
            if (step != 0) & (step % FLAGS.log_frequency == 0):
                # logging for tensorboard
                summary, _ = sess.run([merged, op_train], feed_dict={handle: training_handle, training_var: True})
                #####################################################################################
                train_writer.add_summary(summary, step)
                train_writer.flush()

                summary = sess.run(merged, feed_dict={handle: validation_handle, training_var: False})
                #####################################################################################
                validation_writer.add_summary(summary, step)
                validation_writer.flush()
            else:
                sess.run(op_train, feed_dict={handle: training_handle, training_var: True})

                # _, a, b, c, d, f = sess.run([op_train, loss_kl, loss_point_reconstruction1, loss_point_reconstruction2, loss_pose_estimation, loss_all], feed_dict={handle: training_handle, training_var: True})
                # print(a, b, c, d, f)

            # if step % n_training_batch == n_training_batch-1:
            #     print('***EVALUATION OF TRAINING***')
            #     # END OF EPOCH
            #     duration = time.time() - start_time
            #     print('****Step %d completed. Duration of last %d steps: %.2f sec' % (step, n_training_batch, duration))
            #     print('**INFERENCE ==> VALIDATION**')
            #     inference_validation(sess, loss_kl, loss_point_reconstruction, euc_joint_distance, loss_all, handle, validation_handle, training_var, validation_steps)

            if (step != 0) & (step % 1000 == 0):
                print('***SAVE A MODEL***')
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)

                # visualize the result
                visualize_result(sess, points_occluded, points_clean, joints, points_out, pose_out, handle, training_handle, training_var, FLAGS.batch_size, step) # !!!!!!!!!!!!!!!points_occluded => points_clean


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    aaa = sys.argv[0]
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)
