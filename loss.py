import tensorflow as tf
from tensorflow.python.framework import ops
import os
from config import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
approxmatch_module = tf.load_op_library(os.path.join(BASE_DIR, 'Earth Mover Distance Dependency/tf_approxmatch_so.so'))
nn_distance_module = tf.load_op_library(os.path.join(BASE_DIR, 'Chamfer Distance Dependency/tf_nndistance_so.so'))


def KL_divergence(latent_mean, latent_stddev, batch_size):
    temp = 0.5 * tf.reduce_sum(tf.square(latent_mean)+tf.square(latent_stddev)-tf.log(tf.square(latent_stddev))-1)/batch_size/2
    tf.summary.scalar('loss KL', temp)
    return temp  # kl loss per sample / 1.0(weight)


def approx_match(xyz1, xyz2):
    '''
    input:
    xyz1 : batch_size * #dataset_points * 3
    xyz2 : batch_size * #query_points * 3
    returns:
    match : batch_size * #query_points * #dataset_points
    '''
    return approxmatch_module.approx_match(xyz1, xyz2)


ops.NoGradient('ApproxMatch')


def match_cost(xyz1, xyz2, match, label):
    '''
    input:
    xyz1 : batch_size * #dataset_points * 3
    xyz2 : batch_size * #query_points * 3
    match : batch_size * #query_points * #dataset_points
    returns:
    cost : batch_size
    '''
    return tf.multiply(approxmatch_module.match_cost(xyz1, xyz2, match), tf.squeeze(label))


@tf.RegisterGradient('MatchCost')
def _match_cost_grad(op, grad_cost):
    xyz1 = op.inputs[0]
    xyz2 = op.inputs[1]
    match = op.inputs[2]
    grad_1, grad_2 = approxmatch_module.match_cost_grad(xyz1, xyz2, match)
    return [grad_1*tf.expand_dims(tf.expand_dims(grad_cost, 1), 2), grad_2*tf.expand_dims(tf.expand_dims(grad_cost, 1), 2), None]


def nn_distance(xyz1, xyz2):
    '''
    Computes the distance of nearest neighbors for a pair of point clouds
    input: xyz1: (batch_size,#points_1,3)  the first point cloud
    input: xyz2: (batch_size,#points_2,3)  the second point cloud
    output: dist1: (batch_size,#point_1)   distance from first to second
    output: idx1:  (batch_size,#point_1)   nearest neighbor from first to second
    output: dist2: (batch_size,#point_2)   distance from second to first
    output: idx2:  (batch_size,#point_2)   nearest neighbor from second to first
    '''
    return nn_distance_module.nn_distance(xyz1, xyz2)


@ops.RegisterGradient('NnDistance')
def _nn_distance_grad(op,grad_dist1,grad_idx1,grad_dist2,grad_idx2):
    xyz1 = op.inputs[0]
    xyz2 = op.inputs[1]
    idx1 = op.outputs[1]
    idx2 = op.outputs[3]
    return nn_distance_module.nn_distance_grad(xyz1, xyz2, grad_dist1, idx1, grad_dist2, idx2)


def loss_net_point_reconstruction1(points_out, points_clean, label):
    """ Approxmiate algorithm for computing the Earth Mover's Distance.

    Original author: Haoqiang Fan
    Modified by Charles R. Qi
    """
    match = approx_match(points_out, points_clean)
    loss = tf.cond(tf.reduce_sum(label)>tf.constant(0, dtype=tf.float32), lambda: tf.reduce_sum(match_cost(points_out, points_clean, match, label))/tf.reduce_sum(label)/OUTPUT_POINT_SIZE, lambda: 1e-10)
    # if tf.reduce_sum(label) == 0:   # writing in this way is completely wrong
    #     loss = 1e-10
    # else:
    #     match = approx_match(points_out, points_clean)
    #     loss = tf.reduce_sum(match_cost(points_out, points_clean, match, label))/tf.reduce_sum(label)/OUTPUT_POINT_SIZE
    tf.summary.scalar('loss point reconstruction (EMD)', loss)
    return loss  # Earth Mover Distance (point euclidean loss)


def loss_net_point_reconstruction2(points_out, points_clean, label):
    """ Compute Chamfer's Distance.

    Original author: Haoqiang Fan.
    Modified by Charles R. Qi
    """
    dist1, _, dist2, _ = nn_distance(points_out, points_clean)
    dist1 = tf.multiply(tf.reduce_sum(dist1, axis=1), tf.squeeze(label))
    dist2 = tf.multiply(tf.reduce_sum(dist2, axis=1), tf.squeeze(label))
    loss = tf.cond(tf.reduce_sum(label)>tf.constant(0, dtype=tf.float32), lambda: (tf.reduce_sum(dist1)+tf.reduce_sum(dist2))/tf.reduce_sum(label)/OUTPUT_POINT_SIZE, lambda: 1e-10)
    # if tf.reduce_sum(label) == 0:    # wrong way
    #     loss = 1e-10
    # else:
    #     dist1, _, dist2, _ = nn_distance(points_out, points_clean)
    #     dist1 = tf.multiply(tf.reduce_sum(dist1, axis=1), tf.squeeze(label))
    #     dist2 = tf.multiply(tf.reduce_sum(dist2, axis=1), tf.squeeze(label))
    #     loss = (tf.reduce_sum(dist1)+tf.reduce_sum(dist2))/tf.reduce_sum(label)/OUTPUT_POINT_SIZE
    tf.summary.scalar('loss point reconstruction (CD)', loss)
    return loss  # Chamfer Distance (point euclidean loss)


def loss_net_pose_estimation(pose_out, poseGT, batch_size):
    poseLoss = tf.nn.l2_loss(pose_out - poseGT)/batch_size
    tf.summary.scalar('loss pose estimation', poseLoss)
    return poseLoss  # point euclidean loss


def pose_loss_for_test_and_validation(pose_out, poseGT, batch_size):
    pose_out = tf.reshape(pose_out, [batch_size, 21, 3])
    poseGT = tf.reshape(poseGT, [batch_size, 21, 3])
    euc_distance = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(pose_out-poseGT), 2)))/batch_size/21.0
    tf.summary.scalar('euclidean joint distance', euc_distance)
    return euc_distance

'''
# test for loss
import numpy as np
import time
xyz1=np.random.randn(32,16384,3).astype('float32')
xyz2=np.random.randn(32,1024,3).astype('float32')
inp1=tf.Variable(xyz1)
inp2=tf.constant(xyz2)

loss = loss_net_point_reconstruction1(inp1, inp2)
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    t1 = time.time()
    a=sess.run(loss)
    t = time.time()-t1
    print(a, t)
'''


