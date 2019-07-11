import tensorflow as tf
import math

from config import *


def point_conv(inputs, num_output_channels, kernel_size, scope, stride=[1, 1], padding='VALID', bn=True,
               is_training=True, activation_fn=tf.nn.relu):
    """ 2D convolution with non-linear operation.
    Args:
      inputs: 4-D tensor variable BxHxWxC
      num_output_channels: int
      kernel_size: a list of 2 ints
      scope: string
      stride: a list of 2 ints
      padding: 'SAME' or 'VALID'
      activation_fn: function
      bn: bool, whether to use batch norm
      is_training: bool Tensor variable
    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope):
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w, num_in_channels, num_output_channels]

        kernel = tf.Variable(tf.random_uniform(kernel_shape, minval=-math.sqrt(6/num_in_channels), maxval=math.sqrt(6/num_in_channels)), name='weights')
        stride_h, stride_w = stride

        outputs = tf.nn.conv2d(inputs, kernel, [1, stride_h, stride_w, 1], padding=padding)

        biases = tf.get_variable('biases', shape=[num_output_channels], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = tf.layers.batch_normalization(outputs, axis=3, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON,
                                    center=True, scale=True, renorm=True, training=is_training, fused=True)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def pointnet_residual_global_module(inputs, point_size, num_output_dims, scope, blocks=4, is_training=True):
    """Creates one layer of blocks for the ResNet model, for pointnet layers.
    Args:
        inputs: Image tensor of size [batch,dim]
        filters: Number of filters for the first convolution of the layer
        blocks: Number of blocks contained in the layer
        strides: Stride to use for the first convolution of the layer
        training_var: Boolean indicating if training (True) or inference (False)
        data_format: channels_last or channels_first
    Returns:
        Output tensor of the residual module
    """
    with tf.variable_scope(scope):
        shortcut = inputs
        output = inputs
        for i in range(0, blocks):
            with tf.variable_scope('block' + str(i)):
                maxNet = tf.layers.max_pooling1d(tf.reshape(output, [-1, point_size, num_output_dims]), point_size, 1, name='maxpool')
                lambd = tf.Variable(tf.ones([num_output_dims]), name='lambd')
                gamma = tf.Variable(tf.ones([num_output_dims]), name='gamma')
                output = tf.multiply(lambd, output) - tf.multiply(gamma, tf.reshape(maxNet, [-1, 1, 1, num_output_dims]))
                output = point_conv(output, num_output_channels=num_output_dims, kernel_size=[1, 1], scope='block'+str(i), bn=True, is_training=is_training)
        output = output + shortcut
        return output