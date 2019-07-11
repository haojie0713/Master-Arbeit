from layers import *
from utilities import *

GRID_POINTS = tf.constant(grid_points(), dtype=tf.float32)


def encoder_rPEL(points, training_var):

    with tf.variable_scope('rPEL'):
        points = tf.reshape(points, [-1, INPUT_POINT_SIZE, 3, 1])

        net = point_conv(points, num_output_channels=64, kernel_size=[1, 3], scope='pointConv_initial1', bn=True, is_training=training_var)
        for i in range(0, 2):
            scopename = 'pointResConv1' + str(i)
            with tf.variable_scope(scopename):
                net = pointnet_residual_global_module(net, INPUT_POINT_SIZE, 64, scope=scopename, blocks=4, is_training=training_var)

        net = point_conv(net, num_output_channels=256, kernel_size=[1, 1], scope='pointConv_initial2', bn=True, is_training=training_var)
        for i in range(0, 2):
            scopename = 'pointResConv2' + str(i)
            with tf.variable_scope(scopename):
                net = pointnet_residual_global_module(net, INPUT_POINT_SIZE, 256, scope=scopename, blocks=4, is_training=training_var)

        net = point_conv(net, num_output_channels=1024, kernel_size=[1, 1], scope='pointConv_initial3', bn=True, is_training=training_var)
        for i in range(0, 2):
            scopename = 'pointResConv3' + str(i)
            with tf.variable_scope(scopename):
                net = pointnet_residual_global_module(net, INPUT_POINT_SIZE, 1024, scope=scopename, blocks=4, is_training=training_var)

        with tf.variable_scope('pointConvScore'):
            scores = point_conv(net, num_output_channels=512, kernel_size=[1, 1], scope='pointConvPoseScore0', bn=True, is_training=training_var)
            scores = point_conv(scores, num_output_channels=256, kernel_size=[1, 1], scope='pointConvPoseScore1', bn=False, activation_fn=tf.sigmoid)

        with tf.variable_scope('pointConvValue'):
            values = point_conv(net, num_output_channels=512, kernel_size=[1, 1], scope='pointConvPoseValue0', bn=True, is_training=training_var)
            values = point_conv(values, num_output_channels=256, kernel_size=[1, 1], scope='pointConvPoseValue1', bn=False, activation_fn=None)

        scores = scores + 1e-7
        scoresMax = tf.reshape(tf.reduce_max(scores, axis=1), [-1, 1, 1, 256])
        scores = tf.div(scores, scoresMax)
        weightSum = tf.squeeze(tf.reduce_sum(scores, axis=1))
        pre_latent_vars = tf.div(tf.squeeze(tf.reduce_sum(tf.multiply(scores, values), axis=1)), weightSum)  # shape = (BS, 256)

        # Fully Connected Layer
        with tf.variable_scope('FC_layer'):

            # first hidden layer
            h = tf.layers.dense(tf.reshape(pre_latent_vars, [-1, 256]), units=256, activation=tf.nn.relu, name='hiddenlayer0')
            h = tf.layers.batch_normalization(h, axis=1, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, center=True, scale=True, renorm=True, training=training_var, fused=True)

            # second hidden layer
            h = tf.layers.dense(h, units=1024, activation=tf.nn.relu, name='hiddenlayer1')
            h = tf.layers.batch_normalization(h, axis=1, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, center=True, scale=True, renorm=True, training=training_var, fused=True)

            # third hidden layer
            h = tf.layers.dense(h, units=512, activation=tf.nn.relu, name='hiddenlayer2')
            h = tf.layers.batch_normalization(h, axis=1, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, center=True, scale=True, renorm=True, training=training_var, fused=True)

            # fourth hidden layer
            h = tf.layers.dense(h, units=256, activation=tf.nn.relu, name='hiddenlayer3')
            h = tf.layers.batch_normalization(h, axis=1, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, center=True, scale=True, renorm=True, training=training_var, fused=True)

            # output player
            latent_vars = tf.layers.dense(h, activation=None, units=dim_z*2, name='outputlayer')

        latent_mean = latent_vars[:, :dim_z]
        latent_stddev = 1e-7 + tf.nn.softplus(latent_vars[:, dim_z:])
        tf.summary.image('latent mean', tf.reshape(latent_mean, [-1, dim_z, 1, 1]), max_outputs=1)
        tf.summary.image('latent stddev', tf.reshape(latent_stddev, [-1, dim_z, 1, 1]), max_outputs=1)
        tf.summary.histogram('latent mean', latent_mean)
        tf.summary.histogram('latent stddev', latent_stddev)
    return latent_mean, latent_stddev, scores  # shape = (BS, dim_z)


def decoder_FdNt(z, batch_size, training_var):

    with tf.variable_scope('FdNt'):
        grid_points_expand_dims = tf.reshape(GRID_POINTS, [1, OUTPUT_POINT_SIZE, 1, 2])
        grid_points_repeat = tf.tile(grid_points_expand_dims, [batch_size, 1, 1, 1])
        z_expand_dims = tf.reshape(z, [-1, 1, 1, dim_z])
        z_repeat = tf.tile(z_expand_dims, [1, OUTPUT_POINT_SIZE, 1, 1])

        # concatenate
        folding_1st_input = tf.concat([z_repeat, grid_points_repeat], 3)

        # 4 layer perceptron x1
        folding_1st_MLP0 = point_conv(folding_1st_input, num_output_channels=256, kernel_size=[1, 1], scope='FdNt0_0', bn=True, is_training=training_var, activation_fn=tf.nn.relu)
        folding_1st_MLP1 = point_conv(folding_1st_MLP0, num_output_channels=1024, kernel_size=[1, 1], scope='FdNt0_1', bn=True, is_training=training_var, activation_fn=tf.nn.relu)
        folding_1st_MLP2 = point_conv(folding_1st_MLP1, num_output_channels=512, kernel_size=[1, 1], scope='FdNt0_2', bn=True, is_training=training_var, activation_fn=tf.nn.relu)
        folding_1st_MLP3 = point_conv(folding_1st_MLP2, num_output_channels=256, kernel_size=[1, 1], scope='FdNt0_3', bn=True, is_training=training_var, activation_fn=tf.nn.relu)
        folding_1st_MLP4 = point_conv(folding_1st_MLP3, num_output_channels=3, kernel_size=[1, 1], scope='FdNt0_4', bn=False, activation_fn=None)

        # concatenate
        folding_2nd_input = tf.concat([z_repeat, folding_1st_MLP4], 3)

        # 4 layer perceptron x2
        folding_2nd_MLP0 = point_conv(folding_2nd_input, num_output_channels=256, kernel_size=[1, 1], scope='FdNt1_0', bn=True, is_training=training_var, activation_fn=tf.nn.relu)
        folding_2nd_MLP1 = point_conv(folding_2nd_MLP0, num_output_channels=1024, kernel_size=[1, 1], scope='FdNt1_1', bn=True, is_training=training_var, activation_fn=tf.nn.relu)
        folding_2nd_MLP2 = point_conv(folding_2nd_MLP1, num_output_channels=512, kernel_size=[1, 1], scope='FdNt1_2', bn=True, is_training=training_var, activation_fn=tf.nn.relu)
        folding_2nd_MLP3 = point_conv(folding_2nd_MLP2, num_output_channels=256, kernel_size=[1, 1], scope='FdNt1_3', bn=True, is_training=training_var, activation_fn=tf.nn.relu)
        folding_2nd_MLP4 = point_conv(folding_2nd_MLP3, num_output_channels=3, kernel_size=[1, 1], scope='FdNt1_4', bn=False, activation_fn=None)

    return tf.squeeze(folding_2nd_MLP4)  # shape = (BS, OUTPUT_POINT_SIZE, 3)


def decoder_pose(z, training_var):

    with tf.variable_scope('FC_pose'):
        # first hidden layer
        h = tf.layers.dense(z, units=128, activation=tf.nn.relu, name='hiddenlayer0')
        h = tf.layers.batch_normalization(h, axis=1, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, center=True, scale=True, renorm=True, training=training_var, fused=True)

        # second hidden layer
        h = tf.layers.dense(h, units=1024, activation=tf.nn.relu, name='hiddenlayer1')
        h = tf.layers.batch_normalization(h, axis=1, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, center=True, scale=True, renorm=True, training=training_var, fused=True)

        # third hidden layer
        h = tf.layers.dense(h, units=512, activation=tf.nn.relu, name='hiddenlayer2')
        h = tf.layers.batch_normalization(h, axis=1, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, center=True, scale=True, renorm=True, training=training_var, fused=True)

        # fourth hidden layer
        h = tf.layers.dense(h, units=256, activation=tf.nn.relu, name='hiddenlayer3')
        h = tf.layers.batch_normalization(h, axis=1, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, center=True, scale=True, renorm=True, training=training_var, fused=True)

        # output player
        pose_out = tf.layers.dense(h, units=63, activation=None, name='outputlayer')

        return pose_out  # shape = (BS, 63)


def re_parameterization(latent_mean, latent_stddev, training_var):
    if training_var==True:
        # sampling by re-parameterization technique
        z = latent_mean + latent_stddev * tf.random_normal(tf.shape(latent_mean), 0, 1, dtype=tf.float32)
    else:
        z = latent_mean
    return z


def train_net(loss, learning_rate):
    """Sets up the training operations."""
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):

        # Create the Adam optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)

        op_train = optimizer.minimize(loss, global_step=global_step)
    return op_train

