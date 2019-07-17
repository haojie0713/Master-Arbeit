import tensorflow as tf
import glob
import numpy as np

dir_points = '/home/haojie/Desktop/hand_object/points/'


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))


writer = tf.python_io.TFRecordWriter('/home/haojie/Desktop/HandObject')
for i in range(1, 2966):
    file = dir_points+'image_D{:08d}.npy'.format(i)
    points = np.load(file).astype(np.float32)

    features = {'pointCloud': bytes_feature(points), 'joint': bytes_feature(np.zeros([21, 3]).astype(np.float32)),
                'handScale': bytes_feature(np.array([1.0]).astype(np.float32))}

    example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(example.SerializeToString())
writer.close()

