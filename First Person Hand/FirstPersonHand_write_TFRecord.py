import tensorflow as tf
import glob
import numpy as np


dir_points = '/home/haojie/Desktop/FirstPerson/points/'
dir_joints = '/home/haojie/Desktop/FirstPerson/joints/'


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))


list_points = glob.glob('/home/haojie/Desktop/FirstPersonHand/points/*.npy')
# print(len(list_points)) # 101641
np.random.shuffle(list_points)
counter = 0
writer = tf.python_io.TFRecordWriter('/home/haojie/Desktop/FirstPersonHand_1')
for points in list_points:
    joints = points.replace('points', 'joints')
    points = np.load(points).astype(np.float32)
    joints = np.load(joints).astype(np.float32)

    if (np.mod(counter, 30000) == 0) and (counter != 0):
        writer.close()
        writer = tf.python_io.TFRecordWriter('/home/haojie/Desktop/FirstPersonHand_'+str(counter//30000+1))
    counter += 1

    features = {'pointCloud': bytes_feature(points), 'joint': bytes_feature(joints), 'handScale': bytes_feature(np.array([1.0]).astype(np.float32))}
    example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(example.SerializeToString())
    if counter == 1000:
        break
writer.close()
