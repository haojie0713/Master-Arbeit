import tensorflow as tf
import glob
import numpy as np


dir_points = '/home/haojie/Desktop/EgoDexter/data/Kitchen/points/'
# desk 515 fruits 343 kitchen 251 rotunda 369

label = np.load('/home/haojie/Desktop/EgoDexter/data/Kitchen/label.npy')
indices = np.where(np.sum(label, axis=1)!=0)[0]
print(len(indices))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))


writer = tf.python_io.TFRecordWriter('/home/haojie/Desktop/kitchen')
for i in indices:
    points = np.load(dir_points+str(i)+'.npy').astype(np.float32)
    joints = np.zeros([21, 3]).astype(np.float32)
    label = np.array([0.0]).astype(np.float32)
    features = {'pointCloud': bytes_feature(points), 'joint': bytes_feature(joints),
                'handScale': bytes_feature(np.array([1.0]).astype(np.float32)), 'label': bytes_feature(label)}
    example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(example.SerializeToString())
writer.close()
