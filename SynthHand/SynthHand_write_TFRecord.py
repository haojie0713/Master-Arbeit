import tensorflow as tf
import glob
import numpy as np


list_points_noobject = glob.glob('/home/haojie/Desktop/SynthHand/SynthHands_Release/processed_result/female_noobject/points/*.npy')\
    +glob.glob('/home/haojie/Desktop/SynthHand/SynthHands_Release/processed_result/male_noobject/points/*.npy')
# 92536   get 69402 for training
np.random.shuffle(list_points_noobject)
training_1 = list_points_noobject[:69402]
test_1 = list_points_noobject[69402:]

list_points_object = glob.glob('/home/haojie/Desktop/SynthHand/SynthHands_Release/processed_result/female_object/points/*.npy')\
    +glob.glob('/home/haojie/Desktop/SynthHand/SynthHands_Release/processed_result/male_object/points/*.npy')
# 91600   get 22900 for training
np.random.shuffle(list_points_object)
training_2 = list_points_object[:22900]
test_2 = list_points_object[22900:]

list_training = training_1 + training_2
np.random.shuffle(list_training)
list_test = test_1 + test_2
np.random.shuffle(list_test)


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))


list_all = [list_training, list_test]
list_writer = ['/home/haojie/Desktop/SynthHand_training_1', '/home/haojie/Desktop/SynthHand_test_1']
for i in range(2):
    list_points = list_all[i]
    writer = tf.python_io.TFRecordWriter(list_writer[i])
    counter = 0
    for file_points in list_points:
        file_joints = file_points.replace('points', 'joints')
        points = np.load(file_points).astype(np.float32)
        joints = np.load(file_joints).astype(np.float32)

        if (np.mod(counter, 30000) == 0) and (counter != 0):
            writer.close()
            writer = tf.python_io.TFRecordWriter(list_writer[i][:-1]+str(counter//30000+1))
        counter += 1

        if file_points.find('_object') == -1:  # no object
            label = np.array([1.0]).astype(np.float32)
        else:  # with object
            label = np.array([0.0]).astype(np.float32)

        features = {'pointCloud': bytes_feature(points), 'joint': bytes_feature(joints), 'handScale': bytes_feature(np.array([1.0]).astype(np.float32)), 'label': bytes_feature(label)}
        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())
    writer.close()
