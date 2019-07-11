import tensorflow as tf
from dataprocessing import *


def dataset_filenames(data_dir):
    training_filenames = ["Hands2017TrainPointCloud_1",
                          "Hands2017TrainPointCloud_2",
                          "Hands2017TrainPointCloud_3",
                          "Hands2017TrainPointCloud_4",
                          "Hands2017TrainPointCloud_5",
                          "Hands2017TrainPointCloud_6",
                          "Hands2017TrainPointCloud_7",
                          "Hands2017TrainPointCloud_8",
                          "Hands2017TrainPointCloud_9",
                          "Hands2017TrainPointCloud_10",
                          "Hands2017TrainPointCloud_11",
                          "Hands2017TrainPointCloud_12",
                          "Hands2017TrainPointCloud_13",
                          "Hands2017TrainPointCloud_14",
                          "Hands2017TrainPointCloud_15",
                          "Hands2017TrainPointCloud_16",
                          "Hands2017TrainPointCloud_17",
                          "Hands2017TrainPointCloud_18",
                          "Hands2017TrainPointCloud_19",
                          "Hands2017TrainPointCloud_20",
                          "Hands2017TrainPointCloud_21",
                          "Hands2017TrainPointCloud_22",
                          "Hands2017TrainPointCloud_23",
                          "Hands2017TrainPointCloud_24",
                          "Hands2017TrainPointCloud_25",
                          "Hands2017TrainPointCloud_26",
                          "Hands2017TrainPointCloud_27",
                          "Hands2017TrainPointCloud_28",
                          "Hands2017TrainPointCloud_29",
                          "Hands2017TrainPointCloud_30",
                          "Hands2017TrainPointCloud_31",
                          "Hands2017TrainPointCloud_32"
                          ]
    validation_filenames = ["Hands2017TrainPointCloud_30",
                            "Hands2017TrainPointCloud_31"]
    test_filenames = ["FirstPersonHand_1"]
    # test_filenames = ["FirstPersonHand_1",
    #                   "FirstPersonHand_2",
    #                   "FirstPersonHand_3",
    #                   "FirstPersonHand_4"]
    training_filenames = [data_dir + s for s in training_filenames]
    validation_filenames = [data_dir + s for s in validation_filenames]
    test_filenames = [data_dir + s for s in test_filenames]
    return training_filenames, validation_filenames, test_filenames


def create_datasets_boxnet(training_filenames, validation_filenames, test_filenames, handle, batch_size, thread_count, buffer_count):
    # Define the training with Dataset API
    training_dataset = tf.data.TFRecordDataset(training_filenames)
    training_dataset = training_dataset.map(parse_function_training, num_parallel_calls=thread_count)
    training_dataset = training_dataset.shuffle(buffer_size=buffer_count)
    training_dataset = training_dataset.batch(batch_size)
    training_dataset = training_dataset.repeat()

    # Define the validation dataset for evaluation
    validation_dataset = tf.data.TFRecordDataset(validation_filenames)
    validation_dataset = validation_dataset.map(parse_function_validation, num_parallel_calls=thread_count)
    validation_dataset = validation_dataset.batch(batch_size)
    validation_dataset = validation_dataset.repeat()

    # Define the evaluation on training dataset dataset
    test_dataset = tf.data.TFRecordDataset(test_filenames)
    test_dataset = test_dataset.map(parse_function_test, num_parallel_calls=thread_count)
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.repeat()

    # Create a feedable iterator to consume data
    iterator = tf.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)
    next_occluded_points, next_clean_points, next_joints, next_scales, next_object, next_depth_image = iterator.get_next()

    # Define the different iterators
    training_iterator = training_dataset.make_one_shot_iterator()
    validation_iterator = validation_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    return next_occluded_points, next_clean_points, next_joints, next_object, next_depth_image, training_iterator, validation_iterator, test_iterator


def parse_function_training(example_proto):
    """Parse through current binary batch and extract images and labels"""
    # Parse through features and extract byte string
    parsed_features = tf.parse_single_example(example_proto, features={
        'pointCloud': tf.FixedLenFeature([], tf.string),
        'joint': tf.FixedLenFeature([], tf.string),
        'handScale': tf.FixedLenFeature([], tf.string)
        }, name='features')

    # Decode content into correct types
    points_dec = tf.decode_raw(parsed_features['pointCloud'], tf.float32)
    points_dec = tf.reshape(points_dec, [6000, 3])
    joint_dec = tf.decode_raw(parsed_features['joint'], tf.float32)
    handScale_dec = tf.decode_raw(parsed_features['handScale'], tf.float32)

    # preprocess points
    points_occluded_dec, points_clean_dec, joint_dec, handScale_dec, object_dec, depthimage_dec = tf.py_func(preprocessPoint_training, [points_dec, joint_dec, handScale_dec], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
    return points_occluded_dec, points_clean_dec, joint_dec, handScale_dec, object_dec, depthimage_dec


def parse_function_validation(example_proto):
    """Parse through current binary batch and extract images and labels"""
    # Parse through features and extract byte string
    parsed_features = tf.parse_single_example(example_proto, features={
        'pointCloud': tf.FixedLenFeature([], tf.string),
        'joint': tf.FixedLenFeature([], tf.string),
        'handScale': tf.FixedLenFeature([], tf.string)
        }, name='features')

    # Decode content into correct types
    points_dec = tf.decode_raw(parsed_features['pointCloud'], tf.float32)
    points_dec = tf.reshape(points_dec, [6000, 3])
    joint_dec = tf.decode_raw(parsed_features['joint'], tf.float32)
    handScale_dec = tf.decode_raw(parsed_features['handScale'], tf.float32)

    # preprocess points
    points_occluded_dec, points_clean_dec, joint_dec, handScale_dec, object_dec, depthimage_dec = tf.py_func(preprocessPoint_validation, [points_dec, joint_dec, handScale_dec], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
    return points_occluded_dec, points_clean_dec, joint_dec, handScale_dec, object_dec, depthimage_dec


def parse_function_test(example_proto):
    """Parse through current binary batch and extract images and labels"""
    # Parse through features and extract byte string
    parsed_features = tf.parse_single_example(example_proto, features={
        'pointCloud': tf.FixedLenFeature([], tf.string),
        'joint': tf.FixedLenFeature([], tf.string),
        'handScale': tf.FixedLenFeature([], tf.string)
        }, name='features')

    # Decode content into correct types
    points_dec = tf.decode_raw(parsed_features['pointCloud'], tf.float32)
    points_dec = tf.reshape(points_dec, [6000, 3])
    joint_dec = tf.decode_raw(parsed_features['joint'], tf.float32)
    handScale_dec = tf.decode_raw(parsed_features['handScale'], tf.float32)

    # preprocess points
    points_occluded_dec, joint_dec = tf.py_func(preprocessPoint_test, [points_dec, joint_dec], [tf.float32, tf.float32])
    return points_occluded_dec, tf.zeros([INPUT_POINT_SIZE, 3]), joint_dec, handScale_dec, tf.zeros([INPUT_POINT_SIZE, 3]), tf.zeros([80, 80])


