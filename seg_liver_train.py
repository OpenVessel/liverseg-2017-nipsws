"""
Original code from OSVOS (https://github.com/scaelles/OSVOS-TensorFlow)
Sergi Caelles (scaelles@vision.ee.ethz.ch)

Modified code for liver and lesion segmentation:
Miriam Bellver (miriam.bellver@bsc.es)
"""

import os
import sys
import tensorflow as tf
slim = tf.contrib.slim
import seg_liver as segmentation
from dataset.dataset_seg import Dataset
from config import Config


# gpu_id = 0
# number_slices = 3

# # Training parameters
# batch_size = 1
# iter_mean_grad = 10
# max_training_iters_1 = 15000
# max_training_iters_2 = 30000
# max_training_iters_3 = 50000


# save_step = 2000
# display_step = 2
# ini_learning_rate = 1e-8
# boundaries = [10000, 15000, 25000, 30000, 40000]
# values = [ini_learning_rate, ini_learning_rate * 0.1, ini_learning_rate, ini_learning_rate * 0.1, ini_learning_rate,
#           ini_learning_rate * 0.1]

def seg_liver_train(config, train_df, val_df,
                    gpu_id, number_slices, batch_size, iter_mean_grad, max_training_iters_1,
                    max_training_iters_2, max_training_iters_3, save_step, display_step,
                    ini_learning_rate, boundaries, values):
    """
    train_file: Training DF
    val_file: Testing DF used to evaluate.
    """

    task_name = 'seg_liver'
    # \seg_liver_ck\networks\seg_liver.ckpt
    ### config constants ###
    root_folder = config.root_folder
    database_root = config.database_root
    logs_path = config.get_log('seg_liver')
    imagenet_ckpt = config.imagenet_ckpt
    finetune = config.fine_tune
    trained_weights = config.old_weights
    

    print("finetune", finetune)
    if finetune == 0: 
        print("loading weights path of vgg-16 or resnet",imagenet_ckpt)
        print("logs_path", logs_path)
    else:
        print("trained weights", trained_weights)
    # D:\L_pipe\liver_open\liverseg-2017-nipsws\train_files\seg_liver_ck\networks\seg_liver.ckpt

    # train_file = os.path.join(root_folder, 'seg_DatasetList', 'training_volume_3.txt')
    # val_file = os.path.join(root_folder, 'seg_DatasetList', 'testing_volume_3.txt')

    dataset = Dataset(train_df, None, val_df, database_root, number_slices, store_memory=False)

    # Train the network
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(gpu_id)):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            segmentation.train_seg(dataset, trained_weights, imagenet_ckpt, 1, learning_rate, logs_path, max_training_iters_1, save_step,
                            display_step, global_step, number_slices=number_slices, iter_mean_grad=iter_mean_grad,
                            batch_size=batch_size, resume_training=False, finetune = finetune)

    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(gpu_id)):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            segmentation.train_seg(dataset, trained_weights, imagenet_ckpt, 2, learning_rate, logs_path, max_training_iters_2, save_step,
                            display_step, global_step, number_slices=number_slices, iter_mean_grad=iter_mean_grad,
                            batch_size=batch_size, resume_training=True, finetune = config.fine_tune)

    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(gpu_id)):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            segmentation.train_seg(dataset, trained_weights, imagenet_ckpt, 3, learning_rate, logs_path, max_training_iters_3, save_step,
                            display_step, global_step, number_slices=number_slices, iter_mean_grad=iter_mean_grad,
                            batch_size=batch_size, resume_training=True, finetune = config.fine_tune)
