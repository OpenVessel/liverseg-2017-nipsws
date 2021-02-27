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
import numpy as np
import det_lesion as detection
from dataset.dataset_det import Dataset
from config import Config

def det_lesion_test(config):

    gpu_id = 0

    task_name = 'det_lesion_ck'

    ### config constants ###
    database_root = config.database_root
    logs_path = config.get_log(task_name)
    result_root = config.get_result_root('detection_results/')
    root_folder = config.root_folder
    ###

    model_name = os.path.join(logs_path, "det_lesion.ckpt")
    #liver_open\liverseg-2017-nipsws\det_DatasetList\testing_negative_det_patches_test_patches_dummy.txt
    #liver_open\liverseg-2017-nipsws\det_DatasetList\testing_positive_det_patches_test_patches_dummy.txt
    val_file_pos = os.path.join(root_folder, 'det_DatasetList', 'testing_positive_det_patches_test_patches_OV.txt')
    val_file_neg = os.path.join(root_folder, 'det_DatasetList', 'testing_negative_det_patches_test_patches_OV.txt')

#D:\L_pipe\liver_open\liverseg-2017-nipsws\det_DatasetList\testing_positive_det_patches_test_patches_OVxt
    # val_file_pos = os.path.join(root_folder, 'det_DatasetList', 'testing_positive_det_patches.txt')
    # val_file_neg = os.path.join(root_folder, 'det_DatasetList', 'testing_negative_det_patches.txt')

    dataset = Dataset(None, None, val_file_pos, val_file_neg, None, database_root, store_memory=False)

    result_path = os.path.join(result_root, task_name)
    checkpoint_path = model_name
    detection.validate(dataset, checkpoint_path, result_path, number_slices=1)

    """For testing dataset without labels
    # test_file_pos = os.path.join(root_folder, 'det_DatasetList', 'testing_det_patches.txt')
    # dataset = Dataset(None, None, test_file_pos, None, None, database_root, store_memory=False)
    # detection.test(dataset, checkpoint_path, result_path, number_slices=1, volume=False)
    """

if __name__ =='__main__':
    from config import Config

    config = Config()
    det_lesion_test(config)