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

def det_lesion_test(config, val_file_pos, val_file_neg):
    '''
    val_file_pos and val_file_neg are the validation set
    '''

    gpu_id = 0

    task_name = 'det_lesion_ck'

    ### config constants ###
    database_root = config.database_root
    logs_path = config.get_log(task_name)
    result_root = os.path.join(config.root_folder, 'detection_results/')
    root_folder = config.root_folder
    ###

    model_name = os.path.join(logs_path, "det_lesion.ckpt")

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