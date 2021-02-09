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
import seg_liver as segmentation
from dataset.dataset_seg import Dataset
from config import Config

def seg_liver_test(config, test_volume_txt, number_slices=3):
    """
        segmentation of liver
    """

    task_name = 'seg_liver_ck'

    ### config constants ###
    database_root = config.database_root
    logs_path = config.get_log(task_name)
    result_root = os.path.join(config.root_folder, 'LiTS_database')
    root_folder = config.root_folder
    ###

    model_name = os.path.join(logs_path, "seg_liver.ckpt")

    test_file = os.path.join(root_folder, testing_volume_txt)

    dataset = Dataset(None, test_file, None, database_root, number_slices, store_memory=False)

    result_path = os.path.join(result_root, task_name)
    checkpoint_path = model_name
    segmentation.test(dataset, checkpoint_path, result_path, number_slices)

if __name__ =='__main__':
    config = Config()
    seg_liver_test(config)
