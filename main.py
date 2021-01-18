import os
import sys
import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
import seg_liver as segmentation
from dataset.dataset_seg import Dataset

class LiverLesion:
    def __init__(self, config):
        self.config = config
    
    def seg_liver_test(self, number_slices=3):
        task_name = 'seg_liver_ck'

        ### config constants ###
        database_root = self.config.database_root
        logs_path = self.config.get_log(task_name)
        result_root = self.config.get_result_root('LiTS_database')
        root_folder = self.config.root_folder
        ###

        model_name = os.path.join(logs_path, "seg_liver.ckpt")

        test_file = os.path.join(root_folder, 'seg_DatasetList/testing_volume_3.txt')

        dataset = Dataset(None, test_file, None, database_root, number_slices, store_memory=False)

        result_path = os.path.join(result_root, task_name)
        checkpoint_path = model_name
        segmentation.test(dataset, checkpoint_path, result_path, number_slices)

if __name__ =='__main__':
    from config import Config

    config = Config()
    liver_lesion = LiverLesion(config)
    liver_lesion.seg_liver_test()


