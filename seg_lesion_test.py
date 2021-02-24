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
import seg_lesion as segmentation
from dataset.dataset_seg import Dataset
import utils.crop_to_image
import utils.mask_with_liver
import utils.det_filter

def seg_lesion_test(config, number_slices=3):

    gpu_id = 0
    number_slices = 3

    #crops_list = 'crops_LiTS_gt.txt'
    #crops_predict_gt.txt
    det_results_list = 'detection_lesion_example'

    task_name = 'seg_lesion_ck'

    ### config constants ###
    database_root = config.database_root
    ##
    logs_path = config.get_log(task_name)
    result_root = config.get_result_root('results')
    root_folder = config.root_folder ### root folder?
    crops_list = config.crops_list

    #seg_liver_ck
    liver_results_path = os.path.join(database_root, 'seg_liver_ck')
    model_name = os.path.join(logs_path, "seg_lesion.ckpt")
    #test
    test_file = os.path.join(root_folder, 'seg_DatasetList', 'testing_volume_3_crops.txt')
    dataset = Dataset(None, test_file, None, database_root, number_slices, store_memory=False)

    result_path = os.path.join(result_root, task_name)
    checkpoint_path = model_name

    # 1. Does inference with the lesion segmentation network
    #testing_volume_105_OV.txt
    print("segmenting lesions")
    segmentation.test(dataset, checkpoint_path, result_path, number_slices) 
        # O -> results/seg_lesion_ck/.png


    # 2. Returns results to the original size (from cropped slices to 512x512)
    ## 'crops_LiTS_gt.txt'
    ### D:\L_pipe\liver_open\liverseg-2017-nipsws\utils\crops_list\crops_predict_gt.tx
    ### I ->  
    print("uncropping results")
    utils.crop_to_image.crop(
        base_root=root_folder, 
        input_config=task_name, 
        crops_list=crops_list)
        # O -> results/out_seg_lesion_ck/.png


    # 3. Masks the results with the liver segmentation masks
    print("Masking results")
    utils.mask_with_liver.mask(
        base_root=root_folder, 
        labels_path=liver_results_path, 
        input_config='out_' + task_name, 
        th=0.5)
        # O -> masked_out_seg_lesion_ck

    # 4. Checks positive detections of lesions in the liver. Remove those false positive of the segmentation network using the detection results.
    print("filtering results")
    utils.det_filter.filter(
        base_root=root_folder, 
        crops_list=crops_list, 
        input_config='masked_out_' + task_name,
        results_list=det_results_list, 
        th=0.33)
        # O -> results/det_masked_out_seg_lesion_ck/.png


if __name__ == '__main__':
    from config import Config

    config = Config()
    seg_lesion_test(config)

