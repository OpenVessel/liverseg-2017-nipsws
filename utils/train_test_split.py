import os
import platform
import pandas as pd
import numpy as np
import math
from pprint import pprint

class TrainTestSplit:
    def __init__(self, config):
        self.config = config

    def sort_list(self, images_volumes, item_seg, liver_seg):
        """
        Arguments
        images_volumes: name of ground truths folder
        item_seg: name of item (lesion) folder
        liver_seg: name of liver folder

        return: list of lists that will be fed into a pandas DataFrame
        """


        # ground truths
        images_volumes_fold = os.path.join(self.config.database_root, images_volumes)
        images_volumes_patID = os.listdir(images_volumes_fold)
        images_volumes_patID.sort()
        images_volumes_patID = sorted(images_volumes_patID, key=len)

        # lesions
        item_seg_fold = os.path.join(self.config.database_root, item_seg)
        item_seg_patID = os.listdir(item_seg_fold)
        item_seg_patID.sort()
        item_seg_patID = sorted(item_seg_patID, key=len)

        # liver
        liver_seg_fold = os.path.join(self.config.database_root, liver_seg)
        liver_seg_patID = os.listdir(liver_seg_fold)
        liver_seg_patID.sort()
        liver_seg_patID = sorted(liver_seg_patID, key=len)

        all_paths = []

        for i in range(len(images_volumes_patID) - 1):

            all_patient_data = [[], [], []]

            for j, id in enumerate([images_volumes_patID, item_seg_patID, liver_seg_patID]):
                if id[i] != ".DS_Store":
                    patient_path = os.path.join(images_volumes_fold, images_volumes_patID[i])
                    patient_slice = os.listdir(patient_path)
                    patient_slice.sort()
                    patient_slice = sorted(patient_slice, key=len)
                    for pat_slice in patient_slice:
                        pat_slice_path = os.path.join(patient_path, pat_slice).split(os.path.sep)[self.config.num_slices:]
                        pat_slice_path = os.path.sep.join(pat_slice_path)
                        all_patient_data[j].append(pat_slice_path)

            all_paths.append(all_patient_data)

        return all_paths


    def get_data_volume(self, lol):

        rows = []

        for pat in range(len(lol)):
            # construct row for given num_slices
            num_patSlices = len(lol[pat][1]) - (self.config.num_slices - 1)
            for patSlice in range(num_patSlices):
                row = []
                for i in range(self.config.num_slices):
                    row_segment = []
                    for j in range(self.config.num_slices):
                        row_segment.append( str(lol[pat][j][patSlice] ) )
                    row.append(" ".join(row_segment))
                rows.append(row)
        return pd.DataFrame(rows, columns=['slice {}'.format(i) for i in range(1, self.config.num_slices+1)])


    def split(self, images_volumes, item_seg, liver_seg, train_test_split_ratio = 0.8):
        """
        Arguments
        self.config.database_root: root folder where the dataset is held 
        images_volumes: name of ground truths folder
        item_seg: name of item (lesion) folder
        liver_seg: name of liver folder

        return: training and testing df for seg_liver_train/test()
        """
        
        lol = self.sort_list(images_volumes, item_seg, liver_seg)
        # split data
        print(np.array(lol)[0][1])
        split_point = math.floor(np.array(lol).shape[-1]*train_test_split_ratio)
        training_volume = self.get_data_volume(lol[split_point:])
        testing_volume = self.get_data_volume(lol[:split_point])

        return training_volume, testing_volume