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

        def get_sorted_dir(dir):
            dirContents = os.listdir(dir)
            dirContents.sort()
            return sorted(dirContents, key=len)


        # ground truths
        images_volumes_fold = os.path.join(self.config.database_root, images_volumes)
        images_volumes_patID = get_sorted_dir(images_volumes_fold)

        # lesions
        item_seg_fold = os.path.join(self.config.database_root, item_seg)
        item_seg_patID = get_sorted_dir(item_seg_fold)

        # liver
        liver_seg_fold = os.path.join(self.config.database_root, liver_seg)
        liver_seg_patID = get_sorted_dir(liver_seg_fold)

        all_paths = []

        for i, patientID in enumerate(images_volumes_patID): # loop through each patient
            if self.config.debug:
                print('(Train Test Split) Sorting patient {}'.format(patientID))
            
            all_patient_data = [[], [], []]

            for j, id in enumerate([images_volumes_patID, item_seg_patID, liver_seg_patID]):
                if id[i] != ".DS_Store":
                    
                    patient_path = os.path.join(images_volumes_fold, images_volumes_patID[i])
                    patient_slices = get_sorted_dir(patient_path)

                    for pat_slice in patient_slices:
                        pat_slice_path = os.path.join(patient_path, pat_slice).split(os.path.sep)[self.config.num_slices:]
                        pat_slice_path = os.path.sep.join(pat_slice_path)
                        all_patient_data[j].append(pat_slice_path)

            all_paths.append(all_patient_data)

        return all_paths[0]


    def get_data_volume(self, lol):

        rows = []

        for pat in range(len(lol)):
            # construct row for given num_slices
            num_patSlices = len(lol[pat][1]) - (self.config.num_slices - 1)
            for patSlice in range(num_patSlices):
                row = []
                for i in range(self.config.num_slices):
                    row.append( str(lol[pat][patSlice+i]) )
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
        num_patients = np.array(lol).shape[0]
        split_point = int(math.floor( num_patients * train_test_split_ratio )) - 1
        training_set = lol[:split_point]
        testing_set = lol[split_point:]

        print('')
        print('# patients in dataset: {}'.format(num_patients))
        print('Splitting on patient {}'.format(split_point))
        print('-'*40)
        print('First row of training set')
        print(training_set[0][0]) # patient 0, row 0
        print('')
        print('First row of testing set')
        print(testing_set[0][0])
        print('-'*40)


        training_volume = self.get_data_volume(training_set)
        testing_volume = self.get_data_volume(testing_set)

        return training_volume, testing_volume