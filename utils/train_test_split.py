import os
import platform
import pandas as pd
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

        [
            [
                [images1.mat, images2.mat], [item1.png, item2.png], [liver1.png, liver2.png]
            ], 
            [..]
        ]
        
        """

        def get_sorted_dir(dir):
            dirContents = os.listdir(dir)
            dirContents.sort()
            return sorted(dirContents, key=len)

        # ground truths
        images_volumes_fold = os.path.join(self.config.database_root, images_volumes)
        patient_ids = get_sorted_dir(images_volumes_fold)

        patients = []

        for patient_id in patient_ids:

            patient_fold = os.path.join(images_volumes_fold, patient_id)
            num_patient_slices = len(os.listdir(patient_fold))

            mats = []
            items = []
            livers = []

            for i in range(1, num_patient_slices+1):
                file_id = os.path.join(patient_id, str(i))
                mats.append(os.path.join(images_volumes, file_id + '.mat'))
                items.append(os.path.join(item_seg, file_id + '.png'))
                livers.append(os.path.join(liver_seg, file_id + '.png'))

            patients.append([mats, items, livers])

        return patients


    def get_data_volume(self, lol):

        rows = []

        # PatientID-105 mat1 item1 liver1 mat2 item2 liver 2 ... matn itemn livern  3 * num_slices
        # PatientID-105 

        for pat in range(len(lol)):
            # construct row for given num_slices
            num_pat_rows = len(lol[pat][0]) - self.config.num_slices
            for i in range(0, num_pat_rows + 1):
                row = []
                mats = lol[pat][0]
                items = lol[pat][1]
                livers = lol[pat][2]

                for j in range(0, self.config.num_slices):
                    row.extend([ mats[j + i], items[j + i], livers[j + i] ]) # 1.mat 1.itempng 1.liverpng
                rows.append(row)

        return pd.DataFrame(rows)


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
        num_patients = len(lol)
        split_point = int(math.floor( num_patients * train_test_split_ratio ))
        training_set = lol[:split_point]
        testing_set = lol[split_point:]

        print('')
        print('# patients in dataset: {}'.format(num_patients))
        print('Splitting on patient {}'.format(split_point))
        print('-'*40)
        print('First row of training set patient 0')
        print(training_set[0][0][0]) # patient 0, row 0
        print('')
        print('First row of testing set patient 0')
        print(testing_set[0][0][0])
        print('-'*40)


        training_volume = self.get_data_volume(training_set)
        testing_volume = self.get_data_volume(testing_set)

        return training_volume, testing_volume