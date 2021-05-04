import pandas as pd
import os
import math


class DetTestTrain():
    def __init__(self, config, crops_df):
        self.config = config
        self.labels = config.labels
        self.crops = crops_df

    # bb_images_volumes_alldatabase3_gt_nozoom_common_bb/0/46.mat 
    # bb_liver_lesion_seg_alldatabase3_gt_nozoom_common_bb/0/46.png 
    # bb_liver_seg_alldatabase3_gt_nozoom_common_bb/0/46.png 
    
    # bb_images_volumes_alldatabase3_gt_nozoom_common_bb/0/47.mat 
    # bb_liver_lesion_seg_alldatabase3_gt_nozoom_common_bb/0/47.png 
    # bb_liver_seg_alldatabase3_gt_nozoom_common_bb/0/47.png 
    
    # bb_images_volumes_alldatabase3_gt_nozoom_common_bb/0/48.mat 
    # bb_liver_lesion_seg_alldatabase3_gt_nozoom_common_bb/0/48.png 
    # bb_liver_seg_alldatabase3_gt_nozoom_common_bb/0/48.png 
    # 0.000132 0.000132

    def sort_list(self, images_volumes, lesion_seg, liver_seg, liver_results):
        def get_sorted_dir(dir):
            dirContents = os.listdir(dir)
            dirContents.sort()
            dirContents = sorted(dirContents, key=len)
            return list(filter(lambda x: x != ".DS_Store", dirContents))

        crops_files = []

        for index, row in self.crops.iterrows():
            if row["is_liver"]:
                crops_files.append(row["liver_seg"])

        # ground truths
        images_volumes_fold = os.path.join(self.config.database_root, images_volumes)
        patient_ids = get_sorted_dir(images_volumes_fold)

        patients = []

        for patient_id in patient_ids:

            patient_fold = os.path.join(images_volumes_fold, patient_id)
            num_patient_slices = os.listdir(patient_fold)

            mats = []
            items = []
            livers = []
            list_of_id = []
            res = []
            
            for string_id in num_patient_slices:
                if string_id.endswith('.mat'):
                        string_id = string_id[:-4]
                list_of_id.append(int(string_id))
                list_of_id.sort()

            for i in range(len(num_patient_slices)):
                string_id_file = list_of_id[i]
                file_id = os.path.join(patient_id, str(string_id_file)) ## file won't always be a number
                if file_id in crops_files:
                    mats.append(os.path.join(images_volumes, file_id + '.mat')) 
                    if self.labels:
                        items.append(os.path.join(lesion_seg, file_id + '.png')) 
                        livers.append(os.path.join(liver_seg, file_id + '.png'))
                        res.append(os.path.join(liver_results, file_id + ".png"))

            patients.append([mats, items, livers, res])
            list_of_id = []
        return patients


    def get_data_volume(self, lol):

        rows = []

        for pat in range(len(lol)):
            # construct row for given num_slices
            num_pat_rows = len(lol[pat][0]) - self.config.num_slices
            for i in range(0, num_pat_rows + 1):
                row = []
                mats = lol[pat][0]
                if self.labels:
                    items = lol[pat][1]
                    livers = lol[pat][2]

                for j in range(0, self.config.num_slices):
                    row.extend([mats[j + i]]) # 1.mat 
                    if self.labels:
                        row.extend([items[j + i], livers[j + i] ]) # 1.itempng 1.liverpng
                rows.append(row)

        return pd.DataFrame(rows)


    def split(self, images_volumes, lesion_seg, liver_seg, liver_results, train_test_split_ratio = 0.8):
        """
        Arguments
        self.config.database_root: root folder where the dataset is held 
        images_volumes: name of ground truths folder
        item_seg: name of item (lesion) folder
        liver_seg: name of liver folder

        return: training and testing df for seg_liver_train/test()
        """
        
        lol = self.sort_list(images_volumes, lesion_seg, liver_seg, liver_results)


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

        return lesion_train_no_backprop, lesion_test_no_backprop