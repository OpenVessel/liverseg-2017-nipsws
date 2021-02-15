import os
import tensorflow as tf

from seg_liver_test import seg_liver_test
from seg_liver_train import seg_liver_train

from utils.crops_methods.compute_3D_bbs_from_gt_liver import compute_3D_bbs_from_gt_liver
from utils.sampling_bb.sample_bbs import sample_bbs_test, sample_bbs_train

from det_lesion_test import det_lesion_test
from det_lesion_train import det_lesion_train

from seg_lesion_test import seg_lesion_test
from seg_lesion_train import seg_lesion_train

import time



class LiverLesion:
    def __init__(self, config):
        self.config = config
        self.number_slices = 3

        # Training parameters
        self.gpu_id = 0
        
        self.batch_size = 1
        self.iter_mean_grad = 10
        self.max_training_iters_1 = 15000
        self.max_training_iters_2 = 30000
        self.max_training_iters_3 = 50000


        self.save_step = 2000
        self.display_step = 2
        self.ini_learning_rate = 1e-8
        self.boundaries = [10000, 15000, 25000, 30000, 40000]
        self.values = [self.ini_learning_rate, self.ini_learning_rate * 0.1, self.ini_learning_rate, 
                    self.ini_learning_rate * 0.1, self.ini_learning_rate, self.ini_learning_rate * 0.1]

        # only for det_lesion_train
        self.det_batch_size = 16#64
        self.det_iter_mean_grad = 1
        self.det_max_training_iters = 5000

        self.det_save_step = 200
        self.det_learning_rate = 0.01



    def seg_liver_test(self):
        seg_liver_test(self.config, self.number_slices)
    
    def seg_liver_train(self):
        seg_liver_train(self.config, self.gpu_id, self.number_slices, self.batch_size, self.iter_mean_grad, 
                    self.max_training_iters_1, self.max_training_iters_2, self.max_training_iters_3, 
                    self.save_step, self.display_step, self.ini_learning_rate, self.boundaries, self.values)
    

    def compute_3D_bbs_from_gt_liver(self):
        compute_3D_bbs_from_gt_liver(self.config)
    

    def sample_bbs(self):
        liver_masks_path = os.path.join(self.config.database_root, 'liver_seg')
        lesion_masks_path = os.path.join(self.config.database_root, 'item_seg')
        output_folder_path =  './det_DatasetList/'
        crops_list_sp = './utils/crops_list/crops_LiTS_gt_2.txt'
        #crops_list_sp = '../crops_list/crops_LiTS_gt.txt'
        output_file_name_sp = 'test_patches'

        data_aug_options_sp = 8
        sample_bbs_train(crops_list_sp, output_file_name_sp, data_aug_options_sp, liver_masks_path, lesion_masks_path, output_folder_path)
        sample_bbs_test(crops_list_sp, output_file_name_sp, liver_masks_path, lesion_masks_path, output_folder_path)


    def det_lesion_test(self):
        det_lesion_test(self.config)

    def det_lesion_train(self):
        det_lesion_train(self.config, self.gpu_id, self.det_batch_size, self.det_iter_mean_grad, self.det_max_training_iters, 
                        self.det_save_step, self.display_step, self.det_learning_rate)

    def seg_lesion_test(self):
        seg_lesion_test(self.config, self.number_slices)
    
    def seg_lesion_train(self):
        seg_lesion_train(self.config, self.gpu_id, self.number_slices, self.batch_size, self.iter_mean_grad,
                        self.max_training_iters_1, self.max_training_iters_2, self.max_training_iters_3, self.save_step,
                        self.display_step, self.ini_learning_rate, self.boundaries, self.values)


    def test(self):
        """
            Driver code for testing the model.
        """
        
        test_steps = [
            #['seg_liver_test', self.seg_liver_test], ## seg_liver_test.py
            #['compute_bbs_from_gt_liver', self.compute_3D_bbs_from_gt_liver], ## compute_3D_bbs_from_gt_liver.py
            #['sample_bbs', self.sample_bbs], ### sample_bbs.py
            #['det_lesion_test', self.det_lesion_test], ### det_lesion_test.py
            ['seg_lesion_test', self.seg_lesion_test] ##### seg_lesion_test.py

        ]

        time_list = []

        for name, step in test_steps:
            print('Running step: ' + name + "\n")
            start_time = time.time()
            step()
            tf.reset_default_graph()
            print('\nDone step: '+ name)
            total_time = int(time.time() - start_time)
            time_list.append(total_time)
            print ("\nTime taken: " + str(total_time) + " seconds or, " + str(total_time/60) + " minutes to run\n")

        print("Time for each function: ")
        for func in range(len(test_steps)):
            print("\t" + str(test_steps[func][0]) + ": " + str(time_list[func]) + " seconds, " + str(time_list[func]/60) + " minutes.\n")
        
        print("Total time: " + str(sum(time_list)) + " seconds, " + str(sum(time_list)/60) + " minutes.\n")

    def train(self):
        """
            Driver code for training the model.
        """
        
        train_steps = [
            ['seg_liver_train', self.seg_liver_train], ### seg_liver_train.py
            ['seg_liver_test', self.seg_liver_test], ### seg_liver_test.py

            ['compute_bbs_from_gt_liver', self.compute_3D_bbs_from_gt_liver], ### compute_3D_bbs_from_gt_liver.py

            ['sample_bbs', self.sample_bbs], ### sample_bbs.py

            ['det_lesion_train', self.det_lesion_train], ### det_lesion_train.py
            ['det_lesion_test', self.det_lesion_test], ### det_lesion_test.py

            ['seg_lesion_train', self.seg_lesion_train], ##### seg_lesion_train.py
            ['seg_lesion_test', self.seg_lesion_test] ##### seg_lesion_test.py
        ]

        time_list = []

        for name, step in train_steps:
            print('Running step: ' + name + "\n")
            start_time = time.time()
            step()
            tf.reset_default_graph()
            print('\nDone step: '+ name)
            total_time = int(time.time() - start_time)
            time_list.append(total_time)
            print ("\nTime taken: " + str(total_time) + " seconds or, " + str(total_time/60) + " minutes to run\n")

        print("Time for each function: ")
        for func in range(len(train_steps)):
            print("\t" + str(train_steps[func][0]) + ": " + str(time_list[func]) + " seconds, " + str(time_list[func]/60) + " minutes.\n")
        
        print("Total time: " + str(sum(time_list)) + " seconds, " + str(sum(time_list)/60) + " minutes.\n")

if __name__ =='__main__':
    from config import Config

    config = Config()
    print(config.get_result_root('results'))

    liver_lesion = LiverLesion(config)
    liver_lesion.test()

    #liver_lesion.train()
    


