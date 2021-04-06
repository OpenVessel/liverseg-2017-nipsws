import os
import tensorflow as tf

from seg_liver_test import seg_liver_test
from seg_liver_train import seg_liver_train

from utils.crops_methods.compute_3D_bbs_from_gt_liver import compute_3D_bbs_from_gt_liver
from utils.sampling_bb.sample_bbs import sample_bbs
from utils.train_test_split import TrainTestSplit
from det_lesion_test import det_lesion_test
from det_lesion_train import det_lesion_train

from seg_lesion_test import seg_lesion_test
from utils.train_test_split import TrainTestSplit
import time
import math



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

        self.time_list = []

        tts = TrainTestSplit(self.config)
        self.training_volume, self.testing_volume = tts.split(self.config.images_volumes, self.config.item_seg, self.config.liver_seg)


    def logSummary(self, phase, time_list):
        print("--- SUMMARY ({0}) ---".format(phase))
        for step in time_list:
            print("Step: ", step['name'])
            step_time = step['time']
            print("\nTime taken: {} seconds or {} minutes {}s to run\n".format(step_time, math.floor(step_time/60), step_time % 60))

        total_time = sum(time_list.map(lambda x: x['time']))
        print("\nTotal time taken: {} seconds or {} minutes {}s to run\n".format(total_time, math.floor(total_time/60), total_time % 60))


    def with_time(step):
        def wrapper(self):
            # run step
            print('Running step: ' + step.__name__ + "\n")
            start_time = time.time()

            step_output = step(self)

            print('\nDone step: '+ step.__name__)

            ## run time
            total_time = int(time.time() - start_time)
            self.time_list.append({'name': step.__name__, 'time' :total_time})
            
            floor_var = math.floor(total_time/60)
            mod_var = total_time % 60
            print("\nTime taken: {} seconds or {} minutes {}s to run\n".format(total_time, floor_var, mod_var))
            
            # reset tf graph for memory purposes
            tf.reset_default_graph()

            return step_output
        return wrapper


    @with_time
    def seg_liver_test(self, test_volume_txt):
        return seg_liver_test(self.config, test_volume_txt, self.config.num_slices)
    

    @with_time
    def seg_liver_train(self):
        train_df = self.training_volume
        val_df = self.testing_volume
        seg_liver_train(self.config, train_df, val_df,
                        self.gpu_id, self.number_slices, self.batch_size, self.iter_mean_grad, 
                        self.max_training_iters_1, self.max_training_iters_2, self.max_training_iters_3, 
                        self.save_step, self.display_step, self.ini_learning_rate, self.boundaries, self.values)
    

    @with_time
    def compute_3D_bbs_from_gt_liver(self):
        return compute_3D_bbs_from_gt_liver(self.config)


    @with_time
    def sample_bbs(self, crops_list_sp):
        liver_masks_path = os.path.join(self.config.database_root, 'liver_seg')
        lesion_masks_path = os.path.join(self.config.database_root, 'item_seg')
        data_aug_options = 8
        return sample_bbs(crops_list_sp, data_aug_options, liver_masks_path, lesion_masks_path)


    @with_time
    def det_lesion_test(self, val_file_pos, val_file_neg):
        return det_lesion_test(self.config, val_file_pos, val_file_neg)


    @with_time
    def det_lesion_train(self):
        det_lesion_train(self.config, self.gpu_id, self.det_batch_size, self.det_iter_mean_grad, self.det_max_training_iters, 
                        self.det_save_step, self.display_step, self.det_learning_rate)


    @with_time
    def seg_lesion_test(self):
        return seg_lesion_test(self.config, self.config.num_slices)
    

    @with_time
    def seg_lesion_train(self):
        seg_lesion_train(self.config, self.gpu_id, self.number_slices, self.batch_size, self.iter_mean_grad,
                        self.max_training_iters_1, self.max_training_iters_2, self.max_training_iters_3, self.save_step,
                        self.display_step, self.ini_learning_rate, self.boundaries, self.values)


    def test(self):
        """
            Driver code for testing the model.
        """

        self.config.phase = "test"

        self.time_list = []

        # testing workflow
        self.seg_liver_test(self.testing_volume)
        crops_df = self.compute_3D_bbs_from_gt_liver()
        patches = self.sample_bbs(crops_df)
        self.det_lesion_test(patches["test_pos"], patches["test_neg"])
        self.seg_lesion_test()

        self.logSummary('Testing', self.time_list)
        

    def train(self):
        """
            Driver code for training the model.
        """

        self.config.phase = "train"

        self.time_list = []

        # training workflow
        self.seg_liver_train()
        self.seg_liver_test()
        self.compute_3D_bbs_from_gt_liver()
        self.sample_bbs()
        self.det_lesion_train()
        self.det_lesion_test()
        self.seg_lesion_train()
        self.seg_lesion_test()

        self.logSummary('Training', self.time_list)

# Global vars and driver
if __name__ =='__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train or Test the Liver Lesion Segmentation Model")
    parser.add_argument('mode', help="'test' or 'train' depending on what you wish to do.")
    cmdline = parser.parse_args()

    from config import Config
    

    config = Config()

    config.labels = True # Change to false if we don't have labels

    liver_lesion = LiverLesion(config)
    
    if cmdline.mode == "test":
        liver_lesion.test()
    elif cmdline.mode == "train":
        liver_lesion.train()
    else:
        raise BaseException('Invalid mode. Must be test or train')