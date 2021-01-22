import os

from seg_liver_test import seg_liver_test
from utils.crops_methods.compute_3D_bbs_from_gt_liver import compute_3D_bbs_from_gt_liver
from utils.sampling_bb.sample_bbs import sample_bbs_test, sample_bbs_train
from det_lesion_test import det_lesion_test
from seg_lesion_test import seg_lesion_test
import time


class LiverLesion:
    def __init__(self, config, number_slices=3):
        self.config = config
        self.number_slices=number_slices


    def seg_liver_test(self):
        seg_liver_test(self.config, self.number_slices)
    

    def compute_3D_bbs_from_gt_liver(self):
        compute_3D_bbs_from_gt_liver(self.config)
    

    def sample_bbs_test(self):
        liver_masks_path = os.path.join(self.config.database_root, 'liver_seg')
        lesion_masks_path = os.path.join(self.config.database_root, 'item_seg')
        output_folder_path =  './det_DatasetList/'
        crops_list_sp = './utils/crops_list/crops_LiTS_gt_2.txt'
        #crops_list_sp = '../crops_list/crops_LiTS_gt.txt'
        output_file_name_sp = 'test_patches'

        data_aug_options_sp = 8
        sample_bbs_train(crops_list_sp, output_file_name_sp, data_aug_options_sp, liver_masks_path, lesion_masks_path, output_folder_path)
        # sample_bbs_test(crops_list_sp, output_file_name_sp, liver_masks_path, lesion_masks_path, output_folder_path)


    def det_lesion_test(self):
        det_lesion_test(self.config)


    def seg_lesion_test(self):
        seg_lesion_test(self.config, self.number_slices)
    

    def test(self):
        """
            Driver code for testing the model.
        """
        
        test_steps = [
            # ['seg_liver_test', self.seg_liver_test],
            ['compute_bbs_from_gt_liver', self.compute_3D_bbs_from_gt_liver], 
            # ['sample_bbs_test', self.sample_bbs_test], 
            # ['det_lesion_test', self.det_lesion_test], 
            # ['seg_lesion_test', self.seg_lesion_test]
        ]

        time_list = []

        for name, step in test_steps:
            print('Running step: ' + name + "\n")
            start_time = time.time()
            step()
            print('\nDone step: '+ name)
            total_time = time.time() - start_time
            time_list.append(total_time)
            print ("\nTime taken: " + str(total_time) + " secondsor, " + str(total_time/60) + " minutes to run\n")

        print("Time for each function: ")
        for func in range(len(test_steps)):
            print("\t" + str(test_steps[func][0]) + ": " + str(time_list[func]) + " seconds, " + str(time_list[func]/60) + " minutes.\n")
        
        print("Total time: " + str(sum(time_list)) + " seconds, " + str(sum(time_list)/60) + " minutes.\n")

if __name__ =='__main__':
    from config import Config

    config = Config()
    print(config.get_result_root('results'))

    liver_lesion = LiverLesion(config)
    liver_lesion.test()
    


