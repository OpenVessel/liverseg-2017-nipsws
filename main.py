import os
import tensorflow as tf

from seg_liver_test import seg_liver_test
from utils.crops_methods.compute_3D_bbs_from_gt_liver import compute_3D_bbs_from_gt_liver
from utils.sampling_bb.sample_bbs import sample_bbs
from det_lesion_test import det_lesion_test
from seg_lesion_test import seg_lesion_test
import time


class LiverLesion:
    def __init__(self, config, number_slices=3):
        self.config = config
        self.number_slices=number_slices


    def seg_liver_test(self, test_volume_txt):
        return seg_liver_test(self.config, test_volume_txt, self.number_slices)
    

    def compute_3D_bbs_from_gt_liver(self):
        return compute_3D_bbs_from_gt_liver(self.config)


    def sample_bbs(self, crops_list_sp):
        liver_masks_path = os.path.join(self.config.database_root, 'liver_seg')
        lesion_masks_path = os.path.join(self.config.database_root, 'item_seg')
        data_aug_options = 8
        return sample_bbs(crops_list_sp, data_aug_options, liver_masks_path, lesion_masks_path)



    def det_lesion_test(self, val_file_pos, val_file_neg):
        return det_lesion_test(self.config, val_file_pos, val_file_neg)


    def seg_lesion_test(self):
        return seg_lesion_test(self.config, self.number_slices)
    

    def test(self):
        """
            Driver code for testing the model.
        """
        
        

        time_list = []
        last_step_output = None

        
        def runStepWithTime(name, step):
            # run step
            print('Running step: ' + name + "\n")
            start_time = time.time()

            step_output = step()
            
            print('\nDone step: '+ name)

            ## run time
            total_time = int(time.time() - start_time)
            time_list.append(total_time)
            print ("\nTime taken: " + str(total_time) + " seconds or, " + str(total_time/60) + " minutes to run\n")

            # reset tf graph for memory purposes
            tf.reset_default_graph()

            return step_output


        test_volume_txt = 'seg_DatasetList/testing_volume_OV.txt'

        # run workflow
        #runStepWithTime('seg_liver_test', lambda: self.seg_liver_test(test_volume_txt))
        crops_df = runStepWithTime('compute_bbs_from_gt_liver', lambda: self.compute_3D_bbs_from_gt_liver())
        patches =  runStepWithTime('sample_bbs_test', lambda: self.sample_bbs(crops_df))
        runStepWithTime('det_lesion_test', lambda: self.det_lesion_test(patches["test_pos"], patches["test_neg"]))
        runStepWithTime('seg_lesion_test', lambda: self.seg_lesion_test())


        print("Time for each function: ")
        for func in range(len(test_steps)):
            print("\t" + str(test_steps[func][0]) + ": " + str(time_list[func]) + " seconds, " + str(time_list[func]/60) + " minutes.\n")
        
        print("Total time: " + str(sum(time_list)) + " seconds, " + str(sum(time_list)/60) + " minutes.\n")

if __name__ =='__main__':
    from config import Config

    config = Config()

    liver_lesion = LiverLesion(config)
    liver_lesion.test()
    


