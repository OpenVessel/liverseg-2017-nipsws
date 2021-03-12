import os
import tensorflow as tf

from seg_liver_test import seg_liver_test
from utils.crops_methods.compute_3D_bbs_from_gt_liver import compute_3D_bbs_from_gt_liver
from utils.sampling_bb.sample_bbs import sample_bbs
from det_lesion_test import det_lesion_test
from seg_lesion_test import seg_lesion_test
from utils.train_test_split import TrainTestSplit
import time
import math


class LiverLesion:
    def __init__(self, config):
        self.config = config


    def seg_liver_test(self, test_volume_txt):
        return seg_liver_test(self.config, test_volume_txt, self.config.num_slices)
    

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
        return seg_lesion_test(self.config, self.config.num_slices)
    

    def test(self, images_volumes, item_seg, liver_seg):
        """
            Driver code for testing the model.
        """


        time_list = []
        last_step_output = None

        
        def runStepWithTime(name, step):
            # run step
            print('Running step: ' + name + "\n")
            start_time = time.time()

            try:
                step_output = step()
            except:
                raise Exception('ERROR running')
            
            print('\nDone step: '+ name)

            ## run time
            total_time = int(time.time() - start_time)
            time_list.append({'name': name, 'time' :total_time})
            floor_var = math.floor(total_time/60)
            mod_var = total_time % 60
            print("\nTime taken: {} seconds or {} minutes {}s to run\n".format(total_time, floor_var, mod_var))

            # reset tf graph for memory purposes
            tf.reset_default_graph()

            return step_output

        print('Splitting training/testing volumes')
        tts = TrainTestSplit(self.config)
        training_volume, testing_volume = tts.split(images_volumes, item_seg, liver_seg)

        # run workflow
        runStepWithTime('seg_liver_test', lambda: self.seg_liver_test(testing_volume))
        crops_df = runStepWithTime('compute_bbs_from_gt_liver', lambda: self.compute_3D_bbs_from_gt_liver())
        patches =  runStepWithTime('sample_bbs_test', lambda: self.sample_bbs(crops_df))
        runStepWithTime('det_lesion_test', lambda: self.det_lesion_test(patches["test_pos"], patches["test_neg"]))
        runStepWithTime('seg_lesion_test', lambda: self.seg_lesion_test()) ## TODO: crops_list_gt.txt --> df


        print("---SUMMARY---")
        for step in time_list:
            print("Step: ", step['name'])
            step_time = step['time']
            print("\nTime taken: {} seconds or {} minutes {}s to run\n".format(step_time, math.floor(step_time/60), step_time % 60))
        
        total_time = sum(time_list)
        print("\nTotal time taken: {} seconds or {} minutes {}s to run\n".format(total_time, math.floor(total_time/60), total_time % 60))

if __name__ =='__main__':
    from config import Config
    

    config = Config()



    # liver_lesion = LiverLesion(config)
    # liver_lesion.test('images_volumes', 'item_seg', 'liver_seg')
    


