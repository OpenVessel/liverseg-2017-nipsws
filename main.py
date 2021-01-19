import os

from seg_liver_test import seg_liver_test
from utils.crops_methods.compute_3D_bbs_from_gt_liver import compute_3D_bbs_from_gt_liver
from utils.sampling_bb.sample_bbs import sample_bbs_test
from det_lesion_test import det_lesion_test
from seg_lesion_test import seg_lesion_test

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

        sample_bbs_test(crops_list_sp, output_file_name_sp, liver_masks_path, lesion_masks_path, output_folder_path)


    def det_lesion_test(self):
        det_lesion_test(self.config)


    def seg_lesion_test(self):
        seg_lesion_test(self.config, self.number_slices)
    

    def test(self):
        """
            Driver code for testing the model.
        """

        test_steps = [
            #['seg_liver_test', self.seg_liver_test],
            #['compute_bbs_from_gt_liver', self.compute_3D_bbs_from_gt_liver], 
            #['sample_bbs_test', self.sample_bbs_test], 
            ['det_lesion_test', self.det_lesion_test], 
            #['seg_lesion_test', self.seg_lesion_test]
        ]
        for name, step in test_steps:
            print('Running step: '+ name)
            step()
            print('Done step: '+ name)

if __name__ =='__main__':
    from config import Config

    config = Config()
    print(config.get_result_root('results'))

    liver_lesion = LiverLesion(config)
    liver_lesion.test()
    


