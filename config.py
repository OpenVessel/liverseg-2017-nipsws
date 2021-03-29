import os
import sys

class Config:
    '''
        Config class that contains all pathing
    '''

    def __init__(self):
        self.__database_root = 'LiTS_database'
        #self.__database_root = 'predict_database'

        self.root_folder = os.path.dirname(os.path.realpath(__file__))
        sys.path.append(os.path.abspath(self.root_folder))

        self.database_root = os.path.join(self.root_folder, self.__database_root)
        
        self.resnet_ckpt = os.path.join(self.root_folder, 'train_files', 'resnet_v1_50.ckpt') #"seg_lesion.ckpt.data-00000-of-00001"
        self.imagenet_ckpt = os.path.join(self.root_folder, 'train_files', 'vgg_16.ckpt')

        self.images_volumes = 'images_volumes'
        self.item_seg = 'item_seg'
        self.liver_seg = 'liver_seg'
        
        self.debug = 0 # 0 for false, 1 for true

        self.phase = 'test' ## train or test

        self.crops_list = 'crops_list_OV.txt' # change as per convenience in comment below but leave this be for final
        # self.crops_list = 'crops_LiTS_gt.txt'
        self.patient_range = [105,108] # inclusive

        self.num_slices = 3

    def get_result_root(self, result_name):
        return os.path.join(self.root_folder, result_name)

    def get_crops_list_path(self):
        return os.path.join(self.root_folder, 'utils', 'crops_list', self.crops_list)

    def get_log(self, task_name):
        return os.path.join(self.root_folder, 'train_files', task_name, 'networks')
