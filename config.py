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
        self.imagenet_ckpt = os.path.join(self.root_folder, 'train_files', 'vgg_16.ckpt') #D:\L_pipe\results\3_23_2021 results\train_files\seg_liver_ck\networks
        self.bar_ckpt = os.path.join(self.root_folder, 'train_files', 'seg_liver_ck' , 'networks', 'seg_liver.ckpt')
        self.old_weights =  os.path.join(self.root_folder, 'train_files', 'seg_liver_ck' , 'networks', 'seg_liver.ckpt')
        ## the news weights are saved 
        #"D:\L_pipe\liver_open\liverseg-2017-nipsws\train_files\seg_liver\networks\train"
        self.images_volumes = 'images_volumes'
        self.item_seg = 'item_seg'
        self.liver_seg = 'liver_seg'
        
        self.debug = 0 # 0 for false, 1 for true

        self.phase = 'train' ## train or test

        #self.crops_list = 'crops_list_OV.txt' # change as per convenience in comment below but leave this be for final
        self.crops_list = 'crops_LiTS_gt.txt' # Do we still need this? Most likely, no.
        self.patient_range = [105,108] # inclusive

        self.num_slices = 3

        self.labels = True

        self.fine_tune = 0

    def get_result_root(self, result_name):
        return os.path.join(self.root_folder, result_name)

    def get_crops_list_path(self):
        return os.path.join(self.root_folder, 'utils', 'crops_list', self.crops_list)

    def get_log(self, task_name):
        return os.path.join(self.root_folder, 'train_files', task_name, 'networks')
