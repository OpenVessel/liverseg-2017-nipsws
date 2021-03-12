import os
import sys

class Config:
    '''
        Config class that contains all pathing
    '''

    def __init__(self):
        # self.__database_root = 'D:\L_pipe\liver_open\liverseg-2017-nipsws\predict_database'
        self.__database_root = r'C:\Users\grego\OneDrive\Projects\Python Projects\fake_root'

        self.root_folder = os.path.dirname(os.path.realpath(__file__))
        sys.path.append(os.path.abspath(self.root_folder))

        self.database_root = os.path.join(self.root_folder, self.__database_root)
        
        self.resnet_ckpt = os.path.join(self.root_folder, 'train_files', 'resnet_v1_50.ckpt')
        self.imagenet_ckpt = os.path.join(self.root_folder, 'train_files', 'vgg_16.ckpt')

        self.images_volumes = 'images_volumes'
        self.item_seg = 'item_seg'
        self.liver_seg = 'liver_seg'
        
        self.debug = 0

        self.patient_range = [105,108] # inclusive

        self.num_slices = 3


    def get_log(self, task_name):
        return os.path.join(self.root_folder, 'train_files', task_name, 'networks')