import numpy as np
from scipy import misc
import os
import glob
import math
import scipy.io
import pandas as pd

def compute_3D_bbs_from_gt_liver(config):

    path_list = []

    MIN_AREA_SIZE = 512.0*512.0

    phase = config.phase ## if/else 

    # inputs
    images_path = os.path.join(config.database_root, 'images_volumes')
    labels_path = os.path.join(config.database_root,  'item_seg/') ##GT labels
    labels_liver_path = os.path.join(config.database_root,  'liver_seg/') ## GT labels
    liver_results = os.path.join(config.database_root, 'seg_liver_ck/')

    # outputs
    output_images_path_bb = os.path.join(config.database_root, 'bb_images_volumes_alldatabase3_gt_nozoom_common_bb')
    output_labels_path_bb = os.path.join(config.database_root,  'bb_liver_lesion_seg_alldatabase3_gt_nozoom_common_bb')
    output_labels_liver_path_bb = os.path.join(config.database_root,  'bb_liver_seg_alldatabase3_gt_nozoom_common_bb')
    output_liver_results_path_bb = os.path.join(config.database_root, 'liver_results/')
    crops_df_rows = []

    # This script computes the bounding boxes around the liver from the ground truth, computing
    # a single 3D bb for all the volume.

    ## all paths in a list
    bb_paths = [output_labels_path_bb, output_images_path_bb, output_labels_liver_path_bb, output_liver_results_path_bb]

    for bb_path in bb_paths:
        if not os.path.exists(bb_path):
            os.makedirs(bb_path)

    def integerise(value):
        if value != '.DS_Store':
            return int(value)

    ## If no labels, the masks_folder should contain the results of liver segmentation
    # masks_folders = os.listdir(results_path + 'liver_seg/')
    #print(labels_liver_path)
    masks_folders = os.listdir(labels_liver_path) # liver seg 
    masks_folders = sorted(masks_folders, key=integerise)

    # print(type(masks_folders))
    ## alex code right 
    if phase == 'train':
#calc where the start of the testing files are because liver_results are only being generated on testing data 
        print("mask folders", masks_folders)
        start_test_dir = int(round(.8*len(masks_folders)))
        print(start_test_dir)
        masks_folders = masks_folders[start_test_dir:]
        print(masks_folders)
        #105 - 130 
    if phase == 'train':
        x = 1

    sort_by_path = lambda x: int(os.path.splitext(os.path.basename(x))[0])

    for i in range(len(masks_folders)):

        # change numbers to change range of patients.
        #what does the label data look like
        if masks_folders[i] != '.DS_Store':
            if not masks_folders[i].startswith(('.', '\t')):
                dir_name = masks_folders[i]
                ## If no labels the masks_folder should contain the results of liver segmentation

                # get file names of png ground truths
                masks_of_volume = glob.glob(labels_liver_path + dir_name + '/*.png')
                file_names = (sorted(masks_of_volume, key=sort_by_path))

                depth_of_volume = len(masks_of_volume)
            
            # make directory if it doesn't exist
            for bb_path in bb_paths:
                if not os.path.exists(os.path.join(bb_path, dir_name)):
                    os.makedirs(os.path.join(bb_path, dir_name))

            total_maxa = 0
            total_mina = 10000000
            
            total_maxb = 0
            total_minb = 10000000


            for j in range(0, depth_of_volume):
                img = misc.imread(file_names[j])
                img = img/255.0
                img[np.where(img > 0.5)] = 1
                img[np.where(img < 0.5)] = 0
                a, b = np.where(img == 1)
                
                if len(a) > 0:

                    maxa = np.max(a)
                    maxb = np.max(b)
                    mina = np.min(a)
                    minb = np.min(b)
                    
                    if maxa > total_maxa:
                        total_maxa = maxa
                    if maxb > total_maxb:
                        total_maxb = maxb
                    if mina < total_mina:
                        total_mina = mina
                    if minb < total_minb:
                        total_minb = minb

            for j in range(0, depth_of_volume):
                img = misc.imread(file_names[j])
                img = img/255.0
                img[np.where(img > 0.5)] = 1
                img[np.where(img < 0.5)] = 0

                a, b = np.where(img == 1)
                
                if len(a) > 0:

                    new_img = img[total_mina:total_maxa, total_minb:total_maxb]

                # gt filename
                current_file = os.path.splitext(file_names[j])[0]
                # current_file = file_names[j].split('.png')[0]
                
                if config.debug:
                    print("current file ->",current_file)
                    
                png = os.path.basename(current_file) + '.png'
                mat = os.path.basename(current_file) + '.mat'
                liver_seg = current_file.split('liver_seg/')[-1]

                is_liver = len(np.where(img == 1)[0]) > 500
                if is_liver:

                    # constants
                    area = 1
                    zoom = math.sqrt(MIN_AREA_SIZE/area)
                

                    # write to crops df
                    crops_df_rows.append([liver_seg, is_liver, total_mina, total_maxa, total_minb, total_maxb])

                    ######### apply 3Dbb to files ##########
                    if config.debug:
                        print("images_path",images_path)
                        print("dir_name", dir_name)
                        print("mat", mat)
                        print("png", png)


                    # .mat
                    original_img = np.array(scipy.io.loadmat(os.path.join(images_path, dir_name, mat))['section'], dtype = np.float32)
                    o_new = original_img[total_mina:total_maxa, total_minb:total_maxb]
                    scipy.io.savemat(os.path.join(output_images_path_bb, dir_name, mat), mdict={'section': o_new})
                
                    # lesion png
                    original_label = misc.imread(os.path.join(labels_path, dir_name, png))
                    lbl_new = original_label[total_mina:total_maxa, total_minb:total_maxb]
                    misc.imsave(os.path.join(output_labels_path_bb, dir_name, png), lbl_new)
                    
                    # liver png
                    original_liver_label = misc.imread(os.path.join(labels_liver_path, dir_name, png))
                    lbl_liver_new = original_liver_label[total_mina:total_maxa, total_minb:total_maxb]
                    misc.imsave(os.path.join(output_labels_liver_path_bb, dir_name,  png), lbl_liver_new)

                    # results png
                    original_results_label = misc.imread(os.path.join(liver_results, dir_name, png))
                    res_liver_new = original_results_label[total_mina:total_maxa, total_minb:total_maxb]
                    misc.imsave(os.path.join(output_liver_results_path_bb, dir_name, png), res_liver_new)

                    if config.debug:
                        print("Success" + "Directory:" + str(liver_results) + "Patient:" + str(dir_name) + "File:" + str(png))

                else:
                    crops_df_rows.append([liver_seg, is_liver, None, None, None, None])

    crops_df = pd.DataFrame(crops_df_rows, columns=["liver_seg", "is_liver", "total_mina", "total_maxa", "total_minb", "total_maxb"])
    return crops_df

    pd.DataFrame(path_list)

if __name__ =='__main__':
    from config import Config
    config = Config()
    crops_df = compute_3D_bbs_from_gt_liver(config)
    print(crops_df.head(10))