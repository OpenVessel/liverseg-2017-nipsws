import numpy as np
import scipy.misc
import scipy.io
import os
import pandas as pd
from utils.parse_result import parse_result

# def sample_bbs_test(crops_df, liver_masks_path ):
#     """Samples bounding boxes around liver region for a test image.
#     Args:
#     crops_df: DataFrame, each row with filename, boolean indicating if there is liver, x1, x2, y1, y2, zoom.
#     liver_masks_path: path to gt liver masks

#     Output: 
#     test_df: DataFrame with rows [x1, y1, 0]. (Then, each bb is of 80x80, and the 0 is related
#     to the data augmentation applied, which is none for test images)
#     """

#     # output
#     test_df_rows = []

#     for _, row in crops_df.iterrows():

#         if row["is_liver"]:

#             # constants
#             file = row["liver_seg"].split('images_volumes/')[-1]
#             mask_filename = file.split('.')[0]

#             # binarize liver mask
#             print("Mask filename", mask_filename)
#             print("liver_masks_path", liver_masks_path)
#             mask_liver = scipy.misc.imread(os.path.join(liver_masks_path, mask_filename + '.png'))/255.0
#             mask_liver[mask_liver > 0.5] = 1.0
#             mask_liver[mask_liver < 0.5] = 0.0


#             # add padding to the bounding box
#             padding = 25.0

#             if row["total_mina"] > padding:
#                 row["total_mina"] = row["total_mina"] - padding
#             if row["total_minb"] > padding:
#                 row["total_minb"] = row["total_minb"] - padding
#             if row["total_maxb"] + padding < 512.0:
#                 row["total_maxb"] = row["total_maxb"] + 25.0
#             if row["total_maxa"] + padding < 512.0:
#                 row["total_maxa"] = row["total_maxa"] + padding

#             mult = 50.0

#             max_bbs_a = int((row["total_maxa"]-row["total_mina"])/mult)
#             max_bbs_b = int((row["total_maxb"]-row["total_minb"])/mult)

#             for x in range (0, max_bbs_a):
#                 for y in range (0, max_bbs_b):
#                     mask_liver_aux = mask_liver[int(row["total_mina"] + mult*x):int(row["total_mina"] + (x+1)*mult), int(row["total_minb"] + y*mult):int(row["total_minb"] + (y+1)*mult)]
#                     pos_liver = np.sum(mask_liver_aux)
#                     if pos_liver > (25.0*25.0):
#                         if (row["total_mina"] + mult*x) > 15.0 and ((row["total_mina"] + (x+1)*mult) < 512.0) and (row["total_minb"] + y*mult) > 15.0 and ((row["total_minb"] + (y+1)*mult) < 512.0):
#                             a1 = row["total_mina"] + mult*x - 15.0
#                             b1 = row["total_minb"] + y*mult - 15.0
#                         test_df_rows.append(['images_volumes/{}'.format(file), a1, b1])
#     test_df = pd.DataFrame(test_df_rows, columns=["file_name", "a1", "b1"])
#     return test_df


def sample_bbs(crops_df, data_aug_options, liver_masks_path, lesion_masks_path):

    """
    Samples bounding boxes around liver region for a train image. In this case, we will train two files, one with the positive bounding boxes
    and another with the negative bounding boxes.

    Args:
    crops_df: DataFrame, each row with filename, boolean indicating if there is liver, x1, x2, y1, y2, zoom.
    data_aug_options: How many data augmentation options you want to generate for the training images. The maximum is 8.
    liver_masks_path: path to gt liver masks

    Output:
    dict containing 4 dfs under the keys [test_pos, test_neg, train_pos, train_neg].
    Each df has rows [file name, x1, y1, data_aug_option] (Then, each bb is of 80x80)
    """

    train_positive_df_rows = []
    train_negative_df_rows = []
    test_positive_df_rows = []
    test_negative_df_rows = []

    #print(crops_df)

    # read in bbs from crops df
    for _, row in crops_df.iterrows():
        # constants
        mask_filename = os.path.splitext(row["liver_seg"])[0]
        liver_seg_file = row["liver_seg"].split('liver_seg/')[-1]

        if row["is_liver"] and int(mask_filename.split(os.path.sep)[0])!= 59:

            # binarize masks

            # liver
            mask_liver = scipy.misc.imread(os.path.join(liver_masks_path, mask_filename + '.png'))/255.0
            mask_liver[mask_liver > 0.5] = 1.0
            mask_liver[mask_liver < 0.5] = 0.0

            # lesion
            mask_lesion = scipy.misc.imread(os.path.join(lesion_masks_path, mask_filename + '.png'))/255.0
            mask_lesion[mask_lesion > 0.5] = 1.0
            mask_lesion[mask_lesion < 0.5] = 0.0
            
            

            # add padding

            padding = 25.0

            if row["total_mina"] > padding:
                row["total_mina"] = row["total_mina"] - padding
            if row["total_minb"] > padding:
                row["total_minb"] = row["total_minb"] - padding
            if row["total_maxb"] + padding < 512.0:
                row["total_maxb"] = row["total_maxb"] + padding
            if row["total_maxa"] + padding < 512.0:
                row["total_maxa"] = row["total_maxa"] + padding

            mult = 50.0
            
            max_bbs_a = int((row["total_maxa"]-row["total_mina"])/mult)
            max_bbs_b = int((row["total_maxb"]-row["total_minb"])/mult)
            
            for x in range (0, max_bbs_a):
                for y in range (0, max_bbs_b):
                    bb = np.array([int(row["total_mina"] + x*mult), int(row["total_mina"] + (x+1)*mult), int(row["total_minb"] + y*mult), int(row["total_minb"] + (y+1)*mult)])
                    mask_liver_aux = mask_liver[int(row["total_mina"] + mult*x):int(row["total_mina"] + (x+1)*mult), int(row["total_minb"] + y*mult):int(row["total_minb"] + (y+1)*mult)]
                    pos_liver = np.sum(mask_liver_aux)
                    if pos_liver > (25.0*25.0):
                        mask_lesion_aux = mask_lesion[int(row["total_mina"] + mult*x):int(row["total_mina"] + (x+1)*mult), int(row["total_minb"] + y*mult):int(row["total_minb"] + (y+1)*mult)]
                        pos_lesion = np.sum(mask_lesion_aux)
                        if (row["total_mina"] + mult*x) > 15.0 and ((row["total_mina"] + (x+1)*mult) < 490.0) and (row["total_minb"] + y*mult) > 15.0 and ((row["total_minb"] + (y+1)*mult) < 490.0):
                            a1 = row["total_mina"] + mult*x - 15.0
                            b1 = row["total_minb"] + y*mult - 15.0

                            
                            if pos_lesion > mult:
                                if int(liver_seg_file.split(os.path.sep)[-2]) < 105:
                                    for j in range(data_aug_options):
                                        train_positive_df_rows.append(['images_volumes/{}'.format(liver_seg_file), a1, b1, j+1])
                                else:
                                    test_positive_df_rows.append(['images_volumes/{}'.format(liver_seg_file), a1, b1, 1])
                            else:
                                if int(liver_seg_file.split(os.path.sep)[-2]) < 105:
                                    for j in range(data_aug_options):
                                        train_negative_df_rows.append(['images_volumes/{}'.format(liver_seg_file), a1, b1, j+1])
                                else:
                                    test_negative_df_rows.append(['images_volumes/{}'.format(liver_seg_file), a1, b1, 1])

    # make dfs
    cols = ["file_name", "a1", "b1", "data_aug_option"]
    return {
        "test_pos":  pd.DataFrame(test_positive_df_rows, columns=cols), 
        "test_neg":  pd.DataFrame(test_negative_df_rows, columns=cols),
        "train_pos": pd.DataFrame(train_positive_df_rows, columns=cols), 
        "train_neg": pd.DataFrame(train_negative_df_rows, columns=cols)
    }
    


if __name__ == "__main__":

    database_root = '../../LiTS_database/'

    # Paths for Own Validation set
    images_path = os.path.join(database_root, 'images_volumes')
    liver_masks_path = os.path.join(database_root, 'liver_seg')
    lesion_masks_path = os.path.join(database_root, 'item_seg')

    output_folder_path =  '../../det_DatasetList/'

    # Example of sampling bounding boxes around liver for train images
    crops_list_sp = '../crops_list/crops_LiTS_gt_2.txt'
    #crops_list_sp = '../crops_list/crops_LiTS_gt.txt'
    output_file_name_sp = 'example'
    # all possible combinations of data augmentation
    data_aug_options_sp = 8
    sample_bbs_train(crops_list_sp, output_file_name_sp, data_aug_options_sp, liver_masks_path, lesion_masks_path, output_folder_path)

    # Example of sampling bounding boxes around liver for tests images, when there are no labels
    # uncomment for using this option
    # output_file_name_sp = 'test_patches'
    #sample_bbs_test(crops_list_sp, output_file_name_sp)
    