import numpy as np
import scipy.misc
import scipy.io
import os


def sample_bbs_test(crops_list, output_file_name):
    """Samples bounding boxes around liver region for a test image.
    Args:
    crops_list: Textfile, each row with filename, boolean indicating if there is liver, x1, x2, y1, y2, zoom.
    output_file_name: File name for the output file generated, that will be of the form file name, x1, y1, 0. (Then, each bb is of 80x80, and the 0 is related
    to the data augmentation applied, which is none for test images)
    """

    test_file = open(os.path.join(output_folder_path, output_file_name + '.txt'), 'w')

    if crops_list is not None:
        with open(crops_list) as t:
            crops_lines = t.readlines()

    for i in range(len(crops_lines)):
        result = crops_lines[i].split(' ')

        if len(result) > 2:
            id_img, bool_zoom, mina, maxa, minb, maxb = result # parsed line from txt file
            mina = int(mina)
            maxa = int(maxa)
            minb = int(minb)
            maxb = int(maxb)
        else:
            id_img, bool_zoom = result

        if bool_zoom == '1':

            # constants
            file = id_img.split('images_volumes/')[-1]
            mask_filename = file.split('.')[0]

            # binarize liver mask
            mask_liver = scipy.misc.imread(os.path.join(liver_masks_path, mask_filename + '.png'))/255.0
            mask_liver[mask_liver > 0.5] = 1.0
            mask_liver[mask_liver < 0.5] = 0.0

            # add padding to the bounding box

            padding = 25.0

            if mina > padding:
                mina = mina - padding
            if minb > padding:
                minb = minb - padding
            if maxb + padding < 512.0:
                maxb = maxb + 25.0
            if maxa + padding < 512.0:
                maxa = maxa + padding

            mult = 50.0

            max_bbs_a = int((maxa-mina)/mult)
            max_bbs_b = int((maxb-minb)/mult)

            for x in range (0, max_bbs_a):
                for y in range (0, max_bbs_b):
                    mask_liver_aux = mask_liver[int(mina + mult*x):int(mina + (x+1)*mult), int(minb + y*mult):int(minb + (y+1)*mult)]
                    pos_liver = np.sum(mask_liver_aux)
                    if pos_liver > (25.0*25.0):
                        if (mina + mult*x) > 15.0 and ((mina + (x+1)*mult) < 512.0) and (minb + y*mult) > 15.0 and ((minb + (y+1)*mult) < 512.0):
                            a1 = mina + mult*x - 15.0
                            b1 = minb + y*mult - 15.0
                        test_file.write('images_volumes/{} {} {} 1 \n'.format(file, a1, b1))
    test_file.close()


def sample_bbs_train(crops_list, output_file_name, data_aug_options):

    """
    Samples bounding boxes around liver region for a train image. In this case, we will train two files, one with the positive bounding boxes
    and another with the negative bounding boxes.
    Args:
    crops_list: Textfile, each row with filename, boolean indicating if there is liver, x1, x2, y1, y2, zoom.
    data_aug_options: How many data augmentation options you want to generate for the training images. The maximum is 8.
    output_file_name: Base file name for the outputs file generated, that will be of the form file name, x1, y1, data_aug_option. (Then, each bb is of 80x80)
        In total 4 text files will be generated. For training, a positive and a negative file, and the same for testing.
    """

    train_positive_file = open(os.path.join(output_folder_path, 'training_positive_det_patches_' + output_file_name + '.txt'), 'w')
    train_negative_file = open(os.path.join(output_folder_path, 'training_negative_det_patches_' + output_file_name + '.txt'), 'w')
    test_positive_file = open(os.path.join(output_folder_path, 'testing_positive_det_patches_' + output_file_name + '.txt'), 'w')
    test_negative_file = open(os.path.join(output_folder_path, 'testing_negative_det_patches_' + output_file_name + '.txt'), 'w')


    # read in bbs from txt file
    if crops_list is not None:
        with open(crops_list) as t:
            crops_lines = t.readlines()
    
    for i in range(len(crops_lines)):
        result = crops_lines[i].split(' ')
        
        if len(result) > 2:
            id_img, bool_zoom, mina, maxa, minb, maxb = result
            mina = int(mina)
            maxa = int(maxa)
            minb = int(minb)
            maxb = int(maxb)
        else:
            id_img, bool_zoom = result
            
        # constants
        mask_filename = id_img.split('images_volumes/')[-1].split('.')[0]
        liver_seg_file = id_img.split('liver_seg/')[-1]
        
        if bool_zoom == '1' and int(mask_filename.split('/')[0])!= 59:

            # binarize masks

            # liver
            mask_liver = scipy.misc.imread(os.path.join(liver_masks_path, mask_filename + '.png'))/255.0
            mask_liver[mask_liver > 0.5] = 1.0
            mask_liver[mask_liver < 0.5] = 0.0

            # lesion
            mask_lesion = scipy.misc.imread(os.path.join(lesion_masks_path, mask_filename + '.png'))/255.0
            mask_lesion[mask_lesion > 0.5] = 1.0
            mask_lesion[mask_lesion < 0.5] = 0.0
            
            if 1:

                # add padding

                padding = 25.0

                if mina > padding:
                    mina = mina - padding
                if minb > padding:
                    minb = minb - padding
                if maxb + padding < 512.0:
                    maxb = maxb + padding
                if maxa + padding < 512.0:
                    maxa = maxa + padding

                mult = 50.0
                    ## calulated by average
                max_bbs_a = int((maxa-mina)/mult)
                max_bbs_b = int((maxb-minb)/mult)
                
                for x in range (0, max_bbs_a):
                    for y in range (0, max_bbs_b):
                        bb = np.array([int(mina + x*mult), int(mina + (x+1)*mult), int(minb + y*mult), int(minb + (y+1)*mult)])
                        mask_liver_aux = mask_liver[int(mina + mult*x):int(mina + (x+1)*mult), int(minb + y*mult):int(minb + (y+1)*mult)]
                        pos_liver = np.sum(mask_liver_aux)
                        if pos_liver > (25.0*25.0):
                            mask_lesion_aux = mask_lesion[int(mina + mult*x):int(mina + (x+1)*mult), int(minb + y*mult):int(minb + (y+1)*mult)]
                            pos_lesion = np.sum(mask_lesion_aux)
                            if (mina + mult*x) > 15.0 and ((mina + (x+1)*mult) < 490.0) and (minb + y*mult) > 15.0 and ((minb + (y+1)*mult) < 490.0):
                                a1 = mina + mult*x - 15.0
                                b1 = minb + y*mult - 15.0
                                if pos_lesion > mult:
                                    if int(liver_seg_file.split('/')[-2]) < 105:
                                        for j in range(data_aug_options):
                                            train_positive_file.write('images_volumes/{} {} {} {} \n'.format(liver_seg_file, a1, b1, j + 1))
                                    else:
                                        for j in range(1):
                                            test_positive_file.write('images_volumes/{} {} {} {} \n'.format(liver_seg_file, a1, b1, j + 1))

                                else:
                                    if int(liver_seg_file.split('/')[-2]) < 105:
                                        for j in range(data_aug_options):
                                            train_negative_file.write('images_volumes/{} {} {} {} \n'.format(liver_seg_file, a1, b1, j + 1))
                                    else:
                                        for j in range(1):
                                            test_negative_file.write('images_volumes/{} {} {} {} \n'.format(liver_seg_file, a1, b1, j + 1))

    train_positive_file.close()
    train_negative_file.close()
    test_positive_file.close()
    test_negative_file.close()


if __name__ == "__main__":

    database_root = '../../LiTS_database/'

    # Paths for Own Validation set
    images_path = os.path.join(database_root, 'images_volumes')
    liver_masks_path = os.path.join(database_root, 'liver_seg')
    lesion_masks_path = os.path.join(database_root, 'item_seg')

    output_folder_path =  '../../det_DatasetList/'

    # Example of sampling bounding boxes around liver for train images
    crops_list_sp = '../crops_list/crops_LiTS_gt.txt'
    output_file_name_sp = 'example'
    # all possible combinations of data augmentation
    # data_aug_options_sp = 8
    # sample_bbs_train(crops_list_sp, output_file_name_sp, data_aug_options_sp)

    # Example of sampling bounding boxes around liver for tests images, when there are no labels
    # uncomment for using this option
    output_file_name_sp = 'test_patches'
    sample_bbs_test(crops_list_sp, output_file_name_sp)
    