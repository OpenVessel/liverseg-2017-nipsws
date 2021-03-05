import numpy as np
import scipy.misc
import scipy.io
import os
#from utils.parse_result import parse_result
    #create a function that parses out only the slices with useful liver mask pngs from 
        # the crops_list.txt file generated in 3D bb's 
def parse_result(result): 
    # check to see whether we are in the liver 
        #  (by seeing if the text line has any BB coordinates in them) 
    if len(result) > 2: 
            # we extract all the values from the crops_list.txt file and put them into variables 
            id_img, bool_zoom, mina, maxa, minb, maxb = result 

            # make sure the data type for the extracted BB coordinates are in integer form  
            mina = int(mina) 
            maxa = int(maxa) 
            minb = int(minb) 
            maxb = int(maxb) 

        # if there are no BB coordinates in the read txt line then 
            # extract only the necessary variables & set everything else to None 
    else: 
        id_img, bool_zoom = result 
        mina = minb = maxa = maxb = None     
    
    return id_img, bool_zoom, mina, maxa, minb, maxb


    #this function outputs the test_pathes.txt file which includes: 
                # the file location & the min x & min y coordinates of the sample box 
        # it doesn't return the slices that don't have positive liver patches 
        # because these test patches are only used for ASSUMEDLY as testing inputs to the liver lesion network    
    #QUESTION: why don't we output the maxes (is it bc we always just need the x & y iterators to move through the sample BB's)? 
        # check the ASSUMPTION about where we use the test_patches.txt outputs from this function
def sample_bbs_test(crops_list, output_file_name, liver_masks_path, lesion_masks_path, output_folder_path):
    """Samples bounding boxes around liver region for a test image.
    Args: 
    crops_list: Textfile, each row with filename, boolean indicating if there is liver, x1, x2, y1, y2, zoom.
    output_file_name: File name for the output file generated, that will be of the form file name, x1, y1, 0. (Then, each bb is of 80x80, and the 0 is related
    to the data augmentation applied, which is *none* for test images)
    output_folder_path = det_Dataset_list (where the output file will sit)
    """

    #output file opened in write mode & created if it doesn't exist 
            #Note: in compute_3D_bbs.. we check to see whether the outputs exist and make it if not, this may be a more simple way to do that 
        #so we can create a file that has the patient file location, the min x & min y coordinates of the sample box 
    test_file = open(os.path.join(output_folder_path, output_file_name + '.txt'), 'w')
    
    ### open crops_list text file and read it into a list 
    if crops_list is not None:
        with open(crops_list) as t:
            crops_lines = t.readlines()

    ### for every line within the crops_list text file
    for i in range(len(crops_lines)):
        
        ### split by the space between each value within the text line 
        result = crops_lines[i].split(' ')
        ### prints out the list of values witin each line  
        ###QUESTION: why print? 
        print(result)
        
        ###use parse_result function to extract values from text lines  
        id_img, bool_zoom, mina, maxa, minb, maxb = parse_result(result)

        ### if within the liver then 
        if bool_zoom == '1':

                # constants (OG comment)
                #get the last element within the split of id_img 
                #QUESTION: where is the images_volumes part of the path coming from within id_img?
                    #  id_img is supposed to be in the form "patient"/"slice"
            file = id_img.split('images_volumes/')[-1]
            
            # with the form "patient#.png" we split at the period and take just the patient's ID 
                #QUESTION: can we make this into 1 line of code and do it in a more efficient way? 
            mask_filename = file.split('.')[0]

            #QUESTION: why print? 
            print("Mask filename", mask_filename)
            print("liver_masks_path", liver_masks_path)

            # normalize the liver mask PNG 
            mask_liver = scipy.misc.imread(os.path.join(liver_masks_path, mask_filename + '.png'))/255.0
            
            # binarize the liver mask 
            mask_liver[mask_liver > 0.5] = 1.0
            mask_liver[mask_liver < 0.5] = 0.0

            # create the constant that we'll use to add padding to the entire bounding box 
                # we add padding in order to give the bounding box's cropping of the mask room for error 
                    # and this seemingly allows the algorithm to understand how to deal with the edges of the liver
            padding = 25.0 

            # add padding to the mins & maxs 
                # make sure that the image minimums are above the padding 
                # so that we don't go below the images' possible minimum (0) 
                # since we are subtracting the padding from the minimum 
            if mina > padding:
                mina = mina - padding
            if minb > padding:
                minb = minb - padding 
                
                #make sure that the maximum image height/width is not crossed 
                # since we are adding the padding to our the max & mins, doesn't that make our 
            if maxb + padding < 512.0:
                maxb = maxb + padding 
            if maxa + padding < 512.0:
                maxa = maxa + padding

            #this multiplier is equal to the height/width of the sample bounding box we want to create
            mult = 50.0

            # find the true height & width of the bounding box 
            # then split the height & width into multiple regions! by dividing by our multiplier (50) 
                # by dividing by 50 proportionally on both the x & y axis we create 50 x 50 samples of the liver's BB region  
            max_bbs_a = int((maxa-mina)/mult)
            max_bbs_b = int((maxb-minb)/mult)

            # iterate through the amount of regions that there are in the x direction 
                # Note: x = a & y = b 
            for x in range (0, max_bbs_a):
                #for each x region we will iterate through all of the y direction's regions and then move to the next x region  
                for y in range (0, max_bbs_b):
                
                    #the sample BBs are now being applied to the liver mask using the previously calculated values like mult, mina, maxa, & x & y 
                    # the region counter/tracker = value of x & y at some point within the for loop 
                    # the bounding box's height & width should be equal to "mult" ( = 50 ) as we multiply this with the region counter/iterator
                    # the lower bound of the bounding box's width & height will be x & y respectively (the first region's values)
                    # the upper bound of this bounding box will be x+1 & y+1 (the next region's values) 
                    
                    # we start at the min of the 1st region and create all of our sample bb's on the liver mask 
                        # by working our way through each region by iterating through the FOR loop 
                    # specifically we make the sample BB by starting at the whole bounding boxes minimum a & b 
                        # and multiply the size of the bounding box (mult; = 50) by the region we want to reach (defined by x & y) to calculate that region's minimum 
                        # to get the maximum of this sample BB region we just use the next region's minimum by doing the same thing we did to reach the minimum of the region of interest 
                        # and instead multiplying 50 by x+1 & y+1 
                    mask_liver_aux = mask_liver[int(mina + mult*x):int(mina + (x+1)*mult), int(minb + y*mult):int(minb + (y+1)*mult)]
                    
                    # example: mina = 125 & minb= 75, mult = 50, x region = 0 , y region = 3   
                    
                                # mask_liver_aux = mask_liver[125 + 50*0: 125 + 50*1, 75 + 50*3: 75 + 50*4]
                                                # = mask_liver[125:175, 225:275] 

                    #we check to see whether the liver is more positive or negative within the bounding box 
                        #  we do this by taking the sum of the # of True (= 1) pixels within the range of the BB's 2D array of 1's & 0's 
                    pos_liver = np.sum(mask_liver_aux) 
                        #QUESTION: is this for debugging or do we always want to see for each slice how many true values there were within the bounding box?
                    print("np.sum(mask_liver_aux)", np.sum(mask_liver_aux))

                    #if over half of the pixels within the sample BB range are True 
                        #then we categorize this a positive liver BB sample  
                    if pos_liver > (25.0*25.0): 
                        
                        # if the mins are above 15 and the maxs are below 512 then ... 
                            # 15 is the lower bound because we will be subtracting the minimums for the bounding box range by 15 
                            # this value has to be 15 specifically because 15 is the margin of overlap between BB's that we want to have 
                            # we want to create some overlap so that the bounding boxes aren't cut exactly on their distinguishing lines 
                                # so that some level of context is created 
                        if (mina + mult*x) > 15.0 and ((mina + (x+1)*mult) < 512.0) and (minb + y*mult) > 15.0 and ((minb + (y+1)*mult) < 512.0):
                            #recalculate 
                            a1 = mina + mult*x - 15.0 
                            b1 = minb + y*mult - 15.0
                        
                        # we write our new outputs to the output file we created 
                        # we state the file name (i.e. 204.png (204 = patient ID)), and our new BB limits that we just calculated
                        # the 1 is to indicate that this is within the liver 
                        #QUESTION: instead of putting the string images_volumes/ and then the variable "file" would it be more simple & efficient
                            # to just get rid of the string and use the variable "id_img" instead? 
                        #QUESTION: why not include the maxes as well? Where is this file being translated (probably Liver Lesion)?  
                        # we are writing each bounding box mina & minb starting point for each patient and each slice that is within the liver's slice range
                        test_file.write('images_volumes/{} {} {} 1 \n'.format(file, a1, b1))
                            
    test_file.close() 

# this function outputs the training and testing sets into 2 distinct sets where there have been lesions detected and where there has not been a lesion detected 
    # the parameter difference between sample_bbs_test & sample_bbs_train is that train! has "data_aug_options" extra 
def sample_bbs_train(crops_list, output_file_name, data_aug_options, liver_masks_path, lesion_masks_path, output_folder_path):
    """
    Samples bounding boxes around liver region for a train image. In this case, we will train two files, one with the positive bounding boxes
    and another with the negative bounding boxes.
    Args:
    crops_list: Textfile, each row with filename, boolean indicating if there is liver, x1, x2, y1, y2, zoom.
    data_aug_options: How many data augmentation options you want to generate for the training images. The maximum is 8.
    output_file_name: Base file name for the outputs file generated, that will be of the form file name, x1, y1, data_aug_option. (Then, each bb is of 80x80)
        In total 4 text files will be generated. For training, a positive and a negative file, and the same for testing.
    liver_masks_path & lesion_masks_path = GT labels for liver & lesion respectively 
    """

    # opening training & testing files for both positive and negative sample bb's to use as our outputs
    train_positive_file = open(os.path.join(output_folder_path, 'training_positive_det_patches_' + output_file_name + '_dummy.txt'), 'w')
    train_negative_file = open(os.path.join(output_folder_path, 'training_negative_det_patches_' + output_file_name + '_dummy.txt'), 'w')
    test_positive_file = open(os.path.join(output_folder_path, 'testing_positive_det_patches_' + output_file_name + '_dummy.txt'), 'w')
    test_negative_file = open(os.path.join(output_folder_path, 'testing_negative_det_patches_' + output_file_name + '_dummy.txt'), 'w')

    # read the crop_list txt file (in order to get bb coordinates)
    if crops_list is not None:
        with open(crops_list) as t:
            crops_lines = t.readlines()
    
    # QUESTION/SUGGESTION: !!! implement parse_result !!! 
    # for every line within the txt file 
    
    for i in range(len(crops_lines)):
        
        #split the values into elements for a list 
        result = crops_lines[i].split(' ')
        
        # if we are looking within the lung then 
        if len(result) > 2: 
            # place the values from the txt file within their proper variables 
            id_img, bool_zoom, mina, maxa, minb, maxb = result
            # make the strings of mins/maxs into integers 
            mina = int(mina)
            maxa = int(maxa)
            minb = int(minb)
            maxb = int(maxb)
        
        # if we are not looking within the liver for this slice then 
        else:
            # place those respective values from the txt file within their proper variables 
            id_img, bool_zoom = result
            
        # constants (their OG comment)
        #QUESTION: what does it mean to do os.path.splitext vs variable.split?
        #QUESTION: what's the difference between mask_file_name and liver_seg_file?
        # take the first element out of the split id_img variable's **path???** 
            # which should be in the format "patient\slice#" 
            # so seemingly we are extracting the patient's # 
        mask_filename = os.path.splitext(id_img)[0]
            #print("Mask Filename:", mask_filename)
        
        # take the last element of the split test of id_img which should be in the format images_volumes/"patient"/"slice"
            #QUESTION: I don't know how you split on id_img "liver_seg/" when there is no "liver_seg/" in the variable 
                # it seems like what we'll receive out of this variable is id_img right back?
        liver_seg_file = id_img.split('liver_seg/')[-1]
            #print("Int Thing:", mask_filename.split(os.path.sep)[0], os.path.split(mask_filename)[1])

        # QUESTION: are aux & bool_zoom truly synonymous? 
            # am I reading this != patient? 59 correctly? 
        # if we are looking within the liver & the patient is not # 59 
        if bool_zoom == '1' and int(mask_filename.split(os.path.sep)[0])!= 59:

            # normalize & binarize the liver and lesion masks by the threshold value of 0.5 
                #which equates to 127.5 on the grey pixel scale 

                # liver
            mask_liver = scipy.misc.imread(os.path.join(liver_masks_path, mask_filename + '.png'))/255.0
            mask_liver[mask_liver > 0.5] = 1.0
            mask_liver[mask_liver < 0.5] = 0.0

                # lesion
            mask_lesion = scipy.misc.imread(os.path.join(lesion_masks_path, mask_filename + '.png'))/255.0
            mask_lesion[mask_lesion > 0.5] = 1.0
            mask_lesion[mask_lesion < 0.5] = 0.0
            
            #QUESTION: this if statement seems to not do anything, do we want this? 
            if 1:

                # add padding

                padding = 25.0

                    # add padding to the mins & maxs 
                    # make sure that the image minimums are above the padding 
                    # so that we don't go below the images' possible minimum (0) 
                    # since we are subtracting the padding from the minimum 
                if mina > padding:
                    mina = mina - padding
                if minb > padding:
                    minb = minb - padding 
                #make sure that the maximum image height/width is not crossed 
                    # since we are adding the padding to our the max & mins, doesn't that make our 
                if maxb + padding < 512.0:
                    maxb = maxb + padding 
                if maxa + padding < 512.0:
                    maxa = maxa + padding

                 #this multiplier is equal to the height/width of the sample bounding box we want to create
                mult = 50.0

                    ## calulated by average (their OG comment)
                # find the true height & width of the bounding box 
                    # then split the height & width into multiple regions! by dividing by our multiplier (50) 
                    # By dividing by 50 proportionally on both the x & y axis we create 50 x 50 samples of the liver's BB region 
                max_bbs_a = int((maxa-mina)/mult)
                max_bbs_b = int((maxb-minb)/mult)
                
                # iterate through the amount of regions that there are in the x direction
                for x in range (0, max_bbs_a):
                    # for each x region we will iterate through all of the y direction's regions and then move to the next x region
                    for y in range (0, max_bbs_b):
                        
                        #this is not in the sample_bbs_test and does not seem to be used anywhere else in the rest of this function 
                            #QUESTION: do we need it? 
                        # seems to write within a list all of the mins 
                        bb = np.array([int(mina + x*mult), int(mina + (x+1)*mult), int(minb + y*mult), int(minb + (y+1)*mult)])
                        
                        # we start at the min of the 1st x region and create all of our sample bb's in the y direction
                            # by working our way through each region by iterating through the FOR loop 
                        # apply the sample bounding box generated to the ground truth liver mask 
                        mask_liver_aux = mask_liver[int(mina + mult*x):int(mina + (x+1)*mult), int(minb + y*mult):int(minb + (y+1)*mult)]
                        
                        #we check to see whether the liver is more positive or negative within this sample bounding box range 
                            #  we do this by taking the sum of the number of True (= 1) pixels within the range of the BB's 2D array of 1's & 0's 
                        pos_liver = np.sum(mask_liver_aux)
                        
                        #if over half of the pixels within the BB range are True 
                            #then we categorize this a positive liver BB sample and continue to look for lesions within this BB 
                        if pos_liver > (25.0*25.0): 
                            
                            #if we have a positive liver then we apply the same BB to the ground truth lesion image 
                            mask_lesion_aux = mask_lesion[int(mina + mult*x):int(mina + (x+1)*mult), int(minb + y*mult):int(minb + (y+1)*mult)]
                            
                            # look for a positive lesion by adding all of the True (normalized & binarized pixel = 1) values within the image  
                            pos_lesion = np.sum(mask_lesion_aux)
                            
                            # if lower min of x & y's bound > 15 and the upper min of x & y's bound of the BB is < 490 (482?)
                            if (mina + mult*x) > 15.0 and ((mina + (x+1)*mult) < 490.0) and (minb + y*mult) > 15.0 and ((minb + (y+1)*mult) < 490.0):
                                
                                #subtracting the region's mins for both x & y by 15 to create a margin of overlap with this BB and the previous one 
                                a1 = mina + mult*x - 15.0
                                b1 = minb + y*mult - 15.0
                                
                                #print(train_negative_file)
                                #print(train_positive_file)
                                # print(liver_seg_file.split(os.path.sep))
                                # print(liver_seg_file.split(os.path.sep))

                                # if the number of positive values within the sample bounding boxed labeled lesion PNG is > 50 pixels 
                                    # then we will accept this as a valid lesion
                                if pos_lesion > mult:
                                    # check to see whether we should add the positive lesion detection to the training set or testing set
                                    # it is looking to do this by receiving the # of the slice of the patient we are in 
                                        # and are trying to seperate between the training & testing data set 
                                    #QUESTION/SUGGESTION: THIS WILL BREAK IF NEW DATA IS ADDED SINCE THIS VALUE IS HARDCODED
                                        # need to do something similar to the 80%/20% logic we developed in 3D BB    
                                    # if we are looking within the training set then 
                                    if int(liver_seg_file.split(os.path.sep)[-2]) < 105: 
                                        # the user tells us the amount of data augmentation flips they want , 
                                            # we will notify the lesion detection training network to augment the data that many times
                                            # by creating a new line** for each data flip that will occur
                                        # QUESTION: is it more efficient to just list within the text file the number of data flips they want just once for each slice? 
                                        # we add to the training positive patches file the patient # & the slice #, the minimum x & y of the sample BB, 
                                            #and which data augmentation flip we are in (assuming the # of desired flips (data_aug_options) is > 1)
                                        for j in range(data_aug_options): 
                                            train_positive_file.write('images_volumes/{} {} {} {} \n'.format(liver_seg_file, a1, b1, j + 1))
                                    # if we are developing sample BBs for the testing set then data augmentation options are irrelevant 
                                        # and that variable will still be written down as 1 
                                    #QUESTION: why is it j+1 and not just j (meaning 1)? wouldn't we want that value to equal 0? 
                                        # I  think this is because it is being used as an aux/bool_zoom variable with this text file, 
                                        # but we aren't writing anything for out of the liver so I still don't know. 
                                        # There might be an answer further down the road when we are utilizing these text files
                                    #QUESTION/SUGGESTION: why do we need a for loop if it never loops? 
                                    else: 
                                        for j in range(1):
                                            test_positive_file.write('images_volumes/{} {} {} {} \n'.format(liver_seg_file, a1, b1, j + 1))
                                
                                # we will repeat this same process but on the condition that the # of positive lesions is < 50 
                                else:
                                    if int(liver_seg_file.split(os.path.sep)[-2]) < 105:
                                        for j in range(data_aug_options):
                                            train_negative_file.write('images_volumes/{} {} {} {} \n'.format(liver_seg_file, a1, b1, j + 1))
                                    else:
                                        for j in range(1):
                                            test_negative_file.write('images_volumes/{} {} {} {} \n'.format(liver_seg_file, a1, b1, j + 1))

# close the files we just wrote to 
    train_positive_file.close()
    train_negative_file.close()
    test_positive_file.close()
    test_negative_file.close()

# if we are running main.py then we define the GT liver mask paths, 
    # the GT lesion masks path, the folder we output our files to, & define how many data augmentation flips we want 
    #and then call sample_BBs_train or sample_BBs_test
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
    