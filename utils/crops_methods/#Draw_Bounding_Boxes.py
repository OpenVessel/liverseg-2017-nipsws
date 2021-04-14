#Draw_Bounding_Boxes
#Define parse_result

#############
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

###########



########### start getting max_bbs_a & max_bbs_b 
crops_folder_path = ''

crops_list = open(os.path.join(crops_folder_path, + 'crops_list.txt'))

if crops_list is not None:
    with open(crops_list) as t:
        crops_lines = t.readlines()

for i in range(len(crops_lines)):
    
    ### split by the space between each value within the text line 
    result = crops_lines[i].split(' ')
    
    ###use parse_result function to extract values from text lines  
    id_img, bool_zoom, mina, maxa, minb, maxb = parse_result(result)

    ### if within the liver then 
    if bool_zoom == '1':

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

        ########## end getting max_bbs_a & max_bbs_b 


#THE REAL THING
    #what do we need to fix? 
        #what size are our bounding boxes really? 
        #what is the png patient folder path and what is the test_patches folder path? 
############################################################

import bbox_visualizer as bbv

########## get matlab's png patient folder  

#*** Matlab PNG Patient File Location 
matlab_png_patient_folder = os.path.join(config.database_root, '???')

########## open test_patches.txt

#*** what folder is test_patches in? 
test_patches_folder_path = '???'

#extract test_patches from the folder & open it up 
test_patches = open(os.path.join(test_patches_folder_path, + 'test_patches.txt'))

#read each line of test_patches.txt
if test_patches is not None: 
    test_patches_lines = test_patches.readlines()

a = 0 
id_img_tracker_list=[]
#go through each line of test_patches
for i in len(test_patches_lines):
    
    #split the variables on the spaces between them 
    line = test_patches_lines[i].split(' ')
    
    #pull the variables from each line 
    id_img, mina, minb, aux = line 
    
    #find the maxes of the sample bounding boxes 
        #fix this? 
    maxa= mina + 64 
    maxb= minb + 64

    #check to see if the new id_img being read in, is the same as the last one 
        #if so, then we'll continue adding the coordinates to our bboxes list
    if id_img in id_img_tracker_list: 
        # make our bbox based off the coordinates 
        bbox= (mina, maxa, minb, maxb)

        #input coordinates that'll eventually go to the B_Box Visualizer code 
        bb_boxes = bb_boxes.append(bbox)
    
    #if our id_img is not a repeat of a previous slice then we must be on to a new slice 
    else:
        #so we draw the last patient if there is a last patient
        if a != 0: 
            curr_img= os.path.join(matlab_png_patient_folder,id_img_tracker_list[-1], + '.png')
            # Visualize each box for each slice 
            bbv.draw_multiple_rectangles(curr_img, bboxes)
            bb_boxes = []
        
        #add this new id_img to the tracker_list 
        id_img_tracker_list.append(id_img)
        
        #after we append our first image we set this to 1 to let the code know that we can start drawing now
        a = 1 
        
        #properly make this new slice's first bbox 
        bbox= (mina, maxa, minb, maxb)

        #add first bbox for the new slice 
        #input coordinates to the B_Box Visualizer code 
        bb_boxes = bb_boxes.append(bbox)


        

#######################################################






#a1 & b1 coming out of test_patches .txt will be the margined sample bounding box's minx & miny respectively 

#mina = (mina + mult*x) 
#maxa = ((mina + (x+1)*mult)

#mina = mina + 50

#mask_liver_aux = mask_liver[int(mina + mult*x):int(mina + (x+1)*mult), int(minb + y*mult):int(minb + (y+1)*mult)]

## bb.
#x= ? need a double for loop that reads in the text file, grabs a1, b1 (max x, max y), calculates the minimums   





# -----------------------------------------------------------
# bounding boxes applied to PNGs
#
# By Alex Schweizer & Leslie Wubbel 2/25/2021
# OpenVessel 
# lesliewubbel@gmail.com, 17aschweizer.stem@gmail.com
#
# -----------------------------------------------------------
    #what do we need to fix? 
        #what size are our bounding boxes really? 
############################################################
import bbox_visualizer as bbv
import os
from PIL import Image
import numpy as np
########## get matlab's png patient folder  
#*** Matlab PNG Patient File Location 
## DataScienceDeck Pathing 
bb_path_output = r"D:\L_pipe\DataScienceDeck\DataScienceDeck\Data\Data_output\Output_bounding_boxes"
#matlab_png_patient_folder = os.path.join(config.database_root, '???')
#matlab_png_patient_folder = r"D:\L_pipe\liver_open\liverseg-2017-nipsws\LiTS_database\item_seg"
matlab_png_patient_folder = r"D:\L_pipe\liver_open\liverseg-2017-nipsws\LiTS_database\liver_seg"
#matlab_png_patient_folder = r"D:\L_pipe\DataScienceDeck\DataScienceDeck\Data\Data_output\output_png_matlab"
#D:\L_pipe\liver_open\liverseg-2017-nipsws\LiTS_database\liver_seg
########## open test_patches.txt
## pathway for test_patches
## D:\L_pipe\liver_open\liverseg-2017-nipsws\det_DatasetList\test_patches.txt
test_patches_folder_path = r'D:\L_pipe\liver_open\liverseg-2017-nipsws\det_DatasetList'
#extract test_patches from the folder & open it up 
print(test_patches_folder_path)
#test_patches = open(os.path.join(test_patches_folder_path, 'test_patches.txt'))
test_patches = open(r"D:\L_pipe\liver_open\liverseg-2017-nipsws\det_DatasetList\test_patches.txt")
#test_patches = open(r"D:\L_pipe\liver_open\liverseg-2017-nipsws\detection_results\det_lesion_ck\hard_results.txt")
#read each line of test_patches.txt


if test_patches is not None: 
    test_patches_lines = test_patches.readlines()

a = 0 
id_img_tracker_list = []
bb_boxes = []
#print(test_patches_lines)
#go through each line of test_patches
for i in test_patches_lines:
    print(i)
    #split the variables on the spaces between them 
    line = i.split(' ')
    #pull the variables from each line 
    ##### if the text file is test_patches
    img_volumes, mina, minb, aux, n = line 
    #print("work plz", img_volumes)
    images_volumes, id_img = img_volumes.rsplit("/")
    #### if hard_results is used
    #img_volumes, mina, minb = line 
    #id_img = img_volumes
    print(id_img)
    id_patient, img = id_img.rsplit("\\")
    if not os.path.exists(bb_path_output + '\\' + id_patient):
        os.mkdir(bb_path_output + '\\' + id_patient)
    #find the maxes of the sample bounding boxes 
    mina = int(float(mina))
    minb = int(float(minb))
    maxa= mina + 80 
    maxb= minb + 80
    #check to see if the new id_img being read in, is the same as the last one 
        #if so, then we'll continue adding the coordinates to our bboxes list
    
    if id_img in id_img_tracker_list: 
        # make our bbox based off the coordinates 
        bbox = [mina, maxa, minb, maxb]
        #input coordinates that'll eventually go to the B_Box Visualizer code 
        bb_boxes.append(bbox)
    #if our id_img is not a repeat of a previous slice then we must be on to a new slice 
    else:
        #so we draw the last patient if there is a last patient
        if a != 0: 
            ## .mat
            curr_img= os.path.join(matlab_png_patient_folder,id_img_tracker_list[-1]) + '.png'
            # Visualize each box for each slice 
            ## PIL to read the path curr)img
            ## PIL path to png -> python object
            image = Image.open(curr_img).convert('RGB')
            data_array = np.asarray(image)
            ## PNG to numpy 
            ## numpy to bounding boxes
            ## numpy array as PNG
            print(bb_boxes)
            print(type(bb_boxes))
            print(bb_boxes[0])
            print(type(bb_boxes[0]))
            data = bbv.draw_multiple_rectangles(data_array, bb_boxes, bbox_color= (239,15,15), thickness=2, is_opaque=False )
            bb_boxes = []
            #image2 = Image.fromarray(data)
            ## bb_path_output
            saving_path = bb_path_output + '\\'+ id_img + '.png'
            Image.fromarray(data).save(saving_path)
        #add this new id_img to the tracker_list 
        id_img_tracker_list.append(id_img)
        #after we append our first image we set this to 1 to let the code know that we can start drawing now
        a = 1 
        #properly make this new slice's first bbox 
        bbox = [mina, maxa, minb, maxb]
        #add first bbox for the new slice 
        #input coordinates to the B_Box Visualizer code 
        bb_boxes.append(bbox)