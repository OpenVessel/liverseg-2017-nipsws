import numpy as np
from scipy import misc
import os
import glob
import math
import scipy.io 

def compute_3D_bbs_from_gt_liver(config, image_size= 512.0):

    MIN_AREA_SIZE = image_size*image_size 

    ## this file is generated at the end 
    crops_list_name = 'crops_LiTS_gt_2.txt'

    # 1/18/2021 change 
    #utils_path = '../crops_list/'
    utils_path = os.path.join(config.root_folder, 'utils/crops_list/' )
    #results_path = '../../results/'
    results_path = config.get_result_root('results')

    # inputs
    images_path = os.path.join(config.database_root, 'images_volumes') ##QUESTION: are these the matlab files? 
    labels_path = os.path.join(config.database_root,  'item_seg/') ##GT liver mask labels
    labels_liver_path = os.path.join(config.database_root,  'liver_seg/') ## GT liver lesion labels
    liver_results = os.path.join(config.database_root, 'seg_liver_ck/') ## DL predicted liver mask 

    # outputs
    output_images_path_bb = os.path.join(config.database_root, 'bb_images_volumes_alldatabase3_gt_nozoom_common_bb')
    output_labels_path_bb = os.path.join(config.database_root,  'bb_liver_lesion_seg_alldatabase3_gt_nozoom_common_bb')
    output_labels_liver_path_bb = os.path.join(config.database_root,  'bb_liver_seg_alldatabase3_gt_nozoom_common_bb')
    output_liver_results_path_bb = os.path.join(config.database_root, 'liver_results/')

    # This script computes the bounding boxes around the liver from the ground truth, computing
    # a single 3D bb for all the volume. 

    ###checking to see whether the output folder paths exist, if not; then make it
    bb_paths = [output_labels_path_bb, output_images_path_bb, output_labels_liver_path_bb, output_liver_results_path_bb]

    for bb_path in bb_paths:
        if not os.path.exists(bb_path):
            os.makedirs(bb_path)

    ### used as a way to get the integer value of a patient 
    def integerise(value):
        if value != '.DS_Store':
            return int(value) 

    ## If no labels, the masks_folder should contain the results of liver segmentation
    # masks_folders = os.listdir(results_path + 'liver_seg/')

    ### labeled liver mask folder with many patients  
    masks_folders = os.listdir(labels_liver_path) # liver seg 
    
    ### sort the patients 
    sorted_mask_folder = sorted(masks_folders, key=integerise)

    ### opens a new txt file called "crops_LiTS_gt_2.txt" that we will write within 
    crops_file = open(os.path.join(utils_path, crops_list_name), 'w')
    aux = 0 

    sort_by_path = lambda x: int(os.path.splitext(os.path.basename(x))[0])
    ### sort_by_path = lambda x: int(os.path.splitext(os.path.basename(x))[0], print(os.path.splitext(os.path.basename(x)))
    ### .DS_Store change? 

    ### for each ground truth labeled patient 
    for i in range(len(masks_folders)):
        ### if the folder within the patient folder ISN'T in the format of .DS_Store 
        if masks_folders[i] != '.DS_Store': 
            ### if the patient's folder doesn't start with a "." or a tab then continue 
            ### QUESTION: When DO they equal it?
            if not masks_folders[i].startswith(('.', '\t')):
                ### get the patient's path 
                dir_name = masks_folders[i] 
                
                ### gets all of the ground truth liver mask PNGs from i-th patient  
                masks_of_volume = glob.glob(labels_liver_path + dir_name + '/*.png')
                
                ### sort the liver mask png's into proper sequential order 
                file_names = (sorted(masks_of_volume, key=sort_by_path))
                
                ### answers the question of how many slices are within the ground truth liver mask folder? 
                depth_of_volume = len(masks_of_volume)

            ### QUESTION: Didn't we just do this before this function? If we are looping through each mask folder 
                # are we slowing down our computation time a lot by checking this everytime? Are there any differences at all? 
                #Note: I labeled this chunk more nicely though. 
            
            ### ouput paths in a list 
            bb_paths = [output_labels_path_bb, output_images_path_bb, output_labels_liver_path_bb, output_liver_results_path_bb]
            
            ### check to see whether the output paths exist already; if not then create it 
            for bb_path in bb_paths:
                if not os.path.exists(os.path.join(bb_path, dir_name)):
                    os.makedirs(os.path.join(bb_path, dir_name))

            ### QUESTION: what do these represent? why are they so large and small? why is the minimum larger than the maximum? 
            total_maxa = 0
            total_mina = 10000000
            
            total_maxb = 0
            total_minb = 10000000
            
            ### loop through the whole liver mask folder for each patient
            for j in range(0, depth_of_volume):
                ### read the j-th png within the patient's ground truth liver mask folder 
                img = misc.imread(file_names[j])

                ### 255 * 2 = 510, the pngs should be 510 x 510 pixels, 
                # 0 to 255 grey scale 
                ### QUESTION: Does this have to do with normalizing the image? 
                    #what values do the pixels have before this translation?
                    #as we see in the next step, the pixels are supposed to be between 0 & 1 assumedly since they are 
                    #trying to binarize the image and their threshold is 0.5 
                img = img/255.0 
                img[np.where(img > 0.5)] = 1
                img[np.where(img < 0.5)] = 0
                #QUESTION: are a & b lists of coordinates? 
                    #they should be lists (len (a) is called right after) but what does this look like? 
                a, b = np.where(img == 1)
                #print("a:", a, "b:", b)
                
                ###QUESTION: what data type is "a"? 
                ### If there are multiple truth values (there are values > 0.5 on the normalized pixel scale) then ...
                if len(a) > 0:
                    
                    ###find min & max of both a & b coordinate positions 
                        #QUESTION: would the max values of a & b be the furthest truth values (1's; any pixel value > 0.5 
                            # on the normalized grey matter pixel scale) from the center 
                            # and therefore we would be grabbing the very outer edges of the liver mask so that we can make a clean liver mask bounding box 
                    maxa = np.max(a)
                    maxb = np.max(b)
                    mina = np.min(a)
                    minb = np.min(b)
                    
                    ###total_maxa & total_maxb = 0 & total_mina & total_minb = 1,000,000
                    ### see the questions that I have for when total_maxs & total_mins were made 
                    if maxa > total_maxa:
                        total_maxa = maxa
                    if maxb > total_maxb:
                        total_maxb = maxb
                    if mina < total_mina:
                        total_mina = mina
                    if minb < total_minb:
                        total_minb = minb

            ### QUESTION: Do we have to do this first process again? (besides refiguring the img into a new image)      
            for j in range(0, depth_of_volume):
                img = misc.imread(file_names[j])
                img = img/255.0
                img[np.where(img > 0.5)] = 1
                img[np.where(img < 0.5)] = 0

                a, b = np.where(img == 1)
                
                if len(a) > 0:
                    ### builds off of our last calculations and makes a bounding box? 
                        ### this seems to say that the new_img will keep* anything that is between the total_mins & maxs 
                    ###QUESTION: what is the difference between new_img and o_new? (new_img) is only called in the DEPRECATED section
                    new_img = img[total_mina:total_maxa, total_minb:total_maxb]
                    ### print("new_img", new_img)
                    

                # elements of txt line 
                ### gets the current png file we are working and splits the .png part off so we can refer to this specific patient in other formats
                current_file = file_names[j].split('.png')[0]
                    #for exploratory purposes
                    #current_file_1 = file_names[j].split('.png')[1]  
                    ##print("current_file:", current_file)
                    ##print("current_file with file_names[j].split('.png')[1]:", current_file_1)

                if config.debug:
                    print("current file ->",current_file)
                
                ### from the unchanged* current file: make the png, matlab file, and ??? 
                png = os.path.basename(current_file) + '.png'
                mat = os.path.basename(current_file) + '.mat'
                ###QUESTION: what does the -1 do? and is this supposed to create a folder?
                    ## I believe this is trying to get the liver_seg path by basically going backwards and deleting the 
                    ## already defined liver_seg/ part in the path??? 
                    ## similar to doing cd .. kind of thing but I know I'm missing comprehensive info here
                liver_seg = current_file.split('liver_seg/')[-1]
                
                ###QUESTION: why 500 and what does [0] mean? and of what are we looking through? 
                #print("len(np.where(img == 1)[0])", len(np.where(img == 1)[0]))
                #print("np.where(img == 1)[0]", np.where(img == 1)[0])
                if len(np.where(img == 1)[0]) > 500:

                    # constants
                    area = 1
                    ###QUESTION: what's the purpose of the zoom & aux and why do they divide by 1? 
                    zoom = math.sqrt(MIN_AREA_SIZE/area)
                    aux = 1

                    # write to crops txt file 
                    ###(this is an ouput correct?)
                    ###QUESTION: what does aux represent & what is liver_seg? 
                        ## aux seems to be a binary operator that just represents whether the metric of: 
                            ## "len(np.where(img == 1)[0]) > 500" was true or not 
                    ### Note: do we really still want to write to this crops_file text file if we are transitioning out of the text file system?
                    line = ' '.join([str(x) for x in [liver_seg, aux, total_mina, total_maxa, total_minb, total_maxb]])
                    crops_file.write(line + '\n')

                    ######### apply 3Dbb to files ##########
                    ### Note: really like this style of debugging.
                    ### if debugging, we output the images path (matlab files?), directory name, new matlab file, & new BB png file 
                    if config.debug:
                        print("images_path",images_path)
                        print("dir_name", dir_name)
                        print("mat", mat)
                        print("png", png)

                    # .mat
                    ### QUESTION: what does 'section' do? 
                        # section is the name of the key from within a matlab dictionary in which we want to pull from and write to
                    ### makes a numpy array of the original matlab file input into this function 
                    original_img = np.array(scipy.io.loadmat(os.path.join(images_path, dir_name, mat))['section'], dtype = np.float32)
                    ### altered original image where (I believe) that we crop out everything else outside of the min & max created bounding box 
                        ##QUESTION: what would be outside of the minimum and maximum? 
                    o_new = original_img[total_mina:total_maxa, total_minb:total_maxb] 
                    ### we save the original? matlab file to our OUTPUT path and make the new section variable key = to the new bounded box section
                    scipy.io.savemat(os.path.join(output_images_path_bb, dir_name, mat), mdict={'section': o_new})


                    ### DEPRECATED: masked_original_img is never used
                    masked_original_img = o_new
                    masked_original_img[np.where(new_img == 0)] = 0
                    ###
                
                    ### lesion png, liver png, & results png: 
                        ### 1st- reads in the original png 
                        ### 2nd- makes a new variable to put the bounding box filter on it  
                        ### 3rd- saves it to the respective output path 

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
                    
                    #is this supposed to be a crops_file.write message?  
                    print("Success", "Directory:", liver_results, "Patient:", dir_name, "File:", png)

                ### if (if len(np.where(img == 1)[0]) > 500:) is not true then aux = 0 
                ### QUESTION: again* I don't know what "current_file.split('liver_seg/')[-1]" is actually doing 

                else:
                    aux = 0
                    crops_file.write(current_file.split('liver_seg/')[-1]  + ' ' + str(aux) + '\n')

### closes our output text file that we just wrote to 
###QUESTION": exactly what did we write? 
    crops_file.close()

### if we are running from main, import config and run the configuration that was set within main 
if __name__ =='__main__':
    from config import Config
    config = Config()
    compute_3D_bbs_from_gt_liver(config)