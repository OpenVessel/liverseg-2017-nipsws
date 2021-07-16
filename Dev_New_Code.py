
def create_lesion_data(matlabfilepath, lesionpngspathtosave):
    pos_patches = open(r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\det_DatasetList\training_positive_det_patches.txt")
    #neg_patches = open(r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\det_DatasetList\testing_negative_det_patches.txt")
    if pos_patches is not None: 
        pos_patches_lines = pos_patches.readlines()
    # if neg_patches is not None: 
    #     neg_patches_lines = neg_patches.readlines()
    a = 0
    id_img_tracker_list = [] 
    bb_boxes = [] 
    total_rows = [] 
    ### develop metrics 
    for i in pos_patches_lines:
        line = i.split(' ')
        img_volumes, mina, minb = line 
        images_volumes, patient, slice_number = img_volumes.rsplit("/")
        #print(type(patient))
        #if patient in ['1','10']:
            #make folders for each patient 
        outfolder = os.path.join(lesionpngspathtosave, patient)
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)
        mina = int(float(mina))
        minb = int(float(minb))
        maxa= mina + 80 
        maxb= minb + 80
        id_img = str(patient + "\\" + slice_number) #slice id
        #to handle multiple lesions/bounding boxes 
        #if same patient and slice then add the new coordinates
        if id_img in id_img_tracker_list: 
            # make our bbox based off the coordinates 
            bbox = [mina, maxa, minb, maxb]
            #input coordinates that'll eventually go to the B_Box Visualizer code 
            bb_boxes.append(bbox)
        #if our id_img is not a repeat of a previous slice then we must be on to a new slice 
        else:
        #so we draw the last patient if there is a last patient
            segmentation_path = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\item_seg"
            ### BASE THE FOLDER STRUCTURE OF OF THIS LOGIC (B/C A SHOULD BE NUMBER OF LESIONS i THINK)
            if a != 0: 
                # print(type(id_img_tracker_list[-1]))
                curr_img = np.array(loadmat(os.path.join(matlabfilepath,str(id_img_tracker_list[-1]))+ '.mat')['section'], dtype = np.float32)
                #from here get the hu values then use those for z score
                rows = []
                for x in bb_boxes:
                    # print("Lesion number: ", bb_boxes.index(x))
                    print("ID_img: ",id_img)
                    # print("     Coordinates: ", x)
                    #create individual lesion folders 
                    outpath = os.path.join(os.path.join(lesionpngspathtosave,patient,str(bb_boxes.index(x))))
                    if not os.path.exists(outpath):
                        os.mkdir(outpath)
                    # print("Outpath: ",outpath)
                    new_pos_bb = curr_img[x[0]:x[1], x[2]:x[3]] #size = (80, 80)
                    # #print(x[0],x[1],x[2],x[3])
                    # #print(new_pos_bb.shape)
                    # #save stuff right here (both statistics and pngs)
                    skew = stats.skew(new_pos_bb.ravel()) #2D array to 1D array with same values
                    shapiro =  stats.shapiro(new_pos_bb)
                    rows.append([id_img,
                    np.mean(new_pos_bb), np.var(new_pos_bb), np.std(new_pos_bb), np.min(new_pos_bb),
                    np.max(new_pos_bb), skew, stats.zscore(new_pos_bb), shapiro])
                    cropped = Image.fromarray(new_pos_bb).convert("RGB")
                    segmented = os.path.join(segmentation_path, str(id_img + ".png"))
                    segmented = Image.open(segmented)
                    segmented = np.array(segmented)
                    segmented = segmented[x[0]:x[1], x[2]:x[3]]
                    segmented = Image.fromarray(segmented)
                    cropped.resize((512,512))
                    cropped.save(os.path.join(outpath,
                        "{}.png".format(slice_number))) ## implement a right here 
                    segmented.save(os.path.join(outpath,
                        "segmented_{}.png".format(slice_number)))              
                    #print(len(rows))
                total_rows.extend(rows) 
                bb_boxes = []
            bbox = [mina, maxa, minb, maxb]
            id_img_tracker_list.append(id_img)
            a = 1 
            bb_boxes.append(bbox)
        #print(total_rows)
    df = pd.DataFrame(total_rows, 
            columns=["path","Mean", " Variance",
                "Standard deviation",  "Minimum",  "Maximum",
                "Skew","Z-score", "Shapiro Statistic"]).to_csv(os.path.join(
                        lesionpngspathtosave, "statistics.csv"))
#create_lesion_data(matlabfilepath, 
    r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\Lesion PNGs")
data = pd.read_csv(r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\Lesion PNGs\statistics.csv")



############ ONLY LESION 
from scipy.io import loadmat
from scipy import stats
import os
import numpy as np
from PIL import Image
import pandas as pd 
##### DATA CREATION 
matlabfilepath = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes"


def create_lesion_data(matlabfilepath, lesionpngspathtosave):
    pos_patches = open(r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\det_DatasetList\training_positive_det_patches.txt")
    #neg_patches = open(r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\det_DatasetList\testing_negative_det_patches.txt")
    
    #get GT lesion & liver paths 
    GT_lesion_path = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\item_seg" 
    GT_liver_path = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\liver_seg"
    

    if pos_patches is not None: 
        pos_patches_lines = pos_patches.readlines()
    # if neg_patches is not None: 
    #     neg_patches_lines = neg_patches.readlines()
    a = 0
    id_img_tracker_list = [] 
    bb_boxes = [] 
    total_rows = [] 
    ### develop metrics 
    for i in pos_patches_lines:
        line = i.split(' ')
        img_volumes, mina, minb = line 
        images_volumes, patient, slice_number = img_volumes.rsplit("/")
        mina = int(float(mina)) 
        minb = int(float(minb)) 
        maxa= mina + 80 
        maxb= minb + 80 
        #print(type(patient))
        #if patient in ['1','10']:
            #make folders for each patient 
        outfolder = os.path.join(lesionpngspathtosave, patient)
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)
        
        
        id_img = str(patient + "\\" + slice_number) #slice id  
        #to handle multiple lesions/bounding boxes 
        #if same patient and slice then add the new coordinates
        if id_img in id_img_tracker_list: 
            # make our bbox based off the coordinates 
            bbox = [mina, maxa, minb, maxb]
            #input coordinates that'll eventually go to the B_Box Visualizer code 
            bb_boxes.append(bbox)

        #if our id_img is not a repeat of a previous slice then we must be on to a new slice 
        else:
        #so we draw the last patient if there is a last patient
            segmentation_path = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\item_seg"
            ### BASE THE FOLDER STRUCTURE OF OF THIS LOGIC (B/C A SHOULD BE NUMBER OF LESIONS i THINK)
            if a != 0: 
                # print(type(id_img_tracker_list[-1])) 
                
                #get lesion/liver masks into array form
                GT_lesion_array = np.array(Image.open(os.path.join(GT_lesion_path,str(id_img_tracker_list[-1])) + ".png"))
                GT_liver_array = np.array(Image.open(os.path.join(GT_liver_path, str(id_img_tracker_list[-1])) + ".png"))
                curr_img = np.array(loadmat(os.path.join(matlabfilepath,str(id_img_tracker_list[-1]))+ '.mat')['section'], dtype = np.float32)
                
                # apply liver mask to CT scan before getting BB's of CT scan 
                curr_only_liver_array = GT_liver_array * curr_img
                np.set_printoptions(threshold=np.inf)
                print(curr_only_liver_array)

                #from here get the hu values then use those for z score
                rows = []
                for x in bb_boxes:
                    # print("Lesion number: ", bb_boxes.index(x))
                    print("ID_img: ",id_img)
                    # print("     Coordinates: ", x)
                    #create individual lesion folders 
                    outpath = os.path.join(os.path.join(lesionpngspathtosave,patient,str(bb_boxes.index(x))))
                    if not os.path.exists(outpath):
                        os.mkdir(outpath)
                    # print("Outpath: ",outpath)
                    new_pos_bb = curr_img[x[0]:x[1], x[2]:x[3]] #size = (80, 80)

                    # only_lesion_start_bb = GT_lesion_array[x[0]:x[1], x[2]:x[3]]
                    # a,b = np.where(only_lesion_start_bb == 1) 
                    # only_lesion_end_bb = new_pos_bb[a,b]
                    # print(a)
                    # print(b)
                    # print(only_lesion_end_bb.shape)

                    # #print(x[0],x[1],x[2],x[3])
                    # #print(new_pos_bb.shape)
                    # #save stuff right here (both statistics and pngs)
                    skew = stats.skew(new_pos_bb.ravel()) #2D array to 1D array with same values
                    shapiro =  stats.shapiro(new_pos_bb)
                    rows.append([id_img,
                    np.mean(new_pos_bb), np.var(new_pos_bb), np.std(new_pos_bb), np.min(new_pos_bb),
                    np.max(new_pos_bb), skew, stats.zscore(new_pos_bb), shapiro])
                    cropped = Image.fromarray(new_pos_bb).convert("RGB")
                    segmented = os.path.join(segmentation_path, str(id_img + ".png"))
                    segmented = Image.open(segmented)
                    segmented = np.array(segmented)
                    segmented = segmented[x[0]:x[1], x[2]:x[3]]
                    segmented = Image.fromarray(segmented)
                    cropped.resize((512,512))
                    cropped.save(os.path.join(outpath,
                        "{}.png".format(slice_number))) ## implement a right here 
                    segmented.save(os.path.join(outpath,
                        "segmented_{}.png".format(slice_number)))              
                    #print(len(rows))
                total_rows.extend(rows) 
                bb_boxes = []
            bbox = [mina, maxa, minb, maxb]
            id_img_tracker_list.append(id_img)
            a = 1 
            bb_boxes.append(bbox)
        #print(total_rows)
    df = pd.DataFrame(total_rows, 
            columns=["path","Mean", " Variance",
                "Standard deviation",  "Minimum",  "Maximum",
                "Skew","Z-score", "Shapiro Statistic"]).to_csv(os.path.join(
                        lesionpngspathtosave, "statistics.csv"))
#create_lesion_data(matlabfilepath, 
    r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\Lesion PNGs")
data = pd.read_csv(r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\Lesion PNGs\statistics.csv")

###################  Create Only Lesion Data (no Bounding Boxes)   ####################################
def create_only_lesion_data(matlabfilepath, lesionpngspathtosave):
    pos_patches = open(r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\det_DatasetList\training_positive_det_patches.txt")
    #neg_patches = open(r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\det_DatasetList\testing_negative_det_patches.txt")
    
    #get GT lesion & liver paths 
    GT_lesion_path = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\item_seg" 
    GT_liver_path = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\liver_seg"
    

    if pos_patches is not None: 
        pos_patches_lines = pos_patches.readlines()
    # if neg_patches is not None: 
    #     neg_patches_lines = neg_patches.readlines()
    a = 0
    id_img_tracker_list = [] 
    bb_boxes = [] 
    total_rows = [] 
    ### develop metrics 
    for i in pos_patches_lines:
        line = i.split(' ')
        img_volumes, mina, minb = line 
        images_volumes, patient, slice_number = img_volumes.rsplit("/")
        mina = int(float(mina)) 
        minb = int(float(minb)) 
        maxa= mina + 80 
        maxb= minb + 80 
        #print(type(patient))
        #if patient in ['1','10']:
            #make folders for each patient 
        outfolder = os.path.join(lesionpngspathtosave, patient)
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)
        
        
        id_img = str(patient + "\\" + slice_number) #slice id  
        #to handle multiple lesions/bounding boxes 
        #if same patient and slice then add the new coordinates
        if id_img in id_img_tracker_list: 
            # make our bbox based off the coordinates 
            bbox = [mina, maxa, minb, maxb]
            #input coordinates that'll eventually go to the B_Box Visualizer code 
            bb_boxes.append(bbox)

        #if our id_img is not a repeat of a previous slice then we must be on to a new slice 
        else:
        #so we draw the last patient if there is a last patient
            segmentation_path = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\item_seg"
            ### BASE THE FOLDER STRUCTURE OF OF THIS LOGIC (B/C A SHOULD BE NUMBER OF LESIONS i THINK)
            if a != 0: 
                # print(type(id_img_tracker_list[-1])) 
                
                #get lesion/liver masks into array form
                GT_lesion_array = np.array(Image.open(os.path.join(GT_lesion_path,str(id_img_tracker_list[-1])) + ".png"))
                GT_liver_array = np.array(Image.open(os.path.join(GT_liver_path, str(id_img_tracker_list[-1])) + ".png"))
                curr_img = np.array(loadmat(os.path.join(matlabfilepath,str(id_img_tracker_list[-1]))+ '.mat')['section'], dtype = np.float32)
                
                # apply liver mask to CT scan before getting BB's of CT scan 
                curr_only_liver_array = GT_lesion_array * curr_img 

                #from here get the hu values then use those for z score
                rows = [] 
                for x in bb_boxes: 
                    # print("Lesion number: ", bb_boxes.index(x))
                    print("ID_img: ",id_img) 
                    # print("     Coordinates: ", x)
                    #create individual lesion folders 
                    outpath = os.path.join(os.path.join(lesionpngspathtosave,patient,str(bb_boxes.index(x))))
                    if not os.path.exists(outpath):
                        os.mkdir(outpath)
                    # print("Outpath: ",outpath)
                    new_pos_bb = curr_img[x[0]:x[1], x[2]:x[3]] #size = (80, 80)

                    # only_lesion_start_bb = GT_lesion_array[x[0]:x[1], x[2]:x[3]]
                    # a,b = np.where(only_lesion_start_bb == 1) 
                    # only_lesion_end_bb = new_pos_bb[a,b]
                    # print(a)
                    # print(b)
                    # print(only_lesion_end_bb.shape)

                    # #print(x[0],x[1],x[2],x[3])
                    # #print(new_pos_bb.shape)
                    # #save stuff right here (both statistics and pngs)
                    skew = stats.skew(new_pos_bb.ravel()) #2D array to 1D array with same values
                    shapiro =  stats.shapiro(new_pos_bb)
                    rows.append([id_img,
                    np.mean(new_pos_bb), np.var(new_pos_bb), np.std(new_pos_bb), np.min(new_pos_bb),
                    np.max(new_pos_bb), skew, stats.zscore(new_pos_bb), shapiro])
                    cropped = Image.fromarray(new_pos_bb).convert("RGB")
                    segmented = os.path.join(segmentation_path, str(id_img + ".png"))
                    segmented = Image.open(segmented)
                    segmented = np.array(segmented)
                    segmented = segmented[x[0]:x[1], x[2]:x[3]]
                    segmented = Image.fromarray(segmented)
                    cropped.resize((512,512))
                    cropped.save(os.path.join(outpath,
                        "{}.png".format(slice_number))) ## implement a right here 
                    segmented.save(os.path.join(outpath,
                        "segmented_{}.png".format(slice_number)))              
                    #print(len(rows))
                total_rows.extend(rows) 
                bb_boxes = []
            bbox = [mina, maxa, minb, maxb]
            id_img_tracker_list.append(id_img)
            a = 1 
            bb_boxes.append(bbox)
        #print(total_rows)
    df = pd.DataFrame(total_rows, 
            columns=["path","Mean", " Variance",
                "Standard deviation",  "Minimum",  "Maximum",
                "Skew","Z-score", "Shapiro Statistic"]).to_csv(os.path.join(
                        lesionpngspathtosave, "statistics.csv"))
#create_only_lesion_data(matlabfilepath, 
    r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\Lesion PNGs")
data = pd.read_csv(r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\Lesion PNGs\statistics.csv")


######## Find the 3D BB of all patients and find the max x, y, & z ########## 

crops_list = open(r"/Users/alexschweizer/Documents/GitHub/liverseg-2017-nipsws/utils/crops_list/crops_LiTS_gt.txt")
outputCSVpath = r"/Users/alexschweizer/Documents/..."


def Stats_for_3D_BBs(crops_list):

    BB_2D_Coords = []
    
    BB_Z_Coords = []
    Start_Z_Tracker = []

    Liver_Lengths = [] 
    Liver_Length = 0
    a = 0 
    
    patient_tracker = []
    
    if os.path.exists(crops_list) is True: 
        lines = crops_list.readlines()

    for line in lines: 
        
        line = line.split()

        if len(line) > 2: 

        int(id_img), int(aux), int(minX), int(maxX), int(minY), int(maxY) = line
        #id_img, aux = line

        patient, slice_no = id_img.split("/") 

        if aux == 1: 
            Liver_Length += 1 

            #id_img, aux, minX, maxX, minY, maxY = line
        
            if patient not in patient_tracker: 
                patient_tracker.append(patient)
                
                Start_Z_Tracker.append(slice_no)
                
                bbox = [minX, maxX, minY, maxY]
                minX.append(minX)
                BB_2D_Coords.append(bbox) 
                

        else: 
            ## do something with this data 
            #this assumes that every patient's liver isn't at the end of the image, that there's an aux = 0 after aux = 1  
            if patient in patient_tracker:
                if a == 0:
                    end_z = slice_no - 1 
                    z = [Start_Z_Tracker[0], end_z] 
                    BB_Z_Coords.append(z)
                    Start_Z_Tracker = [] 

                    Liver_Lengths.append(Liver_Length)
                    Liver_Length = 0 
                    
                    a += 1 
                    
            else:
                a = 0




    df = pd.DataFrame(patient_tracker, BB_2D_Coords[i][0] for i in BB_2D_Coords, BB_2D_Coords[i][1] for i in BB_2D_Coords, 
            BB_2D_Coords[i][2] for i in BB_2D_Coords, BB_2D_Coords[i][3] for i in BB_2D_Coords, BB_Z_Coords[i][0], BB_Z_Coords[i][1], Liver_Lengths,
            columns=["patient","Min-X", "Max-X", "Min-Y",  "Max-Y",  "Min-Z", "Max-Z", "Liver Length"]).to_csv(os.path.join(
                        outputCSVpath, "Stats_for_3D_BBs.csv"))

#Stats_for_3D_BBs(crops_list) 


###### Actual Change Preprocessing 

# change the normalization 
# to-do's 
    # find the smallest min-X & Y and the largest max-X & Y 
    # read the matlab PNG's in and convert to numpy 
    # apply bounding boxes on every slice of every patient 
    # output to different dataset folder  


##### Latest Create Lesion Dataset Code 


from scipy.io import loadmat
from scipy import stats
import os
import numpy as np
from PIL import Image
import pandas as pd 
##### DATA CREATION 
matlabfilepath = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes"
import shutil
​
def create_lesion_dataset(matlabfilepath, lesionpngspathtosave):
    pos_patches = open(r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\det_DatasetList\training_positive_det_patches.txt")
    #neg_patches = open(r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\det_DatasetList\testing_negative_det_patches.txt")
​
​
    if pos_patches is not None: 
        pos_patches_lines = pos_patches.readlines()
    # if neg_patches is not None: 
    #     neg_patches_lines = neg_patches.readlines()
​
    a = 0
    id_img_tracker_list = []
    bb_boxes = []
    total_rows = []
​
​
    ### develop metrics 
    for i in pos_patches_lines:
​
        line = i.split(' ')
        
        img_volumes, mina, minb = line 
        
        
        images_volumes, patient, slice_number = img_volumes.rsplit("/")
        #print(type(patient))
        
        #if patient in ['1','10']:
            
            #make folders for each patient 
        outfolder = os.path.join(lesionpngspathtosave, patient)
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)
​
        mina = int(float(mina))
        minb = int(float(minb))
        maxa= mina + 80 
        maxb= minb + 80
        
        
        id_img = str(patient + "\\" + slice_number) #slice id
        #to handle multiple lesions/bounding boxes 
​
        #if same patient and slice then add the new coordinates
        if id_img in id_img_tracker_list: 
            # make our bbox based off the coordinates 
            bbox = [mina, maxa, minb, maxb]
            #input coordinates that'll eventually go to the B_Box Visualizer code 
            bb_boxes.append(bbox)
​
        #if our id_img is not a repeat of a previous slice then we must be on to a new slice 
        else:
        #so we draw the last patient if there is a last patient
            
​
​
            segmentation_path = r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\item_seg"
            
            ### BASE THE FOLDER STRUCTURE OF OF THIS LOGIC (B/C A SHOULD BE NUMBER OF LESIONS i THINK)
            if a != 0: 
                # print(type(id_img_tracker_list[-1]))
                curr_img = np.array(loadmat(os.path.join(matlabfilepath,str(id_img_tracker_list[-1]))+ '.mat')['section'], dtype = np.float32)
                #from here get the hu values then use those for z score
                rows = []
                for x in bb_boxes:
                    
                    # print("Lesion number: ", bb_boxes.index(x))
                    print("ID_img: ",id_img)
                    # print("     Coordinates: ", x)
                    #create individual lesion folders 
                    outpath = os.path.join(os.path.join(lesionpngspathtosave,patient,str(bb_boxes.index(x))))
                    if not os.path.exists(outpath):
                        os.mkdir(outpath)
​
                    # print("Outpath: ",outpath)
​
                    new_pos_bb = curr_img[x[0]:x[1], x[2]:x[3]] #size = (80, 80)
                    # #print(x[0],x[1],x[2],x[3])
                    # #print(new_pos_bb.shape)
                    # #save stuff right here (both statistics and pngs)
                    out = os.path.join(outpath,
                         "{}.png".format(slice_number))
                    
                    skew = stats.skew(new_pos_bb.ravel()) #2D array to 1D array with same values
                    shapiro =  stats.shapiro(new_pos_bb)
                    rows.append([out[-13:],
                    np.mean(new_pos_bb), np.var(new_pos_bb), np.std(new_pos_bb), np.min(new_pos_bb),
                    np.max(new_pos_bb), skew, stats.zscore(new_pos_bb), shapiro])
​
                    
                    cropped = Image.fromarray(new_pos_bb).convert("RGB")
                    
                    segmented = os.path.join(segmentation_path, str(id_img + ".png"))
                    segmented = Image.open(segmented)
                    segmented = np.array(segmented)
                    segmented = segmented[x[0]:x[1], x[2]:x[3]]
                    segmented = Image.fromarray(segmented)
​
                    cropped.resize((512,512))
                    cropped.save(out) ## implement a right here
                    segmented.save(os.path.join(outpath,
                        "segmented_{}.png".format(slice_number)))              
                    
                    #print(len(rows))
                total_rows.extend(rows) 
                bb_boxes = []
                   
            bbox = [mina, maxa, minb, maxb]
            id_img_tracker_list.append(id_img)
​
            a = 1 
            bb_boxes.append(bbox)
​
        #print(total_rows)
​
​
    #ADD SIZE TO THE CSV
    df = pd.DataFrame(total_rows, 
            columns=["path","Mean", " Variance",
                "Standard deviation",  "Minimum",  "Maximum",
                "Skew","Z-score", "Shapiro Statistic"]).to_csv(os.path.join(
                        lesionpngspathtosave, "statistics.csv"))
            
    
#create_lesion_dataset(matlabfilepath, 
#     r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes\Lesion PNGs")




############### Create NEW Lesion Dataset


#To-Do's: 
    # - measure.label to get label matrix 
    # - loop through each labeled connected components 
    # - find min and max coordinates for both x & y 
    # - subtract min and max for both x & y to find the width and height of lesion 
    # 	- use these heights and widths to generate some lesion size statistics 
    # - I would like to do lesion size analysis / generate lesion size statistics on our lesion dataset  
    # 	- do this in order to determine what the best bounding box size should be 
    # - based on determined bounding box size add some width and height to our max and min lesion coordinates so that it's centered
    # - then have to apply the coordinates onto the normal data in numpy arrays

from scipy.io import loadmat
from scipy import stats
import os
import numpy as np
from PIL import Image
import pandas as pd 
import shutil

from skimage.measure import label, regionprops

#gt lesion files, normal PNGs, output folder of cropped PNGs 

gt_lesion_folder_path=  r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\item_seg"
matlab_folder_path =   r"C:\Users\12673\Desktop\Projects\OpenVessel\liverseg-2017-nipsws\LiTS_database\images_volumes"
output_folder_path =  r"C:\Users\12673\Desktop\Projects\OpenVessel\"

def create_centered_lesion_data(gt_lesion_folder_path, matlab_file_path, output_folder_path): 
    
    patients = os.listdir(gt_lesion_folder_path)
    len_x_list = []
    len_y_list = []
    
    for patient in patients: 

        slices = os.listdir(patient)

        for each_slice in slices: 
            GT_lesion_array = np.array(Image.open(os.path.join(gt_lesion_folder_path, patient, each_slice) + ".png"))
            labeled_lesions = label(GT_lesion_array, connectivity = 2)
            
            regionprops= regionprops(labeled_lesions)
            print("regionprops = ", labeled_lesions)
            
            num_lesions = np.max(labeled_lesions)
            print("num_lesions=", num_lesions)

            for i in range(num_lesions - 1): 
                #this is what we'll use to track the lesions inside the excel sheet / dataframe
                id_lesion = str(patient) + "-" + str(each_slice) + "-" +str(i+1)
                id_lesion_list.append(id_lesion)

                #find the coordinates of the labeled lesion
                x,y = np.where(labeled_lesions == (i + 1))
                print("x coordinates of lesion pixels =", x)
                print("y coordinate of lesion pixels =", y)
                print("x & y together =", (x,y))

                z = np.where(labeled_lesions == (i + 1))
                print("z (supposed to represent the number of pixels within lesion)=", z)

                # 50 was used in samble bbs as a threshold for valid lesion slices because we don't want the very bottom of a lesion to be used as training data 
                    # we want the model to find reasonably big enough lesions 
                if len(z) > 50: 

                    max_x = np.max(x)
                    min_x = np.min(x)
                    max_y = np.max(y)
                    min_y = np.min(y)
                    
                    len_x = max_x - min_x 
                    len_y = max_y - min_y 
                    len_x_list.append(len_x)
                    len_y_list.append(len_y)

                    #might have to change the value of 80 after we do lesion size analysis

                    if len_x < 80: 
                        new_max_x = max_x + (80 - len_x)/2 
                        new_min_x = min_x - (80 - len_x)/2 
                    
                    if len_y < 80: 
                        new_max_y = max_y + (80 - len_y)/2 
                        new_min_y = min_y - (80 - len_y)/2 

                    curr_array = np.array(loadmat(os.path.join(matlab_folder_path, patient, each_slice)+ '.mat')['section'], dtype = np.float32)

                    new_bb = curr_array[new_min_x : new_max_x, new_min_y : new_max_y]
                    
                    new_bb_pic = Image.fromarray(new_bb)
                    
                    if not os.path.exists(os.path.join(output_folder_path, patient, each_slice)):
                        os.mkdir(os.path.join(output_folder_path, patient, each_slice))
                    
                    new_bb_pic.save(os.path.join(output_folder_path, patient, each_slice, "cropped_bb_slice#{}_lesion#{}.png".format(each_slice, i+1))) 
    
    total_rows = [id_lesion_list, len_x_list, len_y_list]

    df = pd.DataFrame(total_rows, 
            columns=["id_lesion","Lesion Width", "Lesion Height"]).to_csv(os.path.join(
                        output_folder_path, "Lesion_Size_Stats.csv"))

#create_centered_lesion_data(gt_lesion_folder_path, matlab_file_path, output_folder_path)


########  masterlist #### 
def master_list(root_path):
    #generate nested list for all pathing strings 
    #root_path 
    level_list_1 = [] ## contain patients 0, 1, 2, 3 
    level_list_2 = [] ## individual pathing to each patient
    master_list = []        
    level_list_1 = os.listdir(root_path)        
    if '.DS_Store' in level_list_1:
        level_list_1.remove('.DS_Store')        ## sort this list         
    level_list_1.sort()
    level_list_1 = sorted(level_list_1, key=len)        
    #print(level_list_1)        
    for patient_path_string in level_list_1:
        #print(patient_path_string)
        patient_root_level = os.path.join(root_path, patient_path_string)
        slices_list_path_string = os.listdir(patient_root_level)        
        master_list.append([])                  
        for index in range(len(slices_list_path_string)): 
            slices_path = slices_list_path_string[index] 
            index_i = int(patient_path_string) 
            master_list[index_i].append(slices_path)    
    return master_list


####### Applying GT liver mask 

####### apply GT Liver Mask to Lesion Dataset & just normal liver for K-Means algorithm 
liver_seg_path = r"/home/openvessel/Documents/OpenVessel/LITS_Database/liver_seg_2/liver_seg" 
gt_output_folder = r"/home/openvessel/Documents/OpenVessel/Cropped_Dataset_3" 
images_volumes_path = r"/home/openvessel/Documents/OpenVessel/LITS_Database/images_volumes_2/images_volumes"
def apply_GT_liver_mask(gt_output_path, liver_seg_path, images_volumes_path):
    image_masterlist = master_list(images_volumes_path)    
    patient_no = 0     
    for patient in image_masterlist:         
        for slice_no in patient:            
            #if os.path.exists(os.path.join(liver_seg_path, str(patient_no), slice_no)) and os.path.exists(os.path.join(images_volumes_path, str(patient_no), image_slice_no)):
            slice_no_new, mat = slice_no.split(".")
            original_liver_label = misc.imread(os.path.join(liver_seg_path, str(patient_no), slice_no_new + ".png"))
            original_image_volume = np.array(scipy.io.loadmat(os.path.join(images_volumes_path, str(patient_no), slice_no))['section'], dtype = np.float32)
            if original_liver_label.shape == original_image_volume.shape:
                new_image_volume = original_liver_label * original_image_volume 
                if not os.path.exists(os.path.join(gt_output_folder, "png_images_volumes", str(patient_no))):
                    os.makedirs(os.path.join(gt_output_folder, "png_images_volumes", str(patient_no)))                
                misc.imsave(os.path.join(gt_output_folder, "png_images_volumes", str(patient_no), slice_no_new + ".png"), new_image_volume)        
        patient_no += 1    
#apply_GT_liver_mask(gt_output_folder, liver_seg_path, images_volumes_path)


########

################
from scipy.io import loadmat
#from scipy import stats
import os
import numpy as np
from PIL import Image
import pandas as pd 
#import shutil
from skimage.measure import label #, regionprops, regionprops_table
matlab_folder_path =   r"/home/openvessel/Documents/OpenVessel/LITS_Database/images_volumes_2/images_volumes"
gt_lesion_folder_path= r"/home/openvessel/Documents/OpenVessel/LITS_Database/item_seg" 
output_folder_path =  r"/home/openvessel/Documents/OpenVessel/"
def create_centered_lesion_data(gt_lesion_folder_path, matlab_folder_path, output_folder_path):     
    gt_lesion_master_list = master_list(gt_lesion_folder_path)    
    id_lesion_list = []
    len_x_list = []
    len_y_list = []    
    patient_no = 0
    for patient in gt_lesion_master_list:         
        for each_slice in patient: 
            each_slice_new, png = each_slice.split(".")
            GT_lesion_array = np.array(Image.open(os.path.join(gt_lesion_folder_path, str(patient_no), each_slice_new) + ".png"))
            labeled_lesions = label(GT_lesion_array, connectivity = 2)            
            # region_props= regionprops_table(labeled_lesions)
            # print("regionprops = ", regionprops_table(labeled_lesions))            
            num_lesions = np.max(labeled_lesions)
            print("num_lesions=", num_lesions)   
            for i in range(1, num_lesions + 1): 
                #find the coordinates of the labeled lesion
                x,y = np.where(labeled_lesions == (i))
                # print("x coordinates of lesion pixels =", x)
                # print("y coordinate of lesion pixels =", y)
                # print("x & y together =", (x,y))                
                # z = np.where(labeled_lesions == (i))
                # print("z (supposed to represent the number of pixels within lesion)=", z)
                # print('len(z)=',len(z))         
                # print("np.sum(np.where(labeled_lesions == i+1))/(i+1) =",np.sum(np.where(labeled_lesions == i+1))/(i+1))
                # 50 was used in samble bbs as a threshold for valid lesion slices because we don't want the very bottom of a lesion to be used as training data 
                    # we want the model to find reasonably big enough lesions 
                # a = 0
                # b = 0 
                # for i in x: 
                #     a += 1
                # for d in y: 
                #     b += 1 
                # c = a + b
                print("len(x)=", len(x))
                if len(x) > 50:  
                    #this is what we'll use to track the lesions inside the excel sheet / dataframe
                    id_lesion = str(patient_no) + "-" + str(each_slice_new) + "-" +str(i)
                    id_lesion_list.append(id_lesion)   
                    max_x = np.max(x)
                    min_x = np.min(x)
                    max_y = np.max(y)
                    min_y = np.min(y)                    
                    len_x = max_x - min_x 
                    len_y = max_y - min_y 
                    len_x_list.append(len_x)
                    len_y_list.append(len_y)                    
                    #might have to change the value of 80 after we do lesion size analysis                    
                    if len_x < 80: 
                        new_max_x = max_x + (80 - len_x)/2 
                        new_min_x = min_x - (80 - len_x)/2   
                    else: 
                        new_max_x = max_x + 15 
                        new_min_x = min_x + 15                   
                    if len_y < 80: 
                        new_max_y = max_y + (80 - len_y)/2 
                        new_min_y = min_y - (80 - len_y)/2 
                    else: 
                        new_max_y = max_y + 15 
                        new_min_y = min_y + 15                  
                    curr_array = np.array(loadmat(os.path.join(matlab_folder_path, str(patient_no), each_slice_new)+ '.mat')['section'], dtype = np.float32)                    
                    new_bb = curr_array[new_min_x : new_max_x, new_min_y : new_max_y] 
                    new_gt_bb = GT_lesion_array[new_min_x : new_max_x, new_min_y : new_max_y]                
                    new_bb_pic = Image.fromarray(new_bb)  
                    new_bb_pic = new_bb_pic.convert('RGB')
                    new_gt_bb_pic = Image.fromarray(new_gt_bb)  
                    new_gt_bb_pic = new_gt_bb_pic.convert('RGB')
                    if not os.path.exists(os.path.join(output_folder_path, "New_Lesion_Dataset", "patient-" + str(patient_no), "slice-" + each_slice_new)):
                        os.makedirs(os.path.join(output_folder_path, "New_Lesion_Dataset", "patient-" + str(patient_no), "slice-" + each_slice_new))           
                    if not os.path.exists(os.path.join(output_folder_path, "New_GT_Lesion_Dataset", "patient-" + str(patient_no), "slice-" + each_slice_new)):
                        os.makedirs(os.path.join(output_folder_path, "New_GT_Lesion_Dataset", "patient-" + str(patient_no), "slice-" + each_slice_new))               
                    new_bb_pic.save(os.path.join(output_folder_path, "New_Lesion_Dataset", "patient-" + str(patient_no), "slice-" + each_slice_new, "cropped_bb_slice#" + str(each_slice_new) + "_lesion#" + str(i) + ".png" ))
                    new_gt_bb_pic.save(os.path.join(output_folder_path, "New_GT_Lesion_Dataset", "patient-" + str(patient_no), "slice-" + each_slice_new, "cropped_gt_bb_slice#" + str(each_slice_new) + "_lesion#" + str(i) + ".png" ))
        patient_no += 1
    print("Max Lesion Width: ", np.max(len_x_list))
    print("Average Lesion Width: ", np.average(len_x_list))
    print("Max Lesion Height: ", np.max(len_y_list))
    print("Average Lesion Height: ", np.average(len_y_list))
    df = pd.DataFrame(list(zip(id_lesion_list, len_x_list, len_y_list)), columns=["id_lesion","Lesion Width", "Lesion Height"]).to_csv(os.path.join(
                    output_folder_path, "New_Lesion_Dataset", "Lesion_Size_Stats.csv"))
create_centered_lesion_data(gt_lesion_folder_path, matlab_folder_path, output_folder_path)