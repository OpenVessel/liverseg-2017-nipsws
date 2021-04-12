
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
create_lesion_data(matlabfilepath, 
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
create_lesion_data(matlabfilepath, 
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
create_only_lesion_data(matlabfilepath, 
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
        
        line = line.split(' ')

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
        

Stats_for_3D_BBs(crops_list) 
        

            



