import os
import pandas as pd

def listSort():
    images_volumes_fold = os.path.join(root_dataset, images_volumes)
    images_volumes_patID = os.listdir(images_volumes_fold)
    images_volumes_patID.sort()
    images_volumes_patID = sorted(images_volumes_patID, key=len)

    item_seg_fold = os.path.join(root_dataset, item_seg)
    item_seg_patID = os.listdir(item_seg_fold)
    item_seg_patID.sort()
    item_seg_patID = sorted(item_seg_patID, key=len)

    liver_seg_fold = os.path.join(root_dataset, liver_seg)
    liver_seg_patID = os.listdir(liver_seg_fold)
    liver_seg_patID.sort()
    liver_seg_patID = sorted(liver_seg_patID, key=len)


    all_paths = []

    for i in range(len(images_volumes_patID) - 1):

        all_patient_data = [[], [], []]

        if images_volumes_patID[i] != ".DS_Store":
            patient_path = os.path.join(images_volumes_fold, images_volumes_patID[i])
            patient_slice = os.listdir(patient_path)
            patient_slice.sort()
            patient_slice = sorted(patient_slice, key=len)
            for pat_slice in patient_slice:
                pat_slice_path = os.path.join(patient_path, pat_slice).split(os.path.sep)[3:]
                pat_slice_path = os.path.sep.join(pat_slice_path)
                all_patient_data[0].append(pat_slice_path)

        if item_seg_patID[i] != ".DS_Store":
            patient_path = os.path.join(item_seg_fold, item_seg_patID[i])
            patient_slice = os.listdir(patient_path)
            patient_slice.sort()
            patient_slice = sorted(patient_slice, key=len)
            for pat_slice in patient_slice:
                pat_slice_path = os.path.join(patient_path, pat_slice).split(os.path.sep)[3:]
                pat_slice_path = os.path.sep.join(pat_slice_path)
                all_patient_data[1].append(pat_slice_path)

        if liver_seg_patID[i] != ".DS_Store":
            patient_path = os.path.join(liver_seg_fold, liver_seg_patID[i])
            patient_slice = os.listdir(patient_path)
            patient_slice.sort()
            patient_slice = sorted(patient_slice, key=len)
            for pat_slice in patient_slice:
                pat_slice_path = os.path.join(patient_path, pat_slice).split(os.path.sep)[3:]
                pat_slice_path = os.path.sep.join(pat_slice_path)
                all_patient_data[2].append(pat_slice_path)

        all_paths.append(all_patient_data)

    return all_paths

# all_paths = [pat0, pat1, pat2, pat3, pat4, ... , pat130]
    # patXYZ = [[images_volumes], [item_seg], [liver_seg]]
        # images_volumes = all paths for patientXYZ from that folder
        #       item_seg = all paths for patientXYZ from that folder
        #      liver_seg = all paths for patientXYZ from that folder


# Makes each individual line for training file
def makeTrainLine(list):
    lineList = []
    line = ''

    for pat in range(int(4*(len(list))/5)):
        for patSlice in range(len(list[pat][1]) - 2):
            line = line + str(list[pat][0][patSlice]) + " " + str(list[pat][1][patSlice]) + " "  + str(list[pat][2][patSlice])
            line = line + str(list[pat][0][patSlice + 1]) + " " + str(list[pat][1][patSlice + 1]) + " "  + str(list[pat][2][patSlice + 1])
            line = line + str(list[pat][0][patSlice + 2]) + " " + str(list[pat][1][patSlice + 2]) + " "  + str(list[pat][2][patSlice + 2])
            lineList.append(line)
            line = ""

    return lineList

# Makes each individual line for testing file
def makeTestLine(list):
    lineList = []
    line = ''

    for pat in range(len(list)):
        for patSlice in range(len(list[pat][1]) - 2):
            line = line + str(list[pat][0][patSlice]) + " " + str(list[pat][1][patSlice]) + " "  + str(list[pat][2][patSlice])
            line = line + str(list[pat][0][patSlice + 1]) + " " + str(list[pat][1][patSlice + 1]) + " "  + str(list[pat][2][patSlice + 1])
            line = line + str(list[pat][0][patSlice + 2]) + " " + str(list[pat][1][patSlice + 2]) + " "  + str(list[pat][2][patSlice + 2])
            lineList.append(line)
            line = ""

    return lineList

# Makes each training TXT file
def makeTrainTXT():
    lineList = makeTrainLine(listOfLists)

    TXTfile = open("../../seg_DatasetList/training_volume_OV_OG.txt", "w")

    for line in lineList:
        TXTfile.write(str(line) + "\n")

    TXTfile.close()

# Makes each testing TXT file
def makeTestTXT():
    lineList = makeTestLine(listOfLists)

    TXTfile = open("../../seg_DatasetList/testing_volume_OV_OG.txt", "w")
    
    for line in lineList:
        TXTfile.write(str(line) + "\n")

    TXTfile.close()

# Makes training DF
def makeTrainDF():
    lineList = makeTrainLine(listOfLists)

    return pd.DataFrame(lineList)

# Makes testing DF
def makeTestDF():
    lineList = makeTestLine(listOfLists)

    return pd.DataFrame(lineList)

# Header
if __name__ == "__main__":
    root_dataset = "../../LiTS_database/"

    images_volumes = "images_volumes/"
    item_seg = "item_seg/"
    liver_seg = "liver_seg/"

    listOfLists = listSort()
    makeTestTXT()
    makeTrainTXT()

    testDF = makeTestDF()
    trainDF = makeTrainDF()