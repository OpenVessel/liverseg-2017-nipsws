import os
import platform
import pandas as pd


def sort_list(dataset_root, images_volumes, item_seg, liver_seg):
    """
    Arguments
    dataset_root: root folder where the dataset is held 
    images_volumes: name of ground truths folder
    item_seg: name of item (lesion) folder
    liver_seg: name of liver folder

    return: list of lists that will be fed into a pandas DataFrame
    """



    # ground truths
    images_volumes_fold = os.path.join(dataset_root, images_volumes)
    images_volumes_patID = os.listdir(images_volumes_fold)
    images_volumes_patID.sort()
    images_volumes_patID = sorted(images_volumes_patID, key=len)

    # lesions
    item_seg_fold = os.path.join(dataset_root, item_seg)
    item_seg_patID = os.listdir(item_seg_fold)
    item_seg_patID.sort()
    item_seg_patID = sorted(item_seg_patID, key=len)

    # liver
    liver_seg_fold = os.path.join(dataset_root, liver_seg)
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


################################# TXT START ############################


# Makes each training TXT file
def generate_train_txt(lol):
    lineList = makeTrainLine(lol)

    if platform.system() == "Windows":
        TXTname = r"..\..\seg_DatasetList\training_volume_OV.txt"
    else:
        TXTname = "../../seg_DatasetList/training_volume_OV.txt"

    TXTfile = open(TXTname, "w")

    for line in lineList:
        TXTfile.write(str(line) + "\n")

    TXTfile.close()


# Makes each testing TXT file
def generate_test_txt(lol):
    lineList = makeTestLine(lol)

    if platform.system() == "Windows":
        TXTname = r"..\..\seg_DatasetList\testing_volume_OV.txt"
    else:
        TXTname = "../../seg_DatasetList/testing_volume_OV.txt"

    TXTfile = open(TXTname, "w")
    
    for line in lineList:
        TXTfile.write(str(line) + "\n")

    TXTfile.close()

################################# TXT END ############################


# Makes training DF
def generate_training_volume_df(lol):
    lineList = makeTrainLine(lol)

    return pd.DataFrame(lineList)


# Makes testing DF
def generate_testing_volume_df(lol):
    lineList = makeTestLine(lol)

    return pd.DataFrame(lineList)


def generate_test_train_volume_dfs(dataset_root, images_volumes, item_seg, liver_seg):
    """
    Generates the df for seg_liver_test
    """
    
    lol = sort_list(dataset_root, images_volumes, item_seg, liver_seg)
    testing_volume = generate_testing_volume_df(lol)
    training_volume = generate_training_volume_df(lol)

    return training_volume, testing_volume


if __name__ == "__main__":

    if platform.system() == "Windows":
        dataset_root = r"..\..\LiTS_database"
    else:
        dataset_root = "../../LiTS_database"

    images_volumes = "images_volumes"
    item_seg = "item_seg"
    liver_seg = "liver_seg"

    lol = sort_list(dataset_root, images_volumes, item_seg, liver_seg)
    generate_test_txt(lol)
    generate_train_txt(lol)

    testDF = generate_testing_volume_df(lol)
    trainDF = generate_training_volume_df(lol)