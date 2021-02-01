import os

root_dataset = "../../LiTS_database/"

images_volumes = "images_volumes/"
item_seg = "item_seg/"
liver_seg = "liver_seg/"

# images_volumes/0/1.mat 
# item_seg/0/1.png 
# liver_seg/0/1.png 
# images_volumes/0/2.mat 
# item_seg/0/2.png 
# liver_seg/0/2.png 
# images_volumes/0/3.mat 
# item_seg/0/3.png 
# liver_seg/0/3.png 
# 0.000132 0.028108
# first number: number of voxels that belong to lesion class divided by the total volume 
# second number: same but for the liver class

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

# all_paths = [pat0, pat1, pat2, pat3, pat4, ..., pat131]
    # patXYZ = [[images_volumes], [item_seg], [liver_seg]]
        # images_volumes = all paths for patientXYZ from that folder
        #       item_seg = all paths for patientXYZ from that folder
        #      liver_seg = all paths for patientXYZ from that folder

# TO-DO
    # Make output txt file
    # Easy peasy
