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
print(images_volumes_patID)
print()

item_seg_fold = os.path.join(root_dataset, item_seg)
item_seg_patID = os.listdir(item_seg_fold)
item_seg_patID.sort()
item_seg_patID = sorted(item_seg_patID, key=len)
print(item_seg_patID)
print()

liver_seg_fold = os.path.join(root_dataset, liver_seg)
liver_seg_patID = os.listdir(liver_seg_fold)
liver_seg_patID.sort()
liver_seg_patID = sorted(liver_seg_patID, key=len)
print(liver_seg_patID)
print()


line = ''

for i in range(len(images_volumes_patID) - 1):
    if images_volumes_patID[i] != ".DS_Store":
        patient_path = os.path.join(images_volumes_fold, images_volumes_patID[i])
        patient_slice = os.listdir(patient_path)
        patient_slice.sort()
        patient_slice = sorted(patient_slice, key=len)
        for pat_slice in patient_slice:
            pat_slice_path = os.path.join(patient_path, pat_slice)
            print(pat_slice_path)


            
        print(patient_path)

    # if item_seg_patID[i] != ".DS_Store":
    #     patient_path = os.path.join(item_seg_fold, item_seg_patID[i])
    #     print(patient_path)

    # if liver_seg_patID[i] != ".DS_Store":
    #     patient_path = os.path.join(liver_seg_fold, liver_seg_patID[i])
    #     print(patient_path)