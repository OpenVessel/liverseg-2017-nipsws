#Steps: load png, png to numpy, read in coordinates, apply bounding box, apply BB to z-score  

matlab_png_patient_folder = r"D:\L_pipe\liver_open\liverseg-2017-nipsws\LiTS_database\liver_seg"

pos_patches = open(r"D:\L_pipe\liver_open\liverseg-2017-nipsws\det_DatasetList\testing_positive_det_patches.txt")
neg_patches = open(r"D:\L_pipe\liver_open\liverseg-2017-nipsws\det_DatasetList\testing_negative_det_patches.txt")


if pos_patches is not None: 
    pos_patches_lines = pos_patches.readlines()

if neg_patches is not None: 
    neg_patches_lines = neg_patches.readlines()

for i in pos_patches_lines:
    
    line = i.split(' ')
    
    img_volumes, mina, minb = line 
    
    images_volumes, patient, slice_number = img_volumes.rsplit("/")
    
    if patient == '130': 

        mina = int(float(mina))
        minb = int(float(minb))
        maxa= mina + 80 
        maxb= minb + 80

        curr_img= os.path.join(matlab_png_patient_folder, patient, slice_number)+ '.png'

        np_bb = np.asarray(curr_image)

        new_pos_bb = np_bb[mina : maxa, minb : maxb]

        z_pos_bb = stats.zscore(new_pos_bb) 

        #do something with z_pos_bb ^ 

for i in neg_patches_lines:
        
    line = i.split(' ')
    
    img_volumes, mina, minb = line 
    
    images_volumes, patient, slice_number = img_volumes.rsplit("/")
    
    if patient == '130': 

        mina = int(float(mina))
        minb = int(float(minb))
        maxa= mina + 80 
        maxb= minb + 80

        curr_img= os.path.join(matlab_png_patient_folder, patient, slice_number)+ '.png'

        np_bb = np.asarray(curr_image)

        new_neg_bb = np_bb[mina : maxa, minb : maxb]

        z_neg_bb = stats.zscore(new_neg_bb) 

        #do something with z_neg_bb ^ 





#what are we trying to do: 
# - we still need to run z-score & statistics on z-score (variance, whole z-score array, max z-score, # of pixels with z-score > +/- 1.5 ) but also on the HU values of the 80 x 80 positively detected lesions samples 
# - need to find way to get whole lesions so we can run statistics on different lesion types (practice on LITS, implement on Dr. Jeph's Li-Rads) 
# - 