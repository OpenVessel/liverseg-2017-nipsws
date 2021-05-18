

import numpy as np
import pydicom
from scipy import io
import os
import glob
from PIL import Image
from nibabel.nifti1 import Nifti1Image
from nibabel.nifti2 import Nifti2Image
import logging
import nibabel as nib

def pngs2nifti( pngs_folder, to_file=False, output_filename=None, version=2):
        """Convert all the pngs in a given folder into one NIFTI
        Args:
            pngs_folder (str): Path to the folder containing pngs.
            to_file (bool, optional): Specify if you want the nifti saved to a file or returned as an object. Defaults to False (returns object)
            output_filename (str, optional): Filename to save nifti in. Should end in .nii. Defaults to None (returns object instead)
            version (int, optional): Specify Nifti1 or Nifti2 file. Defaults to 2.
        """
        data = []
        for f in glob.glob(pngs_folder + '/*.png'):
            logging.debug(f)
            data.append(np.asarray(Image.open(f)))
        data = np.array(data)
        data = np.transpose(data)
        
        # TODO: Nifti headers? affine?
        if version == 1:
            nifti = Nifti1Image(data, affine=None)
        elif version == 2:
            nifti = Nifti2Image(data, affine=None)
        else:
            raise BaseException("Nifti version must be 1 or 2")
        if to_file:
            if output_filename is None:
                raise BaseException("if to_file is True, output_filename must not be None") 
            # nifti.saved(output_filename)
            nifti.to_filename(output_filename)
            nib.save(nifti, output_filename)
            print("saved", output_filename)
            return
        return 


pathway = "D:\L_pipe\liver_open\liverseg-2017-nipsws\LiTS_database\seg_liver_ck"
output = "D:\L_pipe\liver_open\liverseg-2017-nipsws\data_output"
list_of_patients = os.listdir(pathway)

print(list_of_patients)
list_of_paths = []
for patient in list_of_patients:
    path = os.path.join(pathway, patient)
    list_of_paths.append(path)
print(list_of_paths)
# for patient in list_of_patients:
#     nifti = patient + '.nii'
#     output_name = os.path.join(output,nifti)
#     print(output_name)

for path in list_of_paths:
    print(path)
    patient = path[-3:]
    nifti = patient + '.nii'
    output_name = os.path.join(output,nifti)
    pngs2nifti(path, to_file=True, output_filename=output_name, version=2)