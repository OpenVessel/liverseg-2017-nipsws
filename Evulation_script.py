from medpy import metric
from surface import Surface
import glob
import nibabel as nb
import numpy as np
import os


## we have to convert the png results 

## all 75 pngs back into 1-segmentation.nii


###########
###########

def get_scores(pred,label,vxlspacing):
	volscores = {}

	volscores['dice'] = metric.dc(pred,label)
	volscores['jaccard'] = metric.binary.jc(pred,label)
	volscores['voe'] = 1. - volscores['jaccard']
	volscores['rvd'] = metric.ravd(label,pred)

	if np.count_nonzero(pred) ==0 or np.count_nonzero(label)==0:
		volscores['assd'] = 0
		volscores['msd'] = 0
	else:
		evalsurf = Surface(pred,label,physical_voxel_spacing = vxlspacing,mask_offset = [0.,0.,0.], reference_offset = [0.,0.,0.])
		volscores['assd'] = evalsurf.get_average_symmetric_surface_distance()

		volscores['msd'] = metric.hd(label,pred,voxelspacing=vxlspacing)

	return volscores

label_path = ''
prob_path = ''

labels = sorted(glob.glob(label_path+'label*.nii'))
probs = sorted(glob.glob(prob_path+'probs*.nii'))

results = []
outpath = '/data/results.csv'

for label, prob in zip(labels,probs):
    loaded_label = nb.load(label)
    loaded_prob = nb.load(prob)
    
    liver_scores = get_scores(loaded_prob.get_data()>=1,loaded_label.get_data()>=1,loaded_label.header.get_zooms()[:3])
    lesion_scores = get_scores(loaded_prob.get_data()==2,loaded_label.get_data()==2,loaded_label.header.get_zooms()[:3])
    print("Liver dice",liver_scores['dice'], "Lesion dice", lesion_scores['dice'])
    
    results.append([label, liver_scores, lesion_scores])

    #create line for csv file
    outstr = str(label) + ','
    for l in [liver_scores, lesion_scores]:
        for k,v in l.iteritems():
            outstr += str(v) + ','
            outstr += '\n'

    #create header for csv file if necessary
    if not os.path.isfile(outpath):
        headerstr = 'Volume,'
        for k,v in liver_scores.iteritems():
            headerstr += 'Liver_' + k + ','
        for k,v in liver_scores.iteritems():
            headerstr += 'Lesion_' + k + ','
        headerstr += '\n'
        outstr = headerstr + outstr

    #write to file
    f = open(outpath, 'a+')
    f.write(outstr)
    f.close()