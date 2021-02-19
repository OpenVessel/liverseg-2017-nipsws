import numpy as np
from scipy import misc
import os
import scipy.io
from PIL import Image


def filter(base_root, config, crops_list='crops_LiTS_gt.txt', input_config='masked_out_lesion', results_list='detection_lesion_example', th=0.5):

    # crops_list = base_root + 'utils/crops_list/' + crops_list
    # results_list = base_root + 'detection_results/' + results_list + '/soft_results.txt'
    
    print("crops_list before" , crops_list)
    print("results_list before" , results_list)
    crops_list = os.path.join( base_root, 'utils/crops_list/', crops_list) ## crop list of ground truths
    results_list = os.path.join( base_root , 'detection_results/' , results_list , 'soft_results.txt')
    print("crops_list after" , crops_list)
    print("results_list after" , results_list)

    if crops_list is not None:
        with open(crops_list) as t:
            crops_lines = t.readlines()

    print("input_config before use ", input_config)
    input_results_path = os.path.join(base_root , 'results/' , input_config) ###
    output_results_path = os.path.join(base_root , 'results/det_' + input_config) ###
    print("input_results_path after ", input_results_path)
    print("output_results_path after ", output_results_path)

    if not os.path.exists(os.path.join(output_results_path)):
        os.makedirs(os.path.join(output_results_path))

    if results_list is not None:
        with open(results_list) as t:
            results_lines = t.readlines()

## hard range input assoicated with error on line 60 #IOError:
## 105 131
## what are the inputs to this?
    for i in range(config.patient_range[0], config.patient_range[1] + 1):
        if i != 106: ## we did this skip of 106 because we test code on 1/19/2021
            print(i)
            folder_name = str(i)
            images = []
            nm = folder_name + '/'
            for x in results_lines:
                if nm in x:
                    images.append(x)

            slices_names = []

            if not os.path.exists(os.path.join(output_results_path, folder_name)):
                os.makedirs(os.path.join(output_results_path, folder_name))

            for j in range(len(images)):
                slices_names.append(images[j].split()[0])

            unique_slices_names = np.unique(slices_names)
            for x in range(len(unique_slices_names)):
                total_mask = []
                for l in range(len(slices_names)):
                    if slices_names[l] == unique_slices_names[x]:
                        if float(images[l].split()[3]) > th:
                            aux_mask = np.zeros([512, 512])
                            x_bb = int(float(images[l].split()[1]))
                            y_bb = int(float(images[l].split()[2].split('\n')[0]))
                            aux_name = images[l].split()[0] + '.png'

                            #IOError: [Errno 2] No such file or directory: 'D:\\L_pipe\\liver_open\\liverseg-2017-nipsws\\results/masked_out_seg_lesion_ck\\106/359.png'
                            total_patch = (np.array(Image.open(os.path.join(input_results_path, aux_name)), dtype=np.uint8))/255.0

                            cropped_patch = total_patch[x_bb: (x_bb + 80), y_bb:(y_bb + 80)]
                            aux_mask[x_bb: (x_bb + 80), y_bb:(y_bb + 80)] = cropped_patch
                            total_mask.append(aux_mask)
                if len(total_mask) > 0:
                    if len(total_mask) > 1:
                        summed_mask = np.sum(total_mask, axis=0)
                    else:
                        summed_mask = np.array(total_mask)[0]

                    thresholded_total_mask = np.greater(total_mask, 0.0).astype(float)
                    summed_thresholded_total_mask = np.sum(thresholded_total_mask, axis= 0)
                    summed_thresholded_total_mask[summed_thresholded_total_mask == 0.0] = 1.0
                    summed_mask = np.divide(summed_mask, summed_thresholded_total_mask)
                    summed_mask = summed_mask*255.0
                    name = unique_slices_names[x].split('.')[0] + '.png'
                    scipy.misc.imsave(os.path.join(output_results_path, name), summed_mask)

    for i in range(len(crops_lines)):
            result = crops_lines[i].split(' ')
            if len(result) > 2:
                id_img, bool_zoom, mina, maxa, minb, maxb  = result
            else:
                id_img, bool_zoom = result
            # 104 is specific to the dataset train/test split 
            training_set_cutoff = 104
            if int(id_img.split(os.path.sep)[0]) > training_set_cutoff:
                if not os.path.exists(os.path.join(output_results_path, id_img + '.png')):
                    mask = np.zeros([512, 512])
                    misc.imsave(os.path.join(output_results_path, id_img + '.png'), mask)
