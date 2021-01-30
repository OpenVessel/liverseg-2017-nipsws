### 1/30/2021
## delete script after developing functions for automation and visualizations


import os
from PIL import Image
patient_ID = "105"

def png_to_gif(patient_ID):

    gif_name = patient_ID + 'outputName'
    #pathway_result = r"D:\L_pipe\liver_open\liverseg-2017-nipsws\results\seg_lesion_ck\105"
    pathway_result = r"D:\L_pipe\liver_open\liverseg-2017-nipsws\predict_database\seg_liver_ck\105"
    file_list = os.listdir(pathway_result)
    #print(file_list)
    #list.sort(file_list, key=lambda x: int(x.split('_')[1].split('.png')[0])) # Sort the images by #, this may need to be tweaked for your use case

    list_of_path = []

    for png in file_list:
        new_path = pathway_result + '\\' + png
        list_of_path.append(new_path)
    
    frames = []
    for i in list_of_path:
        new_frame = Image.open(i)
        frames.append(new_frame)
        new_frame.closed 

    frames[0].save('fire3_PIL.gif', format='GIF',
    append_images=frames[1:],
    save_all=True,
    duration=300, loop=0)

png_to_gif(patient_ID)