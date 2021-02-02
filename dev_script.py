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
    #print(list_of_path)
    en_list = []
    print(enumerate(list_of_path))
    en_list = enumerate(list_of_path)
    number = []
    for j in en_list:
        number.append(j[0])
    x = 10
    print(number[-1])
    frames = []
    for i in number:
        
        print(i)
        print(list_of_path[i])
        if i == x+x:
            for j in range(5):
                i = i + 10
        
        new_frame = Image.open(list_of_path[i])
        frames.append(new_frame)
        
        
        if i == number[-1]:
            print("test")
            frames[0].save('D:\L_pipe\liver_open\liverseg-2017-nipsws\fire3_PIL.gif', format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=300, loop=0)
            new_frame.close()
    

png_to_gif(patient_ID)