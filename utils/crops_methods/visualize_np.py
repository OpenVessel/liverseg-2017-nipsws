import numpy as np 
import os

def save_np(np_array, output_path):
    
    #Get Current working Directory
    currentDirectory = os.getcwd()
    #Change the Current working Directory
        #os.chdir('/home/varun')

    #going to save to an internal folder called "visualized_functions" 
    output_path = '/Users/alexschweizer/Documents/GitHub/liverseg-2017-nipsws/visualized_data_results'
    patient = output_path + '//' + output_patient

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    new_np_array = np.asarray(np_array)
    np.save(output_path, new_np_array)
    
    return print("numpy array was saved at", output_path) 

