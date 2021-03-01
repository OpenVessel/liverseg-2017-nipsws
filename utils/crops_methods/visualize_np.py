import numpy as np 


def save_np(np_array):
    
    #going to save to an internal folder called "visualized_functions" 
    output_path = '/Users/alexschweizer/Documents/GitHub/liverseg-2017-nipsws/visualized_data_results'
    new_np_array = np.asarray(np_array)
    np.save(output_path, new_np_array)
    
    return print("numpy array was saved at", output_path) 

