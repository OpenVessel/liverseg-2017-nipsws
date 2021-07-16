
######## DATA 
data_type = 'mat'  ## 'mat' or 'nifti'
data_path = ''
labels_path = ''
preprocessed_data_path = 
data_input_shape = (??,512,512,2) ### number of slices needed ??  

root_process_database = '' ### this is for nifti file construction  

#### Run Organizer 
#### train_Liver_Detection_model()
test_split_size = 0.15

#### Liver_Detection.py
        #### build_LD_model()
dropout_rate = 0.25 

regularizer_choice = 'l1_l2' 
l1_weight = 0.05 
l2_weight = 0.001

        #### load_weights_into_LD_model()
load_weights = False 
weights_file_path = '' 

        #### load_model_into_LD_model()
load_model = False 
model_to_load = ''

        #### compile_LD_model
loss_function = "binary_crossentropy" 
optimizer = "sgd" 
metrics = ["accuracy"]

       #### train_LD_model()
validation_split = 0.2
epochs = 500
batch_size = 32

        #### Callback Selector 
                ## EarlyStopping
early_stopping = True 
monitor = 'val_loss' 
patience = 20
min_delta = 0.1

                ## CustomVerbose() ... a custom callback 
custom_verbose= True 
train_verbose = False #if custom_verbose = False then would you like default verbose (True) or all verbose off (False) when training 

                ## learning_rate_scheduler
learning_rate_scheduler = True

                ## model_checkpoint callback creator. Choices: 'both', 'model', 'weights'
model_and_weights_saved = 'both' 

        #### test_LD_model()
test_verbose = 2

        
