import tensorflow as tf 
from tf.keras.models import Sequential
from tf.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
from tf.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateScheduler
import LD_Config as config
import pandas as pd 

from datetime import datetime, date
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image


#### THINGS TO RESOLVE
    # when you do model.fit inside of a function, do you need to return the object related to it or does the action itself carry forward??  
        #ex: history = model.fit() --> return history or just do model.fit() and return nothing?? 

    # when and why do we choose to do: 
        # which loss function? 
        # which optimizer? 
        # which regularizer? 
        # when and how much should you do BatchNormalization and Dropout layers? 
        # how many neurons? 
        # how many layers? 
        # which padding? 
        # what kernel_size for pooling? 

    # what is the input_shape coming from a Flatten layer? 

    # get date and time 

    # DICOM --> Numpy 



###################

# this should probably be compiled in the train_Liver_Detection file 
    #helper functions
    #from Data_Conversion import dicom_to_numpy

    # Data ingestion 

# def my_train_test_split(data, labels, test_size = 0.2):
#     training_data, testing_data, training_labels, testing_labels = test_train_split(data, labels, test_size)

#     return training_data, testing_data, training_labels, testing_labels


# Data model 
    # what is our goal for the Liver Detection model? 
        #Goal = classify images as Liver or Not Liver  

# parameters that you might want to change before running our models would be: learning_rate, epochs, optimizer 

def build_LD_model(input_shape = (512,512,2), dropout_rate = 0.25, regularizer_choice = 'l1_l2', l1_weight = 0.05, l2_weight = 0.001):
    """
    regularizer_choice = choice of 'l1', 'l2', 'l1_l2'
    dropout_rate = set the dropout rate for the Dropout layer, which should be a number 
                    between 0 and 1 that represents the probability of any random weight connection becoming 0 in each pass through???? 
    l1_weight = set the rate of the l1 and/or l2 regularizer (if applicable and wanted)
    """

    if regularizer_choice = 'l1_l2': 
        regularizer = tf.keras.regularizers.l1_l2(l1_ = l1_weight, l2 = l2_weight)
    
    elif regularizer_choice = 'l1':
        regularizer = tf.keras.regularizers.l1(l1_weight)

    elif regularizer_choice = 'l2':
        regularizer = tf.keras.regularizers.l2(l2_weight)
    
    model = Sequential([
        Conv2D(units = 128, activation = 'relu', input_shape = input_shape),
        MaxPooling2D(kernel_size = (2,2), padding = 'VALID'),
        Conv2D(units = 128, activation = 'relu'),
        MaxPooling2D(kernel_size = (2,2), padding = 'VALID'),
        Conv2D(units = 64, activation = 'relu'),
        MaxPooling2D(kernel_size = , padding = 'VALID'),
        Conv2D(units = 64, activation = 'relu'),
        MaxPooling2D(kernel_size = (2,2), padding = 'VALID'),
        Flatten(), 
        ##################### see line 1 
        Dense(units = 128, activation = 'relu', kernel_regularizer = regularizer), ##################### need , input_shape = ()?
        Dense(units = 128, activation = 'relu', kernel_regularizer = regularizer),
        BatchNormalization(), 
        Dropout(dropout_rate), 
        Dense(units = 64, activation = 'relu', kernel_regularizer = regularizer),
        Dense(units = 64, activation = 'relu', kernel_regularizer = regularizer),
        BatchNormalization(), 
        Dropout(dropout_rate),
        Dense(units = 1, activation = 'sigmoid')
    ])

    return model 


def load_weights_into_LD_model(model, file_path):
    model.load_weights(file_path)


######## CHECK TO SEE IF THIS IS RIGHT 

def load_model_into_LD_model(file_path):
    model.load_model(file_path)


def compile_LD_model(model, loss_function = "binary_crossentropy", optimizer = "sgd", metrics = ["accuracy"]):
    """
    model = the model to compile for 
    loss_function(str), optimizer(str), metrics(list of str's) = the parameter to specify (if desired) when compiling 
    """

    model.compile(loss = loss_function, optimizer = optimizer, metrics = metrics)


class Custom_Verbose(CallBack, metrics = ['accuracy']): #####################
    def on_train_begin(self, logs = None): #####################
        print("The training session has begun...")
    
    def on_epoch_begin(self, epoch, logs = None):
        print("Epoch #{} has begun...".format(epoch)) #####################
    
    def on_epoch_end(self, epoch, metrics = metrics, logs = None):
        print("Epoch #{} has ended.".format(epoch))
        print("/tResults for Epoch#{}:".format(epoch)) #####################
        for i in metrics: 
            print("/t/t{}: {:7.2f}".format(i, logs[i]))

    def on_train_end(self, metrics = metrics, logs = None): #####################
        print("Training has ended.")
        print("/tTraining Results:")
        for i in metrics: 
            print("/t/t{}: {:7.2f}".format(i, logs[i]))


def learning_rate_scheduler(epoch):
        initial_lrate = 0.1
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop,  
                math.floor((1+epoch)/epochs_drop))
        return lrate
    

def callback_selector(early_stopping = True, monitor = 'val_loss', patience = 20, min_delta = 0.1, learning_rate_scheduler = True, custom_verbose= True, metrics = ['accuracy'],model_and_weights_saved = 'both'):
    
    callbacks = [] 

    now = datetime.now()

    date_and_time = now.strftime("%d/%m/%Y %H:%M:%S")

    #### ModelCheckpoint() / models_and_weights_saved

    if model_and_weights_saved = 'both': 
        checkpoint = ModelCheckpoint(f'Liver_Detection-version:{date_and_time}') #####################
        callbacks.append(checkpoint)

    elif model_and_weights_saved = 'model':
        checkpoint = ModelCheckpoint(f'Liver_Detection-version:{date_and_time}', save_model_only = True) #####################
        callbacks.append(checkpoint) 

    elif model_and_weights_saved = 'weights':
        checkpoint = ModelCheckpoint(f'Liver_Detection-version:{date_and_time}', save_weights_only = True)
        callbacks.append(checkpoint) 

    #### EarlyStopping / early_stopping 
    if early_stopping = True:
        early_stopping = EarlyStopping(monitor = monitor, patience = patience, min_delta = min_delta) ##################### finish 
        callbacks.append(early_stopping)

    #### LearningRateScheduler / learning_rate_scheduler

    if learning_rate_scheduler = True: 
        learning_rate_scheduler = LearningRateScheduler(learning_rate_function) ##################### finish 
        callbacks.append(learning_rate_scheduler)

    #### Custom_Verbose / custom_verbose
    if custom_verbose = True:
        callbacks.append(Custom_Verbose(metrics)) #####################

##################### get rid of this? 


    # callback_object_list = [early_stopping, learning_rate_scheduler]

    # for i in callback_object_list: 
    #     if i == None:
    #         callback_object_list.pop(i) ##################### correct term "pop"? 
    #     else: 
    #         callbacks.append(i)

    # if early_stopping != False: 
    #     callbacks.append(early_stopping)
    
    # if learning_rate_scheduler != False: 
    #     callbacks.append(learning_rate_scheduler) #####################
    
    return callbacks


def train_LD_model(model, training_data, training_labels, validation_split = 0.2, epochs = 500, batch_size = 64, verbose_train = False):
    """
    """

    model.fit(training_data, training labels, validation_split = validation_split, epochs = epochs, batch_size = batch_size, verbose = verbose_train)


def test_LD_model(model, testing_data, testing_labels, verbose_test = 2):
    """
    """

    model.evaluate(testing_data, testing_labels, verbose = verbose_test)


# Results output 


def mat_data_preprocessing(mat_path, labeled_data_path, outpath_root):
    """
    outpath_root = directory where a folder of unlabeled pngs will be saved 
    """

    outpath = os.path.join(outpath_root, "Liver_PNG_Data")

    if not os.path.exists(outpath):
            os.mkdir(outpath)

    dataframe = pd.DataFrame(columns = ["patient/slice", "png_path", "label"])

    ## matlab to png folder pool 
    for patient in os.listdir(mat_path):
        for img_slice in os.listdir(os.path.join(mat_path, patient)):
            
            id_img= str(patient) + '/' + str(img_slice)
            out_png_path = os.path.join(outpath, id_img)

            ## matlab --> png --> new_folder
            mat_slice_path = os.path.join(mat_path, patient, img_slice)
            mat_array = np.array(loadmat(mat_slice_path)['section'])
            mat_png = Image.fromarray(mat_array).convert('RGB')

            mat_png.save(out_png_path)

            ## analyze and label each ground truth png 
            gt_png = os.path.join(labeled_data_path, patient, img_slice)
            gt_array = image.img_to_array(gt_png)

            liver_count = np.count_nonzero(gt_array)

            if liver_count > 0:
                if liver_count > 100: 
                    label = 1 
                else: 
                    label = (liver_count/100)/2
            else:
                label = 0 

            dataframe.append(id_img, out_png_path, label)
    
    IDG = ImageDataGenerator( rescale = 1./255 ) 

    #make training dataset from directory path
    ##QUESTION: what does seed do? 
    data = IDG.flow_from_directory(outpath),
                                        shuffle = True, 
                                        target_size = (512, 512),
                                        batch_size = 32,
                                        class_mode = 'binary',
                                        seed = 44)
    
    return data, dataframe[:,2]


            # image = Image.open(os.path.join(liver_seg_path, patient, str(number + ".png")))
            # image = np.array(image)






###########################################################################################

def pngs_from_mat(mat_file_path, liver_seg_path, outpath):
    ## QUESTION: if this is the point of this function then should it be called this? 
    # generate train, test, validation pngs from .mat files
    # end result --> saves pngs to train, test, validation pathing separated by liver and non-liver

    #Parameters: 
        # mat_file_path: the path that contains each patients .mat files 
        # outpath: folder path to save the generated pngs

        ## QUESTION: can we get this another way or a more simple way? 
        # numpatients: the number of patients used to train the detection model (from 1 - 131)

    liver_train = os.path.join("Train", "Liver")
    liver_validation = os.path.join("Test", "Liver")
    nl_train = os.path.join("Train", "Non-Liver")
    nl_validation = os.path.join("Test", "Non-Liver")

    data_outpaths = [liver_train, liver_test, nl_train, nl_test]
    for path in data_outpaths:
        if not os.path.exists(path):
            os.mkdir(path)


    #generates a list of randomly generated numbers (patients in range 0,130) 
    #to be used for the train, validation data for detection model
    patient_list = random.sample(range(131), numpatients) 
    #converts patient list values into strings
    patient_list = [str(x) for x in patient_list]


    ##QUESTION: do we have to do this from patient list or can we just do 130? 
        ## why would we take 80% of 25? if this is true then it would have ripple effects in the pair of if/else statements at the bottom of this function 
        # (because we are trying to get the training set cutoff point) 
    cutoff = int( len(patient_list) *.8 )

    

    for patient in os.listdir(mat_file_path): #0-131
        ## QUESTION: I believe this should be "for" patient in patient_list instead of this combo of the above for patient in os.listdir and this if patient
            # else handling needed? 

        if patient in patient_list:

            patient_liver_segpath = os.path.join(liver_seg_path, patient)
            mat_file_patientpath = os.path.join(mat_file_path, patient)
            for file in os.listdir(mat_file_patientpath):
                

                number = file[:-4]
                mat_file = os.path.join(mat_file_patientpath, file) #the individual files
                
                mat_array = np.array(loadmat(mat_file)['section'])
                mat_png = Image.fromarray(mat_array).convert('RGB')

                image = Image.open(os.path.join(liver_seg_path, patient, str(number + ".png")))
                image = np.array(image)
                
                #if there is white present in the image (if the ground truth image contains liver/white pixels)
                if np.count_nonzero(image) != 0: 
                    #print(patient, number)
                    print("patient_list[:cutoff] = ", patient_list[:cutoff])
                    ##Question: should this be another for loop instead of if statement? 
                    if patient in patient_list[:cutoff]: #first 80% of the patients in the file list 
                        print("l t: ",liver_train)
                        x = os.path.join(liver_train, str(number + "_" + patient + ".png"))
                        mat_png.save(x)
                        print("saved")
                    else:
                        print("l v:",liver_validation)
                        y = os.path.join(liver_validation, str(number + "_" + patient + ".png")) 
                        mat_png.save(y)             
                #if the image is all black (not containing liver/white pixels)
                else:
                    ##QUESTION: does this cutoff include or exclude the 80th percent patient when calculated and executed in this range?  
                    if patient in patient_list[:cutoff]:
                        print("nl t: ", nl_train)
                        X = os.path.join(nl_train, str(number +  "_" +patient + ".png"))
                        mat_png.save(X)
                    else:
                        print("nl v: ",nl_validation)
                        Y = os.path.join(nl_validation, str(number +  "_" +patient + ".png"))
                        mat_png.save(Y)

def gen_mat_pngs_from_nifti(niftis_path, root_process_database):
    ## QUESTION: do we want to keep the time elapsed in here? 
    start = time.time()
    
    # path constants
    # niftis_path = 'E:\Datasets\LiTS_liver_lesion\LITS17'
    # root_process_database = '../../output_folder/'

    
    ## Folders to be created
    folder_volumes = os.path.join(root_process_database, 'images_volumes/')
    folder_seg_liver = os.path.join(root_process_database, 'liver_seg/')
    folder_seg_item = os.path.join(root_process_database, 'item_seg/')


    # create non-existent paths
    folder_paths = [root_process_database, folder_volumes, folder_seg_liver, folder_seg_item]
        
    for p in folder_paths:
        if not os.path.exists(p):
            os.mkdir(p)

    ##QUESTION: is this just the format of the nifti file structure? should "if" be "where"
    # filter to only files starting with v (volume) or s (segmentation)
    filenames = [filename for filename in os.listdir(niftis_path) if filename[0] in ('v', 's')]


    for filename in filenames:
        path_file = os.path.join(niftis_path, filename)
        ## QUESTION:  wat??
        index = filename.find('.nii') + 1 # +1 to account for matlab --> python

        if filename[0] == 'v':
            print('Processing Volume {}'.format(filename))
            ##QUESTION: "filename[7:index-1]" what's happening here?  
            folder_volume = os.path.join(folder_volumes, filename[7:index-1])
            volume = nib.load(path_file) # load nifti
            imgs = volume.get_fdata() # get 3d NumPy array

            # clipping HU pixel clipping 
            imgs[imgs<-150] = -150
            imgs[imgs>250] = 250

            ## # equivalent to matlab single()
            imgs = imgs.astype(np.float32) 
            img_max, img_min = (np.max(imgs), np.min(imgs))

            # create folder_volume folder
            img_volume = 255*(imgs - img_min)/(img_max-img_min) #convert HU pixels to grey scale pixels
            if not os.path.exists(folder_volume):
                os.mkdir(folder_volume)
            
            ## QUESTION: do you need to save each slice as a different matlab file
            for k in range(img_volume.shape[2]): # for each slice within the CT scan stack
                section = img_volume[:,:,k]
                filename_for_section = os.path.join(folder_volume, str(k+1) + '.mat')
                scipy.io.savemat(filename_for_section, {'section': section})
        else:
            print('Processing Segmentation {}'.format(filename))
            ## QUESTION: what are we cropping out of filename? 
            folder_seg_item_num = os.path.join(folder_seg_item, filename[13:index-1])
            folder_seg_liver_num = os.path.join(folder_seg_liver, filename[13:index-1])
            segmentation = nib.load(path_file)
            img_seg = segmentation.get_fdata().astype(np.uint8)
            
            ## QUESTION: do we want to still print this out? 
            print(img_seg.shape)

            ## QUESTION: binarize & normalize? 
                # binarize and normalize data
            img_seg_item = img_seg.copy()
            img_seg_liver = img_seg.copy()

            ## QUESTION: aren't these binary masks already? why change 1 to 0 with item_seg but not liver_seg? 
            # create masks
            img_seg_item[img_seg_item == 1] = 0
            img_seg_item[img_seg_item == 2] = 1
            img_seg_liver[img_seg_liver == 2] = 1

            # create dirs
            if not os.path.exists(folder_seg_item_num):
                os.mkdir(folder_seg_item_num)
            if not os.path.exists(folder_seg_liver_num):
                os.mkdir(folder_seg_liver_num)
            
            # save images
            for k in range(0, img_seg_item.shape[2]):
                ## QUESTION: do we still want to print this out? 
                print(filename, ", ", k)
                # item
                    ##QUESTION: why are we flipping? 
                item_seg_section = np.fliplr(np.flipud(img_seg_item[:,:,k]*255)) # flip on both axes

                item_seg_filename = os.path.join(folder_seg_item_num, str(k+1) + '.png')
                im_item = Image.fromarray(item_seg_section)
                im_item.save(item_seg_filename)
                
                # liver
                liver_seg_section = np.fliplr(np.flipud(img_seg_liver[:,:,k]*255))
                liver_seg_filename = os.path.join(folder_seg_liver_num, str(k+1) + '.png')
                im_liver = Image.fromarray(liver_seg_section)
                im_liver.save(liver_seg_filename)


    end = time.time()
    print("Elapsed Time is:", end - start)