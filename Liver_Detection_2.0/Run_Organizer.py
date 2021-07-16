
from Liver_Detection.py import mat_data_preprocessing, gen_mat_pngs_from_nifti, build_LD_model, load_weights_into_LD_model, compile_LD_model, callback_selector, Custom_Verbose, train_LD_model, test_LD_model, learning_rate_scheduler
from sklearn.model_selection import train_test_split

def preprocess_data(curr_data_type = 'mat', data_path, labels_path, outpath, root_process_database):
    if curr_data_type = 'mat':
        data, labels = mat_data_preprocessing(mat_path = data_path, labeled_data_path = labels_path, outpath_root = outpath)

    if curr_data_type = 'nifti'
        data, labels = gen_mat_pngs_from_nifti(data, root_process_database)

    return data, labels


def train_Liver_Detection_model(data, labels, test_size = 0.2, load_model = False, model_to_load, data_input_shape = (), dropout_rate = 0.25, regularizer_choice = 'l1_l2', l1_weight = 0.05, l2_weight = 0.001, load_weights = False, weights_file_path, loss_function = "binary_crossentropy", optimizer = "sgd", metrics = ["accuracy"], validation_split = 0.2, epochs = 500, batch_size = 64, verbose_train = False, early_stopping = True, monitor = 'val_loss', patience = 20, min_delta = 0.1, learning_rate_scheduler = True, custom_verbose= True, model_and_weights_saved = 'both'):

    ## split data and labels into training and testing sets 
    training_data, testing_data, training_labels, testing_labels = train_test_split(data, labels, test_size = test_size)

    ## build model from loading or through my function 
    if load_model = True: 
        built_model = load_model_into_LD_model(model_to_load = model_to_load)
    else:
        built_model = build_LD_model(input_shape = data_input_shape, dropout_rate = dropout_rate, regularizer_choice = regularizer_choice, l1_weight = l1_weight, l2_weight = l2_weight)
    
    ## initialize weights? 
    if load_weights = True: 
        built_model = load_weights_into_LD_model(built_model, weights_file_path)

    ## compile model 
    compiled_model = compile_LD_model(built_model, loss_function = loss_function, optimizer = optimizer, metrics = metrics)

    ## create callbacks? 
    if early_stopping or learning_rate_scheduler or custom_verbose or model_and_weights_saved != False: 
        callbacks = callback_selector(early_stopping = early_stopping, monitor = monitor, patience = patience, min_delta = min_delta, learning_rate_scheduler = learning_rate_scheduler, custom_verbose= custom_verbose, metrics = metrics, model_and_weights_saved = model_and_weights_saved)
    else: 
        callbacks = [] 

    ## train the model 
    history = train_LD_model(model, training_data, training_labels, validation_split = validation_split, epochs = epochs, batch_size = batch_size, callbacks = callbacks, verbose_train = verbose_train)

    return history, testing_data, testing_labels


def test_Liver_Detection_model(load_model = False, model_to_load, data, labels, verbose_test = 2): 
    
    ## load already trained model? 
    if load_model = True: 
        model = load_model_into_LD_model(model_to_load)

    ## split data and labels into training and testing sets 
    training_data, testing_data, training_labels, testing_labels = train_test_split(data, labels, test_size = test_size)

    ## test the model 
    results = test_LD_model(model, testing_data, testing_labels, verbose_test = verbose_test)

    return results