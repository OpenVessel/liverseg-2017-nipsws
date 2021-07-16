from Run_Organizer import train_Liver_Detection_model, test_Liver_Detection_model
import LD_Config as config
import argparse


if __name__ =='__main__':
    
    parser = argparse.ArgumentParser(description="Train or Test the Liver Lesion Segmentation Model")
    parser.add_argument('mode', help="'train', 'test', or 'train_and_test' depending on what you wish to do.")
    command_line_input = parser.parse_args()

    if command_line_input.mode = 'train_and_test':
        data, labels = preprocess_data(curr_data_type = config.data_type, data_path = config.data_path, labels_path = config.labels_path, outpath = config.preprocessed_data_path, root_process_database = config.root_process_database)

        history, testing_data, testing_labels = train_Liver_Detection_model(data = data, labels = labels, test_size = config.test_split_size, data_input_shape = config.data_input_shape, dropout_rate = config.dropout_rate, regularizer_choice = config.regularizer_choice, l1_weight = config.l1_weight, l2_weight = config.l2_weight, load_weights = config.load_weights, weights_file_path = config.weights_file_path, loss_function = config.loss_function, optimizer = config.optimizer, metrics = config.metrics, validation_split = config.validation_split, epochs = config.epochs, batch_size = config.batch_size, verbose_train = config.train_verbose, early_stopping = config.early_stopping, monitor = config.monitor, patience = config.patience, min_delta = config.min_delta, learning_rate_scheduler = config.learning_rate_scheduler, custom_verbose = config.custom_verbose, model_and_weights_saved = config.model_and_weights_saved)
        test_Liver_Detection_model(load_model = False, model_to_load = history, data = testing_data, labels = testing_labels, verbose_test = config.test_verbose)

    elif command_line_input.mode = 'train':
        data, labels = preprocess_data(curr_data_type = config.data_type, data_path = config.data_path, labels_path = config.labels_path, outpath = config.preprocessed_data_path, root_process_database = config.root_process_database)
        train_Liver_Detection_model(data = data, labels = labels, test_size = config.test_split_size, data_input_shape = config.data_input_shape, dropout_rate = config.dropout_rate, regularizer_choice = config.regularizer_choice, l1_weight = config.l1_weight, l2_weight = config.l2_weight, load_weights = config.load_weights, weights_file_path = config.weights_file_path, loss_function = config.loss_function, optimizer = config.optimizer, metrics = config.metrics, validation_split = config.validation_split, epochs = config.epochs, batch_size = config.batch_size, verbose_train = config.train_verbose, early_stopping = config.early_stopping, monitor = config.monitor, patience = config.patience, min_delta = config.min_delta, learning_rate_scheduler = config.learning_rate_scheduler, custom_verbose = config.custom_verbose, model_and_weights_saved = config.model_and_weights_saved)

    elif command_line_input.mode = 'test':
        data, labels = preprocess_data(curr_data_type = config.data_type, data_path = config.data_path, labels_path = config.labels_path, outpath = config.preprocessed_data_path, root_process_database = config.root_process_database)
        test_Liver_Detection_model(load_model = True, model_to_load = config.model_to_load, data = data, labels = labels, verbose_test = config.test_verbose)

    else: 
        print("Incorrect input: must be 'train','test', or 'train_and_test'.")

