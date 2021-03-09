"""
Original code from OSVOS (https://github.com/scaelles/OSVOS-TensorFlow)
Sergi Caelles (scaelles@vision.ee.ethz.ch)

Modified code for liver and lesion segmentation:
Miriam Bellver (miriam.bellver@bsc.es)
"""

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.layers.python.layers import initializers
import sys
from datetime import datetime
import os
import scipy.misc
from PIL import Image
slim = tf.contrib.slim
from tensorflow.contrib.slim.nets import resnet_v1
import scipy.io 
import scipy.misc

DTYPE = tf.float32


### Function from old tensorflow that wan't working (their comments or ours?)
def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
### Ignore it. We can find a way to remove this (their comments or ours?)


# set parameters so that Deconvolutional layers compute bilinear interpolation
# Note: this is for deconvolution without groups 
# this function reassigns some of our "-up" global tf variables into being able to be bilinearly interpolated 
#QUESTION: #what goes into variables? 
    #global variables that come form tf.global_variables_initializer()
def interp_surgery(variables):
    # this will be our outputs 
    interp_tensors = []
    for v in variables:
        #QUESTION: what variables have in their name "-up"? 
        # for all variables with "-up" in their name 
        if '-up' in v.name:
            #we are getting the shape of the variable which should be 4D 
                #k & m are inputs and output channels 
                #h & w are filters that we want to be square  
            h, w, k, m = v.get_shape()
            
            #creates a temporary variable that has the same shape as the variable 
                #  except everything inside are 0's and now they switch the order of the variable shape 
                # instead of h,w,k,m it's now m,k,h,w 
            tmp = np.zeros((m, k, h, w)) 

            # checks to see that m & k are the same (since they are apparently input & output channels)
            if m != k:
                print('input + output channels need to be the same')
                raise
            # filters need to be square so the lengths of the filters need to be the same 
                #raises an error if condition not met 
            if h != w:
                print('filters need to be square')
                raise

            #this makes a 2D* bilinear kernel suitable for upsampling* 
            #QUESTION: what is upsampling? what is bilinear interpolation? 
            #QUESTION: it's supposed to be given an (h,w) size parameter but it is only given h
            up_filter = upsample_filt(int(h))
            
            # the up_filter is a 4D input except the 3D & 4D are 0's since these are the filter lengths
            # what is the difference between tmp = np.zeroes((m,k,h,w)) & tmp[range(m), range(k), :, :]
                # but really the better question is, what's the difference between range(m) & : in displaying/accessing a columns' values
            tmp[range(m), range(k), :, :] = up_filter
            
            #appending the newly assigned variables (that were taken in as inputs) into the output list (interp_sensors)
            #How: changes the shape with tmp.transpose 
            # use_locking = Passing use_locking=True when creating a TensorFlow optimizer, or a variable assignment op, causes a lock to be acquired around the relevant updates to the variable. Other optimizers/assignments on the same variable also created with use_locking=True will be serialized. 2 caveats to keep in mind. 
            # validate_shape = True means that we have to pass in a specified shape of a specific input which is apparently the tmp.transpose variable 
            interp_tensors.append(tf.assign(v, tmp.transpose((2, 3, 1, 0)), validate_shape=True, use_locking=True))
    return interp_tensors


def det_lesion_arg_scope(weight_decay=0.0002):
    """Defines the arg scope.
    Args:
    weight_decay: The l2 regularization coefficient. 
        #Small values of L2 can help prevent overfitting the training data.
    Returns: An arg_scope.
    """
    with slim.arg_scope(
                        #List or tuple of operations to set argument scope
                        #QUESTION: what is slim.conv2d, convolution2d_transpose ? 
                        [slim.conv2d, slim.convolution2d_transpose], 
                        # relu is our activation function 
                        # QUESTION: can we get the leaky activation function or others? 
                        activation_fn=tf.nn.relu,
                        # looks like we are going to start with random "normal" weights
                        # QUESTION: what other options are there for starting weights?  
                        weights_initializer=tf.random_normal_initializer(stddev=0.001),
                        # QUESTION: what do different weight decays do to the weight regularizer and what are our options? 
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        # QUESTION: what is this and what are our options?  
                        biases_initializer=tf.zeros_initializer,
                        # QUESTION: what is this? and what are our options? 
                        biases_regularizer=None,
                        # padding = SAME means we will pad the input with the needed extra 0's in order to make sure all values within the input are accounted for 
                            # https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
                        padding='SAME') as arg_sc:
        return arg_sc
        
    #Defines the binary cross entropy loss        
def binary_cross_entropy(output, target, epsilon=1e-8, name='bce_loss'):
    """Defines the binary cross entropy loss
    Args:
    output: the output of the network
    target: the ground truth
    epsilon: what does the variable epsilon 
    name: what do you want to name the output?? 
    Returns:
    A scalar with the loss, the output and the target
    """
    #cast the ground truth target value in the format of a float 
    target = tf.cast(target, tf.float32)
    #cast the output of the deep learning model variable in the format of a float 
    #QUESTION: what is tf.squeeze? 
    output = tf.cast(tf.squeeze(output), tf.float32)
    
    # name the output ? 
    with tf.name_scope(name):
        # calculate the loss based off of binary cross-entropy mathematics 
        #QUESTION: what is tf.reduce_mean? describe the math going on here.
        return tf.reduce_mean(-(target * tf.log(output + epsilon) +
                              (1. - target) * tf.log(1. - output + epsilon))), output, target

#Pre-process the image to adapt it to network requirements
# sample bounding boxes are manipulated here 
def preprocess_img(image, x_bb, y_bb, ids=None):
    """Preprocess the image to adapt it to network requirements
    Args: 
    image: Image we want to input the network (W,H,3) numpy array (W= width, H= height, 3 = ??)
    x_bb: ?? 
    y_bb: ?? 
    ids: number of flips that you want the image to go through 
    Returns: 
    Image ready to input the network (1,W,H,3) (1 = ???)
    """

    #if ids= None then it is assumed we don't want any flips 
    if ids == None:
        #makes an array of 1's in the shape of the width of the image 
        ids = np.ones(np.array(image).shape[0])

    # this variable is a list that creates a list of lists that has the same width as the image 
        # this seems to create a 2d array of lists 
    images = [[] for i in range(np.array(image).shape[0])]
    
    # for loop that goes through the length of the width of the image 
    #QUESTION: what does j really represent? are we going through the width of the image? 
    for j in range(np.array(image).shape[0]):
        
        #QUESTION: why is this a static 3? for what reason 3? 
            # is it about 3 dimensions of the image? 
        for i in range(3):
            #get matlab file into a numpy array 
            #QUESTION: how does image[j] & x_bb[j] work? 
            aux = np.array(scipy.io.loadmat(image[j])['section'], dtype=np.float32)
            
            #cropping the data arrays into proper bounding box shape (80 x 80)
            crop = aux[int(float(x_bb[j])):int((float(x_bb[j])+80)), int(float(y_bb[j])): int((float(y_bb[j])+80))]
            """Different data augmentation options""" 
            # I believe this section is set up like this because our txt files that come out of sample_BBs 
            # have the same information patient and slice information and coordinates for as many data augmentation options we wanted at the time 
            # this is shown within this function by the fact that they are all elif statements and will only do that data augmentation option that is correlated with it's number and won't do all of the data augmentations at once.  
            #QUESTION: how are we using this function? are we looping through the sample BB generated text files?
                # 
            if id == '2':
                crop = np.fliplr(crop)
            elif id == '3':
                crop = np.fliphr(crop)
            elif id == '4':
                crop = np.fliphr(crop)
                crop = np.fliplr(crop)
            elif id == '5':
                crop = np.rot90(crop)
            elif id == '6':
                crop = np.rot90(crop, 2)
            elif id == '7':
                crop = np.fliplr(crop)
                crop = np.rot90(crop)
            elif id == '8':
                crop = np.fliplr(crop)
                crop = np.rot90(crop, 2)

            #add to the images list of lists each crop, the crop for that pixel(?? again what is j?) 
            #QUESTION: How does this account for how many flips we get? 
            #QUESTION: why again do we have a for loop of 3? does this same crop need to exist for 3 dimensions? 
            images[j].append(crop)
    #
    in_ = np.array(images)
    #QUESTION: w
    in_ = in_.transpose((0,2,3,1))
    #QUESTION: why? why these specific numbers? 
    in_ = np.subtract(in_, np.array((104.00699, 116.66877, 122.67892), dtype=np.float32))

    return in_
    
# i believe we are now correlating all of our inputs with their proper labels 
#QUESTION: how are we calling pre-process_labels? what do our labels look like? 
    #probably within a for loop that sends in a ??? each time  
def preprocess_labels(label):
    """Preprocess the labels to adapt them to the loss computation requirements
    Args:
    Label corresponding to the input image (W,H) numpy array
    Returns:
    Label ready to compute the loss (1,W,H,1)
    """

    # this variable is a list that creates a list of lists that has the same width as the image 
        # this seems to create a 2d array of lists in the same shape of the label
    
    labels = [[] for i in range(np.array(label).shape[0])]  
    
    # for loop that goes through the length of the width of the image 
    #QUESTION: again what does j really mean? 
    print("label.shape[0]=",np.array(label).shape[0])

    for j in range(np.array(label).shape[0]):
        #make sure that label is not a multidimensional array 
        if type(label) is not np.ndarray:
            
            for i in range(3):
                print("label=", label)
                print("label[j][i] =", label[j][i])
                #QUESTION: what is a np.uint8?
                aux = np.array(Image.open(label[j][i]), dtype=np.uint8)
                #QUESTION: where does x_bb come from? is this legal? 
                crop = aux[int(float(x_bb[j])):int((float(x_bb[j])+80)), int(float(y_bb[j])): int((float(y_bb[j])+80))]
                #QUESTION: for every ??? append a bounding box  
                labels[j].append(crop)
    
    # we grab the first label from the labels list 
    label = np.array(labels[0])
    # make the 3rd dimension the 1st, the 1st the 2nd, and the 2nd the 3rd 
    label = label.transpose((1,2,0))
    # get the max of the label array
    max_mask = np.max(label) * 0.5
    # returns a True/False array in the same shape as the inputs depending on whether the label 
    label = np.greater(label, max_mask)
    #make a new axis placed as the first dimension now on top of the previous np.greater(label)/True/False array
        #QUESTION: this new 1st dimension is ??? 0's? nothing (just a placeholder)? 
    label = np.expand_dims(label, axis=0)

    return label
        

#defines the network (but what does this entail??)
def det_lesion_resnet(inputs, is_training_option=False, scope='det_lesion'):
    """ Defines the network
    Args:
    inputs: Tensorflow placeholder that contains the input image
    scope: Scope name for the network
    Returns:
    net: Output Tensor of the network
    end_points: Dictionary with all Tensors of the network
    """ 

    #this var_scope defines the variable end_points_collection below it 
    # QUESTION: do all of the variables below this arg_scope get included under this one 
    #    #probably yes  
    # scope is defined as ()
    with tf.variable_scope(scope, 'det_lesion', [inputs]) as sc:
        
        #sets the end point dictionary's name within the variable scope of "det_lesion" (??)
        end_points_collection = sc.name + '_end_points'
        
        #I think we are pulling the argument scope from resnet_v1 
        #resnet_v1 is imported from tf at the top 
            # I believe this could be a variable file 
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            # resnet takes our input of an image and specifies to our net (Tensor) and end_points (Tensor dictionary) trains if we specify that it should be training 
            net, end_points = resnet_v1.resnet_v1_50(inputs, is_training=is_training_option)
            #QUESTION: what does scope = flatten5 mean? 
                # I believe this is a specific type of flattening method that is named as the variable "flatten5"
                # might be a tf.Variable, but where are these variables held? Hint: they are apparently global 
            net = slim.flatten(net, scope='flatten5')
            #our fully connected layer (actual neural network) has an sigmoid activation function (define?), it's weights 
            net = slim.fully_connected(net, 1, 
                                        #this is where a sigmoid activation function is being applied 
                                        #QUESTION: what is a sigmoid? 
                                        activation_fn=tf.nn.sigmoid,
                                        #QUESTION: how does this weights initializer 
                                        weights_initializer=initializers.xavier_initializer(), 
                                        # QUESTION: does scope NAME this scope or does it PULL a scope that already has this name 
                                        scope='output')
                    #QUESTION: what does this do? how? why do we do both this and return outputs from this method? 
                        #SEEMS to be a Tensorflow method to package outputs in *some kind of way* 
            utils.collect_named_outputs(end_points_collection, 'det_lesion/output', net)
        # make a dictionary from the end_points_collection 
        #QUESTION: what is a collection? what is in end_points_collection? 
    end_points = slim.utils.convert_collection_to_dict(end_points_collection)

    #return the network & the end_points collection 
    # Note: describe what these are specifically. 
    return net, end_points

# Initialize the network parameters from the Resnet-50 pre-trained model provided by TF-SLIM\
#Note: describe further 
def load_resnet_imagenet(ckpt_path):
    """Initialize the network parameters from the Resnet-50 pre-trained model provided by TF-SLIM
    Args:
    Path to the checkpoint
    Returns:
    Function that takes a session and initializes the network
    """
    # read in the checkpoint of the Resnet-50 pre-trained model 
    reader = tf.train.NewCheckpointReader(ckpt_path)
    # QUESTION: what is a variable_to_shape_map? and how does it work? 
    var_to_shape_map = reader.get_variable_to_shape_map()
    # create a dictionary 
    #QUESTION: for what? 
    vars_corresp = dict()
    
    # for each *object* in var_to_shape_map (what is an object in var_to_shape_map?)
    for v in var_to_shape_map:
        # if we see bottleneck_v1 or conv1 as our object then continue 
        if "bottleneck_v1" in v or "conv1" in v:
            
            #QUESTION: if v is a string, how does the indexing work exactly here? what does [0] result in here from the output of .get_model_variables? 
            # gets the model variables from v and changes the variable name into our style so that this variable sits inside 
            vars_corresp[v] = slim.get_model_variables(v.replace("resnet_v1_50", "det_lesion/resnet_v1_50"))[0]
        # QUESTION: what does fn stand for? what it going on with this method? what is the relationship between ckpt_path & vars_corresp? 
    init_fn = slim.assign_from_checkpoint_fn(ckpt_path, vars_corresp) 
    return init_fn

# uses Binary cross entropy to calculate the accuracy and defines it as a variable name called accuracy
#QUESTION: what is the difference between this and binary_cross_entropy function we created? 
def my_accuracy(output, target, name='accuracy'):
    """Accuracy for detection
    Args:
    The output and the target
    Returns:
    The accuracy based on the binary cross entropy
    """

    # make the target (prediction) into a float 
    target = tf.cast(target, tf.float32)
    
    #QUESTION: what does tf.squeeze do? what are we sending in for the output parameter? 
    output = tf.squeeze(output)
    
    #return the accuracy score under the name of "accuracy"
    with tf.name_scope(name):
        #QUESTION: what does reduce_mean do? what is the math going on here? 
        return tf.reduce_mean((target * output) + (1. - target) * (1. - output))


def train(dataset, initial_ckpt, learning_rate, logs_path, max_training_iters, save_step, display_step,
           global_step, iter_mean_grad=1, batch_size=1, momentum=0.9, resume_training=False, config=None, finetune=1):

    """Train network
    Args:
    dataset: Reference to a Dataset object instance
    initial_ckpt: Path to the checkpoint to initialize the network (May be parent network or pre-trained Imagenet)
    supervison: Level of the side outputs supervision: 1-Strong 2-Weak 3-No supervision
    learning_rate: Value for the learning rate. It can be number or an instance to a learning rate object.
    logs_path: Path to store the checkpoints
    max_training_iters: Number of training iterations
    save_step: A checkpoint will be created every save_steps
    display_step: Information of the training will be displayed every display_steps
    global_step: Reference to a Variable that keeps track of the training steps (what are training steps?)
    iter_mean_grad: Number of gradient computations that are average before updating the weights
    batch_size: (???) 
    momentum: Value of the momentum parameter for the Momentum optimizer (what is a Momentum optimizer?)
    resume_training: Boolean to try to restore from a previous checkpoint (True) or not (False)
    config: Reference to a Configuration object used in the creation of a Session
    finetune: Use to select to select type of training, 0 for the parent network and 1 for finetunning
    Returns:
    """
    # creating the path for this checkpoint 
    #QUESTION: do we always want to overwrite this checkpoint or make new checkpoints for each training session (what are the repurcussions of this?)
    model_name = os.path.join(logs_path, "det_lesion.ckpt")
    
    # set configurations if there are None 
    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

    # start spittin facts 
    tf.logging.set_verbosity(tf.logging.INFO)

    # Prepare the input data
    input_image = tf.placeholder(tf.float32, [batch_size, 80, 80, 3])
    input_label = tf.placeholder(tf.float32, [batch_size])
    is_training = tf.placeholder(tf.bool, shape=())
    
    # input_label = 1 or is re-defined from batch_size parameter 
    # QUESTION: what are parameters for histogram and what is being output for this histogram?  
    tf.summary.histogram('input_label', input_label)

    # Create the network
    # we use our det_lesion_arg_scope function to pass in the arg_scope 
    with slim.arg_scope(det_lesion_arg_scope()):
        # defines the network, acquiring locations? of the nets and end_points (describe further?)
        net, end_points = det_lesion_resnet(input_image, is_training_option=is_training)

    # Initialize weights from pre-trained model 
    # if we are not finetuning then we have to initalize the weights of the model from Resnet
    if finetune == 0:
        init_weights = load_resnet_imagenet(initial_ckpt)

    # Define loss
    #outputs loss scores to variable losses
    with tf.name_scope('losses'):
        #uses the net as the output to calculate the binary_cross_entropy based on the target (input_label)
        #QUESTION: input_label is defined from batch_size and it's default is 1, is this a proper target? 
            #my assumption is that we use the det_lesion_positive_patches as our target but I guess
            #that we are aiming to get the prediction score to 1 and that's why the target is 1 
        loss, output, target = binary_cross_entropy(net, input_label)
        
        #QUESTION: what does tf.add_n do? what are regularization losses? 
        total_loss = loss + tf.add_n(tf.losses.get_regularization_losses())
        
        #QUESTION: what do the histograms look like? why? what does tf.summary.scalar do? 
        tf.summary.scalar('losses/total_loss', total_loss)
        tf.summary.histogram('losses/output', output)
        tf.summary.histogram('losses/target', target)

    # Define optimization method
    with tf.name_scope('optimization'):
        # output the learning_rate 
        #QUESTION: does it matter for this input whether learning rate is an instance or constant?  
        tf.summary.scalar('learning_rate', learning_rate)

        # QUESTION: what does MomentumOptimizer output exactly? how do learning rate & momentum work together? 
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)

        # uses total_loss to compute the gradient 
        #QUESTION: how does .compute_gradients work? 
        grads_and_vars = optimizer.compute_gradients(total_loss) 

        #QUESTION: what is the point of a gradient accumulator? 
        with tf.name_scope('grad_accumulator'): 
            #create gradient accumulator list 
            grad_accumulator = []

            # for the length of each grads_and_vars (write out what it is)
            for ind in range(0, len(grads_and_vars)):
                #QUESTION: what is in grads_and_vars[ind][0]? 
                if grads_and_vars[ind][0] is not None:
                    #QUESTION: what is .dtype? why are we doing only grads_and_vars[0][0].dtype? 
                        # what does ConditionalAccumulator do? what's the point? 
                    grad_accumulator.append(tf.ConditionalAccumulator(grads_and_vars[0][0].dtype))
        
        #apply the gradients 
        with tf.name_scope('apply_gradient'):
            #create the gradient accumulator operations list 
            grad_accumulator_ops = []

            #for the length of the gradient accumalator 
            for ind in range(0, len(grad_accumulator)):
                
                # if the ???? is not None 
                if grads_and_vars[ind][0] is not None:
                    # QUESTION: what is in grads_and_vars[ind][1]? what are we getting out of this? 
                        # what does this say about the total loss values that we put into this variable with optimizer.compute_gradients(total_loss)? 
                    var_name = str(grads_and_vars[ind][1].name).split(':')[0]
                    
                    #Note: fill in from previous questions  
                        #seems to be the gradient part of grads_and_vars 
                    var_grad = grads_and_vars[ind][0]
                    
                    # if the variable name that we pulled represent weights 
                    if "weights" in var_name:
                        #QUESTION: what does this mean/do? 
                        aux_layer_lr = 1.0

                    # if the variable name that we pulled represents biases 
                    elif "biases" in var_name:
                        #QUESTION: what does this mean/do? 
                        aux_layer_lr = 2.0
                    
                    # append to the gradient accumulator the applied gradient that multiplies the gradient * the aux_layer_lr
                    #QUESTION: how does aux_layer_lr impact the gradient, why? what does ConditionalAccumulator.apply_grad do? 
                    grad_accumulator_ops.append(grad_accumulator[ind].apply_grad(var_grad*aux_layer_lr,
                                                                                local_step=global_step))
        # 
        with tf.name_scope('take_gradients'):
            # create the list that will hold the ??? 
            mean_grads_and_vars = [] 

            for ind in range(0, len(grad_accumulator)):
                if grads_and_vars[ind][0] is not None:
                    # QUESTION: what does ConditionalAccumulator.take_grad do? 
                    # iter_mean_grad: Number of gradient computations that are average before updating the weights
                    mean_grads_and_vars.append((grad_accumulator[ind].take_grad(iter_mean_grad), grads_and_vars[ind][1]))
            # apply the gradients on the mean_grad_and_vars 
            apply_gradient_op = optimizer.apply_gradients(mean_grads_and_vars, global_step=global_step)
    
    with tf.name_scope('metrics'):
        acc_op = my_accuracy(net, input_label)
        tf.summary.scalar('metrics/accuracy', acc_op)
        
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        tf.logging.info('Gathering update_ops')
        with tf.control_dependencies(tf.tuple(update_ops)):
            total_loss = tf.identity(total_loss)
       
    merged_summary_op = tf.summary.merge_all()

    # Initialize variables
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        print('Init variable')
        sess.run(init)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path + '/train', graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter(logs_path + '/test')

        # Create saver to manage checkpoints
        saver = tf.train.Saver(max_to_keep=None)

        last_ckpt_path = tf.train.latest_checkpoint(logs_path)
        if last_ckpt_path is not None and resume_training:
            # Load last checkpoint
            print('Initializing from previous checkpoint...')
            saver.restore(sess, last_ckpt_path)
            step = global_step.eval() + 1
        else:
            # Load pre-trained model
            if finetune == 0:
                print('Initializing from pre-trained imagenet model...')
                init_weights(sess)
            else:
                print('Initializing from pre-trained model...')
                # init_weights(sess)
                var_list = []
                for var in tf.global_variables():
                    var_type = var.name.split('/')[-1]
                    if 'weights' in var_type or 'bias' in var_type:
                        var_list.append(var)
                saver_res = tf.train.Saver(var_list=var_list)
                saver_res.restore(sess, initial_ckpt)
            step = 1
        sess.run(interp_surgery(tf.global_variables()))
        print('Weights initialized')

        print('Start training')
        while step < max_training_iters + 1:
            # Average the gradient
            for iter_steps in range(0, iter_mean_grad):
                batch_image, batch_label, x_bb_train, y_bb_train, ids_train = dataset.next_batch(batch_size, 'train', 0.5)
                batch_image_val, batch_label_val, x_bb_val, y_bb_val, ids_val = dataset.next_batch(batch_size, 'val', 0.5)
                image = preprocess_img(batch_image, x_bb_train, y_bb_train, ids_train)
                label = batch_label
                val_image = preprocess_img(batch_image_val, x_bb_val, y_bb_val)
                label_val = batch_label_val
                run_res = sess.run([total_loss, merged_summary_op, acc_op] + grad_accumulator_ops,
                                   feed_dict={input_image: image, input_label: label, is_training: True})
                batch_loss = run_res[0]
                summary = run_res[1]
                acc = run_res[2]
                if step % display_step == 0:
                    val_run_res = sess.run([total_loss, merged_summary_op, acc_op],
                                           feed_dict={input_image: val_image, input_label: label_val, is_training: False})
                    val_batch_loss = val_run_res[0]
                    val_summary = val_run_res[1]
                    val_acc = val_run_res[2]

            # Apply the gradients
            sess.run(apply_gradient_op)

            # Save summary reports
            summary_writer.add_summary(summary, step)
            if step % display_step == 0:
                test_writer.add_summary(val_summary, step)

            # Display training status
            if step % display_step == 0:
                print >> sys.stderr, "{} Iter {}: Training Loss = {:.4f}".format(datetime.now(), step, batch_loss)
                print >> sys.stderr, "{} Iter {}: Validation Loss = {:.4f}".format(datetime.now(), step, val_batch_loss)
                print >> sys.stderr, "{} Iter {}: Training Accuracy = {:.4f}".format(datetime.now(), step, acc)
                print >> sys.stderr, "{} Iter {}: Validation Accuracy = {:.4f}".format(datetime.now(), step, val_acc)

            # Save a checkpoint
            if step % save_step == 0:
                save_path = saver.save(sess, model_name, global_step=global_step)
                print "Model saved in file: %s" % save_path

            step += 1

        if (step-1) % save_step != 0:
            save_path = saver.save(sess, model_name, global_step=global_step)
            print "Model saved in file: %s" % save_path

        print('Finished training.')


def validate(dataset, checkpoint_path, result_path, number_slices=1, config=None):
    """Test one sequence
    Args:
    dataset: Reference to a Dataset object instance
    checkpoint_path: Path of the checkpoint to use for the evaluation
    result_path: Path to save the output images
    config: Reference to a Configuration object used in the creation of a Session
    Returns:
    net:
    """
    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True
    tf.logging.set_verbosity(tf.logging.INFO)

    # Input data
    batch_size = 64
    number_of_slices = number_slices
    depth_input = number_of_slices
    if number_of_slices < 3:
        depth_input = 3

    pos_size = dataset.get_val_pos_size()
    neg_size = dataset.get_val_neg_size()
        
    input_image = tf.placeholder(tf.float32, [batch_size, None, None, depth_input])

    # Create the cnn
    with slim.arg_scope(det_lesion_arg_scope()):
        net, end_points = det_lesion_resnet(input_image, is_training_option=False)
    probabilities = end_points['det_lesion/output']
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create a saver to load the network
    saver = tf.train.Saver([v for v in tf.global_variables() if '-up' not in v.name and '-cr' not in v.name])

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(interp_surgery(tf.global_variables()))
        saver.restore(sess, checkpoint_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        
        results_file_soft = open(os.path.join(result_path, 'soft_results.txt'), 'w')
        results_file_hard = open(os.path.join(result_path, 'hard_results.txt'), 'w')
        
        # Test positive windows
        count_patches = 0
        for frame in range(0, pos_size/batch_size + (pos_size % batch_size > 0)):
            img, label, x_bb, y_bb = dataset.next_batch(batch_size, 'val', 1)
            curr_ct_scan = img[0]
            print('Testing ' + curr_ct_scan)
            image = preprocess_img(img, x_bb, y_bb)
            res = sess.run(probabilities, feed_dict={input_image: image})
            label = np.array(label).astype(np.float32).reshape(batch_size, 1)
            
            for i in range(0, batch_size):
                count_patches +=1
                img_part = img[i]
                res_part = res[i][0]
                label_part = label[i][0]
                if count_patches < (pos_size + 1):
                    results_file_soft.write(img_part.split('images_volumes/')[-1] + ' ' + str(x_bb[i]) + ' ' +
                                            str(y_bb[i]) + ' ' + str(res_part) + ' ' + str(label_part) + '\n')
                    if res_part > 0.5:
                        results_file_hard.write(img_part.split('images_volumes/')[-1] + ' ' +
                                                str(x_bb[i]) + ' ' + str(y_bb[i]) + '\n')

        # Test negative windows
        count_patches = 0
        for frame in range(0, neg_size/batch_size + (neg_size % batch_size > 0)):
            img, label, x_bb, y_bb = dataset.next_batch(batch_size, 'val', 0)
            curr_ct_scan = img[0]
            print('Testing ' + curr_ct_scan)
            image = preprocess_img(img, x_bb, y_bb)
            res = sess.run(probabilities, feed_dict={input_image: image})
            label = np.array(label).astype(np.float32).reshape(batch_size, 1)
           
            for i in range(0, batch_size):
                count_patches += 1
                img_part = img[i]
                res_part = res[i][0]
                label_part = label[i][0]
                if count_patches < (neg_size + 1):
                    results_file_soft.write(img_part.split('images_volumes/')[-1] + ' ' +
                                            str(x_bb[i]) + ' ' + str(y_bb[i]) + ' ' + str(res_part) + ' ' +
                                            str(label_part) + '\n')
                    if res_part > 0.5:
                        results_file_hard.write(img_part.split('images_volumes/')[-1] + ' ' +
                                                str(x_bb[i]) + ' ' + str(y_bb[i]) + '\n')
        
        results_file_soft.close()
        results_file_hard.close()


def test(dataset, checkpoint_path, result_path, number_slices=1, volume=False, config=None):
    """Test one sequence
    Args:
    dataset: Reference to a Dataset object instance
    checkpoint_path: Path of the checkpoint to use for the evaluation
    result_path: Path to save the output images
    config: Reference to a Configuration object used in the creation of a Session
    Returns:
    net:
    """

    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True
    tf.logging.set_verbosity(tf.logging.INFO)

    # Input data
    batch_size = 64
    number_of_slices = number_slices
    depth_input = number_of_slices
    if number_of_slices < 3:
        depth_input = 3

    total_size = dataset.get_val_pos_size()
        
    input_image = tf.placeholder(tf.float32, [batch_size, None, None, depth_input])

    # Create the cnn
    with slim.arg_scope(det_lesion_arg_scope()):
        net, end_points = det_lesion_resnet(input_image, is_training_option=False)
    probabilities = end_points['det_lesion/output']
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create a saver to load the network
    saver = tf.train.Saver([v for v in tf.global_variables() if '-up' not in v.name and '-cr' not in v.name])

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(interp_surgery(tf.global_variables()))
        saver.restore(sess, checkpoint_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        results_file_soft = open(os.path.join(result_path, 'soft_results.txt'), 'w')
        results_file_hard = open(os.path.join(result_path, 'hard_results.txt'), 'w')
        
        # Test all windows
        count_patches = 0
        for frame in range(0, total_size/batch_size + (total_size % batch_size > 0)):
            img, x_bb, y_bb = dataset.next_batch(batch_size, 'test', 1)
            curr_ct_scan = img[0]
            print('Testing ' + curr_ct_scan)
            image = preprocess_img(img, x_bb, y_bb)
            res = sess.run(probabilities, feed_dict={input_image: image})

            for i in range(0, batch_size):
                count_patches += 1
                img_part = img[i]
                res_part = res[i][0]
                if count_patches < (total_size + 1):
                    results_file_soft.write(img_part.split('images_volumes/')[-1] + ' ' + str(x_bb[i]) + ' ' +
                                            str(y_bb[i]) + ' ' + str(res_part) + '\n')
                    if res_part > 0.5:
                        results_file_hard.write(img_part.split('images_volumes/')[-1] + ' ' + str(x_bb[i]) + ' ' +
                                                str(y_bb[i]) + '\n')
        
        results_file_soft.close()
        results_file_hard.close()
