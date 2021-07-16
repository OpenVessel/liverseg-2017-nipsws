"""
Original code from OSVOS (https://github.com/scaelles/OSVOS-TensorFlow)
Sergi Caelles (scaelles@vision.ee.ethz.ch)

Modified code for liver and lesion segmentation:
Miriam Bellver (miriam.bellver@bsc.es)
"""

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import utils
import sys
from datetime import datetime
import os
import scipy.misc
from PIL import Image
slim = tf.contrib.slim
import scipy.io
import timeit

DTYPE = tf.float32


def seg_liver_arg_scope(weight_decay=0.0002):
    """Defines the arg scope.
    Args:
    weight_decay: The l2 regularization coefficient.
    Returns:
    An arg_scope.
    """
    with slim.arg_scope(
            #List or tuple of operations to set argument scope
            #QUESTION: what is slim.conv2d, convolution2d_transpose ? 
            [slim.conv2d, slim.convolution2d_transpose], 
            # relu is our activation function 
            # QUESTION: can we get the leaky activation function or others? 
            activation_fn= tf.nn.relu,
            # looks like we are going to start with normally distributed weights
            # default stddev = 0.5 
                #QUESTION: how does 0.001 change a normal distribution? 
            # QUESTION: what other options are there for starting weights & standard devs? what does random mean for normal distribution  
            weights_initializer= tf.random_normal_initializer(stddev=0.001),
            # QUESTION: what do different weight decays do to the weight regularizer and what are our options? 
            #we use our only argument weight decay as the regularizer's input (it's coefficient)
            weights_regularizer= slim.l2_regularizer(weight_decay),
            # QUESTION: what is this and what are our options?  
                # Initializer that generates tensors initialized to 0.
                # I guess we don't want to start with any implicit biases 
            biases_initializer= tf.zeros_initializer,
            # QUESTION: what is this? and what are our options? 
            biases_regularizer=None,
            # padding = SAME means we will pad the input with the needed extra 0's in order to make sure all values within the input are accounted for 
            padding='SAME') as arg_sc:
        return arg_sc


def crop_features(feature, out_size):
    """Crop the center of a feature map
    Args:
    feature: Feature map to crop 
    out_size: Size of the output feature map 
        - QUESTION: is this the size we want it to be? what size do we want it to be?
            - when it's called, out_size is being passed in as the image shape of the input image 
            to the model which will be in the shape of the tensorflow placeholder
                tf.placeholder(tf.float32, [batch_size, None, None, input_depth])
                - so it sounds like we want the size to be 512 x 512? 
    Returns: 
    Tensor that performs the cropping 
    """

    #get the shape of the feature map 
    up_size = tf.shape(feature) 
    print("shape of the feature map: ", up_size)
    print("shape of the input image: ", out_size)
    
    #subtract the width/height of the feature map by the  width/height of the image 
    # and then divide this result size in half to get to the center? 
    ini_w = tf.div(tf.subtract(up_size[1], out_size[1]), 2)
    ini_h = tf.div(tf.subtract(up_size[2], out_size[2]), 2)
    
    print("tf.div(tf.subtract(up_size[1], out_size[1]), 2) *(width)*:", ini_w)
    print("tf.div(tf.subtract(up_size[1], out_size[1]), 2) *(height)*:", ini_h)
    
    #format of tf.slice = tf.slice(input_, begin, size)
        # begin tells you where to start cutting out a slice
        # size tells you how long that cut should be 
    slice_input = tf.slice(feature, (0, ini_w, ini_h, 0), (-1, out_size[1], out_size[2], -1))
    
    print("cropped feature map:", slice_input)

    #add whatever is in the input placeholder to this cropped feature map and the cropped feature map should be getting resized to 512 x 512  
    
    return tf.reshape(slice_input, [int(feature.get_shape()[0]), out_size[1], out_size[2], int(feature.get_shape()[3])])
    print("output shape:",[int(feature.get_shape()[0]), out_size[1], out_size[2], int(feature.get_shape()[3])])
    print("the output itself:", tf.reshape(slice_input, [int(feature.get_shape()[0]), out_size[1], out_size[2], int(feature.get_shape()[3])]))

    # call the weight variable with an initializer on it 
def _weight_variable(name, shape):
    #QUESTION: what is tf.truncated_normal_initializer? 
    return tf.get_variable(name, shape, DTYPE, tf.truncated_normal_initializer(stddev=0.1))

    # call the bias variable with an initializer added to it 
def _bias_variable(name, shape):
    #QUESTION: what is tf.constant_initializer?
    return tf.get_variable(name, shape, DTYPE, tf.constant_initializer(0.1, dtype=DTYPE))


def seg_liver(inputs, number_slices=1, volume=False, scope='seg_liver'):
    """Defines the network
    Args:
    inputs: Tensorflow placeholder that contains the input image
    number_slices: ?? is this the distinction between whether we are sending in 3 images or not, to create depth
    scope: Scope name for the network
    Returns:
    net: Output Tensor of the network
    end_points: Dictionary with all Tensors of the network
    """

    # size of the image 
    im_size = tf.shape(inputs)

    #QUESTION: what are the parameters of tf.variable_scope?
    with tf.variable_scope(scope, 'seg_liver', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs of all intermediate layers.
        with slim.arg_scope(
            slim.conv2d, slim.max_pool2d], padding='SAME', outputs_collections=end_points_collection):
            #QUESTION: what are the parameters for slim.repeat? 
            #QUESTION: why does it go 64 , 128 , 256, 512, 512? 
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net_2 = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net_2, [2, 2], scope='pool2')
            net_3 = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net_3, [2, 2], scope='pool3')
            net_4 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net_4, [2, 2], scope='pool4')
            net_5 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')

            # Get side outputs of the network
                #QUESTION: what are side outputs? 
            #QUESTION: notice there is no activation function but are we still convoluting these already convuluted outputs? 
                # what does slim.conv2d exactly output 
            with slim.arg_scope([slim.conv2d],
                                activation_fn=None):
                #QUESTION: why are the scopes named in that way? 
                side_2 = slim.conv2d(net_2, 16, [3, 3], scope='conv2_2_16')
                side_3 = slim.conv2d(net_3, 16, [3, 3], scope='conv3_3_16')
                side_4 = slim.conv2d(net_4, 16, [3, 3], scope='conv4_3_16')
                side_5 = slim.conv2d(net_5, 16, [3, 3], scope='conv5_3_16')

                # Supervise side outputs
                    #QUESTION: what does it mean to supervise the side outputs? 
                        # understand how the parameters are impacting the input data
                side_2_s = slim.conv2d(side_2, number_slices, [1, 1], scope='score-dsn_2')
                side_3_s = slim.conv2d(side_3, number_slices, [1, 1], scope='score-dsn_3')
                side_4_s = slim.conv2d(side_4, number_slices, [1, 1], scope='score-dsn_4')
                side_5_s = slim.conv2d(side_5, number_slices, [1, 1], scope='score-dsn_5')
                
                #QUESTION: is this the supervision part?
                with slim.arg_scope([slim.convolution2d_transpose],
                                    activation_fn=None, biases_initializer=None, padding='VALID',
                                    outputs_collections=end_points_collection, trainable=False):
                    side_2_s = slim.convolution2d_transpose(side_2_s, number_slices, 4, 2, scope='score-dsn_2-up')
                    side_2_s = crop_features(side_2_s, im_size)
                    utils.collect_named_outputs(end_points_collection, 'seg_liver/score-dsn_2-cr', side_2_s)
                    side_3_s = slim.convolution2d_transpose(side_3_s, number_slices, 8, 4, scope='score-dsn_3-up')
                    side_3_s = crop_features(side_3_s, im_size)
                    utils.collect_named_outputs(end_points_collection, 'seg_liver/score-dsn_3-cr', side_3_s)
                    side_4_s = slim.convolution2d_transpose(side_4_s, number_slices, 16, 8, scope='score-dsn_4-up')
                    side_4_s = crop_features(side_4_s, im_size)
                    utils.collect_named_outputs(end_points_collection, 'seg_liver/score-dsn_4-cr', side_4_s)
                    side_5_s = slim.convolution2d_transpose(side_5_s, number_slices, 32, 16, scope='score-dsn_5-up')
                    side_5_s = crop_features(side_5_s, im_size)
                    
                    utils.collect_named_outputs(end_points_collection, 'seg_liver/score-dsn_5-cr', side_5_s)

                    # Main output
                    side_2_f = slim.convolution2d_transpose(side_2, 16, 4, 2, scope='score-multi2-up')
                    side_2_f = crop_features(side_2_f, im_size)
                    utils.collect_named_outputs(end_points_collection, 'seg_liver/side-multi2-cr', side_2_f)
                    side_3_f = slim.convolution2d_transpose(side_3, 16, 8, 4, scope='score-multi3-up')
                    side_3_f = crop_features(side_3_f, im_size)
                    utils.collect_named_outputs(end_points_collection, 'seg_liver/side-multi3-cr', side_3_f)
                    side_4_f = slim.convolution2d_transpose(side_4, 16, 16, 8, scope='score-multi4-up')
                    side_4_f = crop_features(side_4_f, im_size)
                    utils.collect_named_outputs(end_points_collection, 'seg_liver/side-multi4-cr', side_4_f)
                    side_5_f = slim.convolution2d_transpose(side_5, 16, 32, 16, scope='score-multi5-up')
                    side_5_f = crop_features(side_5_f, im_size)
                    utils.collect_named_outputs(end_points_collection, 'seg_liver/side-multi5-cr', side_5_f)
                #QUESTION: what does tf.concat do? 
                concat_side = tf.concat([side_2_f, side_3_f, side_4_f, side_5_f], 3)
                #QUESTION: this is our final output of net so what does this do to the concatenated sides? 
                net = slim.conv2d(concat_side, number_slices, [1, 1], scope='upscore-fuse')
    #QUESTION: what is end_points all comprised of now? 
    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
    print("end points", end_points)
    return net, end_points

# all of these comments were said about this function that is in det_lesion.py
### Function from old tensorflow that wasn't working (their comments or ours?)
def upsample_filt(size):
    """ 
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """ 
    # analyze 
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

#Note: everything underneath this is what was commented for this same function in det_lesion.py 
# this function reassigns some of our "-up" global tf variables into being able to be bilinearly interpolated 
#QUESTION: #what goes into variables? 
    # global variables that come from tf.global_variables_initializer() 
def interp_surgery(variables):
    # initate the list that we will return as outputs 
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
            up_filter = upsample_filt(int(h))
            
            # the up_filter is a 4D input except the 3D & 4D are 0's since these are the filter lengths
            #QUESTION: what is the difference between tmp = np.zeroes((m,k,h,w)) & tmp[range(m), range(k), :, :]
                # but really the better question is, what's the difference between range(m) & : in displaying/accessing a columns' values
            tmp[range(m), range(k), :, :] = up_filter
            
            #appending the newly assigned variables (that were taken in as inputs) into the output list (interp_sensors)
            #How: changes the shape with tmp.transpose 
            # use_locking = Passing use_locking=True when creating a TensorFlow optimizer, or a variable assignment op, causes a lock to be acquired around the relevant updates to the variable. Other optimizers/assignments on the same variable also created with use_locking=True will be serialized. 2 caveats to keep in mind. 
            # validate_shape = True means that we have to pass in a specified shape of a specific input which is apparently the tmp.transpose variable 
            interp_tensors.append(tf.assign(v, tmp.transpose((2, 3, 1, 0)), validate_shape=True, use_locking=True))
    return interp_tensors


def preprocess_img(image, number_slices):
    """Preprocess the image to adapt it to network requirements
    Args:
    Image we want to input the network (W,H,3) numpy array
    Returns:
	Image ready to input the network (1,W,H,3)
    """
    # this variable is a list that creates a list of lists that has the same width as the image 
        # this seems to create a 2d array of lists 
    images = [[] for i in range(np.array(image).shape[0])]


    if number_slices > 2:
        for j in range(np.array(image).shape[0]):
            if type(image) is not np.ndarray:
                for i in range(number_slices):
                    # this seems to ineffectively append each image or only truly appends the last slice within the image to the images variable
                    # this is because j will be the same value inside of this for loop so it should overwrite each appended slice until we reach the final slice within this image
                    images[j].append(np.array(scipy.io.loadmat(image[0][i])['section'], dtype=np.float32))
            
            # this seems to handle nothing if the type of the image IS a numpy array 
            else: 
                img = image
    
    else:
        for j in range(np.array(image).shape[0]):
            #again this for loop seems to contribute nothing except return the last image within the set
            for i in range(3):
                images[j].append(np.array(scipy.io.loadmat(image[0][0])['section'], dtype=np.float32))
    #QUESTION: we appended multiple images but we only want to return the first one?
    in_ = np.array(images[0])
    in_ = in_.transpose((1, 2, 0))
    in_ = np.expand_dims(in_, axis=0)

    return in_


def preprocess_labels(label, number_slices):
    """Preprocess the labels to adapt them to the loss computation requirements
    Args:
    Label corresponding to the input image (W,H) numpy array
    Returns:
    Label ready to compute the loss (1,W,H,1)
    """

    # this variable is a list that creates a list of lists that has the same width as the image 
        # this seems to create a 2d array of lists in the same shape of the rows of the label

    labels = [[] for i in range(np.array(label).shape[0])]

    print"np.array(label)=", np.array(label)
    print"np.array(label.shape[0])=",np.array(label).shape[0]
    print"labels = [[] for i in range(np.array(label).shape[0])]" , labels


    for j in range(np.array(label).shape[0]):
        #make sure that label is not a multidimensional array 
        if type(label) is not np.ndarray:

            for i in range(number_slices):
                #again this for loop seems to contribute nothing except truly append the last image within the set
                print"label[0][i] = ", label[0][i]
                labels[j].append(np.array(Image.open(label[0][i]), dtype=np.uint8))

    label = np.array(labels[0])
    print"label = np.array(labels[0]) = ", np.array(labels[0])
    label = label.transpose((1, 2, 0))

    max_mask = np.max(label) * 0.5
    print"max_mask = np.max(label) * 0.5 = ", max_mask
    label = np.greater(label, max_mask)
    print"label = np.greater(label, max_mask) = ", label 
    label = np.expand_dims(label, axis=0)

    return label


def load_vgg_imagenet(ckpt_path, number_slices): 
    """Initialize the network parameters from the VGG-16 pre-trained model provided by TF-SLIM 
    Args: 
    Path to the checkpoint 
    Returns: 
    Function that takes a session and initializes the network 
    """ 
    reader = tf.train.NewCheckpointReader(ckpt_path) 
    var_to_shape_map = reader.get_variable_to_shape_map() 
    vars_corresp = dict() 
    for v in var_to_shape_map: 
        if "conv" in v: 
            if not "conv1/conv1_1/weights" in v or number_slices < 4:
                vars_corresp[v] = slim.get_model_variables(v.replace("vgg_16", "seg_liver"))[0]
    init_fn = slim.assign_from_checkpoint_fn(
        ckpt_path,
        vars_corresp)
    return init_fn


def class_balanced_cross_entropy_loss(output, label):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network (what is the range of values for the output?)
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """

    print("gt_label=", label)
    #I thought ground truth labels were 0's & 1's 
    # seems to me that we are finding the points that are the truth points within the gt labels 
    labels = tf.cast(tf.greater(label, 0.5), tf.float32)
    print("labels = tf.cast(tf.greater(label, 0.5), tf.float32) = ", labels)

    #find the outputs greater than 0 
    #QUESTION: what is tf.greater_equal? by the variable name I thought this would just want greater than 0 not equal? 
    # looks like we're finding all of the positive liver predictions 
    output_gt_zero = tf.cast(tf.greater_equal(output, 0), tf.float32)
    print("output_gt_zero = tf.cast(tf.greater_equal(output, 0), tf.float32) = ", output_gt_zero)

    # find the difference between the positive label and the positive predictions and multiply it by the total output
        #QUESTION: what is the point of this? what does this accomplish? 
    #QUESTION: what does tf.exp() do?, what does tf.log() do 
    # multiply the total predictions by the positive predictions, what does this do? 
        # then multiply that by 2 and have the 

    print("A = tf.multiply(output, output_gt_zero)= ", tf.multiply(output, output_gt_zero))
    print("B = tf.multiply(output, (labels - output_gt_zero)) =", tf.multiply(output, (labels - output_gt_zero))) 
    
    loss_val = tf.multiply(output, (labels - output_gt_zero)) - tf.log(
        1 + tf.exp(output - 2 * tf.multiply(output, output_gt_zero)))
    
    print("loss_val = A - B ==", loss_val)

    # what does reduce_sum 
    
    loss_pos = tf.reduce_sum(-tf.multiply(labels, loss_val))
    print"A = -tf.multiply(labels, loss_val) = ", -tf.multiply(labels, loss_val)
    print"tf.reduce_sum( A ) = ", tf.reduce_sum(-tf.multiply(labels, loss_val))
    #
    loss_neg = tf.reduce_sum(-tf.multiply(1.0 - labels, loss_val))
    print"B = -tf.multiply(1.0 - labels, loss_val) = ", -tf.multiply(1.0 - labels, loss_val)
    print"tf.reduce_sum( B ) = ", tf.reduce_sum(-tf.multiply(1.0 - labels, loss_val))
    
    #How to calulate weights
    print"0.931 * loss_pos = ", 0.931 * loss_pos
    print"0.069 * loss_neg = ", 0.069 * loss_neg
    print"final_loss = ", 0.931 * loss_pos + 0.069 * loss_neg

    final_loss = 0.931 * loss_pos + 0.069 * loss_neg

    return final_loss



def dice_coef_theoretical(y_pred, y_true):
    """Define the dice coefficient
        Args:
        y_pred: Prediction
        y_true: Ground truth Label
        Returns:
        Dice coefficient
        """

    
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)

    print"tf.cast(tf.reshape(y_true, [-1]), tf.float32) = ", y_true_f

    y_pred_f = tf.nn.sigmoid(y_pred)
    print"y_pred_f = tf.nn.sigmoid(y_pred) = ", y_pred_f 
    
    y_pred_f = tf.cast(tf.greater(y_pred_f, 0.5), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred_f, [-1]), tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    dice = (2. * intersection) / (union + 0.00001)

    if (tf.reduce_sum(y_pred) == 0) and (tf.reduce_sum(y_true) == 0):
        dice = 1

    return dice


def parameter_lr():
    """Specify the learning rate for every parameter
    Args:
    Returns:
    Dictionary with the learning rate for every parameter
    """

    #weights are set to 1 and biases are set to 2, convolutions 1 & 2 have only 2 subsets of weights & biases while 3 - 5 have 3 subsets  

    vars_corresp = dict()
    vars_corresp['seg_liver/conv1/conv1_1/weights'] = 1
    vars_corresp['seg_liver/conv1/conv1_1/biases'] = 2
    vars_corresp['seg_liver/conv1/conv1_2/weights'] = 1
    vars_corresp['seg_liver/conv1/conv1_2/biases'] = 2


    vars_corresp['seg_liver/conv2/conv2_1/weights'] = 1
    vars_corresp['seg_liver/conv2/conv2_1/biases'] = 2
    vars_corresp['seg_liver/conv2/conv2_2/weights'] = 1
    vars_corresp['seg_liver/conv2/conv2_2/biases'] = 2


    vars_corresp['seg_liver/conv3/conv3_1/weights'] = 1
    vars_corresp['seg_liver/conv3/conv3_1/biases'] = 2
    vars_corresp['seg_liver/conv3/conv3_2/weights'] = 1
    vars_corresp['seg_liver/conv3/conv3_2/biases'] = 2
    vars_corresp['seg_liver/conv3/conv3_3/weights'] = 1
    vars_corresp['seg_liver/conv3/conv3_3/biases'] = 2


    vars_corresp['seg_liver/conv4/conv4_1/weights'] = 1
    vars_corresp['seg_liver/conv4/conv4_1/biases'] = 2
    vars_corresp['seg_liver/conv4/conv4_2/weights'] = 1
    vars_corresp['seg_liver/conv4/conv4_2/biases'] = 2
    vars_corresp['seg_liver/conv4/conv4_3/weights'] = 1
    vars_corresp['seg_liver/conv4/conv4_3/biases'] = 2


    vars_corresp['seg_liver/conv5/conv5_1/weights'] = 1
    vars_corresp['seg_liver/conv5/conv5_1/biases'] = 2
    vars_corresp['seg_liver/conv5/conv5_2/weights'] = 1
    vars_corresp['seg_liver/conv5/conv5_2/biases'] = 2
    vars_corresp['seg_liver/conv5/conv5_3/weights'] = 1
    vars_corresp['seg_liver/conv5/conv5_3/biases'] = 2


    vars_corresp['seg_liver/conv2_2_16/weights'] = 1
    vars_corresp['seg_liver/conv2_2_16/biases'] = 2
    vars_corresp['seg_liver/conv3_3_16/weights'] = 1
    vars_corresp['seg_liver/conv3_3_16/biases'] = 2
    vars_corresp['seg_liver/conv4_3_16/weights'] = 1
    vars_corresp['seg_liver/conv4_3_16/biases'] = 2
    vars_corresp['seg_liver/conv5_3_16/weights'] = 1
    vars_corresp['seg_liver/conv5_3_16/biases'] = 2


    vars_corresp['seg_liver/score-dsn_2/weights'] = 0.1
    vars_corresp['seg_liver/score-dsn_2/biases'] = 0.2
    vars_corresp['seg_liver/score-dsn_3/weights'] = 0.1
    vars_corresp['seg_liver/score-dsn_3/biases'] = 0.2
    vars_corresp['seg_liver/score-dsn_4/weights'] = 0.1
    vars_corresp['seg_liver/score-dsn_4/biases'] = 0.2
    vars_corresp['seg_liver/score-dsn_5/weights'] = 0.1
    vars_corresp['seg_liver/score-dsn_5/biases'] = 0.2

    vars_corresp['seg_liver/upscore-fuse/weights'] = 0.01
    vars_corresp['seg_liver/upscore-fuse/biases'] = 0.02
    return vars_corresp


def _train(dataset, initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step, display_step,
           global_step, number_slices=1, volume=False, iter_mean_grad=1, batch_size=1, task_id=2, loss=1, momentum=0.9, resume_training=False, config=None, finetune=1):
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
    global_step: Reference to a Variable that keeps track of the training steps
    iter_mean_grad: Number of gradient computations that are average before updating the weights
    batch_size:
    momentum: Value of the momentum parameter for the Momentum optimizer
    resume_training: Boolean to try to restore from a previous checkpoint (True) or not (False)
    config: Reference to a Configuration object used in the creation of a Session
    finetune: Use to select to select type of training, 0 for the parent network and 1 for finetunning
    Returns:
    """

    model_name = os.path.join(logs_path, "seg_liver.ckpt")
    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True

    tf.logging.set_verbosity(tf.logging.INFO)

    input_depth = 3
    if number_slices > 3:
        input_depth = number_slices

    # Prepare the input data
    input_image = tf.placeholder(tf.float32, [batch_size, None, None, input_depth])
    input_label = tf.placeholder(tf.float32, [batch_size, None, None, number_slices])

    # Create the network
    with slim.arg_scope(seg_liver_arg_scope()):
        net, end_points = seg_liver(input_image, number_slices, volume)

    # Initialize weights from pre-trained model
    if finetune == 0:
        init_weights = load_vgg_imagenet(initial_ckpt, number_slices)

    # Define loss
    with tf.name_scope('losses'): 
        dsn_2_loss = class_balanced_cross_entropy_loss(end_points['seg_liver/score-dsn_2-cr'], input_label)
        tf.summary.scalar('losses/dsn_2_loss', dsn_2_loss)
        dsn_3_loss = class_balanced_cross_entropy_loss(end_points['seg_liver/score-dsn_3-cr'], input_label)
        tf.summary.scalar('losses/dsn_3_loss', dsn_3_loss)
        dsn_4_loss = class_balanced_cross_entropy_loss(end_points['seg_liver/score-dsn_4-cr'], input_label)
        tf.summary.scalar('losses/dsn_4_loss', dsn_4_loss)
        dsn_5_loss = class_balanced_cross_entropy_loss(end_points['seg_liver/score-dsn_5-cr'], input_label)
        tf.summary.scalar('losses/dsn_5_loss', dsn_5_loss)

        main_loss = class_balanced_cross_entropy_loss(net, input_label)
        tf.summary.scalar('losses/main_loss', main_loss)

        if supervison == 1:
            output_loss = dsn_2_loss + dsn_3_loss + dsn_4_loss + dsn_5_loss + main_loss
        elif supervison == 2:
            output_loss = 0.5 * dsn_2_loss + 0.5 * dsn_3_loss + 0.5 * dsn_4_loss + 0.5 * dsn_5_loss + main_loss
        elif supervison == 3:
            output_loss = main_loss
        else:
            sys.exit('Incorrect supervision id, select 1 for supervision of the side outputs, 2 for weak supervision '
                     'of the side outputs and 3 for no supervision of the side outputs')
        # total_loss = output_loss + tf.add_n(slim.losses.get_regularization_losses())
        total_loss = output_loss + tf.add_n(tf.losses.get_regularization_losses())
        tf.summary.scalar('losses/total_loss', total_loss)

        # total_loss = output_loss + 0.001 * tf.add_n(slim.losses.get_regularization_losses())
        total_loss = output_loss + 0.001 * tf.add_n(tf.losses.get_regularization_losses())

        tf.summary.scalar('losses/total_loss', total_loss)

    # Define optimization method
    with tf.name_scope('optimization'):
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        with tf.name_scope('grad_accumulator'):
            grad_accumulator = []
            for ind in range(0, len(grads_and_vars)):
                if grads_and_vars[ind][0] is not None:
                    grad_accumulator.append(tf.ConditionalAccumulator(grads_and_vars[0][0].dtype))
        with tf.name_scope('apply_gradient'):
            layer_lr = parameter_lr()
            grad_accumulator_ops = []
            for ind in range(0, len(grad_accumulator)):
                if grads_and_vars[ind][0] is not None:
                    var_name = str(grads_and_vars[ind][1].name).split(':')[0]
                    var_grad = grads_and_vars[ind][0]
                    grad_accumulator_ops.append(grad_accumulator[ind].apply_grad(var_grad * layer_lr[var_name],
                                                                                 local_step=global_step))
        with tf.name_scope('take_gradients'):
            mean_grads_and_vars = []
            for ind in range(0, len(grad_accumulator)):
                if grads_and_vars[ind][0] is not None:
                    mean_grads_and_vars.append(
                        (grad_accumulator[ind].take_grad(iter_mean_grad), grads_and_vars[ind][1]))
            apply_gradient_op = optimizer.apply_gradients(mean_grads_and_vars, global_step=global_step)
            # Log training info

    with tf.name_scope('metrics'):
        dice_coef_op = dice_coef_theoretical(net, input_label)
        tf.summary.scalar('metrics/dice_coeff', dice_coef_op)

    merged_summary_op = tf.summary.merge_all()

    # Initialize variables
    init = tf.global_variables_initializer()

    # Create objects to record timing and memory of the graph execution
    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) # Option in the session options=run_options
    # run_metadata = tf.RunMetadata() # Option in the session run_metadata=run_metadata
    # summary_writer.add_run_metadata(run_metadata, 'step%d' % i)
    with tf.Session(config=config) as sess:
        print 'Init variable'
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

        print ('Start training')
        while step < max_training_iters + 1:
            # Average the gradient
            for iter_steps in range(0, iter_mean_grad):
                batch_image, batch_label, batch_label_liver = dataset.next_batch(batch_size, 'train')
                batch_image_val, batch_label_val, batch_label_liver_val = dataset.next_batch(batch_size, 'val')
                image = preprocess_img(batch_image, number_slices)
                val_image = preprocess_img(batch_image_val, number_slices)
                if task_id == 2:
                    batch_label = batch_label_liver
                    batch_label_val = batch_label_liver_val
                label = preprocess_labels(batch_label, number_slices)
                label_val = preprocess_labels(batch_label_val, number_slices)
                run_res = sess.run([total_loss, merged_summary_op, dice_coef_op] + grad_accumulator_ops,
                                   feed_dict={input_image: image, input_label: label})
                batch_loss = run_res[0]
                summary = run_res[1]
                train_dice_coef = run_res[2]
                if step % display_step == 0:
                    val_run_res = sess.run([total_loss, merged_summary_op, dice_coef_op],
                                           feed_dict={input_image: val_image, input_label: label_val})
                    val_batch_loss = val_run_res[0]
                    val_summary = val_run_res[1]
                    val_dice_coef = val_run_res[2]

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
                print >> sys.stderr, "{} Iter {}: Training Dice = {:.4f}".format(datetime.now(), step, train_dice_coef)
                print >> sys.stderr, "{} Iter {}: Validation Dice = {:.4f}".format(datetime.now(), step, val_dice_coef)

            # Save a checkpoint
            if step % save_step == 0:
                save_path = saver.save(sess, model_name, global_step=global_step)
                print "Model saved in file: %s" % save_path

            step += 1

        if (step - 1) % save_step != 0:
            save_path = saver.save(sess, model_name, global_step=global_step)
            print "Model saved in file: %s" % save_path

        print('Finished training.')


def train_seg(dataset, initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step,
                 display_step, global_step, number_slices=1, volume=False, iter_mean_grad=1, batch_size=1, task_id=2,
                 loss=1, momentum=0.9, resume_training=False,
                 config=None):
    """Train parent network
    Args:
    See _train()
    Returns:
    """

    _train(dataset, initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step, display_step,
           global_step, number_slices, volume, iter_mean_grad, batch_size, task_id, loss, momentum,
           resume_training, config, finetune=0)


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
    batch_size = 1
    number_of_slices = number_slices
    depth_input = number_of_slices
    if number_of_slices < 3:
        depth_input = 3

    input_image = tf.placeholder(tf.float32, [batch_size, None, None, depth_input])

    # Create the cnn
    with slim.arg_scope(seg_liver_arg_scope()):
        net, end_points = seg_liver(input_image, number_slices, volume)
    probabilities = tf.nn.sigmoid(net)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create a saver to load the network
    saver = tf.train.Saver([v for v in tf.global_variables() if '-up' not in v.name and '-cr' not in v.name])

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(interp_surgery(tf.global_variables()))
        saver.restore(sess, checkpoint_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        for frame in range(0, dataset.get_test_size()):
            img, curr_img = dataset.next_batch(batch_size, 'test')
            curr_ct_scan = curr_img[0][0].split('/')[-2]
            curr_frames = []
            if 1:
                for i in range(number_of_slices):
                    curr_frames.append([curr_img[0][i].split('/')[-1].split('.')[0] + '.png'])
                if not os.path.exists(os.path.join(result_path, curr_ct_scan)):
                    os.makedirs(os.path.join(result_path, curr_ct_scan))
                image = preprocess_img(curr_img, number_slices)
                res = sess.run(probabilities, feed_dict={input_image: image})
                res_np = res.astype(np.float32)[0, :, :, number_of_slices / 2]

                aux_var = curr_frames[number_of_slices / 2][0]
                scipy.misc.imsave(os.path.join(result_path, curr_ct_scan, aux_var), res_np)
                print 'Saving ' + os.path.join(result_path, curr_ct_scan, aux_var)
                for i in range(number_of_slices):
                    aux_var = curr_frames[i][0]
                    if not os.path.exists(os.path.join(result_path, curr_ct_scan, aux_var)):
                        res_np = res.astype(np.float32)[0, :, :, i]
                        scipy.misc.imsave(os.path.join(result_path, curr_ct_scan, aux_var), res_np)
                        print 'Saving ' + os.path.join(result_path, curr_ct_scan, aux_var)
