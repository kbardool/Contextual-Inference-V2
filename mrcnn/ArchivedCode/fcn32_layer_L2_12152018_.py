"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import tensorflow as tf
# import keras
import keras.backend as KB
import keras.layers as KL
import keras.models as KM
from   keras.regularizers     import l2
from   mrcnn.utils import logt
# sys.path.append('..')

# Requires TensorFlow 1.3+ and Keras 2.0.8+.

def normalize(x):
    #--------------------------------------------------------------------------------------------
    # this normalization moves values to [0  +1] 
    #--------------------------------------------------------------------------------------------    

    # x   = x / tf.reduce_max(x, axis=[1,2], keepdims = True)
    # x   = tf.where(tf.is_nan(x),  tf.zeros_like(x), x, name = 'fcn_heatmap_norm')

    #--------------------------------------------------------------------------------------------
    # this normalization moves values to [-1 +1] 
    #--------------------------------------------------------------------------------------------    
    reduce_max = tf.reduce_max(x, axis = [1,2], keepdims=True)
    reduce_min = tf.reduce_min(x, axis = [1,2], keepdims=True)  ## epsilon    = tf.ones_like(reduce_max) * 1e-7
    print('     size of reduce max is ', reduce_max.shape)
    y  = tf.divide( (x- reduce_min) , (reduce_max - reduce_min), name='fcn_heatmap_norm')
    print('     size of y is : ', y.shape)

    return y

    
###############################################################
# Fully Convolutional Network Layer 
###############################################################
# def fcn_layer(context_tensor, num_classes,weight_decay=0., batch_momentum=0.9):
def fcn32_l2_graph(feature_map , config, mode):
    '''Builds the computation graph of Region Proposal Network.

    feature_map:            Contextual Tensor [batch, num_classes, width, depth]

    Returns:


    '''
    print()
    print('------------------------------------------------------')
    print('>>> FCN32 Layer with L2 regularization - mode: ', mode)
    print('------------------------------------------------------')
    height, width     = config.FCN_INPUT_SHAPE[0:2]
    num_classes       = config.NUM_CLASSES
    rois_per_class    = config.TRAIN_ROIS_PER_IMAGE
    weight_decay      = config.WEIGHT_DECAY
    # In the original implementatoin , batch_momentum was used for batch normalization layers for the ResNet
    # backbone. We are not using this backbone in FCN, therefore it is unused.
    # batch_momentum    = config.BATCH_MOMENTUM
    verbose           = config.VERBOSE
    feature_map_shape = (width, height, num_classes)
    
    
    print('     feature map      :', feature_map.shape)
    print('     height :', height, 'width :', width, 'classes :' , num_classes)
    print('     image_data_format: ', KB.image_data_format())
    print('     rois_per_class   : ', KB.image_data_format())
    print('     FCN L2 weight decay : ', weight_decay)
    
    # feature_map = KL.Input(shape= feature_map_shape, name="input_fcn_feature_map")
    # TODO: Assert proper shape of input [batch_size, width, height, num_classes]
    
    # TODO: check if stride of 2 causes alignment issues if the featuremap is not even.
    
    # if batch_shape:
        # img_input = Input(batch_shape=batch_shape)
        # image_size = batch_shape[1:3]
    # else:
        # img_input = Input(shape=input_shape)
        # image_size = input_shape[0:2]
    
    ## , kernel_regularizer=l2(weight_decay)
    
    # Block 1    data_format='channels_last',
    
    x = KL.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', 
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(feature_map)
    print('   FCN Block 11 shape is : ' ,KB.int_shape(x))
    
    x = KL.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', 
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 12 shape is : ' ,KB.int_shape(x))         
    
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    print('   FCN Block 13 shape is : ' ,KB.int_shape(x))
    x0 = x
    
    # Block 2
    x = KL.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', 
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 21 shape is : ' , KB.int_shape(x))
    
    x = KL.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 22 shape is : ' ,KB.int_shape(x))    
    
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    print('   FCN Block 23 (Max pooling) shape is : ' ,KB.int_shape(x))    
    x1 = x
    
    # Block 3
    x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 31 shape is : ' ,KB.int_shape(x))            
    
    x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', 
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 32 shape is : ' ,KB.int_shape(x))    
    
    x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 33 shape is : ' ,KB.int_shape(x))            
    
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    print('   FCN Block 34 (Max pooling) shape is : ' ,KB.int_shape(x))    
    
    # Block 4
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', 
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 41 shape is : ' ,KB.int_shape(x))            
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 42 shape is : ' ,KB.int_shape(x))            
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 43 shape is : ' ,KB.int_shape(x))                
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    print('   FCN Block 44 (Max pooling) shape is : ' ,KB.int_shape(x))    
    
    # Block 5
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 51 shape is : ' ,KB.int_shape(x))                 
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 52 shape is : ' ,KB.int_shape(x))                 
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 53 shape is : ' ,KB.int_shape(x))                    
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    print('   FCN Block 54 (Max pooling) shape is : ' ,KB.int_shape(x))    

    ##-------------------------------------------------------------------------------------------------------
    ## FCN32 Specific Structure 
    ##-------------------------------------------------------------------------------------------------------
    # Convolutional layers transfered from fully-connected layers
    # changed from 4096 to 2048 - reduction of weights from 42,752,644 to                       
    # changed ftom 2048 to 1024 - 11-05-2018    

    # FC_SIZE = 2048 
    FC_SIZE = 4096
    x = KL.Conv2D(FC_SIZE, (7, 7), activation='relu', padding='same', name="fc1"',
                        kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                        kernel_regularizer=l2(weight_decay))(x)
    print()
    print('   --- FCN32 ----------------------------')
    print('   FCN fully connected 1 (fcn_fc1) shape is : ' ,KB.int_shape(x))        
    x = KL.Dropout(0.5)(x)
    
    #fc2
    x = KL.Conv2D(FC_SIZE, (1, 1), activation='relu', padding='same', name="fc2",
                        kernel_initializer='glorot_uniform', bias_initializer='zeros',
                        kernel_regularizer=l2(weight_decay))(x)
    print('   FCN fully connected 2 (fcn_fc2) shape is : ' ,KB.int_shape(x))        
    x = KL.Dropout(0.5)(x)
    
    #classifying layer
    x = KL.Conv2D(num_classes, (1, 1), kernel_initializer='he_normal',  bias_initializer='zeros', 
                  activation='linear', padding='valid', strides=(1, 1),
                  kernel_regularizer=l2(weight_decay), name="fcn32_deconv2D")(x)
    
    print('   FCN conv2d (fcn32_decond2D) shape is     : ' , KB.int_shape(x),' keras_tensor ', KB.is_keras_tensor(x))                      
    
    fcn_classify_shape = KB.int_shape(x)
    h_factor = height / fcn_classify_shape[1]
    w_factor = height / fcn_classify_shape[2]
    print('   h_factor : ', h_factor, 'w_factor : ', w_factor)

    # x = BilinearUpSampling2D(size=(h_factor, w_factor), name='fcn_bilinear')(x)
    # print('   FCN Bilinear upsmapling layer  shape is : ' , x.get_shape(), ' Keras tensor ', KB.is_keras_tensor(x) )  
    ##-------------------------------------------------------------------------------------------------------
    ## fcn output heatmap
    ##-------------------------------------------------------------------------------------------------------        
    # Upsampling  (padding was originally "valid", I changed it to "same" )
    fcn_hm = KL.Deconvolution2D(num_classes, kernel_size=(32,32), strides = (32,32),
                           kernel_initializer='glorot_uniform', bias_initializer='zeros',
                           padding = "valid", activation = None, name = "fcn32_classify")(x)

    # fcn_hm = tf.identity(fcn_hm)                           
    fcn_hm.set_shape(feature_map.shape)
    logt('FCN fcn8_classify/heatmap  (Deconv(fuse_Pool4)) ' , fcn_hm, verbose = verbose) 
    fcn_hm = KL.Lambda(lambda z: tf.identity(z, name='fcn_hm'), name='fcn_heatmap_lambda') (fcn_hm)
    logt('fcn_hm (final)' , fcn_hm, verbose = verbose) 
    print()

    ##-------------------------------------------------------------------------------------------------------
    ## fcn_SOFTMAX
    ##-------------------------------------------------------------------------------------------------------    
    fcn_sm = KL.Activation("softmax", name = "fcn32_softmax")(fcn_hm)
    logt('fcn8_softmax  ', fcn_sm, verbose = verbose) 
    fcn_sm = KL.Lambda(lambda z: tf.identity(z, name='fcn_sm'), name='fcn_softmax_lambda') (fcn_hm)
    logt('fcn_sm (final)', fcn_sm, verbose = verbose) 
    print()
    
    #---------------------------------------------------------------------------------------------
    # heatmap L2 normalization
    # Normalization using the  `gauss_sum` (batchsize , num_classes, height, width) 
    # 17-05-2018 (New method, replace dthe previous method that usedthe transposed gauss sum
    # 17-05-2018 Replaced with normalization across the CLASS axis 
    #                         normalize along the CLASS axis 
    #---------------------------------------------------------------------------------------------
    # print('\n    L2 normalization ------------------------------------------------------')   
    # fcn_hm_L2norm = KL.Lambda(lambda z: tf.nn.l2_normalize(z, axis = 3, name = 'fcn_heatmap_L2norm'),\
                        # name = 'fcn_heatmap_L2norm')(x)
    # print('\n    normalization ------------------------------------------------------')   
    # fcn_hm_norm   = KL.Lambda(normalize, name="fcn_heatmap_norm") (x)

    print('    fcn_heatmap       : ', fcn_hm.shape        ,' Keras tensor ', KB.is_keras_tensor(fcn_hm) )
    # print('    fcn_heatmap_norm  : ', fcn_hm_norm.shape   ,' Keras tensor ', KB.is_keras_tensor(fcn_hm_norm) )
    # print('    fcn_heatmap_L2norm: ', fcn_hm_L2norm.shape ,' Keras tensor ', KB.is_keras_tensor(fcn_hm_L2norm) )

    return fcn_hm, fcn_sm 



    
    