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
# import keras.initializers as KI
# import keras.engine as KE

sys.path.append('..')

from mrcnn.BilinearUpSampling import BilinearUpSampling2D
from   mrcnn.batchnorm_layer import BatchNorm
# import mrcnn.utils as utils
# from   mrcnn.datagen import data_generator
# import mrcnn.loss  as loss

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
    
###############################################################
## Fully Convolutional Network Layer 
###############################################################
def fcn8_graph(feature_map , config , weight_decay=0.0002, batch_momentum=0.9):
    '''Builds the computation graph of Region Proposal Network.

    feature_map:            Contextual Tensor [batch, num_classes, width, depth]

    Returns:


    '''
    print('\n>>> FCN Layer ')
    height, width     = config.FCN_INPUT_SHAPE[0:2]
    num_classes       = config.NUM_CLASSES
    rois_per_class    = config.TRAIN_ROIS_PER_IMAGE
    
    print('     feature map      :', feature_map.shape)
    print('     height :', height, 'width :', width, 'classes :' , num_classes)
    print('     image_data_format: ', KB.image_data_format())
    print('     rois_per_class   : ', KB.image_data_format())
    feature_map_shape = (width, height, num_classes)
    
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
    
    ## Block 1    data_format='channels_last',
    
    x = KL.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', 
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(feature_map)
    print('   Input feature map                   : ', feature_map.shape)
    print('   FCN Block 11 shape is               : ' ,x.get_shape())
    
    x = KL.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', 
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 12 shape is               : ' ,x.get_shape())         
    
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    print('   FCN Block 13 (Max pooling) shape is : ' ,x.get_shape())
    x0 = x
    
    ## Block 2
    x = KL.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', 
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 21 shape is               : ' , x.get_shape())
    
    x = KL.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 22 shape is               : ' ,x.get_shape())    
    
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    print('   FCN Block 23 (Max pooling) shape is : ' ,x.get_shape())    
    x1 = x
    
    ## Block 3
    x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 31 shape is               : ' ,x.get_shape())            
    
    x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', 
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 32 shape is               : ' ,x.get_shape())    
    
    x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 33 shape is               : ' ,x.get_shape())            
    
    Pool3 = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    print('   FCN Block 34 (Max pooling) shape is : ' ,Pool3.get_shape())    
    
    ## Block 4
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', 
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(Pool3)
    print('   FCN Block 41 shape is               : ' ,x.get_shape())            
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 42 shape is               : ' ,x.get_shape())            
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 43 shape is               : ' ,x.get_shape())                
    Pool4 = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    print('   FCN Block 44 (Max pooling) shape is : ' ,Pool4.get_shape())    
    
    ## Block 5
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(Pool4)
    print('   FCN Block 51 shape is               : ' ,x.get_shape())                 
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 52 shape is               : ' ,x.get_shape())                 
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(weight_decay))(x)
    print('   FCN Block 53 shape is               : ' ,x.get_shape())                    
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    print('   FCN Block 54 (Max pooling) shape is : ' ,x.get_shape())    

    # Convolutional layers transfered from fully-connected layers
    # changed from 4096 to 2048 - reduction of weights from 42,752,644 to                       
    # changed ftom 2048 to 1024 - 11-05-2018
    # Fully Connected Layers
    # FC_SIZE = 2048 
    FC_SIZE = 4096
    x = KL.Conv2D(FC_SIZE, (7, 7), activation='relu', padding='same',
                        kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                        kernel_regularizer=l2(weight_decay), name='fc1')(x)
    print('   FCN fully connected 1 (fc1) shape   : ' ,x.get_shape())        
    x = KL.Dropout(0.5)(x)
    x = KL.Conv2D(FC_SIZE, (1, 1), activation='relu', padding='same',
                        kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                        kernel_regularizer=l2(weight_decay), name='fc2')(x)
    print('   FCN fully connected 2 (fc2) shape   : ' ,x.get_shape())        
    x = KL.Dropout(0.5)(x)
    
    # Classifying layer
    x = KL.Conv2D(num_classes, (1, 1), activation='linear', padding='valid', strides=(1, 1),
                  kernel_initializer='he_normal',  bias_initializer='zeros', 
                  kernel_regularizer=l2(weight_decay), name='fcn32_classify')(x)
    print('   FCN conv2d (fcn32_classify) shape   : ' , x.get_shape(),' keras_tensor ', KB.is_keras_tensor(x))                      
    
    ##-------------------------------------------------------------------------------------------------------
    ## FCN16 Specific Structure 
    ##-------------------------------------------------------------------------------------------------------
    # Score Pool4
    scorePool4 = KL.Conv2D(num_classes, (1,1), activation="relu", padding="valid",
                          kernel_initializer='glorot_uniform', bias_initializer='zeros',
                          kernel_regularizer=l2(weight_decay), name="score_pool4")(Pool4)                    
    print()
    print('   --- FCN16 ----------------------------')
    print('   FCN scorePool4 (Conv2D(Pool4)) shape is                   : ' , KB.int_shape(scorePool4),
          '   keras_tensor ', KB.is_keras_tensor(scorePool4))                      
                  
    # 2x Upsampling to generate Score2 (padding was originally "valid")
    score2   = KL.Deconvolution2D(num_classes,kernel_size=(4,4), strides = (2,2),
                                  padding = "valid", activation=None, name = "score2")(x)
    score2_c = KL.Cropping2D(cropping=((1,1),(1,1)))(score2)
    print('   FCN 2x Upsampling (Deconvolution2D(fcn32_classify)) shape : ' , KB.int_shape(score2),
          '   keras_tnsr ', KB.is_keras_tensor(score2))                      
    print('   FCN 2x Upsampling/Cropped (Cropped2D(score2)) shape       : ' , KB.int_shape(score2_c),
          '   keras_tnsr ', KB.is_keras_tensor(score2_c))
    
    # Sum Score2, Pool4
    x = KL.Add(name='fuse_pool4')([score2_c,scorePool4])    
    print('   FCN Add Score2,scorePool4 Add(score2_c, scorePool4) shape : ' , KB.int_shape(x), 
          '   keras_tensor ', KB.is_keras_tensor(x))                      

    # Upsampling  (padding was originally "valid", I changed it to "same" )
    upscore_pool4 = KL.Deconvolution2D(num_classes, kernel_size=(4,4), strides = (2,2), 
                                       padding = "same", activation = None, name = "upscore_pool4")(x)
                                       
    print('   FCN upscore_pool4 (Deconv(fuse_Pool4)) shape              : ' , KB.int_shape(upscore_pool4),
          '   keras_tensor ', KB.is_keras_tensor(upscore_pool4))                      

    
    
    ##-------------------------------------------------------------------------------------------------------
    ## FCN8 Specific Structure 
    ##-------------------------------------------------------------------------------------------------------
    # Score Pool3
    scorePool3 = KL.Conv2D(num_classes, (1,1), activation="relu", padding="valid", 
                           kernel_initializer='glorot_uniform',  bias_initializer='zeros', 
                           kernel_regularizer=l2(weight_decay), name="score_pool3")(Pool3)                    
    print()
    print('   --- FCN8 ----------------------------')
    print('   FCN scorePool4 (Conv2D(Pool4)) shape                      : ' , KB.int_shape(scorePool3),
          '   keras_tensor ', KB.is_keras_tensor(scorePool3))                      

    upscore_pool4_c = KL.Cropping2D(cropping=((0,0),(0,0)))(upscore_pool4)
    print('   FCN 2x Upsampling/Cropped (Cropped2D(score2)) shape       : ' , KB.int_shape(upscore_pool4_c),
          '   keras_tensor ', KB.is_keras_tensor(upscore_pool4_c))                      

    # Sum  upscore_pool4_c, scorePool3
    x = KL.Add(name='fuse_pool3')([upscore_pool4_c,scorePool3])    
    print('   FCN Add Score2,scorePool4 shape is                        : ' ,  KB.int_shape(x), 
          '   keras_tensor ', KB.is_keras_tensor(x))                      

    # Upsampling  (padding was originally "valid", I changed it to "same" )
    fcn_hm = KL.Deconvolution2D(num_classes,kernel_size=(16,16),strides = (8,8),
                                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                padding = "same",activation = None,name = "fcn_heatmap")(x)
    print('   FCN fcn16_classify (Deconv(fuse_Pool4)) shape             : ' , KB.int_shape(fcn_hm),
          '   keras_tensor ', KB.is_keras_tensor(fcn_hm))                      

    fcn_classify_shape = KB.int_shape(fcn_hm)
    h_factor = height/ fcn_classify_shape[1]
    w_factor = width / fcn_classify_shape[2]
    print('   h_factor : ', h_factor, 'w_factor : ', w_factor)
    
    # x = BilinearUpSampling2D(size=(h_factor, w_factor), name='fcn_bilinear')(x)
    # print('   FCN Bilinear upsmapling layer  shape is : ' , x.get_shape(), ' Keras tensor ', KB.is_keras_tensor(x) )  
    
    # fcn_hm = KL.Lambda(lambda z: tf.identity(z, name="fcn_heatmap"), name="fcn_heatmap") (x)

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

    print('\n   Lambda Layer: (fcn_heatmap)       : ', KB.int_shape(fcn_hm) ,' Keras tensor ', KB.is_keras_tensor(fcn_hm) )
    # print('    fcn_heatmap_norm  : ', fcn_hm_norm.shape   ,' Keras tensor ', KB.is_keras_tensor(fcn_hm_norm) )
    # print('    fcn_heatmap_L2norm: ', fcn_hm_L2norm.shape ,' Keras tensor ', KB.is_keras_tensor(fcn_hm_L2norm) )

    return fcn_hm   # fcn_hm_norm, fcn_hm_L2norm

