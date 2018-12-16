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
# from   keras.activations as KA
# import keras.initializers as KI
# import keras.engine as KE

sys.path.append('..')

from  mrcnn.utils           import logt

# from  mrcnn.helper_layers   import BilinearUpSampling2D
# from   mrcnn.batchnorm_layer import BatchNorm
# import mrcnn.utils as utils
# from   mrcnn.datagen import data_generator
# import mrcnn.loss  as loss

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
    
###############################################################
## Fully Convolutional Network Layer 
###############################################################
def fcn8_graph(feature_map , config, mode = None ):
    '''Builds the computation graph of Region Proposal Network.

    feature_map:            Contextual Tensor [batch, num_classes, width, depth]

    Returns:


    '''
    print()
    print('---------------')
    print('>>> FCN8 Layer - mode:', mode)
    print('---------------')
    batch_size        = config.BATCH_SIZE
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

    
    if mode == 'training':
        KB.set_learning_phase(1)
    else:
        KB.set_learning_phase(0)
    print('     Set learning phase to :', KB.learning_phase())

    # feature_map = KL.Input(shape= feature_map_shape, name="input_fcn_feature_map")
    
    # TODO: Assert proper shape of input [batch_size, width, height, num_classes]    
    # TODO: check if stride of 2 causes alignment issues if the featuremap is not even.
    
    # if batch_shape:
        # img_input = Input(batch_shape=batch_shape)
        # image_size = batch_shape[1:3]
    # else:
        # img_input = Input(shape=input_shape)
        # image_size = input_shape[0:2]
    
    ##-------------------------------------------------------------------------------------------------------
    ## Block 1    data_format='channels_last',
    ##-------------------------------------------------------------------------------------------------------    
    x = KL.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', 
                    kernel_initializer='glorot_uniform', bias_initializer='zeros')(feature_map)
    print('   Input feature map                   : ', feature_map.shape)
    logt('Input feature map ', feature_map, verbose = 1)
    
    logt('FCN Block 11 ' , x, verbose = verbose)
    
    x = KL.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', 
                    kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    logt('FCN Block 12 ' , x, verbose = verbose)
    
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    logt('FCN Block 13 (Max pooling) ', x, verbose = verbose)
    
    ##-------------------------------------------------------------------------------------------------------
    ## Block 2
    ##-------------------------------------------------------------------------------------------------------
    x = KL.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', 
                    kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    logt('FCN Block 21  ', x, verbose = verbose)
    
    x = KL.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    logt('FCN Block 22 ', x, verbose = verbose)
    
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    logt('FCN Block 23 (Max pooling) ', x, verbose = verbose)
    
    ##-------------------------------------------------------------------------------------------------------
    ## Block 3
    ##-------------------------------------------------------------------------------------------------------
    x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    logt('FCN Block 31  ', x, verbose = verbose)
    
    x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', 
                    kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    logt('FCN Block 32 ', x, verbose = verbose)
    
    x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    logt('FCN Block 33 ', x, verbose = verbose)
    
    Pool3 = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    logt('FCN Block 34 (Max pooling) ', Pool3, verbose = verbose)
    
    ##-------------------------------------------------------------------------------------------------------
    ## Block 4
    ##-------------------------------------------------------------------------------------------------------
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', 
                    kernel_initializer='glorot_uniform', bias_initializer='zeros')(Pool3)
    logt('FCN Block 41 ', x, verbose = verbose)
    
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    logt('FCN Block 42 ', x, verbose = verbose)
    
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    logt('FCN Block 43 ', x, verbose = verbose)           
    
    Pool4 = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    logt('FCN Block 44 (Max pooling) ', Pool4, verbose = verbose)
    
    ##-------------------------------------------------------------------------------------------------------
    ## Block 5
    ##-------------------------------------------------------------------------------------------------------
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros')(Pool4)
    logt('FCN Block 51 ' , x, verbose = verbose)
    
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    logt('FCN Block 52 ' , x, verbose = verbose)

    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    logt('FCN Block 53 ' , x, verbose = verbose)

    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    logt('FCN Block 54 (Max pooling) ' , x, verbose = verbose)

    ##-------------------------------------------------------------------------------------------------------
    ## FCN32 Specific Structure 
    ##-------------------------------------------------------------------------------------------------------
    # Convolutional layers transfered from fully-connected layers
    # changed from 4096 to 2048 - reduction of weights from 42,752,644 to                       
    # changed ftom 2048 to 1024 - 11-05-2018
    # FC_SIZE = 2048 
    FC_SIZE = 4096
    x = KL.Conv2D(FC_SIZE, (7, 7), activation='relu', padding='same', name='fcn32_fc1',
                        kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    print()
    print('   --- FCN32 ----------------------------')
    logt(' FCN fully connected 1 (fc1) ' , x, verbose = verbose) 
    
    x = KL.Dropout(0.5)(x)
    x = KL.Conv2D(FC_SIZE, (1, 1), activation='relu', padding='same', name='fcn32_fc2',
                        kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    logt('FCN fully connected 2 (fc2) ' , x, verbose = verbose)
    
    x = KL.Dropout(0.5)(x)
    # Classifying layer
    x = KL.Conv2D(num_classes, (1, 1), activation='linear', padding='valid', strides=(1, 1), name='fcn32_deconv2D',
                  kernel_initializer='he_normal',  bias_initializer='zeros')(x)                  
    logt('FCN conv2d (fcn32_deconv2D)  ' , x, verbose = verbose)
    
    ##-------------------------------------------------------------------------------------------------------
    ## FCN16 Specific Structure 
    ##-------------------------------------------------------------------------------------------------------
    # Score Pool4 - Reduce Pool4 filters from 512 to num_classes (81)
    scorePool4 = KL.Conv2D(num_classes, (1,1), activation='relu', padding='valid', name='fcn16_score_pool4',
                           kernel_initializer='glorot_uniform', bias_initializer='zeros')(Pool4)                    
    print()
    print('   --- FCN16 ----------------------------')
    logt('FCN scorePool4 (Conv2D(Pool4)) ' , scorePool4, verbose = verbose)
                  
    # 2x Upsampling of fcn_deconv2D  to generate Score2 (padding was originally "valid")
    x  = KL.Deconvolution2D(num_classes,kernel_size=(4,4), activation=None, padding = 'valid', name = 'fcn16_score2', 
                            strides = (2,2))(x)                    
    logt('FCN 2x Upsampling (Deconvolution2D(fcn32_classify))  ' , x, verbose = verbose)
    
    # Crop to appropriate shape if required      
    score2_c = KL.Cropping2D(cropping=((1,1),(1,1)), name='fcn16_crop_score2' )(x)                                
    logt('FCN 2x Upsampling/Cropped (Cropped2D(score2)) ', score2_c, verbose = verbose)
   
    # Sum Score2, scorePool4
    x = KL.Add(name='fcn16_fuse_pool4')([score2_c,scorePool4])    
    logt('FCN Add Score2,scorePool4 Add(score2_c, scorePool4)  ' , x, verbose = verbose)
       

    # 2x Upsampling  (padding was originally "valid", I changed it to "same" )
    x  = KL.Deconvolution2D(num_classes, kernel_size=(4,4),  activation = None,  padding = 'same',  name = 'fcn16_upscore_pool4',
                           kernel_initializer='glorot_uniform', bias_initializer='zeros', strides = (2,2) )(x)
                                       
    logt('FCN upscore_pool4 (Deconv(fuse_Pool4)) ' , x, verbose = verbose)
    
    ##-------------------------------------------------------------------------------------------------------
    ## FCN8 Specific Structure 
    ##-------------------------------------------------------------------------------------------------------
    # Score Pool3 - Reduce Pool3 filters from 256 to num_classes (81)
    scorePool3 = KL.Conv2D(num_classes, (1,1), activation='relu', padding='valid', name='fcn8_score_pool3', 
                           kernel_initializer='glorot_uniform',  bias_initializer='zeros')(Pool3)                    
    print()
    print('   --- FCN8 ----------------------------')
    logt('FCN scorePool3 (Conv2D(Pool3))  ' , scorePool3, verbose = verbose)

    upscore_pool4_c = KL.Cropping2D(cropping=((0,0),(0,0)), name='fcn8_crop_pool4')(x)    
    logt('FCN 2x Upsampling/Cropped (Cropped2D(score2)) ' , upscore_pool4_c, verbose = verbose)

    # Sum  upscore_pool4_c, scorePool3
    x = KL.Add(name='fcn8_fuse_pool3')([upscore_pool4_c,scorePool3])    
    logt('FCN Add Score2,scorePool4' ,  x, verbose = verbose) 
    
    print()

    
    ##-------------------------------------------------------------------------------------------------------
    ## fcn_heatmap
    ##-------------------------------------------------------------------------------------------------------    
    # 8x Upsampling  (padding was originally "valid", I changed it to "same" )
    fcn_hm = KL.Deconvolution2D(num_classes, kernel_size=(16,16), activation = None, padding = 'same', name = 'fcn8_heatmap',
                           kernel_initializer='glorot_uniform', bias_initializer='zeros' , strides = (8,8))(x)
    # fcn_hm = tf.identity(fcn_hm)                           
    fcn_hm.set_shape(feature_map.shape)
    logt('FCN fcn8_classify/heatmap  (Deconv(fuse_Pool4)) ' , fcn_hm, verbose = verbose) 
    fcn_hm = KL.Lambda(lambda z: tf.identity(z, name='fcn_hm'), name='fcn_heatmap_lambda') (fcn_hm)
    logt('fcn_hm (final)' , fcn_hm, verbose = verbose) 
    print()

    # fcn_classify_shape = KB.int_shape(fcn_hm)
    # h_factor = height / fcn_classify_shape[1]
    # w_factor = width  / fcn_classify_shape[2]
    # print('   fcn_classify_shape:',fcn_classify_shape,'   h_factor : ', h_factor, '  w_factor : ', w_factor)
    
    # x = BilinearUpSampling2D(size=(h_factor, w_factor), name='fcn_bilinear')(x)
    # print('   FCN Bilinear upsmapling layer  shape is : ' , KB.int_shape(x), ' Keras tensor ', KB.is_keras_tensor(x) )  
    
    ##-------------------------------------------------------------------------------------------------------
    ## fcn_heatmap
    ##-------------------------------------------------------------------------------------------------------    
    fcn_sm = KL.Activation("softmax", name = "fcn8_softmax")(fcn_hm)
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

    return fcn_hm, fcn_sm
    #,fcn_softmax   # fcn_hm_norm, fcn_hm_L2norm
    


    