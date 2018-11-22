"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import glob
import random
import math
import datetime
import numpy as np
import keras.layers  as KL
import keras.backend as KB
import keras.engine  as KE
import tensorflow    as tf
# from keras.layers import *

############################################################
##  BatchNorm Layer
############################################################

class BatchNorm(KL.BatchNormalization):
    """Batch Normalization class. Subclasses the Keras BN class and
    hardcodes training=False so the BN layer doesn't update
    during training.

    Batch normalization has a negative effect on training if batches are small
    so we disable it here.
    """

    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=False)
        
############################################################
##  BilinearUpSampling2D Layer
############################################################

def resize_images_bilinear(X, height_factor=1, width_factor=1, target_height=None, target_width=None, data_format='default'):
    '''
    Resize the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'channels_first' data_format)
    - [batch, height, width, channels] (for 'channels_last' data_format)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    
    X                   Input tensor
    
    target_height       If specified, identifies target height and width post resize  
    target_width
    
    height_factor       If target_height/target_width are not specified, identifies
    width_factor        factor that image height/width will be multiplied by in resize op.
    
    data_format         Identifies format of input tensor: 'default', 'channels_first', 'channels_last'
    '''
    if data_format == 'default':
        data_format = KB.image_data_format()
    
    if data_format == 'channels_first':
        original_shape = KB.int_shape(X)

        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[2:]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        
        # transpose: [batch, channels, height, width] --> [batch, height, width, channels]
        X = permute_dimensions(X, [0, 2, 3, 1])
        X = tf.image.resize_bilinear(X, new_shape, name='fcn_heatmap_channels_first')
        
        # transpose: [batch, height, width, channels] --> [batch, channels, height, width]         
        X = permute_dimensions(X, [0, 3, 1, 2])

        if target_height and target_width:
            X.set_shape((None, None, target_height, target_width))
        else:
            X.set_shape((None, None, original_shape[2] * height_factor, original_shape[3] * width_factor))
        return X
    
    elif data_format == 'channels_last':
        original_shape = KB.int_shape(X)
        print('     CHANNELS LAST: X: ', X.get_shape(), ' KB.int_shape() : ', original_shape)
        print('     target_height   : ', target_height, ' target_width  : ', target_width )
        
        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
            print('     new_shape (1): ' , new_shape.get_shape())            

        else:
            new_shape = tf.shape(X)[1:3]
            print('     new_shape (2): ' , new_shape.get_shape(), new_shape.shape)                        
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
            print('     new_shape (3): ' , new_shape.get_shape(), new_shape.shape)                        
        
        X = tf.image.resize_bilinear(X, new_shape, name = 'fcn_heatmap_channels_last')
        print('     X after image.resize_bilinear: ' , X.get_shape())            
        
        if target_height and target_width:
            X.set_shape((None, target_height, target_width, None))
        else:
            X.set_shape((None, original_shape[1] * height_factor, original_shape[2] * width_factor, None))
        print('     Dimensions of X after set_shape() : ', X.get_shape())
        return X
        
    else:
        raise Exception('Invalid data_format: ' + data_format)


class BilinearUpSampling2D(KE.Layer):
    '''
    Deinfes the Bilinear Upsampling layer

    Returns:
    -------

    pred_tensor :       [batch, NUM_CLASSES, TRAIN_ROIS_PER_IMAGE    , (index, class_prob, y1, x1, y2, x2, class_id, old_idx)]
                                in normalized coordinates   
    pred_cls_cnt:       [batch, NUM_CLASSES] 
    gt_tensor:          [batch, NUM_CLASSES, DETECTION_MAX_INSTANCES, (index, class_prob, y1, x1, y2, x2, class_id, old_idx)]
    gt_cls_cnt:         [batch, NUM_CLASSES]
    
     Note: Returned arrays might be zero padded if not enough target ROIs.
    
    '''
    
    def __init__(self, size=(1, 1), target_size=None, data_format='default', **kwargs):
        if data_format == 'default':
            data_format = KB.image_data_format()
            
        self.size = tuple(size)
        
        if target_size is not None:
            self.target_size = tuple(target_size)
        else:
            self.target_size = None
        
        assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {tf, th}'
        
        self.data_format = data_format
        self.input_spec = [KL.InputSpec(ndim=4)]
        print()
        print('-------------------------------' )
        print('>>> BilinearUpSampling2D layer ' )
        print('-------------------------------' )
        print('     data_format : ', self.data_format)
        print('     size        : ', self.size   )
        print('     target_size : ', self.target_size)
        print('     input_spec  : ', self.input_spec)
        
        # super(BilinearUpSampling2D, self).__init__(**kwargs)
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape): 
        print('     BilinearUpSampling2D. compute_output_shape()' )    
        if self.data_format == 'channels_first':
            width = int(self.size[0] * input_shape[2] if input_shape[2] is not None else None)
            height = int(self.size[1] * input_shape[3] if input_shape[3] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    input_shape[1],
                    width,
                    height)
        elif self.data_format == 'channels_last':
            width  = int(self.size[0] * input_shape[1] if input_shape[1] is not None else None)
            height = int(self.size[1] * input_shape[2] if input_shape[2] is not None else None)
            if self.target_size is not None:
                width  = self.target_size[0]
                height = self.target_size[1]
            print('     Bilinear output shape is:', input_shape[0],',',width,',',height,',',input_shape[3])
            return (input_shape[0],
                    width,
                    height,
                    input_shape[3])
        else:
            raise Exception('Invalid data_format: ' + self.data_format)

    def call(self, x, mask=None):
        if self.target_size is not None:
            print('     call resize_images_bilinear with target_size: ', self.target_size)
            return resize_images_bilinear(x, target_height=self.target_size[0], target_width=self.target_size[1],
                                             data_format=self.data_format)
        else:
            print('     call resize_images_bilinear with size: ', self.size)        
            return resize_images_bilinear(x, height_factor=self.size[0], width_factor=self.size[1], 
                                             data_format=self.data_format)

    def get_config(self):
        print('    BilinearUpSampling2D. get_config()' )
        config = {'size': self.size, 'target_size': self.target_size}
        base_config = super(BilinearUpSampling2D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

############################################################
##  ROIAlign Layer
############################################################

def log2_graph(x):
    """Implementatin of Log2. TF doesn't have a native implemenation."""
    return tf.log(x) / tf.log(2.0)


class PyramidROIAlign(KE.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, chanells]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, image_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape  = tuple(pool_shape)
        self.image_shape = tuple(image_shape)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]
        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[1:]

        print('   > PyramidRoI Alignment Layer Call() ', len(inputs))
        mrcnn_class , mrcnn_bbox,  output_rois, gt_class_ids, gt_bboxes = inputs
        print('     boxes.shape    :',    KB.int_shape( mrcnn_class ))
        

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1

        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(self.image_shape[0] * self.image_shape[1], tf.float32)
        roi_level  = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level  = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level  = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indicies for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            #------------------------------------------------------------------
            ## Crop and Resize
            #   From Mask R-CNN paper: "We sample four regular locations, so
            #   that we can evaluate either max or average pooling. In fact,
            #   interpolating only a single value at each bin center (without
            #   pooling) is nearly as effective."
            #
            #   Here we use the simplified approach of a single value per bin,
            #   which is how it's done in tf.crop_and_resize()
            #   Result: [batch * num_boxes, pool_height, pool_width, channels]
            #------------------------------------------------------------------
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0] ).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        pooled = tf.expand_dims(pooled, 0)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[1][-1], )

        