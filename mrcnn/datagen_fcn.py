"""
Mask R-CNN
Mask R-CNN Data Generator implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Modified to remove MASK related code

"""

import os
import sys
import glob
import random
import math
import datetime
import itertools
import json
import re
import logging
# from collections import OrderedDict
import numpy as np

# import scipy.misc
# import tensorflow as tf
# import keras
# import keras.backend as K
# import keras.layers as KL
# import keras.initializers as KI
# import keras.engine as KE
# import keras.models as KM

import mrcnn.utils as utils

############################################################
##  Data Generator
############################################################
"""
List of Modules:
   load_heatmap_npz :       Load and return ground truth data for an image (image, mask, bboxes)
   
   data_generator   :       A generator that returns images and corresponding target class ids,
                            bounding box deltas, and masks.
   
"""

##----------------------------------------------------------------------
## LOAD_HEATMAP_NPZ
##----------------------------------------------------------------------

def load_heatmap_npz(dataset, config, image_id, augment=False):
    
    """
    Load and return heatmap and score nparrays for an image .

    Inputs:
    --------    
    augment:            If true, apply random image augmentation. Currently, only
                        horizontal flipping is offered.
                        
    Returns:
    ---------
    image:              [height, width, 3]
    pr_hm:              
    pr_hm_scores:    
    gt_hm:      
    gt_hm_scores:               [instance_count, (y1, x1, y2, x2)]
    
    """
    
    # Load image and mask
    # print('=========================')
    # print(' Load Image GT: ', image_id)
    # print('=========================')    
    image         = dataset.load_image(image_id)
    heatmap_data  = dataset.load_image_heatmap(image_id)
    gt_hm         = heatmap_data['gt_hm_norm']
    gt_hm_scores  = heatmap_data['gt_hm_scores']
    pr_hm         = heatmap_data['pr_hm_norm']
    pr_hm_scores  = heatmap_data['pr_hm_scores']
    image_meta    = heatmap_data['input_image_meta']
    print(' load_heatmap_npz() :  Load Image id: ', image_id, ' image_id in image_meta[]: ', image_meta[0], ' coco_id' ,
                dataset.image_info[image_id]['id'])
    print('     coco path: ', dataset.image_info[image_id]['path'])
    print('  heatmap path: ', dataset.image_info[image_id]['heatmap_path'])
    print('  image_meta[0] chaged from :', image_meta[0], ' to : ', image_id)
    image_meta[0] = image_id
    
    # print(  heatmap_data.keys())
    print(' Image shape        : ', image.shape, image.dtype, np.min(image), np.max(image))
    # image, window, scale, padding = utils.resize_image(image,
    image, _ , _ , _  = utils.resize_image(image,
                                     min_dim=config.IMAGE_MIN_DIM,
                                     max_dim=config.IMAGE_MAX_DIM,
                                     padding=config.IMAGE_PADDING)
    
    # print(' gt_heatmap_norm    : ', gt_hm.shape )  
    # print(' gt_heatmap_scores  : ', gt_hm_scores.shape)  
    # print(' pred_heatmap_norm  : ', pr_hm.shape )  
    # print(' pred_heatmap_scores: ', pr_hm_scores.shape)  
    # print(' input_image_meta   : ', image_meta.shape)   
    # print(image_meta)

    # mask, class_ids = dataset.load_mask(image_id)
    # print(mask.shape, class_ids.shape)
    # for  i in range( class_ids.shape[-1]) :
        # print( 'mask ',i, ' class_id :', class_ids[i], mask[:,:,i].shape)
        # print()
        # print(np.array2string(np.where(mask[:,:,i],1,0),max_line_width=134, separator = ''))
    
    # shape = image.shape
    # image, window, scale, padding = utils.resize_image(image,
                                                       # min_dim=config.IMAGE_MIN_DIM,
                                                       # max_dim=config.IMAGE_MAX_DIM,
                                                       # padding=config.IMAGE_PADDING)
    # mask = utils.resize_mask(mask, scale, padding)

    # print('after resize_mask shape is :',mask.shape)
    # Random horizontal flips.
    # if augment:
        # if random.randint(0, 1):
            # image = np.fliplr(image)
            # mask  = np.fliplr(mask)

    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    # bbox = utils.extract_bboxes(mask)
    
    # print('boxes are: \n', bbox)
    
    ## Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    
    # active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    # source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    # active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    # if use_mini_mask:
        # mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)
        # print('after use_mini_mask  shape is :',mask.shape)
    # Image meta data
    # image_meta = utils.compose_image_meta(image_id, shape, window, active_class_ids)

    return  image, pr_hm, pr_hm_scores, gt_hm, gt_hm_scores, image_meta 

    
##----------------------------------------------------------------------
## DATA_GENERATOR
##----------------------------------------------------------------------
def fcn_data_generator(dataset, config, shuffle=True, augment=True, 
                   batch_size=1,  image_index = -1):
    '''
    A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.
    
    Inputs:
    -------
    dataset:                The Dataset object to pick data from
    config:                 The model config object
    shuffle:                If True, shuffles the samples before every epoch
    augment:                If True, applies image augmentation to images (currently only
                            horizontal flips are supported)
                            
    batch_size:             How many images to return in each call
    
    image_index             -1     : Start from beginning (or random position)
                            n <> -1: start from item n+1 in the list when shuffle is False 
                            
    Returns:                A Python generator. Upon calling next() on it, the
    --------                generator returns two lists, [inputs] and [outputs]. The containtes
                            of the lists differs depending on the received arguments:
    [Inputs] return list:
    --------------------
  0 batch_images:           [batch_sz, H, W, C]                                                [1, 128,128,3]
  1 batch_image_meta:       [batch_sz, size of image meta]                                     [1,  12]
  
  2 batch_rpn_match:        [batch_sz, N] Integer (1=positive anchor, -1=negative, 0=neutral)  [1,4092, 1]
  3 batch_rpn_bbox:         [batch_sz, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.      [1, 256, 4]
  
  4 batch_gt_class_ids:     [batch_sz, MAX_GT_INSTANCES] Integer class IDs                     [1, 100]
  5 batch_gt_boxes:         [batch_sz, MAX_GT_INSTANCES, (y1, x1, y2, x2)]                     [1, 100, 4]
  6 batch_gt_masks:         [batch_sz, height, width, MAX_GT_INSTANCES]. The height and width  [1,  56, 56, 100]
                            are those of the image unless use_mini_mask is True, in which
                            case they are defined in MINI_MASK_SHAPE.
                            
    [Outputs] :             Usually empty in regular training.
                            
                            
    Operation outline:
    ------------------
    - generate_pyramid_anchors
        - load ground truth information for current image using load_image_gt
        - at to batch being created 
        - If batch is complete, build [Inputs] list
    '''
    b           = 0  # batch item index
    # commented on 24-10-18 to allow for restarting generator at a certain point. 
    # -1 is now default parm , if not specified in call
    #  image_index = -1  

    image_ids   = np.copy(dataset.image_ids)
    print(' FCN DATAGEN starting image_index  ',image_index, 'len of image ids :', len(image_ids))
    error_count = 0

    # Keras requires a generator to run indefinately.
    while True:
        try:
            #-----------------------------------------------------------------------           
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            #-----------------------------------------------------------------------            
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            #-----------------------------------------------------------------------           
            # Get GT bounding boxes and masks for image.
            #-----------------------------------------------------------------------            
            image_id = image_ids[image_index]
            print('Image index: ', image_index, 'image_id: ', image_id)
            # image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
            image, pr_hm, pr_hm_scores, gt_hm, gt_hm_scores, image_meta = \
                load_heatmap_npz(dataset, config, image_id, augment=augment)

            #-----------------------------------------------------------------------           
            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            #-----------------------------------------------------------------------            
            # if not np.any(gt_class_ids > 0):
                # continue

            #-----------------------------------------------------------------------
            # Init batch arrays
            #-----------------------------------------------------------------------
            if b == 0:
                batch_images      = np.zeros( (batch_size,) + image.shape       , dtype=np.float32)
                batch_image_meta  = np.zeros( (batch_size,) + image_meta.shape  , dtype=image_meta.dtype)
                batch_pr_hm       = np.zeros( (batch_size,) + pr_hm.shape       , dtype=np.float32)
                batch_pr_hm_scores= np.zeros( (batch_size,) + pr_hm_scores.shape, dtype=np.float32)
                batch_gt_hm       = np.zeros( (batch_size,) + gt_hm.shape       , dtype=np.float32)
                batch_gt_hm_scores= np.zeros( (batch_size,) + gt_hm_scores.shape, dtype=np.float32)
                        
            #-----------------------------------------------------------------------    
            # Add to batch
            #-----------------------------------------------------------------------            
            batch_images[b]                               = utils.mold_image(image.astype(np.float32), config)            
            batch_image_meta[b]                           = image_meta
            batch_pr_hm[b]                                = pr_hm
            batch_pr_hm_scores[b]                         = pr_hm_scores
            batch_gt_hm[b]                                = gt_hm
            batch_gt_hm_scores[b]                         = gt_hm_scores
            b += 1
            
            #-----------------------------------------------------------------------            
            # Batch full? send out inputs, outputs
            #-----------------------------------------------------------------------            
            if b >= batch_size:
                images = batch_images 
                inputs = [batch_image_meta, 
                          batch_pr_hm,       
                          batch_pr_hm_scores,
                          batch_gt_hm,       
                          batch_gt_hm_scores
                         ]
                
                outputs = []

                yield inputs, outputs, images

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise

                
##----------------------------------------------------------------------
## DATA_GENERATOR SIMULAION
##----------------------------------------------------------------------
def fcn_data_gen_simulate(dataset, config, image_index):
    '''
    Simulate generator operations, on a specific given image id 
    
    Inputs:
    -------
    dataset:                The Dataset object to pick data from
    config:                 The model config object
    batch_size:             How many images to return in each call
    
    Returns:                Returns two lists, [inputs] and [outputs]. The containtes 
    --------                of the lists differs depending on the received arguments:
                            
    [Inputs] return list:
    --------------------
  0 batch_images:           [batch_sz, H, W, C]                                                [1, 128,128,3]
  1 batch_image_meta:       [batch_sz, size of image meta]                                     [1,  12]
  
  2 batch_rpn_match:        [batch_sz, N] Integer (1=positive anchor, -1=negative, 0=neutral)  [1,4092, 1]
  3 batch_rpn_bbox:         [batch_sz, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.      [1, 256, 4]
  
  4 batch_gt_class_ids:     [batch_sz, MAX_GT_INSTANCES] Integer class IDs                     [1, 100]
  5 batch_gt_boxes:         [batch_sz, MAX_GT_INSTANCES, (y1, x1, y2, x2)]                     [1, 100, 4]
  6 batch_gt_masks:         [batch_sz, height, width, MAX_GT_INSTANCES]. The height and width  [1,  56, 56, 100]
                            are those of the image unless use_mini_mask is True, in which
                            case they are defined in MINI_MASK_SHAPE.
                               
    [Outputs] :             Usually empty in regular training. 
                            
    '''

    augment=False
    # shuffle=True    
    b = 0  # batch item index

    # image_ids   = np.copy(dataset.image_ids)
    
    error_count = 0
    if not isinstance(image_index, list):
        image_index = [image_index]
        print(' Converted to image index --> ',image_index)
    batch_size  = len(image_index)
    print(' batch size is :', batch_size)


    # Keras requires a generator to run indefinately.
    for img_idx in image_index:
        try:
            #-----------------------------------------------------------------------           
            # Get GT bounding boxes and masks for image.
            #-----------------------------------------------------------------------            
            print(' load image id: ', img_idx)
            image_id = dataset.image_ids[img_idx]

            print(' Image index: ', image_index, 'image_id: ', image_id)
            # image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
            image, pr_hm, pr_hm_scores, gt_hm, gt_hm_scores, image_meta = \
                load_heatmap_npz(dataset, config, image_id, augment=augment)

            #-----------------------------------------------------------------------           
            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            #-----------------------------------------------------------------------            
            # if not np.any(gt_class_ids > 0):
                # continue

            #-----------------------------------------------------------------------
            # Init batch arrays
            #-----------------------------------------------------------------------
            if b == 0:
                batch_images      = np.zeros( (batch_size,) + image.shape       , dtype=np.float32)
                batch_image_meta  = np.zeros( (batch_size,) + image_meta.shape  , dtype=image_meta.dtype)
                batch_pr_hm       = np.zeros( (batch_size,) + pr_hm.shape       , dtype=np.float32)
                batch_pr_hm_scores= np.zeros( (batch_size,) + pr_hm_scores.shape, dtype=np.float32)
                batch_gt_hm       = np.zeros( (batch_size,) + gt_hm.shape       , dtype=np.float32)
                batch_gt_hm_scores= np.zeros( (batch_size,) + gt_hm_scores.shape, dtype=np.float32)
                        
            #-----------------------------------------------------------------------    
            # Add to batch
            #-----------------------------------------------------------------------            
            batch_images[b]                               = utils.mold_image(image.astype(np.float32), config)            
            batch_image_meta[b]                           = image_meta
            batch_pr_hm[b]                                = pr_hm
            batch_pr_hm_scores[b]                         = pr_hm_scores
            batch_gt_hm[b]                                = gt_hm
            batch_gt_hm_scores[b]                         = gt_hm_scores
            b += 1
            
            #-----------------------------------------------------------------------            
            # Batch full? send out inputs, outputs
            #-----------------------------------------------------------------------            
            if b >= batch_size:
                print(' Batch size met' )
                images = batch_images
                inputs = [ batch_image_meta, 
                          batch_pr_hm,       
                          batch_pr_hm_scores,
                          batch_gt_hm,       
                          batch_gt_hm_scores
                         ]
                
                outputs = []
                 
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise

    return inputs, outputs, images