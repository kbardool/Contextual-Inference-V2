"""
Mask R-CNN
Contextual Heatmap Layer for Training Mode - Version 1

Copyright (c) 2018 K.Bardool 
Licensed under the MIT License (see LICENSE for details)
Written by Kevin Bardool
"""

import os, sys, glob, random, math, datetime, itertools, logging, pprint, copy 
import numpy as np
# from scipy.stats import  multivariate_normal
import tensorflow as tf
import keras.backend as KB
import keras.layers as KL
import keras.engine as KE
# import tensorflow.contrib.util as tfc

from mrcnn.utils   import logt
import mrcnn.utils as utils


def normalize_scores(x, low = 0.0, high = 1.0, verbose = 0):
    #--------------------------------------------------------------------------------------------
    # this normalization moves values to [low , high]  using he general normalization formula :
    #    
    #                 (X - Xmin) (high - low)
    #     X = low + -------------------------
    #                     (Xmax - Xmin)
    #--------------------------------------------------------------------------------------------    
    reduce_max = tf.reduce_max(x, axis = -2, keepdims=True)
    reduce_min = tf.reduce_min(x, axis = -2, keepdims=True)  ## epsilon    = tf.ones_like(reduce_max) * 1e-7
    reduce_max = tf.where(reduce_max  < +1.0e-15,  tf.ones_like(reduce_max) * high, reduce_max)
    reduce_min = tf.where(reduce_min  > -1.0e-15,  tf.ones_like(reduce_min) * low,  reduce_min)

    y  = (x- reduce_min) / (reduce_max - reduce_min)
    y  = y * (high - low) + low 
    
    logt(' ', verbose = verbose)
    logt(' Normalize_scores() ------------------------------------------------------', verbose = verbose)
    logt('   input shape      ', x, verbose = verbose)
    logt('   reduce_min shape ', reduce_min, verbose = verbose)
    logt('   reduce_max shape ', reduce_max, verbose = verbose)
    logt('       output shape ', y, verbose = verbose)

    return y
    
##--------------------------------------------------------------------------------------------------------
## Build Mask and Score - version 3 
##   Applies a reduced area mask using cx,cy and covar, similar to what is being done for score calc in FCN
##--------------------------------------------------------------------------------------------------------        
def build_hm_score_v3(input_list):
    '''
    Inputs:
    -----------
        heatmap_tensor :    [ image height, image width ]
        input_row      :    [y1, x1, y2, x2] in absolute (non-normalized) scale

    Returns
    -----------
        gaussian_sum :      sum of gaussian heatmap vlaues over the area covered by the bounding box
        bbox_area    :      bounding box area (in pixels)
    '''
    heatmap_tensor, cy, cx, covar = input_list
    with tf.variable_scope('mask_routine'):
        start_y      = tf.maximum(cy-covar[1], 0.0)
        end_y        = tf.minimum(cy+covar[1], KB.int_shape(heatmap_tensor)[0])
        start_x      = tf.maximum(cx-covar[0], 0.0)
        end_x        = tf.minimum(cx+covar[0], KB.int_shape(heatmap_tensor)[1])
        
        #---------------------------------------------------------------------------------------
        # though rounding was an option, after analyzig the output data, opted to not use it. 
        # Also not used in FCN Scoring layer routine    11-26-2018
        #---------------------------------------------------------------------------------------
        # y_extent     = tf.range(tf.round(start_y), tf.round(end_y))  ##  Rounding is NOT USED 
        # x_extent     = tf.range(tf.round(start_x), tf.round(end_x))  ##  here or in FCN scoring
        y_extent     = tf.range(start_y, end_y) 
        x_extent     = tf.range(start_x, end_x)        
        Y,X          = tf.meshgrid(y_extent, x_extent)
    
        mask_indices = tf.stack([Y,X],axis=2)        
        mask_indices = tf.reshape(mask_indices,[-1,2])
        mask_indices = tf.to_int32(mask_indices)
        mask_size    = tf.shape(mask_indices)[0]
        mask_updates = tf.ones([mask_size], dtype = tf.float32)    
        
        mask         = tf.scatter_nd(mask_indices, mask_updates, tf.shape(heatmap_tensor))
        
        heatmap_tensor = tf.multiply(heatmap_tensor, mask, name = 'mask_applied')
        score        = tf.reduce_sum(heatmap_tensor)
        
        mask_area    = tf.to_float((end_y -start_y) * (end_x-  start_x))        
        mask_sum     = tf.reduce_sum(mask)

    # return tf.stack([ score, mask_area, mask_sum,  score/mask_area, score/mask_sum], axis = -1)  
    return tf.stack([ score, mask_sum, score/mask_sum], axis = -1)  


    
##-----------------------------------------------------------------------------------------------------------
## Build Mask and Score - version 2
##----------------------------------------------------------------------------------------------------------- 
def build_hm_score_v2(input_list):
    '''
    Inputs:
    -----------
        heatmap_tensor :    [ image height, image width ]
        input_row      :    [y1, x1, y2, x2] coordinates of bounding box
        input_score    :    input score 
        
    Returns
    -----------
        gaussian_sum :      sum of gaussian heatmap vlaues over the area covered by the bounding box
        bbox_area    :      bounding box area (in pixels)
        weighted_sum :      gaussian_sum * bbox_score
    '''
    heatmap_tensor, input_bbox, input_score = input_list
    
    with tf.variable_scope('mask_routine'):
        y_extent     = tf.range(input_bbox[0], input_bbox[2])
        x_extent     = tf.range(input_bbox[1], input_bbox[3])
        Y,X          = tf.meshgrid(y_extent, x_extent)
        mask_indices = tf.stack([Y,X],axis=2)        
        mask_indices = tf.reshape(mask_indices,[-1,2])
        mask_indices = tf.to_int32(mask_indices)
        mask_size    = tf.shape(mask_indices)[0]
        mask_updates = tf.ones([mask_size], dtype = tf.float32)    
        mask         = tf.scatter_nd(mask_indices, mask_updates, tf.shape(heatmap_tensor))
        # mask_sum    =  tf.reduce_sum(mask)
        heatmap_tensor = tf.multiply(heatmap_tensor, mask, name = 'mask_applied')
        bbox_area    = tf.to_float((input_bbox[2]-input_bbox[0]) * (input_bbox[3]-input_bbox[1]))
        gaussian_sum = tf.reduce_sum(heatmap_tensor)

        # Multiply gaussian_sum by score to obtain weighted sum    
        # weighted_sum = gaussian_sum * input_row[5]
        # Replaced lines above with following lines 21-09-2018
        # Multiply gaussian_sum by normalized score to obtain weighted_norm_sum 
        weighted_norm_sum = gaussian_sum * input_score    # input_list[7]

    return tf.stack([gaussian_sum, bbox_area, weighted_norm_sum], axis = -1)


##-----------------------------------------------------------------------------------------------------------
## Clip Heatmap
##      Clips heatmap to a predefined vicinity (+/- 5 pixels of cy,cx)
##----------------------------------------------------------------------------------------------------------- 
def clip_heatmap(input_list):
    '''
    Inputs:
    -----------
        heatmap_tensor :    [ image height, image width ]
        cy,cx, covar   :   
        
    Returns
    -----------
        Clipped heatmap tensor 
    '''
    heatmap_tensor, cy, cx, covar = input_list

    with tf.variable_scope('mask_routine'):
        start_y      = tf.maximum(cy-covar[1],0)
        end_y        = tf.minimum(cy+covar[1], KB.int_shape(heatmap_tensor)[0])
        start_x      = tf.maximum(cx-covar[0],0)
        end_x        = tf.minimum(cx+covar[0], KB.int_shape(heatmap_tensor)[1])
        y_extent     = tf.range(start_y, end_y)
        x_extent     = tf.range(start_x, end_x)
        Y,X          = tf.meshgrid(y_extent, x_extent)
        mask_indices = tf.stack([Y,X],axis=2)        
        mask_indices = tf.reshape(mask_indices,[-1,2])
        mask_indices = tf.to_int32(mask_indices)
        mask_size    = tf.shape(mask_indices)[0]
        mask_updates = tf.ones([mask_size], dtype = tf.float32)    
        mask         = tf.scatter_nd(mask_indices, mask_updates, tf.shape(heatmap_tensor))
        # mask_sum    =  tf.reduce_sum(mask)
        heatmap_tensor = tf.multiply(heatmap_tensor, mask, name = 'mask_applied')

    return  heatmap_tensor

    
##-----------------------------------------------------------------------------------------------------------
##  build_pr_heatmap : Build gaussian heatmaps using pred_tensor
##-----------------------------------------------------------------------------------------------------------  
##  v3: In this version we replace the score generation routine from build_hm_score_v2 which operated over 
##      The complete bounding box area, to build_hm_score_v3, which calculates the score in a method similar to what 
##      is done in the fcn_scoring_layer (clipping on a small box and calulcating the sum over this region. 
##      
##      A number of different scores (alt_score_*) were calcualted and one was finally settled on, the others are
##      commented out for reference purposes for the time being.
##
##  v2: in this version, 
##      For heatmap generation, prob_grid is passed through "clip_heatmap" which clips the gaussian distribution
##      based on Cx, Cy and the Covar parms for each bounding box. This prob_grid_clipped is then passed on to 
##      next steps (per-class normalization, 
##      "build_hm_score_v2" function which calculates scores is applied to "prob_grid", 
##       
## Inputs:
##      in_tensor - [BATCH_SIZE, NUM_CLASSES, DETECTIONS_PER_CLASS, 8]
##                  per-class tensor of bounding boxes (Y1,X1,Y2,X2), class_ids, mrcnn_predicted scores, 
##                  sequence_id, and per-class normalized scores       
##      config    - Model configuration object
##
## Outputs:
##
##   gauss_heatmap
##   gauss_scores - [BATCH_SIZE, NUM_CLASSES, DETECTIONS_PER_CLASS, 11]
##                  same as in_tensor, adding :
##                  - sum of heatmap in masked area
##                  - area of bounding box in pixes
##                  - (sum of heatmap in masked area) * (bbox per-class normalized score from in_tensor)
##------------------------------------------------------------------------------------------------------------
def build_pr_heatmap(in_tensor, config, names = None):
    '''
    input:
    -------
    pred_tensor:        [ Bsz, Num_Classes, Num_Rois, 8: 
                            {y1, x1, y2, x2, class_id, score, sequence_id, score normalized per class}]
                        
    output:
    -------
        pr_heatmap      (None,  Heatmap-height, Heatmap_width, num_classes)
        pr_scores       (None, num_classes, 200, 23) 
                        [batchSz, Detection_Max_instance, (y1,x1,y2,x2, class, score, sequence_id, normalized_score,
                                                           scores-0: gaussian_sum, bbox_area, weighted_norm_sum 
                                                           scores-1: score, mask_sum, score/mask_sum, (score, mask_sum, score/mask_sum) normalized by class
                                                           scores-2: score, mask_sum, score/mask_sum, (score, mask_sum, score/mask_sum) normalized by class ]
    '''
    verbose           = config.VERBOSE
    img_h, img_w      = config.IMAGE_SHAPE[:2]
    batch_size        = config.BATCH_SIZE
    num_classes       = config.NUM_CLASSES  
    heatmap_scale     = config.HEATMAP_SCALE_FACTOR
    grid_h, grid_w    = config.IMAGE_SHAPE[:2] // heatmap_scale    
    CLASS_COLUMN      = 4
    SCORE_COLUMN      = 5
    DT_TYPE_COLUMN    = 6
    SEQUENCE_COLUMN   = 7
    NORM_SCORE_COLUMN = 8

    # rois per image is determined by size of input tensor 
    #   detection mode:   config.TRAIN_ROIS_PER_IMAGE 
    #   ground_truth  :   config.DETECTION_MAX_INSTANCES
    #   strt_cls        = 0 if rois_per_image == 32 else 1
    # rois_per_image  = config.DETECTION_PER_CLASS
    rois_per_image  = (in_tensor.shape)[2]  

    if verbose:
        print('\n ')
        print('  > build_pr_heatmap() for : ', names )
        print('    in_tensor shape        : ', in_tensor.shape)       
        print('    num bboxes per class   : ', rois_per_image )
        print('    heatmap scale          : ', heatmap_scale, 'Dimensions:  w:', grid_w,' h:', grid_h)

    ##-----------------------------------------------------------------------------    
    ## Stack non_zero bboxes from in_tensor into pt2_dense 
    ##-----------------------------------------------------------------------------
    # pt2_ind shape is [?, 3]. 
    #    pt2_ind[0] corresponds to image_index 
    #    pt2_ind[1] corresponds to class_index 
    #    pt2_ind[2] corresponds to roi row_index 
    # pt2_dense shape is [?, 7]
    #    pt2_dense[0:3]  roi coordinates 
    #    pt2_dense[4]    is class id 
    #    pt2_dense[5]    is score from mrcnn    
    #    pt2_dense[6]    is bbox sequence id    
    #    pt2_dense[7]    is normalized score (per class)    
    #-----------------------------------------------------------------------------
    pt2_sum = tf.reduce_sum(tf.abs(in_tensor[:,:,:,:4]), axis=-1)
    pt2_ind = tf.where(pt2_sum > 0)
    pt2_dense = tf.gather_nd( in_tensor, pt2_ind)

    logt('pt2_sum   ', pt2_sum, verbose = verbose)
    logt('pt2_ind   ', pt2_ind, verbose = verbose)
    logt('pt2_dense ', pt2_dense, verbose = verbose)

    ##-----------------------------------------------------------------------------
    ## Build mesh-grid to hold pixel coordinates  
    ##-----------------------------------------------------------------------------
    X = tf.range(grid_w, dtype=tf.int32)
    Y = tf.range(grid_h, dtype=tf.int32)
    X, Y = tf.meshgrid(X, Y)

    # duplicate (repeat) X and Y into a  batch_size x rois_per_image tensor
    ones = tf.ones([tf.shape(pt2_dense)[0] , 1, 1], dtype = tf.int32)
    rep_X = ones * X
    rep_Y = ones * Y 
    
    if verbose:
        print('    X/Y shapes :',  X.get_shape(), Y.get_shape())
        print('    Ones:    ', ones.shape)                
        print('    ones_exp * X', ones.shape, '*', X.shape, '= ',rep_X.shape)
        print('    ones_exp * Y', ones.shape, '*', Y.shape, '= ',rep_Y.shape)

    # # stack the X and Y grids 
    pos_grid = tf.to_float(tf.stack([rep_X,rep_Y], axis = -1))
    logt('pos_grid before transpse ', pos_grid, verbose = verbose)
    pos_grid = tf.transpose(pos_grid,[1,2,0,3])
    logt('pos_grid after transpose ', pos_grid, verbose = verbose)  

    ##-----------------------------------------------------------------------------
    ##  Build mean and convariance tensors for Multivariate Normal Distribution 
    ##-----------------------------------------------------------------------------
    bboxes_scaled = pt2_dense[:,:4]/heatmap_scale
    width  = bboxes_scaled[:,3] - bboxes_scaled[:,1]      # x2 - x1
    height = bboxes_scaled[:,2] - bboxes_scaled[:,0]
    cx     = bboxes_scaled[:,1] + ( width  / 2.0)
    cy     = bboxes_scaled[:,0] + ( height / 2.0)
    means  = tf.stack((cx,cy),axis = -1)
    covar  = tf.stack((width * 0.5 , height * 0.5), axis = -1)
    covar  = tf.sqrt(covar)

    ##-----------------------------------------------------------------------------
    ##  Compute Normal Distribution for bounding boxes
    ##-----------------------------------------------------------------------------    
    tfd = tf.contrib.distributions
    mvn = tfd.MultivariateNormalDiag(loc = means,  scale_diag = covar)
    prob_grid = mvn.prob(pos_grid)
    logt('Input to MVN.PROB: pos_grid (meshgrid) ', pos_grid, verbose = verbose)
    logt('Prob_grid shape from mvn.probe  ',prob_grid, verbose = verbose)
    prob_grid = tf.transpose(prob_grid,[2,0,1])
    logt('Prob_grid shape after tanspose ', prob_grid, verbose = verbose)
    logt('Output probabilities shape   '  , prob_grid, verbose = verbose)

    ##--------------------------------------------------------------------------------------------
    ## (0) Generate scores using prob_grid and pt2_dense - (NEW METHOD added 09-21-2018)
    ##--------------------------------------------------------------------------------------------
    old_style_scores = tf.map_fn(build_hm_score_v2, [prob_grid, bboxes_scaled, pt2_dense[ :, NORM_SCORE_COLUMN ] ], 
                                 dtype = tf.float32, swap_memory = True)
    old_style_scores = tf.scatter_nd(pt2_ind, old_style_scores, 
                                     [batch_size, num_classes, rois_per_image, KB.int_shape(old_style_scores)[-1]],
                                     name = 'scores_scattered')
    logt('old_style_scores        :',  old_style_scores, verbose = verbose)

    ##----------------------------------------------------------------------------------------------------
    ## Generate scores using same method as FCN, over the prob_grid
    ## using (prob_grid_clipped) as input is superfluous == RETURNS EXACT SAME Results AS prob_grid above
    ##----------------------------------------------------------------------------------------------------
    # alt_scores_0 = tf.map_fn(build_hm_score_v3, [prob_grid, cy, cx,covar], dtype=tf.float32)    
    # print('    alt_scores_0 : ', KB.int_shape(alt_scores_0), ' Keras tensor ', KB.is_keras_tensor(alt_scores_0) )
    # alt_scores_0 = tf.scatter_nd(pt2_ind, alt_scores_0, 
    #                                  [batch_size, num_classes, rois_per_image, KB.int_shape(alt_scores_0)[-1]], name = 'alt_scores_0')

    ##---------------------------------------------------------------------------------------------
    ## (NEW STEP - Clipped heatmaps) 
    ## (1)  Clip heatmap to region surrounding Cy,Cx and Covar X, Y 
    ##      Similar ro what is being done for gt_heatmap in CHMLayerTarget 
    ##---------------------------------------------------------------------------------------------    
    prob_grid_clipped = tf.map_fn(clip_heatmap, [prob_grid, cy,cx, covar], dtype = tf.float32, swap_memory = True)  
    logt('    prob_grid_clipped : ', prob_grid_clipped, verbose = verbose)


    ##---------------------------------------------------------------------------------------------
    ## (2) apply normalization per bbox heatmap instance
    ##---------------------------------------------------------------------------------------------
    logt('\n    normalization ------------------------------------------------------', verbose = verbose)   
    normalizer = tf.reduce_max(prob_grid_clipped, axis=[-2,-1], keepdims = True)
    normalizer = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    logt('    normalizer     : ', normalizer, verbose = verbose)
    prob_grid_cns = prob_grid_clipped / normalizer


    ##---------------------------------------------------------------------------------------------
    ## (3) multiply normalized heatmap by normalized score in in_tensor/ (pt2_dense NORM_SCORE_COLUMN)
    ##     broadcasting : https://stackoverflow.com/questions/49705831/automatic-broadcasting-in-tensorflow
    ##---------------------------------------------------------------------------------------------    
    prob_grid_cns = tf.transpose(tf.transpose(prob_grid_cns) * pt2_dense[ :, NORM_SCORE_COLUMN ])
    logt('    prob_grid_cns: clipped/normed/scaled : ', prob_grid_cns, verbose = verbose)


    ##---------------------------------------------------------------------------------------------
    ## - Build alternative scores based on normalized/scaled/clipped heatmap
    ##---------------------------------------------------------------------------------------------
    alt_scores_1 = tf.map_fn(build_hm_score_v3, [prob_grid_cns, cy, cx,covar], dtype=tf.float32)    
    logt('alt_scores_1 ', alt_scores_1, verbose = verbose)
    alt_scores_1 = tf.scatter_nd(pt2_ind, alt_scores_1, 
                                     [batch_size, num_classes, rois_per_image, KB.int_shape(alt_scores_1)[-1]], name = 'alt_scores_1')  

    logt('alt_scores_1(by class) ', alt_scores_1, verbose = verbose)
    alt_scores_1_norm = normalize_scores(alt_scores_1)
    logt('alt_scores_1_norm(by_class) ', alt_scores_1_norm, verbose = verbose)
    # alt_scores_1_norm = tf.gather_nd(alt_scores_1_norm, pt2_ind)
    # print('    alt_scores_1_norm(by_image)  : ', alt_scores_1_norm.shape, KB.int_shape(alt_scores_1_norm))
                                     
    ##-------------------------------------------------------------------------------------
    ## (3) scatter out the probability distributions based on class 
    ##-------------------------------------------------------------------------------------
    gauss_heatmap   = tf.scatter_nd(pt2_ind, prob_grid_cns, 
                                    [batch_size, num_classes, rois_per_image, grid_w, grid_h], name = 'gauss_scatter')
    logt('\n    Scatter out the probability distributions based on class --------------', verbose = verbose)
    logt('pt2_ind shape      ', pt2_ind      , verbose = verbose)
    logt('prob_grid_clippped ', prob_grid_cns, verbose = verbose)
    logt('gauss_heatmap      ', gauss_heatmap, verbose = verbose)   # batch_sz , num_classes, num_rois, image_h, image_w

    ##-------------------------------------------------------------------------------------
    ## Construction of Gaussian Heatmap output using Reduce SUM
    ##
    ## (4) SUM : Reduce and sum up gauss_heatmaps by class  
    ## (5) heatmap normalization (per class)
    ## (6) Transpose heatmap to shape required for FCN
    ##-------------------------------------------------------------------------------------
    gauss_heatmap_sum = tf.reduce_sum(gauss_heatmap, axis=2, name='gauss_heatmap_sum')
    logt('\n    Reduce SUM based on class and normalize within each class -----------------------', verbose = verbose)
    logt('gaussian_heatmap_sum ', gauss_heatmap_sum , verbose = verbose)

    ## normalize in class
    normalizer = tf.reduce_max(gauss_heatmap_sum, axis=[-2,-1], keepdims = True)
    normalizer = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    gauss_heatmap_sum = gauss_heatmap_sum / normalizer
    # gauss_heatmap_sum_normalized = gauss_heatmap_sum / normalizer
    logt('normalizer shape   : ', normalizer, verbose = verbose)
    logt('normalized heatmap : ', gauss_heatmap_sum, verbose = verbose)
    
    ##---------------------------------------------------------------------------------------------
    ##  Score on reduced sum heatmaps. 
    ##
    ##  build indices and extract heatmaps corresponding to each bounding boxes' class id
    ##  build alternative scores#  based on normalized/sclaked clipped heatmap
    ##---------------------------------------------------------------------------------------------
    hm_indices = tf.cast(pt2_ind[:, :2],dtype=tf.int32)
    logt('hm_indices   ',  hm_indices, verbose = verbose)
    
    pt2_heatmaps = tf.gather_nd(gauss_heatmap_sum, hm_indices )
    logt('pt2_heatmaps ',  pt2_heatmaps, verbose = verbose)

    alt_scores_2 = tf.map_fn(build_hm_score_v3, [pt2_heatmaps, cy, cx,covar], dtype=tf.float32)    
    logt('    alt_scores_2    : ', alt_scores_2, verbose = verbose)
    
    alt_scores_2 = tf.scatter_nd(pt2_ind, alt_scores_2, 
                                [batch_size, num_classes, rois_per_image, KB.int_shape(alt_scores_2)[-1]], name = 'alt_scores_2')  
    
    logt('alt_scores_2(scattered)     ', alt_scores_2, verbose = verbose)
    alt_scores_2_norm = normalize_scores(alt_scores_2)
    logt('alt_scores_2_norm(by_class) ', alt_scores_2_norm, verbose = verbose)
 
    # ##---------------------------------------------------------------------------------------------
    # ## Construction of Gaussian Heatmap output using Reduce MAX
    # ## (4) MAX : Reduce and sum up gauss_heatmaps by class  
    # ## (5) heatmap normalization
    # ## (6) Transpose heatmap to shape required for FCN
    # ##---------------------------------------------------------------------------------------------
    # print('\n    Reduce MAX based on class and normalize within each class -------------------------------------')         
    # gauss_heatmap_max = tf.reduce_max(gauss_heatmap, axis=2, name='gauss_heatmap_max')
    # print('    gaussian_heatmap : ', gauss_heatmap_max.get_shape(), 'Keras tensor ', KB.is_keras_tensor(gauss_heatmap_max) )      
    # ## normalize in class
    # normalizer = tf.reduce_max(gauss_heatmap_max, axis=[-2,-1], keepdims = True)
    # normalizer = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    # gauss_heatmap_max            = gauss_heatmap_max / normalizer
    # # gauss_heatmap_max_normalized = gauss_heatmap_max / normalizer
    # print('    normalizer shape   : ', normalizer.shape)   
    # print('    normalized heatmap_max : ', gauss_heatmap_max.shape   ,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap_max) )

    
    # ##---------------------------------------------------------------------------------------------
    # ##  Score on reduced max heatmaps. 
    # ##
    # ##   # ##  build indices and extract heatmaps corresponding to each bounding boxes' class id
    # ##  build alternative scores based on normalized/sclaked clipped heatmap
    # ##---------------------------------------------------------------------------------------------
    # hm_indices = tf.cast(pt2_ind[:, :2],dtype=tf.int32)
    # print('    hm_indices shape         :',  hm_indices.get_shape(), KB.int_shape(hm_indices))
    # pt2_heatmaps = tf.gather_nd(gauss_heatmap_max_normalized, hm_indices )
    # print('    pt2_heatmaps             :',  pt2_heatmaps.get_shape(), KB.int_shape(pt2_heatmaps))

    # alt_scores_3 = tf.map_fn(build_hm_score_v3, [pt2_heatmaps, cy, cx,covar], dtype=tf.float32)    
    # print('    alt_scores_3    : ', KB.int_shape(alt_scores_3), ' Keras tensor ', KB.is_keras_tensor(alt_scores_3) )
    # alt_scores_3 = tf.scatter_nd(pt2_ind, alt_scores_3, 
    #                                  [batch_size, num_classes, rois_per_image, KB.int_shape(alt_scores_3)[-1]], name = 'alt_scores_3')  

    ##---------------------------------------------------------------------------------------------
    ## (6) Transpose heatmaps to shape required for FCN [batchsize , width, height, num_classes]
    ##---------------------------------------------------------------------------------------------    
    gauss_heatmap_sum            = tf.transpose(gauss_heatmap_sum           ,[0,2,3,1], name = names[0])
    logt('reshaped heatmap ', gauss_heatmap_sum, verbose = verbose)
    # gauss_heatmap_sum_normalized = tf.transpose(gauss_heatmap_sum_normalized,[0,2,3,1], name = names[0]+'_norm')   
    # print('    reshaped heatmap normalized    : ', gauss_heatmap_sum_normalized.shape,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap_sum_normalized) )

    # gauss_heatmap_max            = tf.transpose(gauss_heatmap_max           ,[0,2,3,1], name = names[0]+'_max')
    # print('    reshaped heatmap_max           : ', gauss_heatmap_max.shape,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap_max) )
    # gauss_heatmap_max_normalized = tf.transpose(gauss_heatmap_max_normalized,[0,2,3,1], name = names[0]+'_max_norm') 
    # print('    reshaped heatmap_max normalized: ', gauss_heatmap_max_normalized.shape,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap_max_normalized) )
    
    ##--------------------------------------------------------------------------------------------
    ## APPEND ALL SCORES TO input score tensor TO YIELD output scores tensor
    ##--------------------------------------------------------------------------------------------
    gauss_scores     = tf.concat([in_tensor, old_style_scores, alt_scores_1, alt_scores_1_norm, alt_scores_2, alt_scores_2_norm],
                                  axis = -1,name = names[0]+'_scores')
    logt('    gauss_scores    : ', gauss_scores, verbose = verbose)
    logt('    complete', verbose = verbose)

    return   gauss_heatmap_sum, gauss_scores  
    
    
    
    
    
##-----------------------------------------------------------------------------------------------------------
##  build_pr_tensor : Build pred_tensor
##-----------------------------------------------------------------------------------------------------------
##  09-21-2018: 
##    **  build_pr_tensor --> build_refined_predictions
##        build_predicitions routine now applies predicted refinement (from mrcnn_bbox deltas) to the 
##        output_rois. After applying the delta, the refined output rois are clipped so they dont exceed
##        the NN image dimensions. The modified coordiantes of the bboxes are placed in pred_tensor for 
##        futher processing           
##
##    **  renamed build_mask_routine --> build_hm_bbox_score
##        better reflects the function
##
##    ** Modified build_headmap : 
##     - Added HEATMAP_SCALE_FACTOR
##  
##     - Scores are now built using dense tensors : Instead of using `gauss_heatmap` (gauss_sum) and 
##       `in_tensor`, where the process of flattening and running build_hm_bbox_score against them 
##       includes a large number of empty [0 0 0 0] bboxes (since gauss_heatmap is replicated  
##       [num_classes x num_bbox_per_class] times), the score calculation will be done on prob_grid 
##       and pt2_dense , which only include the real number of non-zero bboxes.                  
##       The scores are build using the same method (map_fn)
##       Scores are 1) sum of gaussian distribution within bounding box 
##                      2) area of bounding box in pixels (not used)
##                      3) sum of gaussian within bounding bouding box * normalized score. 
##
##-----------------------------------------------------------------------------------------------------------   
def build_pr_tensor(norm_input_rois, mrcnn_class, mrcnn_bbox, config):
    '''
    Split output_rois by class id, and add class_id and class_score 
    
    input
    ------
    input_rois          Normalized Proposed ROIs   [Batch Size, NumRois, 4]
    mrcnn_class
    mrcnn_bbox 
    
    
    output:
    -------
    pred_tensor:        [ Bsz, Num_Classes, Num_Rois, 8: 
                            {y1, x1, y2, x2, class_id, score, sequence_id, score normalized per class}]
                        
                        y1,x1, y2,x2 are in image dimension format
    '''
    verbose         = config.VERBOSE
    batch_size      = config.BATCH_SIZE
    num_classes     = config.NUM_CLASSES
    h, w            = config.IMAGE_SHAPE[:2]
    # num_rois        = config.TRAIN_ROIS_PER_IMAGE
    num_rois        = KB.int_shape(norm_input_rois)[1]
    scale           = tf.constant([h,w,h,w], dtype = tf.float32)
    # dup_scale       = tf.reshape(tf.tile(scale, [num_rois]),[num_rois,-1])
    dup_scale       = scale * tf.ones([batch_size, num_rois, 1], dtype = 'float32')
    det_per_class   = config.TRAIN_ROIS_PER_IMAGE
    CLASS_COLUMN      = 4
    SCORE_COLUMN      = 5
    DT_TYPE_COLUMN    = 6
    SEQUENCE_COLUMN   = 7
    NORM_SCORE_COLUMN = 8
    
    if verbose:
        print()
        print('  > build_pr_tensor()')
        print('    num_rois               : ', num_rois )
        print('    norm_input_rois.shape  : ', type(norm_input_rois), KB.int_shape(norm_input_rois))
        print('    scale.shape            : ', type(scale), KB.int_shape(scale), scale.get_shape())
        print('    dup_scale.shape        : ', type(dup_scale), KB.int_shape(dup_scale), dup_scale.get_shape())
        print()
        print('    mrcnn_class shape      : ', KB.int_shape(mrcnn_class))
        print('    mrcnn_bbox.shape       : ', KB.int_shape(mrcnn_bbox), mrcnn_bbox.shape )
        print('    config image shape     : ', config.IMAGE_SHAPE, 'h:',h,'w:',w)

    #---------------------------------------------------------------------------
    # Build a meshgrid for image id and bbox to use in gathering of bbox delta information 
    #---------------------------------------------------------------------------
    batch_grid, bbox_grid = tf.meshgrid( tf.range(batch_size, dtype=tf.int32),
                                         tf.range(num_rois, dtype=tf.int32), indexing = 'ij' )

    #------------------------------------------------------------------------------------
    # use the argmaxof each row to determine the dominating (predicted) class
    #------------------------------------------------------------------------------------
    pred_classes = tf.argmax( mrcnn_class,axis=-1,output_type = tf.int32)
    pred_classes_exp = tf.to_float(tf.expand_dims(pred_classes ,axis=-1))    
    
    gather_ind   = tf.stack([batch_grid , bbox_grid, pred_classes],axis = -1)
    pred_scores  = tf.gather_nd(mrcnn_class, gather_ind)
    pred_scores  = tf.expand_dims(pred_scores, axis = -1)    
    pred_deltas  = tf.gather_nd(mrcnn_bbox , gather_ind)
    
    ##------------------------------------------------------------------------------------
    ## apply delta refinements to the rois,  based on deltas provided by the mrcnn head 
    ##------------------------------------------------------------------------------------
    pred_deltas  = tf.multiply(pred_deltas, config.BBOX_STD_DEV, name = 'pred_deltas')
    input_rois   = tf.multiply(norm_input_rois , dup_scale )
    refined_rois = utils.apply_box_deltas_tf(input_rois, pred_deltas)

    ##   Clip refined boxes to image window    
    window = tf.constant([[0,0,h,w]], dtype = tf.float32)
    refined_rois  = utils.clip_to_window_tf( window, refined_rois)
    
    logt('refined rois ', refined_rois, verbose = verbose)
    logt('input_rois   ', input_rois  , verbose = verbose)
    logt('refined_rois ', refined_rois, verbose = verbose)

    ##------------------------------------------------------------------------------------
    ## 31-01-2019 Added to prevent NaN that occur when delta refinement causes the 
    ## boxes coordiantes to generate a zero area bbox. 
    ##------------------------------------------------------------------------------------
    refined_roi_area = (refined_rois[...,3] - refined_rois[...,1])*(refined_rois[...,2] - refined_rois[...,0])
    nonzero_ix = tf.where(refined_roi_area > 0)
    logt('refined_roi_area ', refined_roi_area, verbose = verbose)
    logt(' nonzero_ix', nonzero_ix, verbose = verbose)

    nonzero_rois    = tf.gather_nd( refined_rois, nonzero_ix) 
    nonzero_scores  = tf.gather_nd( pred_scores , nonzero_ix) 
    nonzero_classes = tf.gather_nd( pred_classes_exp , nonzero_ix) 

    clean_rois      = tf.scatter_nd(nonzero_ix, nonzero_rois  , refined_rois.shape)
    clean_scores    = tf.scatter_nd(nonzero_ix, nonzero_scores, pred_scores.shape)
    clean_classes   = tf.scatter_nd(nonzero_ix, nonzero_classes, pred_classes_exp.shape)

    logt('clean_rois   ', clean_rois, verbose = verbose)
    logt('clean_scores ', clean_scores, verbose = verbose)
    logt('clean_classes ', clean_classes, verbose = verbose)

    flag = tf.where(clean_classes > 0, tf.ones_like(clean_classes, dtype=tf.float32), tf.zeros_like(clean_classes, dtype=tf.float32))
    logt('flag:', flag)
    
    ##------------------------------------------------------------------------------------
    ##  Build Pred_Scatter: tensor of bounding boxes by Image / Class
    ##------------------------------------------------------------------------------------
    ## sequence id is used to preserve the order of rois as passed to this routine
    ##  This may be important in the post matching process but for now it's not being used.
    ## 22-09-18 : We need to use this sequence as the sort process based on score will cause
    ##            mismatch between the bboxes from output_rois and roi_gt_bboxes
    ##------------------------------------------------------------------------------------
    sequence = tf.ones_like(pred_classes, dtype = tf.int32) * (bbox_grid[...,::-1] + 1) 
    sequence = tf.to_float(tf.expand_dims(sequence, axis = -1))   
    logt('sequence ', sequence, verbose = verbose)
    
    
    ##--------------------------------------------------------------------------------------------------
    ##  Build array of bbox coordinates, pred_classes, scores, and sequence ids
    ##  Scatter by class_id to create pred_scatter (Batch_Sz, Num_classes, Num_detections, num_columns)
    ##--------------------------------------------------------------------------------------------------
    # pred_array  = tf.concat([ refined_rois, pred_classes_exp , pred_scores, sequence],
                            # axis=-1, name = 'pred_array')
    ## 31-01-2019: Replaced above line with following line to insert clean roi and scores:
    ##-----------------------------------------------------------------------------------
    # pred_array  = tf.concat([ clean_rois, pred_classes_exp , clean_scores , sequence], axis=-1, name = 'pred_array')
    pred_array  = tf.concat([ clean_rois, clean_classes , clean_scores , flag, sequence], axis=-1, name = 'pred_array')

    
    scatter_ind = tf.stack([batch_grid , pred_classes, bbox_grid],axis = -1)
    pred_scatt  = tf.scatter_nd(scatter_ind, pred_array, [batch_size, num_classes, num_rois, pred_array.shape[-1]])
    logt('pred_array   ', pred_array, verbose = verbose)
    logt('scatter_ind  ', scatter_ind, verbose = verbose)
    logt('pred_scatter ', pred_scatt, verbose = verbose)
    
    ##--------------------------------------------------------------------------------------------
    ##  Apply a per class score normalization using the score column (COLUMN 5)
    ##--------------------------------------------------------------------------------------------
    normalizer   = tf.reduce_max(pred_scatt[...,5], axis = -1, keepdims=True)
    normalizer   = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    norm_score   = tf.expand_dims(pred_scatt[...,5]/normalizer, axis = -1)
    pred_scatt   = tf.concat([pred_scatt, norm_score],axis = -1)   
    logt('- Add normalized score --\n', verbose = verbose)
    logt('normalizer   ', normalizer, verbose = verbose)
    logt('norm_score   ', norm_score, verbose = verbose)
    logt('pred_scatter ', pred_scatt, verbose = verbose)
    
    ##------------------------------------------------------------------------------------
    ## Sort pred_scatt in each class dimension based on sequence number, to push valid  
    ##      to top for each class dimension
    ##
    ## 22-09-2018: sort is now based on sequence which was added as last column
    ##             (previously sort was on bbox scores)
    ##------------------------------------------------------------------------------------
    _, sort_inds = tf.nn.top_k(pred_scatt[...,6], k=pred_scatt.shape[2])
    
    # build indexes to gather rows from pred_scatter based on sort order    
    class_grid, batch_grid, roi_grid = tf.meshgrid(tf.range(num_classes),tf.range(batch_size), tf.range(num_rois))
    
    gather_inds  = tf.stack([batch_grid , class_grid, sort_inds],axis = -1)
    pred_tensor  = tf.gather_nd(pred_scatt, gather_inds, name = 'pred_tensor')    

    logt('sort_inds      ', sort_inds  , verbose = verbose)
    logt('class_grid     ', class_grid , verbose = verbose)
    logt('batch_grid     ', batch_grid , verbose = verbose)
    logt('roi_grid shape ', roi_grid   , verbose = verbose)
    logt('gather_inds    ', gather_inds, verbose = verbose)
    logt('pred_tensor    ', pred_tensor, verbose = verbose)

    return  pred_tensor
    


##-----------------------------------------------------------------------------------------------------------
##  build_pr_tensor_np : Build pr_tensor in numpy mode
##-----------------------------------------------------------------------------------------------------------
def build_pr_tensor_np(norm_output_rois, mrcnn_class, mrcnn_bbox, target_class_ids, config, verbose = 0):
    '''
    Split output_rois by class id, and add class_id and class_score 
    
    input
    ------
    input_rois          Normalized Proposed ROIs   [Batch Size, NumRois, 4]
    mrcnn_class
    mrcnn_bbox 

    output:
    -------
    pred_tensor:        [ Bsz, Num_Classes, Num_Rois, 8: 
                          {y1, x1, y2, x2, class_id, score, sequence_id, score normalized per class}]                    
                          y1,x1, y2,x2 are in image dimension format
    '''
#     verbose         = config.VERBOSE
    batch_size      = config.BATCH_SIZE
    num_classes     = config.NUM_CLASSES
    h, w            = config.IMAGE_SHAPE[:2]
    num_rois1       = config.TRAIN_ROIS_PER_IMAGE
    num_rois2       = norm_output_rois.shape[1]    
    scale           = np.array([h,w,h,w], dtype = np.float32)
    det_per_class   = config.TRAIN_ROIS_PER_IMAGE
    
    CLASS_COLUMN      = 4
    SCORE_COLUMN      = 5
    DT_TYPE_COLUMN    = 6
    SEQUENCE_COLUMN   = 7
    NORM_SCORE_COLUMN = 8
    
    if verbose:
        logt()
        logt('  > build_pr_tensor()')
        logt('    num_rois1  ', num_rois1 )
        logt('    num_rois2  ', num_rois2 )
        logt('    norm_input_rois ', norm_output_rois)
        logt('    scale      ', scale)
        logt()
        logt('    mrcnn_class', mrcnn_class)
        logt('    mrcnn_bbox ', mrcnn_bbox )
        logt('    config.image_shape ', config.IMAGE_SHAPE)

    #---------------------------------------------------------------------------
    # Build a meshgrid for image id and bbox to use in gathering of bbox delta information 
    #---------------------------------------------------------------------------
    bbox_idx = np.arange(config.TRAIN_ROIS_PER_IMAGE)
    
    #------------------------------------------------------------------------------------
    # use the argmaxof each row to determine the dominating (predicted) class
    #------------------------------------------------------------------------------------
    pred_scores  = np.max(mrcnn_class, axis = -1, keepdims= True)
    pred_classes = np.argmax(mrcnn_class, axis = -1)
    pred_deltas  = mrcnn_bbox[bbox_idx, pred_classes]
    logt('* pred_scores', pred_scores, verbose = verbose)
    logt('* pred_classes', pred_classes, verbose = verbose)
    logt('* pred_deltas', pred_deltas, verbose = verbose)
    
    ##------------------------------------------------------------------------------------
    ## apply delta refinements to the rois,  based on deltas provided by the mrcnn head 
    ##------------------------------------------------------------------------------------
    pred_deltas  *= config.BBOX_STD_DEV
    output_rois   = norm_output_rois * scale
    refined_rois  = utils.apply_box_deltas_np(output_rois, pred_deltas)
    
    ##   Clip refined boxes to image window        
    clip_window   = np.array([0,0,h,w], dtype = np.float32)
    refined_rois  = utils.clip_to_window_np(clip_window, refined_rois)
    logt('clip widow', clip_window, verbose = verbose)
    logt('refined rois',refined_rois, verbose = verbose)
    logt('refined rois ', refined_rois, verbose = verbose)
    
    ##------------------------------------------------------------------------------------
    ## Remove RoIs that have an area of zero 
    ## 31-01-2019 Added to prevent NaN that occur when delta refinement causes the 
    ## boxes coordiantes to generate a zero area bbox. 
    ##------------------------------------------------------------------------------------
    roi_area           = (refined_rois[...,3] - refined_rois[...,1])*(refined_rois[...,2] - refined_rois[...,0]) > 0
    roi_area_condition = (refined_rois[...,3] - refined_rois[...,1])*(refined_rois[...,2] - refined_rois[...,0]) > 0
    nonzero_idx        = np.where(roi_area > 0)


    clean_rois    = refined_rois[roi_area_condition]
    clean_scores  = pred_scores[roi_area_condition]  
    clean_classes = pred_classes[roi_area_condition]  
    class_counts  = np.bincount(clean_classes, minlength=config.NUM_CLASSES)
    first_neg_example = np.count_nonzero(target_class_ids)
    
    if verbose:
        logt(' roi_area ', roi_area, verbose = verbose)
        logt(' roi_area_condition ', roi_area_condition, verbose = verbose)
        logt(' nonzero_idx   ', nonzero_idx, verbose = verbose)
        logt(' clean_rois    ', clean_rois, verbose = verbose)
        logt(' clean_scores  ', clean_scores, verbose = verbose)
        logt(' clean_classes ', clean_classes, verbose = verbose)
        logt(' class_counts  ', class_counts, verbose = verbose)
        logt(' clean_classes ', clean_classes, verbose = verbose)
        logt(' target_class_ids', target_class_ids, verbose = verbose)
        logt(' first_neg_example', first_neg_example, verbose = verbose)

    clean_classes_exp = np.expand_dims(clean_classes, -1)
    
    pos_array  = np.concatenate([ clean_rois       [:first_neg_example, :], 
                                  clean_classes_exp[:first_neg_example, :] ,
                                  clean_scores     [:first_neg_example, :] ,
                                  np.ones_like(clean_scores[:first_neg_example])], axis = -1)
    
    neg_array  = np.concatenate([ clean_rois       [first_neg_example:, :], 
                                  clean_classes_exp[first_neg_example:, :],
                                  clean_scores     [first_neg_example:, :],
                                  np.ones_like(clean_scores[first_neg_example:]) * -1 ], axis = -1)
    
    pos_class_counts  = np.bincount(clean_classes[:first_neg_example], minlength=config.NUM_CLASSES)
    ttl_pos_examples  = pos_array.shape[0]
    ttl_neg_examples  = neg_array.shape[0]
    
    
    ## determine how many false examples should be assigned to each class
    assign_neg_to_classes = ((pos_class_counts * ttl_neg_examples) /  ttl_pos_examples)
    assign_negs = np.rint(assign_neg_to_classes).astype(np.int)

    if verbose:
        print('pos_class_counts: ', pos_class_counts, 'ttl ', ttl_pos_examples)
    #     print(pos_array)
        print(' neg_array ttl :', ttl_neg_examples)
    #     print(neg_array)
        print(' pos_class_counts: ', pos_class_counts, 'ttl ', ttl_pos_examples)
        print(assign_negs.sum() == ttl_neg_examples, ' ttl_neg_examples: ', ttl_neg_examples,'  assign_negs     : ', assign_negs, 'sum : ', assign_negs.sum())   

    ## Calculate IoUs between Positive and Negative examples 
    overlaps = utils.compute_overlaps(neg_array[:,:CLASS_COLUMN], pos_array[:, :CLASS_COLUMN])
    overlaps_argsorts = np.argsort(overlaps, axis = 1)

    #     print(' overalps: ',overlaps.shape)
    #     print(overlaps)
    #     print('arg_max: ',np.argmax(overlaps, axis = 1))
    #     print('arg_min: ',np.argmin(overlaps, axis = 1))
    #     print('argsort \n', overlaps_argsorts)
    
    assign_neg_bin_count = copy.deepcopy(assign_negs)
    negs_assigned = 0
    negs_not_assigned = assign_negs.sum()
    neg_idx = 0
    pos_indexes = np.arange(ttl_pos_examples)
    
    while neg_idx < ttl_neg_examples and negs_not_assigned:
        for pos_idx in pos_indexes:
            min_idx = overlaps_argsorts[neg_idx, pos_idx]
            min_cls_id = clean_classes[min_idx]
#             print('neg_idx {} pos_idx: {} min_idx: {} min_cls_id: {}  overlap: {}'.format(neg_idx, pos_idx, min_idx, min_cls_id, overlaps[neg_idx, min_idx]))
            if assign_neg_bin_count[min_cls_id] > 0:
#                 print('found non zero bin count for class', min_cls_id, ' bin is ', assign_neg_bin_count)
                assign_neg_bin_count[min_cls_id] -= 1
                negs_not_assigned  -= 1
                break
#         print('      Assign neg_array[{}] to class {}  bin is {} '.format(neg_idx, min_cls_id, assign_neg_bin_count))
        neg_array[neg_idx, CLASS_COLUMN] = min_cls_id
        negs_assigned += 1
        neg_idx += 1
        
    #     print('While loop finisehd - negs_not_assigned :', negs_not_assigned, 'negs_assigned: ', negs_assigned)
    #     print()
    #     print('pos_class_counts: ', pos_class_counts, 'ttl ', ttl_pos_examples)
    #     print(pos_array)    
    #     print(' neg_array ttl :', ttl_neg_examples)
    #     print(neg_array)

    pos_neg_array = np.vstack([pos_array, neg_array])
    
    ## Pad resulting array as needed ......
    gap = config.TRAIN_ROIS_PER_IMAGE - pos_neg_array.shape[0]
    assert gap >= 0
    if gap > 0:
        pos_neg_array = np.pad(pos_neg_array, [(0, gap), (0, 0)], 'constant', constant_values=0)
 
    ## Add a sequence column to assist with sorting and keeing the order 
    sequence = np.expand_dims(np.arange(config.TRAIN_ROIS_PER_IMAGE,0,-1),-1)
    pos_neg_array = np.hstack([pos_neg_array, sequence])
    #     print(' pos_neg_array ttl :', pos_neg_array.shape)
    #     print(pos_neg_array)
      
    ## Equivalent of scatter_nd
    class_idx    = pos_neg_array[:,CLASS_COLUMN].astype(np.int32)
    pred_scatter = np.zeros((num_classes, num_rois1, pos_neg_array.shape[-1]))
    pred_scatter[class_idx, bbox_idx] = pos_neg_array[bbox_idx]
    
    #     print('pred_scatter :', pred_scatter.shape)
    #     for i in range(pred_scatter.shape[0]):
    #         print('pred_scatter[',i,']')
    #         print(pred_scatter[i])

    ## Normalization 
    normalizer   = np.amax(pred_scatter[:,:,SCORE_COLUMN], axis = -1, keepdims=True)
    normalizer   = np.where(normalizer < 1.0e-8,  1.00, normalizer)    
    norm_score   = pred_scatter[...,SCORE_COLUMN]/normalizer 
    pred_scatter = np.dstack([pred_scatter, norm_score])

    #     print(' normalizer: ',normalizer.shape)
    #     print(normalizer)
    #     print(' norm_score: ',norm_score.shape)
    #     print(norm_score)
    #     print(' pred_scatter ', pred_scatter.shape)
              

    ## sort by descending sequence id in each class 
    sort_inds = np.argsort(-pred_scatter[:,:, SEQUENCE_COLUMN])
    pred_tensor = np.zeros_like(pred_scatter)
    cls_ix, box_ix = np.indices(pred_tensor.shape[:2])
    
    #     print(sort_inds.shape)
    #     print(sort_inds)
    #     print(' pred_tensor: ', pred_tensor.shape)
    #     print('cls_ix', cls_ix.shape)
    #     print(cls_ix)
    #     print('box_ix', box_ix.shape)
    #     print(box_ix)
    #     print('sort_inds[cls_ix, box_ix]')
    #     print(sort_inds[cls_ix, box_ix])
    
    pred_tensor[cls_ix, box_ix] = pred_scatter[cls_ix, sort_inds[cls_ix, box_ix] ]    
    #     for i in range(pred_tensor.shape[0]):
    #         print('pred_tensor[',i,']')
    #         print(pred_tensor[i])
            
             
    #     logt('sort_inds      ', sort_inds  , verbose = verbose)
    #     logt('class_ix       ', cls_ix , verbose = verbose)
    #     logt('bbox_ix        ', box_ix , verbose = verbose)
    #     logt('pred_tensor    ', pred_tensor, verbose = verbose)        
    return pred_tensor    
    
    
    
    
##------------------------------------------------------------------------------------------------------------
##
##------------------------------------------------------------------------------------------------------------     
class CHMLayer(KE.Layer):
    '''
    Contextual Heatmap Layer - Training Mode
    Receives the bboxes, their repsective classification and roi_outputs and 
    builds the per_class tensor

    Returns:
    -------
    Returns the following tensors:

    pred_tensor :       [batch, NUM_CLASSES, TRAIN_ROIS_PER_IMAGE, 
                                             (index, class_prob, y1, x1, y2, x2, class_id, old_idx)]
                                in normalized coordinates   
    pred_cls_cnt:       [batch, NUM_CLASSES] 
    
    gt_tensor:          [batch, NUM_CLASSES, DETECTION_MAX_INSTANCES, 
                                                (index, class_prob, y1, x1, y2, x2, class_id, old_idx)]
    gt_cls_cnt:         [batch, NUM_CLASSES]
    
    Note: Returned arrays might be zero padded if not enough target ROIs.
    
    '''

    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        print('-----------------------------------------')
        print('>>>  CHM Layer (Prediction Generation)   ')
        print('-----------------------------------------')
        self.config = config

    def build_pr_tensor_wrapper(self, norm_output_rois, mrcnn_class, mrcnn_bbox, target_class_ids):
        pr_tensor_batch = []
        for b in range(self.config.BATCH_SIZE):

            pr_tensor = build_pr_tensor_np(norm_output_rois[b], mrcnn_class[b], mrcnn_bbox[b], target_class_ids[b], self.config, verbose = 0)
            pr_tensor_batch.append(pr_tensor)

        pr_tensor_batch = np.array(pr_tensor_batch).astype(np.float32)
        return pr_tensor_batch
         
        
    def call(self, inputs):
        verbose = self.config.VERBOSE
        mrcnn_class , mrcnn_bbox,  output_rois, target_class_ids  = inputs

        logt('> CHMLayer Call() ', inputs)
        logt('  mrcnn_class',  mrcnn_class, verbose = verbose)
        logt('  mrcnn_bbox ',  mrcnn_bbox , verbose = verbose) 
        logt('  output_rois',  output_rois, verbose = verbose) 
        logt('  target_class_ids ',  target_class_ids, verbose = verbose)          
        
        # pr_tensor = build_pr_tensor(output_rois, mrcnn_class, mrcnn_bbox, self.config)
        
        pr_tensor = tf.py_func(self.build_pr_tensor_wrapper, [output_rois, mrcnn_class, mrcnn_bbox, target_class_ids], tf.float32, name="pr_tensor")
        pr_tensor.set_shape([None, self.config.NUM_CLASSES, self.config.TRAIN_ROIS_PER_IMAGE, 9])
        logt(' pr tensor' , pr_tensor, verbose = verbose)
        pr_hm, pr_hm_scores  = build_pr_heatmap(pr_tensor, self.config, names = ['pred_heatmap'])
        # pred_cls_cnt = KL.Lambda(lambda x: tf.count_nonzero(x[:,:,:,-1],axis = -1), name = 'pred_cls_count')(pred_tensor)        

        logt(' ', verbose = verbose) 
        logt('\npred_refined_heatmap      ', pr_hm, verbose = verbose) 
        logt('pred_refnined_heatmap_scores', pr_hm_scores, verbose = verbose) 
        logt('complete', verbose = verbose)
        
        return [  pr_hm, pr_hm_scores, pr_tensor]

    def compute_output_shape(self, input_shape):
        logt('> CHMLayer compute_output_shape() ', verbose = self.config.VERBOSE)
        # may need to change dimensions of first return from IMAGE_SHAPE to MAX_DIM
        if self.config.VERBOSE:
            for i,j in enumerate(input_shape):
                print('input: ', i, '     ',type(j), j)
        return [
                  (None, self.config.IMAGE_SHAPE[0], self.config.IMAGE_SHAPE[1], self.config.NUM_CLASSES)  # pred_heatmap_norm
                , (None, self.config.NUM_CLASSES , self.config.TRAIN_ROIS_PER_IMAGE , 24)                  # pred_heatmap_scores 
                , (None, self.config.NUM_CLASSES, self.config.TRAIN_ROIS_PER_IMAGE, 9)
               ]


