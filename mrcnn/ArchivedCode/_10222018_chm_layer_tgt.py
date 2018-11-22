"""
Mask R-CNN
Contextual Heatmap Layer for Training Mode - Ground Truth Generation

Copyright (c) 2018 K.Bardool 
Licensed under the MIT License (see LICENSE for details)
Written by Kevin Bardool
"""

import os, sys, glob, random, math, datetime, itertools, json, re, logging
import numpy as np
from scipy.stats import  multivariate_normal
import tensorflow as tf
# import keras
import keras.backend as KB
import keras.layers as KL
import keras.engine as KE
sys.path.append('..')
import mrcnn.utils as utils
import tensorflow.contrib.util as tfc
import pprint

##-----------------------------------------------------------------------------------------------------------
## 
##-----------------------------------------------------------------------------------------------------------
def build_gt_tensor(gt_class_ids, norm_gt_bboxes, config):

    batch_size      = config.BATCH_SIZE
    num_classes     = config.NUM_CLASSES
    h, w            = config.IMAGE_SHAPE[:2]
    num_bboxes      = KB.int_shape(norm_gt_bboxes)[1]

    scale           = tf.constant([h,w,h,w], dtype = tf.float32)
    # dup_scale       = tf.reshape(tf.tile(scale, [num_rois]),[num_rois,-1])
    dup_scale       = scale * tf.ones([batch_size, num_bboxes, 1], dtype = 'float32')
    gt_bboxes       = tf.multiply(norm_gt_bboxes , dup_scale )
    det_per_class    = config.DETECTION_PER_CLASS
 
    # num of bounding boxes is determined by bbox_list.shape[1] instead of config.DETECTION_MAX_INSTANCES
    # use of this routine for both input_gt_boxes, and target_gt_deltas
    if  num_bboxes == config.DETECTION_MAX_INSTANCES:
        tensor_name = "gt_tensor_max"
    else:
        tensor_name = "gt_tensor"
        
    print('\n')
    print('  > BUILD_GROUND TRUTH_TF()' )        
    print('    num_bboxes             : ', num_bboxes, '(building ', tensor_name , ')' )    
    print('    gt_class_ids shape     : ', gt_class_ids.get_shape(), '  ', KB.int_shape(gt_class_ids))
    print('    norm_gt_bboxes.shape   : ', norm_gt_bboxes.get_shape()   , '  ', KB.int_shape(norm_gt_bboxes))
    print('    gt_bboxes.shape        : ', gt_bboxes.get_shape()   , '  ', KB.int_shape(gt_bboxes))
        
    #---------------------------------------------------------------------------
    # use the argmaxof each row to determine the dominating (predicted) class
    # mask identifies class_ids > 0 
    #---------------------------------------------------------------------------
    gt_classes_exp = tf.to_float(tf.expand_dims(gt_class_ids ,axis=-1))
    print('    gt_classes_exp         : ', gt_classes_exp.get_shape() )

    ones = tf.ones_like(gt_class_ids)
    zeros= tf.zeros_like(gt_class_ids)
    mask = tf.greater(gt_class_ids , 0)

    gt_scores     = tf.where(mask, ones, zeros)
    # pred_scores      = tf.reduce_max(mrcnn_class ,axis=-1, keep_dims=True)   # (32,)
    gt_scores_exp = tf.to_float(KB.expand_dims(gt_scores, axis=-1))
    print('    gt_scores_exp          : ', gt_scores_exp.get_shape())

    ##------------------------------------------------------------------------------------
    ## Generate GT_ARRAY
    ##    Note that we add gt_scores_exp also at the end, to match the the dimensions of  
    ##    pred_tensor generated in build_predictions (corresponds to the normalized score)
    ##
    ##    sequence id is used to preserve the order of rois as passed to this routine
    ##------------------------------------------------------------------------------------
    batch_grid, bbox_grid = tf.meshgrid( tf.range(batch_size, dtype=tf.int32), 
                                         tf.range(num_bboxes, dtype=tf.int32), indexing = 'ij' )
    
    sequence = gt_scores * (bbox_grid[...,::-1] + 1) 
    sequence = tf.to_float(tf.expand_dims(sequence, axis = -1))   
    gt_array = tf.concat([gt_bboxes, gt_classes_exp, gt_scores_exp, sequence, gt_scores_exp ], 
                         axis=-1, name = 'gt_array')

    # print('    batch_grid shape  ', batch_grid.get_shape())
    # print('    bbox_grid  shape  ', bbox_grid.get_shape())
    # print('    sequence shape    ', sequence.get_shape())

    ##------------------------------------------------------------------------------
    ## Create indicies to scatter rois out to multi-dim tensor by image id and class
    ## resulting tensor is batch size x num_classes x num_bboxes x 7 (num columns)
    ##------------------------------------------------------------------------------
    scatter_ind = tf.stack([batch_grid , gt_class_ids, bbox_grid],axis = -1)
    gt_scatter = tf.scatter_nd(scatter_ind, gt_array, [batch_size, num_classes, num_bboxes, gt_array.shape[-1] ])
    
    print('    gt_array shape         : ', gt_array.shape   , gt_array.get_shape())
    print('    scatter_ind shape      : ', scatter_ind.shape, scatter_ind.get_shape())
    print('    tf.shape(gt_array)[-1] : ', gt_array.shape[-1], KB.int_shape(gt_array))
    print('    gt_scatter shape       : ', gt_scatter.shape , gt_scatter.get_shape())
    
    ##-------------------------------------------------------------------------------
    ## sort in each class dimension based on on sequence number (column 6)
    ##     scatter_nd places bboxs in a sparse fashion --- this sort is to place all bboxes
    ## at the top of the class bbox array
    ##-------------------------------------------------------------------------------
    _ , sort_inds = tf.nn.top_k(tf.abs(gt_scatter[:,:,:,6]), k=gt_scatter.shape[2])

    # build indexes to gather rows from pred_scatter based on sort order 
    class_grid, batch_grid, bbox_grid = tf.meshgrid(tf.range(num_classes),tf.range(batch_size), tf.range(num_bboxes))
    bbox_grid_exp = tf.to_float(tf.expand_dims(bbox_grid, axis = -1))
 
    gather_inds = tf.stack([batch_grid , class_grid, sort_inds],axis = -1)
    gt_tensor   = tf.gather_nd(gt_scatter, gather_inds[...,:det_per_class,:] , name = tensor_name)
    # append an index to the end of each row --- commented out 30-04-2018
    # gt_tensor = tf.concat([gt_tensor, bbox_grid_exp], axis = -1)
    print('    sort_inds              : ', type(sort_inds)   , ' shape ', sort_inds.shape)
    print('    class_grid             : ', type(class_grid)  , ' shape ', class_grid.get_shape())
    print('    batch_grid             : ', type(batch_grid)  , ' shape ', batch_grid.get_shape())
    print('    gather_inds            : ', gather_inds.get_shape())
    print('    gt_tensor.shape        : ', KB.int_shape(gt_tensor), gt_tensor.get_shape())

    return  gt_tensor 

##-----------------------------------------------------------------------------------------------------------
##  build_gt_heatmap : Build Ground Truth heatmaps using pred_gt_tensor
##------------------------------------------------------------------------------------------------------------
##  v2: in this version, 
##      For heatmap generation, prob_grid is passed through "clip_heatmap" which clips the gaussian distribution
##      based on Cx, Cy and the Covar parms for each bounding box. This prob_grid_clipped is then passed on to 
##      next steps (per-class normalization, 
##      "build_hm_score" function which calculates scores is applied to "prob_grid", 
##       
## Inputs:
##      in_tensor - [BATCH_SIZE, NUM_CLASSES, DETECTIONS_PER_CLASS, 8]
##                  per-class tensor of bounding boxes (Y1,X1,Y2,X2), class_ids, mrcnn_predicted scores, 
##                  sequence_id, and per-class normalized scores       
##      config    - Model configuration object
##
## Outputs:
##
##   gauss_heatmap_norm
##   gauss_scores - [BATCH_SIZE, NUM_CLASSES, DETECTIONS_PER_CLASS, 11]
##                  same as in_tensor, adding :
##  TODO  update these for Ground_Truth function 
##                  - sum of heatmap in masked area
##                  - area of bounding box in pixes
##                  - (sum of heatmap in masked area) * (bbox per-class normalized score from in_tensor)
##------------------------------------------------------------------------------------------------------------
def build_gt_heatmap(in_tensor, config, names = None):
  
    num_detections  = config.DETECTION_MAX_INSTANCES
    img_h, img_w    = config.IMAGE_SHAPE[:2]
    batch_size      = config.BATCH_SIZE
    num_classes     = config.NUM_CLASSES  
    heatmap_scale   = config.HEATMAP_SCALE_FACTOR
    grid_h, grid_w  = config.IMAGE_SHAPE[:2] // heatmap_scale    
    # rois per image is determined by size of input tensor 
    #   detection mode:   config.TRAIN_ROIS_PER_IMAGE 
    #   ground_truth  :   config.DETECTION_MAX_INSTANCES
    #   strt_cls        = 0 if rois_per_image == 32 else 1
    # rois_per_image  = config.DETECTION_PER_CLASS
    rois_per_image  = (in_tensor.shape)[2]  

    print('\n ')
    print('  > build_heatmap() for ', names )
    print('    in_tensor shape        : ', in_tensor.shape)       
    print('    num bboxes per class   : ', rois_per_image )
    print('    heatmap scale        : ', heatmap_scale, 'Dimensions:  w:', grid_w,' h:', grid_h)
    
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

    print('    pt2_sum shape  : ', pt2_sum.shape)
    print('    pt2_ind shape  : ', pt2_ind.shape)
    print('    pt2_dense shape: ', pt2_dense.get_shape())

    ##-----------------------------------------------------------------------------
    ## Build mesh-grid to hold pixel coordinates  
    ##-----------------------------------------------------------------------------
    # X = tf.range(grid_w, dtype=tf.int32)
    # Y = tf.range(grid_h, dtype=tf.int32)
    # X, Y = tf.meshgrid(X, Y)

    # duplicate (repeat) X and Y into a  batch_size x rois_per_image tensor
    # print('    X/Y shapes :',  X.get_shape(), Y.get_shape())
    # ones = tf.ones([tf.shape(pt2_dense)[0] , 1, 1], dtype = tf.int32)
    # rep_X = ones * X
    # rep_Y = ones * Y 
    # print('    Ones:       ', ones.shape)                
    # print('    ones_exp * X', ones.shape, '*', X.shape, '= ',rep_X.shape)
    # print('    ones_exp * Y', ones.shape, '*', Y.shape, '= ',rep_Y.shape)

    # # stack the X and Y grids 
    # pos_grid = tf.to_float(tf.stack([rep_X,rep_Y], axis = -1))
    # print('    pos_grid before transpose : ', pos_grid.get_shape())
    # pos_grid = tf.transpose(pos_grid,[1,2,0,3])
    # print('    pos_grid after  transpose : ', pos_grid.get_shape())    

    ##-----------------------------------------------------------------------------
    ##  Build mean and convariance tensors for Multivariate Normal Distribution 
    ##-----------------------------------------------------------------------------
    pt2_dense_scaled = pt2_dense[:,:4]/heatmap_scale
    width  = pt2_dense_scaled[:,3] - pt2_dense_scaled[:,1]      # x2 - x1
    height = pt2_dense_scaled[:,2] - pt2_dense_scaled[:,0]
    cx     = pt2_dense_scaled[:,1] + ( width  / 2.0)
    cy     = pt2_dense_scaled[:,0] + ( height / 2.0)
    means  = tf.stack((cx,cy),axis = -1)
    covar  = tf.stack((width * 0.5 , height * 0.5), axis = -1)
    covar  = tf.sqrt(covar)

    ##-----------------------------------------------------------------------------
    ##  Compute Normal Distribution for bounding boxes
    ##-----------------------------------------------------------------------------    
    prob_grid = tf.ones([tf.shape(pt2_dense)[0] , grid_h, grid_w], dtype = tf.float32)
    print('     Prob_grid shape : ', prob_grid.shape)    
    
    # tfd = tf.contrib.distributions
    # mvn = tfd.MultivariateNormalDiag(loc = means,  scale_diag = covar)
    # prob_grid = mvn.prob(pos_grid)
    # print('    >> input to MVN.PROB: pos_grid (meshgrid) shape: ', pos_grid.shape)
    # print('     box_dims: ', box_dims.shape)
    # print('     Prob_grid shape from mvn.probe: ', prob_grid.shape)
    # prob_grid = tf.transpose(prob_grid,[2,0,1])
    # print('     Prob_grid shape after tanspose: ', prob_grid.shape)    
    # print('    << output probabilities shape  : ', prob_grid.shape)

    #--------------------------------------------------------------------------------
    # Kill distributions of NaN boxes (resulting from bboxes with height/width of zero
    # which cause singular sigma cov matrices
    #--------------------------------------------------------------------------------
    # prob_grid = tf.where(tf.is_nan(prob_grid),  tf.zeros_like(prob_grid), prob_grid)

    #---------------------------------------------------------------------------------------------
    # (1) apply normalization per bbox heatmap instance
    #---------------------------------------------------------------------------------------------
    # print('\n    normalization ------------------------------------------------------')   
    # normalizer = tf.reduce_max(prob_grid, axis=[-2,-1], keepdims = True)
    # normalizer = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    # print('    normalizer     : ', normalizer.shape) 
    # prob_grid_norm = prob_grid / normalizer

    #---------------------------------------------------------------------------------------------
    # (2) multiply normalized heatmap by normalized score in in_tensor/ (pt2_dense column 7)
    #     broadcasting : https://stackoverflow.com/questions/49705831/automatic-broadcasting-in-tensorflow
    #---------------------------------------------------------------------------------------------    
    # prob_grid_norm_scaled = tf.transpose(tf.transpose(prob_grid_norm) * pt2_dense[:,7])
    # print('    prob_grid_norm_scaled : ', prob_grid_norm_scaled.shape)
    prob_grid_clipped = tf.map_fn(clip_gt_heatmap, [prob_grid, cy,cx, covar], 
                                 dtype = tf.float32, swap_memory = True)
                                 
                                 
    ##--------------------------------------------------------------------------------------------
    ##  Generate scores using prob_grid and pt2_dense - NEW METHOD
    ##  pt2_dense[:,7] is the per-class-normalized score from in_tensor
    ##  added 09-21-2018
    ##--------------------------------------------------------------------------------------------
    scores_from_sum2 = tf.map_fn(build_gt_hm_score, [prob_grid_clipped, pt2_dense_scaled, pt2_dense[:,7]], 
                                     dtype = tf.float32, swap_memory = True)
    scores_scattered = tf.scatter_nd(pt2_ind, scores_from_sum2, 
                                     [batch_size, num_classes, rois_per_image, 3], name = 'scores_scattered')
    gauss_scores = tf.concat([in_tensor, scores_scattered], axis = -1,name = names[0]+'_scores')
    print('    prob_grid_clipped      : ', prob_grid_clipped.shape) 
    print('    scores_scattered shape : ', scores_scattered.shape) 
    print('    gauss_scores           : ', gauss_scores.shape, ' Name:   ', gauss_scores.name)
    print('    gauss_scores  (FINAL)  : ', gauss_scores.shape, ' Keras tensor ', KB.is_keras_tensor(gauss_scores) )
        
    ##-------------------------------------------------------------------------------------
    ## (3) scatter out the probability distribution heatmaps based on class 
    ##-------------------------------------------------------------------------------------
    print('\n    Scatter out the probability distributions based on class --------------') 
    gauss_scatt   = tf.scatter_nd(pt2_ind, prob_grid_clipped, 
                                  [batch_size, num_classes, rois_per_image, grid_w, grid_h], 
                                  name = 'gauss_scatter')
    print('    pt2_ind shape   : ', pt2_ind.shape)  
    print('    prob_grid shape : ', prob_grid.shape)  
    print('    gauss_scatt     : ', gauss_scatt.shape)   # batch_sz , num_classes, num_rois, image_h, image_w

    ##-------------------------------------------------------------------------------------
    ## (4) MAX : Reduce_MAX up gauss_scattered heatmaps by class 
    ##           Since all values are set to '1' in the 'heatmap', there is no need to 
    ##           sum or normalize. We Reduce_max on the class axis, and as a result the 
    ##           correspoding areas in the heatmap are set to '1'
    ##-------------------------------------------------------------------------------------
    print('\n    Reduce MAX based on class ---------------------------------------------')         
    gauss_heatmap = tf.reduce_max(gauss_scatt, axis=2, name='pred_heatmap2')
    print('    gaussian_heatmap : ', gauss_heatmap.get_shape(), 'Keras tensor ', KB.is_keras_tensor(gauss_heatmap) )
    
    #---------------------------------------------------------------------------------------------
    # (5) heatmap normalization
    #     normalizer is set to one when the max of class is zero     
    #     this prevents elements of gauss_heatmap_norm computing to nan
    #---------------------------------------------------------------------------------------------
    # print('\n    normalization ------------------------------------------------------')   
    # normalizer = tf.reduce_max(gauss_heatmap, axis=[-2,-1], keepdims = True)
    # normalizer = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    # gauss_heatmap_norm = gauss_heatmap / normalizer
    # print('    normalizer shape : ', normalizer.shape)   
    # print('    gauss norm       : ', gauss_heatmap_norm.shape   ,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap_norm) )
    
    ##--------------------------------------------------------------------------------------------
    ##  Transpose tensor to [BatchSz, Height, Width, Num_Classes]
    ##--------------------------------------------------------------------------------------------
    gauss_heatmap  = tf.transpose(gauss_heatmap,[0,2,3,1], name = names[0])
    print('    gauss_heatmap : ', gauss_heatmap.shape,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap))
    
    # gauss_heatmap_norm = tf.transpose(gauss_heatmap_norm,[0,2,3,1], name = names[0]+'_norm')
    # print('    gauss_heatmap_norm : ', gauss_heatmap_norm.shape,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap_norm) )
    # print('    complete')

    return   gauss_heatmap, gauss_scores  

##-----------------------------------------------------------------------------------------------------------
## Build Mask and Score 
##----------------------------------------------------------------------------------------------------------- 
def build_gt_hm_score(input_list):
    '''
    Inputs:
    -----------
        heatmap_tensor :    [ image height, image width ]
        input_row      :    [y1, x1, y2, x2] in absolute (non-normalized) scale
        input_norm_score:   class-normalized score 
        
    Returns
    -----------
        gaussian_sum :      sum of gaussian heatmap vlaues over the area covered by the bounding box
        bbox_area    :      bounding box area (in pixels)
        weighted_sum :      gaussian_sum * bbox_score
    '''
    heatmap_tensor, input_bbox, input_norm_score = input_list
    
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

#         Multiply gaussian_sum by score to obtain weighted sum    
#         weighted_sum = gaussian_sum * input_row[5]

#       Replaced lines above with following lines 21-09-2018
        # Multiply gaussian_sum by normalized score to obtain weighted_norm_sum 
        weighted_norm_sum = gaussian_sum * input_norm_score    # input_list[7]

    return tf.stack([gaussian_sum, bbox_area, weighted_norm_sum], axis = -1)
    
    
##-----------------------------------------------------------------------------------------------------------
## Build Mask and Score - Using CY,CX, and COVAR to define masking area
##----------------------------------------------------------------------------------------------------------- 
def build_gt_hm_score_v2(input_list):
    '''
    Inputs:
    -----------
        heatmap_tensor :    [ image height, image width ]
        input_row      :    [y1, x1, y2, x2] in absolute (non-normalized) scale
        input_norm_score:   class-normalized score 
        
    Returns
    -----------
        gaussian_sum :      sum of gaussian heatmap vlaues over the area covered by the bounding box
        bbox_area    :      bounding box area (in pixels)
        weighted_sum :      gaussian_sum * bbox_score
    '''
    heatmap_tensor, input_bbox, input_norm_score = input_list
    
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

#         Multiply gaussian_sum by score to obtain weighted sum    
#         weighted_sum = gaussian_sum * input_row[5]

#       Replaced lines above with following lines 21-09-2018
        # Multiply gaussian_sum by normalized score to obtain weighted_norm_sum 
        weighted_norm_sum = gaussian_sum * input_norm_score    # input_list[7]

    return tf.stack([gaussian_sum, bbox_area, weighted_norm_sum], axis = -1)
    

##-----------------------------------------------------------------------------------------------------------
## Clip Heatmap
##      Clips heatmap to a predefined vicinity (+/- Covar_Y, Covar_X pixels of cy,cx)
##----------------------------------------------------------------------------------------------------------- 
def clip_gt_heatmap(input_list):
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
        
##------------------------------------------------------------------------------------------------------------
##
##------------------------------------------------------------------------------------------------------------     
class CHMLayerTarget(KE.Layer):
    '''
    Contextual Heatmap Layer - Training Mode
    Receives the bboxes, their repsective classification and roi_outputs and 
    builds the per_class tensor

    Returns:
    -------
    Returns the following tensors:

    gt_tensor:          [batch, NUM_CLASSES, DETECTION_MAX_INSTANCES, 
                                                (index, class_prob, y1, x1, y2, x2, class_id, old_idx)]
    gt_cls_cnt:         [batch, NUM_CLASSES]
    
    Note: Returned arrays might be zero padded if not enough target ROIs.
    
    '''

    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        print()
        print('-----------------------------------------')
        print('>>>  CHM Layer (Ground Truth Generation) ')
        print('-----------------------------------------')
        self.config = config

        
    def call(self, inputs):

        tgt_class_ids, tgt_bboxes = inputs
        print('  > CHMLayerTgt Call() ', len(inputs))
        print('    tgt_class_ids.shape  :', tgt_class_ids.shape, KB.int_shape(tgt_class_ids )) 
        print('    tgt_bboxes.shape     :',    tgt_bboxes.shape, KB.int_shape(   tgt_bboxes )) 
         
        gt_tensor     = build_gt_tensor (tgt_class_ids,  tgt_bboxes, self.config)  
        gt_hm_norm, gt_hm_scores  = build_gt_heatmap(gt_tensor, self.config, names = ['gt_heatmap'])
        # gt_cls_cnt   = KL.Lambda(lambda x: tf.count_nonzero(x[:,:,:,-1],axis = -1), name = 'gt_cls_count')(gt_tensor)

        print()
        print('    gt_heatmap                  : ', gt_hm_norm.shape   , 'Keras tensor ', KB.is_keras_tensor(gt_hm_norm))
        print('    gt_heatmap_scores           : ', gt_hm_scores.shape , 'Keras tensor ', KB.is_keras_tensor(gt_hm_scores))
        print('    complete')
        
        return [  gt_hm_norm, gt_hm_scores, gt_tensor]

         
    def compute_output_shape(self, input_shape):
        # may need to change dimensions of first return from IMAGE_SHAPE to MAX_DIM
        return [
                 (None, self.config.IMAGE_SHAPE[0], self.config.IMAGE_SHAPE[1], self.config.NUM_CLASSES)  # gt_heatmap_norm
              ,  (None, self.config.NUM_CLASSES   , self.config.DETECTION_PER_CLASS ,11)                  # gt_heatmap+scores   
              # ----extra stuff for now ---------------------------------------------------------------------------------------------------
              ,  (None, self.config.NUM_CLASSES   , self.config.DETECTION_PER_CLASS ,8)                  # gt_tensor               
              ]

              
'''    
##-----------------------------------------------------------------------------------------------------------
##  build_gt_heatmap : Build gaussian heatmaps using pred_gt_tensor
## 
##  INPUTS :
##
##------------------------------------------------------------------------------------------------------------
def build_gt_heatmap_original(in_tensor, config, names = None):
  
    num_detections  = config.DETECTION_MAX_INSTANCES
    img_h, img_w    = config.IMAGE_SHAPE[:2]
    batch_size      = config.BATCH_SIZE
    num_classes     = config.NUM_CLASSES  
    heatmap_scale   = config.HEATMAP_SCALE_FACTOR
    grid_h, grid_w  = config.IMAGE_SHAPE[:2] // heatmap_scale    
    # rois per image is determined by size of input tensor 
    #   detection mode:   config.TRAIN_ROIS_PER_IMAGE 
    #   ground_truth  :   config.DETECTION_MAX_INSTANCES
    #   strt_cls        = 0 if rois_per_image == 32 else 1
    # rois_per_image  = config.DETECTION_PER_CLASS
    rois_per_image  = (in_tensor.shape)[2]  

    print('\n ')
    print('  > build_heatmap() for ', names )
    print('    in_tensor shape        : ', in_tensor.shape)       
    print('    num bboxes per class   : ', rois_per_image )
    print('    heatmap scale        : ', heatmap_scale, 'Dimensions:  w:', grid_w,' h:', grid_h)
    
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

    print('    pt2_sum shape  : ', pt2_sum.shape)
    print('    pt2_ind shape  : ', pt2_ind.shape)
    print('    pt2_dense shape: ', pt2_dense.get_shape())

    ##-----------------------------------------------------------------------------
    ## Build mesh-grid to hold pixel coordinates  
    ##-----------------------------------------------------------------------------
    X = tf.range(grid_w, dtype=tf.int32)
    Y = tf.range(grid_h, dtype=tf.int32)
    X, Y = tf.meshgrid(X, Y)

    # duplicate (repeat) X and Y into a  batch_size x rois_per_image tensor
    print('    X/Y shapes :',  X.get_shape(), Y.get_shape())
    ones = tf.ones([tf.shape(pt2_dense)[0] , 1, 1], dtype = tf.int32)
    rep_X = ones * X
    rep_Y = ones * Y 
    print('    Ones:    ', ones.shape)                
    print('    ones_exp * X', ones.shape, '*', X.shape, '= ',rep_X.shape)
    print('    ones_exp * Y', ones.shape, '*', Y.shape, '= ',rep_Y.shape)

    # # stack the X and Y grids 
    pos_grid = tf.to_float(tf.stack([rep_X,rep_Y], axis = -1))
    print('    pos_grid before transpse : ', pos_grid.get_shape())
    pos_grid = tf.transpose(pos_grid,[1,2,0,3])
    print('    pos_grid after transpose : ', pos_grid.get_shape())    

    ##-----------------------------------------------------------------------------
    ##  Build mean and convariance tensors for Multivariate Normal Distribution 
    ##-----------------------------------------------------------------------------
    pt2_dense_scaled = pt2_dense[:,:4]/heatmap_scale
    width  = pt2_dense_scaled[:,3] - pt2_dense_scaled[:,1]      # x2 - x1
    height = pt2_dense_scaled[:,2] - pt2_dense_scaled[:,0]
    cx     = pt2_dense_scaled[:,1] + ( width  / 2.0)
    cy     = pt2_dense_scaled[:,0] + ( height / 2.0)
    means  = tf.stack((cx,cy),axis = -1)
    covar  = tf.stack((width * 0.5 , height * 0.5), axis = -1)
    covar  = tf.sqrt(covar)

    ##-----------------------------------------------------------------------------
    ##  Compute Normal Distribution for bounding boxes
    ##-----------------------------------------------------------------------------    
    tfd = tf.contrib.distributions
    mvn = tfd.MultivariateNormalDiag(loc = means,  scale_diag = covar)
    prob_grid = mvn.prob(pos_grid)
    print('    >> input to MVN.PROB: pos_grid (meshgrid) shape: ', pos_grid.shape)
    print('     Prob_grid shape from mvn.probe: ',prob_grid.shape)
    prob_grid = tf.transpose(prob_grid,[2,0,1])
    print('     Prob_grid shape after tanspose: ',prob_grid.shape)    
    print('    << output probabilities shape  : ' , prob_grid.shape)

    #--------------------------------------------------------------------------------
    # Kill distributions of NaN boxes (resulting from bboxes with height/width of zero
    # which cause singular sigma cov matrices
    #--------------------------------------------------------------------------------
    # prob_grid = tf.where(tf.is_nan(prob_grid),  tf.zeros_like(prob_grid), prob_grid)

    ##---------------------------------------------------------------------------------------------
    ## (1) apply normalization per bbox heatmap instance
    ##---------------------------------------------------------------------------------------------
    print('\n    normalization ------------------------------------------------------')   
    normalizer = tf.reduce_max(prob_grid, axis=[-2,-1], keepdims = True)
    normalizer = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    print('    normalizer     : ', normalizer.shape) 
    prob_grid_norm = prob_grid / normalizer

    ##---------------------------------------------------------------------------------------------
    ## (2) multiply normalized heatmap by normalized score in in_tensor/ (pt2_dense column 7)
    ##     broadcasting : https://stackoverflow.com/questions/49705831/automatic-broadcasting-in-tensorflow
    ##---------------------------------------------------------------------------------------------    
    #  Using the double tf.transpose, we dont need this any more    
    #     scr = tf.expand_dims(tf.expand_dims(pt2_dense[:,7],axis = -1), axis =-1)

    prob_grid_norm_scaled = tf.transpose(tf.transpose(prob_grid_norm) * pt2_dense[:,7])
    print('    prob_grid_norm_scaled : ', prob_grid_norm_scaled.shape)
    #     maxes2 = tf.reduce_max(prob_grid_norm_scaled, axis=[-2,-1], keepdims = True)
    #     print('    shape of maxes2       : ', maxes2.shape)

    ##-------------------------------------------------------------------------------------
    ## (3) scatter out the probability distributions based on class 
    ##-------------------------------------------------------------------------------------
    print('\n    Scatter out the probability distributions based on class --------------') 
    gauss_scatt   = tf.scatter_nd(pt2_ind, prob_grid_norm_scaled, 
                                  [batch_size, num_classes, rois_per_image, grid_w, grid_h], 
                                  name = 'gauss_scatter')
    print('    pt2_ind shape   : ', pt2_ind.shape)  
    print('    prob_grid shape : ', prob_grid.shape)  
    print('    gauss_scatt     : ', gauss_scatt.shape)   # batch_sz , num_classes, num_rois, image_h, image_w

    ##-------------------------------------------------------------------------------------
    ## (4) SUM : Reduce and sum up gauss_scattered by class  
    ##-------------------------------------------------------------------------------------
    print('\n    Reduce sum based on class ---------------------------------------------')         
    gauss_heatmap = tf.reduce_sum(gauss_scatt, axis=2, name='pred_heatmap2')
    #--------------------------------------------------------------------------------------
    # force small sums to zero - for now (09-11-18) commented out but could reintroduce based on test results
    # gauss_heatmap = tf.where(gauss_heatmap < 1e-12, gauss_heatmap, tf.zeros_like(gauss_heatmap), name='Where1')
    #--------------------------------------------------------------------------------------
    print('    gaussian_heatmap : ', gauss_heatmap.get_shape(), 'Keras tensor ', KB.is_keras_tensor(gauss_heatmap) )      
    
    ##---------------------------------------------------------------------------------------------
    ## (5) heatmap normalization
    ##     normalizer is set to one when the max of class is zero     
    ##     this prevents elements of gauss_heatmap_norm computing to nan
    ##---------------------------------------------------------------------------------------------
    print('\n    normalization ------------------------------------------------------')   
    normalizer = tf.reduce_max(gauss_heatmap, axis=[-2,-1], keepdims = True)
    normalizer = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    gauss_heatmap_norm = gauss_heatmap / normalizer
    print('    normalizer shape : ', normalizer.shape)   
    print('    gauss norm       : ', gauss_heatmap_norm.shape   ,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap_norm) )
    
    ##--------------------------------------------------------------------------------------------
    ##  Generate scores using prob_grid and pt2_dense - NEW METHOD
    ##  added 09-21-2018
    ##--------------------------------------------------------------------------------------------
    scores_from_sum2 = tf.map_fn(build_gt_hm_score_original,
                                 [prob_grid, pt2_dense_scaled, pt2_dense[:,7]], 
                                 dtype = tf.float32, swap_memory = True)
    scores_scattered = tf.scatter_nd(pt2_ind, scores_from_sum2, 
                                     [batch_size, num_classes, rois_per_image, 3], name = 'scores_scattered')
    gauss_scores = tf.concat([in_tensor, scores_scattered], axis = -1,name = names[0]+'_scores')
    print('    scores_scattered shape : ', scores_scattered.shape) 
    print('    gauss_scores           : ', gauss_scores.shape, ' Name:   ', gauss_scores.name)
    print('    gauss_scores  (FINAL)  : ', gauss_scores.shape, ' Keras tensor ', KB.is_keras_tensor(gauss_scores) )      
    
    
    ##--------------------------------------------------------------------------------------------
    ## //create heatmap Append `in_tensor` and `scores_from_sum` to form `bbox_scores`
    ##--------------------------------------------------------------------------------------------
    # gauss_heatmap  = tf.transpose(gauss_heatmap,[0,2,3,1], name = names[0])
    # print('    gauss_heatmap : ', gauss_heatmap.shape,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap))
    
    gauss_heatmap_norm = tf.transpose(gauss_heatmap_norm,[0,2,3,1], name = names[0]+'_norm')
    print('    gauss_heatmap_norm : ', gauss_heatmap_norm.shape,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap_norm) )
    print('    complete')

    return   gauss_heatmap_norm, gauss_scores  
    # , gauss_heatmap   gauss_heatmap_L2norm    # [gauss_heatmap, gauss_scatt, means, covar]    

    
##-----------------------------------------------------------------------------------------------------------
## Build Mask and Score 
##----------------------------------------------------------------------------------------------------------- 
def build_gt_hm_score_original(input_list):
    """
    Inputs:
    -----------
        heatmap_tensor :    [ image height, image width ]
        input_row      :    [y1, x1, y2, x2] in absolute (non-normalized) scale
        input_norm_score:   class-normalized score 
        
    Returns
    -----------
        gaussian_sum :      sum of gaussian heatmap vlaues over the area covered by the bounding box
        bbox_area    :      bounding box area (in pixels)
        weighted_sum :      gaussian_sum * bbox_score
    """
    heatmap_tensor, input_bbox, input_norm_score = input_list
    
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
        mask_applied = tf.multiply(heatmap_tensor, mask, name = 'mask_applied')
        bbox_area    = tf.to_float((input_bbox[2]-input_bbox[0]) * (input_bbox[3]-input_bbox[1]))
        gaussian_sum = tf.reduce_sum(mask_applied)

#         Multiply gaussian_sum by score to obtain weighted sum    
#         weighted_sum = gaussian_sum * input_row[5]

#       Replaced lines above with following lines 21-09-2018
        # Multiply gaussian_sum by normalized score to obtain weighted_norm_sum 
        weighted_norm_sum = gaussian_sum * input_norm_score    # input_list[7]

    return tf.stack([gaussian_sum, bbox_area, weighted_norm_sum], axis = -1)

'''    
              