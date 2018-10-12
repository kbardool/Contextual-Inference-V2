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
from scipy.stats import  multivariate_normal
# import scipy.misc
import tensorflow as tf
# import keras
import keras.backend as KB
import keras.layers as KL
import keras.engine as KE
sys.path.append('..')
import mrcnn.utils as utils
import tensorflow.contrib.util as tfc
import pprint

##----------------------------------------------------------------------------------------------------------------------          
##
##  09-21-2018: 
##    **  build_predictions --> build_refined_predictions
##              build_predicitions routine now applies predicted refinement (from mrcnn_bbox deltas) to the output_rois.
##              after applying the delta, the refined output rois are clipped so they dont exceed the NN image dimensions.
##              the modified coordiantes of the bboxes are placed in pred_tensor for futher processing           
##
##    ** renamed build_mask_routine --> build_hm_bbox_score
##              better reflects the function
##
##    ** Modified build_headmap : 
##       -  Added HEATMAP_SCALE_FACTOR
##  
##       -  Scores are now built using dense tensors : Instead of using `gauss_heatmap` (gauss_sum) and `in_tensor`, where 
##          the process of flattening and running build_hm_bbox_score against them includes a large number of empty
##          [0 0 0 0] bboxes (since gauss_heatmap is replicated  [num_classes x num_bbox_per_class] times),
##          The score calculation will be done on prob_grid and pt2_dense , which only include the real number 
##          of non-zero bboxes.                  
##          The scores are build using the same method (map_fn)
##          Scores are 1) sum of gaussian distribution within bounding box 
##                         2) area of bounding box in pixels (not used)
##                         3) sum of gaussian within bounding bouding box * normalized score. 
##
##----------------------------------------------------------------------------------------------------------------------          
   
##----------------------------------------------------------------------------------------------------------------------          
## build_refined_predictions 
##----------------------------------------------------------------------------------------------------------------------              
def build_refined_predictions(norm_input_rois, mrcnn_class, mrcnn_bbox, config):
    '''
    Split output_rois by class id, and add class_id and class_score 
    
    output:
    -------
    
    pred_tensor:        [ Batchsz, Num_Classes, Num_Rois, 7: (y1, x1, y2, x2, class_id, class_score, normalized class score)]
                        
                        y1,x1, y2,x2 are in image dimension format
    '''
    batch_size      = config.BATCH_SIZE
    num_classes     = config.NUM_CLASSES
    h, w            = config.IMAGE_SHAPE[:2]
    # num_rois        = config.TRAIN_ROIS_PER_IMAGE
    num_rois        = KB.int_shape(norm_input_rois)[1]
    scale           = tf.constant([h,w,h,w], dtype = tf.float32)
    # dup_scale       = tf.reshape(tf.tile(scale, [num_rois]),[num_rois,-1])
    dup_scale       = scale * tf.ones([batch_size, num_rois, 1], dtype = 'float32')

    det_per_class   = config.DETECTION_PER_CLASS
    
    print()
    print('  > build_predictions()')
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
    pred_classes     = tf.argmax( mrcnn_class,axis=-1,output_type = tf.int32)
    pred_classes_exp = tf.to_float(tf.expand_dims(pred_classes ,axis=-1))    
    #     print('    pred_classes : ', pred_classes.shape)
    #     print(pred_classes.eval())
    #     print('    pred_scores  : ', pred_scores.shape ,'\n', pred_scores.eval())
    #     print('    pred_classes_exp : ', pred_classes_exp.shape)
    
    gather_ind   = tf.stack([batch_grid , bbox_grid, pred_classes],axis = -1)
    pred_scores  = tf.gather_nd(mrcnn_class, gather_ind)
    pred_deltas  = tf.gather_nd(mrcnn_bbox , gather_ind)

    ##------------------------------------------------------------------------------------
    ## apply delta refinements to the  rois,  based on deltas provided by the mrcnn head 
    ##------------------------------------------------------------------------------------
    pred_deltas  = tf.multiply(pred_deltas, config.BBOX_STD_DEV, name = 'pred_deltas')
    input_rois   = tf.multiply(norm_input_rois , dup_scale )

    ## compute "refined rois"  utils.apply_box_deltas_tf(input_rois, pred_deltas)
    refined_rois   = utils.apply_box_deltas_tf(input_rois, pred_deltas)

    ##   Clip boxes to image window    
    window = tf.constant([[0,0,h,w]], dtype = tf.float32)
    refined_rois  = utils.clip_to_window_tf( window, refined_rois)
    
    print('    refined rois clipped   : ', refined_rois.shape)
    print('    input_rois.shape       : ', type(input_rois), KB.int_shape(input_rois), input_rois.get_shape())
    print('    refined_rois.shape     : ', type(refined_rois), KB.int_shape(refined_rois), refined_rois.get_shape())
    # print('    mrcnn_class : ', mrcnn_class.shape, mrcnn_class)
    # print('    gather_ind  : ', gather_ind.shape, gather_ind)
    # print('    pred_scores : ', pred_scores.shape )
    # print('    pred_deltas : ', pred_deltas.shape )   
    # print('    input_rois : ', input_rois.shape, input_rois)
    # print('    refined rois: ', refined_rois.shape, refined_rois)
        
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
    print('    shape of sequence      : ', sequence.shape)
    pred_array  = tf.concat([ refined_rois, pred_classes_exp , tf.expand_dims(pred_scores, axis = -1), sequence], 
                            axis=-1, name = 'pred_array')
     
    #--------------------------------------------------------------------------------------------
    # pred_array  = tf.concat([refined_rois, pred_classes_exp , tf.expand_dims(pred_scores, axis = -1)], axis=-1)
    #---------------------------------------------------------------------------------------------
    
    scatter_ind = tf.stack([batch_grid , pred_classes, bbox_grid],axis = -1)
    pred_scatt  = tf.scatter_nd(scatter_ind, pred_array, [batch_size, num_classes, num_rois, pred_array.shape[-1]])
    print('    pred_array             : ', pred_array.shape)  
    print('    scatter_ind            : ', type(scatter_ind), 'shape', scatter_ind.shape)
    print('    pred_scatter           : ', pred_scatt.get_shape())
    
    ##--------------------------------------------------------------------------------------------
    ##  Apply a per class score normalization using the score column (COLUMN 5)
    ##  
    ##--------------------------------------------------------------------------------------------
    normalizer   = tf.reduce_max(pred_scatt[...,5], axis = -1, keepdims=True)
    normalizer   = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    norm_score   = tf.expand_dims(pred_scatt[...,5]/normalizer, axis = -1)
    pred_scatt   = tf.concat([pred_scatt, norm_score],axis = -1)   
    print('    - Add normalized score --\n')
    print('    normalizer             : ', normalizer.shape)  
    print('    norm_score             : ', norm_score.shape)
    print('    pred_scatter           : ', pred_scatt.get_shape())
    
    ##------------------------------------------------------------------------------------
    ## 22-09-2018: sort is now based on sequence which was added as last column
    ##             (previously sort was on bbox scores)
    ##------------------------------------------------------------------------------------
    _, sort_inds = tf.nn.top_k(pred_scatt[...,6], k=pred_scatt.shape[2])
    
    # build indexes to gather rows from pred_scatter based on sort order    
    class_grid, batch_grid, roi_grid = tf.meshgrid(tf.range(num_classes),tf.range(batch_size), tf.range(num_rois))
    roi_grid_exp = tf.to_float(tf.expand_dims(roi_grid, axis = -1))
    
    gather_inds  = tf.stack([batch_grid , class_grid, sort_inds],axis = -1)
    pred_tensor  = tf.gather_nd(pred_scatt, gather_inds[...,:det_per_class,:], name = 'pred_tensor')    

    # append an index to the end of each row --- commented out 30-04-2018
    # pred_tensor  = tf.concat([pred_tensor, roi_grid_exp], axis = -1)

    print('    sort_inds              : ', type(sort_inds)   , ' shape ', sort_inds.shape)
    print('    class_grid             : ', type(class_grid)  , ' shape ', class_grid.get_shape())
    print('    batch_grid             : ', type(batch_grid)  , ' shape ', batch_grid.get_shape())
    print('    roi_grid shape         : ', type(roi_grid)    , ' shape ', roi_grid.get_shape()) 
    print('    roi_grid_exp           : ', type(roi_grid_exp), ' shape ', roi_grid_exp.get_shape())
    print('    gather_inds            : ', type(gather_inds) , ' shape ', gather_inds.get_shape())
    print('    pred_tensor            : ', pred_tensor.get_shape())

    return  pred_tensor
    
    
##----------------------------------------------------------------------------------------------------------------------          
## 
##----------------------------------------------------------------------------------------------------------------------          
def build_ground_truth(gt_class_ids, norm_gt_bboxes, config):

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

    
    
    
##----------------------------------------------------------------------------------------------------------------------          
##  build_heatmap
## 
##  Build gaussian heatmaps using pred_tensor
## 
##  INPUTS :
##
##----------------------------------------------------------------------------------------------------------------------          
def build_heatmap(in_tensor, config, names = None):
  
    num_detections  = config.DETECTION_MAX_INSTANCES
    img_h, img_w    = config.IMAGE_SHAPE[:2]
    batch_size      = config.BATCH_SIZE
    num_classes     = config.NUM_CLASSES  
    heatmap_scale   = config.HEATMAP_SCALE_FACTOR
    grid_h, grid_w  = config.IMAGE_SHAPE[:2] // heatmap_scale    
    # rois per image is determined by size of input tensor 
    #   detection mode:   config.TRAIN_ROIS_PER_IMAGE 
    #   ground_truth  :   config.DETECTION_MAX_INSTANCES
    # strt_cls        = 0 if rois_per_image == 32 else 1
    rois_per_image  = (in_tensor.shape)[2]  
    print('\n ')
    print('  > NEW build_heatmap() for ', names )
    print('    in_tensor shape        : ', in_tensor.shape)       
    print('    num bboxes per class   : ', rois_per_image )
    print('    heatmap scale        : ', heatmap_scale, 'Dimensions:  w:', grid_w,' h:', grid_h)
    #-----------------------------------------------------------------------------    
    ## Stack non_zero bboxes from in_tensor into pt2_dense 
    #-----------------------------------------------------------------------------
    # pt2_ind shape is [?, 3]. 
    #    pt2_ind[0] corresponds to image_index 
    #    pt2_ind[1] corresponds to class_index 
    #    pt2_ind[2] corresponds to roi row_index 
    # pt2_dense shape is [?, 7]
    #    pt2_dense[0:3]  roi coordinates 
    #    pt2_dense[4]    is class id 
    #    pt2_dense[5]    is score from mrcnn    
    #    pt2_dense[6]    is bbox sequence id    
    #    pt2_dense[7]    is normalized score (pre class)    
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
    mvn = tfd.MultivariateNormalDiag( loc  = means,  scale_diag = covar)
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
    gauss_scatt   = tf.scatter_nd(pt2_ind, prob_grid_norm_scaled, [batch_size, num_classes, rois_per_image, grid_w, grid_h], name = 'gauss_scatter')
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
    print('    gaussian_heatmap shape     : ', gauss_heatmap.get_shape(), 'Keras tensor ', KB.is_keras_tensor(gauss_heatmap) )      
    
    ##---------------------------------------------------------------------------------------------
    ## (5) heatmap normalization
    ##     normalizer is set to one when the max of class is zero     
    ##     this prevents elements of gauss_heatmap_norm computing to nan
    ##---------------------------------------------------------------------------------------------
    print('\n    normalization ------------------------------------------------------')   
    normalizer = tf.reduce_max(gauss_heatmap, axis=[-2,-1], keepdims = True)
    normalizer = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    gauss_heatmap_norm = gauss_heatmap / normalizer
    print('    normalizer shape       : ', normalizer.shape)   
    print('    gauss norm            : ', gauss_heatmap_norm.shape   ,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap_norm) ) 
    
    #-------------------------------------------------------------------------------------
    # scatter out the probability distributions based on class 
    #-------------------------------------------------------------------------------------
    # print('\n    Scatter out the probability distributions based on class --------------') 
    # gauss_scatt   = tf.scatter_nd(pt2_ind, prob_grid, [batch_size, num_classes, rois_per_image, grid_w, grid_h], name = 'gauss_scatter')
    # print('    pt2_ind shape   : ', pt2_ind.shape)  
    # print('    prob_grid shape : ', prob_grid.shape)  
    # print('    gauss_scatt     : ', gauss_scatt.shape)   # batch_sz , num_classes, num_rois, image_h, image_w
    
    #-------------------------------------------------------------------------------------
    # SUM : Reduce and sum up gauss_scattered by class  
    #-------------------------------------------------------------------------------------
    # print('\n    Reduce sum based on class ---------------------------------------------')         
    # gauss_heatmap = tf.reduce_sum(gauss_scatt, axis=2, name='pred_heatmap2')
    # force small sums to zero - for now (09-11-18) commented out but could reintroduce based on test results
    # gauss_heatmap = tf.where(gauss_heatmap < 1e-12, gauss_heatmap, tf.zeros_like(gauss_heatmap), name='Where1')
    # print('    gaussian_hetmap shape     : ', gauss_heatmap.get_shape(), 'Keras tensor ', KB.is_keras_tensor(gauss_heatmap) )   
    
 
    #---------------------------------------------------------------------------------------------
    # heatmap normalization per class
    # normalizer is set to one when the max of class is zero     
    # this prevents elements of gauss_heatmap_norm computing to nan
    #---------------------------------------------------------------------------------------------
    # print('\n    normalization ------------------------------------------------------')   
    # normalizer = tf.reduce_max(gauss_heatmap, axis=[-2,-1], keepdims = True)
    # print('   normalizer shape       : ', normalizer.shape)
    # normalizer = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    # gauss_heatmap_norm = gauss_heatmap / normalizer
    # print('    gauss norm            : ', gauss_heatmap_norm.shape   ,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap_norm) )

    ##--------------------------------------------------------------------------------------------
    ##  Generate scores using prob_grid and pt2_dense - NEW METHOD
    ##  added 09-21-2018
    ##--------------------------------------------------------------------------------------------
    scores_from_sum2 = tf.map_fn(build_hm_score, [prob_grid, pt2_dense_scaled, pt2_dense[:,7]], dtype = tf.float32, swap_memory = True)
    scores_scattered = tf.scatter_nd(pt2_ind, scores_from_sum2, [batch_size, num_classes, rois_per_image, 3], name = 'scores_scattered')
    gauss_scores = tf.concat([in_tensor, scores_scattered], axis = -1,name = names[0]+'_scores')
    print('    scores_scattered shape : ', scores_scattered.shape) 
    print('    gauss_scores           : ', gauss_scores.shape, ' Name:   ', gauss_scores.name)
    print('    gauss_scores  (FINAL)  : ', gauss_scores.shape, ' Keras tensor ', KB.is_keras_tensor(gauss_scores) )      
    
    #---------------------------------------------------------------------------------------------
    #   Normalization is already perfored on the scores at a per_class leve, so we dont use this 
    #  code below anympre
    #
    #  This is a regular normalization that moves everything between [0, 1]. 
    #  This causes negative values to move to -inf, which is a problem in FCN scoring. 
    #  To address this a normalization between [-1 and +1] was introduced in FCN.
    #  Not sure how this will work with training tho.
    #----------------------------------------------------------------------------------------------
    #     normalizer   = tf.reduce_max(scores_scatt[...,-1], axis = -1, keepdims=True)
    #     print('norm',normalizer.shape)
    #     normalizer   = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    #     norm_score2   = tf.expand_dims(scores_scatt[...,-1]/normalizer, axis = -1)
    #     print('norm_SCORE2',norm_score2.shape)
    #----------------------------------------------------------------------------------------------
    
    
    #----------------------------------------------------------------------------------------------
    #  Generate scores using GAUSS_SUM -- OLD METHOD
    #  removed 09-21-2018
    #----------------------------------------------------------------------------------------------
    #   Generate scores : 
    #   -----------------
    #  NOTE: Score is generated on NORMALIZED gaussian distributions (GAUSS_NORM)
    #        If want to do this on NON-NORMALIZED, we need to apply it on GAUSS_SUM
    #        Testing demonstated that the NORMALIZED score generated from using GAUSS_SUM 
    #        and GAUSS_NORM are the same. 
    #        For now we will use GAUSS_SUM score and GAUSS_NORM heatmap. The reason being that 
    #        the raw score generated in GAUSS_SUM is much smaller. 
    #        We may need to change this base on the training results from FCN 
    #---------------------------------------------------------------------------------------------
    #   duplicate GAUSS_NORM <num_roi> times to pass along with bboxes to map_fn function
    # 
    #   Here we have a choice to calculate scores using the GAUSS_SUM (unnormalized) or GAUSS_NORM (normalized)
    #   after looking at the scores and ratios for each option, I decided to go with the normalized 
    #   as the numbers are larger
    #
    #   Examples>
    #   Using GAUSS_SUM
    # [   3.660313    3.513489   54.475536   52.747402    1.   0.999997    4.998889 2450.          0.00204     0.444867]
    # [   7.135149    1.310972   50.020126   44.779854    1.   0.999991    4.981591 1892.          0.002633    0.574077]
    # [  13.401865    0.         62.258957   46.636948    1.   0.999971    4.957398 2303.          0.002153    0.469335]
    # [   0.          0.         66.42349    56.123024    1.   0.999908    4.999996 3696.          0.001353    0.294958]
    # [   0.          0.         40.78952    60.404335    1.   0.999833    4.586552 2460.          0.001864    0.406513]    
    #                                                       
    #   Using GAUSS_NORM:                             class   r-cnn scr   
    # [   3.660313    3.513489   54.475536   52.747402    1.   0.999997 1832.9218   2450.          0.748131    0.479411]
    # [   7.135149    1.310972   50.020126   44.779854    1.   0.999991 1659.3965   1892.          0.877059    0.56203 ]
    # [  13.401865    0.         62.258957   46.636948    1.   0.999971 1540.4974   2303.          0.668909    0.428645]
    # [   0.          0.         66.42349    56.123024    1.   0.999908 1925.3267   3696.          0.520922    0.333813]
    # [   0.          0.         40.78952    60.404335    1.   0.999833 1531.321    2460.          0.622488    0.398898]
    # 
    #  to change the source, change the following line gauss_heatmap_norm <--> gauss_heatmap
    #---------------------------------------------------------------------------------------------------------------------------
    # flatten guassian scattered and input_tensor, and pass on to build_bbox_score routine 
    # in_shape = tf.shape(in_tensor)
    # print('    shape of in_tensor is : ', KB.int_shape(in_tensor))
    # in_tensor_flattened  = tf.reshape(in_tensor, [-1, in_shape[-1]])  <-- not a good reshape style!! 
    # replaced with following line:
    # in_tensor_flattened  = tf.reshape(in_tensor, [-1, in_tensor.shape[-1]])
    #
    #  bboxes = tf.to_int32(tf.round(in_tensor_flattened[...,0:4]))
    #
    # print('    in_tensor             : ', in_tensor.shape)
    # print('    in_tensor_flattened   : ', in_tensor_flattened.shape)
    # print('    Rois per class        : ', rois_per_image)
    #
    #     print('\n    Scores from gauss_heatmap ----------------------------------------------')
    #     temp = tf.expand_dims(gauss_heatmap, axis =2)
    #     print('    temp expanded          : ', temp.shape)
    #     temp = tf.tile(temp, [1,1, rois_per_image ,1,1])
    #     print('    temp tiled shape       : ', temp.shape)
    # 
    #     temp = KB.reshape(temp, (-1, temp.shape[-2], temp.shape[-1]))
    #     
    #     print('    temp flattened         : ', temp.shape)
    #     print('    in_tensor_flattened    : ', in_tensor_flattened.shape)
    # 
    #     scores_from_sum = tf.map_fn(build_hm_score, [temp, in_tensor_flattened], dtype=tf.float32)
    #     scores_shape    = [in_tensor.shape[0], in_tensor.shape[1], in_tensor.shape[2], -1]
    #     scores_from_sum = tf.reshape(scores_from_sum, scores_shape)    
    #     print('    reshaped scores        : ', scores_from_sum.shape)
    #--------------------------------------------------------------------------------------------
    #  tf.reduce_max(scores_from_sum[...,-1], axis = -1, keepdims=True) result is [num_imgs, num_class, 1]
    #
    #  This is a regular normalization that moves everything between [0, 1]. 
    #  This causes negative values to move to -inf, which is a problem in FCN scoring. 
    #  To address this a normalization between [-1 and +1] was introduced in FCN.
    #  Not sure how this will work with training tho.
    #--------------------------------------------------------------------------------------------
    #     normalizer   = tf.reduce_max(scores_from_sum[...,-1], axis = -1, keepdims=True)
    #     normalizer   = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    #     norm_score   = tf.expand_dims(scores_from_sum[...,-1]/normalizer, axis = -1)
    #--------------------------------------------------------------------------------------------
    # Append `in_tensor` and `scores_from_sum` to form `bbox_scores`
    #--------------------------------------------------------------------------------------------
    #     gauss_scores = tf.concat([in_tensor, scores_from_sum, norm_score], axis = -1,name = names[0]+'_scores')
    #     print('    scores_from_sum final  : ', scores_from_sum.shape)    
    #     print('    norm_score             : ', norm_score.shape)
    #     print('    gauss_scores           : ', gauss_scores.shape,  '   name:   ', gauss_scores.name)
    #     print('    gauss_scores  (FINAL)  : ', gauss_scores.shape, ' Keras tensor ', KB.is_keras_tensor(gauss_scores) )    
    #--------------------------------------------------------------------------------------------------------------------
    
    ##--------------------------------------------------------------------------------------------
    ## //create heatmap Append `in_tensor` and `scores_from_sum` to form `bbox_scores`
    ##--------------------------------------------------------------------------------------------
    # gauss_heatmap  = tf.transpose(gauss_heatmap,[0,2,3,1], name = names[0])
    # print('    gauss_heatmap       shape : ', gauss_heatmap.shape     ,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap) )  
    
    gauss_heatmap_norm = tf.transpose(gauss_heatmap_norm,[0,2,3,1], name = names[0]+'_norm')
    print('    gauss_heatmap_norm  shape : ', gauss_heatmap_norm.shape,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap_norm) )  
    print('    complete')

    return   gauss_heatmap_norm, gauss_scores  # , gauss_heatmap   gauss_heatmap_L2norm    # [gauss_heatmap, gauss_scatt, means, covar]    

    
##----------------------------------------------------------------------------------------------------------------------          
##
##----------------------------------------------------------------------------------------------------------------------          
    
def build_hm_score(input_list):
    '''
    Inputs:
    -----------
        heatmap_tensor :    [ image height, image width ]
        input_row      :    [y1, x1, y2, x2] in absolute (non-normalized) scale

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
        bbox_mask    = tf.stack([Y,X],axis=2)        
        mask_indices = tf.reshape(bbox_mask,[-1,2])
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
                  
##----------------------------------------------------------------------------------------------------------------------          
##
##----------------------------------------------------------------------------------------------------------------------          
     
class CHMLayer(KE.Layer):
    '''
    Contextual Heatmap Layer  (previously CHMLayerTF)
    Receives the bboxes, their repsective classification and roi_outputs and 
    builds the per_class tensor

    Returns:
    -------
    The CHM layer returns the following tensors:

    pred_tensor :       [batch, NUM_CLASSES, TRAIN_ROIS_PER_IMAGE    , (index, class_prob, y1, x1, y2, x2, class_id, old_idx)]
                                in normalized coordinates   
    pred_cls_cnt:       [batch, NUM_CLASSES] 
    gt_tensor:          [batch, NUM_CLASSES, DETECTION_MAX_INSTANCES, (index, class_prob, y1, x1, y2, x2, class_id, old_idx)]
    gt_cls_cnt:         [batch, NUM_CLASSES]
    
     Note: Returned arrays might be zero padded if not enough target ROIs.
    
    '''

    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        print('--------------------------------')
        print('>>>  CHM Layer  ')
        print('--------------------------------')
        self.config = config

        
    def call(self, inputs):

        mrcnn_class , mrcnn_bbox,  output_rois, tgt_class_ids, tgt_bboxes = inputs
        print('  > CHMLayer Call() ', len(inputs))
        print('    mrcnn_class.shape    :',   mrcnn_class.shape, KB.int_shape(  mrcnn_class ))
        print('    mrcnn_bbox.shape     :',    mrcnn_bbox.shape, KB.int_shape(   mrcnn_bbox )) 
        print('    output_rois.shape    :',   output_rois.shape, KB.int_shape(  output_rois )) 
        print('    tgt_class_ids.shape  :', tgt_class_ids.shape, KB.int_shape(tgt_class_ids )) 
        print('    tgt_bboxes.shape     :',    tgt_bboxes.shape, KB.int_shape(   tgt_bboxes )) 
         

        # pred_tensor  = build_predictions(output_rois, mrcnn_class, mrcnn_bbox, self.config)
        # pr_hm_norm, pr_hm_scores  = build_heatmap(pred_tensor, self.config, names = ['pred_heatmap'])
        # pred_cls_cnt = KL.Lambda(lambda x: tf.count_nonzero(x[:,:,:,-1],axis = -1), name = 'pred_cls_count')(pred_tensor)        

        
        pred_tensor   = build_refined_predictions(output_rois, mrcnn_class, mrcnn_bbox, self.config)
        pr_hm_norm, pr_hm_scores  = build_heatmap(pred_tensor, self.config, names = ['pred_heatmap'])
                     

        gt_tensor     = build_ground_truth (tgt_class_ids,  tgt_bboxes, self.config)  
        gt_hm_norm, gt_hm_scores  = build_heatmap(gt_tensor, self.config, names = ['gt_heatmap'])
        # gt_cls_cnt   = KL.Lambda(lambda x: tf.count_nonzero(x[:,:,:,-1],axis = -1), name = 'gt_cls_count')(gt_tensor)

        print()
        print('    pred_refined_heatmap        : ', pr_hm_norm.shape   , 'Keras tensor ', KB.is_keras_tensor(pr_hm_norm))
        print('    pred_refnined_heatmap_scores: ', pr_hm_scores.shape , 'Keras tensor ', KB.is_keras_tensor(pr_hm_scores))
        print('    gt_heatmap                  : ', gt_hm_norm.shape   , 'Keras tensor ', KB.is_keras_tensor(gt_hm_norm))
        print('    gt_heatmap_scores           : ', gt_hm_scores.shape , 'Keras tensor ', KB.is_keras_tensor(gt_hm_scores))
        print('    complete')
        
        return [  pr_hm_norm, pr_hm_scores, 
                  gt_hm_norm  ,gt_hm_scores,  
                  pred_tensor , gt_tensor]

         
    def compute_output_shape(self, input_shape):
        # may need to change dimensions of first return from IMAGE_SHAPE to MAX_DIM
        return [
                 (None, self.config.IMAGE_SHAPE[0], self.config.IMAGE_SHAPE[1], self.config.NUM_CLASSES)  # pred_refined_heatmap_norm
              ,  (None, self.config.NUM_CLASSES   , self.config.DETECTION_PER_CLASS ,12)                  # pred_refined_heatmap_scores 

              ,  (None, self.config.IMAGE_SHAPE[0], self.config.IMAGE_SHAPE[1], self.config.NUM_CLASSES)  # gt_heatmap_norm
              ,  (None, self.config.NUM_CLASSES   , self.config.DETECTION_PER_CLASS ,12)                  # gt_heatmap+scores   

              # ----extra stuff for now ---------------------------------------------------------------------------------------------------
              ,  (None, self.config.NUM_CLASSES   , self.config.DETECTION_PER_CLASS ,8)                  # pred_refined_tensor               
              ,  (None, self.config.NUM_CLASSES   , self.config.DETECTION_PER_CLASS ,8)                  # gt_tensor               
              
              
              # ,  (None, self.config.NUM_CLASSES   , self.config.DETECTION_PER_CLASS ,4)                  # pred_deltas 
              
              # ,  (None, self.config.NUM_CLASSES , self.config.TRAIN_ROIS_PER_IMAGE    ,10)            # pred_heatmap_scores (expanded) 
              # ,  (None, self.config.NUM_CLASSES , self.config.DETECTION_MAX_INSTANCES ,10)            # gt_heatmap+scores   (expanded) 
              # ,  (None, self.config.NUM_CLASSES , self.config.TRAIN_ROIS_PER_IMAGE    , 7)            # pred_tensor
              # ,  (None, self.config.NUM_CLASSES , self.config.DETECTION_MAX_INSTANCES , 7)            # gt_tensor   (expanded) 

              ]


##----------------------------------------------------------------------------------------------------------------------          
##
##
##
##----------------------------------------------------------------------------------------------------------------------

          
##----------------------------------------------------------------------------------------------------------------------          
## removed 25-09-2018  and replaced with version that appled  delta refinements and clips bounding boxes to image boundaries
##----------------------------------------------------------------------------------------------------------------------          

"""              
##----------------------------------------------------------------------------------------------------------------------          
## build_predictions -- will be replaced by build_refined_predictions( )
##----------------------------------------------------------------------------------------------------------------------              
def build_predictions(norm_input_rois, mrcnn_class, mrcnn_bbox, config):
    '''
    Split output_rois by class id, and add class_id and class_score 
    
    output:
    -------
    
    pred_tensor:        [ Batchsz, Num_Classes, Num_Rois, 7: (y1, x1, y2, x2, class_id, class_score, normalized class score)]
                        
                        y1,x1, y2,x2 are in image dimension format
    '''
    batch_size      = config.BATCH_SIZE
    num_classes     = config.NUM_CLASSES
    h, w            = config.IMAGE_SHAPE[:2]
    # num_rois        = config.TRAIN_ROIS_PER_IMAGE
    num_rois        = KB.int_shape(norm_input_rois)[1]
    scale           = tf.constant([h,w,h,w], dtype = tf.float32)
    # dup_scale       = tf.reshape(tf.tile(scale, [num_rois]),[num_rois,-1])
    dup_scale       = scale * tf.ones([batch_size, num_rois, 1], dtype = 'float32')

    det_per_class   = config.DETECTION_PER_CLASS
    
    print()
    print('  > build_predictions()')
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
    pred_classes     = tf.argmax( mrcnn_class,axis=-1,output_type = tf.int32)
    pred_classes_exp = tf.to_float(tf.expand_dims(pred_classes ,axis=-1))    
    #     print('    pred_classes : ', pred_classes.shape)
    #     print(pred_classes.eval())
    #     print('    pred_scores  : ', pred_scores.shape ,'\n', pred_scores.eval())
    #     print('    pred_classes_exp : ', pred_classes_exp.shape)
    
    gather_ind   = tf.stack([batch_grid , bbox_grid, pred_classes],axis = -1)
    pred_scores  = tf.gather_nd(mrcnn_class, gather_ind)
    pred_deltas  = tf.gather_nd(mrcnn_bbox , gather_ind)

    ##------------------------------------------------------------------------------------
    ## convert input_rois to unnormalized coordiantes 
    ##------------------------------------------------------------------------------------
    input_rois   = tf.multiply(norm_input_rois , dup_scale )
    
    print('    input_rois.shape       : ', type(input_rois), KB.int_shape(input_rois), input_rois.get_shape())
    # print('    mrcnn_class : ', mrcnn_class.shape, mrcnn_class)
    # print('    gather_ind  : ', gather_ind.shape, gather_ind)
    # print('    pred_scores : ', pred_scores.shape )
    # print('    pred_deltas : ', pred_deltas.shape )   
    # print('    input_rois : ', input_rois.shape, input_rois)
    # print('    refined rois: ', refined_rois.shape, refined_rois)
        
    #------------------------------------------------------------------------------------
    # 22-05-2018 - stopped using the following code as it was clipping too many bouding 
    # boxes to 0 or 128 causing zero area generation
    #------------------------------------------------------------------------------------
    # ##   Clip boxes to image window    
    # # for now we will consider the window [0,0, 128,128]
    # #     _, _, window, _ =  parse_image_meta(image_meta)    
    # window        = tf.constant([[0,0,128,128]], dtype =tf.float32)   
    # refined_rois  = utils.clip_to_window_tf(window, refined_rois)
    # print('    refined rois clipped: ', refined_rois.shape, refined_rois)
    #------------------------------------------------------------------------------------
    
    
    ##------------------------------------------------------------------------------------
    ##  Build Pred_Scatter: tensor of bounding boxes by Image / Class
    ##------------------------------------------------------------------------------------
    # sequence id is used to preserve the order of rois as passed to this routine
    #  This may be important in the post matching process but for now it's not being used.
    #     sequence = tf.ones_like(pred_classes, dtype = tf.int32) * (bbox_grid[...,::-1] + 1) 
    #     sequence = tf.to_float(tf.expand_dims(sequence, axis = -1))   
    #     print(sequence.shape)
    #     print(sequence.eval())
    #     pred_array  = tf.concat([ refined_rois, pred_classes_exp , tf.expand_dims(pred_scores, axis = -1), sequence], axis=-1)
    #------------------------------------------------------------------------------------
    pred_array  = tf.concat([input_rois, pred_classes_exp , tf.expand_dims(pred_scores, axis = -1)], axis=-1)
    scatter_ind = tf.stack([batch_grid , pred_classes, bbox_grid],axis = -1)
    pred_scatt  = tf.scatter_nd(scatter_ind, pred_array, [batch_size, num_classes, num_rois, pred_array.shape[-1]])
    print('    pred_array             : ', pred_array.shape)  
    print('    scatter_ind            : ', type(scatter_ind), 'shape', scatter_ind.shape)
    print('    pred_scatter           : ', pred_scatt.get_shape())
    
    ##--------------------------------------------------------------------------------------------
    ##  Apply a per class score normalization
    ##--------------------------------------------------------------------------------------------
    normalizer   = tf.reduce_max(pred_scatt[...,-1], axis = -1, keepdims=True)
    normalizer   = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    norm_score   = tf.expand_dims(pred_scatt[...,-1]/normalizer, axis = -1)
    pred_scatt   = tf.concat([pred_scatt, norm_score],axis = -1)   
    print('    - Add normalized score --\n')
    print('    normalizer             : ', normalizer.shape)  
    print('    norm_score             : ', norm_score.shape)
    print('    pred_scatter           : ', pred_scatt.get_shape())
    
    ##------------------------------------------------------------------------------------
    ## sort pred_scatter in each class dimension based on bbox scores (last column)
    ##------------------------------------------------------------------------------------
    _, sort_inds = tf.nn.top_k(pred_scatt[...,-1], k=pred_scatt.shape[2])
    
    # build indexes to gather rows from pred_scatter based on sort order    
    class_grid, batch_grid, roi_grid = tf.meshgrid(tf.range(num_classes),tf.range(batch_size), tf.range(num_rois))
    roi_grid_exp = tf.to_float(tf.expand_dims(roi_grid, axis = -1))
    
    gather_inds  = tf.stack([batch_grid , class_grid, sort_inds],axis = -1)
    pred_tensor  = tf.gather_nd(pred_scatt, gather_inds[...,:det_per_class,:], name = 'pred_tensor')    

    # append an index to the end of each row --- commented out 30-04-2018
    # pred_tensor  = tf.concat([pred_tensor, roi_grid_exp], axis = -1)

    print('    sort_inds              : ', type(sort_inds)   , ' shape ', sort_inds.shape)
    print('    class_grid             : ', type(class_grid)  , ' shape ', class_grid.get_shape())
    print('    batch_grid             : ', type(batch_grid)  , ' shape ', batch_grid.get_shape())
    print('    roi_grid shape         : ', type(roi_grid)    , ' shape ', roi_grid.get_shape()) 
    print('    roi_grid_exp           : ', type(roi_grid_exp), ' shape ', roi_grid_exp.get_shape())
    print('    gather_inds            : ', type(gather_inds) , ' shape ', gather_inds.get_shape())
    print('    pred_tensor            : ', pred_tensor.get_shape())

    return  pred_tensor    
    
 """
        
##----------------------------------------------------------------------------------------------------------------------          
## removed 21-09-2018  and replaced with version that appled changes documented at top of program 
##----------------------------------------------------------------------------------------------------------------------          
 
"""
##----------------------------------------------------------------------------------------------------------------------          
##  INPUTS :
##    FCN_HEATMAP    [ numn_images x height x width x num classes ] 
##    PRED_HEATMAP_SCORES 
##----------------------------------------------------------------------------------------------------------------------          
    
def build_heatmap(in_tensor, config, names = None):
  
    num_detections  = config.DETECTION_MAX_INSTANCES
    img_h, img_w    = config.IMAGE_SHAPE[:2]
    batch_size      = config.BATCH_SIZE
    num_classes     = config.NUM_CLASSES  
    # rois per image is determined by size of input tensor 
    #   detection mode:   config.TRAIN_ROIS_PER_IMAGE 
    #   ground_truth  :   config.DETECTION_MAX_INSTANCES
    # strt_cls        = 0 if rois_per_image == 32 else 1
    rois_per_image  = (in_tensor.shape)[2]  
    print('\n ')
    print('  > NEW build_heatmap() for ', names )
    print('    in_tensor shape        : ', in_tensor.shape)       
    print('    num bboxes per class   : ', rois_per_image )

    #-----------------------------------------------------------------------------    
    ## Stack non_zero bboxes from in_tensor into pt2_dense 
    #-----------------------------------------------------------------------------
    # pt2_ind shape is [?, 3]. 
    #   pt2_ind[0] corresponds to image_index 
    #   pt2_ind[1] corresponds to class_index 
    #   pt2_ind[2] corresponds to roi row_index 
    # pt2_dense shape is [?, 6]
    #    pt2_dense[0] is image index
    #    pt2_dense[1:4]  roi cooridnaytes 
    #    pt2_dense[5]    is class id 
    #-----------------------------------------------------------------------------
    pt2_sum = tf.reduce_sum(tf.abs(in_tensor[:,:,:,:-2]), axis=-1)
    print('    pt2_sum shape ',pt2_sum.shape)
    # print(pt2_sum[0].eval())
    pt2_ind = tf.where(pt2_sum > 0)

    ## replaced the two operations below with the one above - 15-05-2018
    # pt2_mask = tf.greater(pt2_sum , 0)
    # pt2_ind  = tf.where(pt2_mask)
    # print(' pt2_mask shape ', pt2_mask.get_shape())
    # print(pt2_mask.eval())
    # print('    pt2_ind shape ', pt2_ind.get_shape())
    # print(pt2_ind.eval())

    pt2_dense = tf.gather_nd( in_tensor, pt2_ind)
    print('    dense shape ',pt2_dense.get_shape())

    #-----------------------------------------------------------------------------
    ## Build mesh-grid to hold pixel coordinates  
    #-----------------------------------------------------------------------------
    X = tf.range(img_w, dtype=tf.int32)
    Y = tf.range(img_h, dtype=tf.int32)
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
    bef_pos = tf.to_float(tf.stack([rep_X,rep_Y], axis = -1))
    print('    before transpse ', bef_pos.get_shape())
    pos_grid = tf.transpose(bef_pos,[1,2,0,3])
    print('    after transpose ', pos_grid.get_shape())    

    ##-----------------------------------------------------------------------------
    ##  Build mean and convariance tensors for Multivariate Normal Distribution 
    ##-----------------------------------------------------------------------------
    width  = pt2_dense[:,3] - pt2_dense[:,1]      # x2 - x1
    height = pt2_dense[:,2] - pt2_dense[:,0]
    cx     = pt2_dense[:,1] + ( width  / 2.0)
    cy     = pt2_dense[:,0] + ( height / 2.0)
    means  = tf.stack((cx,cy),axis = -1)
    covar  = tf.stack((width * 0.5 , height * 0.5), axis = -1)
    covar  = tf.sqrt(covar)

    ##-----------------------------------------------------------------------------
    ##  Compute Normal Distribution for bounding boxes
    ##-----------------------------------------------------------------------------    
    tfd = tf.contrib.distributions
    mvn = tfd.MultivariateNormalDiag( loc  = means,  scale_diag = covar)
    prob_grid = mvn.prob(pos_grid)
    print('     Prob_grid shape before tanspose: ',prob_grid.get_shape())
    prob_grid = tf.transpose(prob_grid,[2,0,1])
    print('     Prob_grid shape after tanspose: ',prob_grid.get_shape())    
    print('    >> input to MVN.PROB: pos_grid (meshgrid) shape: ', pos_grid.get_shape())
    print('    << output probabilities shape:' , prob_grid.get_shape())

    ##--------------------------------------------------------------------------------
    ## IMPORTANT: kill distributions of NaN boxes (resulting from bboxes with height/width of zero
    ## which cause singular sigma cov matrices
    ##--------------------------------------------------------------------------------
    prob_grid = tf.where(tf.is_nan(prob_grid),  tf.zeros_like(prob_grid), prob_grid)
    

    ##-------------------------------------------------------------------------------------
    ## scatter out the probability distributions based on class 
    ##-------------------------------------------------------------------------------------
    print('\n    Scatter out the probability distributions based on class --------------') 
    gauss_scatt   = tf.scatter_nd(pt2_ind, prob_grid, [batch_size, num_classes, rois_per_image, img_w, img_h])
    print('    pt2_ind shape   : ', pt2_ind.shape)  
    print('    prob_grid shape : ', prob_grid.shape)  
    print('    gauss_scatt     : ', gauss_scatt.shape)   # batch_sz , num_classes, num_rois, image_h, image_w
    
    ##-------------------------------------------------------------------------------------
    ## SUM : Reduce and sum up gauss_scattered by class  
    ##-------------------------------------------------------------------------------------
    print('\n    Reduce sum based on class ---------------------------------------------')         
    gauss_heatmap = tf.reduce_sum(gauss_scatt, axis=2, name='pred_heatmap2')

    # force small sums to zero - for now (09-11-18) commented out but could reintroduce based on test results
    # gauss_heatmap = tf.where(gauss_heatmap < 1e-12, gauss_heatmap, tf.zeros_like(gauss_heatmap), name='Where1')
    print('    gaussian_sum shape     : ', gauss_heatmap.get_shape(), 'Keras tensor ', KB.is_keras_tensor(gauss_heatmap) )      
    
        ### Normalize `gauss_heatmap`  --> `gauss_norm`
    
    
    #---------------------------------------------------------------------------------------------
    # heatmap L2 normalization
    # Normalization using the  `gauss_heatmap` (batchsize , num_classes, height, width) 
    # 17-05-2018 (New method, replace dthe previous method that usedthe transposed gauss sum
    # 17-05-2018 Replaced with normalization across the CLASS axis 
    #---------------------------------------------------------------------------------------------
    # print('\n    L2 normalization ------------------------------------------------------')   
    # gauss_L2norm   = KB.l2_normalize(gauss_heatmap, axis = +1)   # normalize along the CLASS axis 
    # print('    gauss L2 norm   : ', gauss_L2norm.shape   ,' Keras tensor ', KB.is_keras_tensor(gauss_L2norm) )
    #---------------------------------------------------------------------------------------------

    ##---------------------------------------------------------------------------------------------
    ## gauss_heatmap normalization
    ## normalizer is set to one when the max of class is zero 
    
    ## this prevents elements of gauss_norm computing to nan
    ##---------------------------------------------------------------------------------------------
    print('\n    normalization ------------------------------------------------------')   
    normalizer = tf.reduce_max(gauss_heatmap, axis=[-2,-1], keepdims = True)
    normalizer = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    gauss_norm = gauss_heatmap / normalizer
    # gauss_norm    = gauss_heatmap / tf.reduce_max(gauss_heatmap, axis=[-2,-1], keepdims = True)
    # gauss_norm    = tf.where(tf.is_nan(gauss_norm),  tf.zeros_like(gauss_norm), gauss_norm, name = 'Where2')
    print('    gauss norm            : ', gauss_norm.shape   ,' Keras tensor ', KB.is_keras_tensor(gauss_norm) )

    ##--------------------------------------------------------------------------------------------
    ## generate score based on gaussian using bounding box masks 
    ## NOTE: Score is generated on NORMALIZED gaussian distributions (GAUSS_NORM)
    ##       If want to do this on NON-NORMALIZED, we need to apply it on GAUSS_SUM
    ##--------------------------------------------------------------------------------------------
    # flatten guassian scattered and input_tensor, and pass on to build_bbox_score routine 
    in_shape = tf.shape(in_tensor)
    print('    shape of in_tensor is : ', KB.int_shape(in_tensor))
    # in_tensor_flattened  = tf.reshape(in_tensor, [-1, in_shape[-1]])  <-- not a good reshape style!! 
    # replaced with following line:
    in_tensor_flattened  = tf.reshape(in_tensor, [-1, in_tensor.shape[-1]])
    
    #  bboxes = tf.to_int32(tf.round(in_tensor_flattened[...,0:4]))
    
    print('    in_tensor             : ', in_tensor.shape)
    print('    in_tensor_flattened   : ', in_tensor_flattened.shape)
    print('    Rois per class        : ', rois_per_image)
    
    #--------------------------------------------------------------------------------------------------------------------------
    # duplicate GAUSS_NORM <num_roi> times to pass along with bboxes to map_fn function
    #   Here we have a choice to calculate scores using the GAUSS_SUM (unnormalized) or GAUSS_NORM (normalized)
    #   after looking at the scores and ratios for each option, I decided to go with the normalized 
    #   as the numbers are larger
    #
    # Examples>
    #   Using GAUSS_SUM
    # [   3.660313    3.513489   54.475536   52.747402    1.   0.999997    4.998889 2450.          0.00204     0.444867]
    # [   7.135149    1.310972   50.020126   44.779854    1.   0.999991    4.981591 1892.          0.002633    0.574077]
    # [  13.401865    0.         62.258957   46.636948    1.   0.999971    4.957398 2303.          0.002153    0.469335]
    # [   0.          0.         66.42349    56.123024    1.   0.999908    4.999996 3696.          0.001353    0.294958]
    # [   0.          0.         40.78952    60.404335    1.   0.999833    4.586552 2460.          0.001864    0.406513]    
    #                                                       
    #   Using GAUSS_NORM:                             class   r-cnn scr   
    # [   3.660313    3.513489   54.475536   52.747402    1.   0.999997 1832.9218   2450.          0.748131    0.479411]
    # [   7.135149    1.310972   50.020126   44.779854    1.   0.999991 1659.3965   1892.          0.877059    0.56203 ]
    # [  13.401865    0.         62.258957   46.636948    1.   0.999971 1540.4974   2303.          0.668909    0.428645]
    # [   0.          0.         66.42349    56.123024    1.   0.999908 1925.3267   3696.          0.520922    0.333813]
    # [   0.          0.         40.78952    60.404335    1.   0.999833 1531.321    2460.          0.622488    0.398898]
    # 
    #  to change the source, change the following line gauss_norm <--> gauss_heatmap
    #---------------------------------------------------------------------------------------------------------------------------

    ##--------------------------------------------------------------------------------------------
    ##  Generate scores : 
    ##  Testing demonstated that the NORMALIZED score generated from using GAUSS_SUM and GAUSS_NORM
    ##  Are the same. 
    ##  For now we will use GAUSS_SUM score and GAUSS_NORM heatmap. The reason being that the 
    ##  raw score generated in GAUSS_SUM is much smaller. 
    ##  We may need to change this base on the training results from FCN 
    ##--------------------------------------------------------------------------------------------
    
    ##--------------------------------------------------------------------------------------------
    ##  Generate scores using GAUSS_SUM
    ##--------------------------------------------------------------------------------------------
    print('\n    Scores from gauss_heatmap ----------------------------------------------')
    temp = tf.expand_dims(gauss_heatmap, axis =2)
    print('    temp expanded          : ', temp.shape)

    temp = tf.tile(temp, [1,1, rois_per_image ,1,1])
    print('    temp tiled shape       : ', temp.shape)

    temp = KB.reshape(temp, (-1, temp.shape[-2], temp.shape[-1]))
    
    print('    temp flattened         : ', temp.shape)
    print('    in_tensor_flattened    : ', in_tensor_flattened.shape)

    scores_from_sum = tf.map_fn(build_hm_score, [temp, in_tensor_flattened], dtype=tf.float32)
    print('    Scores_from_sum (after build mask routine) : ', scores_from_sum.shape)   # [(num_batches x num_class x num_rois ), 3]    

    scores_shape    = [in_tensor.shape[0], in_tensor.shape[1], in_tensor.shape[2], -1]
    scores_from_sum = tf.reshape(scores_from_sum, scores_shape)    
    print('    reshaped scores        : ', scores_from_sum.shape)

    ##--------------------------------------------------------------------------------------------
    ##  tf.reduce_max(scores_from_sum[...,-1], axis = -1, keepdims=True) result is [num_imgs, num_class, 1]
    ##
    ##  This is a regular normalization that moves everything between [0, 1]. 
    ##  This causes negative values to move to -inf, which is a problem in FCN scoring. 
    ##  To address this a normalization between [-1 and +1] was introduced in FCN.
    ##  Not sure how this will work with training tho.
    ##--------------------------------------------------------------------------------------------
    normalizer   = tf.reduce_max(scores_from_sum[...,-1], axis = -1, keepdims=True)
    normalizer   = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    norm_score   = tf.expand_dims(scores_from_sum[...,-1]/normalizer, axis = -1)
    # scores_from_sum = tf.concat([scores_from_sum, norm_score],axis = -1)  <-- added to concat down below 18-9-18
     
    
    '''
    ##--------------------------------------------------------------------------------------------
    ##  Generate scores using normalized GAUSS_SUM (GAUSS_NORM)
    ##--------------------------------------------------------------------------------------------
    print('==== Scores from gauss_norm ================')
    temp = tf.expand_dims(gauss_norm, axis =2)
    print('    temp expanded shape       : ', temp.shape)
    
    temp = tf.tile(temp, [1,1, rois_per_image ,1,1])
    print('    temp tiled shape          : ', temp.shape)

    temp_reshape = KB.reshape(temp, (-1, temp.shape[-2], temp.shape[-1]))
    print('    temp flattened shape      : ', temp_reshape.shape)
    print('    in_tensor_flattened       : ', in_tensor_flattened.shape)
    
    scores_from_norm = tf.map_fn(build_mask_routine_inf, [temp_reshape, in_tensor_flattened], dtype=tf.float32)
    print('    Scores_from_norm (after build mask routine) : ', scores_from_norm.shape)   # [(num_batches x num_class x num_rois ), 3]    
    
    scores_shape    = [in_tensor.shape[0], in_tensor.shape[1],in_tensor.shape[2], -1]
    scores_from_norm = tf.reshape(scores_from_norm, scores_shape)    
    print('    reshaped scores       : ', scores_from_norm.shape)
    
    ##--------------------------------------------------------------------------------------------
    ##  normalize score between [0, 1]. 
    ##--------------------------------------------------------------------------------------------
    normalizer   = tf.reduce_max(scores_from_norm[...,-1], axis = -1, keepdims=True)
    normalizer   = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    
    print('    normalizer            : ',normalizer.shape)
    norm_score   = tf.expand_dims(scores_from_norm[...,-1]/normalizer, axis = -1)
    scores_from_norm = tf.concat([scores_from_norm, norm_score],axis = -1)
         
    print('    norm_score            : ', norm_score.shape)
    print('    scores_from_norm final: ', scores_from_norm.shape)   
    
    '''

    ##--------------------------------------------------------------------------------------------
    ## Append `in_tensor` and `scores_from_sum` to form `bbox_scores`
    ##--------------------------------------------------------------------------------------------
    gauss_scores = tf.concat([in_tensor, scores_from_sum, norm_score], axis = -1,name = names[0]+'_scores')
    print('    in_tensor              : ', in_tensor.shape)
    print('    scores_from_sum final  : ', scores_from_sum.shape)    
    print('    norm_score             : ', norm_score.shape)
    print('    gauss_scores           : ', gauss_scores.shape,  '   name:   ', gauss_scores.name)
    print('    gauss_scores  (FINAL)  : ', gauss_scores.shape, ' Keras tensor ', KB.is_keras_tensor(gauss_scores) )     
    
    ##--------------------------------------------------------------------------------------------
    ## //create heatmap Append `in_tensor` and `scores_from_sum` to form `bbox_scores`
    ##--------------------------------------------------------------------------------------------
    #     gauss_heatmap      = KB.identity(tf.transpose(gauss_heatmap,[0,2,3,1]), name = names[0])

    gauss_heatmap  = tf.transpose(gauss_heatmap,[0,2,3,1], name = names[0])
    gauss_norm = tf.transpose(gauss_norm,[0,2,3,1], name = names[0]+'_norm')

    # print('    gauss_heatmap       shape : ', gauss_heatmap.shape     ,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap) )  
    # print('    gauss_heatmap_norm  shape : ', gauss_heatmap_norm.shape,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap_norm) )  
    # print(gauss_heatmap)
    
    # gauss_heatmap_norm   = KB.identity(tf.transpose(gauss_norm,[0,2,3,1]), name = names[0]+'_norm')
    # print('    gauss_heatmap_norm final shape : ', gauss_heatmap_norm.shape,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap_norm) )  
    # gauss_heatmap_L2norm = KB.identity(tf.transpose(gauss_L2norm,[0,2,3,1]), name = names[0]+'_L2norm')
 
    print('    complete')
    return   gauss_norm, gauss_scores  # , gauss_heatmap   gauss_heatmap_L2norm    # [gauss_heatmap, gauss_scatt, means, covar]    

    
    '''
    
    17-9-2018 -- routine was cloned from chm_layer_inf, and this code was commented out as we dont use L2 normalization
    kept for history
    
    # consider the two new columns for reshaping the gaussian_bbox_scores
    new_shape   = tf.shape(in_tensor)+ [0,0,0, tf.shape(scores)[-1]]        
    bbox_scores = tf.concat([in_tensor_flattened, scores], axis = -1)
    bbox_scores = tf.reshape(bbox_scores, new_shape)
    # print('    new shape is            : ', new_shape.eval())
    print('    in_tensor_flattened     : ', in_tensor_flattened.shape)
    print('    Scores shape            : ', scores.shape)   # [(num_batches x num_class x num_rois ), 3]
    print('    boxes_scores (rehspaed) : ', bbox_scores.shape)    

    ##--------------------------------------------------------------------------------------------
    ## Normalize computed score above, and add it to the heatmap_score tensor as last column
    ##--------------------------------------------------------------------------------------------
    scr_L2norm   = tf.nn.l2_normalize(bbox_scores[...,-1], axis = -1)   # shape (num_imgs, num_class, num_rois)
    scr_L2norm   = tf.expand_dims(scr_L2norm, axis = -1)

    ##--------------------------------------------------------------------------------------------
    # shape of tf.reduce_max(bbox_scores[...,-1], axis = -1, keepdims=True) is (num_imgs, num_class, 1)
    #  This is a regular normalization that moves everything between [0, 1]. 
    #  This causes negative values to move to -inf, which is a problem in FCN scoring. 
    # To address this a normalization between [-1 and +1] was introduced in FCN.
    # Not sure how this will work with training tho.
    ##--------------------------------------------------------------------------------------------
    scr_norm     = bbox_scores[...,-1]/ tf.reduce_max(bbox_scores[...,-1], axis = -1, keepdims=True)
    scr_norm     = tf.where(tf.is_nan(scr_norm),  tf.zeros_like(scr_norm), scr_norm)     
    
    #--------------------------------------------------------------------------------------------
    # this normalization moves values to [-1, +1] which we use in FCN, but not here. 
    #--------------------------------------------------------------------------------------------    
    # reduce_max = tf.reduce_max(bbox_scores[...,-1], axis = -1, keepdims=True)
    # reduce_min = tf.reduce_min(bbox_scores[...,-1], axis = -1, keepdims=True)  ## epsilon    = tf.ones_like(reduce_max) * 1e-7
    # scr_norm  = (2* (bbox_scores[...,-1] - reduce_min) / (reduce_max - reduce_min)) - 1     

    scr_norm     = tf.where(tf.is_nan(scr_norm),  tf.zeros_like(scr_norm), scr_norm)  
    scr_norm     = tf.expand_dims(scr_norm, axis = -1)                             # shape (num_imgs, num_class, 32, 1)
    bbox_scores  = tf.concat([bbox_scores, scr_norm, scr_L2norm], axis = -1)
    
    gauss_heatmap        = KB.identity(tf.transpose(gauss_heatmap,[0,2,3,1]), name = names[0])
    gauss_heatmap_norm   = KB.identity(tf.transpose(gauss_norm,[0,2,3,1]), name = names[0]+'_norm')
    gauss_heatmap_L2norm = KB.identity(tf.transpose(gauss_L2norm,[0,2,3,1]), name = names[0]+'_L2norm')
    gauss_scores         = KB.identity(bbox_scores, name = names[0]+'_scores') 
    
    print('    gauss_heatmap final shape : ', gauss_heatmap.shape   ,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap) )  
    print('    gauss_scores  final shape : ', gauss_scores.shape ,' Keras tensor ', KB.is_keras_tensor(gauss_scores) )  
    print('    complete')

    return   gauss_heatmap_norm, gauss_scores, gauss_heatmap,gauss_heatmap_L2norm    # [gauss_heatmap, gauss_scatt, means, covar]    
    '''

"""

##----------------------------------------------------------------------------------------------------------------------          
## back up copy 25-09-2018  - before creating build_heatmap2 which uses pred_array instead of pred_tensor and 
## introduces heatmap scaling
##----------------------------------------------------------------------------------------------------------------------          


"""
##----------------------------------------------------------------------------------------------------------------------          
##  INPUTS :
##    FCN_HEATMAP    [ numn_images x height x width x num classes ] 
##    PRED_HEATMAP_SCORES 
##----------------------------------------------------------------------------------------------------------------------          
    
def build_heatmap(in_tensor, config, names = None):
  
    num_detections  = config.DETECTION_MAX_INSTANCES
    img_h, img_w    = config.IMAGE_SHAPE[:2]
    batch_size      = config.BATCH_SIZE
    num_classes     = config.NUM_CLASSES  
    # rois per image is determined by size of input tensor 
    #   detection mode:   config.TRAIN_ROIS_PER_IMAGE 
    #   ground_truth  :   config.DETECTION_MAX_INSTANCES
    # strt_cls        = 0 if rois_per_image == 32 else 1
    rois_per_image  = (in_tensor.shape)[2]  
    print('\n ')
    print('  > NEW build_heatmap() for ', names )
    print('    in_tensor shape        : ', in_tensor.shape)       
    print('    num bboxes per class   : ', rois_per_image )

    #-----------------------------------------------------------------------------    
    ## Stack non_zero bboxes from in_tensor into pt2_dense 
    #-----------------------------------------------------------------------------
    # pt2_ind shape is [?, 3]. 
    #   pt2_ind[0] corresponds to image_index 
    #   pt2_ind[1] corresponds to class_index 
    #   pt2_ind[2] corresponds to roi row_index 
    # pt2_dense shape is [?, 6]
    #    pt2_dense[0] is image index
    #    pt2_dense[1:4]  roi cooridnaytes 
    #    pt2_dense[5]    is class id 
    #-----------------------------------------------------------------------------
    pt2_sum = tf.reduce_sum(tf.abs(in_tensor[:,:,:,:-2]), axis=-1)
    print('    pt2_sum shape ',pt2_sum.shape)
    # print(pt2_sum[0].eval())
    pt2_ind = tf.where(pt2_sum > 0)
    pt2_dense = tf.gather_nd( in_tensor, pt2_ind)
    print('    dense shape ',pt2_dense.get_shape())

    #-----------------------------------------------------------------------------
    ## Build mesh-grid to hold pixel coordinates  
    #-----------------------------------------------------------------------------
    X = tf.range(img_w, dtype=tf.int32)
    Y = tf.range(img_h, dtype=tf.int32)
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
    bef_pos = tf.to_float(tf.stack([rep_X,rep_Y], axis = -1))
    print('    before transpse ', bef_pos.get_shape())
    pos_grid = tf.transpose(bef_pos,[1,2,0,3])
    print('    after transpose ', pos_grid.get_shape())    

    ##-----------------------------------------------------------------------------
    ##  Build mean and convariance tensors for Multivariate Normal Distribution 
    ##-----------------------------------------------------------------------------
    width  = pt2_dense[:,3] - pt2_dense[:,1]      # x2 - x1
    height = pt2_dense[:,2] - pt2_dense[:,0]
    cx     = pt2_dense[:,1] + ( width  / 2.0)
    cy     = pt2_dense[:,0] + ( height / 2.0)
    means  = tf.stack((cx,cy),axis = -1)
    covar  = tf.stack((width * 0.5 , height * 0.5), axis = -1)
    covar  = tf.sqrt(covar)

    ##-----------------------------------------------------------------------------
    ##  Compute Normal Distribution for bounding boxes
    ##-----------------------------------------------------------------------------    
    tfd = tf.contrib.distributions
    mvn = tfd.MultivariateNormalDiag( loc  = means,  scale_diag = covar)
    prob_grid = mvn.prob(pos_grid)
    print('     Prob_grid shape before tanspose: ',prob_grid.get_shape())
    prob_grid = tf.transpose(prob_grid,[2,0,1])
    print('     Prob_grid shape after tanspose: ',prob_grid.get_shape())    
    print('    >> input to MVN.PROB: pos_grid (meshgrid) shape: ', pos_grid.get_shape())
    print('    << output probabilities shape:' , prob_grid.get_shape())

    ##--------------------------------------------------------------------------------
    ## IMPORTANT: kill distributions of NaN boxes (resulting from bboxes with height/width of zero
    ## which cause singular sigma cov matrices
    ##--------------------------------------------------------------------------------
    prob_grid = tf.where(tf.is_nan(prob_grid),  tf.zeros_like(prob_grid), prob_grid)
    

    ##-------------------------------------------------------------------------------------
    ## scatter out the probability distributions based on class 
    ##-------------------------------------------------------------------------------------
    print('\n    Scatter out the probability distributions based on class --------------') 
    gauss_scatt   = tf.scatter_nd(pt2_ind, prob_grid, [batch_size, num_classes, rois_per_image, img_w, img_h], name = 'gauss_scatter')
    print('    pt2_ind shape   : ', pt2_ind.shape)  
    print('    prob_grid shape : ', prob_grid.shape)  
    print('    gauss_scatt     : ', gauss_scatt.shape)   # batch_sz , num_classes, num_rois, image_h, image_w
    
    ##-------------------------------------------------------------------------------------
    ## SUM : Reduce and sum up gauss_scattered by class  
    ##-------------------------------------------------------------------------------------
    print('\n    Reduce sum based on class ---------------------------------------------')         
    gauss_heatmap = tf.reduce_sum(gauss_scatt, axis=2, name='pred_heatmap2')
    # force small sums to zero - for now (09-11-18) commented out but could reintroduce based on test results
    # gauss_heatmap = tf.where(gauss_heatmap < 1e-12, gauss_heatmap, tf.zeros_like(gauss_heatmap), name='Where1')
    print('    gaussian_sum shape     : ', gauss_heatmap.get_shape(), 'Keras tensor ', KB.is_keras_tensor(gauss_heatmap) )      
    
    #---------------------------------------------------------------------------------------------
    # heatmap L2 normalization
    # Normalization using the  `gauss_heatmap` (batchsize , num_classes, height, width) 
    # 17-05-2018 (New method, replace dthe previous method that usedthe transposed gauss sum
    # 17-05-2018 Replaced with normalization across the CLASS axis 
    #---------------------------------------------------------------------------------------------
    # print('\n    L2 normalization ------------------------------------------------------')   
    # gauss_L2norm   = KB.l2_normalize(gauss_heatmap, axis = +1)   # normalize along the CLASS axis 
    # print('    gauss L2 norm   : ', gauss_L2norm.shape   ,' Keras tensor ', KB.is_keras_tensor(gauss_L2norm) )
    #---------------------------------------------------------------------------------------------

    ##---------------------------------------------------------------------------------------------
    ## heatmap normalization
    ## normalizer is set to one when the max of class is zero     
    ## this prevents elements of gauss_heatmap_norm computing to nan
    ##---------------------------------------------------------------------------------------------
    print('\n    normalization ------------------------------------------------------')   
    normalizer = tf.reduce_max(gauss_heatmap, axis=[-2,-1], keepdims = True)
    normalizer = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    gauss_heatmap_norm = gauss_heatmap / normalizer
    # gauss_heatmap_norm    = gauss_heatmap / tf.reduce_max(gauss_heatmap, axis=[-2,-1], keepdims = True)
    # gauss_heatmap_norm    = tf.where(tf.is_nan(gauss_heatmap_norm),  tf.zeros_like(gauss_heatmap_norm), gauss_heatmap_norm, name = 'Where2')
    print('    gauss norm            : ', gauss_heatmap_norm.shape   ,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap_norm) )

    ##--------------------------------------------------------------------------------------------
    ##  Generate scores using prob_grid and pt2_dense - NEW METHOD
    ##  added 09-21-2018
    ##--------------------------------------------------------------------------------------------
    scores_from_sum2 = tf.map_fn(build_hm_score, [prob_grid, pt2_dense], dtype = tf.float32, swap_memory = True)
    scores_scattered = tf.scatter_nd(pt2_ind, scores_from_sum2, [batch_size, num_classes, rois_per_image, 3], name = 'scores_scattered')
    gauss_scores = tf.concat([in_tensor, scores_scattered], axis = -1,name = names[0]+'_scores')
    print('    scores_scattered shape : ', scores_scattered.shape) 
    print('    gauss_scores           : ', gauss_scores.shape, ' Name:   ', gauss_scores.name)
    print('    gauss_scores  (FINAL)  : ', gauss_scores.shape, ' Keras tensor ', KB.is_keras_tensor(gauss_scores) )      
    
    ##--------------------------------------------------------------------------------------------
    ##   Normalization is already perfored on the scores at a per_class leve, so we dont use this 
    ##  code below anympre
    ##
    ##  This is a regular normalization that moves everything between [0, 1]. 
    ##  This causes negative values to move to -inf, which is a problem in FCN scoring. 
    ##  To address this a normalization between [-1 and +1] was introduced in FCN.
    ##  Not sure how this will work with training tho.
    ##--------------------------------------------------------------------------------------------
    #     normalizer   = tf.reduce_max(scores_scatt[...,-1], axis = -1, keepdims=True)
    #     print('norm',normalizer.shape)
    #     normalizer   = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    #     norm_score2   = tf.expand_dims(scores_scatt[...,-1]/normalizer, axis = -1)
    #     print('norm_SCORE2',norm_score2.shape)
    
    
    #-------------------------------------------------------------------------------------------------------------------
    #  Generate scores using GAUSS_SUM -- OLD METHOD
    #  removed 09-21-2018
    #-------------------------------------------------------------------------------------------------------------------
    #   Generate scores : 
    #   -----------------
    #  NOTE: Score is generated on NORMALIZED gaussian distributions (GAUSS_NORM)
    #        If want to do this on NON-NORMALIZED, we need to apply it on GAUSS_SUM
    #        Testing demonstated that the NORMALIZED score generated from using GAUSS_SUM 
    #        and GAUSS_NORM are the same. 
    #        For now we will use GAUSS_SUM score and GAUSS_NORM heatmap. The reason being that 
    #        the raw score generated in GAUSS_SUM is much smaller. 
    #        We may need to change this base on the training results from FCN 
    #---------------------------------------------------------------------------------------------
    #   duplicate GAUSS_NORM <num_roi> times to pass along with bboxes to map_fn function
    # 
    #   Here we have a choice to calculate scores using the GAUSS_SUM (unnormalized) or GAUSS_NORM (normalized)
    #   after looking at the scores and ratios for each option, I decided to go with the normalized 
    #   as the numbers are larger
    #
    #   Examples>
    #   Using GAUSS_SUM
    # [   3.660313    3.513489   54.475536   52.747402    1.   0.999997    4.998889 2450.          0.00204     0.444867]
    # [   7.135149    1.310972   50.020126   44.779854    1.   0.999991    4.981591 1892.          0.002633    0.574077]
    # [  13.401865    0.         62.258957   46.636948    1.   0.999971    4.957398 2303.          0.002153    0.469335]
    # [   0.          0.         66.42349    56.123024    1.   0.999908    4.999996 3696.          0.001353    0.294958]
    # [   0.          0.         40.78952    60.404335    1.   0.999833    4.586552 2460.          0.001864    0.406513]    
    #                                                       
    #   Using GAUSS_NORM:                             class   r-cnn scr   
    # [   3.660313    3.513489   54.475536   52.747402    1.   0.999997 1832.9218   2450.          0.748131    0.479411]
    # [   7.135149    1.310972   50.020126   44.779854    1.   0.999991 1659.3965   1892.          0.877059    0.56203 ]
    # [  13.401865    0.         62.258957   46.636948    1.   0.999971 1540.4974   2303.          0.668909    0.428645]
    # [   0.          0.         66.42349    56.123024    1.   0.999908 1925.3267   3696.          0.520922    0.333813]
    # [   0.          0.         40.78952    60.404335    1.   0.999833 1531.321    2460.          0.622488    0.398898]
    # 
    #  to change the source, change the following line gauss_heatmap_norm <--> gauss_heatmap
    #---------------------------------------------------------------------------------------------------------------------------
    # flatten guassian scattered and input_tensor, and pass on to build_bbox_score routine 
    # in_shape = tf.shape(in_tensor)
    # print('    shape of in_tensor is : ', KB.int_shape(in_tensor))
    # in_tensor_flattened  = tf.reshape(in_tensor, [-1, in_shape[-1]])  <-- not a good reshape style!! 
    # replaced with following line:
    # in_tensor_flattened  = tf.reshape(in_tensor, [-1, in_tensor.shape[-1]])
    #
    #  bboxes = tf.to_int32(tf.round(in_tensor_flattened[...,0:4]))
    #
    # print('    in_tensor             : ', in_tensor.shape)
    # print('    in_tensor_flattened   : ', in_tensor_flattened.shape)
    # print('    Rois per class        : ', rois_per_image)
    #
    #     print('\n    Scores from gauss_heatmap ----------------------------------------------')
    #     temp = tf.expand_dims(gauss_heatmap, axis =2)
    #     print('    temp expanded          : ', temp.shape)
    #     temp = tf.tile(temp, [1,1, rois_per_image ,1,1])
    #     print('    temp tiled shape       : ', temp.shape)
    # 
    #     temp = KB.reshape(temp, (-1, temp.shape[-2], temp.shape[-1]))
    #     
    #     print('    temp flattened         : ', temp.shape)
    #     print('    in_tensor_flattened    : ', in_tensor_flattened.shape)
    # 
    #     scores_from_sum = tf.map_fn(build_hm_score, [temp, in_tensor_flattened], dtype=tf.float32)
    #     scores_shape    = [in_tensor.shape[0], in_tensor.shape[1], in_tensor.shape[2], -1]
    #     scores_from_sum = tf.reshape(scores_from_sum, scores_shape)    
    #     print('    reshaped scores        : ', scores_from_sum.shape)
    #--------------------------------------------------------------------------------------------
    #  tf.reduce_max(scores_from_sum[...,-1], axis = -1, keepdims=True) result is [num_imgs, num_class, 1]
    #
    #  This is a regular normalization that moves everything between [0, 1]. 
    #  This causes negative values to move to -inf, which is a problem in FCN scoring. 
    #  To address this a normalization between [-1 and +1] was introduced in FCN.
    #  Not sure how this will work with training tho.
    #--------------------------------------------------------------------------------------------
    #     normalizer   = tf.reduce_max(scores_from_sum[...,-1], axis = -1, keepdims=True)
    #     normalizer   = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    #     norm_score   = tf.expand_dims(scores_from_sum[...,-1]/normalizer, axis = -1)
    #--------------------------------------------------------------------------------------------
    # Append `in_tensor` and `scores_from_sum` to form `bbox_scores`
    #--------------------------------------------------------------------------------------------
    #     gauss_scores = tf.concat([in_tensor, scores_from_sum, norm_score], axis = -1,name = names[0]+'_scores')
    #     print('    scores_from_sum final  : ', scores_from_sum.shape)    
    #     print('    norm_score             : ', norm_score.shape)
    #     print('    gauss_scores           : ', gauss_scores.shape,  '   name:   ', gauss_scores.name)
    #     print('    gauss_scores  (FINAL)  : ', gauss_scores.shape, ' Keras tensor ', KB.is_keras_tensor(gauss_scores) )    
    #--------------------------------------------------------------------------------------------------------------------

    
    ##--------------------------------------------------------------------------------------------
    ## //create heatmap Append `in_tensor` and `scores_from_sum` to form `bbox_scores`
    ##--------------------------------------------------------------------------------------------
    # gauss_heatmap  = tf.transpose(gauss_heatmap,[0,2,3,1], name = names[0])
    gauss_heatmap_norm = tf.transpose(gauss_heatmap_norm,[0,2,3,1], name = names[0]+'_norm')
    
    # print('    gauss_heatmap       shape : ', gauss_heatmap.shape     ,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap) )  
    # print('    gauss_heatmap_norm  shape : ', gauss_heatmap_norm.shape,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap_norm) )  
    
    # gauss_heatmap_norm   = KB.identity(tf.transpose(gauss_heatmap_norm,[0,2,3,1]), name = names[0]+'_norm')
    # print('    gauss_heatmap_norm final shape : ', gauss_heatmap_norm.shape,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap_norm) )  
    # gauss_heatmap_L2norm = KB.identity(tf.transpose(gauss_L2norm,[0,2,3,1]), name = names[0]+'_L2norm')
 
    print('    complete')
    return   gauss_heatmap_norm, gauss_scores  # , gauss_heatmap   gauss_heatmap_L2norm    # [gauss_heatmap, gauss_scatt, means, covar]    

 
"""