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
import mrcnn.utils as utils
import tensorflow.contrib.util as tfc
import pprint
from   mrcnn.utils       import logt
from   mrcnn.chm_layer   import build_hm_score_v2, build_hm_score_v3, clip_heatmap, normalize_scores

##-----------------------------------------------------------------------------------------------------------
## 
##-----------------------------------------------------------------------------------------------------------
def build_gt_tensor(gt_class_ids, norm_gt_bboxes, config):
    verbose         = config.VERBOSE
    batch_size      = config.BATCH_SIZE
    num_classes     = config.NUM_CLASSES
    h, w            = config.IMAGE_SHAPE[:2]
    det_per_class   = config.DETECTION_PER_CLASS
    num_bboxes      = KB.int_shape(norm_gt_bboxes)[1]

    scale           = tf.constant([h,w,h,w], dtype = tf.float32)
    # dup_scale       = tf.reshape(tf.tile(scale, [num_rois]),[num_rois,-1])
    dup_scale       = scale * tf.ones([batch_size, num_bboxes, 1], dtype = 'float32')
    gt_bboxes       = tf.multiply(norm_gt_bboxes , dup_scale )
 
    # num of bounding boxes is determined by bbox_list.shape[1] instead of config.DETECTION_MAX_INSTANCES
    # use of this routine for both input_gt_boxes, and target_gt_deltas
    if  num_bboxes == config.DETECTION_MAX_INSTANCES:
        tensor_name = "gt_tensor_max"
    else:
        tensor_name = "gt_tensor"
        
    if verbose:
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
    logt('gt_classes_exp ', gt_classes_exp, verbose = verbose)

    ones = tf.ones_like(gt_class_ids)
    zeros= tf.zeros_like(gt_class_ids)
    mask = tf.greater(gt_class_ids , 0)

    gt_scores     = tf.where(mask, ones, zeros)
    gt_scores_exp = tf.to_float(KB.expand_dims(gt_scores, axis=-1))
    logt('gt_scores_exp  ', gt_scores_exp, verbose = verbose)

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
    
    logt('gt_array    ', gt_array   , verbose = verbose)
    logt('scatter_ind ', scatter_ind, verbose = verbose)
    logt('gt_array    ', gt_array   , verbose = verbose)
    logt('gt_scatter  ', gt_scatter , verbose = verbose)
    
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
    gt_tensor   = tf.gather_nd(gt_scatter, gather_inds , name = tensor_name)
    # append an index to the end of each row --- commented out 30-04-2018
    # gt_tensor = tf.concat([gt_tensor, bbox_grid_exp], axis = -1)
    logt('sort_inds   ', sort_inds   , verbose = verbose)
    logt('class_grid  ', class_grid  , verbose = verbose)
    logt('batch_grid  ', batch_grid  , verbose = verbose)
    logt('gather_inds ', gather_inds , verbose = verbose)
    logt('gt_tensor   ', gt_tensor   , verbose = verbose)

    return  gt_tensor 

##-----------------------------------------------------------------------------------------------------------
##  build_gt_heatmap : Build Ground Truth heatmaps using pred_gt_tensor
##------------------------------------------------------------------------------------------------------------
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
##   gauss_heatmap_norm
##   gauss_scores - [BATCH_SIZE, NUM_CLASSES, DETECTIONS_PER_CLASS, 11]
##                  same as in_tensor, adding :
##  TODO  update these for Ground_Truth function 
##                  - sum of heatmap in masked area
##                  - area of bounding box in pixes
##                  - (sum of heatmap in masked area) * (bbox per-class normalized score from in_tensor)
##------------------------------------------------------------------------------------------------------------
def build_gt_heatmap(in_tensor, config, names = None):
    verbose         = config.VERBOSE
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

    if verbose:
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

    logt('pt2_sum   ', pt2_sum, verbose = verbose)
    logt('pt2_ind   ', pt2_ind, verbose = verbose)
    logt('pt2_dense ', pt2_dense, verbose = verbose)

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
    logt('Prob_grid  ', prob_grid, verbose = verbose)    
    
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
    # (2) multiply normalized heatmap by normalized score in i  n_tensor/ (pt2_dense column 7)
    #     broadcasting : https://stackoverflow.com/questions/49705831/automatic-broadcasting-in-tensorflow
    #---------------------------------------------------------------------------------------------    
    # prob_grid_norm_scaled = tf.transpose(tf.transpose(prob_grid_norm) * pt2_dense[:,7])
    # print('    prob_grid_norm_scaled : ', prob_grid_norm_scaled.shape)
    
    ##---------------------------------------------------------------------------------------------
    ## (NEW STEP) Clip heatmap to region surrounding Cy,Cx and Covar X, Y 
    ##---------------------------------------------------------------------------------------------        
    prob_grid_clipped = tf.map_fn(clip_heatmap, [prob_grid, cy,cx, covar], 
                                 dtype = tf.float32, swap_memory = True)
    logt('prob_grid_clipped ', prob_grid_clipped, verbose = verbose) 
    
    ##--------------------------------------------------------------------------------------------
    ## (0) Generate scores using prob_grid and pt2_dense - (NEW METHOD added 09-21-2018)
    ##  pt2_dense[:,7] is the per-class-normalized score from in_tensor
    ##
    ## 11-27-2018: (note - here, build_hm_score_v2 is being applied to prob_grid_clipped, 
    ## unlilke chm_layer) - Changed to prob_grid to make it consistent with chm_layer.py
    ##
    ## When using prob_grid:
    ## [ 1.0000     1.0000   138.0000     1.0000  4615.0000  4531.1250  4615.0000 
    ## [ 3.0000     1.0000   179.0000     1.0000   570.0000   547.5000   570.0000 
    ## 
    ## When using prob_grid_clipped:
    ## [ 1.0000     1.0000   138.0000     1.0000   144.0000  4531.1250   144.0000         
    ## [ 3.0000     1.0000   179.0000     1.0000    56.0000   547.5000    56.0000 
    ##--------------------------------------------------------------------------------------------
    old_style_scores = tf.map_fn(build_hm_score_v2, [prob_grid, pt2_dense_scaled, pt2_dense[:,7]], 
                                 dtype = tf.float32, swap_memory = True)
    old_style_scores = tf.scatter_nd(pt2_ind, old_style_scores, 
                                     [batch_size, num_classes, rois_per_image, 3], name = 'scores_scattered')
    logt('old_style_scores ', old_style_scores, verbose = verbose)
    
     
    ##---------------------------------------------------------------------------------------------
    ## - Build alternative scores based on normalized/scaled/clipped heatmap
    ##---------------------------------------------------------------------------------------------
    alt_scores_1 = tf.map_fn(build_hm_score_v3, [prob_grid_clipped, cy, cx,covar], dtype=tf.float32)    
    logt('alt_scores_1    ', alt_scores_1, verbose = verbose)
    alt_scores_1 = tf.scatter_nd(pt2_ind, alt_scores_1, 
                                 [batch_size, num_classes, rois_per_image, KB.int_shape(alt_scores_1)[-1]],
                                 name = 'alt_scores_1')  

    alt_scores_1_norm = normalize_scores(alt_scores_1)
    logt('alt_scores_1(by class)      ', alt_scores_1, verbose = verbose)
    logt('alt_scores_1_norm(by_class) ', alt_scores_1_norm, verbose = verbose)
                                     
    ##-------------------------------------------------------------------------------------
    ## (3) scatter out the probability distribution heatmaps based on class 
    ##-------------------------------------------------------------------------------------
    gauss_heatmap   = tf.scatter_nd(pt2_ind, prob_grid_clipped, 
                                  [batch_size, num_classes, rois_per_image, grid_w, grid_h], 
                                  name = 'gauss_heatmap')
    logt('\n    Scatter out the probability distributions based on class --------------') 
    logt('pt2_ind       ', pt2_ind, verbose = verbose)
    logt('prob_grid     ', prob_grid, verbose = verbose)
    logt('gauss_heatmap ', gauss_heatmap, verbose = verbose)   # batch_sz , num_classes, num_rois, image_h, image_w

    ##-------------------------------------------------------------------------------------
    ## (4) MAX : Reduce_MAX up gauss_heatmaps by class 
    ##           Since all values are set to '1' in the 'heatmap', there is no need to 
    ##           sum or normalize. We Reduce_max on the class axis, and as a result the 
    ##           correspoding areas in the heatmap are set to '1'
    ##-------------------------------------------------------------------------------------
    gauss_heatmap = tf.reduce_max(gauss_heatmap, axis=2, name='gauss_heatmap')
    logt('\n    Reduce MAX based on class -------------------------------------', verbose = verbose)
    logt(' gaussian_heatmap : ', gauss_heatmap, verbose = verbose)
    
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
    
    ##---------------------------------------------------------------------------------------------
    ##  build indices and extract heatmaps corresponding to each bounding boxes' class id
    ##  build alternative scores#  based on normalized/sclaked clipped heatmap
    ##---------------------------------------------------------------------------------------------
    hm_indices = tf.cast(pt2_ind[:, :2],dtype=tf.int32)
    pt2_heatmaps = tf.gather_nd(gauss_heatmap, hm_indices )
    logt('hm_indices   ',  hm_indices, verbose = verbose)
    logt('pt2_heatmaps ',  pt2_heatmaps, verbose = verbose)

    alt_scores_2 = tf.map_fn(build_hm_score_v3, [pt2_heatmaps, cy, cx,covar], dtype=tf.float32)    
    logt('alt_scores_2  ', alt_scores_2, verbose = verbose)

    alt_scores_2 = tf.scatter_nd(pt2_ind, alt_scores_2, 
                                     [batch_size, num_classes, rois_per_image, KB.int_shape(alt_scores_2)[-1]], name = 'alt_scores_2')  
    
    alt_scores_2_norm = normalize_scores(alt_scores_2)
    logt('alt_scores_2(by class)       : ', alt_scores_2, verbose = verbose)
    logt('alt_scores_2_norm(by_class)  : ', alt_scores_2_norm, verbose = verbose)
    
        
    ##--------------------------------------------------------------------------------------------
    ##  Transpose tensor to [BatchSz, Height, Width, Num_Classes]
    ##--------------------------------------------------------------------------------------------
    gauss_heatmap  = tf.transpose(gauss_heatmap,[0,2,3,1], name = names[0])
    
    # gauss_heatmap_norm = tf.transpose(gauss_heatmap_norm,[0,2,3,1], name = names[0]+'_norm')
    # print('    gauss_heatmap_norm : ', gauss_heatmap_norm.shape,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap_norm) )
    # print('    complete')

    ##--------------------------------------------------------------------------------------------
    ## APPEND ALL SCORES TO input score tensor TO YIELD output scores tensor
    ##--------------------------------------------------------------------------------------------
    gauss_scores     = tf.concat([in_tensor, old_style_scores, alt_scores_1, alt_scores_1_norm, alt_scores_2, alt_scores_2_norm],
                                    axis = -1,name = names[0]+'_scores')
    #                                 alt_scores_2[...,:3], alt_scores_3],
    logt('gauss_heatmap  ', gauss_heatmap, verbose = verbose)
    logt('gauss_scores', gauss_scores, verbose = verbose)
    logt('complete    ', verbose = verbose)
        

    return   gauss_heatmap, gauss_scores  

        
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
        verbose         = self.config.VERBOSE

        tgt_class_ids, tgt_bboxes = inputs
        logt('  > CHMLayerTgt Call()   :', inputs, verbose = verbose)
        logt('    tgt_class_ids.shape  :', tgt_class_ids, verbose = verbose)
        logt('    tgt_bboxes.shape     :',    tgt_bboxes, verbose = verbose)
         
        gt_tensor     = build_gt_tensor (tgt_class_ids,  tgt_bboxes, self.config)  
        gt_hm, gt_hm_scores  = build_gt_heatmap(gt_tensor, self.config, names = ['gt_heatmap'])
        # gt_cls_cnt   = KL.Lambda(lambda x: tf.count_nonzero(x[:,:,:,-1],axis = -1), name = 'gt_cls_count')(gt_tensor)

        logt(' ', verbose = verbose)
        logt('gt_heatmap        ', gt_hm, verbose = verbose)
        logt('gt_heatmap_scores ', gt_hm_scores, verbose = verbose)
        logt('complete', verbose = verbose)
        
        return [ gt_hm, gt_hm_scores]

         
    def compute_output_shape(self, input_shape):
        # may need to change dimensions of first return from IMAGE_SHAPE to MAX_DIM
        return [
                 (None, self.config.IMAGE_SHAPE[0], self.config.IMAGE_SHAPE[1], self.config.NUM_CLASSES)  # gt_heatmap
              ,  (None, self.config.NUM_CLASSES   , self.config.TRAIN_ROIS_PER_IMAGE , 23)                  # gt_heatmap_scores   
              ]
