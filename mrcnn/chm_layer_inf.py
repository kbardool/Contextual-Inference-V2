"""
Mask R-CNN
Contextual Heatmap Layer for Inference Mode 

Copyright (c) 2018 K.Bardool 
Licensed under the MIT License (see LICENSE for details)
Written by Kevin Bardool
"""

import os, sys, glob, random, math, datetime, itertools, json, re, logging
# from collections import OrderedDict
# from scipy.stats import  multivariate_normal
import numpy as np
import tensorflow as tf
import keras.backend as KB
import keras.layers as KL
import keras.engine as KE
import mrcnn.utils as utils
import tensorflow.contrib.util as tfc
import pprint
from mrcnn.chm_layer import build_hm_score_v2, build_hm_score_v3, clip_heatmap, normalize_scores
              
##-----------------------------------------------------------------------------------------------------------
## build_predictions 
##-----------------------------------------------------------------------------------------------------------              
def build_predictions_inference(detected_rois, config):  
    '''
    input: 
        detected_rois:   (None, 200, 7)    
                         [batchSz, Detection_Max_instance, (y1,x1,y2,x2, class, score, det_type)]
    output:
        pred_tensor      (None, nu_classes, 200, 9)
                         [batchSz, Detection_Max_instance, (y1,x1,y2,x2, class, score, det_type, sequence_id, normalized_score)]
    '''
    batch_size      = config.BATCH_SIZE
    num_classes     = config.NUM_CLASSES
    h, w            = config.IMAGE_SHAPE[:2]
    CLASS_COLUMN    = 4
    SCORE_COLUMN    = 5
    DT_TYPE_COLUMN  = 6
    SEQUENCE_COLUMN = 7
    NORM_SCORE_COLUMN = 8
    
    # num_rois        = config.DETECTION_MAX_INSTANCES 
    num_rois        = KB.int_shape(detected_rois)[1]
    num_cols        = KB.int_shape(detected_rois)[-1]
    det_per_class   = config.DETECTION_PER_CLASS

    print()
    print('  > build_predictions Inference mode ()')
    print('    config image shape     : ', config.IMAGE_SHAPE, 'h:',h,'w:',w)
    print('    Detection Max Instacnes: ', config.DETECTION_MAX_INSTANCES)
    print('    num_rois               : ', num_rois )
    print('    num_cols               : ', num_cols )
    print('    detected_rois.shape    : ', KB.int_shape(detected_rois))

    ##-------------------------------------------------------------------------------------
    ## Build a meshgrid for image id and bbox to use in gathering of bbox delta information 
    ##   indexing = 'ij' provides matrix indexing conventions
    ##-------------------------------------------------------------------------------------
    batch_grid, bbox_grid = tf.meshgrid( tf.range(batch_size, dtype=tf.int32),
                                         tf.range(num_rois, dtype=tf.int32), indexing = 'ij' )
    print('    batch_grid: ', KB.int_shape(batch_grid))
    print('    bbox_grid : ', KB.int_shape(bbox_grid))
    # print( batch_grid.eval())
    # print( bbox_grid.eval())

    #---------------------------------------------------------------------------
    # column -2 contains the prediceted class 
    #  (NOT USED)   pred_classes_exp = tf.to_float(tf.expand_dims(pred_classes ,axis=-1))    
    #---------------------------------------------------------------------------
    pred_classes = tf.to_int32(detected_rois[..., CLASS_COLUMN])
    # print(pred_classes.eval())

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
    pred_array  = tf.concat([ detected_rois, sequence], axis=-1, name = 'pred_array')
    
    scatter_ind = tf.stack([batch_grid , pred_classes, bbox_grid],axis = -1)
    pred_scatt  = tf.scatter_nd(scatter_ind, pred_array, [batch_size, num_classes, num_rois, num_cols + 1])
    print('    pred_array             : ', pred_array.shape)  
    print('    scatter_ind            : ', scatter_ind.shape)
    print('    pred_scatter           : ', pred_scatt.shape)

    ##--------------------------------------------------------------------------------------------
    ##  Apply a per class score normalization using the score column (COLUMN 5)
    ##  
    ##--------------------------------------------------------------------------------------------
    normalizer   = tf.reduce_max(pred_scatt[..., SCORE_COLUMN], axis = -1, keepdims=True)
    normalizer   = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    norm_score   = tf.expand_dims(pred_scatt[..., SCORE_COLUMN]/normalizer, axis = -1)
    pred_scatt   = tf.concat([pred_scatt, norm_score],axis = -1)   
    print('    - Add normalized score --\n')
    print('    normalizer             : ', normalizer.shape)  
    print('    norm_score             : ', norm_score.shape)
    print('    pred_scatter           : ', pred_scatt.get_shape())

    ##------------------------------------------------------------------------------------
    ## Sort pred_scatt in each class dimension based on sequence number, to push valid  
    ##      to top for each class dimension
    ##
    ## 22-09-2018: sort is now based on sequence which was added as last column
    ##             (previously sort was on bbox scores)
    ##------------------------------------------------------------------------------------
    _, sort_inds = tf.nn.top_k(pred_scatt[..., SEQUENCE_COLUMN], k=pred_scatt.shape[2])
    
    # build indexes to gather rows from pred_scatter based on sort order    
    class_grid, batch_grid, roi_grid = tf.meshgrid(tf.range(num_classes),tf.range(batch_size), tf.range(num_rois))
    
    gather_inds  = tf.stack([batch_grid , class_grid, sort_inds],axis = -1)
    
    ##
    # 12-02-2018
    # difference between config.DETECTION_PER_CLASS and config.DETECTION_MAX_INSTANCES causes
    # problems when converting arrays from <by_class> to <by Image>. As a result commented the following line
    # out gather_inds[...,:det_per_class,:]  and replaced with gather_inds in next line
    # pred_tensor  = tf.gather_nd(pred_scatt, gather_inds[...,:det_per_class,:], name = 'pred_tensor')    
    pred_tensor  = tf.gather_nd(pred_scatt, gather_inds, name = 'pred_tensor')    

    print('    sort_inds              : ', KB.int_shape(sort_inds)   , 'Keras Tensor:', KB.is_keras_tensor(sort_inds))
    print('    class_grid             : ', KB.int_shape(class_grid)  , 'Keras Tensor:', KB.is_keras_tensor(class_grid))
    print('    batch_grid             : ', KB.int_shape(batch_grid)  , 'Keras Tensor:', KB.is_keras_tensor(batch_grid))
    print('    roi_grid shape         : ', KB.int_shape(roi_grid)    , 'Keras Tensor:', KB.is_keras_tensor(roi_grid)) 
    print('    gather_inds            : ', KB.int_shape(gather_inds) , 'Keras Tensor:', KB.is_keras_tensor(gather_inds))
    print('    pred_tensor            : ', KB.int_shape(pred_tensor) , 'Keras Tensor:', KB.is_keras_tensor(pred_tensor))
    
    return  pred_tensor    
            
              
##-----------------------------------------------------------------------------------------------------
##  build_heatmap_inference()
##
##  INPUTS :
##    pred_tensor        [ batch_size, num_classes, num_bboxes, 7 ] 
##
##  rois_per_image is determined by size of input tensor is one of the following: 
##    training/trainfcn mode :   config.TRAIN_ROIS_PER_IMAGE 
##    inference mode         :   config.DETECTION_MAX_INSTANCES
##    
##-----------------------------------------------------------------------------------------------------          
def build_heatmap_inference(in_tensor, config, names = None):
    '''
    input:
        pred_tensor      (None, num_classes, 200, 9)
                        [batchSz, Detection_Max_instance, (y1,x1,y2,x2, class, score, det_type, sequence_id, normalized_score)]
                         
    output:
        pr_heatmap      (None,  Heatmap-height, Heatmap_width, num_classes)
        pr_scores       (None, num_classes, 200, 24) 
                        [batchSz, Detection_Max_instance, (y1,x1,y2,x2, class, score, det_type, sequence_id, normalized_score,
                                                           scores-0: gaussian_sum, bbox_area, weighted_norm_sum 
                                                           scores-1: score, mask_sum, score/mask_sum, (score, mask_sum, score/mask_sum) normalized by class
                                                           scores-2: score, mask_sum, score/mask_sum, (score, mask_sum, score/mask_sum) normalized by class ]
    '''

    num_detections    = config.DETECTION_MAX_INSTANCES
    img_h, img_w      = config.IMAGE_SHAPE[:2]
    batch_size        = config.BATCH_SIZE
    num_classes       = config.NUM_CLASSES 
    heatmap_scale     = config.HEATMAP_SCALE_FACTOR
    grid_h, grid_w    = config.IMAGE_SHAPE[:2] // heatmap_scale    
    # rois_per_image  = config.DETECTION_PER_CLASS
    rois_per_image    = (in_tensor.shape)[2]  
    CLASS_COLUMN      = 4
    SCORE_COLUMN      = 5
    DT_TYPE_COLUMN    = 6
    SEQUENCE_COLUMN   = 7
    NORM_SCORE_COLUMN = 8

    print('\n ')
    print('  > build_inference_heatmap() for ', names )
    print('    in_tensor shape        : ', in_tensor.shape)       
    print('    num bboxes per class   : ', rois_per_image )
    print('    heatmap scale        : ', heatmap_scale, 'Dimensions:  w:', grid_w,' h:', grid_h)

    ##-----------------------------------------------------------------------------    
    ## Stack non_zero bboxes from in_tensor into pt2_dense 
    ##-----------------------------------------------------------------------------
    # pt2_ind shape is [?, 3].                    pt2_dense shape is [?, 7]
    #    pt2_ind[0] corresponds to image_index       pt2_dense[0:3]  roi coordinates 
    #    pt2_ind[1] corresponds to class_index       pt2_dense[4]    class id 
    #    pt2_ind[2] corresponds to roi row_index     pt2_dense[5]    score from mrcnn    
    #                                                pt2_dense[6]    bbox sequence id    
    #                                                pt2_dense[7]    per-class normalized score 
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
    print('    >> input to MVN.PROB: pos_grid (meshgrid) shape: ', pos_grid.shape)
    print('     Prob_grid shape from mvn.probe: ',prob_grid.shape)
    prob_grid = tf.transpose(prob_grid,[2,0,1])
    print('     Prob_grid shape after tanspose: ',prob_grid.shape)    
    print('    << output probabilities shape  : ' , prob_grid.shape)
    
    ##--------------------------------------------------------------------------------------------
    ## (0) Generate scores using prob_grid and pt2_dense - (NEW METHOD added 09-21-2018)
    ##--------------------------------------------------------------------------------------------
    old_style_scores = tf.map_fn(build_hm_score_v2, [prob_grid, bboxes_scaled, pt2_dense[:, NORM_SCORE_COLUMN]], 
                                 dtype = tf.float32, swap_memory = True)
    old_style_scores = tf.scatter_nd(pt2_ind, old_style_scores, 
                                     [batch_size, num_classes, rois_per_image, KB.int_shape(old_style_scores)[-1]],
                                     name = 'scores_scattered')
    print('    old_style_scores        :',  old_style_scores.get_shape(), KB.int_shape(old_style_scores))      

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
    print('    prob_grid_clipped : ', prob_grid_clipped.shape)


    ##---------------------------------------------------------------------------------------------
    ## (2) apply normalization per bbox heatmap instance
    ##---------------------------------------------------------------------------------------------
    print('\n    normalization ------------------------------------------------------')   
    normalizer = tf.reduce_max(prob_grid_clipped, axis=[-2,-1], keepdims = True)
    normalizer = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    print('    normalizer     : ', normalizer.shape) 
    prob_grid_cns = prob_grid_clipped / normalizer

    
    ##---------------------------------------------------------------------------------------------
    ## (3) multiply normalized heatmap by normalized score in in_tensor/ (pt2_dense column 7)
    ##     broadcasting : https://stackoverflow.com/questions/49705831/automatic-broadcasting-in-tensorflow
    ##---------------------------------------------------------------------------------------------    
    prob_grid_cns = tf.transpose(tf.transpose(prob_grid_cns) * pt2_dense[:, NORM_SCORE_COLUMN])
    print('    prob_grid_cns: clipped/normed/scaled : ', prob_grid_cns.shape)


    ##---------------------------------------------------------------------------------------------
    ## - Build alternative scores based on normalized/scaled/clipped heatmap
    ##---------------------------------------------------------------------------------------------
    alt_scores_1 = tf.map_fn(build_hm_score_v3, [prob_grid_cns, cy, cx,covar], dtype=tf.float32)    
    print('    alt_scores_1    : ', KB.int_shape(alt_scores_1), ' Keras tensor ', KB.is_keras_tensor(alt_scores_1) )
    alt_scores_1 = tf.scatter_nd(pt2_ind, alt_scores_1, 
                                     [batch_size, num_classes, rois_per_image, KB.int_shape(alt_scores_1)[-1]], name = 'alt_scores_1')  

    print('    alt_scores_1(by class)       : ', alt_scores_1.shape ,' Keras tensor ', KB.is_keras_tensor(alt_scores_1) )  
    alt_scores_1_norm = normalize_scores(alt_scores_1)
    print('    alt_scores_1_norm(by_class)  : ', alt_scores_1_norm.shape, KB.int_shape(alt_scores_1_norm))
    # alt_scores_1_norm = tf.gather_nd(alt_scores_1_norm, pt2_ind)
    # print('    alt_scores_1_norm(by_image)  : ', alt_scores_1_norm.shape, KB.int_shape(alt_scores_1_norm))
                                                                          
    ##-------------------------------------------------------------------------------------
    ## (3) scatter out the probability distributions based on class 
    ##-------------------------------------------------------------------------------------
    print('\n    Scatter out the probability distributions based on class --------------') 
    gauss_heatmap   = tf.scatter_nd(pt2_ind, prob_grid_cns, 
                                    [batch_size, num_classes, rois_per_image, grid_w, grid_h], name = 'gauss_scatter')
    print('    pt2_ind shape      : ', pt2_ind.shape)  
    print('    prob_grid_clippped : ', prob_grid_cns.shape)  
    print('    gauss_heatmap      : ', gauss_heatmap.shape)   # batch_sz , num_classes, num_rois, image_h, image_w

    ##-------------------------------------------------------------------------------------
    ## Construction of Gaussian Heatmap output using Reduce SUM
    ##
    ## (4) SUM : Reduce and sum up gauss_heatmaps by class  
    ## (5) heatmap normalization (per class)
    ## (6) Transpose heatmap to shape required for FCN
    ##-------------------------------------------------------------------------------------
    print('\n    Reduce SUM based on class and normalize within each class -------------------------------------')         
    gauss_heatmap_sum = tf.reduce_sum(gauss_heatmap, axis=2, name='gauss_heatmap_sum')
    print('    gaussian_heatmap_sum : ', gauss_heatmap_sum.get_shape(), 'Keras tensor ', KB.is_keras_tensor(gauss_heatmap_sum) )      
    ## normalize in class
    normalizer = tf.reduce_max(gauss_heatmap_sum, axis=[-2,-1], keepdims = True)
    normalizer = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    gauss_heatmap_sum = gauss_heatmap_sum / normalizer
    # gauss_heatmap_sum_normalized = gauss_heatmap_sum / normalizer
    print('    normalizer shape   : ', normalizer.shape)   
    print('    normalized heatmap : ', gauss_heatmap_sum.shape   ,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap_sum) )
    
    ##---------------------------------------------------------------------------------------------
    ##  Score on reduced sum heatmaps. 
    ##
    ##  build indices and extract heatmaps corresponding to each bounding boxes' class id
    ##  build alternative scores#  based on normalized/sclaked clipped heatmap
    ##---------------------------------------------------------------------------------------------
    hm_indices = tf.cast(pt2_ind[:, :2],dtype=tf.int32)
    print('    hm_indices shape         :',  hm_indices.get_shape(), KB.int_shape(hm_indices))
    pt2_heatmaps = tf.gather_nd(gauss_heatmap_sum, hm_indices )
    print('    pt2_heatmaps             :',  pt2_heatmaps.get_shape(), KB.int_shape(pt2_heatmaps))

    alt_scores_2 = tf.map_fn(build_hm_score_v3, [pt2_heatmaps, cy, cx,covar], dtype=tf.float32)    
    print('    alt_scores_2    : ', KB.int_shape(alt_scores_2), ' Keras tensor ', KB.is_keras_tensor(alt_scores_2) )
    alt_scores_2 = tf.scatter_nd(pt2_ind, alt_scores_2, 
                                     [batch_size, num_classes, rois_per_image, KB.int_shape(alt_scores_2)[-1]], name = 'alt_scores_2')  

    print('    alt_scores_2(scattered)       : ', alt_scores_2.shape ,' Keras tensor ', KB.is_keras_tensor(alt_scores_2) )  
    alt_scores_2_norm = normalize_scores(alt_scores_2)
    print('    alt_scores_2_norm(by_class)  : ', alt_scores_2_norm.shape, KB.int_shape(alt_scores_2_norm))

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
    print('    reshaped heatmap   : ', gauss_heatmap_sum.shape,' Keras tensor ', KB.is_keras_tensor(gauss_heatmap_sum) )
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
    print('    gauss_scores    : ', gauss_scores.shape, ' Keras tensor ', KB.is_keras_tensor(gauss_scores) )      
    print('    complete')

    return   gauss_heatmap_sum, gauss_scores  
 



##----------------------------------------------------------------------------------------------------------------------          
##
##----------------------------------------------------------------------------------------------------------------------          
class CHMLayerInference(KE.Layer):
    '''
    Contextual Heatmap Layer - Inference mode
    Receives the bboxes, their repsective classification and roi_outputs and 
    builds the per_class tensor

    Returns:
    -------
    Returns the following tensors:

    pred_tensor :       [batch, NUM_CLASSES, TRAIN_ROIS_PER_IMAGE, 
                                             (index, class_prob, y1, x1, y2, x2, class_id, old_idx)]
                                in normalized coordinates   
    pred_cls_cnt:       [batch, NUM_CLASSES] 
    
    Note: Returned arrays might be zero padded if not enough target ROIs.
    
    '''

    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        print('\n--------------------------------')
        print('>>>  CHM Inference Layer  ')
        print('--------------------------------')
        self.config = config

        
    def call(self, inputs):

        detections = inputs[0]        
        print('  > CHM Inference Layer: call ', type(inputs), len(inputs))
        print('     detections.shape     :',  KB.int_shape(detections)) 

        pred_tensor  = build_predictions_inference(detections, self.config)
        pr_hm_norm, pr_hm_scores = build_heatmap_inference(pred_tensor, self.config, names = ['pred_heatmap'])
        # pred_cls_cnt = KL.Lambda(lambda x: tf.count_nonzero(x[:,:,:,-1],axis = -1), name = 'pred_cls_count')(pred_tensor)        
        print()
        print('    Output of CHMLayerInference: ')
        print('     pred_tensor        : ', pred_tensor.shape  , 'Keras tensor ', KB.is_keras_tensor(pred_tensor) )
        print('     pred_heatmap_norm  : ', pr_hm_norm.shape   , 'Keras tensor ', KB.is_keras_tensor(pr_hm_norm ))
        print('     pred_heatmap_scores: ', pr_hm_scores.shape , 'Keras tensor ', KB.is_keras_tensor(pr_hm_scores))
        # print('     pred_cls_cnt shape : ', pred_cls_cnt.shape , 'Keras tensor ', KB.is_keras_tensor(pred_cls_cnt) )
        print('     complete')

        return [ pr_hm_norm , pr_hm_scores]  ##  pred_tensor] 
        
        
    def compute_output_shape(self, input_shape):
        # may need to change dimensions of first return from IMAGE_SHAPE to MAX_DIM
        # return [ (None, self.config.NUM_CLASSES, self.config.DETECTION_MAX_INSTANCES,  6)]                 # pred_tensor
        return [ (None, self.config.IMAGE_SHAPE[0], self.config.IMAGE_SHAPE[1], self.config.NUM_CLASSES)     # pred_heatmap_norm
               , (None, self.config.NUM_CLASSES, self.config.DETECTION_PER_CLASS, 24)                        # pred_heatmap_scores (expanded) 
               ]

            