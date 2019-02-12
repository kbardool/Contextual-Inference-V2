import os, sys, pprint
import numpy as np
import tensorflow as tf
import keras.backend as KB
import keras.layers as KL
import keras.engine as KE
import mrcnn.utils as utils

from mrcnn.utils   import  logt
from mrcnn.chm_layer import build_hm_score_v2, build_hm_score_v3, normalize_scores

   
##-------------------------------------------------------------------------------------------------------
##   score fcn heatmaps : gen scores from heatmap
##-------------------------------------------------------------------------------------------------------
##   We use the coordinates of the bounding boxes passed in pr_scores to calculate 
##   the score of bounding boxes overlaid on the heatmap produced by the fcn_layer
##   - convert the pr_scores (or gt_hm_scores) from a per_class/per_bbox tensor to a per_class tensor
##     [BATCH_SIZE, NUM_CLASSES, DETECTIONS_PER_CLASS, 11] --> [BATCH_SIZE, DETECTIONS_MAX_INSTANCES, 11]
##   - Extract non-zero bounding boxes
##   - calculate the Cy, Cx, and Covar of the bounding boxes 
##   - Clip the heatmap by using masks centered on Cy,Cx and +/- Covar_Y, Covar_X
##-------------------------------------------------------------------------------------------------------
def fcn_scoring_graph(input, config, mode):
    in_heatmap, pr_scores = input
    rois_per_image        = KB.int_shape(pr_scores)[2] 
    img_h, img_w          = config.IMAGE_SHAPE[:2]
    batch_size            = config.BATCH_SIZE
    num_classes           = config.NUM_CLASSES  
    heatmap_scale         = config.HEATMAP_SCALE_FACTOR
    verbose               = config.VERBOSE
    CLASS_COLUMN          = 4
    SCORE_COLUMN          = 5
    DT_TYPE_COLUMN        = 6
    SEQUENCE_COLUMN       = 7
    NORM_SCORE_COLUMN     = 8
    # if mode == 'training':
        # SEQUENCE_COLUMN       = 6
        # NORM_SCORE_COLUMN     = 7
    # else:
        # DT_TYPE_COLUMN        = 6
        # SEQUENCE_COLUMN       = 7
        # NORM_SCORE_COLUMN     = 8
        
    print('\n ')
    print('---------------------------------------------')
    print('>>> FCN Scoring Graph  - mode:', mode)
    print('---------------------------------------------')
    logt('in_heatmap.shape  ', in_heatmap, verbose = verbose)
    logt('pr_hm_scores.shape', pr_scores, verbose = verbose )
    # rois per image is determined by size of input tensor 
    #   detection mode:   config.TRAIN_ROIS_PER_IMAGE 
    #   ground_truth  :   config.DETECTION_MAX_INSTANCES

    logt('pr_scores shape ', pr_scores, verbose = verbose)
    logt('rois_per_image  ', rois_per_image , verbose = verbose)
    logt('config.DETECTION_MAX_INSTANCES ', config.DETECTION_MAX_INSTANCES, verbose = verbose)
    logt('config.DETECTIONS_PER_CLASS    ', config.DETECTION_PER_CLASS, verbose = verbose)
    logt('SEQUENCE_COLUMN                ', SEQUENCE_COLUMN, verbose = verbose)
    logt('NORM_SCORE_COLUMN              ', NORM_SCORE_COLUMN, verbose = verbose)
    
    ##---------------------------------------------------------------------------------------------
    ## Stack non_zero bboxes from PR_SCORES into pt2_dense 
    ##---------------------------------------------------------------------------------------------
    # pt2_ind shape  : [?, 3] : [ {image_index, class_index , roi row_index }]
    # pt2_dense shape: [?, 11] : 
    #    pt2_dense[0:3]  roi coordinates 
    #    pt2_dense[4]    is class id 
    #    pt2_dense[5]    is score from mrcnn    
    #    pt2_dense[6]    is bbox sequence id    
    #    pt2_dense[7]    is normalized score (per class)    
    #-----------------------------------------------------------------------------
    pt2_sum = tf.reduce_sum(tf.abs(pr_scores[:,:,:,:CLASS_COLUMN]), axis=-1)
    pt2_ind = tf.where(pt2_sum > 0)
    pt2_dense = tf.gather_nd(pr_scores, pt2_ind)
    logt('in_heatmap       ', in_heatmap, verbose = verbose)
    logt('pr_scores.shape  ', pr_scores , verbose = verbose)
    logt('pt2_sum shape    ', pt2_sum   , verbose = verbose)
    logt('pt2_ind shape    ', pt2_ind   , verbose = verbose)
    logt('pt2_dense shape  ', pt2_dense , verbose = verbose)


    ##---------------------------------------------------------------------------------------------
    ##  Build mean and convariance tensors for bounding boxes
    ##---------------------------------------------------------------------------------------------
    # bboxes_scaled = tf.to_int32(tf.round(pt2_dense[...,0:4])) / heatmap_scale
    bboxes_scaled = pt2_dense[...,0:CLASS_COLUMN] / heatmap_scale
    width  = bboxes_scaled[:,3] - bboxes_scaled[:,1]      # x2 - x1
    height = bboxes_scaled[:,2] - bboxes_scaled[:,0]
    cx     = bboxes_scaled[:,1] + ( width  / 2.0)
    cy     = bboxes_scaled[:,0] + ( height / 2.0)
    # means  = tf.stack((cx,cy),axis = -1)
    covar  = tf.stack((width * 0.5 , height * 0.5), axis = -1)
    covar  = tf.sqrt(covar)            

    
    ##---------------------------------------------------------------------------------------------
    ##  build indices and extract heatmaps corresponding to each bounding boxes' class id
    ##---------------------------------------------------------------------------------------------
    hm_indices = tf.cast(pt2_ind[:, :2],dtype=tf.int32)
    logt('hm_indices  ',  hm_indices, verbose = verbose)
    
    pt2_heatmaps = tf.transpose(in_heatmap, [0,3,1,2])
    logt('pt2_heatmaps',  pt2_heatmaps, verbose = verbose)
    
    pt2_heatmaps = tf.gather_nd(pt2_heatmaps, hm_indices )
    logt('pt2_heatmaps',  pt2_heatmaps, verbose = verbose)

    ##--------------------------------------------------------------------------------------------
    ## (0) Generate scores using prob_grid and pt2_dense
    ##--------------------------------------------------------------------------------------------
    old_style_scores = tf.map_fn(build_hm_score_v2, [pt2_heatmaps, bboxes_scaled, pt2_dense[:, NORM_SCORE_COLUMN]], 
                                 dtype = tf.float32, swap_memory = True)
    logt('old_style_scores',  old_style_scores, verbose = verbose)                      
                                                                      
    # old_style_scores = tf.scatter_nd(pt2_ind, old_style_scores, 
                                     # [batch_size, num_classes, rois_per_image, KB.int_shape(old_style_scores)[-1]],
                                     # name = 'scores_scattered')
    # print('    old_style_scores        :',  old_style_scores.get_shape(), KB.int_shape(old_style_scores))                                 
                                     
    ##---------------------------------------------------------------------------------------------
    ## generate score based on gaussian using bounding box masks 
    ##---------------------------------------------------------------------------------------------
    alt_scores_1 = tf.map_fn(build_hm_score_v3, [pt2_heatmaps, cy, cx,covar], dtype=tf.float32)    
    logt('alt_scores_1 ', alt_scores_1 , verbose = verbose)

    ##---------------------------------------------------------------------------------------------
    ##  Scatter back to per-class tensor /  normalize by class
    ##---------------------------------------------------------------------------------------------
    alt_scores_1_norm = tf.scatter_nd(pt2_ind, alt_scores_1, 
                                    [batch_size, num_classes, rois_per_image, KB.int_shape(alt_scores_1)[-1]],
                                    name='alt_scores_1_norm')
    logt('alt_scores_1_scattered', alt_scores_1_norm, verbose = verbose)
    
    alt_scores_1_norm = normalize_scores(alt_scores_1_norm)
    logt('alt_scores_1_norm(by_class)', alt_scores_1_norm, verbose = verbose)
    
    alt_scores_1_norm = tf.gather_nd(alt_scores_1_norm, pt2_ind)
    logt('alt_scores_1_norm(by_image)', alt_scores_1_norm, verbose = verbose)

    ##---------------------------------------------------------------------------------------------
    ## Normalize input heatmap normalization (per class) to calculate alt_score_2
    ##--------------------------------------------------------------------------------------------
    logt('Normalize heatmap within each class !-------------------------------------', verbose = verbose)         
    in_heatmap_norm = tf.transpose(in_heatmap, [0,3,1,2])

    logt('in_heatmap_norm  ', in_heatmap_norm, verbose = verbose)
    ## normalize in class
    normalizer = tf.reduce_max(in_heatmap_norm, axis=[-2,-1], keepdims = True)
    normalizer = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    in_heatmap_norm = in_heatmap_norm / normalizer
    
    # gauss_heatmap_sum_normalized = gauss_heatmap_sum / normalizer
    logt('normalizer shape ', normalizer, verbose = verbose)   
    logt('normalized heatmap  ', in_heatmap_norm, verbose = verbose)

    ##---------------------------------------------------------------------------------------------
    ##  build indices and extract heatmaps corresponding to each bounding boxes' class id
    ##  build alternative scores#  based on normalized/sclaked clipped heatmap
    ##---------------------------------------------------------------------------------------------
    hm_indices = tf.cast(pt2_ind[:, :2],dtype=tf.int32)    
    pt2_heatmaps = tf.gather_nd(in_heatmap_norm, hm_indices)
    alt_scores_2 = tf.map_fn(build_hm_score_v3, [pt2_heatmaps, cy, cx,covar], dtype=tf.float32)    
    
    logt('hm_indices shape',  hm_indices, verbose = verbose)
    logt('pt2_heatmaps',  pt2_heatmaps, verbose = verbose)
    logt('alt_scores_2',alt_scores_2, verbose = verbose)
    
    alt_scores_2_norm = tf.scatter_nd(pt2_ind, alt_scores_2, 
                                     [batch_size, num_classes, rois_per_image, KB.int_shape(alt_scores_2)[-1]], name = 'alt_scores_2')  
    logt('alt_scores_2(scattered)', alt_scores_2_norm , verbose = verbose)
    
    alt_scores_2_norm = normalize_scores(alt_scores_2_norm)
    logt('alt_scores_2_norm(by_class)', alt_scores_2_norm, verbose = verbose)
    
    alt_scores_2_norm = tf.gather_nd(alt_scores_2_norm, pt2_ind)
    logt('alt_scores_2_norm(by_image)', alt_scores_2_norm, verbose = verbose)

    
    ##--------------------------------------------------------------------------------------------
    ##  Append alt_scores_1, alt_scores_1_norm to yield fcn_scores_dense 
    ##--------------------------------------------------------------------------------------------
    fcn_scores_dense = tf.concat([pt2_dense[:, : NORM_SCORE_COLUMN+1], old_style_scores, alt_scores_1, alt_scores_1_norm, alt_scores_2, alt_scores_2_norm], 
                                  axis = -1, name = 'fcn_scores_dense')
    logt('fcn_scores_dense    ', fcn_scores_dense , verbose = verbose)

    ##---------------------------------------------------------------------------------------------
    ##  Scatter back to per-image,class  tensor 
    ##---------------------------------------------------------------------------------------------
    seq_ids = tf.to_int32( rois_per_image - pt2_dense[:, SEQUENCE_COLUMN] )
    scatter_ind= tf.stack([hm_indices[:,0], seq_ids], axis = -1, name = 'scatter_ind')
    fcn_scores_by_class = tf.scatter_nd(pt2_ind, fcn_scores_dense, 
                                        [batch_size, num_classes, rois_per_image, fcn_scores_dense.shape[-1]], name='fcn_hm_scores')
    # fcn_scores_by_image = tf.scatter_nd(scatter_ind, fcn_scores_dense, 
                                        # [batch_size, rois_per_image, fcn_scores_dense.shape[-1]], name='fcn_hm_scores_by_image')
    logt('seq_ids             ', seq_ids, verbose = verbose)
    logt('sscatter_ids        ', scatter_ind, verbose = verbose)
    logt('fcn_scores_by_class ', fcn_scores_by_class, verbose = verbose)
    # logt('fcn_scores_by_image ', fcn_scores_by_image) 
    logt('complete', verbose = verbose)
   
    return fcn_scores_by_class
    
##------------------------------------------------------------------------------------------------------------
##
##------------------------------------------------------------------------------------------------------------
class FCNScoringLayer(KE.Layer):
    '''
    FCN Scoring Layer  
    Receives the heatmap out of FCN, bboxes information and builds FCN scores 

    Returns:
    -------
    The CHM layer returns the following tensors:

    fcn_scores :       [batch, NUM_CLASSES, TRAIN_ROIS_PER_IMAGE, 16] 
                       ( --- same as pr_hm_scores --- + fcn_score)]
    '''

    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        print()
        print('----------------------')
        print('>>> FCN Scoring Layer ')
        print('----------------------')
        self.config = config

        
    def call(self, inputs):
        fcn_heatmap, pr_hm_scores = inputs
        
        logt('> FCNScoreLayer Call() ', len(inputs) , verbose = verbose)
        logt('  fcn_heatmap.shape    ', fcn_heatmap , verbose = verbose)
        logt('  pr_hm_scores.shape   ', pr_hm_scores, verbose = verbose)

        fcn_scores  = fcn_scoring_graph([fcn_heatmap, pr_hm_scores], self.config)

        logt('\n   Output build_fcn_score ', verbose = verbose)
        logt('     fcn_scores   ', fcn_scores, verbose = verbose)
        logt('     complete', verbose = verbose)
        
        return [fcn_scores]

        
    def compute_output_shape(self, input_shape):
        # may need to change dimensions of first return from IMAGE_SHAPE to MAX_DIM
        input_num_classes= input_shape[1][1]
        input_detections = input_shape[1][2]
        input_columns    = input_shape[1][3]
        logt('   FCNScoringLayer - Compute output shape() ', verbose = self.config.VERBOSE)
        logt('   input_num_classes : ', input_num_classes, verbose = self.config.VERBOSE)
        logt('   input_detections  : ', input_detections, verbose = self.config.VERBOSE)
        logt('   input_columns     : ', input_columns, verbose = self.config.VERBOSE)
            
        return [ (None, input_num_classes, input_detections , input_columns)  ]
