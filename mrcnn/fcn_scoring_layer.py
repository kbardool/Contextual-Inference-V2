import os, sys, pprint
import numpy as np
import tensorflow as tf
# import keras
import keras.backend as KB
import keras.layers as KL
import keras.engine as KE
sys.path.append('..')
import mrcnn.utils as utils
from mrcnn.utils   import  logt
# import tensorflow.contrib.util as tfc
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
    detections_per_image  = pr_scores.shape[2] 
    rois_per_image        = KB.int_shape(pr_scores)[2] 
    img_h, img_w          = config.IMAGE_SHAPE[:2]
    batch_size            = config.BATCH_SIZE
    num_classes           = config.NUM_CLASSES  
    heatmap_scale         = config.HEATMAP_SCALE_FACTOR
    class_column          = 4
    score_column          = 5
    if mode == 'training':
        sequence_column       = 6
        norm_score_column     = 7
    else:
        dt_type_column        = 6
        sequence_column       = 7
        norm_score_column     = 8
        
    print('\n ')
    print('----------------------')
    print('>>> FCN Scoring Layer - mode:', mode)
    print('----------------------')
    logt('in_heatmap.shape  ', in_heatmap)
    logt('pr_hm_scores.shape', pr_scores)
    # rois per image is determined by size of input tensor 
    #   detection mode:   config.TRAIN_ROIS_PER_IMAGE 
    #   ground_truth  :   config.DETECTION_MAX_INSTANCES

    print('    detctions_per_image : ', detections_per_image, 'pr_scores shape', pr_scores.shape )
    print('    rois_per_image      : ', rois_per_image )
    print('    config.DETECTION_MAX_INSTANCES   : ', config.DETECTION_MAX_INSTANCES)
    print('    config.DETECTIONS_PER_CLASS      : ', config.DETECTION_PER_CLASS)
    print('    sequence_column                  : ', sequence_column)
    print('    norm_score_column                : ', norm_score_column)
    
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
    pt2_sum = tf.reduce_sum(tf.abs(pr_scores[:,:,:,:class_column]), axis=-1)
    pt2_ind = tf.where(pt2_sum > 0)
    pt2_dense = tf.gather_nd(pr_scores, pt2_ind)
    logt('in_heatmap       ', in_heatmap)
    logt('pr_scores.shape  ', pr_scores)
    logt('pt2_sum shape    ', pt2_sum)
    logt('pt2_ind shape    ', pt2_ind)
    logt('pt2_dense shape  ', pt2_dense)


    ##---------------------------------------------------------------------------------------------
    ##  Build mean and convariance tensors for bounding boxes
    ##---------------------------------------------------------------------------------------------
    # bboxes_scaled = tf.to_int32(tf.round(pt2_dense[...,0:4])) / heatmap_scale
    bboxes_scaled = pt2_dense[...,0:class_column] / heatmap_scale
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
    logt('hm_indices  ',  hm_indices)
    pt2_heatmaps = tf.transpose(in_heatmap, [0,3,1,2])
    logt('pt2_heatmaps',  pt2_heatmaps)
    pt2_heatmaps = tf.gather_nd(pt2_heatmaps, hm_indices )
    logt('pt2_heatmaps',  pt2_heatmaps)

    ##--------------------------------------------------------------------------------------------
    ## (0) Generate scores using prob_grid and pt2_dense
    ##--------------------------------------------------------------------------------------------
    old_style_scores = tf.map_fn(build_hm_score_v2, [pt2_heatmaps, bboxes_scaled, pt2_dense[:, norm_score_column]], 
                                 dtype = tf.float32, swap_memory = True)
    logt('old_style_scores',  old_style_scores)                                 
                                                                      
    # old_style_scores = tf.scatter_nd(pt2_ind, old_style_scores, 
                                     # [batch_size, num_classes, rois_per_image, KB.int_shape(old_style_scores)[-1]],
                                     # name = 'scores_scattered')
    # print('    old_style_scores        :',  old_style_scores.get_shape(), KB.int_shape(old_style_scores))                                 
                                     
    ##---------------------------------------------------------------------------------------------
    ## generate score based on gaussian using bounding box masks 
    ##---------------------------------------------------------------------------------------------
    alt_scores_1 = tf.map_fn(build_hm_score_v3, [pt2_heatmaps, cy, cx,covar], dtype=tf.float32)    
    logt('alt_scores_1 ', alt_scores_1 )

    ##---------------------------------------------------------------------------------------------
    ##  Scatter back to per-class tensor /  normalize by class
    ##---------------------------------------------------------------------------------------------
    alt_scores_1_norm = tf.scatter_nd(pt2_ind, alt_scores_1, 
                                    [batch_size, num_classes, detections_per_image, KB.int_shape(alt_scores_1)[-1]],
                                    name='alt_scores_1_norm')
    logt('alt_scores_1_scattered', alt_scores_1_norm )  
    alt_scores_1_norm = normalize_scores(alt_scores_1_norm)
    logt('alt_scores_1_norm(by_class)', alt_scores_1_norm)
    alt_scores_1_norm = tf.gather_nd(alt_scores_1_norm, pt2_ind)
    logt('alt_scores_1_norm(by_image)', alt_scores_1_norm)

    ###################################################################################################################
    ## Note: Running this scoring method yields the exact same final result as alt_score_1, and is therefore redundant
    ##-------------------------------------------------------------------------------------
    ## Normalize input heatmap normalization (per class) to calculate alt_score_2
    ##-------------------------------------------------------------------------------------
    print('\n    Normalize heatmap within each class !-------------------------------------')         
    in_heatmap_norm = tf.transpose(in_heatmap, [0,3,1,2])

    print('    in_heatmap_norm : ', in_heatmap_norm.get_shape(), 'Keras tensor ', KB.is_keras_tensor(in_heatmap_norm) )      
    ## normalize in class
    normalizer = tf.reduce_max(in_heatmap_norm, axis=[-2,-1], keepdims = True)
    normalizer = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    in_heatmap_norm = in_heatmap_norm / normalizer
    # gauss_heatmap_sum_normalized = gauss_heatmap_sum / normalizer
    print('    normalizer shape   : ', normalizer.shape)   
    print('    normalized heatmap : ', in_heatmap_norm.shape   ,' Keras tensor ', KB.is_keras_tensor(in_heatmap_norm) )

    ##---------------------------------------------------------------------------------------------
    ##  build indices and extract heatmaps corresponding to each bounding boxes' class id
    ##  build alternative scores#  based on normalized/sclaked clipped heatmap
    ##---------------------------------------------------------------------------------------------
    hm_indices = tf.cast(pt2_ind[:, :2],dtype=tf.int32)
    logt('hm_indices shape',  hm_indices)
    
    pt2_heatmaps = tf.gather_nd(in_heatmap_norm, hm_indices )
    logt('pt2_heatmaps',  pt2_heatmaps)

    alt_scores_2 = tf.map_fn(build_hm_score_v3, [pt2_heatmaps, cy, cx,covar], dtype=tf.float32)    
    logt('alt_scores_2',alt_scores_2)
    
    alt_scores_2_norm = tf.scatter_nd(pt2_ind, alt_scores_2, 
                                     [batch_size, num_classes, rois_per_image, KB.int_shape(alt_scores_2)[-1]], name = 'alt_scores_2')  
    logt('alt_scores_2(scattered)', alt_scores_2_norm ) 
    
    alt_scores_2_norm = normalize_scores(alt_scores_2_norm)
    logt('alt_scores_2_norm(by_class)', alt_scores_2_norm)
    
    alt_scores_2_norm = tf.gather_nd(alt_scores_2_norm, pt2_ind)
    logt('alt_scores_2_norm(by_image)', alt_scores_2_norm)
    ####################################################################################################################
    
    ##--------------------------------------------------------------------------------------------
    ##  Append alt_scores_1, alt_scores_1_norm to yield fcn_scores_dense 
    ##--------------------------------------------------------------------------------------------
    fcn_scores_dense = tf.concat([pt2_dense[:, : norm_score_column+1], old_style_scores, alt_scores_1, alt_scores_1_norm, alt_scores_2, alt_scores_2_norm], 
                                  axis = -1, name = 'fcn_scores_dense')
    logt('fcn_scores_dense    ', fcn_scores_dense )  

    ##---------------------------------------------------------------------------------------------
    ##  Scatter back to per-image tensor 
    ##---------------------------------------------------------------------------------------------
    seq_ids = tf.to_int32( rois_per_image - pt2_dense[:, sequence_column] )
    scatter_ind= tf.stack([hm_indices[:,0], seq_ids], axis = -1, name = 'scatter_ind')
    fcn_scores_by_class = tf.scatter_nd(pt2_ind, fcn_scores_dense, 
                                        [batch_size, num_classes, detections_per_image, fcn_scores_dense.shape[-1]], name='fcn_hm_scores')
    # fcn_scores_by_image = tf.scatter_nd(scatter_ind, fcn_scores_dense, 
                                        # [batch_size, detections_per_image, fcn_scores_dense.shape[-1]], name='fcn_hm_scores_by_image')
    logt('seq_ids             ', seq_ids) 
    logt('sscatter_ids        ', scatter_ind)
    logt('fcn_scores_by_class ', fcn_scores_by_class) 
    # logt('fcn_scores_by_image ', fcn_scores_by_image) 
    logt('complete')    
   
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
        
        print('   > FCNScoreLayer Call() ', len(inputs))
        print('     fcn_heatmap.shape    :', fcn_heatmap.shape , KB.int_shape(fcn_heatmap), 'Keras tensor ', KB.is_keras_tensor(fcn_heatmap))
        print('     pr_hm_scores.shape   :', pr_hm_scores.shape, KB.int_shape(pr_hm_scores), 'Keras tensor ', KB.is_keras_tensor(pr_hm_scores))

        # fcn_scores  = batch_slice_fcn([fcn_heatmap, pr_hm_scores, gt_hm_scores], my_score_fcn_heatmaps, 
                                            # self.config.IMAGES_PER_GPU, self.config)

        fcn_scores  = fcn_scoring_graph([fcn_heatmap, pr_hm_scores], self.config)

        print('\n    Output build_fcn_score ')
        print('     fcn_scores  : ', fcn_scores.shape  , 'Keras tensor ', KB.is_keras_tensor(fcn_scores))
        print('     complete')
        
        return [fcn_scores]

        
    def compute_output_shape(self, input_shape):
        # may need to change dimensions of first return from IMAGE_SHAPE to MAX_DIM
        print(' FCNScoringLayer - Compute output shape() ')
        input_num_classes= input_shape[1][1]
        input_detections = input_shape[1][2]
        input_columns    = input_shape[1][3]
        print('   input_num_classes : ', input_num_classes)
        print('   input_detections  : ', input_detections)
        print('   input_columns     : ', input_columns)
            
        # if self.config.mode == 'training':
            # detection_count = self.config.TRAIN_ROIS_PER_IMAGE
        # else:
            # detection_count = self.config.DETECTION_MAX_INSTANCES
            
        return [ (None, input_num_classes, input_detections , input_columns)  ]

              
"""
archived 11-15-2018              
    ##-------------------------------------------------------------------------------------------------------
    ##   score fcn heatmaps : gen scores from heatmap
    ##-------------------------------------------------------------------------------------------------------
    ##   We use the coordinates of the bounding boxes passed in pr_scores (or gt_scores), to calculate 
    ##   the score of bounding boxes overlaid on the heatmap produced by the fcn_layer
    ##   - convert the pr_scores (or gt_hm_scores) from a per_class/per_bbox tensor to a per_class tensor
    ##     [BATCH_SIZE, NUM_CLASSES, DETECTIONS_PER_CLASS, 11] --> [BATCH_SIZE, DETECTIONS_MAX_INSTANCES, 11]
    ##   - Extract non-zero bounding boxes
    ##   - calculate the Cy, Cx, and Covar of the bounding boxes 
    ##   - Clip the heatmap by using masks centered on Cy,Cx and +/- Covar_Y, Covar_X
    ##-------------------------------------------------------------------------------------------------------
    def score_fcn_heatmaps(self, in_heatmap, pr_scores, gt_scores):

        num_detections  = tf.shape(pr_scores)[1] 
        img_h, img_w    = self.config.IMAGE_SHAPE[:2]
        batch_size      = self.config.BATCH_SIZE
        num_classes     = self.config.NUM_CLASSES  
        heatmap_scale   = self.config.HEATMAP_SCALE_FACTOR
        print('\n ')
        print('  > Score_fcn_heatmap() for ')
        print('    in_heatmap shape : ', in_heatmap.shape)       
        print('    pr_scores  shape : ', pr_scores.shape)       
        print('    gt_scores  shape : ', gt_scores.shape)       
        
        # rois per image is determined by size of input tensor 
        #   detection mode:   config.TRAIN_ROIS_PER_IMAGE 
        #   ground_truth  :   config.DETECTION_MAX_INSTANCES
        
        rois_per_image  = KB.int_shape(pr_scores)[2] 
        # strt_cls        = 0 if rois_per_image == 32 else 1
        print('    num_detctions     : ', num_detections )
        print('    rois_per_image    : ', rois_per_image )

        ##----------------------------------------------------------------------------------------------------
        ## flatten guassian scattered and input_tensor, and pass on to build_bbox_score routine 
        ##----------------------------------------------------------------------------------------------------
        # pr_scores_shape = tf.shape(pr_scores)
        # pr_scores_flat  = tf.reshape(pr_scores, [-1, pr_scores_shape[-1]])
        # bboxes = tf.to_int32(tf.round(pr_scores_flat[...,0:4]))
        # print('    pr_scores_shape : ', pr_scores_shape.eval() )
        # print('    pr_scores_flat  : ', tf.shape(pr_scores_flat).eval())
        # print('    boxes shape     : ', tf.shape(bboxes).eval())
        # print('    Rois per image  : ', rois_per_image)
        
        ##-----------------------------------------------------------------------------    
        ## Stack non_zero bboxes from PR_SCORES into pt2_dense 
        ##-----------------------------------------------------------------------------
        # pt2_ind shape  : [?, 3] : [ {image_index, class_index , roi row_index }]
        # pt2_dense shape: [?, 11] : 
        #    pt2_dense[0:3]  roi coordinates 
        #    pt2_dense[4]    is class id 
        #    pt2_dense[5]    is score from mrcnn    
        #    pt2_dense[6]    is bbox sequence id    
        #    pt2_dense[7]    is normalized score (per class)    
        #-----------------------------------------------------------------------------
        pt2_sum = tf.reduce_sum(tf.abs(pr_scores[:,:,:4]), axis=-1)
        pt2_ind = tf.where(pt2_sum > 0)
        pt2_dense = tf.gather_nd(pr_scores, pt2_ind)
        bboxes = tf.to_int32(tf.round(pt2_dense[:,0:4]))
        hm_indices =  tf.to_int32(tf.expand_dims(pt2_dense[:,4], axis = -1))
        print('    pt2_sum shape     : ', KB.int_shape(pt2_sum))
        print('    pt2_ind shape     : ', KB.int_shape(pt2_ind))
        print('    pt2_dense shape   : ', KB.int_shape(pt2_dense))
        print('    bboxes  shape     : ', KB.int_shape(bboxes))
        print('    hm_indices shape  :',  KB.int_shape(hm_indices))
        
        ##-----------------------------------------------------------------------------
        ##  Build mean and convariance tensors for Multivariate Normal Distribution 
        ##-----------------------------------------------------------------------------
        bboxes = bboxes / heatmap_scale
        width  = bboxes[:,3] - bboxes[:,1]      # x2 - x1
        height = bboxes[:,2] - bboxes[:,0]
        cx     = bboxes[:,1] + ( width  / 2.0)
        cy     = bboxes[:,0] + ( height / 2.0)
        means  = tf.stack((cx,cy),axis = -1)
        covar  = tf.stack((width * 0.5 , height * 0.5), axis = -1)
        covar  = tf.sqrt(covar)        

        #-----------------------------------------------------------------------------------------------------
        # duplicate GAUSS_NORM <num_roi> times to pass along with bboxes to map_fn function
        #   Here we have a choice to calculate scores using the GAUSS_SUM (unnormalized) or GAUSS_NORM 
        #   (normalized). After looking at the scores and ratios for each option, I decided to go with 
        #   the normalized as the numbers are large
        #-----------------------------------------------------------------------------------------------------
        # hm_indices =  tf.cast(pt2_dense[:,4],dtype=tf.int32)

        pt2_heatmaps = tf.gather(in_heatmap, hm_indices[:,0], axis = -1)
        print('    selected heatmaps shape  :',  pt2_heatmaps.get_shape(), KB.int_shape(pt2_heatmaps))
        pt2_heatmaps = tf.transpose(pt2_heatmaps, [2,0,1])
        print('    selected heatmaps shape  :',  pt2_heatmaps.get_shape(), KB.int_shape(pt2_heatmaps))


        ##----------------------------------------------------------------------------------------------------
        ## generate score based on gaussian using bounding box masks 
        ## NOTE: Score is generated on NORMALIZED gaussian distributions (GAUSS_NORM)
        ##       If want to do this on NON-NORMALIZED, we need to apply it on GAUSS_SUM
        ##----------------------------------------------------------------------------------------------------    
        print(KB.int_shape(pt2_heatmaps),KB.int_shape(cy),KB.int_shape(cx), KB.int_shape(covar))
        scores = tf.map_fn(self.build_fcn_score, [pt2_heatmaps, cy, cx,covar], dtype=tf.float32)    
        
        ##---------------------------------------------------------------------------------------------
        ## Apply normalization per bbox scores
        ##---------------------------------------------------------------------------------------------
        print('\n    normalization ------------------------------------------------------')   
        normalizer = tf.reduce_max(scores, axis=[-1], keepdims = True)
        normalizer = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
        scores_norm = scores / normalizer
        print('    normalizer     : ', normalizer.shape) 
        print('    scores_norm    : ', scores_norm.shape)
        
        both_scores = tf.stack([scores, scores_norm], axis = -1)
        print('    both_scores    : ', both_scores.shape)
        
        # consider the two new columns for reshaping the gaussian_bbox_scores
        # new_shape   = pr_scores_shape + [0,0,0, tf.shape(scores)[-1]]        
        # bbox_scores = tf.concat([pr_scores_flat, scores], axis = -1)
        # bbox_scores = tf.reshape(bbox_scores, new_shape)
        # print('    new shape is            : ', new_shape.eval())
        # print('    pr_scores_flat          : ', tf.shape(pr_scores_flat).eval())
        # print('    Scores shape            : ', tf.shape(scores).eval())   # [(num_batches x num_class x num_rois ), 3]
        # print('    boxes_scores (rehspaed) : ', tf.shape(bbox_scores).eval())    

        #--------------------------------------------------------------------------------------------
        # this normalization moves values to [-1 +1] 
        #--------------------------------------------------------------------------------------------    
        # reduce_max = tf.reduce_max(bbox_scores[...,-1], axis = -1, keepdims=True)
        # reduce_min = tf.reduce_min(bbox_scores[...,-1], axis = -1, keepdims=True)  ## epsilon    = tf.ones_like(reduce_max) * 1e-7
        # scr_norm  = (2* (bbox_scores[...,-1] - reduce_min) / (reduce_max - reduce_min)) - 1     

        # scr_norm     = tf.expand_dims(scr_norm, axis = -1)                             # shape (num_imgs, num_class, 32, 1)
        # fcn_scores   = KB.identity(tf.concat([bbox_scores, scr_norm, scr_L2norm], axis = -1), name = 'fcn_heatmap_scores') 
        ##--------------------------------------------------------------------------------------------
        ## Add returned values from scoring to the end of the input score 
        ##--------------------------------------------------------------------------------------------    
        fcn_scores  = tf.concat([pt2_dense, both_scores], axis = -1, name='fcn_heatmap_scores')
        Padding    = tf.maximum(num_detections- tf.shape(fcn_scores)[0], 0)
        padded_fcn_scores = tf.pad(fcn_scores, [(0, Padding), (0, 0)])

        print('    both_scores        : ', both_scores.shape ,' Keras tensor ', KB.is_keras_tensor(both_scores) )  
        print('    fcn_scores         : ', fcn_scores.shape ,' Keras tensor ', KB.is_keras_tensor(fcn_scores) )  
        print('    padding            : ', Padding.shape ,' Keras tensor ', KB.is_keras_tensor(Padding) )  
        print('    fcn_padded_scores  : ', fcn_padded_scores.shape ,' Keras tensor ', KB.is_keras_tensor(fcn_padded_scores) )  
        print('    complete')        

        return padded_fcn_scores     
"""        

"""        
    ##--------------------------------------------------------------------------------------------------------
    ##
    ##--------------------------------------------------------------------------------------------------------        
    def build_fcn_score(self, input_list):
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
        heatmap_tensor, input_row = input_list
        with tf.variable_scope('mask_routine'):
            y_extent     = tf.range(input_row[0], input_row[2])
            x_extent     = tf.range(input_row[1], input_row[3])
            Y,X          = tf.meshgrid(y_extent, x_extent)
            bbox_mask    = tf.stack([Y,X],axis=2)        
            mask_indices = tf.reshape(bbox_mask,[-1,2])
            mask_indices = tf.to_int32(mask_indices)
            mask_size    = tf.shape(mask_indices)[0]
            mask_updates = tf.ones([mask_size], dtype = tf.float32)    
            mask         = tf.scatter_nd(mask_indices, mask_updates, tf.shape(heatmap_tensor))
            # mask_sum    =  tf.reduce_sum(mask)
            heatmap_tensor = tf.multiply(heatmap_tensor, mask, name = 'mask_applied')
            bbox_area    = tf.to_float((input_row[2]-input_row[0]) * (input_row[3]-input_row[1]))
            gaussian_sum = tf.reduce_sum(mask_applied)
            
            ratio        = gaussian_sum / bbox_area 
            ratio        = tf.where(tf.is_nan(ratio),  0.0, ratio)  
        return tf.stack([gaussian_sum, bbox_area, ratio], axis = -1)
"""
        
"""
    ##-------------------------------------------------------------------------------------------------------
    ##   score fcn heatmaps : gen scores from heatmap
    ##-------------------------------------------------------------------------------------------------------
    ##   We use the coordinates of the bounding boxes passed in pr_scores (or gt_scores), to calculate 
    ##   the score of bounding boxes overlaid on the heatmap produced by the fcn_layer
    ##   - convert the pr_scores (or gt_hm_scores) from a per_class/per_bbox tensor to a per_class tensor
    ##     [BATCH_SIZE, NUM_CLASSES, DETECTIONS_PER_CLASS, 11] --> [BATCH_SIZE, DETECTIONS_MAX_INSTANCES, 11]
    ##   - Extract non-zero bounding boxes
    ##   - calculate the Cy, Cx, and Covar of the bounding boxes 
    ##   - Clip the heatmap by using masks centered on Cy,Cx and +/- Covar_Y, Covar_X
    ##-------------------------------------------------------------------------------------------------------
    def score_fcn_heatmaps(self, in_heatmap, pr_scores, gt_hm_scores, names = None):

        num_detections  = self.config.DETECTION_MAX_INSTANCES
        img_h, img_w    = self.config.IMAGE_SHAPE[:2]
        batch_size      = self.config.BATCH_SIZE
        num_classes     = self.config.NUM_CLASSES  
        print('\n ')
        print('  > NEW build_heatmap() for ', names )
        print('    orignal in_heatmap shape : ', in_heatmap.shape)       
        # rois per image is determined by size of input tensor 
        #   detection mode:   config.TRAIN_ROIS_PER_IMAGE 
        #   ground_truth  :   config.DETECTION_MAX_INSTANCES
        rois_per_image  = KB.int_shape(pr_scores)[2] 
        # strt_cls        = 0 if rois_per_image == 32 else 1
        print('    num of bboxes per class is : ', rois_per_image )

        ##----------------------------------------------------------------------------------------------------
        ## flatten guassian scattered and input_tensor, and pass on to build_bbox_score routine 
        # generate score based on gaussian using bounding box masks 
        # NOTE: Score is generated on NORMALIZED gaussian distributions (GAUSS_NORM)
        #       If want to do this on NON-NORMALIZED, we need to apply it on GAUSS_SUM
        ##----------------------------------------------------------------------------------------------------

        pr_scores_shape = tf.shape(pr_scores)
        pr_scores_flat  = tf.reshape(pr_scores, [-1, pr_scores_shape[-1]])
        bboxes = tf.to_int32(tf.round(pr_scores_flat[...,0:4]))
        # print('    pr_scores_shape : ', pr_scores_shape.eval() )
        # print('    pr_scores_flat  : ', tf.shape(pr_scores_flat).eval())
        # print('    boxes shape     : ', tf.shape(bboxes).eval())
        print('    Rois per image  : ', rois_per_image)

        #-----------------------------------------------------------------------------------------------------
        # duplicate GAUSS_NORM <num_roi> times to pass along with bboxes to map_fn function
        #   Here we have a choice to calculate scores using the GAUSS_SUM (unnormalized) or GAUSS_NORM 
        #   (normalized). After looking at the scores and ratios for each option, I decided to go with 
        #   the normalized as the numbers are large
        #-----------------------------------------------------------------------------------------------------
        dup_heatmap = tf.transpose(in_heatmap, [0,3,1,2])
        print('    heatmap original shape   : ', in_heatmap.shape)
        print('    heatmap transposed shape :',  dup_heatmap.get_shape())
        dup_heatmap = tf.expand_dims(dup_heatmap, axis =2)
        # print('    heatmap expanded shape   :',  tf.shape(dup_heatmap).eval())
        dup_heatmap = tf.tile(dup_heatmap, [1,1, rois_per_image ,1,1])
        print('    heatmap tiled            : ', dup_heatmap.get_shape())
        dup_heatmap_shape   = KB.int_shape(dup_heatmap)
        dup_heatmap         = KB.reshape(dup_heatmap, (-1, dup_heatmap_shape[-2], dup_heatmap_shape[-1]))
        print('    dup_heatmap reshaped     : ', tf.shape(dup_heatmap).eval())

        ##----------------------------------------------------------------------------------------------------
        ## generate score based on gaussian using bounding box masks 
        ## NOTE: Score is generated on NORMALIZED gaussian distributions (GAUSS_NORM)
        ##       If want to do this on NON-NORMALIZED, we need to apply it on GAUSS_SUM
        ##----------------------------------------------------------------------------------------------------
        scores = tf.map_fn(self.build_mask_routine, [dup_heatmap, bboxes], dtype=tf.float32)    

        ##--------------------------------------------------------------------------------------------
        ## Add returned values from scoring to the end of the input score 
        ##--------------------------------------------------------------------------------------------    
        # consider the two new columns for reshaping the gaussian_bbox_scores
        new_shape   = pr_scores_shape + [0,0,0, tf.shape(scores)[-1]]        
        bbox_scores = tf.concat([pr_scores_flat, scores], axis = -1)
        bbox_scores = tf.reshape(bbox_scores, new_shape)
        # print('    new shape is            : ', new_shape.eval())
        # print('    pr_scores_flat          : ', tf.shape(pr_scores_flat).eval())
        # print('    Scores shape            : ', tf.shape(scores).eval())   # [(num_batches x num_class x num_rois ), 3]
        # print('    boxes_scores (rehspaed) : ', tf.shape(bbox_scores).eval())    

        ##--------------------------------------------------------------------------------------------
        ## Normalize computed score above, and add it to the heatmap_score tensor as last column
        ##--------------------------------------------------------------------------------------------
        scr_L2norm   = tf.nn.l2_normalize(bbox_scores[...,-1], axis = -1)   # shape (num_imgs, num_class, num_rois)
        scr_L2norm   = tf.expand_dims(scr_L2norm, axis = -1)
       
        ##--------------------------------------------------------------------------------------------
        # shape of tf.reduce_max(bbox_scores[...,-1], axis = -1, keepdims=True) is (num_imgs, num_class, 1)
        #  This is a regular normalization that moves everything between [0, 1]. This causes negative values to move
        #  to -inf. 
        # To address this a normalization between [-1 and +1] was introduced. Not sure how this will work with 
        # training tho.
        ##--------------------------------------------------------------------------------------------
        scr_norm     = bbox_scores[...,-1]/ tf.reduce_max(bbox_scores[...,-1], axis = -1, keepdims=True)
        scr_norm     = tf.where(tf.is_nan(scr_norm),  tf.zeros_like(scr_norm), scr_norm)     
        
        #--------------------------------------------------------------------------------------------
        # this normalization moves values to [-1 +1] 
        #--------------------------------------------------------------------------------------------    
        # reduce_max = tf.reduce_max(bbox_scores[...,-1], axis = -1, keepdims=True)
        # reduce_min = tf.reduce_min(bbox_scores[...,-1], axis = -1, keepdims=True)  ## epsilon    = tf.ones_like(reduce_max) * 1e-7
        # scr_norm  = (2* (bbox_scores[...,-1] - reduce_min) / (reduce_max - reduce_min)) - 1     

        # scr_norm     = tf.where(tf.is_nan(scr_norm),  tf.zeros_like(scr_norm), scr_norm)  
        scr_norm     = tf.expand_dims(scr_norm, axis = -1)                             # shape (num_imgs, num_class, 32, 1)
        fcn_scores   = KB.identity(tf.concat([bbox_scores, scr_norm, scr_L2norm], axis = -1), name = 'fcn_heatmap_scores') 

        print('    fcn_scores  final shape : ', fcn_scores.shape ,' Keras tensor ', KB.is_keras_tensor(fcn_scores) )  
        print('    complete')

        return fcn_scores     
"""        
                  
"""
### Batch Slicing -------------------------------------------------------------------
##   Some custom layers support a batch size of 1 only, and require a lot of work
##   to support batches greater than 1. This function slices an input tensor
##   across the batch dimension and feeds batches of size 1. Effectively,
##   an easy way to support batches > 1 quickly with little code modification.
##   In the long run, it's more efficient to modify the code to support large
##   batches and getting rid of this function. Consider this a temporary solution
##   batch dimension size:
##       DetectionTargetLayer    IMAGES_PER_GPU  * # GPUs (batch size)
##-----------------------------------------------------------------------------------

def my_batch_slice_fcn(inputs, graph_fn, batch_size, config = None, names=None):
    '''
    Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs:     list of tensors. All must have the same first dimension length
    graph_fn:   A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names:      If provided, assigns names to the resulting tensors.
    '''
    print(' batch slice fcn')
    if not isinstance(inputs, list):
        inputs = [inputs]
    fcn_hm       = inputs[0]   
    fcn_hm_shape = KB.int_shape(fcn_hm)
    pr_hm_scores = inputs[1]
    gt_hm_scores = inputs[2]
    for i,x in enumerate(inputs):
        print('input ', i ,' ----' , KB.int_shape(x), 'Keras tensor ', KB.is_keras_tensor(x))
    print(' fcn_hm_shape:', fcn_hm_shape)
    outputs = []
    
    
    # fcn_scores = tf.map_fn(my_score_fcn_heatmaps, [fcn_hm, pr_hm_scores, gt_hm_scores], dtype=tf.float32)    
    
    
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]    # inputs is a list eg. [sc, ix] => input_slice = [sc[0], ix[0],...]
        # for i in inputs_slice:
        fcn_slice = KL.Lambda(lambda x: x[i,:,:,:], output_shape=(1,)+ fcn_hm_shape[1:])(fcn_hm)
        print(' fcn_hm           ----' , KB.int_shape(fcn_hm))
        print(' fcn_hm[i]        ----' , KB.int_shape(fcn_hm[i]))
        print(' fcn_slice        ----' , KB.int_shape(fcn_slice))
        print(' pr_hm_scores[i]  ----' , KB.int_shape(pr_hm_scores[i]))
        print(' gt_hm_scores[i]  ----' , KB.int_shape(gt_hm_scores[i]))
        
        output_slice = graph_fn(fcn_hm[i], pr_hm_scores[i], gt_hm_scores[i], config)   # pass list of inputs_slices through function => graph_fn(sc[0], ix[0],...)
    
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)

    # Change outputs from:
    #    a list of slices where each is a list of outputs, e.g.  [ [out1[0],out2[0]], [out1[1], out2[1]],.....
    # to 
    #    a list of outputs and each has a list of slices ==>    [ [out1[0],out1[1],...] , [out2[0], out2[1],....],.....    
    
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    for o,n in zip(outputs,names):
        print(' outputs shape: ', len(o), 'name: ',n)
        for i in range(len(o)):
            print(' shape of item ',i, 'in tuple', o[i].shape)
        
    result = [tf.stack(o, axis=0, name=n) for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


##-------------------------------------------------------------------------------------------------------
##   score fcn heatmaps : gen scores from heatmap
##-------------------------------------------------------------------------------------------------------
##   We use the coordinates of the bounding boxes passed in pr_scores (or gt_scores), to calculate 
##   the score of bounding boxes overlaid on the heatmap produced by the fcn_layer
##   - convert the pr_scores (or gt_hm_scores) from a per_class/per_bbox tensor to a per_class tensor
##     [BATCH_SIZE, NUM_CLASSES, DETECTIONS_PER_CLASS, 11] --> [BATCH_SIZE, DETECTIONS_MAX_INSTANCES, 11]
##   - Extract non-zero bounding boxes
##   - calculate the Cy, Cx, and Covar of the bounding boxes 
##   - Clip the heatmap by using masks centered on Cy,Cx and +/- Covar_Y, Covar_X
##-------------------------------------------------------------------------------------------------------
def my_score_fcn_heatmaps(in_heatmap, pr_scores, gt_scores, config):

    num_detections  = tf.shape(pr_scores)[1] 
    img_h, img_w    = config.IMAGE_SHAPE[:2]
    batch_size      = config.BATCH_SIZE
    num_classes     = config.NUM_CLASSES  
    heatmap_scale   = config.HEATMAP_SCALE_FACTOR
    print('\n ')
    print('  > my_score_fcn_heatmap() for ')
    print('    in_heatmap shape : ', in_heatmap.shape)       
    print('    pr_scores  shape : ', pr_scores.shape)       
    print('    gt_scores  shape : ', gt_scores.shape)       
    
    # rois per image is determined by size of input tensor 
    #   detection mode:   config.TRAIN_ROIS_PER_IMAGE 
    #   ground_truth  :   config.DETECTION_MAX_INSTANCES
    
    rois_per_image  = KB.int_shape(pr_scores)[2] 
    # strt_cls        = 0 if rois_per_image == 32 else 1
    print('    num_detctions     : ', num_detections )
    print('    rois_per_image    : ', rois_per_image )

    ##----------------------------------------------------------------------------------------------------
    ## flatten guassian scattered and input_tensor, and pass on to build_bbox_score routine 
    ##----------------------------------------------------------------------------------------------------
    # pr_scores_shape = tf.shape(pr_scores)
    # pr_scores_flat  = tf.reshape(pr_scores, [-1, pr_scores_shape[-1]])
    # bboxes = tf.to_int32(tf.round(pr_scores_flat[...,0:4]))
    # print('    pr_scores_shape : ', pr_scores_shape.eval() )
    # print('    pr_scores_flat  : ', tf.shape(pr_scores_flat).eval())
    # print('    boxes shape     : ', tf.shape(bboxes).eval())
    # print('    Rois per image  : ', rois_per_image)
    
    ##-----------------------------------------------------------------------------    
    ## Stack non_zero bboxes from PR_SCORES into pt2_dense 
    ##-----------------------------------------------------------------------------
    # pt2_ind shape  : [?, 3] : [ {image_index, class_index , roi row_index }]
    # pt2_dense shape: [?, 11] : 
    #    pt2_dense[0:3]  roi coordinates 
    #    pt2_dense[4]    is class id 
    #    pt2_dense[5]    is score from mrcnn    
    #    pt2_dense[6]    is bbox sequence id    
    #    pt2_dense[7]    is normalized score (per class)    
    #-----------------------------------------------------------------------------
    pt2_sum = tf.reduce_sum(tf.abs(pr_scores[:,:,:4]), axis=-1)
    pt2_ind = tf.where(pt2_sum > 0)
    pt2_dense = tf.gather_nd(pr_scores, pt2_ind)
    bboxes = tf.to_int32(tf.round(pt2_dense[:,0:4]))
    hm_indices =  tf.to_int32(tf.expand_dims(pt2_dense[:,4], axis = -1))
    print('    pt2_sum shape     : ', KB.int_shape(pt2_sum))
    print('    pt2_ind shape     : ', KB.int_shape(pt2_ind))
    print('    pt2_dense shape   : ', KB.int_shape(pt2_dense))
    print('    bboxes  shape     : ', KB.int_shape(bboxes))
    print('    hm_indices shape  :',  KB.int_shape(hm_indices))
    
    ##-----------------------------------------------------------------------------
    ##  Build mean and convariance tensors for Multivariate Normal Distribution 
    ##-----------------------------------------------------------------------------
    bboxes = bboxes / heatmap_scale
    width  = bboxes[:,3] - bboxes[:,1]      # x2 - x1
    height = bboxes[:,2] - bboxes[:,0]
    cx     = bboxes[:,1] + ( width  / 2.0)
    cy     = bboxes[:,0] + ( height / 2.0)
    means  = tf.stack((cx,cy),axis = -1)
    covar  = tf.stack((width * 0.5 , height * 0.5), axis = -1)
    covar  = tf.sqrt(covar)        

    #-----------------------------------------------------------------------------------------------------
    # duplicate GAUSS_NORM <num_roi> times to pass along with bboxes to map_fn function
    #   Here we have a choice to calculate scores using the GAUSS_SUM (unnormalized) or GAUSS_NORM 
    #   (normalized). After looking at the scores and ratios for each option, I decided to go with 
    #   the normalized as the numbers are large
    #-----------------------------------------------------------------------------------------------------
    # hm_indices =  tf.cast(pt2_dense[:,4],dtype=tf.int32)

    pt2_heatmaps = tf.gather(in_heatmap, hm_indices[:,0], axis = -1)
    print('    selected heatmaps shape  :',  pt2_heatmaps.get_shape(), KB.int_shape(pt2_heatmaps))
    pt2_heatmaps = tf.transpose(pt2_heatmaps, [2,0,1])
    print('    selected heatmaps shape  :',  pt2_heatmaps.get_shape(), KB.int_shape(pt2_heatmaps))


    ##----------------------------------------------------------------------------------------------------
    ## generate score based on gaussian using bounding box masks 
    ## NOTE: Score is generated on NORMALIZED gaussian distributions (GAUSS_NORM)
    ##       If want to do this on NON-NORMALIZED, we need to apply it on GAUSS_SUM
    ##----------------------------------------------------------------------------------------------------    
    print(KB.int_shape(pt2_heatmaps),KB.int_shape(cy),KB.int_shape(cx), KB.int_shape(covar))
    scores = tf.map_fn(my_build_fcn_score, [pt2_heatmaps, cy, cx,covar], dtype=tf.float32)    
    
    ##---------------------------------------------------------------------------------------------
    ## Apply normalization per bbox scores
    ##---------------------------------------------------------------------------------------------
    print('\n    normalization ------------------------------------------------------')   
    normalizer = tf.reduce_max(scores, axis=[-1], keepdims = True)
    normalizer = tf.where(normalizer < 1.0e-15,  tf.ones_like(normalizer), normalizer)
    scores_norm = scores / normalizer
    print('    normalizer     : ', normalizer.shape) 
    print('    scores_norm    : ', scores_norm.shape)
    
    both_scores = tf.stack([scores, scores_norm], axis = -1)
    print('    both_scores    : ', both_scores.shape)
    
    # consider the two new columns for reshaping the gaussian_bbox_scores
    # new_shape   = pr_scores_shape + [0,0,0, tf.shape(scores)[-1]]        
    # bbox_scores = tf.concat([pr_scores_flat, scores], axis = -1)
    # bbox_scores = tf.reshape(bbox_scores, new_shape)
    # print('    new shape is            : ', new_shape.eval())
    # print('    pr_scores_flat          : ', tf.shape(pr_scores_flat).eval())
    # print('    Scores shape            : ', tf.shape(scores).eval())   # [(num_batches x num_class x num_rois ), 3]
    # print('    boxes_scores (rehspaed) : ', tf.shape(bbox_scores).eval())    

    #--------------------------------------------------------------------------------------------
    # this normalization moves values to [-1 +1] 
    #--------------------------------------------------------------------------------------------    
    # reduce_max = tf.reduce_max(bbox_scores[...,-1], axis = -1, keepdims=True)
    # reduce_min = tf.reduce_min(bbox_scores[...,-1], axis = -1, keepdims=True)  ## epsilon    = tf.ones_like(reduce_max) * 1e-7
    # scr_norm  = (2* (bbox_scores[...,-1] - reduce_min) / (reduce_max - reduce_min)) - 1     

    # scr_norm     = tf.expand_dims(scr_norm, axis = -1)                             # shape (num_imgs, num_class, 32, 1)
    # fcn_scores   = KB.identity(tf.concat([bbox_scores, scr_norm, scr_L2norm], axis = -1), name = 'fcn_heatmap_scores') 
    ##--------------------------------------------------------------------------------------------
    ## Add returned values from scoring to the end of the input score 
    ##--------------------------------------------------------------------------------------------    
    fcn_scores  = tf.concat([pt2_dense, both_scores], axis = -1, name='fcn_heatmap_scores')
    Padding    = tf.maximum(num_detections- tf.shape(fcn_scores)[0], 0)
    padded_fcn_scores = tf.pad(fcn_scores, [(0, Padding), (0, 0)])

    print('    both_scores        : ', both_scores.shape ,' Keras tensor ', KB.is_keras_tensor(both_scores) )  
    print('    fcn_scores         : ', fcn_scores.shape ,' Keras tensor ', KB.is_keras_tensor(fcn_scores) )  
    print('    padding            : ', Padding.shape ,' Keras tensor ', KB.is_keras_tensor(Padding) )  
    print('    fcn_padded_scores  : ', fcn_padded_scores.shape ,' Keras tensor ', KB.is_keras_tensor(fcn_padded_scores) )  
    print('    complete')        

    return padded_fcn_scores     
    
##--------------------------------------------------------------------------------------------------------
##
##--------------------------------------------------------------------------------------------------------        
def my_build_fcn_score(input_list):
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
    print('    build_fcn_score()')
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

        heatmap_tensor = tf.multiply(heatmap_tensor, mask, name = 'mask_applied')
        score        = tf.reduce_sum(heatmap_tensor)
        print(' heatmapshape:', heatmap_tensor.get_shape())
    return score        
"""
              