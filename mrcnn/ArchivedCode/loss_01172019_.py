"""
Mask R-CNN
Dataset functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import numpy         as np
import tensorflow    as tf
import keras.backend as KB
import keras.layers  as KL
import keras.initializers as KI
import keras.engine  as KE
import keras.losses  as KLosses
import mrcnn.utils   as utils
from mrcnn.utils import logt
import pprint
pp = pprint.PrettyPrinter(indent=2, width=100)


##-----------------------------------------------------------------------
##  Loss Functions
##-----------------------------------------------------------------------
def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typicallly: [N, 4], but could be any shape.
    """
    diff = KB.abs(y_true - y_pred)
    less_than_one = KB.cast(KB.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss

##-----------------------------------------------------------------------
##  RPN anchor classifier loss
##-----------------------------------------------------------------------
def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    '''
    RPN anchor classifier loss.

    calculates the loss between:
    
    rpn_match:          [batch, anchors, 1]. Anchor match type. 1=positive,
                        -1=negative, 0=neutral anchor.
                        
    rpn_class_logits:   [batch, anchors, 2]. RPN classifier logits for FG/BG.
    '''
    print('\n>>> rpn_class_loss_graph' )
    print('    rpn_match size :', rpn_match.shape)
    print('    tf default session: ', tf.get_default_session())

    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = KB.cast(KB.equal(rpn_match, 1), tf.int32)
    
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(KB.not_equal(rpn_match, 0))
    
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class     = tf.gather_nd(anchor_class, indices)
    
    # Crossentropy loss
    loss = KB.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    
    print('    loss      :', loss.get_shape(), KB.shape(loss), 'KerasTensor: ', KB.is_keras_tensor(loss))
    loss = KB.switch(tf.size(loss) > 0, KB.mean(loss), tf.constant(0.0))
    print('    mean loss :', loss.get_shape(), KB.shape(loss), 'KerasTensor: ', KB.is_keras_tensor(loss))
    loss = tf.reshape(loss, [1, 1], name = 'rpn_class_loss')    
    print('    reshaped mean loss :', loss.get_shape(), KB.shape(loss), 'KerasTensor: ', KB.is_keras_tensor(loss))
    return loss



##-----------------------------------------------------------------------
##  RPN bbox loss  
##  new and improved version which uses call to smooth_l1_loss() 
##-----------------------------------------------------------------------    
def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    '''
    Return the RPN bounding box loss graph.

    config:             the model config object.
    
    target_bbox:        [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
                        Uses 0 padding to fill in unsed bbox deltas.
    
    rpn_match:          [batch, anchors, 1]. Anchor match type. 1=positive,
                        -1=negative, 0=neutral anchor.
    
    rpn_bbox:           [batch, anchors, (dy, dx, log(dh), log(dw))]
    
    '''

    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = KB.squeeze(rpn_match, -1)
    indices   = tf.where(KB.equal(rpn_match, 1))
    print('\n>>> rpn_bbox_loss_graph' )
    print('    rpn_match size :', rpn_match.shape)
    print('    rpn_bbox  size :', rpn_bbox.shape)
    # print(rpn_match.eval())
    
    # Pick bbox deltas that contribute to the loss
    rpn_bbox  = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = KB.sum(KB.cast(KB.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox  = utils.batch_pack_graph(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU)
  
    # Smooth-L1 Loss
    loss = KB.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=rpn_bbox),
                    tf.constant(0.0))
    print('    loss      :', loss.get_shape(), KB.shape(loss), 'KerasTensor: ', KB.is_keras_tensor(loss))
    loss = KB.mean(loss)
    print('    mean loss :', loss.get_shape(), KB.shape(loss), 'KerasTensor: ', KB.is_keras_tensor(loss))
    loss = tf.reshape(loss, [1, 1], name = 'rpn_bbox_loss')
    print('    reshaped mean loss :', loss.get_shape(), KB.shape(loss), 'KerasTensor: ', KB.is_keras_tensor(loss))
    return loss

##-----------------------------------------------------------------------
##  MRCNN Class loss  
##-----------------------------------------------------------------------    
def mrcnn_class_loss_graph(target_class_ids, pred_class_logits, active_class_ids):
    '''
    Loss for the classifier head of Mask RCNN.

    target_class_ids:       [batch, num_rois]. Integer class IDs. Uses zero
                            padding to fill in the array.
    
    pred_class_logits:      [batch, num_rois, num_classes]
    
    active_class_ids:       [batch, num_classes]. Has a value of 1 for
                            classes that are in the dataset of the image, and 0
                            for classes that are not in the dataset. 
    '''
    print('\n>>> mrcnn_class_loss_graph ' )
    print('    target_class_ids  size :', target_class_ids.shape)
    print('    pred_class_logits size :', pred_class_logits.shape)
    print('    active_class_ids  size :', active_class_ids.shape)    
    target_class_ids = tf.cast(target_class_ids, 'int64')
    
    # Find predictions of classes that are not in the dataset.
    
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)

    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)
    
    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    loss = loss * pred_active
    print('    loss      :', loss.get_shape(), KB.shape(loss), 'KerasTensor: ', KB.is_keras_tensor(loss))

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    print('    mean loss :', loss.get_shape(), KB.shape(loss), 'KerasTensor: ', KB.is_keras_tensor(loss))
    loss = tf.reshape(loss, [1, 1], name = 'mrcnn_class_loss')
    print('    reshaped mean loss :', loss.get_shape(), KB.shape(loss), 'KerasTensor: ', KB.is_keras_tensor(loss))
    return loss

##-----------------------------------------------------------------------
##  MRCNN bbox loss
##-----------------------------------------------------------------------    
def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    ''' Loss for Mask R-CNN bounding box refinement.

    target_bbox:        [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids:   [batch, num_rois]. Integer class IDs.
    pred_bbox:          [batch, num_rois, num_classes,  4: (dy, dx, log(dh), log(dw))]
    
    '''
    print('\n>>> mrcnn_bbox_loss_graph ' )
    print('    target_class_ids  size :', target_class_ids.shape)
    print('    pred_bbox size         :', pred_bbox.shape)
    print('    target_bbox size       :', target_bbox.shape)    
    
    # Reshape to merge batch and roi dimensions for simplicity.
    # target_class_ids:  reshaped into [ (batch * num_rois) ]
    # target_bbox     :  reshaped into [ (batch * num_rois) , 4]
    # pred_bbox       :  reshaped into [ (batch * num_rois) , num_classes, 4]
    
    target_class_ids = KB.reshape(target_class_ids, (-1,))
    target_bbox      = KB.reshape(target_bbox, (-1, 4))
    pred_bbox        = KB.reshape(pred_bbox, (-1, KB.int_shape(pred_bbox)[2], 4))
    print('    reshpaed pred_bbox size         :', pred_bbox.shape)
    print('    reshaped target_bbox size       :', target_bbox.shape)    

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indicies.
    positive_roi_ix        = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast( tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices                = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox   = tf.gather_nd(pred_bbox, indices)
    print('    pred_bbox size         :', pred_bbox.shape)
    print('    target_bbox size       :', target_bbox.shape)    
    
    # Smooth-L1 Loss
    loss        = KB.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    print('    loss      :', loss.get_shape(), KB.shape(loss), 'KerasTensor: ', KB.is_keras_tensor(loss))
    loss        = KB.mean(loss)
    print('    mean loss :', loss.get_shape(), KB.shape(loss), 'KerasTensor: ', KB.is_keras_tensor(loss))
    loss        = tf.reshape(loss, [1, 1], name = 'mrcnn_bbox_loss')
    print('    reshaped mean loss :', loss.get_shape(), KB.shape(loss), 'KerasTensor: ', KB.is_keras_tensor(loss))
    return loss

##-----------------------------------------------------------------------
##  MRCNN mask loss
##-----------------------------------------------------------------------    
def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks:       [batch, num_rois, height, width].
                        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids:   [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks:         [batch, proposals, height, width, num_classes] float32 tensor
                        with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    print('\n>>> mrcnn_mask_loss_graph ' )
    print('    target_class_ids shape :', target_class_ids.shape)
    print('    target_masks     shape :', target_masks.shape)
    print('    pred_masks       shape :', pred_masks.shape)    
    
    target_class_ids = KB.reshape(target_class_ids, (-1,))
    print('    target_class_ids shape :', target_class_ids.shape)
    
    target_shape     = tf.shape(target_masks)
    print('    target_shape       shape :', target_shape.shape)    
    
    target_masks     = KB.reshape(target_masks, (-1, target_shape[2], target_shape[3]))
    print('    target_masks     shape :', target_masks.shape)        
    
    pred_shape       = tf.shape(pred_masks)
    print('    pred_shape       shape :', pred_shape.shape)        
    
    pred_masks       = KB.reshape(pred_masks, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    print('    pred_masks       shape :', pred_masks.get_shape())        
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix        = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
    indices            = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)
    print('     y_true shape:', y_true.get_shape())
    print('     y_pred shape:', y_pred.get_shape())
    
    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = KB.switch(tf.size(y_true) > 0,
                    KB.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    print('    loss      :', loss.get_shape(), KB.shape(loss), 'KerasTensor: ', KB.is_keras_tensor(loss))
    loss = KB.mean(loss)
    print('    mean loss :', loss.get_shape(), KB.shape(loss), 'KerasTensor: ', KB.is_keras_tensor(loss))
    loss = tf.reshape(loss, [1, 1], name = 'mrcnn_mask_loss')
    print('    reshaped mean loss :', loss.get_shape(), KB.shape(loss), 'KerasTensor: ', KB.is_keras_tensor(loss))
    return loss
 

##-----------------------------------------------------------------------
##  FCN normalized loss
##-----------------------------------------------------------------------    
def fcn_score_loss_graph(input_target,  input_pred):

    '''
    Generate Loss based on Normalized score in PRED_HEATMAP_SCORES and FCN_HEATMAP_SCORES 
    
    Inputs:            
    gt_heatmap_scores   [batch, num_classes, num_rois, 11 ] --> column 9 contains normalized score.
    pred_heatmap:       [batch, num_classes, num_rois, 16 ] --> column 14 contains normalized score
    '''
    pred_scores   = input_pred[:,1:,:,14]
    target_scores = input_target[:,1:,:,9]
    # Reshape for simplicity. Merge first two dimensions into one.
    print('\n>>> fcn_norm_loss_graph ' )
    print('    target_scores shape :', target_scores.shape)
    print('    pred_scores   shape :', pred_scores.shape)    

    target_scores1 = KB.reshape(target_scores, (-1,1))
    print('    target_scores1 shape :', target_scores1.get_shape(), KB.int_shape(target_scores1))        
    pred_scores1   = KB.reshape(pred_scores  , (-1,1))
    print('    pred_scores1  shape :', pred_scores1.get_shape())        

#     # Compute binary cross entropy. If no positive ROIs, then return 0.
#     # shape: [batch, roi, num_classes]
#     # Smooth-L1 Loss
    loss        = KB.switch(tf.size(target_scores1) > 0,
                    smooth_l1_loss(y_true=target_scores1, y_pred=pred_scores1),
                    tf.constant(0.0))
    loss        = KB.mean(loss)
    loss        = tf.reshape(loss, [1, 1], name = 'fcn_norm_loss')
    print('    loss type is :', type(loss))
    return loss

    
    
##-----------------------------------------------------------------------
##  FCN BBOX loss
##-----------------------------------------------------------------------    
def fcn_bbox_loss_graph(target_bbox_deltas, target_class_ids, fcn_bbox_deltas):
    ''' 
    Loss for FCN heatmap corresponding bounding box refinement.

    target_bbox_deltas  :   [batch, num_classes, num_rois, (dy, dx, log(dh), log(dw), class_id, score)]
                            last two are removed for loss calculation via [...,:-2]
    target_class_ids    :   [batch, num_rois]. Integer class IDs.
    fcn_bbox_deltas     :   [batch, num_classes, num_rois,  (dy, dx, log(dh), log(dw))]
    
    '''
    print('\n>>> fcn_bbox_loss_graph ' )
    print('    target_class_ids  :', target_class_ids.shape)
    print('    fcn_bbox_deltas   :', fcn_bbox_deltas.shape)
    print('    target_bbox_deltas    :', target_bbox_deltas.shape)    

    ## Reshape to merge batch and roi dimensions for simplicity.
    class_array   = KB.reshape(target_bbox_deltas[...,-2]  , (-1, 1))
    tgt_bbox      = KB.reshape(target_bbox_deltas[...,:-2] , (-1, 4))
    pred_bbox     = KB.reshape(fcn_bbox_deltas, (-1, 4))
    print('    reshaped class_array            :', class_array.shape)
    print('    reshaped pred_bbox size         :', pred_bbox.shape)
    print('    reshaped target_bbox size       :', tgt_bbox.shape)    

    ## Only positive ROIs contribute to the loss. And only the right 
    ## class_id of each ROI. Get their indicies.

    pos_ix = tf.where(target_bbox_deltas[...,-2] > 0)
 
    ## Gather the deltas (predicted and true) that contribute to loss
  
    # IMPORTANT: THE :-2 IS TO PREVENT ADDITIONAL ELEMENTS FROM BEING COPIED
    y_true = tf.gather_nd(target_bbox_deltas[...,:-2], pos_ix)
    y_pred = tf.gather_nd(fcn_bbox_deltas, pos_ix)
    # print(y_pred.eval(session=sess))
    # print(tf.shape(y_pred).eval(session=sess), tf.shape(y_true).eval(session=sess))    
    print('    y_true shape:', y_true.get_shape())
    print('    y_pred shape:', y_pred.get_shape())

    
    ## Smooth-L1 Loss
    loss        = KB.switch(tf.size(y_true) > 0,
                    smooth_l1_loss(y_true=y_true, y_pred=y_pred),
                    tf.constant(0.0))
    loss        = KB.mean(loss)
    loss        = tf.reshape(loss, [1, 1], name = 'fcn_bbox_loss')
    return loss

    
##-----------------------------------------------------------------------
##  FCN Heatmap Mean Square Error loss
##-----------------------------------------------------------------------    
def fcn_heatmap_MSE_loss_graph(target_heatmap, pred_heatmap):
    """
    Binary cross-entropy loss for the heatmaps.

    target_heatmap:       [batch, height, width, num_classes].
    pred_heatmap:         [batch, height, width, num_classes] 
    """
    print()
    print('-------------------------------' )
    print('>>> fcn_heatmap_MSE_loss_graph ' )
    print('-------------------------------' )
    print('    target_masks :', target_heatmap.get_shape(), KB.int_shape(target_heatmap), 'KerasTensor: ', KB.is_keras_tensor(target_heatmap))
    print('    pred_heatmap :', pred_heatmap.get_shape()  , KB.int_shape(pred_heatmap)  , 'KerasTensor: ', KB.is_keras_tensor(pred_heatmap))
    loss = KLosses.mean_squared_error(target_heatmap[...,1:], pred_heatmap[...,1:])
    loss_mean  = KB.mean(loss)
    loss_final = tf.reshape(loss_mean, [1, 1], name = "fcn_MSE_loss")
    print('    loss         :', loss.get_shape()       , KB.int_shape(loss)       , 'KerasTensor: ', KB.is_keras_tensor(loss))
    print('    loss mean    :', loss_mean.get_shape()  , KB.int_shape(loss_mean)  , 'KerasTensor: ', KB.is_keras_tensor(loss_mean))
    print('    loss final   :', loss_final.get_shape() , KB.int_shape(loss_final) , 'KerasTensor: ', KB.is_keras_tensor(loss_final))

    # Permute predicted & target heatmaps to [N, num_classes, height, width]
    
    # pred_heatmap       = tf.transpose(pred_heatmap, [0, 3, 1, 2])    
    # target_heatmap     = tf.transpose(target_heatmap, [0, 3, 1, 2])    
    # print('    target_heatmap :', target_heatmap.get_shape(), KB.shape(target_heatmap), 'KerasTensor: ', KB.is_keras_tensor(target_heatmap))
    # print('    pred_heatmap   :', pred_heatmap.get_shape()  , KB.shape(pred_heatmap)  , 'KerasTensor: ', KB.is_keras_tensor(pred_heatmap))
    
    # Reshape predicted & target heatmaps to [N * num_classes, height, width]
    
    # target_shape     = KB.shape(target_heatmap)
    # target_heatmap     = KB.reshape(target_heatmap, (-1, target_shape[1], target_shape[2]))
    # pred_shape       = KB.shape(pred_heatmap)
    # pred_heatmap       = KB.reshape(pred_heatmap, (-1, pred_shape[1], pred_shape[2]))

    # print('    target_shape     shape :', target_shape.shape)    
    # print('    target_heatmap     shape :', target_heatmap.shape)        
    # print('    pred_shape       shape :', pred_shape.shape)        
    # print('    pred_heatmap       shape :', pred_heatmap.get_shape())        

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    # Smooth-L1 Loss
    # loss        = KB.switch(tf.size(target_heatmap) > 0,
                    # smooth_l1_loss(y_true=target_heatmap, y_pred=pred_heatmap),
                    # KB.constant(0.0))
    # print('    loss is keras tensor:', KB.is_keras_tensor(loss))                    
    # loss        = KB.mean(loss)
    # loss        = tf.reshape(loss, [1, 1], name = 'fcn_loss')
    
    # loss = KB.binary_crossentropy(target_heatmap[...,1:], pred_heatmap[...,1:], from_logits=True)
    # loss = KB.categorical_crossentropy(target_heatmap[...,1:], pred_heatmap[...,1:], from_logits=True)
    # loss = KLosses.mean_absolute_error(target_heatmap[...,1:], pred_heatmap[...,1:])
    # loss = KLosses.squared_hinge(target_heatmap[...,1:], pred_heatmap[...,1:])
    # loss = KLosses.hinge(target_heatmap[...,1:], pred_heatmap[...,1:])
    
                    
    return loss_final
    

##-----------------------------------------------------------------------
##  FCN Categorical Cross Entropy loss  
##-----------------------------------------------------------------------    
def fcn_heatmap_CE_loss_graph(target_heatmap, pred_heatmap, active_class_ids):
    '''
    Categorical Cross Entropy Loss for the FCN heatmaps.

    target_class_ids:       [batch, num_rois]. Integer class IDs. Uses zero
                            padding to fill in the array.
    
    pred_class_logits:      [batch, num_rois, num_classes]
    
    active_class_ids:       [batch, num_classes]. Has a value of 1 for
                            classes that are in the dataset of the image, and 0
                            for classes that are not in the dataset. 
    '''
    print()
    print('-------------------------------' )
    print('>>> fcn_heatmap_CE_loss_graph  ' )
    print('-------------------------------' )
    print('    target_class_ids  :', KB.int_shape(target_heatmap))
    print('    pred_class_logits :', KB.int_shape(pred_heatmap))
    print('    active_class_ids  :', KB.int_shape(active_class_ids))
    # target_class_ids = tf.cast(target_class_ids, 'int64')
    
    # Find predictions of classes that are not in the dataset.
    pred_class_ids = KB.argmax(pred_heatmap  , axis=-1)
    gt_class_ids   = KB.argmax(target_heatmap, axis=-1)
    print('    pred_class_ids    :', KB.int_shape(pred_class_ids), pred_class_ids.dtype ) 
    print('    gt_class_ids      :', KB.int_shape(gt_class_ids  ), gt_class_ids.dtype) 

    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)
    print('    pred_active       :', KB.int_shape(pred_active),  pred_active.dtype)  
    
    # Loss
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_heatmap, logits=pred_heatmap)
    print('    loss              :', KB.int_shape(loss), loss.dtype)    

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    loss = loss * pred_active
    print('    loss*pred_active  :', KB.int_shape(loss), 'KerasTensor: ', KB.is_keras_tensor(loss))

    # Compute  loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_sum(loss)   ##/ tf.reduce_sum(pred_active)
    loss_mean  = KB.mean(loss)
    loss_final = tf.reshape(loss_mean, [1, 1], name = "fcn_CE_loss")
    
    print('    loss              :', loss.get_shape()       , KB.int_shape(loss)       , 'KerasTensor: ', KB.is_keras_tensor(loss))
    print('    loss mean         :', loss_mean.get_shape()  , KB.int_shape(loss_mean)  , 'KerasTensor: ', KB.is_keras_tensor(loss_mean))
    print('    loss final        :', loss_final.get_shape() , KB.int_shape(loss_final) , 'KerasTensor: ', KB.is_keras_tensor(loss_final))
    
    return loss_final


##-----------------------------------------------------------------------
##  FCN Binary Cross Entropy loss  
##-----------------------------------------------------------------------    
def fcn_heatmap_BCE_loss_graph(target_heatmap, pred_heatmap):
    '''
    Binary Cross Entropy Loss for the FCN heatmaps.
    
    Apply a per-pixel sigmoid and binary loss, similar to the Lmask loss calculation
    in MaskRCNN. 
    Two approaches :
    1- Only calaculate loss for classes which have active GT bounding boxes
    2- Calculate for all classes 
    
    We will implement approach 1. 
    
    
    target_heatmaps:    [batch, height, width, num_classes].
                        A float32 tensor of values 0 or 1. Uses zero padding to fill array.

    target_class_ids:   [batch, num_rois]. Integer class IDs. Zero padded.

    pred_masks:         [batch, height, width, num_classes]  float32 tensor
                        with values from 0 to 1.

    # active_class_ids:       [batch, num_classes]. Has a value of 1 for
                            # classes that are in the dataset of the image, and 0
                            # for classes that are not in the dataset. 
    '''
    print()
    print('-------------------------------' )
    print('>>> fcn_heatmap_BCE_loss_graph  ' )
    print('-------------------------------' )
    logt('    target_class_ids  :', target_heatmap)
    logt('    pred_class_logits :', pred_heatmap)
    # target_class_ids = tf.cast(target_class_ids, 'int64')
    
    # Find predictions of classes that are active (present in the GT heatmaps)  
    target_heatmap = tf.transpose(target_heatmap, [0,3,1,2])
    pred_heatmap   = tf.transpose(  pred_heatmap, [0,3,1,2])
    logt(' trgt_heatmap ', target_heatmap)
    logt(' trgt_heatmap ', pred_heatmap  )

    tgt_hm_sum = tf.reduce_sum(target_heatmap, axis = [2,3])
    logt(' tgt_hm_sum ',tgt_hm_sum)

    class_idxs = tf.where(tgt_hm_sum > 0)
    logt(' class indeixes ', class_idxs)

    active_tgt_heatmaps  = tf.gather_nd(target_heatmap, class_idxs)
    active_pred_heatmaps = tf.gather_nd(pred_heatmap, class_idxs)
    logt('active_tgt_heatmaps  ',active_tgt_heatmaps)
    logt('active_pred_heatmaps ',active_pred_heatmaps)
    y_true = tf.reshape(active_tgt_heatmaps, (-1,))
    y_pred = tf.reshape(active_pred_heatmaps, (-1,))
    logt('y_true : ', y_true)
    logt('y_pred : ', y_pred)

    loss = KB.switch(tf.size(y_true) > 0,
                    KB.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    logt('loss', loss)
    loss_mean = KB.mean(loss)
    logt('mean loss ', loss_mean)  
    loss_final = tf.reshape(loss_mean, [1, 1], name = 'fcn_BCE_loss')
    logt('loss (final) ', loss_final)
    # return loss    
    print('    loss              :', loss.get_shape()       , KB.int_shape(loss)       , 'KerasTensor: ', KB.is_keras_tensor(loss))
    print('    loss mean         :', loss_mean.get_shape()  , KB.int_shape(loss_mean)  , 'KerasTensor: ', KB.is_keras_tensor(loss_mean))
    print('    loss final        :', loss_final.get_shape() , KB.int_shape(loss_final) , 'KerasTensor: ', KB.is_keras_tensor(loss_final))
    
    return loss_final

    
    
##-----------------------------------------------------------------------
##  FCN Categorical Cross Entropy loss  
##-----------------------------------------------------------------------    
def fcn_heatmap_CE_loss_graph_2(target_heatmap, pred_heatmap, active_class_ids):
    '''
    Categorical Cross Entropy Loss for the FCN heatmaps.

    target_class_ids:       [batch, num_rois]. Integer class IDs. Uses zero
                            padding to fill in the array.
    
    pred_class_logits:      [batch, num_rois, num_classes]
    
    active_class_ids:       [batch, num_classes]. Has a value of 1 for
                            classes that are in the dataset of the image, and 0
                            for classes that are not in the dataset. 
    '''
    print()
    print('--------------------------------' )
    print('>>> fcn_heatmap_CE_loss_graph_2 ' )
    print('--------------------------------' )
    logt('target_class_ids  ', target_heatmap)
    logt('pred_class_logits ', pred_heatmap  )
    logt('active_class_ids  ', active_class_ids)
    # target_class_ids = tf.cast(target_class_ids, 'int64')
    
    # Find predictions of classes that are not in the dataset.
    pred_class_ids = KB.argmax(pred_heatmap  , axis=-1)
    gt_class_ids   = KB.argmax(target_heatmap, axis=-1)
    logt('pred_class_ids    ', pred_class_ids) 
    logt('gt_class_ids      ', gt_class_ids  ) 

    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)
    
    # Loss
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_heatmap, logits=pred_heatmap)
    logt('pred_active       ', pred_active)
    logt('loss              ', loss)    

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    # loss = loss * pred_active
    # print('loss*pred_active ', loss)

    # Compute  loss mean. Use only predictions that contribute to the loss to get a correct mean.
    loss = tf.reduce_sum(loss)   ##/ tf.reduce_sum(pred_active)
    loss_mean  = KB.mean(loss)
    loss_final = tf.reshape(loss_mean, [1, 1], name = "fcn_CE_loss")
    
    logt('loss      ', loss)
    logt('loss mean ', loss_mean)
    logt('loss final', loss_final)
    
    return loss_final
    
    
    
    
    
    
"""    
#-----------------------------------------------------------------------
#  RPN bbox loss - obsolete -- use rpn_bbox_loss_graph()
#-----------------------------------------------------------------------    
# def rpn_bbox_loss_graph_old(config, target_bbox, rpn_match, rpn_bbox):
    # '''
    # Return the RPN bounding box loss graph.

    # config:             the model config object.
    
    # target_bbox:        [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
                        # Uses 0 padding to fill in unsed bbox deltas.
    
    # rpn_match:          [batch, anchors, 1]. Anchor match type. 1=positive,
                        # -1=negative, 0=neutral anchor.
    
    # rpn_bbox:           [batch, anchors, (dy, dx, log(dh), log(dw))]
    
    # '''
    # # Positive anchors contribute to the loss, but negative and
    # # neutral anchors (match value of 0 or -1) don't.
    # rpn_match = KB.squeeze(rpn_match, -1)
    # indices   = tf.where(KB.equal(rpn_match, 1))

    # print('>>> rpn_bbox_loss_graph_old' )
    # print('    rpn_match size    : ', rpn_match.shape)
    # print('    rpn_bbox  size    : ', rpn_bbox.shape)
    # print('    target_bbox size n: ', target_bbox.shape)
    
    # # Pick bbox deltas that contribute to the loss
    # rpn_bbox  = tf.gather_nd(rpn_bbox, indices)

    # # Trim target bounding box deltas to the same length as rpn_bbox.
    # batch_counts = KB.sum(KB.cast(KB.equal(rpn_match, 1), tf.int32), axis=1)
    # target_bbox  = utils.batch_pack_graph(target_bbox, batch_counts,
                                   # config.IMAGES_PER_GPU)

    # # TODO: use smooth_l1_loss() rather than reimplementing here
    # #       to reduce code duplication
    # diff = KB.abs(target_bbox - rpn_bbox)
    # less_than_one = KB.cast(KB.less(diff, 1.0), "float32")
    # loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    # loss = KB.switch(tf.size(loss) > 0, KB.mean(loss), tf.constant(0.0))
    
    # return loss
"""
    