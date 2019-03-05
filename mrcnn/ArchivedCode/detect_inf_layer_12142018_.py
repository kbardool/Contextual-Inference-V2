"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

 
import numpy as np
import tensorflow as tf
import keras.backend as KB
import keras.engine as KE
from mrcnn.utils import apply_box_deltas_np, non_max_suppression, logt
import pprint

pp = pprint.PrettyPrinter(indent=2, width=100)
np.set_printoptions(linewidth=100)


############################################################
##  Inference mode Detection Layer
############################################################

def clip_to_window(window, boxes):
    '''
    window: (y1, x1, y2, x2). The window in the image we want to clip to.
    boxes: [N, (y1, x1, y2, x2)]
    '''
    
    boxes[:, 0] = np.maximum(np.minimum(boxes[:, 0], window[2]), window[0])
    boxes[:, 1] = np.maximum(np.minimum(boxes[:, 1], window[3]), window[1])
    boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], window[2]), window[0])
    boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], window[3]), window[1])
    return boxes


def refine_detections(rois, probs, deltas, window, config):
    '''
    Refine classified proposals and filter overlaps and return final detections.

    Inputs:
    ------
        
    rois:           rpn_rois    - [N, (y1, x1, y2, x2)] in normalized coordinates
    
                    passed from PROPOSAL_LAYER
                                  
    probs:          mrcnn_class - [N, num_classes]. Class probabilities.
    deltas:         mrcnn_bbox  - [N, num_classes, (dy, dx, log(dh), log(dw))]. 
                                  Class-specific bounding box deltas.
                    
                    passed from FPN_CLASSIFIER_GRAPH              
    window:         
    (y1, x1, y2, x2) in image coordinates. The part of the image
                    that contains the image excluding the padding.

    Returns:
    --------
    detections      [M, (y1, x1, y2, x2, class_id, score)]
                    M - determined by DETECTION_MAX_INSTANCES
                    
                    detection bounding boxes -- these have had the corresponding 
                    deltas applied, and their boundries clipped to the image window
    '''

    
    ##----------------------------------------------------------------------------
    ##  1. Find Class IDs with higest scores for each per ROI
    ##----------------------------------------------------------------------------
    class_ids       = np.argmax(probs, axis=1)
    
    ##----------------------------------------------------------------------------
    ##  2. Get Class probability(score) and bbox delta of the top class of each ROI
    ##----------------------------------------------------------------------------
    class_scores    =  probs[np.arange(class_ids.shape[0]), class_ids]
    deltas_specific = deltas[np.arange(deltas.shape[0])   , class_ids]
    
    ##----------------------------------------------------------------------------
    ##  3. Apply bounding box delta to the corrsponding rpn_proposal
    ##----------------------------------------------------------------------------
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois    = apply_box_deltas_np(rois, deltas_specific * config.BBOX_STD_DEV)
    
    ##----------------------------------------------------------------------------
    ##  4. Convert the refined roi coordiates from normalized to NN image domain
    ##  5.  Clip boxes to image window
    ##  6.  Round and cast to int since we're deadling with pixels now
    ##----------------------------------------------------------------------------
    # TODO: better to keep them normalized until later   
    height, width   = config.IMAGE_SHAPE[:2]
    refined_rois   *= np.array([height, width, height, width])
    refined_rois    = clip_to_window(window, refined_rois)
    refined_rois    = np.rint(refined_rois).astype(np.int32)

    ##----------------------------------------------------------------------------
    ##  7.  TODO: Filter out boxes with zero area
    ##----------------------------------------------------------------------------

    ##----------------------------------------------------------------------------
    ##  8.  Filter out background boxes
    ##      keep : contains indices of non-zero elements in class_ids
    ##      config.DETECTION_MIN_CONFIDENCE == 0 
    ##      np.intersect: find indices into class_ids that satisfy:
    ##        -  class_id     >  0 
    ##        -  class_scores >=            config.DETECTION_MIN_CONFIDENCE (0.3)
    ##----------------------------------------------------------------------------
    keep = np.where(class_ids > 0)[0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        keep = np.intersect1d(keep, np.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[0])

    ##----------------------------------------------------------------------------
    ##  9.  Apply per-class NMS
    ##----------------------------------------------------------------------------
    pre_nms_class_ids = class_ids[keep]
    pre_nms_scores    = class_scores[keep]
    pre_nms_rois      = refined_rois[keep]
    nms_keep          = []
    # print(' apply per class nms')    
    for class_id in np.unique(pre_nms_class_ids):
        # Pick detections of this class
        ixs = np.where(pre_nms_class_ids == class_id)[0]

        # print('class_id : ', class_id)
        # print('pre_nms_rois.shape:', pre_nms_rois[ixs].shape)
        # pp.pprint(pre_nms_rois[ixs])
        # print('pre_nms_scores.shape :', pre_nms_scores[ixs].shape)
        # pp.pprint(pre_nms_scores[ixs])    
        # Apply NMS
        class_keep = non_max_suppression(pre_nms_rois[ixs], 
                                         pre_nms_scores[ixs],
                                         config.DETECTION_NMS_THRESHOLD)
        # Map indicies
        class_keep = keep[ixs[class_keep]]
        nms_keep   = np.union1d(nms_keep, class_keep)
    
    keep = np.intersect1d(keep, nms_keep).astype(np.int32)

    ##----------------------------------------------------------------------------
    ## 10.  Keep top detections
    ##----------------------------------------------------------------------------
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids   = np.argsort(class_scores[keep])[::-1][:roi_count]
    keep      = keep[top_ids]

    ##----------------------------------------------------------------------------
    ## 11.  Add a detect_ind = +1 to differentiate from false postives added in eval layer
    ##----------------------------------------------------------------------------
    detect_ind    = np.ones((top_ids.shape[0],1))
 

    ##----------------------------------------------------------------------------
    ## 11.  Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    ##      Coordinates are in image domain.
    ##----------------------------------------------------------------------------
    result = np.hstack((refined_rois[keep],
                        class_ids   [keep][..., np.newaxis],
                        class_scores[keep][..., np.newaxis],
                        detect_ind))

    return result


class DetectionInferenceLayer(KE.Layer):
    '''
    Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.
    Input:
    -------
    rpn_proposal_rois:   [batch, N, (y1, x1, y2, x2)] in normalized coordinates.
                           Proposal bboxes generated by RPN
                           Might be zero padded if there are not enough proposals.
    
    mrcnn_class : 
    mrcnn_bbox  :  
    input_image_meta:
    
    
    Returns:
    --------
    
    detections:         [batch, num_max_detections, 6 (y1, x1, y2, x2, class_id, class_score)] in pixels
    
    '''

    def __init__(self, config=None, **kwargs):
        # super(DetectionLayer, self).__init__(**kwargs)
        super().__init__(**kwargs)
        print('\n>>> Detection Layer (Inference Mode)')
        self.config  = config
        self.verbose = config.VERBOSE

    def call(self, inputs):
        print('    Detection Layer : call() ', type(inputs), len(inputs))    
        logt('rpn_proposals_roi ',  inputs[0], verbose = self.verbose)
        logt('mrcnn_class.shape ',  inputs[1], verbose = self.verbose) 
        logt('mrcnn_bboxes.shape',  inputs[2], verbose = self.verbose)
        logt('input_image_meta  ',  inputs[3], verbose = self.verbose) 
    
        def wrapper(rois, mrcnn_class, mrcnn_bbox, image_meta):
            from mrcnn.utils import parse_image_meta
            detections_batch = []
            # logt('detection wrapper - rpn_proposals_roi  ',  rois       , verbose = self.verbose)
            # logt('detection wrapper - mrcnn_class.shape  ',  mrcnn_class, verbose = self.verbose)
            # logt('detection wrapper - mrcnn_bboxes.shape ',  mrcnn_bbox , verbose = self.verbose)
            # logt('detection wrapper - image_meta         ',  image_meta , verbose = self.verbose)
            # process item per item in batch 
            
            for b in range(self.config.BATCH_SIZE):
                _, _, window, _ =  parse_image_meta(image_meta)

                detections = refine_detections(rois[b], mrcnn_class[b], mrcnn_bbox[b], window[b], self.config)
                # if self.verbose:
                    # print('\n\n config.DETECTION_MAX_INSTANCES: ', self.config.DETECTION_MAX_INSTANCES)
                    # print(' Detections shape:', detections.shape)
                # print(detections)
                
                # Pad with zeros if detections < DETECTION_MAX_INSTANCES
                gap = self.config.DETECTION_MAX_INSTANCES - detections.shape[0]
                assert gap >= 0
                if gap > 0:
                    detections = np.pad(detections, [(0, gap), (0, 0)], 'constant', constant_values=0)

                detections_batch.append(detections)

            # Stack detections and cast to float32
            # TODO: track where float64 is introduced
            detections_batch = np.array(detections_batch).astype(np.float32)
            num_columns = detections_batch.shape[-1]

            # Reshape output
            # [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
            return np.reshape(detections_batch, [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, num_columns])

        # Return wrapped function
        return tf.py_func(wrapper, inputs, tf.float32, name="detections")

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 7)