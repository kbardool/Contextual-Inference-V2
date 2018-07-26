"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
## Generate FCN model using FCN graph that does NOT have 
## L2 normalization layers (import fcn_layer_no_L2)
##
##
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
from collections import OrderedDict
import numpy as np
import pprint
import scipy.misc
import tensorflow as tf

import keras
import keras.backend as KB
import keras.layers  as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM
#sys.path.append('..')

import mrcnn.utils            as utils
import mrcnn.loss             as loss
from   mrcnn.datagen          import data_generator
from   mrcnn.utils            import log
from   mrcnn.utils            import parse_image_meta_graph, parse_image_meta

from   mrcnn.RPN_model        import build_rpn_model
from   mrcnn.resnet_model     import resnet_graph

from   mrcnn.chm_layer        import CHMLayer
from   mrcnn.chm_inf_layer    import CHMLayerInference
from   mrcnn.proposal_layer   import ProposalLayer

from   mrcnn.fcn_layer_no_L2         import fcn_graph
from   mrcnn.fcn_scoring_layer import FCNScoringLayer
from   mrcnn.detect_layer      import DetectionLayer  
from   mrcnn.detect_tgt_layer_mod import DetectionTargetLayer_mod

from   mrcnn.fpn_layers       import fpn_graph, fpn_classifier_graph, fpn_mask_graph
from   mrcnn.callbacks        import get_layer_output_1,get_layer_output_2
from   mrcnn.callbacks        import MyCallback
from   mrcnn.batchnorm_layer  import BatchNorm

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3.0")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')
pp = pprint.PrettyPrinter(indent=4, width=100)
tf.get_variable_scope().reuse_variables()

############################################################
#  Code                                     moved to 
#  -----------------------------------      ----------------
#  BatchNorm                              batchnorm_layer.py      
#  Miscellenous Graph Functions                     utils.py 
#  Loss Functions                                    loss.py
#  Data Generator                                 datagen.py
#  Data Formatting                                  utils.py
#  Proposal Layer                          proposal_layer.py
#  ROIAlign Layer                         roiialign_layer.py
#  FPN Layers                                   fpn_layer.py
#  FPN Head Layers                         fpnhead_layers.py
#  Detection Target Layer                detect_tgt_layer.py
#  Detection Layer                        detection_layer.py
#  Region Proposal Network (RPN)                rpn_model.py
#  Resnet Graph                              resnet_model.py
############################################################
from keras.callbacks import TensorBoard

class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': KB.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)
        
############################################################
##  FCN Class
############################################################
class FCN():
    """Encapsulates the FCN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir, FCN_layers = True):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']

        print('>>> Initialize FCN model, mode: ',mode)

        self.mode      = mode
        self.config    = config
        self.model_dir = model_dir
        self.set_log_dir()
        # Pre-defined layer regular expressions
        self.layer_regex = {
            # ResNet from a specific stage and up
            "res3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)",
            "res4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)",
            "res5+": r"(res5.*)|(bn5.*)",

            # fcn only 
            "fcn" : r"(fcn\_.*)",
            # fpn
            "fpn" : r"(fpn\_.*)",
            # rpn
            "rpn" : r"(rpn\_.*)",
            # rpn
            "mrcnn" : r"(mrcnn\_.*)",

            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # all layers but the backbone
            "allheads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(fcn\_.*)",
          
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        

        self.keras_model = self.build(mode=mode, config=config, FCN_layers = FCN_layers)

        print('>>> FCN initialization complete. mode: ',mode) 

    
    def build(self, mode, config, FCN_layers = False):
        '''
        Build MODIFIED Mask R-CNN architecture (NO MASK PROCESSING)
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        '''
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")
        num_classes = config.NUM_CLASSES
        num_bboxes  = config.DETECTION_MAX_INSTANCES   # 100
        num_bboxes  = config.TRAIN_ROIS_PER_IMAGE      # 32



        ##------------------------------------------------------------------                            
        ##  Input Layer
        ##------------------------------------------------------------------
        # input_image      = KL.Input(shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        # input_image_meta = KL.Input(shape=[None], name="input_image_meta")

        pr_hm_norm   = KL.Input(shape=[h,w, num_classes], name="input_pr_hm_norm" , dtype=tf.float32 )
        pr_hm_scores = KL.Input(shape=[num_classes, num_bboxes, 11], name="input_pr_hm_scores", dtype=tf.float32)
        
        if mode == "training":

            gt_hm_norm   = KL.Input(shape=[h,w, num_classes], name="input_gt_hm_norm"  , dtype=tf.float32)
            gt_hm_scores = KL.Input(shape=[num_classes, num_bboxes, 11], name="input_gt_hm_scores", dtype=tf.float32)
            
           
        ## End if mode == 'training'
        
        ##----------------------------------------------------------------------------                
        ## Training Mode Layers
        ##----------------------------------------------------------------------------                
        if mode == "training":
            ##------------------------------------------------------------------------
            ##  FCN Network Head
            ##------------------------------------------------------------------------
            if FCN_layers :
                print('\n')
                print('---------------------------------------------------')
                print('    Adding  FCN layers')
                print('---------------------------------------------------')
            
                fcn_hm_norm, fcn_hm,  _ = fcn_graph(pr_hm_norm, config)
                # fcn_heatmap_norm = fcn_graph(pred_heatmap, config)
                print('   fcn_heatmap      : ', KB.int_shape(fcn_hm), ' Keras tensor ', KB.is_keras_tensor(fcn_hm) )        
                print('   fcn_heatmap_norm : ', KB.int_shape(fcn_hm_norm), ' Keras tensor ', KB.is_keras_tensor(fcn_hm_norm) )        
                         
                fcn_hm_scores = FCNScoringLayer(config, name='fcn_scoring') ([fcn_hm_norm, pr_hm_scores])
                
                ##------------------------------------------------------------------------
                ##  Loss layer definitions
                ##------------------------------------------------------------------------
                print('\n')
                print('---------------------------------------------------')
                print('    building Loss Functions ')
                print('---------------------------------------------------')
                # print(' gt_deltas         :', KB.is_keras_tensor(gt_deltas)      , KB.int_shape(gt_deltas        ))
                # print(' target_class_ids  :', KB.is_keras_tensor(target_class_ids), KB.int_shape(target_class_ids ))
                
                fcn_norm_loss  = KL.Lambda(lambda x: loss.fcn_norm_loss_graph(*x),  name="fcn_norm_loss") \
                                 ([gt_hm_scores, fcn_hm_scores])
                # fcn_score_loss = KL.Lambda(lambda x: loss.fcn_loss_graph(*x), name="fcn_loss") \
                                # ([gt_hm, fcn_hm_norm])
                                
            # Model Inputs 
            inputs  = [ pr_hm_norm, pr_hm_scores, gt_hm_norm, gt_hm_scores]
            outputs = [fcn_hm_norm, fcn_hm_scores, fcn_hm, fcn_norm_loss]

        # end if Training
        ##----------------------------------------------------------------------------                
        ##    Inference Mode
        ##----------------------------------------------------------------------------                
        else:
            ##------------------------------------------------------------------------
            ##  FPN Layer
            ##------------------------------------------------------------------------
            # Network Heads
            # Proposal classifier and BBox regressor heads
            # mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                # fpn_classifier_graph(rpn_proposal_rois, mrcnn_feature_maps, config.IMAGE_SHAPE,
                                     # config.POOL_SIZE, config.NUM_CLASSES)

            ##------------------------------------------------------------------------
            ##  Detetcion Layer
            ##------------------------------------------------------------------------
            #  Generate detection targets
            #    generated RPNs + mrcnn predictions ----> Target ROIs
            #
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in image coordinates
            #------------------------------------------------------------------------           
            # detections = DetectionLayer(config, name="mrcnn_detection")\
                                        # ([rpn_proposal_rois, mrcnn_class, mrcnn_bbox, input_image_meta])
            # print('<<<  shape of DETECTIONS : ', KB.int_shape(detections), ' Keras tensor ', KB.is_keras_tensor(detections) )                         
            
            # Convert boxes to normalized coordinates
            # TODO: let DetectionLayer return normalized coordinates to avoid unnecessary conversions
            # h, w = config.IMAGE_SHAPE[:2]
            # detection_boxes = KL.Lambda(lambda x: x[..., :4] / np.array([h, w, h, w]))(detections)
            # print('<<<  shape of DETECTION_BOXES : ', KB.int_shape(detection_boxes),
                  # ' Keras tensor ', KB.is_keras_tensor(detection_boxes) )                         

            ##------------------------------------------------------------------------
            ##  FPN Mask Layer
            ##------------------------------------------------------------------------
            # Create masks for detections
            # mrcnn_mask = fpn_mask_graph(detection_boxes,
                                        # mrcnn_feature_maps,
                                        # config.IMAGE_SHAPE,
                                        # config.MASK_POOL_SIZE,
                                        # config.NUM_CLASSES)

            ##---------------------------------------------------------------------------
            ## CHM Inference Layer(s) to generate contextual feature maps using outputs from MRCNN 
            ##----------------------------------------------------------------------------         
            # pr_hm_norm,  pr_hm_scores, pr_tensor, pr_hm =\
                                     # CHMLayerInference(config, name = 'cntxt_layer' ) ([detections])
                                
            # print('<<<  shape of pred_tensor   : ', pr_tensor.shape, ' Keras tensor ', KB.is_keras_tensor(pr_tensor) )                         
                                        
            ##------------------------------------------------------------------------
            ## FCN Network Head
            ##------------------------------------------------------------------------
            print('---------------------------------------------------')
            print('    Adding  FCN layers')
            print('---------------------------------------------------')
                    
            fcn_hm_norm, fcn_hm,  _ = fcn_graph(pr_hm_norm, config)
            # fcn_heatmap_norm = fcn_graph(pred_heatmap, config)
            print('   fcn_heatmap      : ', KB.int_shape(fcn_hm), ' Keras tensor ', KB.is_keras_tensor(fcn_hm) )        
            print('   fcn_heatmap_norm : ', KB.int_shape(fcn_hm_norm), ' Keras tensor ', KB.is_keras_tensor(fcn_hm_norm) )        

            fcn_hm_scores = FCNScoringLayer(config, name='fcn_scoring') ([fcn_hm_norm, pr_hm_scores])            
            # fcn_hm_norm = fcn_graph(pr_hm, config)
            # print('   fcn_heatmap_norm  shape is : ', KB.int_shape(fcn_hm_norm), ' Keras tensor ', KB.is_keras_tensor(fcn_hm_norm) )        

            inputs = [ pr_hm_norm, pr_hm_scores]                                        
            # inputs  = [ input_image, input_image_meta]
            outputs = [ fcn_hm_norm, fcn_hm_scores, fcn_hm]
            # end if Inference Mode        
        model = KM.Model( inputs, outputs,  name='FCN')

        print(' ================================================================')
        print(' self.keras_model.losses : ', len(model.losses))
        print(model.losses)
        print(' ================================================================')
        
        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        print('\n>>> FCN build complete. mode: ',mode)
        return model

        
##-------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------        
##-------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------        
        
        
        
                
    def detect(self, images, verbose=0):
        '''
        Runs the detection pipeline.

        images:         List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois:           [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids:      [N] int class IDs
        scores:         [N] float probability scores for the class IDs
        masks:          [H, W, N] instance binary masks
        '''
        # print('>>> model detect()')
        
        assert self.mode   == "inference", "Create model in inference mode."
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)
                
        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)
        if verbose:
            log("molded_images", molded_images)
            log("image_metas"  , image_metas)
            
        ## Run object detection pipeline
        # print('    call predict()')
        # rpn_proposal_rois, rpn_class, rpn_bbox,\
        # detections, \
        # mrcnn_class, mrcnn_bbox,  \
        # pr_hm_norm, pr_hm_scores, \
        fcn_hm_norm, fcn_hm_scores, fcn_hm = \
                  self.keras_model.predict([pr_hm_norm, pr_hm_scores], verbose=0)
            
        print('    return from  predict()')
        print('    Length of detections : ', len(detections))
        # print('    detections \n', detections)
        # print('    Length of rpn_proposal_rois   : ', len(rpn_proposal_rois   ))
        # print('    Length of rpn_class  : ', len(rpn_class  ))
        # print('    Length of rpn_bbox   : ', len(rpn_bbox   ))
        # print('    Length of mrcnn_class: ', len(mrcnn_class))
        # print('    Length of mrcnn_bbox : ', len(mrcnn_bbox ))
        # print('    Length of mrcnn_mask : ', len(mrcnn_mask ))

        # Process detections
        results = []
        for i, image in enumerate(images):
            # , final_masks =\
            final_rois, final_class_ids, final_scores, \
            final_pre_scores, final_fcn_scores          \
              =  self.unmold_detections_new(fcn_hm_scores[i],        # detections[i], 
                                       image.shape  ,
                                       windows[i])    
           # mrcnn_mask[i],

            results.append({
                "rois"        : final_rois,
                "class_ids"   : final_class_ids,
                "scores"      : final_scores,
                "pre_scores"  : final_pre_scores,
                "fcn_scores"  : final_fcn_scores,
                "pre_hm_norm" : pr_hm_norm,
                'fcn_hm_norm' : fcn_hm_norm,
                'fcn_hm'      : fcn_hm
                # "masks"    : final_masks,
            })
        return results 


    def mold_inputs(self, images):
        '''
        Takes a list of images and modifies them to the format expected as an 
        input to the neural network.
        
        - resize to IMAGE_MIN_DIM / IMAGE_MAX_DIM  : utils.RESIZE_IMAGE()
        - subtract mean pixel vals from image pxls : utils.MOLD_IMAGE()
        - build numpy array of image metadata      : utils.COMPOSE_IMAGE_META()
        
        images       :     List of image matricies [height,width,depth]. Images can have
                           different sizes.

        Returns 3 Numpy matrices:
        
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas  : [N, length of meta data]. Details about each image.
        windows      : [N, (y1, x1, y2, x2)]. The portion of the image that has the
                       original image (padding excluded).
        '''
        
        molded_images = []
        image_metas   = []
        windows       = []
        
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                padding=self.config.IMAGE_PADDING)
            
            # subtract mean pixel values from image pixels
            molded_image = utils.mold_image(molded_image, self.config)
            
            # Build image_meta
            image_meta = utils.compose_image_meta( 0, 
                                                   image.shape, 
                                                   window,
                                                   np.zeros([self.config.NUM_CLASSES],
                                                   dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            image_metas.append(image_meta)
            windows.append(window)
        
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas   = np.stack(image_metas)
        windows       = np.stack(windows)
        return molded_images, image_metas, windows

        
    # def unmold_detections(self, detections, mrcnn_mask, image_shape, window):
    def unmold_detections(self, detections, image_shape, window):
        '''
        Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the application.

        detections  : [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_mask  : [N, height, width, num_classes]
        image_shape : [height, width, depth] Original size of the image before resizing
        window      : [y1, x1, y2, x2] Box in the image where the real image is
                       (i.e.,  excluding the padding surrounding the real image)

        Returns:
        boxes       : [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids   : [N] Integer class IDs for each bounding box
        scores      : [N] Float probability scores of the class_id
        masks       : [height, width, num_instances] Instance masks
        '''
        
        # print('>>>  unmold_detections ')
        # print('     detections.shape : ', detections.shape)
        # print('     mrcnn_mask.shape : ', mrcnn_mask.shape)
        # print('     image_shape.shape: ', image_shape)
        # print('     window.shape     : ', window)
        # print(detections)
        
        # How many detections do we have?
        # Detections array is padded with zeros. detections[:,4] identifies the class 
        # Find all rows in detection array with class_id == 0 , and place their row indices
        # into zero_ix. zero_ix[0] will identify the first row with class_id == 0.
        print()
        np.set_printoptions(linewidth=100)        

        zero_ix = np.where(detections[:, 4] == 0)[0]
    
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # print(' np.where() \n', np.where(detections[:, 4] == 0))
        # print('     zero_ix.shape     : ', zero_ix.shape)
        # print('     N is :', N)
        
        # Extract boxes, class_ids, scores, and class-specific masks
        boxes     = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores    = detections[:N, 5]
        # masks     = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Compute scale and shift to translate coordinates to image domain.
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale   = min(h_scale, w_scale)
        shift   = window[:2]  # y, x
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

        # Translate bounding boxes to image domain
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

        # Filter out detections with zero area. Often only happens in early
        # stages of training when the network weights are still a bit random.
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
            
        if exclude_ix.shape[0] > 0:
            boxes     = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores    = np.delete(scores, exclude_ix, axis=0)
            # masks     = np.delete(masks, exclude_ix, axis=0)
            N         = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        # full_masks = []
        # for i in range(N):
            # Convert neural network mask to full size mask
            # full_mask = utils.unmold_mask(masks[i], boxes[i], image_shape)
            # full_masks.append(full_mask)
        # full_masks = np.stack(full_masks, axis=-1)\
            # if full_masks else np.empty((0,) + masks.shape[1:3])

        return boxes, class_ids, scores     # , full_masks


    def unmold_detections_new(self, detections, image_shape, window):
        '''
        RUNS DETECTIONS ON FCN_SCORE TENSOR
        
        Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the application.

        detections  : [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_mask  : [N, height, width, num_classes]
        image_shape : [height, width, depth] Original size of the image before resizing
        window      : [y1, x1, y2, x2] Box in the image where the real image is
                       (i.e.,  excluding the padding surrounding the real image)

        Returns:
        boxes       : [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids   : [N] Integer class IDs for each bounding box
        scores      : [N] Float probability scores of the class_id
        masks       : [height, width, num_instances] Instance masks
        '''
        
        # print('>>>  unmold_detections ')
        # print('     detections.shape : ', detections.shape)
        # print('     mrcnn_mask.shape : ', mrcnn_mask.shape)
        # print('     image_shape.shape: ', image_shape)
        # print('     window.shape     : ', window)
        # print(detections)
        
        # How many detections do we have?
        # Detections array is padded with zeros. detections[:,4] identifies the class 
        # Find all rows in detection array with class_id == 0 , and place their row indices
        # into zero_ix. zero_ix[0] will identify the first row with class_id == 0.
        print()
        np.set_printoptions(linewidth=100)  
        p1 = detections 
        p2 = np.reshape(p1, (-1, p1.shape[-1]))
        p2 = p2[p2[:,5].argsort()[::-1]]
        detections = p2
         
        zero_ix = np.where(detections[:, 4] == 0)[0]
    
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # print(' np.where() \n', np.where(detections[:, 4] == 0))
        # print('     zero_ix.shape     : ', zero_ix.shape)
        # print('     N is :', N)
        
        # Extract boxes, class_ids, scores, and class-specific masks
        boxes        = detections[:N, :4]
        class_ids    = detections[:N, 4].astype(np.int32)
        scores       = detections[:N, 5]
        pre_scores   = detections[:N, 8:11]
        fcn_scores   = detections[:N, 13:]
        # masks      = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Compute scale and shift to translate coordinates to image domain.
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale   = min(h_scale, w_scale)
        shift   = window[:2]  # y, x
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

        # Translate bounding boxes to image domain
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

        # Filter out detections with zero area. Often only happens in early
        # stages of training when the network weights are still a bit random.
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
            
        if exclude_ix.shape[0] > 0:
            boxes      = np.delete(boxes, exclude_ix, axis=0)
            class_ids  = np.delete(class_ids, exclude_ix, axis=0)
            scores     = np.delete(scores, exclude_ix, axis=0)
            pre_scores = np.delete(pre_scores, exclude_ix, axis=0)
            fcn_scores = np.delete(fcn_scores, exclude_ix, axis=0)
            # masks     = np.delete(masks, exclude_ix, axis=0)
            N         = class_ids.shape[0]

        return boxes, class_ids, scores, pre_scores, fcn_scores     # , full_masks

    ##------------------------------------------------------------------------------------    
    ## LOAD MODEL
    ##------------------------------------------------------------------------------------        
        
    def load_model_weights(self,init_with = None, exclude = None, new_folder = False):
        '''
        methods to load weights
        1 - load a specific file
        2 - find a last checkpoint in a specific folder 
        3 - use init_with keyword 
        '''    
        # Which weights to start with?
        print('-----------------------------------------------')
        print(' Load FCN model with init parm: [',init_with,']')
        # print(' find last chkpt :', model.find_last())
        if exclude is not None:
            print(' Exclude layers: ')
            for la in exclude: 
                print('    - ',la)
        print('-----------------------------------------------')
       
        ## 1- look for a specific weights file 
        ## Load trained weights (fill in path to trained weights here)
        # model_path  = 'E:\\Models\\mrcnn_logs\\shapes20180428T1819\\mask_rcnn_shapes_5784.h5'
        # print(' model_path : ', model_path )

        # print("Loading weights from ", model_path)
        # model.load_weights(model_path, by_name=True)    
        # print('Load weights complete')

        # ## 2- look for last checkpoint file in a specific folder (not working correctly)
        # model.config.LAST_EPOCH_RAN = 5784
        # model.model_dir = 'E:\\Models\\mrcnn_logs\\shapes20180428T1819'
        # last_model_found = model.find_last()
        # print(' last model in MODEL_DIR: ', last_model_found)
        # # loc= model.load_weights(model.find_last()[1], by_name=True)
        # # print('Load weights complete :', loc)


        ## 3- Use init_with keyword
        ## Which weights to start with?
        # init_with = "last"  # imagenet, coco, or last

        if init_with == "imagenet":
        #     loc=model.load_weights(model.get_imagenet_weights(), by_name=True)
            loc=self.load_weights(RESNET_MODEL_PATH, by_name=True)
        elif init_with == "init":
            print(' ---> init')
            # Load weights trained on MS COCO, but skip layers that 
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            loc=self.load_weights(VGG16_MODEL_PATH, by_name=True, exclude = exclude)
        elif init_with == "coco":
            print(' ---> coco')
            # Load weights trained on MS COCO, but skip layers that 
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            loc=self.load_weights(COCO_MODEL_PATH, by_name=True,
                               exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        elif init_with == "last":
            print(' ---> last')

            # Load the last model you trained and continue training, placing checkpouints in same folder
            last_file = self.find_last()[1]
            print('   Last file is :', last_file)
            loc= self.load_weights(self.find_last()[1], by_name=True)
        else:
            assert init_with != "", "Provide path to trained weights"
            print("   Loading weights from provided file :", init_with)
            loc = self.load_weights(init_with, by_name=True, exclude = exclude, new_folder= new_folder)    
        # print('Load weights complete', loc)        
        return     


        
    def find_last(self):
        '''
        Finds the last checkpoint file of the last trained model in the
        model directory.
        
        Returns:
        --------
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        '''
        
        # Get directory names. Each directory corresponds to a model
        
        print('>>> find_last checkpoint in : ', self.model_dir)
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        print(' Key :',key)
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("fcn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        
        # log("    find_last info:   dir_name: {}".format(dir_name))
        # log("    find_last info: checkpoint: {}".format(checkpoint))

        return dir_name, checkpoint

        
    def load_weights(self, filepath, by_name=False, exclude=None, new_folder = False):
        """
        Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import topology
        log('>>> load_weights() from : {}'.format(filepath))
        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers
            
        # print('\n\n')
        # print('--------------------------------')
        # print(' List of all Layers in Model    ')
        # print('--------------------------------')
        # print('\n\n')
        
        # for idx,layer in enumerate(layers):
            # print('>layer {} : name : {:40s}  type: {}'.format(idx,layer.name,layer))
            
        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)
            
        print('--------------------------------------' )       
        print(' layers to load (not in exclude list) ' )
        print('--------------------------------------' )
        for idx,layer in enumerate(layers):
            print('>layer {} : name : {:40s}  type: {}'.format(idx,layer.name,layer))
        print('\n\n')
            
        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()
        
        log('    load_weights: Log directory set to : {}'.format(filepath))
        # Update the log directory
        self.set_log_dir(filepath, new_folder)
        print('    Load weights complete : ',filepath)        
        return(filepath)

        
    def set_log_dir(self, model_path=None, new_folder = False):
        '''
        Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        '''
        # Set date and epoch counter as if starting a new model
        # print('>>> Set_log_dir() -- model dir is ', self.model_dir)
        # print('    model_path           :   ', model_path)
        # print('    config.LAST_EPOCH_RAN:   ', self.config.LAST_EPOCH_RAN)

        self.tb_dir = os.path.join(self.model_dir,'tensorboard')
        self.epoch  = 0
        regex_match = False
        last_checkpoint_epoch = 0
        now = datetime.datetime.now()
        
        # If we have a model path with date and epochs use them
        
        
        if model_path:
            # Continue from we left off. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            model_path = model_path.replace('\\' , "/")
            # print('    set_log_dir: model_path (input) is : {}  '.format(model_path))        

            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/fcn\w+(\d{4})\.h5"
            regex_match  = re.match(regex, model_path)
            
            if regex_match:             
                now = datetime.datetime(int(regex_match.group(1)), int(regex_match.group(2)), int(regex_match.group(3)),
                                        int(regex_match.group(4)), int(regex_match.group(5)))
                last_checkpoint_epoch = int(regex_match.group(6)) + 1
                # print('    set_log_dir: self.epoch set to {}  (Next epoch to run)'.format(self.epoch))
                # print('    set_log_dir: tensorboard path: {}'.format(self.tb_dir))
                if last_checkpoint_epoch > 0 and  self.config.LAST_EPOCH_RAN > last_checkpoint_epoch: 
                    self.epoch = self.config.LAST_EPOCH_RAN
                else :
                    self.epoch = last_checkpoint_epoch
        
        
        
        # Set directory for training logs
        # if new_folder = True or appropriate checkpoint filename was not found, generate new folder
        if new_folder:
            now = datetime.datetime.now()

        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "fcn_{}_*epoch*.h5".format(self.config.NAME.lower()))
            
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")
        
        log('    set_log_dir(): Checkpoint path/filename set to          : {} '.format(self.checkpoint_path))
        log('    set_log_dir(): Last completed epoch (self.epoch) set to : {} '.format(self.epoch))


        
    def save_model(self, filepath, filename = None, by_name=False, exclude=None):
        """
        Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        print('>>> save_model_architecture()')

        model_json = self.keras_model.to_json()
        full_filepath = os.path.join(filepath, filename)
        log('    save model to  {}'.format(full_filepath))

        with open(full_filepath , 'w') as f:
            # json.dump(model_json, full_filepath)               
            if hasattr(f, 'close'):
                f.close()
                print('file closed')
                
                
        print('    save_weights: save directory is  : {}'.format(filepath))
        print('    save model Load weights complete')        
        return(filepath)

        
    def get_imagenet_weights(self):
        """
        Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/'\
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_path


        
    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

        
    def run_graph(self, images, outputs):
        """Runs a sub-set of the computation graph that computes the given
        outputs.

        outputs: List of tuples (name, tensor) to compute. The tensors are
            symbolic TensorFlow tensors and the names are for easy tracking.

        Returns an ordered dict of results. Keys are the names received in the
        input and values are Numpy arrays.
        """
        model = self.keras_model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(KB.learning_phase(), int):
            inputs += [KB.learning_phase()]
        kf = KB.function(model.inputs, list(outputs.values()))

        # Run inference
        molded_images, image_metas, windows = self.mold_inputs(images)
        # TODO: support training mode?
        # if TEST_MODE == "training":
        #     model_in = [molded_images, image_metas,
        #                 target_rpn_match, target_rpn_bbox,
        #                 input_normalized_gt_boxes, gt_masks]
        #     if not config.USE_RPN_ROIS:
        #         model_in.append(target_rois)
        #     if model.uses_learning_phase and not isinstance(KB.learning_phase(), int):
        #         model_in.append(1.)
        #     outputs_np = kf(model_in)
        # else:

        model_in = [molded_images, image_metas]
        if model.uses_learning_phase and not isinstance(KB.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v)
                                  for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np

      
    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

        
    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
            else:
                # print('   Layer: ', l.name, ' doesn''t have any weights !!!')
                pass
        return layers


    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=0):
        '''
        Sets model layers as trainable if their names match the given
        regular expression.
        '''       
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("\nSelecting layers to train")
            log("-------------------------")
            log("{:5}    {:20}     {}".format( 'Layer', 'Layer Name', 'Layer Type'))

        keras_model = keras_model or self.keras_model
              
      
        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # go through layers one by one, if the layer matches a layer reg_ex, set it to trainable 
        for ind,layer in enumerate(layers):
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                if verbose > 0:
                    print("Entering model layer: ", layer.name, '------------------------------')
                
                self.set_trainable(layer_regex, keras_model=layer, indent=indent + 4)
                indent -= 4
                
                if verbose >  0 :
                    print("Exiting model layer ", layer.name, '--------------------------------')
                continue

            if not layer.weights:
                if verbose > 0:
                    log(" {}{:3}  {:20}   ({:20})   ............................no weights to train ]". \
                    format(" " * indent, ind, layer.name,layer.__class__.__name__))
                continue
                
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))

            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names

            if trainable :
                log(" {}{:3}  {:20}   ({:20})   TRAIN ".\
                    format(" " * indent, ind, layer.name, layer.__class__.__name__))
            else:
                if verbose > 0:
                    log(" {}{:3}  {:20}   ({:20})   ............................not a layer we want to train ]". \
                    format(" " * indent, ind, layer.name, layer.__class__.__name__))                
                pass
        return   
        
           
    def train(self, 
              train_dataset, 
              val_dataset, 
              learning_rate, 
              layers            = None,
              losses            = None,
              epochs            = 0,
              epochs_to_run     = 0,
              batch_size        = 0, 
              steps_per_epoch   = 0,
              min_lr            = 0):
        '''
        Train the model.
        train_dataset, 
        val_dataset:    Training and validation Dataset objects.
        
        learning_rate:  The learning rate to train with
        
        layers:         Allows selecting wich layers to train. It can be:
                        - A regular expression to match layer names to train
                        - One of these predefined values:
                        heads: The RPN, classifier and mask heads of the network
                        all: All the layers
                        3+: Train Resnet stage 3 and up
                        4+: Train Resnet stage 4 and up
                        5+: Train Resnet stage 5 and up
        
        losses:         List of losses to monitor.
        
        epochs:         Number of training epochs. Note that previous training epochs
                        are considered to be done already, so this actually determines
                        the epochs to train in total rather than in this particaular
                        call.
        
        epochs_to_run:  Number of epochs to run, will update the 'epochs parm.                        
                        
        '''
        assert self.mode == "training", "Create model in training mode."
        
        if batch_size == 0 :
            batch_size = self.config.BATCH_SIZE
        if epochs_to_run > 0 :
            epochs = self.epoch + epochs_to_run
        if steps_per_epoch == 0:
            steps_per_epoch = self.config.STEPS_PER_EPOCH
        if min_lr == 0:
            min_lr = self.config.MIN_LR
            
            
        # use Pre-defined layer regular expressions
        # if layers in self.layer_regex.keys():
            # layers = self.layer_regex[layers]
        print(layers)
        # train_regex_list = []
        # for x in layers:
            # print( ' layers ias : ',x)
            # train_regex_list.append(x)
        train_regex_list = [self.layer_regex[x] for x in layers]
        print(train_regex_list)
        layers = '|'.join(train_regex_list)        
        print('layers regex :', layers)
        

            
        
        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         batch_size=batch_size)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                        batch_size=batch_size,
                                        augment=False)

        # my_callback = MyCallback()

        # Callbacks
        ## call back for model checkpoint was originally (?) loss. chanegd to val_loss (which is default) 2-5-18
        callbacks = [
              keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                          histogram_freq=0,
                                          batch_size=self.config.BATCH_SIZE,
                                          write_graph=True,
                                          write_grads=False,
                                          write_images=True,
                                          embeddings_freq=0,
                                          embeddings_layer_names=None,
                                          embeddings_metadata=None)

            , keras.callbacks.ModelCheckpoint(self.checkpoint_path, 
                                              mode = 'auto', 
                                              period = 1, 
                                              monitor='val_loss', 
                                              verbose=1, 
                                              save_best_only = True, 
                                              save_weights_only=True)
                                            
            , keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                mode     = 'auto', 
                                                factor   = self.config.REDUCE_LR_FACTOR,   
                                                cooldown = self.config.REDUCE_LR_COOLDOWN,
                                                patience = self.config.REDUCE_LR_PATIENCE,
                                                min_lr   = self.config.MIN_LR, 
                                                verbose  = 1)                                            
                                                
            , keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                mode      = 'auto', 
                                                min_delta = 0.00001, 
                                                patience  = self.config.EARLY_STOP_PATIENCE, 
                                                verbose   = 1)                                            
            # , my_callback
        ]

        # Train

        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM, losses)
        
        log("Starting at epoch   {} of {} epochs. LR={}\n".format(self.epoch, epochs, learning_rate))
        log("Steps per epochs    {} ".format(steps_per_epoch))
        log("Batch size          {} ".format(batch_size))
        log("Checkpoint Path:    {} ".format(self.checkpoint_path))
        log("Weight Decay:       {} ".format(self.config.WEIGHT_DECAY       ))
        log("VALIDATION_STEPS    {} ".format(self.config.VALIDATION_STEPS   ))
        log("REDUCE_LR_FACTOR    {} ".format(self.config.REDUCE_LR_FACTOR   ))
        log("REDUCE_LR_COOLDOWN  {} ".format(self.config.REDUCE_LR_COOLDOWN ))
        log("REDUCE_LR_PATIENCE  {} ".format(self.config.REDUCE_LR_PATIENCE ))
        log("MIN_LR              {} ".format(self.config.MIN_LR             ))
        log("EARLY_STOP_PATIENCE {} ".format(self.config.EARLY_STOP_PATIENCE))        
        
        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            validation_data=next(val_generator),
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=1,                                  # max(self.config.BATCH_SIZE // 2, 2),
            use_multiprocessing=False
        )
        self.epoch = max(self.epoch, epochs)

        print('Final : self.epoch {}   epochs {}'.format(self.epoch, epochs))

        
    def train_in_batches(self,
              mrcnn_model,
              train_dataset, 
              val_dataset,  
              layers            = None,
              losses            = None,
              learning_rate     = 0,              
              epochs            = 0,
              epochs_to_run     = 0, 
              batch_size        = 0, 
              steps_per_epoch   = 0,
              min_LR            = 0):
              
        '''
        Train the model.
        train_dataset, 
        val_dataset:    Training and validation Dataset objects.
        
        learning_rate:  The learning rate to train with
        
        epochs:         Number of training epochs. Note that previous training epochs
                        are considered to be done already, so this actually determines
                        the epochs to train in total rather than in this particaular
                        call.
                        
        layers:         Allows selecting wich layers to train. It can be:
                        - A regular expression to match layer names to train
                        - One of these predefined values:
                        heads: The RPN, classifier and mask heads of the network
                        all: All the layers
                        3+: Train Resnet stage 3 and up
                        4+: Train Resnet stage 4 and up
                        5+: Train Resnet stage 5 and up
        '''
        assert self.mode == "training", "Create model in training mode."
        
        if batch_size == 0 :
            batch_size = self.config.BATCH_SIZE
        
        if epochs_to_run ==  0 :
            epochs_to_run = self.config.EPOCHS_TO_RUN
        
        if steps_per_epoch == 0:
            steps_per_epoch = self.config.STEPS_PER_EPOCH

        if min_LR == 0 :
            min_LR = self.config.MIN_LR
        
        if learning_rate == 0:
            learning_rate = self.config.LEARNING_RATE
            
        epochs = self.epoch + epochs_to_run
            
        # use Pre-defined layer regular expressions
        # if layers in self.layer_regex.keys():
            # layers = self.layer_regex[layers]
        print(layers)
        # train_regex_list = []
        # for x in layers:
            # print( ' layers ias : ',x)
            # train_regex_list.append(x)
        train_regex_list = [self.layer_regex[x] for x in layers]
        print(train_regex_list)
        layers = '|'.join(train_regex_list)        
        print('layers regex :', layers)
        
        
        ##--------------------------------------------------------------------------------
        ## Data generators
        ##--------------------------------------------------------------------------------
        train_generator = data_generator(train_dataset, mrcnn_model.config, shuffle=True,
                                         batch_size=batch_size)
        val_generator   = data_generator(val_dataset, mrcnn_model.config, shuffle=True,
                                         batch_size=batch_size,
                                         augment=False)

        ##--------------------------------------------------------------------------------
        ## Set trainable layers and compile
        ##--------------------------------------------------------------------------------
        self.set_trainable(layers)            
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM, losses)        


        ##--------------------------------------------------------------------------------
        ## Create checkpoint folder if it doesn't exists
        ##--------------------------------------------------------------------------------
        from tensorflow.python.platform import gfile
        if not gfile.IsDirectory(self.log_dir):
            log('Creating checkpoint folder : {}'.format(self.log_dir))
            gfile.MakeDirs(self.log_dir)
        else:
            log('Checkpoint folder already exists: {}'.format(self.log_dir))
        # my_callback = MyCallback()

        ##--------------------------------------------------------------------------------
        ## Callbacks
        ##--------------------------------------------------------------------------------
        # call back for model checkpoint was originally (?) loss. chanegd to val_loss (which is default) 2-5-18

        # copied from \keras\engine\training.py
        # def _get_deduped_metrics_names(self):
        ## get metrics from keras_model.metrics_names
        out_labels = self.get_deduped_metrics_names()
        print()
        print(' out_labels from get_deduped_metrics_names() : ')
        print(' --------------------------------------------- ')
        print(out_labels)

        ## setup Progress Bar callback
        callback_metrics = out_labels + ['val_' + n for n in out_labels]
        print()
        print(' Callback metrics monitored by progbar :')
        print(' ---------------------------------------')
        pp.pprint(callback_metrics)
        
        # progbar = keras.callbacks.ProgbarLogger(count_mode='steps')
        # progbar.set_model(self.keras_model)
        # progbar.set_params({
            # 'epochs': epochs,
            # 'steps': steps_per_epoch,
            # 'verbose': 1,
            # 'do_validation': False,
            # 'metrics': callback_metrics,
        # })
        # progbar.set_model(self.keras_model) 

        # setup Checkpoint callback
        # chkpoint = keras.callbacks.ModelCheckpoint(self.checkpoint_path, 
                                                   # monitor='val_loss', 
                                                   # verbose=1, 
                                                   # save_best_only = True, 
                                                   # save_weights_only=True)
        # chkpoint.set_model(self.keras_model)
        # progbar.on_train_begin()



        callbacks_list = [
            keras.callbacks.ProgbarLogger(count_mode='steps'),
             
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                          histogram_freq=0,
                                          batch_size=self.config.BATCH_SIZE,
                                          write_graph=True,
                                          write_grads=True,
                                          write_images=True,
                                          embeddings_freq=0,
                                          embeddings_layer_names=None,
                                          embeddings_metadata=None)

            , keras.callbacks.ModelCheckpoint(self.checkpoint_path, 
                                              mode = 'auto', 
                                              period = 1, 
                                              monitor='val_loss', 
                                              verbose=1, 
                                              save_best_only = True, 
                                              save_weights_only=True)
                                            
            , keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                mode     = 'auto', 
                                                factor   = self.config.REDUCE_LR_FACTOR,   
                                                cooldown = self.config.REDUCE_LR_COOLDOWN,
                                                patience = self.config.REDUCE_LR_PATIENCE,
                                                min_lr   = self.config.MIN_LR, 
                                                verbose  = 1)                                            
                                                
            , keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                mode      = 'auto', 
                                                min_delta = 0.00001, 
                                                patience  = self.config.EARLY_STOP_PATIENCE, 
                                                verbose   = 1)                                            
            # , my_callback
        ]

        
        callbacks =  keras.callbacks.CallbackList(callbacks = callbacks_list)
        callbacks.set_model(self.keras_model)
        callbacks.set_params({
            'epochs': epochs,
            'steps': steps_per_epoch,
            'verbose': 1 ,
            'do_validation': True,
            'metrics': callback_metrics
        })
        
        

        log("Starting at epoch {} of {} epochs. LR={}\n".format(self.epoch, epochs, learning_rate))
        log("Steps per epochs {} ".format(steps_per_epoch))
        log("    Last epoch completed : {} ".format(self.epoch))
        log("    Starting from epoch  : {} for {} epochs".format(self.epoch, epochs_to_run))
        log("    Learning Rate        : {} ".format(learning_rate))
        log("    Steps per epoch      : {} ".format(steps_per_epoch))
        log("    Batch Size           : {} ".format(batch_size))
        log("    Checkpoint Folder    : {} ".format(self.checkpoint_path))

        

        ##--------------------------------------------------------------------------------
        ## Start main training loop
        ##--------------------------------------------------------------------------------
        epoch_idx = self.epoch
        # progbar.on_train_begin()
        callbacks.on_train_begin()

        if epoch_idx >= epochs:
            print('Final epoch {} has already completed - Training will not proceed'.format(epochs))
        else:
        
            while epoch_idx < epochs :
                # progbar.on_epoch_begin(epoch_idx)
                callbacks.on_epoch_begin(epoch_idx)
                epoch_logs = {}
                
                for steps_index in range(steps_per_epoch):
                    
                    # print(' self.epoch {}   epochs {}  step {} '.format(self.epoch, epochs, steps_index))
                    batch_logs = {}
                    batch_logs['batch'] = steps_index
                    batch_logs['size']  = batch_size    
                    # progbar.on_batch_begin(steps_index, batch_logs)

                    callbacks.on_batch_begin(steps_index, batch_logs)
                    train_batch_x, train_batch_y = next(train_generator)

                    # print(len(model_output))
                    # print(type(output_rois))
                    # for i in model_output:
                        # print( i.shape)                    
                    
                    results = mrcnn_model.keras_model.predict(train_batch_x)
                    pr_hm_norm, gt_hm_norm, pr_hm_scores, gt_hm_scores = results[11:]                 

                    # print('pr_hm_norm shape   :', pr_hm_norm.shape)
                    # print('pr_hm_scores shape :', pr_hm_scores.shape)
                    # print('gt_hm_norm shape   :', gt_hm_norm.shape)
                    # print('gt_hm_scores shape :', gt_hm_scores.shape)
                    
                    outs = self.keras_model.train_on_batch([pr_hm_norm,  pr_hm_scores,gt_hm_norm, gt_hm_scores], train_batch_y)
                
                    if not isinstance(outs, list):
                        outs = [outs]

                    for l, o in zip(out_labels, outs):
                        batch_logs[l] = o
    
                    callbacks.on_batch_end(steps_index, batch_logs)

                ##-------------------------------
                ## end of epoch operations     
                ##-------------------------------
                val_steps_done  = 0
                val_all_outs    = []
                val_batch_sizes = []
                while val_steps_done < self.config.VALIDATION_STEPS:

                    val_batch_x, val_batch_y = next(val_generator)
                    val_results = mrcnn_model.keras_model.predict(val_batch_x)
                    pr_hm_norm, gt_hm_norm, pr_hm_scores, gt_hm_scores = val_results[11:]                 
                    
                    # print('pr_hm_norm shape   :', pr_hm_norm.shape)
                    # print('pr_hm_scores shape :', pr_hm_scores.shape)
                    # print('gt_hm_norm shape   :', gt_hm_norm.shape)
                    # print('gt_hm_scores shape :', gt_hm_scores.shape)
 
                    x = [pr_hm_norm, pr_hm_scores, gt_hm_norm, gt_hm_scores]
                    # print('type ',type(x), len(x), x[0].shape[0])
                    outs2 = self.keras_model.test_on_batch( x , val_batch_y)
                    
                    
                    if isinstance(x, list):
                        batch_size = x[0].shape[0]
                    elif isinstance(x, dict):
                        batch_size = list(x.values())[0].shape[0]
                    else:
                        batch_size = x.shape[0]
                        
                    if batch_size == 0:
                        raise ValueError('Received an empty batch. '
                                         'Batches should at least contain one item.')
                    # else:
                        # print('batch size:', batch_size)
                        
                    val_all_outs.append(outs2)
                    val_steps_done += 1
                    val_batch_sizes.append(batch_size)

                ## calculate val_outs after all validations steps complete
                # print(len(val_batch_sizes), len(val_all_outs))
                # print(val_batch_sizes)
                print('\n val_all_outs:', np.asarray(val_all_outs).shape)
                pp.pprint(val_all_outs)

                if not isinstance(outs2, list):
                    val_outs =  np.average(np.asarray(val_all_outs), weights=val_batch_sizes)
                else:
                    averages = []
                    for i in range(len(outs2)):
                        averages.append(np.average([out[i] for out in val_all_outs], axis = 0, weights=val_batch_sizes))
                    val_outs = averages
                print('val_outs is ', val_outs)
                    
                    
                # write_log(callback, val_names, logs, batch_no//10)
                print('\n  validation logs output: ', val_outs)
                if not isinstance(val_outs, list):
                    val_outs = [val_outs]
                
                # Same labels assumed.
                for l, o in zip(out_labels, val_outs):
                    epoch_logs['val_' + l] = o

                callbacks.on_epoch_end(epoch_idx, epoch_logs)
                epoch_idx += 1

                
            ##-------------------------------
            ## end of training operations
            ##--------------------------------
            # if epoch_idx != self.epoch:
            # chkpoint.on_epoch_end(epoch_idx -1, batch_logs)
            callbacks.on_train_end()
            self.epoch = max(epoch_idx - 1, epochs)
            print('Final : self.epoch {}   epochs {}'.format(self.epoch, epochs))
            
        ##--------------------------------------------------------------------------------
        ## End main training loop
        ##--------------------------------------------------------------------------------
        
    ## copied from github but not used
    # def write_log(callback, names, logs, batch_no):
        # for name, value in zip(names, logs):
            # summary = tf.Summary()
            # summary_value = summary.value.add()
            # summary_value.simple_value = value
            # summary_value.tag = name
            # callback.writer.add_summary(summary, batch_no)
            # callback.writer.flush()        


            
    def compile(self, learning_rate, momentum, losses):
        '''
        Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        '''
        assert isinstance(losses, list) , "A loss function must be defined as the objective"
        # Optimizer object
        print('\n')
        print(' Compile Model :')
        print('----------------')
        print('    losses        : ', losses)
        print('    learning rate : ', learning_rate)
        print('    momentum      : ', momentum )
        print()
        #optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, clipnorm=5.0)
        # optimizer= tf.train.GradientDescentOptimizer(learning_rate, momentum)
        # optimizer = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
        optimizer = keras.optimizers.Adagrad(lr=learning_rate, epsilon=None, decay=0.01)
        # optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        # optimizer = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

        ##------------------------------------------------------------------------
        ## Add Losses
        ## These are the losses aimed for minimization
        ## Normally Keras expects the same number of losses as outputs. Since we 
        ## are returning more outputs , we go deep and set the losses in the base
        ## layers () 
        ##------------------------------------------------------------------------    
        print()
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}

        # loss_names = [  "rpn_class_loss", "rpn_bbox_loss", "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss" ]
        # loss_names = [ "fcn_loss", "fcn_norm_loss" ]

        print('Initial self.keras_model.losses :')
        print('---------------------------------')
        print('    losses: ', losses)
        print('    keras_model.losses :') 
        pp.pprint(self.keras_model.losses)

        print('Added losses: ',losses)
        print('------------- ')
        loss_names = losses              
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            print('    Loss: {}  Related Layer is : {}'.format(name, layer.name))
            if layer.output in self.keras_model.losses:
                print('      ',layer.output,' is in self.keras_model.losses, and wont be added to list')
                continue
            print('      >> Add add loss for ', layer.output, ' to list of losses...')
            self.keras_model.add_loss(tf.reduce_mean(layer.output, keepdims=True))
            
        print()    
        print(' self.keras_model.losses after adding losses passed to compile() : ') 
        print('------------------------------------------------------------------ ') 
        pp.pprint(self.keras_model.losses)

        print()    
        print('Keras_model._losses:' ) 
        print('---------------------' ) 
        pp.pprint(self.keras_model._losses)

        print()    
        print('Keras_model._per_input_losses:')
        print('------------------------------')
        pp.pprint(self.keras_model._per_input_losses)
            
            
        ##------------------------------------------------------------------------    
        ## Add L2 Regularization as loss to list of losses
        ##------------------------------------------------------------------------    
        # Skip gamma and beta weights of batch normalization layers.
        # reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                      # for w in self.keras_model.trainable_weights
                      # if 'gamma' not in w.name and 'beta' not in w.name]
        # self.keras_model.add_loss(tf.add_n(reg_losses))

        print()    
        print('Final list of keras_model.losses, after adding L2 regularization as loss to list : ') 
        print('---------------------------------------------------------------------------------- ') 
        pp.pprint(self.keras_model.losses)

        ##------------------------------------------------------------------------    
        ## Compile
        ##------------------------------------------------------------------------   
        print()
        print('----- Compile -----------------------------------------------------------')
        print (' Length of Keras_Model.outputs:', len(self.keras_model.outputs))
        self.keras_model.compile(optimizer=optimizer, 
                                 loss=[None] * len(self.keras_model.outputs))
        print('-------------------------------------------------------------------------')

        ##------------------------------------------------------------------------    
        ## Add metrics for losses
        ##------------------------------------------------------------------------    
        print()
        print(' Add Metrics :')
        print('--------------')                                 
        print (' Initial Keras metric_names:', self.keras_model.metrics_names)                                 

        for name in loss_names:
            if name in self.keras_model.metrics_names:
                print('      ' , name , 'is already in in self.keras_model.metrics_names')
                continue
            layer = self.keras_model.get_layer(name)
            print('    Loss name : {}  Related Layer is : {}'.format(name, layer.name))
            self.keras_model.metrics_names.append(name)
            self.keras_model.metrics_tensors.append(tf.reduce_mean(layer.output, keepdims=True))
            print('      >> Add metric ', name, ' with metric tensor: ', layer.output.name, ' to list of metrics ...')
        
        print()
        print ('Final Keras metric_names :') 
        print ('--------------------------') 
        pp.pprint(self.keras_model.metrics_names)                                 
        print()
        print(' self.keras_model.losses after adding losses passed to compile() : ') 
        print('------------------------------------------------------------------ ') 
        pp.pprint(self.keras_model.losses)

        print()    
        print('Keras_model._losses:' ) 
        print('---------------------' ) 
        pp.pprint(self.keras_model._losses)

        print()    
        print('Keras_model._per_input_losses:')
        print('------------------------------')
        pp.pprint(self.keras_model._per_input_losses)        
        
        return


    def compile_only(self, learning_rate, layers):
        '''
        Compile the model without adding loss info
        learning_rate:  The learning rate to train with
        
        layers:         Allows selecting wich layers to train. It can be:
                        - A regular expression to match layer names to train
                        - One of these predefined values:
                        heads: The RPN, classifier and mask heads of the network
                        all: All the layers
                        3+: Train Resnet stage 3 and up
                        4+: Train Resnet stage 4 and up
                        5+: Train Resnet stage 5 and up
        '''
        # Use Pre-defined layer regular expressions
        if layers in self.layer_regex.keys():
            layers = self.layer_regex[layers]
            
        # Train
        log("Compile with learing rate; {} Learning Moementum: {} ".format(learning_rate,self.config.LEARNING_MOMENTUM))
        log("Checkpoint Folder:  {} ".format(self.checkpoint_path))
        
        self.set_trainable(layers)            
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)        

        out_labels = self.get_deduped_metrics_names()
        callback_metrics = out_labels + ['val_' + n for n in out_labels]
        print('Callback_metrics are:  ( val + _get_deduped_metrics_names() )\n')
        pp.pprint(callback_metrics)
        return
        
        
    def get_deduped_metrics_names(self):
        out_labels = self.keras_model.metrics_names

        # Rename duplicated metrics name
        # (can happen with an output layer shared among multiple dataflows).
        deduped_out_labels = []
        for i, label in enumerate(out_labels):
            new_label = label
            if out_labels.count(label) > 1:
                dup_idx = out_labels[:i].count(label)
                new_label += '_' + str(dup_idx + 1)
            deduped_out_labels.append(new_label)
        return deduped_out_labels        

    def layer_info(self):
        print('\n')
        print(' Inputs:') 
        print(' -------')
    
        for i,x in enumerate(self.keras_model.inputs):
            print(' index: {:2d}    input name : {:40s}   Type: {:15s}   Shape: {}'.format( i, x.name, x.dtype.name, x.shape) )

        print('\n')
        print(' Outputs:') 
        print(' --------')

        for i,x in enumerate(self.keras_model.outputs):
            print(' layer: {:2d}    output name: {:40s}   Type: {:15s}   Shape: {}'.format( i, x.name, x.dtype.name, x.shape) )
        
        return