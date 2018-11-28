"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os, sys, glob, random, math, datetime, itertools, json, re, logging, pprint
from   collections import OrderedDict
import numpy as np
# import scipy.misc
import tensorflow         as tf
import keras
import keras.backend      as KB
import keras.layers       as KL
import keras.initializers as KI
import keras.engine       as KE
import keras.models       as KM
#sys.path.append('..')

import mrcnn.utils                as utils
import mrcnn.loss                 as loss
from   mrcnn.datagen              import data_generator
from   mrcnn.utils                import log, parse_image_meta_graph, parse_image_meta, write_stdout
from   mrcnn.model_base           import ModelBase
from   mrcnn.RPN_model            import build_rpn_model
from   mrcnn.resnet_model         import resnet_graph
from   mrcnn.chm_layer            import CHMLayer
from   mrcnn.chm_layer_tgt        import CHMLayerTarget
from   mrcnn.chm_layer_inf        import CHMLayerInference
from   mrcnn.proposal_layer       import ProposalLayer
from   mrcnn.detect_layer         import DetectionLayer  
from   mrcnn.detect_tgt_layer_mod import DetectionTargetLayer_mod
from   mrcnn.fpn_layers           import fpn_graph, fpn_classifier_graph, fpn_mask_graph
# from   mrcnn.callbacks            import MyCallback
# from   mrcnn.batchnorm_layer      import BatchNorm

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

############################################################
##  MaskRCNN Class
############################################################
class MaskRCNN(ModelBase):
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'trainfcn', 'inference']
        super().__init__(mode, config)
        
        print('>>> ---Initialize MRCNN model, mode: ',mode)
        if mode in ['training']:
            self.set_log_dir()

        # self.model_dir = model_dir
        # not needed as we do this later when we load the model weights
        # self.set_log_dir()
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

        self.keras_model = self.build(mode=mode, config=config)

        print('>>> MaskRCNN initialiation complete. Mode: ',mode)

    
    def build(self, mode, config, FCN_layers = False):
        '''
        Build MODIFIED Mask R-CNN architecture (NO MASK PROCESSING)
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        '''
        assert mode in ['training', 'trainfcn', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        ##------------------------------------------------------------------                            
        ##  Input Layer for all modes
        ##------------------------------------------------------------------
        input_image      = KL.Input(shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        input_image_meta = KL.Input(shape=[None], name="input_image_meta")

        ##------------------------------------------------------------------                            
        ##  Input Layer for training and trainfcn modes
        ##------------------------------------------------------------------
        
        if mode in ["training", "trainfcn"]:
            # RPN GT
            input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox  = KL.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            
            # Normalize coordinates
            h, w = KB.shape(input_image)[1], KB.shape(input_image)[2]
            image_scale = KB.cast(KB.stack([h, w, h, w], axis=0), tf.float32)
            input_normlzd_gt_boxes = KL.Lambda(lambda x: x / image_scale, name = 'input_normalized_gt_boxes')(input_gt_boxes)
            
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            # If using USE_MINI_MASK the mask is 56 x 56 x None 
            #    else:    image h x w x None
        #-------			
            # if config.USE_MINI_MASK:
                # input_gt_masks = KL.Input(
                    # shape=[config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None],
                    # name="input_gt_masks", dtype=bool)
            # else:
                # input_gt_masks = KL.Input(
                    # shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    # name="input_gt_masks", dtype=bool)
        #-------
        # End if mode in ['training', "trainfcn"]
        #-----------------------------------------------------------------------------

        ##----------------------------------------------------------------------------
        ##
        ##  COMMON LAYERS FOR ALL MODES:  training, trainfcn, AND inference  modes
        ##
        ##----------------------------------------------------------------------------


        
        ##----------------------------------------------------------------------------
        ##  Resnet Backbone
        ##----------------------------------------------------------------------------
        # Build the Resnet shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the head (stage 5), so we pick the 4th item in the list.
        #----------------------------------------------------------------------------
        Resnet_Layers      = resnet_graph(input_image, "resnet101", stage5=True)
        
        ##----------------------------------------------------------------------------
        ##  FPN network - Build the Feature Pyramid Network (FPN) layers.
        ##----------------------------------------------------------------------------
        P2, P3, P4, P5, P6 = fpn_graph(Resnet_Layers)
        
        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps   = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        ##----------------------------------------------------------------------------        
        ##  Generate Anchors Box Coordinates
        ##  shape.anchors will contain an array of anchor box coordinates (y1,x1,y2,x2)
        ##----------------------------------------------------------------------------        
        self.anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                      config.RPN_ANCHOR_RATIOS,
                                                      config.BACKBONE_SHAPES,
                                                      config.BACKBONE_STRIDES,
                                                      config.RPN_ANCHOR_STRIDE)


        ##----------------------------------------------------------------------------        
        ##  RPN Model - 
        ##  model which is applied on the feature maps produced by the resnet backbone
        ##----------------------------------------------------------------------------                
        RPN_model = build_rpn_model(config.RPN_ANCHOR_STRIDE, len(config.RPN_ANCHOR_RATIOS), 256)

        
        #----------------------------------------------------------------------------         
        # Loop through pyramid layers (P2 ~ P6) and pass each layer to the RPN network
        # for each layer rpn network returns [rpn_class_logits, rpn_probs, rpn_bbox]
        #      
        #  rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
        #----------------------------------------------------------------------------
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            print('     append {} to layer_outputs '.format(p))
            layer_outputs.append(RPN_model([p]))

            
        #----------------------------------------------------------------------------                    
        # Concatenate  layer outputs
        #----------------------------------------------------------------------------                        
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        # 
        # the final output is a list consisting of three tensors:
        #
        # a1,..., a5: rpn_class_logits : Tensor("rpn_class_logits, shape=(?, ?, 2), dtype=float32)
        # b1,..., b5: rpn_probs        : Tensor("rpn_class       , shape=(?, ?, 2), dtype=float32)
        # c1,..., c5: rpn_bbox         : Tensor("rpn_bbox_11     , shape=(?, ?, 4), dtype=float32)
        #----------------------------------------------------------------------------                
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]     
        outputs = list(zip(*layer_outputs))
        # concatinate the list of tensors in each group (logits, probs, bboxes)
                
        # outputs = [KL.Concatenate(axis=1, name=n)(list(o))  for o, n in zip(outputs, output_names)]
        # print('\n>>> RPN Outputs ',  type(outputs))
        # for i in outputs:
            # print('     ', i.name)

        # outputs =  [KL.Lambda(lambda  o: KB.identity(o , name = n), name = n) (o) for o, n in zip(outputs, output_names)]
        outputs =  [KL.Lambda(lambda  o: tf.concat(o ,axis=1, name=n) ,name = n) (list(o)) for o, n in zip(outputs, output_names)]
        print('\n>>> RPN Outputs ',  type(outputs))
        for i in outputs:
            print('     ', i.name)

            
            
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        ##----------------------------------------------------------------------------                
        ##  RPN Proposal Layer
        ##
        ##  Generate ROI proposals from bboxes and classes generated by the RPN model
        ##  ROI Proposals are [batch, proposal_count, 4 (y1, x1, y2, x2)] in NORMALIZED coordinates
        ##  and zero padded.
        ##
        ##  proposal_count : number of proposal regions to generate:
        ##    Training  mode :        2000 proposals 
        ##    Inference mode :        1000 proposals
        ##----------------------------------------------------------------------------                
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode in [ "training", "trainfcn"] \
                    else config.POST_NMS_ROIS_INFERENCE

        rpn_roi_proposals = ProposalLayer(proposal_count=proposal_count,            # num proposals to generate
                                 nms_threshold=config.RPN_NMS_THRESHOLD,
                                 name="ROI",
                                 anchors=self.anchors,
                                 config=config)([rpn_class, rpn_bbox])
                                 
        ##----------------------------------------------------------------------------                
        ## Training/ trainfcn Mode Layers
        ##----------------------------------------------------------------------------                
        if mode in [ "training" , "trainfcn"]:
            # Class ID mask to mark class IDs supported by the dataset the image came from. 
            _, _, _, active_class_ids = KL.Lambda(lambda x:  parse_image_meta_graph(x), mask=[None, None, None, None])(input_image_meta)
            
            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs - Normalize and use ROIs provided as an input to the model.
                input_rois  = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4], name="input_roi", dtype=np.int32)
                rpn_roi_proposals = KL.Lambda(lambda x: KB.cast(x, tf.float32) / image_scale[:4], name='rpn_roi_proposals')(input_rois)
            else:
                pass
                # target_rois = rpn_roi_proposals
            
            ##--------------------------------------------------------------------------------------
            ##  Detection_Target_Layer
            ##
            #  Generate detection targets
            #    generated RPNs ----> Target ROIs
            #
            #    target_* returned from this layer are the 'processed' versions of gt_*  
            # 
            #    Subsamples proposals and generates target outputs for training
            #    Note that proposal class IDs, input_normalized_gt_boxes, and gt_masks are zero padded. 
            #    Equally, returned rois and targets are zero padded.
            # 
            #   Note : roi (first output of DetectionTargetLayer) was testing and verified to b
            #          be equal to output_rois. Therefore, the output_rois layer was removed, 
            #          and the first output below was renamed rois --> output_rois
            #   
            #    output_rois :       (?, TRAIN_ROIS_PER_IMAGE, 4),    # output bounindg boxes            
            #    target_class_ids :  (?, 1),                          # gt class_ids            
            #    target_bbox_deltas: (?, TRAIN_ROIS_PER_IMAGE, 4),    # gt bounding box deltas            
            #    roi_gt_bboxes:      (?, TRAIN_ROIS_PER_IMAGE, 4)     # gt bboxes            
            #
            ##--------------------------------------------------------------------------------------
            # remove target_mask for build_new   05-11-2018
            
            output_rois, target_class_ids, target_bbox_deltas,  roi_gt_boxes = \
                DetectionTargetLayer_mod(config, name="proposal_targets") \
                                    ([rpn_roi_proposals, input_gt_class_ids, input_normlzd_gt_boxes])

            #--------------------------------------------------------------------------------------
            # TODO: clean up (use tf.identify if necessary)
            # replace with KB.identity -- 03-05-2018
            # renamed output from DetectionTargetLayer abouve from roi to output_roi and 
            # following lines were removed. 
            # 
            # output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)
            # output_rois = KL.Lambda(lambda x: KB.identity(x), name= "output_rois")(rois)
            #------------------------------------------------------------------------------------

            ##------------------------------------------------------------------------------------
            ##  MRCNN Network Classification Head
            ##  TODO: verify that this handles zero padded ROIs
            ##------------------------------------------------------------------------------------
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
                fpn_classifier_graph(output_rois, mrcnn_feature_maps, config.IMAGE_SHAPE, config.POOL_SIZE, config.NUM_CLASSES)

            ##----------------------------------------------------------------------------
            ##  Contextual Layer(CHM) to generate contextual feature maps using outputs from MRCNN 
            ##----------------------------------------------------------------------------         
            # pr_hm, pr_hm_scores , gt_hm, gt_hm_scores, pr_tensor,  gt_tensor \
                # =  CHMLayer(config, name = 'cntxt_layer' ) \
                    # ([mrcnn_class, mrcnn_bbox, output_rois, target_class_ids, roi_gt_boxes])
            pr_hm, pr_hm_scores \
                =  CHMLayer(config, name = 'cntxt_layer' ) ([mrcnn_class, mrcnn_bbox, output_rois])
                
            gt_hm, gt_hm_scores \
                =  CHMLayerTarget(config, name = 'cntxt_layer_gt' ) ([target_class_ids, roi_gt_boxes])
                
            print('<<<  shape of pred_heatmap   : ', pr_hm.shape, ' Keras tensor ', KB.is_keras_tensor(pr_hm) )                         
            print('<<<  shape of gt_heatmap     : ', gt_hm.shape, ' Keras tensor ', KB.is_keras_tensor(gt_hm) )
            

            ##------------------------------------------------------------------------
            ##  training mode Loss layer definitions
            ##------------------------------------------------------------------------
            if mode == "training":                      
                print('\n')
                print('---------------------------------------------------')
                print('    building Loss Functions ')
                print('---------------------------------------------------')

                rpn_class_loss   = KL.Lambda(lambda x: loss.rpn_class_loss_graph(*x),   name="rpn_class_loss")\
                                     ([input_rpn_match   , rpn_class_logits])
                
                rpn_bbox_loss    = KL.Lambda(lambda x: loss.rpn_bbox_loss_graph(config, *x),  name="rpn_bbox_loss")\
                                     ([input_rpn_bbox    , input_rpn_match   , rpn_bbox])

                mrcnn_class_loss = KL.Lambda(lambda x: loss.mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")\
                                     ([target_class_ids  , mrcnn_class_logits, active_class_ids])
                
                mrcnn_bbox_loss  = KL.Lambda(lambda x: loss.mrcnn_bbox_loss_graph(*x),  name="mrcnn_bbox_loss") \
                                     ([target_bbox_deltas, target_class_ids  , mrcnn_bbox])


            ##------------------------------------------------------------------------
            ##  training and trainfcn  mode input / output definitions 
            ##    add losses to output when in mrcnn training mode
            ##------------------------------------------------------------------------
            # Model Inputs 
            inputs = [  input_image              #  
                      , input_image_meta         #   
                      , input_rpn_match          # [batch_sz, N, 1:<pos,neg,nutral>)                  [ 1,4092, 1]
                      , input_rpn_bbox           # [batch_sz, RPN_TRAIN_ANCHORS_PER_IMAGE, 4]         [ 1, 256, 4]
                      , input_gt_class_ids       # [batch_sz, MAX_GT_INSTANCES] Integer class IDs         [1, 100]
                      , input_gt_boxes           # [batch_sz, MAX_GT_INSTANCES, 4]                     [1, 100, 4]
                     ]
                        
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)

            # outputs =  # , rpn_class_logits  , rpn_class         , rpn_bbox            , rpn_roi_proposals     
                         # , output_rois       , target_class_ids  , target_bbox_deltas  , roi_gt_boxes    
                         # , mrcnn_class_logits, mrcnn_class       , mrcnn_bbox                            
                        
            outputs = [    pr_hm                                                                       # 15
                         , pr_hm_scores                                                                     # 17
                         , gt_hm                                                                       # 16
                         , gt_hm_scores                                                                     # 18    
                         , mrcnn_class, mrcnn_bbox, output_rois
                         , target_class_ids, roi_gt_boxes  
                         , mrcnn_class_logits, active_class_ids 
                       ]
            
            if mode == 'training':
                outputs.extend([rpn_class_loss , rpn_bbox_loss, mrcnn_class_loss , mrcnn_bbox_loss])


        ##----------------------------------------------------------------------------                
        ## END   Training Mode Layers
        ##----------------------------------------------------------------------------                
        ## BEGIN Inference Mode
        ##----------------------------------------------------------------------------                
        else:
            ##------------------------------------------------------------------------------------
            ##  FPN Layer - Network Heads - Proposal classifier and BBox regressor heads
            ##------------------------------------------------------------------------------------
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rpn_roi_proposals, mrcnn_feature_maps, config.IMAGE_SHAPE,
                                     config.POOL_SIZE, config.NUM_CLASSES)
            ##------------------------------------------------------------------------------------
            ##  Detection Layer
            ##------------------------------------------------------------------------------------
            ##  Generate detection targets
            ##    generated RPNs + mrcnn predictions ----> Target ROIs
            ##
            ## output is [batch_sz, num_detections, (y1, x1, y2, x2, class_id, score)] 
            ##           in image coordinates
            ##------------------------------------------------------------------------------------
            detections = DetectionLayer(config, name="mrcnn_detection")\
                                        ([rpn_roi_proposals, mrcnn_class, mrcnn_bbox, input_image_meta])
            print('<<<  shape of DETECTIONS : ', KB.int_shape(detections), ' Keras tensor ', KB.is_keras_tensor(detections) )                         
            

            ##------------------------------------------------------------------------------------
            ## CHM Inference Layer(s) to generate contextual feature maps using outputs from MRCNN 
            ##------------------------------------------------------------------------------------ 
            pr_hm,  pr_hm_scores, pr_tensor = CHMLayerInference(config, name = 'cntxt_layer' ) ([detections])
            print('<<<  shape of pr_hm   : ', pr_hm.shape  , ' Keras tensor ', KB.is_keras_tensor(pr_hm) )                         
            print('<<<  shape of pr_hm_scores : ', pr_hm_scores.shape, ' Keras tensor ', KB.is_keras_tensor(pr_hm_scores) )                         
            print('<<<  shape of pr_tensor    : ', pr_tensor.shape   , ' Keras tensor ', KB.is_keras_tensor(pr_tensor) )                         
                                        
            inputs  = [ input_image, input_image_meta]
            outputs = [ detections , rpn_roi_proposals, mrcnn_class, mrcnn_bbox, pr_hm, pr_hm_scores]
                        #rpn_class, rpn_bbox, pr_hm, pr_hm_scores]

            # end if Inference Mode        
        model = KM.Model( inputs, outputs,  name='mask_rcnn')

        if mode == "training":
            print(' ================================================================')
            print(' self.keras_model.losses : ', len(model.losses))
            print(model.losses)
            print(' ================================================================')
        
        # Add multi-GPU support.
        # if config.GPU_COUNT > 1:
            # from parallel_model import ParallelModel
            # model = ParallelModel(model, config.GPU_COUNT)

        print('\n>>> Build MaskRCNN build complete. mode: ', mode)
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

        assert self.mode   == "inference", "Create model in inference mode."
        assert len(images) == self.config.BATCH_SIZE, "len(images): {:3d} must be equal to BATCH_SIZE: {:3d}".format(len(images),self.config.BATCH_SIZE)
        
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
        # rpn_roi_proposals, rpn_class, rpn_bbox,\
        # pr_hm, pr_hm_scores, , pr_tensor
        detections, rpn_roi_proposals, mrcnn_class, mrcnn_bbox, pr_hm, pr_scores_by_class =  \
                  self.keras_model.predict([molded_images, image_metas], verbose=0)

        print('Return from  predict()')
        print('    Length of detections : ', len(detections))
        # print('    detections \n', detections)
        print('    Length of rpn_roi_proposals   : ', len(rpn_roi_proposals   ))
        # print('    Length of rpn_class  : ', len(rpn_class  ))
        # print('    Length of rpn_bbox   : ', len(rpn_bbox   ))
        print('    Length of mrcnn_class  : ', len(mrcnn_class))
        print('    Length of mrcnn_bbox   : ', len(mrcnn_bbox ))
        print('    Length of pr_hm        : ', len(pr_hm))
        print('    Length of pr_hm_scores : ', len(pr_scores_by_class))

        ## Process detections
        results = []
        for i, image in enumerate(images):

            # final_rois, final_class_ids, final_scores, \
            # final_pre_scores, final_fcn_scores          \
            # =  self.unmold_detections_new(fcn_hm_scores[i],  detections[i], image.shape, windows[i])    
            final_rois, final_class_ids, final_scores, molded_rois = self.unmold_detections(detections[i], 
                                                                               image.shape  ,
                                                                               windows[i])    

            ## reshape pr_scores from pre_class to per_image
            ## pr_hm_scores is by image/class/bounding box
            # Convert pr_hm_scores bboxes from NN coordinates to image coordinates
            # pr_boxes_adj = utils.boxes_to_image_domain(pr_scores_by_class[i,:,:,:4],image_metas[i])
            # pr_scores_by_class= np.dstack((pr_boxes_adj, pr_scores_by_class[i,:,:,4:]))            
            pr_scores_by_image = utils.byclass_to_byimage_np(pr_scores_by_class[i], 6)
            
            
            np.set_printoptions(linewidth=180,precision=4,threshold=10000, suppress = True)
            print(' pr_scores_by_class shape:', pr_scores_by_class.shape)
            print(' molded_rois:', molded_rois.shape)
            print(molded_rois)    
            print(' final_rois:', final_rois.shape)
            print(final_rois)
            print(' pr_scores_by_image:', pr_scores_by_image.shape)
            print(pr_scores_by_image)

            
            results.append({
                "image"        : images[i],
                "molded_image" : molded_images[i], 
                "image_meta"   : image_metas[i],

                "rois"         : final_rois,
                "molded_rois"  : molded_rois,
                "class_ids"    : final_class_ids,
                "scores"       : final_scores,

                "pr_scores"    : pr_scores_by_image,
                "pr_scores_by_class"    : pr_scores_by_class[i],
                "pr_hm"        : pr_hm,
                
            })
        return results 


    ##-------------------------------------------------------------------------------------
    ## Mold Inputs
    ##-------------------------------------------------------------------------------------        
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
            ## Resize image to fit the model expected size
            ## TODO: move resizing to mold_image()
            molded_image, window, scale, padding = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                padding=self.config.IMAGE_PADDING)
            
            ## subtract mean pixel values from image pixels
            molded_image = utils.mold_image(molded_image, self.config)
            
            ## Build image_meta - here active_class_ids are all set to 0
            image_meta = utils.compose_image_meta( 0, 
                                                   image.shape, 
                                                   window,
                                                   np.zeros([self.config.NUM_CLASSES],
                                                   dtype=np.int32))
            ## Append
            molded_images.append(molded_image)
            image_metas.append(image_meta)
            windows.append(window)
        
        ## Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas   = np.stack(image_metas)
        windows       = np.stack(windows)
        return molded_images, image_metas, windows

        
    ##-------------------------------------------------------------------------------------
    ## Unmold Detections 
    ##-------------------------------------------------------------------------------------        
    # def unmold_detections(self, detections, mrcnn_mask, image_shape, window):
    def unmold_detections(self, detections, image_shape, window):
        '''
        Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the application.

        Input:
        --------
        detections  : [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_mask  : [N, height, width, num_classes]
        image_shape : [height, width, depth] Original size of the image before resizing
        window      : [y1, x1, y2, x2] Box in the image where the real image is
                       (i.e.,  excluding the padding surrounding the real image)

        Returns:
        --------
        boxes       : [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids   : [N] Integer class IDs for each bounding box
        scores      : [N] Float probability scores of the class_id
        --- masks       : [height, width, num_instances] Instance masks
        '''
        
        # print('>>>  unmold_detections ')
        # print('     detections.shape : ', detections.shape)
        # print('     mrcnn_mask.shape : ', mrcnn_mask.shape)
        # print('     image_shape.shape: ', image_shape)
        # print('     window.shape     : ', window)
        # print(detections)
        
        ##-----------------------------------------------------------------------------------------
        ## How many detections do we have?
        ##  Detections array is padded with zeros. detections[:,4] identifies the class 
        ##  Find all rows in detection array with class_id == 0 , and place their row indices
        ##  into zero_ix. zero_ix[0] will identify the first row with class_id == 0.
        ##-----------------------------------------------------------------------------------------
        print()
        np.set_printoptions(linewidth=100)        

        zero_ix = np.where(detections[:, 4] == 0)[0]
    
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # print(' np.where() \n', np.where(detections[:, 4] == 0))
        # print('     zero_ix.shape     : ', zero_ix.shape)
        # print('     N is :', N)
        
        ##-----------------------------------------------------------------------------------------
        ## Extract boxes, class_ids, scores, and class-specific masks
        ##-----------------------------------------------------------------------------------------
        
        boxes        = detections[:N, :4]
        molded_boxes = detections[:N, :4]
        class_ids    = detections[:N, 4].astype(np.int32)
        scores       = detections[:N, 5]
        # masks     = mrcnn_mask[np.arange(N), :, :, class_ids]

        ##-----------------------------------------------------------------------------------------
        ## Compute scale and shift to translate coordinates to image domain.
        ##-----------------------------------------------------------------------------------------
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale   = min(h_scale, w_scale)
        shift   = window[:2]  # y, x
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

        ##-----------------------------------------------------------------------------------------
        ## Translate bounding boxes to image domain
        ##-----------------------------------------------------------------------------------------
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

        ##-----------------------------------------------------------------------------------------
        ## Filter out detections with zero area. Often only happens in early
        ## stages of training when the network weights are still a bit random.
        ##-----------------------------------------------------------------------------------------
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
            
        if exclude_ix.shape[0] > 0:
            boxes        = np.delete(boxes, exclude_ix, axis=0)
            molded_boxes = np.delete(molded_boxes, exclude_ix, axis=0)
            class_ids    = np.delete(class_ids, exclude_ix, axis=0)
            scores       = np.delete(scores, exclude_ix, axis=0)
            N            = class_ids.shape[0]

        return boxes, class_ids, scores, molded_boxes     # , full_masks
        
        
    ##-------------------------------------------------------------------------------------
    ##  Train 
    ##-------------------------------------------------------------------------------------        
    def train(self, 
              train_dataset, 
              val_dataset, 
              learning_rate     = 1.0, 
              layers            = None,
              losses            = None,
              epochs            = 0,
              epochs_to_run     = 0,
              batch_size        = 0, 
              steps_per_epoch   = 0,
              min_lr            = 0,
              debug             = False):
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
        

        print('type train_dataset:', type(train_dataset))
        print('type val_dataset:', type(val_dataset))
        
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
                                          batch_size=32,
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

        ##----------------------------------------------------------------------------------------------
        ## Setup optimizaion method 
        ##----------------------------------------------------------------------------------------------            
        optimizer = self.set_optimizer()

        # Train

        self.set_trainable(layers)
        # self.compile(learning_rate, self.config.LEARNING_MOMENTUM, losses)
        self.compile(losses, optimizer)

        ##----------------------------------------------------------------------------------------------
        ## If in debug mode write stdout intercepted IO to output file  
        ##----------------------------------------------------------------------------------------------            
        if self.config.SYSOUT == 'FILE':
            utils.write_sysout(self.log_dir)
        
        log("Starting at epoch   {} of {} epochs. LR={}\n".format(self.epoch, epochs, learning_rate))
        log("Steps per epoch     {} ".format(steps_per_epoch))
        log("Batch size          {} ".format(batch_size))
        log("Checkpoint Path:    {} ".format(self.checkpoint_path))
        log("Learning Rate       {} ".format(self.config.LEARNING_RATE))
        log("Momentum            {} ".format(self.config.LEARNING_MOMENTUM))
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

                
    ##-------------------------------------------------------------------------------------
    ## Compile
    ##-------------------------------------------------------------------------------------        
    # def compile(self, learning_rate, momentum, losses):
    def compile(self, losses, optimizer):
        '''
        Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        '''
        assert isinstance(losses, list) , "A loss function must be defined as the objective"

        ##----------------------------------------------------------------------------------------------
        ## Setup optimizaion method 
        ##----------------------------------------------------------------------------------------------            
        optimizer = self.set_optimizer()

        # Optimizer object
        print('\n')
        print(' Compile Model :')
        print('----------------')
        print('    losses        : ', losses)
        print('    optimizer     : ', optimizer)
        print('    learning rate : ', self.config.LEARNING_RATE)
        print('    momentum      : ', self.config.LEARNING_MOMENTUM)
        print()
        
        ##------------------------------------------------------------------------
        ## Add Losses
        ## These are the losses aimed for minimization
        ## Normally Keras expects the same number of losses as outputs. Since we 
        ## are returning more outputs , we go deep and set the losses in the base
        ## layers () 
        ##------------------------------------------------------------------------    
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}

        print(' Add losses:')
        print('----------------')
        print('    losses: ', losses)
        print('    keras_model.losses           :', self.keras_model.losses)
        print()

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
        print('Keras model.losses : ') 
        print('---------------------') 
        for ls in self.keras_model.losses:
            print('  ',ls, '   name:',ls.name )
        
        print()    
        print('Keras_model._losses:' ) 
        print('---------------------' ) 
        for ls in self.keras_model._losses:
            print('  ',ls, '   name:',ls.name )

        print()    
        print('Keras_model._per_input_losses:')
        print('------------------------------')
        for ls in self.keras_model._per_input_losses:
            print('  ',ls, '   name:',type(ls))
        pp.pprint(self.keras_model._per_input_losses)
            
        ##-------------------------------------------------------------------------------    
        ## Add L2 Regularization as loss to list of losses
        ## Skip gamma and beta weights of batch normalization layers.
        ##-------------------------------------------------------------------------------
        reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                      for w in self.keras_model.trainable_weights
                      if 'gamma' not in w.name and 'beta' not in w.name]
                      
                      
        print()    
        print('L2 Regularization losses:')
        print('-------------------------')
        for ls in reg_losses:
            print('  ',ls, '   name:',ls.name )

        self.keras_model.add_loss(tf.add_n(reg_losses))

        print()
        print('    Final list of keras_model.losses ') 
        print('    -------------------------------- ') 
        pp.pprint(self.keras_model.losses)
        

        ##------------------------------------------------------------------------    
        ## Compile
        ##------------------------------------------------------------------------   
        print()
        print (' Length of Keras_Model.outputs:', len(self.keras_model.outputs))
        self.keras_model.compile(optimizer=optimizer, 
                                 loss=[None] * len(self.keras_model.outputs))

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
        print(' Final Keras metric_names:') 
        print(' -------------------------')
        pp.pprint(self.keras_model.metrics_names)                                 
        print()
        return

"""

    ##-------------------------------------------------------------------------------------
    ## Unmold Detections - New 
    ##-------------------------------------------------------------------------------------        
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
"""
