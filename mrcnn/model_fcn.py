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
import os, sys, glob, random, math, datetime, itertools, json, re, logging, pprint
from   collections import OrderedDict
import numpy as np
import scipy.misc
import tensorflow as tf

import keras
import keras.backend as KB
import keras.layers  as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM
from   keras.utils.generic_utils import Progbar

#sys.path.append('..')

import mrcnn.utils                 as utils
import mrcnn.loss                  as loss
from   mrcnn.datagen               import data_generator
from   mrcnn.utils                 import (log, logt,  parse_image_meta_graph, parse_image_meta, 
                                           parse_active_class_ids_graph)
from   mrcnn.model_base            import ModelBase
# from   mrcnn.fcn16_layer           import fcn16_graph
# from   mrcnn.fcn_layer_no_L2       import fcn_graph
# from   mrcnn.fcn_scoring_layer     import FCNScoringLayer 
from   mrcnn.fcn_scoring_layer     import fcn_scoring_graph

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3.0")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')
pp = pprint.PrettyPrinter(indent=4, width=100)
tf.get_variable_scope().reuse_variables()

#  from keras.callbacks import TensorBoard
#  
#  class LRTensorBoard(TensorBoard):
#      def __init__(self, log_dir):  # add other arguments to __init__ if you need
#          super().__init__(log_dir=log_dir)
#  
#      def on_epoch_end(self, epoch, logs=None):
#          logs.update({'lr': KB.eval(self.model.optimizer.lr)})
#          super().on_epoch_end(epoch, logs)
        
############################################################
##  FCN Class
############################################################
class FCN(ModelBase):
    """Encapsulates the FCN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, arch, config):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        assert arch in ['FCN32', 'FCN16', 'FCN8', 'FCN32L2', 'FCN8L2']
        super().__init__(mode, config)

        print('>>> Initialize FCN model, mode: ',mode, 'architecture: ', arch)
        if mode == 'training':
            if config.NEW_LOG_FOLDER:
                self.set_log_dir()
            else:
                print('    Use existing folder if possible')
                last_log_dir = self.find_last()[1]
                print('    Last log dir is :', last_log_dir)
                self.set_log_dir(last_log_dir)

        
        self.arch = arch
        self.mode = mode
        
        if arch == 'FCN8':
            print('    arch set to FCN8 - with No L2 Regularization')
            from   mrcnn.fcn8_layer        import fcn8_graph as fcn_graph

        elif arch == 'FCN8L2':
            print('    arch set to FCN8 - with L2 Regularization')
            from   mrcnn.fcn8_layer_L2     import fcn8_l2_graph as fcn_graph
        
        elif arch == 'FCN32':
            print('    arch set to FCN32')
            from   mrcnn.fcn32_layer       import fcn32_graph as fcn_graph

        elif arch == 'FCN32L2':
            print('    arch set to FCN32 - with L2 Regularization')
            from   mrcnn.fcn32_layer_L2     import fcn32_l2_graph as fcn_graph
            
        else :    # arch == 'FCN16':
            print('    arch set to FCN16')
            from   mrcnn.fcn16_layer       import fcn16_graph as fcn_graph

        
        self.fcn_graph = fcn_graph
        print(self.fcn_graph)

        
        # Pre-defined layer regular expressions
        self.layer_regex = {
            # "res5+": r"(res5.*)|(bn5.*)",
            # All layers
            "all": ".*",
            # fcn32+ 
            "fcn32+" : r"(fcn32\_.*)|(fcn16\_.*)|(fcn8\_.*)",
            # fcn16+ 
            "fcn16+" : r"(fcn16\_.*)|(fcn8\_.*)",

            # fcn32 only 
            "fcn32" : r"(fcn32\_.*)",
            # fcn16 only 
            "fcn16" : r"(fcn16\_.*)",
            # fcn8 only 
            "fcn8" : r"(fcn8\_.*)",
            # fcn only 
            "fcn" : r"(fcn\_.*)",
            # fcn and fc2
            "fc2+" : r"(fcn\_.*)|(fc2.*)",
            # fcn, fc2, fc1
            "fc1+" : r"(fcn\_.*)|(fc2*)|(fc1*)",
            # block5
            "block5" : r"(block5\_.*)",
            # block5+
            "block5+" : r"(block5\_.*)|(fcn\_.*)|(fc2*)|(fc1*)|(fcn32\_.*)|(fcn16\_.*)|(fcn8\_.*)",
            # block4
            "block4" : r"(block4\_.*)",
            # block4+
            "block4+" : r"(block4\_.*)|(block5\_.*)|(fcn32\_.*)|(fcn16\_.*)|(fcn8\_.*)"
            # -- mrcnn
            # "mrcnn" : r"(mrcnn\_.*)",
            # -- all layers but the backbone
            # "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # -- all layers but the backbone
            # "allheads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(fcn\_.*)",
            # -- ResNet from a specific stage and up
            # "res3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)",
            # "res4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)",
            # -- From a specific Resnet stage and up
            # "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
        }
        
        self.keras_model = self.build(mode=mode, config=config  )

        print('>>> FCN initialization complete. mode: ',mode) 
        return
    
    
    ##------------------------------------------------------------------------------------    
    ## Build Model 
    ##------------------------------------------------------------------------------------                    
    def build(self, mode, config, FCN_layers = True):
        '''
        Build FCN architecture 
            input_shape: The shape of the input heatmap from MRCNN.
            mode:        Either "training" or "inference". The inputs and
                         outputs of the model differ accordingly.
        '''
        assert mode in ['training', 'inference']
        verbose = config.VERBOSE


        print('\n')
        print('---------------------------------------------------')
        print(' Build FCN Model -  Arch: ',self.arch, ' mode: ', mode)
        print('---------------------------------------------------')
        
        # Image size must be dividable by 2 multiple times
        h, w = config.FCN_INPUT_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")
        
        num_classes = config.NUM_CLASSES
        if mode == 'training':
            num_scores_columns = 23
            num_bboxes  = config.TRAIN_ROIS_PER_IMAGE      # 200
        else:    
            num_scores_columns = 24
            num_bboxes  = config.DETECTION_MAX_INSTANCES   # 100


        ##------------------------------------------------------------------                            
        ##  Input Layer
        ##------------------------------------------------------------------
        # input_image      = KL.Input(shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        input_image_meta = KL.Input(shape=[None], name="input_image_meta")
        pr_hm            = KL.Input(shape=[h,w, num_classes], name="input_pr_hm_norm" , dtype=tf.float32 )
        pr_hm_scores     = KL.Input(shape=[num_classes, num_bboxes, num_scores_columns], name="input_pr_hm_scores", dtype=tf.float32)
        
        ##----------------------------------------------------------------------------                
        ## FCN Training Mode Layers
        ##----------------------------------------------------------------------------                
        if mode == "training":
            gt_hm        = KL.Input(shape=[ h,w, num_classes], name="input_gt_hm_norm"  , dtype=tf.float32)
            gt_hm_scores = KL.Input(shape=[ num_classes, num_bboxes, num_scores_columns], name="input_gt_hm_scores", dtype=tf.float32)

            _, _, _, active_class_ids = KL.Lambda(lambda x:  parse_image_meta_graph(x), mask=[None, None, None, None],
                                                  name = 'active_class_ids')(input_image_meta)
            # active_class_ids = KL.Lambda(lambda x:  parse_active_class_ids_graph(x), name = 'active_class_ids')(input_image_meta)
             
            print('   active_class_ids  shape is : ', KB.int_shape(active_class_ids), ' Keras tensor ', KB.is_keras_tensor(active_class_ids) )        

            fcn_hm, fcn_sm = self.fcn_graph(pr_hm , config, mode = mode)
            
            print('  * gt_hm_scores shape: ', KB.int_shape(gt_hm_scores), ' Keras tensor ', KB.is_keras_tensor(gt_hm_scores) )        
            print('  * pr_hm_scores shape: ', KB.int_shape(pr_hm_scores), ' Keras tensor ', KB.is_keras_tensor(pr_hm_scores) )        
            print('  * fcn_heatmap shape : ', KB.int_shape(fcn_hm), ' Keras tensor ', KB.is_keras_tensor(fcn_hm) )        
            print('  * fcn_softmax shape : ', KB.int_shape(fcn_sm), ' Keras tensor ', KB.is_keras_tensor(fcn_sm) )        

            fcn_scores   = KL.Lambda(lambda x:  fcn_scoring_graph(x, config, mode), name = 'fcn_scoring')([fcn_hm , pr_hm_scores])
            
            print('  * fcn_scores shape: ', KB.int_shape(fcn_scores), ' Keras tensor ', KB.is_keras_tensor(fcn_scores) )        
            
            ##------------------------------------------------------------------------
            ##  Loss layer definitions
            ##------------------------------------------------------------------------
            print('\n')
            print('---------------------------------------------------')
            print('    building Loss Functions ')
            print('---------------------------------------------------')
            fcn_MSE_loss = KL.Lambda(lambda x: loss.fcn_heatmap_MSE_loss_graph(*x), name="fcn_MSE_loss") \
                            ([gt_hm, fcn_hm])                                                 
                            
            fcn_CE_loss  = KL.Lambda(lambda x: loss.fcn_heatmap_CE_loss_graph(*x) , name="fcn_CE_loss") \
                            ([gt_hm, fcn_hm, active_class_ids])
                            
            fcn_BCE_loss  = KL.Lambda(lambda x: loss.fcn_heatmap_BCE_loss_graph(*x) , name="fcn_BCE_loss") \
                            ([gt_hm, fcn_hm])
                            
            # Model Inputs 
            inputs  = [input_image_meta, pr_hm, pr_hm_scores, gt_hm, gt_hm_scores]
            outputs = [fcn_hm, fcn_sm, fcn_MSE_loss, fcn_BCE_loss, fcn_scores]
            
        # end if Training
        ##----------------------------------------------------------------------------                
        ## FCN Inference Mode Layers
        ##----------------------------------------------------------------------------                
        else:
                    
            fcn_hm, fcn_sm = self.fcn_graph(pr_hm , config, mode)
            logt('* fcn_heatmap shape: ', fcn_hm, verbose = verbose)        
            logt('* fcn_softmax shape: ', fcn_sm, verbose = verbose)        

            fcn_scores   = KL.Lambda(lambda x:  fcn_scoring_graph(x, config, mode), name = 'fcn_scoring')([fcn_hm , pr_hm_scores])
            logt('* fcn_scores shape : ', fcn_scores, verbose = verbose )        
            
            inputs  = [ pr_hm , pr_hm_scores]                                        
            outputs = [ fcn_hm, fcn_sm, fcn_scores]

        # end if Inference Mode        
            
            
        model = KM.Model( inputs, outputs,  name='FCN')
                
        print(' ================================================================')
        print(' self.keras_model.losses : ', len(model.losses))
        for idx,ls in enumerate(model.losses):
            print(idx, '    ', ls)
        print(' ================================================================')
            
        # Add multi-GPU support.
        # if config.GPU_COUNT > 1:
            # from parallel_model import ParallelModel
            # model = ParallelModel(model, config.GPU_COUNT)

        print('\n>>> FCN build complete. mode: ',mode)
        return model

        
##-------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------        
##-------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------        

    def detect(self, input, verbose=0):
        '''
        Runs the FCN detection pipeline on an input batch (heatmaps + scores).
        Input:
        --------    
        pr_hm :         Heatmap [Bsz, hm_w, hm_h, num_classes]  
        pr_hm_scores:   Heatmap Scores by class [BSz, num_classes, num_detections, columns]
        image_metas:    Image Meta information required for unmolding bounding box coordinates

        Returns:        a list of dicts, one dict per image. The dict contains:
        --------
        rois:           [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids:      [N] int class IDs
        scores:         [N] float probability scores for the class IDs
        masks:          [H, W, N] instance binary masks
        '''
        
        pr_hm, pr_hm_scores, image_metas = input
        sequence_column = 7
        if verbose:
            print('===> call fcn predict()')
            print('     pr_hm         ', pr_hm.shape)
            print('     pr_hm_scores  ', pr_hm_scores.shape)
            print('     image_metas   ', image_metas.shape)
        fcn_hm, fcn_sm, fcn_hm_scores = self.keras_model.predict([pr_hm, pr_hm_scores], verbose = 0)

        if verbose:
            print('    results from fcn.keras_model.predict()')
            print('    Length of fcn_heatmaps : ', len(fcn_hm), fcn_hm.shape)
            print('    Length of fcn_softmax  : ', len(fcn_sm), fcn_sm.shape)
            print('    Length of fcn_hm_scores: ', len(fcn_hm_scores), fcn_hm_scores.shape)
         
        ## Process detections                
        ## build results list. Takes images, mrcnn_detections, pr_hm, pr_hm_scores and fcn outputs

        results = []
        
        for i in range(pr_hm.shape[0]):
            
            ## reshape fcn_hm_scores from per_class to per_image tensor
            ## fcn_hm_scores is by class  
            ## Convert fcn_hm_scores bboxes from NN coordinates to image coordinates
            fcn_boxes_adj = utils.boxes_to_image_domain(fcn_hm_scores[i,:,:,:4],image_metas[i])
            fcn_scores_by_class= np.dstack((fcn_boxes_adj, fcn_hm_scores[i,:,:,4:]))
            fcn_scores_by_image = utils.byclass_to_byimage_np(fcn_scores_by_class, sequence_column)
            if verbose:
                print(' Process input/results ', i)
                print(' pr_hm              :', pr_hm[i].shape)
                print(' pr_hm_scores       :', pr_hm_scores[i].shape)
                print(' fcn_hm             :', fcn_hm[i].shape)
                print(' fcn_hm_scores      :', fcn_hm_scores[i].shape)
                print(' fcn_scores_by_class:', fcn_scores_by_class.shape)
            
            results.append({
                "fcn_scores_by_class" : fcn_scores_by_class,
                "fcn_scores"          : fcn_scores_by_image,
                "fcn_hm_scores"       : fcn_hm_scores[i], 
                "fcn_hm"              : fcn_hm[i],
                "fcn_sm"              : fcn_sm[i]
            })            
                    
        return results 
        
    ##-------------------------------------------------------------------------------------
    ##  detect_from_images
    ##-------------------------------------------------------------------------------------        
        
    def detect_from_images(self, mrcnn_model, images, verbose=0):
        '''
        Runs the detection pipeline from an input of images.

        images:         List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois:           [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids:      [N] int class IDs
        scores:         [N] float probability scores for the class IDs
        masks:          [H, W, N] instance binary masks
        
        detections           (200, 6)
        fcn_hm               (256, 256, 81)
        fcn_scores           (8, 23)
        fcn_scores_by_class  (81, 200, 23)
        fcn_sm               (256, 256, 81)
        image                (640, 480, 3)
        image_meta           (89,)
        molded_image         (1024, 1024, 3)
        molded_rois          (8, 4)
        
        pr_hm                (256, 256, 81)
        pr_hm_scores         (81, 200, 23)
        pr_scores            (8, 23)
        pr_scores_by_class   (81, 200, 23)
        
        
        '''
        
        # print('call fcn.detect_from_images()')
        results = mrcnn_model.detect(images, verbose = verbose)

        if verbose:
            print('===>   fcn.detect_from_images() : return from  mrcnn.detect() : ', len(results))
            for i, r in enumerate(results):
                print('\noutputs returned from mrcnn.detect()  ', i, '  ',sorted(r.keys()))
                for key in sorted(r):
                    print(key.ljust(20), r[key].shape)        
        
        ## prep results from mrcnn detections to pass to fcn detection
        # for result in results:
        fcn_input_hm = np.stack([r['pr_hm'] for r in results] )
        fcn_input_hm_scores = np.stack([r['pr_hm_scores'] for r in results] )
        fcn_input_image_metas = np.stack([r['image_meta'] for r in results] )

        # print('===> fcn.detect_from_images() :call fcn.detect()')        
        # fcn_hm, fcn_sm, fcn_hm_scores = self.keras_model.predict([fcn_input_hm, fnc_input_hm_scores], verbose = 1)
        fcn_results = self.detect([fcn_input_hm, fcn_input_hm_scores, fcn_input_image_metas], verbose = verbose)

        if verbose:
            print('===>  fcn.detect_from_images() : return from  fcn.detect() : ', len(fcn_results))
            for i, r in enumerate(fcn_results):
                print('\n outputs returned from fcn.detect() ', i, '  ',sorted(r.keys()))
                for key in sorted(r):
                    print(key.ljust(20), r[key].shape)        
                
        ## Process detections        
        ## build results list. Takes images, mrcnn_detections, pr_hm, pr_hm_scores and fcn outputs

        for i, image in enumerate(images):
            results[i].update(fcn_results[i])
            if verbose:
                print('Unmold image ', i, 'image_shape: ', image.shape)
                print('    pr_hm_scores          : ', results[i]['pr_hm_scores'].shape)
                print('    mrcnn_class_ids       : ', results[i]['class_ids'])
                print('    fcn_hm_scores         : ', fcn_results[i]['fcn_scores'].shape)
                print('    fcn_hm_scores         : ', results[i]['fcn_scores'].shape)
                print('    fcn_hm_scores_by_class: ', fcn_results[i]['fcn_scores_by_class'].shape)
                print('    fcn_hm_scores_by_class: ', results[i]['fcn_scores_by_class'].shape)
                print('--- fcn.detect_from_images() complete')
            
        return results 
        
        
    ##-------------------------------------------------------------------------------------
    ##  detect_from_images
    ##-------------------------------------------------------------------------------------                
    def evaluate(self, mrcnn_model, evaluate_batch, verbose=0):
        '''
        Runs the evaluation pipeline:
        Pass Input --> MRCNN (evaluation mode) ---> FCN (inference mode) --> Results
        
        In evaluation mode, the MRCNN detection process adds False Positive object bounding detections 
        for all (or a selected ssubset) of ground truth annotations to the set of  "detected ROIs".
    
        These modified detections are then passed to FCN for detection. 
        

        evaluate_batch:          [input_image, input_image_meta, input_gt_class_ids, input_gt_boxes]
           input_image:          List of images, potentially of different sizes.        
        
        Returns a list of dicts, one dict per image. The dict contains:
            N : number of detections 
            
            rois:                [N, (y1, x1, y2, x2)] detection bounding boxes
            class_ids:           [N] int class IDs
            scores:              [N] float probability scores for the class IDs
            masks:               [H, W, N] instance binary masks
            
            detections           (DETECTION_MAX_INSTANCES, 6)
            image                (image_height, img_width, num_channels)
            image_meta           (89,)
            molded_image         (1024, 1024, 3)
            molded_rois          (N, 4)
            pr_hm                (1, 256, 256, 81)
            pr_scores            (N, 23)
            pr_scores_by_class   (81, 200, 23)
        '''

        assert self.mode   == "inference", "Create model in evaluate mode."
        assert len(evaluate_batch) == 5, " length of eval batch must be 4"
        sequence_column = 7
        
        results = mrcnn_model.evaluate(evaluate_batch, verbose = verbose)

        if verbose:
            print('===>   return from  MRCNN evaluate() : ', len(results))
            for i, r in enumerate(results):
                print('\n output ', i, '  ',sorted(r.keys()))
                for key in sorted(r):
                    print(key.ljust(20), r[key].shape)        
        
        ## prep results from mrcnn detections to pass to fcn detection
        fcn_input_hm          = np.stack([r['pr_hm'] for r in results] )
        fcn_input_hm_scores   = np.stack([r['pr_hm_scores'] for r in results] )
        fcn_input_image_metas = np.stack([r['image_meta'] for r in results] )

        # fcn_hm, fcn_sm, fcn_hm_scores = self.keras_model.predict([fcn_input_hm, fnc_input_hm_scores], verbose = 1)
        fcn_results = self.detect([fcn_input_hm, fcn_input_hm_scores, fcn_input_image_metas], verbose = verbose)

        if verbose:
            print('===>  return from  FCN detect() : ', len(fcn_results))
            for i, r in enumerate(fcn_results):
                print('\n output ', i, '  ',sorted(r.keys()))
                for key in sorted(r):
                    print(' output: ', key.ljust(20), '      ', r[key].shape)        
                
        ## Process detections        
        ## build results list. Takes images, mrcnn_detections, pr_hm, pr_hm_scores and fcn outputs

        for i in range(len(fcn_results)):
            results[i].update(fcn_results[i])
                    
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
        
    ##-------------------------------------------------------------------------------------
    ## unmold_detections
    ## 18-11-2018
    ## currently we are not using this, the operations applied here are perfromed by 
    ## utils.boxes_to_image_domain
    ##-------------------------------------------------------------------------------------        
    def unmold_detections(self, detections, image_shape, window):
        '''
        Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the application.

        Input:
        --------
        detections  : [N, (y1, x1, y2, x2, class_id, score, seq_id, norm_score, hm_score, bbox_area, norm_hm_score, fcn_score, norm_fcn_score )]
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
        print('     detections.shape : ', detections.shape)
        # print('     mrcnn_mask.shape : ', mrcnn_mask.shape)
        print('     image_shape.shape: ', image_shape)
        print('     window.shape     : ', window)
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
        print('     zero_ix.shape     : ', zero_ix.shape)
        print('     N is :', N)

        ##-----------------------------------------------------------------------------------------
        ## Extract boxes, class_ids, scores, and class-specific masks
        ##-----------------------------------------------------------------------------------------
        boxes        = detections[:N, :4]
        class_ids    = detections[:N, 4].astype(np.int32)
        mrcnn_scores = detections[:N, 7]
        fcn_scores   = detections[:N, 12]
        non_zero_detections = detections[:N, 4:]

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
            non_zero_detections = np.delete(non_zero_detections, exclude_ix, axis=0)
            boxes     = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            mrcnn_scores= np.delete(mrcnn_scores, exclude_ix, axis=0)
            fcn_scores= np.delete(fcn_scores, exclude_ix, axis=0)
            N         = class_ids.shape[0]
            
        print('     N is :', N)
        non_zero_detections = np.hstack((boxes, non_zero_detections))
        
        return [non_zero_detections, boxes, np.expand_dims(class_ids, axis =-1), np.expand_dims(mrcnn_scores, axis =-1), np.expand_dims(fcn_scores, axis =-1) ]

        
        
    ##---------------------------------------------------------------------------------------------
    ## Compile 
    ## 
    ##    Note: Using l2 regularizers in the model adds a loss for each layer using the regularization
    ##---------------------------------------------------------------------------------------------            
    def compile(self, loss_functions, optimizer):
        '''
        Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        '''
        assert isinstance(loss_functions, list) , "A loss function must be defined as the objective"
        
        # Optimizer object
        print('\n')
        print('  Compile Model :')
        print(' ----------------')
        print('    losses        : ', loss_functions)
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

        print(' Initial self.keras_model.losses :')
        print(' ---------------------------------')
        print(' losses passed to compile : ', loss_functions)
        print(' self.keras_model.losses  : ')
        for idx, i in enumerate(self.keras_model.losses):
            print('     ',idx, '  ', i)
        
        print()
        print(' Add loss_functions to self.keras_model.losses')
        print(' -------------------------------------')
        
        for name in loss_functions:
            layer = self.keras_model.get_layer(name)
            print(' --  Loss: {}  Related Layer is : {}'.format(name, layer.name))
            if layer.output in self.keras_model.losses:
                print('      ',layer.output,' is already in self.keras_model.losses, and wont be added to list')
                continue
            print('    >> Add add loss for ', layer.output, ' to list of losses...')
            self.keras_model.add_loss(tf.reduce_mean(layer.output, keepdims=True))
            
        print()    
        print(' self.keras_model.losses after adding loss_functions passed to compile() : ') 
        print(' ------------------------------------------------------------------------- ') 
        for idx, i in enumerate(self.keras_model.losses):
            print('     ',idx, '  ', i)

        print()    
        print(' Keras_model._losses:' ) 
        print(' --------------------' ) 
        for idx, i in enumerate(self.keras_model._losses):
            print('     ',idx, '  ', i)

        print()    
        print(' Keras_model._per_input_losses:')
        print(' ------------------------------')
        for idx, i in enumerate(self.keras_model._per_input_losses):
            print('     ',idx, '  ', i)
        # pp.pprint(self.keras_model._per_input_losses)
            
            
        ##------------------------------------------------------------------------    
        ## Add L2 Regularization as loss to list of losses
        ##------------------------------------------------------------------------    
        # Skip gamma and beta weights of batch normalization layers.
        # reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                      # for w in self.keras_model.trainable_weights
                      # if 'gamma' not in w.name and 'beta' not in w.name]
        # self.keras_model.add_loss(tf.add_n(reg_losses))

        print()    
        print(' Final list of keras_model.losses, after adding L2 regularization as loss to list : ') 
        print(' ---------------------------------------------------------------------------------- ') 
        for idx, i in enumerate(self.keras_model.losses):
            print('     ',idx, '  ', i)

        ##------------------------------------------------------------------------    
        ## Compile
        ##------------------------------------------------------------------------   
        print()
        print(' Compile ')
        print(' --------')
        print (' Length of Keras_Model.outputs:', len(self.keras_model.outputs))
        self.keras_model.compile(optimizer=optimizer, loss=[None] * len(self.keras_model.outputs))

        #------------------------------------------------------------------------    
        # Add metrics for losses
        #------------------------------------------------------------------------    
        print()
        print(' Add Metrics for losses :')
        print(' -------------------------')                                 
        print(' Initial Keras metric_names:', self.keras_model.metrics_names)                                 

        for name in loss_functions:
            if name in self.keras_model.metrics_names:
                print('      ' , name , 'is already in in self.keras_model.metrics_names')
                continue
            layer = self.keras_model.get_layer(name)
            print('    Loss name : {}  Related Layer is : {}'.format(name, layer.name))
            self.keras_model.metrics_names.append(name)
            self.keras_model.metrics_tensors.append(tf.reduce_mean(layer.output, keepdims=True))
            print('    >> Add metric ', name, ' with metric tensor: ', layer.output.name, ' to list of metrics ...')
        
        print()
        print (' Final Keras metric_names :') 
        print (' --------------------------') 
        for idx, i in enumerate(self.keras_model.metrics_names):
            print('     ',idx, '  ', i)

        print()
        print(' self.keras_model.losses after adding losses passed to compile() : ') 
        print(' ----------------------------------------------------------------- ') 
        for idx, i in enumerate(self.keras_model.losses):
            print('     ',idx, '  ', i)

        print()    
        print(' Keras_model._losses:' ) 
        print(' ---------------------' ) 
        for idx, i in enumerate(self.keras_model._losses):
            print('     ',idx, '  ', i)

        print()    
        print(' Keras_model._per_input_losses:')
        print(' ------------------------------')
        for idx, i in enumerate(self.keras_model._per_input_losses):
            print('     ',idx, '  ', i)
        print()
        
        return
        
        
        
    ##---------------------------------------------------------------------------------------------
    ##   Train
    ##---------------------------------------------------------------------------------------------
    def train(self, 
              train_dataset, 
              val_dataset, 
              learning_rate     = 0, 
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
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         batch_size=batch_size, augment=False)
        val_generator   = data_generator(val_dataset, self.config, shuffle=True,
                                         batch_size=batch_size, augment=False)
                                        

        ##--------------------------------------------------------------------------------
        ## Callbacks
        ## call back for model checkpoint was originally (?) loss. chanegd to val_loss (which is default) 2-5-18
        ##--------------------------------------------------------------------------------
            
        callbacks_list = [
              # keras.callbacks.ProgbarLogger(count_mode='steps',
                                            # stateful_metrics=self.keras_model.stateful_metric_names)
            
            keras.callbacks.BaseLogger(stateful_metrics=self.keras_model.stateful_metric_names)
            
            , keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                          histogram_freq=1,
                                          write_graph=True,
                                          write_images=False, 
                                          write_grads=True,
                                          batch_size=self.config.BATCH_SIZE)
                                          # write_graph=True,
                                          # write_images=True,
                                          # embeddings_freq=0,
                                          # embeddings_layer_names=None,
                                          # embeddings_metadata=None)
                                          
            , keras.callbacks.ModelCheckpoint(self.checkpoint_path, 
                                              mode    = 'auto', 
                                              period  = self.config.CHECKPOINT_PERIOD, 
                                              monitor = 'val_loss', 
                                              verbose = 1, 
                                              save_best_only = True, 
                                              save_weights_only=True)

            , keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                mode     = 'auto', 
                                                factor   = self.config.REDUCE_LR_FACTOR,   
                                                cooldown = self.config.REDUCE_LR_COOLDOWN,
                                                patience = self.config.REDUCE_LR_PATIENCE,
                                                min_delta= self.config.REDUCE_LR_MIN_DELTA,
                                                min_lr   = self.config.MIN_LR, 
                                                verbose  = 1)                                            
                                                
            , keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                mode      = 'auto', 
                                                min_delta = self.config.EARLY_STOP_MIN_DELTA, 
                                                patience  = self.config.EARLY_STOP_PATIENCE, 
                                                verbose   = 1)                                            
            , keras.callbacks.History() 
        ]
         
        
        callbacks =  keras.callbacks.CallbackList(callbacks = callbacks_list)
        callbacks.set_model(self.keras_model)
        callbacks.set_params({
            'batch_size': batch_size,
            'epochs': epochs,
            'steps': steps_per_epoch,
            'verbose': 1 ,
            'do_validation': True,
            'metrics': callback_metrics
        })
        
        
        
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
        log("Steps per epochs    {} ".format(steps_per_epoch))
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
        return

        
        
    ##---------------------------------------------------------------------------------------------
    ##   Train In Batches 
    ##---------------------------------------------------------------------------------------------
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
              min_LR            = 0,
              shuffle           = True,
              augment           = False):
              
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
        train_generator = data_generator(train_dataset, mrcnn_model.config, 
                                         shuffle = shuffle, 
                                         augment = augment,
                                         batch_size=batch_size)
                                         
        val_generator   = data_generator(val_dataset, mrcnn_model.config, 
                                         shuffle = shuffle, 
                                         augment = False,
                                         batch_size=batch_size)
                                         
        ##--------------------------------------------------------------------------------
        ## Set trainable layers and compile
        ##--------------------------------------------------------------------------------
        self.set_trainable(layers)            

        ##----------------------------------------------------------------------------------------------
        ## Setup optimizaion method 
        ##----------------------------------------------------------------------------------------------            
        optimizer = self.set_optimizer()

        # self.compile(learning_rate, self.config.LEARNING_MOMENTUM, losses)        
        self.compile(losses, optimizer)

        ##--------------------------------------------------------------------------------
        ## get metrics from keras_model.metrics_names and setup callback metrics 
        ##--------------------------------------------------------------------------------
        out_labels = self.get_deduped_metrics_names()
        callback_metrics = out_labels + ['val_' + n for n in out_labels]
        
        print()
        print(' Post-compile out_labels from get_deduped_metrics_names() : ')
        print(' ---------------------------------------------------------- ')
        for i in out_labels:
            print('     -',i)
        print()
        print(' Post-compile Callback metrics monitored by progbar :')
        print(' ----------------------------------------------------')
        for i in callback_metrics:
            print('     -',i)

        print()
        print(' Post-compile Keras metric_names :') 
        print(' ---------------------------------') 
        for idx, i in enumerate(self.keras_model.metrics_names):
            print('     ',idx, '  ', i)
            
        print()
        print(' Post-compile Keras stateful_metric_names :') 
        print(' ------------------------------------------') 
        for idx, i in enumerate(self.keras_model.stateful_metric_names):
            print('     ',idx, '  ', i)

        ## Setup for stateful_metric_indices Validation process 
        ##--------------------------------------------------------------------------------
        stateful_metric_indices = []
        if hasattr(self, 'metrics'):
            for m in self.stateful_metric_functions:
                m.reset_states()
            stateful_metric_indices = [
                i for i, name in enumerate(self.metrics_names)
                if str(name) in self.stateful_metric_names]
        else:
            stateful_metric_indices = []
            
        ##--------------------------------------------------------------------------------
        ## Callbacks
        ##--------------------------------------------------------------------------------
        # call back for model checkpoint was originally (?) loss. chanegd to val_loss (which is default) 2-5-18
        # copied from \keras\engine\training.py
        # def _get_deduped_metrics_names(self):

            
        callbacks_list = [
              keras.callbacks.ProgbarLogger(count_mode='steps',
                                            stateful_metrics=self.keras_model.stateful_metric_names)
            
            , keras.callbacks.BaseLogger(stateful_metrics=self.keras_model.stateful_metric_names)
            
            , keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                          histogram_freq=0,
                                          write_graph=True,
                                          write_images=False, 
                                          write_grads=True,
                                          batch_size=self.config.BATCH_SIZE)
                                          # write_graph=True,

                                          # write_images=True,
                                          # embeddings_freq=0,
                                          # embeddings_layer_names=None,
                                          # embeddings_metadata=None)

            , keras.callbacks.ModelCheckpoint(self.checkpoint_path, 
                                              mode    = 'auto', 
                                              period  = self.config.CHECKPOINT_PERIOD, 
                                              monitor = 'val_loss', 
                                              verbose = 1, 
                                              save_best_only = True, 
                                              save_weights_only=True)
                                            
            , keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                mode     = 'auto', 
                                                factor   = self.config.REDUCE_LR_FACTOR,   
                                                cooldown = self.config.REDUCE_LR_COOLDOWN,
                                                patience = self.config.REDUCE_LR_PATIENCE,
                                                min_delta= self.config.REDUCE_LR_MIN_DELTA,
                                                min_lr   = self.config.MIN_LR, 
                                                verbose  = 1)                                            
                                                
            , keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                mode      = 'auto', 
                                                min_delta = self.config.EARLY_STOP_MIN_DELTA, 

                                                patience  = self.config.EARLY_STOP_PATIENCE, 
                                                verbose   = 1)                                            
            , keras.callbacks.History() 
        ]
         
        
        callbacks =  keras.callbacks.CallbackList(callbacks = callbacks_list)
        callbacks.set_model(self.keras_model)
        callbacks.set_params({
            'batch_size': batch_size,
            'epochs': epochs,
            'steps': steps_per_epoch,
            'verbose': 1 ,
            'do_validation': True,
            'metrics': callback_metrics
        })
        
            # 'samples': num_train_samples,
            # 'verbose': verbose,
            # 'do_validation': do_validation,
            # 'metrics': callback_metrics or [],
        
        log(" ")
        log("Training Start Parameters:")
        log("--------------------------")
        log("Starting at epoch     {} of {} epochs.".format(self.epoch, epochs))
        log("Steps per epochs      {} ".format(steps_per_epoch))
        log("Last epoch completed  {} ".format(self.epoch))
        log("Batch size            {} ".format(batch_size))
        log("Learning Rate         {} ".format(self.config.LEARNING_RATE))
        log("Momentum              {} ".format(self.config.LEARNING_MOMENTUM))
        log("Weight Decay:         {} ".format(self.config.WEIGHT_DECAY       ))
        log("VALIDATION_STEPS      {} ".format(self.config.VALIDATION_STEPS   ))
        log("REDUCE_LR_FACTOR      {} ".format(self.config.REDUCE_LR_FACTOR   ))
        log("REDUCE_LR_COOLDOWN    {} ".format(self.config.REDUCE_LR_COOLDOWN ))
        log("REDUCE_LR_PATIENCE    {} ".format(self.config.REDUCE_LR_PATIENCE ))
        log("MIN_LR                {} ".format(self.config.MIN_LR             ))
        log("EARLY_STOP_PATIENCE   {} ".format(self.config.EARLY_STOP_PATIENCE))        
        log("Checkpoint Path:      {} ".format(self.checkpoint_path))


        ##----------------------------------------------------------------------------------------------
        ## If in debug mode write stdout intercepted IO to output file  
        ##----------------------------------------------------------------------------------------------            
        if self.config.SYSOUT == 'FILE':
            utils.write_sysout(self.log_dir)

        ##--------------------------------------------------------------------------------
        ## Start main training loop
        ##--------------------------------------------------------------------------------
        early_stopping  = False
        val_steps = self.config.VALIDATION_STEPS
        epoch_idx = self.epoch
        callbacks.on_train_begin()

        if epoch_idx >= epochs:
            print('Final epoch {} has already completed - Training will not proceed'.format(epochs))
        else:
        
            while epoch_idx < epochs :
                callbacks.on_epoch_begin(epoch_idx)
                epoch_logs = {}
                
                ##------------------------------------------------------------------------
                ## TRAINING Phase - emulating fit_generator()
                ##------------------------------------------------------------------------
                for steps_index in range(steps_per_epoch):
                    
                    # print(' self.epoch {}   epochs {}  step {} '.format(self.epoch, epochs, steps_index))
                    batch_logs = {}
                    batch_logs['batch'] = steps_index
                    batch_logs['size']  = batch_size    

                    callbacks.on_batch_begin(steps_index, batch_logs)

                    train_batch_x, train_batch_y = next(train_generator)

                    
                    # print('len of train batch x' ,len(train_batch_x))
                    # for idx, i in  enumerate(train_batch_x):
                        # print(idx, 'type: ', type(i), 'shape: ', i.shape)
                    # print('len of train batch y' ,len(train_batch_y))
                    # for idx, i in  enumerate(train_batch_y):
                        # print(idx, 'type: ', type(i), 'shape: ', i.shape)
                    # print(type(output_rois))
                    # for i in model_output:
                        # print( i.shape)       
                        
                    ## Run prediction on MRCNN  
                    try:
                        results = mrcnn_model.keras_model.predict(train_batch_x)
                        fcn_x = [train_batch_x[1]]
                        fcn_x.extend(results[:4])
                    
                    except Exception as e :
                        print('failure on mrcnn predict - epoch {} , image ids: {} '.format(epoch_idx, train_batch_x[1][:,0]))
                        print('Exception information:')
                        print(str(e))
                    
                    # print('size of results : ', len(results))
                    # for idx, i in  enumerate(x):
                        # print(idx, 'type: ', type(i), 'shape: ', i.shape)
                    
                    ## Train on FCN
                    try:
                        outs = self.keras_model.train_on_batch(fcn_x , train_batch_y)                                            
                    except Exception as e :
                        print('failure on fcn train - epoch {} , image ids: {} '.format(epoch_idx, train_batch_x[1][:,0]))
                        print('Exception information:')
                        print(str(e))                
                        
                    # print('size of outputs from train_on_batch : ', len(outs), outs)
                    # for idx, i in  enumerate(outs):
                        # print(idx, 'type: ', type(i), 'shape: ', i.shape)
                        
                    if not isinstance(outs, list):
                        outs = [outs]

                    for l, o in zip(out_labels, outs):
                        # print(' out label: ', l, ' out value: ', o,' shape: ', o.shape)
                        batch_logs[l] = o
    
                    callbacks.on_batch_end(steps_index, batch_logs)

                ##------------------------------------------------------------------------
                ## VALIDATION Phase - emulating evaluate_generator()
                ##------------------------------------------------------------------------
                # print(' Start validation ')
                # print(' ---------------- ')
                # print(' Stateful metric indices:' )
                # pp.pprint(stateful_metric_indices)
                
                
                val_steps_done      = 0
                val_outs_per_batch  = []
                val_batch_sizes     = []
                
                # setup validation progress bar if we wish
                # progbar = Progbar(target=val_steps)

                while val_steps_done < val_steps:
                    # print(' ** Validation step: ', val_steps_done)
                    
                    mrcnn_val_x, mrcnn_val_y = next(val_generator)
                    
                    # print('len of train batch x' ,len(val_x))
                    # for idx, i in  enumerate(val_x):
                        # print(idx, 'type: ', type(i), 'shape: ', i.shape)
                        
                    ## Run prediction on MRCNN  
                    try:
                        val_results = mrcnn_model.keras_model.predict(mrcnn_val_x)
                        fcn_val_x = [mrcnn_val_x[1]]
                        fcn_val_x.extend(val_results[:4])   ## image_meta, pr_hm, pr_hm_scores, gt_hm, gt_hm_scores
                    except Exception as e :
                        print('failure on mrcnn predict (validation)- epoch {} , image ids: {} '.format(epoch_idx, mrcnn_val_x[1][:,0]))
                        print('Exception information:')
                        print(str(e))                

                    # print('    mrcnn_model.predict() size of results : ', len(val_results))
                    # for idx, i in  enumerate(xval_results):
                        # print('    ',idx, 'type: ', type(i), 'shape: ', i.shape)
                                       
                    ## Train on FCN
                    try:
                        outs2 = self.keras_model.test_on_batch( fcn_val_x , mrcnn_val_y)
                        # print('\n valstep {} outs2 len:{} '.format(val_steps_done, len(outs2)))
                        val_outs_per_batch.append(outs2)
                    except Exception as e :
                        print('failure on fcn train (validation)- epoch {} , image ids: {} '.format(epoch_idx, mrcnn_val_x[1][:,0]))                    
                        print('Exception information:')
                        print(str(e))                

                    # print('fcn_model.test_on_batch() size of results : ', len(outs2))
                    # for idx, i in  enumerate(outs2):
                        # print(idx, 'type: ', type(i), 'shape: ', i.shape)
                    
                    if isinstance(fcn_val_x, list):
                        batch_size = fcn_val_x[0].shape[0]
                    elif isinstance(fcn_val_x, dict):
                        batch_size = list(fcn_val_x.values())[0].shape[0]
                    else:
                        batch_size = fcn_val_x.shape[0]
                        
                    if batch_size == 0:
                        raise ValueError('Received an empty batch. '
                                         'Batches should at least contain one item.')
                    # else:
                        # print('batch size:', batch_size)
                        
                    val_steps_done += 1
                    val_batch_sizes.append(batch_size)
                    # print validation progress bar if we wish
                    # progbar.update(val_steps_done)

                ## calculate val_averages after all validations steps complete, which is passed 
                ## back to fit_generator() as val_outs 
                
                # print('    val_batch_sizes            :', type(val_batch_sizes),' len :', len(val_batch_sizes), val_batch_sizes)
                # print('    val_batch_sizes-shape      :', np.asarray(val_batch_sizes).shape)                
                # print('    val_outs_per_batch:        :', type(val_batch_sizes),' len :', len(val_outs_per_batch))
                # print('    val_outs_per_batch - shape :', np.asarray(val_outs_per_batch).shape)
                # for i,j in enumerate(val_outs_per_batch):
                    # print('        batch: ', i, '  ', j)
                
                val_averages = []
                for i in range(len(outs2)):
                    if i not in stateful_metric_indices:
                        tt = [out[i] for out in val_outs_per_batch]
                        # print(' tt type: ',type(tt), tt)
                        # print('val_batch_sizes.shape' , type(val_batch_sizes), len(val_batch_sizes))
                        val_averages.append(
                                np.average([out[i] for out in val_outs_per_batch], axis = 0, weights=val_batch_sizes)
                                           )
                    else:
                        val_averages.append(float(val_outs_per_batch[-1][i]))
                if len(val_averages) == 1:
                    val_averages = val_averages[0]
                print()
                print('val_averages :', val_averages)
                print()
                
                #-- (unsuccessful) attempt to add histogram info to tensoflow summary 
                #--------------------------------------------------------------------
                # print(' Tensordlow histogram attempt')
                # print('-----------------------------')
                # fcn_val_y = self.keras_model.targets
                # val_sample_weight = self.keras_model.sample_weights
                # print(' len(fcn_val_x)  : ',len(fcn_val_x))
                # print(' len(fcn_val_y)  : ',len(fcn_val_y))
                # print(' len(mrcnn_val_y): ',len(mrcnn_val_y))

                # fcn_val_x, fcn_val_y, fcn_val_sample_weights = self.keras_model._standardize_user_data(fcn_val_x, fcn_val_y, val_sample_weight)
                # fcn_val_data = fcn_val_x + fcn_val_y  + fcn_val_sample_weights

                # print(' len(fcn_val_x)             : ',len(fcn_val_x))
                # print(' len(fcn_val_y)             : ',len(fcn_val_y))
                # print(' len(fcn_val_sample_weights): ',len(fcn_val_sample_weights))
                # print(' len(fcn_val_data)          : ',len(fcn_val_data))
                # if self.keras_model.uses_learning_phase and not isinstance(KB.learning_phase(), int):
                    # fcn_val_data += [0.]
                # for cbk in callbacks:
                    # cbk.validation_data = fcn_val_data
                #-------------------------------------------------------------------------------
                
                ##------------------------------------------------------------------------
                ## END OF EPOCH Phase 
                ##------------------------------------------------------------------------
                ## end of evaluate_generator() emulation
                ## val_averages returned back to fit_generator() as val_outs
                ## calculate val_outs after all validations steps complete
                ##------------------------------------------------------------------------
                if not isinstance(val_averages, list):
                    val_averages = [val_averages]
                # Same labels assumed.
                for l, o in zip(out_labels, val_averages):
                    epoch_logs['val_' + l] = o
                    
                #----commented 31-10-18 replaced with above lines -------------------------------------------
                # if not isinstance(outs2, list):
                    # val_outs =  np.average(np.asarray(val_all_outs), weights=val_batch_sizes)
                # else:
                    # averages = []
                    # for i in range(len(outs2)):
                        # averages.append(np.average([out[i] for out in val_all_outs], axis = 0, weights=val_batch_sizes))
                    # val_outs = averages
                # if not isinstance(val_outs, list):
                    # val_outs = [val_outs]
                
                # # Same labels assumed.
                # for l, o in zip(out_labels, val_outs):
                    # # print(' Validations : out label: val_', l, ' out value: ', o)
                    # epoch_logs['val_' + l] = o
                #-------------------------------------------------------------------------------------
                
                # write_log(callback, val_names, logs, batch_no//10)
                # print('\n    validation logs output: ', val_outs)
                
                    
                epoch_logs.update({'lr': KB.eval(self.keras_model.optimizer.lr)})    
                callbacks.on_epoch_end(epoch_idx, epoch_logs)
                epoch_idx += 1
                

                for callback in callbacks:
                    # print(callback)
                    # pp.pprint(dir(callback.model))
                    if hasattr(callback.model, 'stop_training') and (callback.model.stop_training ==True):
                        print(' +++++++++++ ON EPOCH END CALLBACKS TRIGGERED STOP_TRAINING +++++++++++++')
                        print(callback.model, ' triggered stop_training +++++++++++++')
                        early_stopping = True
                        
                if early_stopping:
                    print('{}  Early Stopping triggered on epoch {} of {} epochs'.format(callback, epoch_idx, epochs))
                    break    
                
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
            
            
            
            
    ##---------------------------------------------------------------------------------------------
    ## write_log
    ## copied from github but not used
    ##---------------------------------------------------------------------------------------------
    # def write_log(callback, names, logs, batch_no):
        # for name, value in zip(names, logs):
            # summary = tf.Summary()
            # summary_value = summary.value.add()
            # summary_value.simple_value = value
            # summary_value.tag = name
            # callback.writer.add_summary(summary, batch_no)
            # callback.writer.flush()        

            
                       
"""
    ##---------------------------------------------------------------------------------------------
    ## Learning Rate scheduler copied from Keras-FCN
    ##---------------------------------------------------------------------------------------------
    def lr_scheduler(epoch, mode='power_decay'):
        '''
        Usage:
            
            scheduler = LearningRateScheduler(lr_scheduler)            
                
        Default values:
            lr_base     = 0.01 * (float(batch_size) / 16)
            lr_power    = 0.9

        '''
        lr_base     = 0.01 * (float(self.config.BATCH_SIZE) / 16)
        lr_power    = 0.9

        if mode is 'power_decay':
            # original lr scheduler
            lr = lr_base * ((1 - float(epoch)/epochs) ** lr_power)
        if mode is 'exp_decay':
            # exponential decay
            lr = (float(lr_base) ** float(lr_power)) ** float(epoch+1)
        # adam default lr
        if mode is 'adam':
            lr = 0.001

        if mode is 'progressive_drops':
            # drops as progression proceeds, good for sgd
            if epoch > 0.9 * epochs:
                lr = 0.0001
            elif epoch > 0.75 * epochs:
                lr = 0.001
            elif epoch > 0.5 * epochs:
                lr = 0.01
            else:
                lr = 0.1

        print('lr: %f' % lr)
        return lr

        
"""        
        
        
        
"""        
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

        ##----------------------------------------------------------------------------------------------
        ## Setup optimizaion method 
        ##----------------------------------------------------------------------------------------------            
        optimizer = self.set_optimizer()

        # self.compile(learning_rate, self.config.LEARNING_MOMENTUM, losses)
        self.compile(losses, optimizer)
        
        log("Starting at epoch   {} of {} epochs. LR={}\n".format(self.epoch, epochs, learning_rate))
        log("Steps per epochs    {} ".format(steps_per_epoch))
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
"""


"""        
    ##---------------------------------------------------------------------------------------------
    ## Compile 2
    ## 
    ##    Note: Using l2 regularizers in the model adds a loss for each layer using the regularization
    ##---------------------------------------------------------------------------------------------            
    def compile2(self, loss_functions, optimizer):
        '''
        Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        '''
        assert isinstance(loss_functions, list) , "A loss function must be defined as the objective"
        
        # Optimizer object
        print('\n')
        print('  Compile Model :')
        print(' ----------------')
        print('    losses        : ', loss_functions)
        print('    optimizer     : ', optimizer)
        print('    learning rate : ', self.config.LEARNING_RATE)
        print('    momentum      : ', self.config.LEARNING_MOMENTUM)
        print()
        
        #------------------------------------------------------------------------
        # Add Losses
        # These are the losses aimed for minimization
        # Normally Keras expects the same number of losses as outputs. Since we 
        # are returning more outputs , we go deep and set the losses in the base
        # layers () 
        #------------------------------------------------------------------------    
        # First, clear previously set losses to avoid duplication
        # self.keras_model._losses = []
        # self.keras_model._per_input_losses = {}

        print(' Initial self.keras_model.losses :')
        print(' ---------------------------------')
        print(' losses passed to compile : ', loss_functions)
        print(' self.keras_model.losses  : ')
        for idx, i in enumerate(self.keras_model.losses):
            print('     ',idx, '  ', i)
        
        # print()
        # print(' Add loss_functions to self.keras_model.losses')
        # print(' -------------------------------------')
        
        # for name in loss_functions:
            # layer = self.keras_model.get_layer(name)
            # print(' --  Loss: {}  Related Layer is : {}'.format(name, layer.name))
            # if layer.output in self.keras_model.losses:
                # print('      ',layer.output,' is already in self.keras_model.losses, and wont be added to list')
                # continue
            # print('    >> Add add loss for ', layer.output, ' to list of losses...')
            # self.keras_model.add_loss(tf.reduce_mean(layer.output, keepdims=True))
            
        print()    
        print(' self.keras_model.losses after adding loss_functions passed to compile() : ') 
        print(' ------------------------------------------------------------------------- ') 
        for idx, i in enumerate(self.keras_model.losses):
            print('     ',idx, '  ', i)

        # print()    
        # print(' Keras_model._losses:' ) 
        # print(' --------------------' ) 
        # for idx, i in enumerate(self.keras_model._losses):
            # print('     ',idx, '  ', i)

        print()    
        print(' Keras_model._per_input_losses:')
        print(' ------------------------------')
        for idx, i in enumerate(self.keras_model._per_input_losses):
            print('     ',idx, '  ', i)
        # pp.pprint(self.keras_model._per_input_losses)
            
            
        #------------------------------------------------------------------------    
        # Add L2 Regularization as loss to list of losses
        #------------------------------------------------------------------------    
        # Skip gamma and beta weights of batch normalization layers.
        # reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                      # for w in self.keras_model.trainable_weights
                      # if 'gamma' not in w.name and 'beta' not in w.name]
        # self.keras_model.add_loss(tf.add_n(reg_losses))

        print()    
        print(' Final list of keras_model.losses, after adding L2 regularization as loss to list : ') 
        print(' ---------------------------------------------------------------------------------- ') 
        for idx, i in enumerate(self.keras_model.losses):
            print('     ',idx, '  ', i)

        ##------------------------------------------------------------------------    
        ## Compile
        ##------------------------------------------------------------------------   
        print()
        print(' Compile ')
        print(' --------')
        print (' Length of Keras_Model.outputs:', len(self.keras_model.outputs))
        # self.keras_model.compile(optimizer=optimizer, loss=[None] * len(self.keras_model.outputs))
        self.keras_model.compile(optimizer=optimizer, loss=loss.fcn_heatmap_MSE_loss_graph)

        ##------------------------------------------------------------------------    
        ## Add metrics for losses
        ##------------------------------------------------------------------------    
        print()
        print(' Add Metrics for losses :')
        print(' -------------------------')                                 
        print(' Initial Keras metric_names:', self.keras_model.metrics_names)                                 

        # for name in loss_functions:
            # if name in self.keras_model.metrics_names:
                # print('      ' , name , 'is already in in self.keras_model.metrics_names')
                # continue
            # layer = self.keras_model.get_layer(name)
            # print('    Loss name : {}  Related Layer is : {}'.format(name, layer.name))
            # self.keras_model.metrics_names.append(name)
            # self.keras_model.metrics_tensors.append(tf.reduce_mean(layer.output, keepdims=True))
            # print('    >> Add metric ', name, ' with metric tensor: ', layer.output.name, ' to list of metrics ...')
        
        print()
        print (' Final Keras metric_names :') 
        print (' --------------------------') 
        for idx, i in enumerate(self.keras_model.metrics_names):
            print('     ',idx, '  ', i)

        print()
        print(' self.keras_model.losses after adding losses passed to compile() : ') 
        print(' ----------------------------------------------------------------- ') 
        for idx, i in enumerate(self.keras_model.losses):
            print('     ',idx, '  ', i)

        # print()    
        # print(' Keras_model._losses:' ) 
        # print(' ---------------------' ) 
        # for idx, i in enumerate(self.keras_model._losses):
            # print('     ',idx, '  ', i)

        print()    
        print(' Keras_model._per_input_losses:')
        print(' ------------------------------')
        for idx, i in enumerate(self.keras_model._per_input_losses):
            print('     ',idx, '  ', i)
        print()
        
        return
"""



               
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

"""
        
        
        
##-------------------------------------------------------------------------------------
## detect old 
##-------------------------------------------------------------------------------------        
        
    def detect_old(self, mrcnn_model, images, verbose=0):
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
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"
    
        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)
        if verbose:
            log("molded_images", molded_images)
            log("image_metas"  , image_metas)        # print('>>> model detect()')
        
        ## Run object detection pipeline

        print('call mrcnn predict()')
        mrcnn_detections, rpn_roi_proposals, mrcnn_class, mrcnn_bbox, pr_hm, pr_hm_scores =  \
                  mrcnn_model.keras_model.predict([molded_images, image_metas], verbose=0)

        print('    return from  MRCNN predict()')
        print('    Length of detections          : ', len(mrcnn_detections), mrcnn_detections.shape)
        print('    Length of rpn_roi_proposals   : ', len(rpn_roi_proposals), rpn_roi_proposals.shape)
        print('    Length of mrcnn_class         : ', len(mrcnn_class), mrcnn_class.shape)
        print('    Length of mrcnn_bbox          : ', len(mrcnn_bbox ), mrcnn_bbox.shape)
        print('    Length of pr_hm               : ', len(pr_hm), pr_hm.shape)
        print('    Length of pr_hm_scores        : ', len(pr_hm_scores), pr_hm_scores.shape)


        print('call fcn predict()')
        fcn_hm, fcn_sm, fcn_hm_scores = self.keras_model.predict([pr_hm, pr_hm_scores], verbose = 1)

        print('    return from  FCN predict()')
        print('    Length of fcn_heatmaps : ', len(fcn_hm), fcn_hm.shape)
        print('    Length of fcn_softmax  : ', len(fcn_sm), fcn_sm.shape)
        print('    Length of fcn_hm_scores: ', len(fcn_hm_scores), fcn_hm_scores.shape)
         
        ## Process detections        
        
        results = []

        
        ## build results list. Takes images, mrcnn_detections, pr_hm, pr_hm_scores and fcn outputs
        
        for i, image in enumerate(images):
            print(' Unmold image ', i, 'image_shape: ', image.shape,  '  windows[]:', windows[i])
            mrcnn_rois, mrcnn_class_ids, mrcnn_scores, mrcnn_molded_rois = \
                        mrcnn_model.unmold_detections(mrcnn_detections[i], image.shape, windows[i])    
          # final_rois, final_class_ids, final_scores, molded_rois = self.unmold_detections(detections[i], 
                                                                               # image.shape  ,
                                                                               # windows[i])    
            print(pr_hm_scores[i].shape)
            print(fcn_hm_scores[i].shape)
            print('mrcnn_class_ids: ', mrcnn_class_ids)
            ## pr_hm_scores is by image/class/bounding box
            ## fcn_hm_scores is by image/class/bounding box
            
            ## reshape pr_hm_scores from per_class to per_image tensor
            ## Convert pr_scores_by_image bboxes from NN coordinates to image coordinates
            #     pr_scores_by_image = utils.byclass_to_byimage_np(pr_hm_scores[i], 6)
            #     print(' pr_scores_by_class shape',pr_scores_by_image.shape)
            #     pr_boxes_adj = utils.boxes_to_image_domain(pr_scores_by_image[:,:4],image_metas[i])
            #     pr_scores_by_image = np.hstack((pr_boxes_adj, pr_scores_by_image[:,4:]))
            #     print(' pr_boxes_adj')
            #     print(pr_boxes_adj)
            #     print(' pr_scores_by_image')
            #     print(pr_scores_by_image)

            ## pr_hm_scores is by class  
            ## Convert pr_hm_scores bboxes from NN coordinates to image coordinates
            pr_boxes_adj = utils.boxes_to_image_domain(pr_hm_scores[i,:,:,:4],image_metas[i])
            pr_scores_by_class= np.dstack((pr_boxes_adj, pr_hm_scores[i,:,:,4:]))
            print(' pr_scores_by_class shape', pr_scores_by_class.shape)
            pr_scores_by_image = utils.byclass_to_byimage_np(pr_scores_by_class, 6)

            #     print(' pr_boxes_adj_2')
            #     print(pr_boxes_adj_2[mrcnn_class_ids, :10])
            #     print(' pr_scores_by_class')
            #     print(pr_scores_by_class[mrcnn_class_ids, :10])

            ## fcn_hm_scores is by class  
            ## Convert pr_hm_scores bboxes from NN coordinates to image coordinates
            fcn_boxes_adj = utils.boxes_to_image_domain(fcn_hm_scores[i,:,:,:4],image_metas[i])
            fcn_scores_by_class= np.dstack((pr_boxes_adj, fcn_hm_scores[i,:,:,4:]))
            print(' fcn_scores_by_class shape', fcn_scores_by_class.shape)
            fcn_scores_by_image = utils.byclass_to_byimage_np(fcn_scores_by_class, 6)
            #     print(' fcn_boxes_adj')
            #     print(fcn_boxes_adj[mrcnn_class_ids, :10])
            #     print(' fcn_scores_by_class')
            #     print(fcn_scores_by_class[mrcnn_class_ids, :10])  

            #     fcn_scores_adj,fcn_rois, fcn_class_ids, mrcnn_scores_norm, fcn_scores =  \
            #                                    self.unmold_detections(fcn_model, fcn_hm_scores[i], image.shape, windows[i]) 
            #     print(fcn_scores_adj.shape, fcn_rois.shape, fcn_class_ids.shape, mrcnn_scores_norm.shape, fcn_scores.shape)
            
            results.append({
                "image"        : images[i],
                "molded_image" : molded_images[i],                 
                "image_meta"   : image_metas[i],
                
                "rois"         : mrcnn_rois,
                "molded_rois"  : mrcnn_molded_rois,
                "class_ids"    : mrcnn_class_ids,
                "mrcnn_scores" : mrcnn_scores, 

                "pr_scores"    : pr_scores_by_image,
                "pr_scores_by_class"    : pr_scores_by_class,
                "pr_hm"        : pr_hm[i],

                "fcn_scores"   : fcn_scores_by_image,
                "fcn_scores_by_class"   : fcn_scores_by_class,
                "fcn_hm"       : fcn_hm[i],
                "fcn_sm"       : fcn_sm[i]
            })
                    
        return results 
"""