"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

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
from   collections import OrderedDict
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

from   mrcnn.fcn_layer         import fcn_graph
from   mrcnn.fcn_scoring_layer import FCNScoringLayer
from   mrcnn.detect_layer      import DetectionLayer  
from   mrcnn.detect_tgt_layer_mod import DetectionTargetLayer_mod

from   mrcnn.fpn_layers       import fpn_graph, fpn_classifier_graph, fpn_mask_graph
from   mrcnn.callbacks        import MyCallback
from   mrcnn.batchnorm_layer  import BatchNorm

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3.0")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')
pp = pprint.PrettyPrinter(indent=4, width=100)
tf.get_variable_scope().reuse_variables()


############################################################
##  ModelBase Class
############################################################
class ModelBase():
    """Encapsulates the genral model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        print('>>> Initialize ModelBase model ')

        # self.mode      = mode
        # self.config    = config
        # self.model_dir = model_dir
        # self.set_log_dir()
        # # Pre-defined layer regular expressions
        # self.layer_regex = {
            # # ResNet from a specific stage and up
            # "res3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)",
            # "res4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)",
            # "res5+": r"(res5.*)|(bn5.*)",

            # # fcn only 
            # "fcn" : r"(fcn\_.*)",
            # # fpn
            # "fpn" : r"(fpn\_.*)",
            # # rpn
            # "rpn" : r"(rpn\_.*)",
            # # rpn
            # "mrcnn" : r"(mrcnn\_.*)",

            # # all layers but the backbone
            # "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # # all layers but the backbone
            # "allheads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(fcn\_.*)",
          
            # # From a specific Resnet stage and up
            # "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # # All layers
            # "all": ".*",
        # }
        

        # self.keras_model = self.build(mode=mode, config=config, FCN_layers = FCN_layers)

        print('>>> ModelBase initialiation complete')

    
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
            print(" ---> Explicit weight file")
            loc = self.load_weights(init_with, by_name=True, exclude = exclude, new_folder= new_folder)    
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
        log('   >>> load_weights() from : {}'.format(filepath))
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
            
        print('   --------------------------------------' )       
        print('    layers to load (not in exclude list) ' )
        print('   --------------------------------------' )
        for idx,layer in enumerate(layers):
            print('    >layer {} : name : {:40s}  type: {}'.format(idx,layer.name,layer))
        print('\n\n')
            
        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()
        
        # Update the log directory
        self.set_log_dir(filepath, new_folder)
        print('   Load weights complete : ',filepath)        
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
        if new_folder or self.config.NEW_LOG_FOLDER:
            now = datetime.datetime.now()

        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "fcn_{}_*epoch*.h5".format(self.config.NAME.lower()))
            
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")
        
        log('  set_log_dir(): self.Checkpoint_path: {} '.format(self.checkpoint_path))
        log('  set_log_dir(): self.log_dir        : {} '.format(self.log_dir))
        log('  set_log_dir(): Last completed epoch (self.epoch): {} '.format(self.epoch))


        
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