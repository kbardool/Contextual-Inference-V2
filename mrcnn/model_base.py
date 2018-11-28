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

# from   mrcnn.fpn_layers       import fpn_graph, fpn_classifier_graph, fpn_mask_graph

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

    def __init__(self, mode, config):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        print('>>> Initialize ModelBase model ')

        self.mode      = mode
        self.config    = config
        self.model_dir = config.TRAINING_PATH
        print('   Mode      : ', self.mode)
        print('   Model dir : ', self.model_dir)

        print('>>> ModelBase initialiation complete')

    
    ##------------------------------------------------------------------------------------    
    ## LOAD MODEL
    ##------------------------------------------------------------------------------------                
    def load_model_weights(self,init_with = None, exclude = None, new_folder = False, verbose = 0):
        '''
        methods to load weights
        1 - look for a specific weights file 
            Load trained weights (fill in path to trained weights here)
        2 - look for last checkpoint file in a specific folder (not working correctly)
        3 - Use init_with keyword
        -- Which weights to start with?
        '''    

        # Display layers we intent to exclude from loading process
        print('-----------------------------------------------')
        print(' Load Model with init parm: [',init_with,']')
        # print(' find last chkpt :', model.find_last())
        if exclude is not None:
            print(' Exclude layers: ')
            for la in exclude: 
                print('    - ',la)
        print('-----------------------------------------------')
       

        if init_with == "imagenet":
            # loc=model.load_weights(model.get_imagenet_weights(), by_name=True)
            loc=self.load_weights(self.config.RESNET_MODEL_PATH, by_name=True, exclude = exclude, verbose = verbose)
            
        elif init_with == "init":
            print(' ---> init :', self.config.VGG16_MODEL_PATH)
            # Load weights trained on MS COCO, but skip layers that 
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            
            loc=self.load_weights(self.config.VGG16_MODEL_PATH, by_name=True, exclude = exclude, verbose = verbose)
            
        elif init_with == "vgg16":
            print(' ---> vgg16 :', self.config.VGG16_MODEL_PATH)
            loc=self.load_weights(self.config.VGG16_MODEL_PATH, by_name=True, exclude = exclude, verbose = verbose)
            
        elif init_with == "coco":
            print(' ---> coco :', self.config.COCO_MODEL_PATH)
            # use pretrained coco weights file:  "mask_rcnn_coco.h5"
            # Load weights trained on MS COCO, but skip layers that 
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            # loc=self.load_weights(self.config.COCO_MODEL_PATH, by_name=True, exclude = exclude)
            loc=self.load_weights(self.config.COCO_MODEL_PATH, by_name=True, exclude = exclude, verbose = verbose)
            # exclude=["mrcnn_class_logits"])  # ,"mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

        elif init_with == "last":
            print(' ---> last')
            # Load the last model you trained and continue training, placing checkpouints in same folder
            # last_file = self.find_last()[1]
            # print('   Last file is :', last_file)
            loc= self.load_weights(self.find_last()[1], by_name=True, verbose = verbose)
        else:
            assert init_with != "", "Provide path to trained weights"
            print(" ---> Explicit weight file")
            loc = self.load_weights(init_with, by_name=True, exclude = exclude, new_folder= new_folder, verbose = verbose)  


        print('==========================================')
        print( self.config.NAME.upper(), " MODEL Load weight file COMPLETE ")
        print('==========================================')
        return     

        
    ##----------------------------------------------------------------------------------------------
    ## Load weights file
    ##----------------------------------------------------------------------------------------------                    
    def load_weights(self, filepath, by_name=False, exclude=None, new_folder = False, verbose = 0):
        '''
        Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        '''
        import h5py

        # from keras.engine import topology
        log('>>> load_weights() from : {}'.format(filepath))
        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        import inspect
        f = h5py.File(filepath, mode='r')

        # pp.pprint([i for i in dir(f) if not inspect.ismethod(i)])
        
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            print('im here')
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
        # for idx,layer in enumerate(layers): 
            # print('>layer {} : name : {:40s}  type: {}'.format(idx,layer.name,layer))
            
        # Exclude some layers
        if exclude:
            layers = list(filter(lambda l: l.name not in exclude, layers))
            
            
        # print('--------------------------------------' )       
        # print(' layers to load (not in exclude list) ' )
        # print('--------------------------------------' )
        # for idx,layer in enumerate(layers):
            # print('>layer {} : name : {:40s}  type: {}'.format(idx,layer.name,layer))
        # print('\n\n')
        
        if by_name:
            # topology.load_weights_from_hdf5_group_by_name(f, layers)
            utils.load_weights_from_hdf5_group_by_name(f, layers, verbose = verbose)
        else:
            # topology.load_weights_from_hdf5_group(f, layers)
            utils.load_weights_from_hdf5_group(f, layers, verbose = verbose)
            
        if hasattr(f, 'close'):
            f.close()
            
        # Update the log directory
        print('    Weights file loaded: {} '.format(filepath))        
        print('    Weights file loaded: {} '.format(filepath), file = sys.__stdout__)

        # if self.mode == 'training':
            # self.set_log_dir(filepath, new_folder)
            
        # print(" load_weights() :  MODEL Load weight file COMPLETE    ")

        return(filepath)


    ##----------------------------------------------------------------------------------------------
    ##  set checkpoint directory 
    ##----------------------------------------------------------------------------------------------                    
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
        print('>>> set_log_dir(): model_path: ', model_path)
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
            print('    set_log_dir(): model_path has been provided : {} '.format(model_path))
            # print('    set_log_dir: model_path (input) is : {}  '.format(model_path))        

            ## serch for something like yyyymmddThhmm/
            # regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/fcn\w+(\d{4})\.h5"
            ## trying this 27-10-18
            regex = ".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/\w+(\d{4}).h5"
            regex_match  = re.match(regex, model_path)
            
            if regex_match:             
                now = datetime.datetime(int(regex_match.group(1)), int(regex_match.group(2)), int(regex_match.group(3)),
                                        int(regex_match.group(4)), int(regex_match.group(5)))
                last_checkpoint_epoch = int(regex_match.group(6)) + 1
                if last_checkpoint_epoch > 0 and  self.config.LAST_EPOCH_RAN > last_checkpoint_epoch: 
                    self.epoch = self.config.LAST_EPOCH_RAN
                else :
                    self.epoch = last_checkpoint_epoch
                print('    set_log_dir(): File match found')
                print('    set_log_dir(): self.epoch set to {}  (Next epoch to run)'.format(self.epoch))
                print('    set_log_dir(): tensorboard path: {}'.format(self.tb_dir))
        else:
            print('    set_log_dir(): model_path has NOT been provided : {} '.format(model_path))
            print('                  NewFolder: {}  config.NEW_LOG_FOLDER: {} '.format(new_folder, self.config.NEW_LOG_FOLDER))
            now = datetime.datetime.now()
        
        # Set directory for training logs
        # if new_folder = True or appropriate checkpoint filename was not found, generate new folder
        # if new_folder or self.config.NEW_LOG_FOLDER:
            # print('NewFolder: {}  config.NEW_LOG_FOLDER: {} '.format(new_folder, self.config.NEW_LOG_FOLDER))
            # now = datetime.datetime.now()

        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(self.config.NAME.lower(), now))

        ##--------------------------------------------------------------------------------
        ## Create checkpoint folder if it doesn't exists
        ##--------------------------------------------------------------------------------
        from tensorflow.python.platform import gfile
        if not gfile.IsDirectory(self.log_dir):
            print('    set_log_dir(): NEW folder     : {}'.format(self.log_dir), file = sys.__stdout__)
            gfile.MakeDirs(self.log_dir)
        else:
            print('    set_log_dir(): EXISTING folder: {}'.format(self.log_dir), file = sys.__stdout__)

            
        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "{}_*epoch*.h5".format(self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")
        
        log('    set_log_dir(): weight file template (self.checkpoint_path): {} '.format(self.checkpoint_path))
        log('    set_log_dir(): weight file dir      (self.log_dir)        : {} '.format(self.log_dir))
        log('    set_log_dir(): Last completed epoch (self.epoch)          : {} '.format(self.epoch))

        return
        

        
    ##----------------------------------------------------------------------------------------------
    ## Search for last checkpoint folder and weight file
    ##----------------------------------------------------------------------------------------------                    
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
        dir_name, checkpoint = None, None    
        print('>>> find_last checkpoint in : ', self.model_dir)
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = list(filter(lambda f: f.startswith(key), dir_names))
        print('    Dir starting with ' , key, ' :', dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        
        
        ## Loop over folders to find most recent foder with a valid weights file 
        for search_dir in dir_names[-1::-1]:
            dir_name = os.path.join(self.model_dir, search_dir)
            # Find the last checkpoint in this dir
            
            checkpoints = next(os.walk(dir_name))[2]
            checkpoints = filter(lambda f: f.startswith(key), checkpoints)

            checkpoints = sorted(checkpoints)
            print('    Folder: ' ,dir_name)
            print('    Checkpoints: ', checkpoints)
            if not checkpoints:
                continue
                # return dir_name, None
            checkpoint = os.path.join(dir_name, checkpoints[-1])
            break
                
        # old method
        # dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        # checkpoints = next(os.walk(dir_name))[2]
        # checkpoints = filter(lambda f: f.startswith(key), checkpoints)
        # checkpoints = sorted(checkpoints)
        # if not checkpoints:
            # return dir_name, None
        # checkpoint = os.path.join(dir_name, checkpoints[-1])
        
        log("    find_last():   dir_name: {}".format(dir_name))
        log("    find_last(): checkpoint: {}".format(checkpoint))

        return dir_name, checkpoint

        
        
    ##----------------------------------------------------------------------------------------------
    ##
    ##----------------------------------------------------------------------------------------------                    
    def save_model(self, filepath = None, filename = None, by_name=False, exclude=None):
        '''
        Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        '''
        print('>>> save_model() -- Weights only')
        if os.path.splitext(filename)[1] != '.h5':
            filename += '.h5'
        
        if filepath is None:
            full_filepath = os.path.join(self.log_dir, filename)
        else:
            full_filepath = os.path.join(self.log_dir, filename)
            
        log('    save model to  {}'.format(full_filepath))
        self.keras_model.save_weights(full_filepath, overwrite=True)
        
        # Following doesnt' work - some objects are not JSON serializable
        # self.keras_model.save_model(model, filepath, overwrite=True, include_optimizer=True):

        # Following doesnt' work - some objects are not JSON serializable
        # model_json = self.keras_model.to_json()
        # with open(full_filepath , 'w') as f:
            # json.dump(model_json, full_filepath)               
            # if hasattr(f, 'close'):
                # f.close()
                # print('file closed')
        print('    save_weights: save directory is  : {}'.format(filepath))
        print('    save model weights complete')        
        return(full_filepath)

        
    ##----------------------------------------------------------------------------------------------
    ##
    ##----------------------------------------------------------------------------------------------                    
    def get_imagenet_weights(self):
        '''
        Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        '''
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/'\
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_path

        
    ##----------------------------------------------------------------------------------------------
    ##
    ##----------------------------------------------------------------------------------------------                    
    def ancestor(self, tensor, name, checked=None):
        '''Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        '''
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

        
    ##----------------------------------------------------------------------------------------------
    ##
    ##----------------------------------------------------------------------------------------------                    
    def run_graph(self, images, outputs):
        '''Runs a sub-set of the computation graph that computes the given
        outputs.

        outputs: List of tuples (name, tensor) to compute. The tensors are
            symbolic TensorFlow tensors and the names are for easy tracking.

        Returns an ordered dict of results. Keys are the names received in the
        input and values are Numpy arrays.
        '''
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

      
    ##----------------------------------------------------------------------------------------------
    ##
    ##----------------------------------------------------------------------------------------------                    
    def find_trainable_layer(self, layer):
        '''If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        '''
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    ##----------------------------------------------------------------------------------------------
    ## Get Trainable Layers    
    ##----------------------------------------------------------------------------------------------                    
    def get_trainable_layers(self):
        '''Returns a list of layers that have weights.'''
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


    ##----------------------------------------------------------------------------------------------
    ## Set Trainable Layers    
    ##----------------------------------------------------------------------------------------------            
    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
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
        
        
    ##----------------------------------------------------------------------------------------------
    ## Setup optimizaion method 
    ##----------------------------------------------------------------------------------------------            
    def set_optimizer(self):
        print('    learning rate : ', self.config.LEARNING_RATE)
        print('    momentum      : ', self.config.LEARNING_MOMENTUM)
        print()

        opt = self.config.OPTIMIZER
        if   opt == 'ADAGRAD':
            optimizer = keras.optimizers.Adagrad(lr=self.config.LEARNING_RATE, epsilon=None, decay=0.01)                                 
        elif opt == 'SGD':
            optimizer = keras.optimizers.SGD(lr=self.config.LEARNING_RATE, 
                                             momentum=self.config.LEARNING_MOMENTUM, clipnorm=5.0)
        elif opt == 'RMSPROP':                                 
            optimizer = keras.optimizers.RMSprop(lr=self.config.LEARNING_RATE, rho=0.9, epsilon=None, decay=0.0)
        elif opt == 'ADAM':
            optimizer = keras.optimizers.Adam(lr=self.config.LEARNING_RATE, 
                                              beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        elif opt == 'NADAM':
            optimizer = keras.optimizers.Nadam(lr=self.config.LEARNING_RATE,
                                              beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        else:
            print('ERROR: Invalid optimizer specified:',opt)
            if debug:
                write_stdout(self.log_dir, '_sysout', sys.stdout )        
                sys.stdout = sys.__stdout__
            print('\n  Run information written to ', self.log_dir+'_sysout.out')
            print('  ERROR: Invalid optimizer specified:',opt)
            sys.exit('  Execution Terminated')

        # optimizer= tf.train.GradientDescentOptimizer(learning_rate, momentum)
        # optimizer= tf.train.GradientDescentOptimizer(learning_rate, momentum)
        # optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, clipnorm=5.0)
        # optimizer = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
        # optimizer = keras.optimizers.Adagrad(lr=learning_rate, epsilon=None, decay=0.01)
        # optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        # optimizer = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

        
        return optimizer
        
    ##----------------------------------------------------------------------------------------------
    ## Compile Model 
    ##----------------------------------------------------------------------------------------------
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
        
        
    ##----------------------------------------------------------------------------------------------
    ## 
    ##----------------------------------------------------------------------------------------------
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

    ##----------------------------------------------------------------------------------------------
    ## 
    ##----------------------------------------------------------------------------------------------
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
        
    ##----------------------------------------------------------------------------------------------
    ## 
    ##----------------------------------------------------------------------------------------------        
    def get_layer_outputs(self, model_input, requested_layers = None , verbose = True, training_flag = True):
        ''' get model layer outputs using a list of layer indices 
        '''
        # _my_input = model_input + [training_flag]
        if verbose: 
            print('/* Inputs */')
            for i, (input_layer,input) in enumerate(zip(self.keras_model.input, model_input)):
                print('Input {:2d}:  ({:40s}) \t  Input shape: {}'.format(i, input_layer.name, input.shape))
        
        if requested_layers is not None:
            requested_layers_tensors = [self.keras_model.outputs[x] for x in requested_layers]    
        else:
            requested_layers = range(len(self.keras_model.outputs))
            requested_layers_tensors = [x for x in self.keras_model.outputs]    

        if verbose:
            print('\nRequested layers:')
            print('-----------------')
            for i,j in zip(requested_layers, requested_layers_tensors):
                print('Layer:  ',i, '   ', j.name, '   ', j.shape)
        
        get_output = KB.function(self.keras_model.input, requested_layers_tensors)   
        results = get_output(model_input)                    
        
        if verbose:
            print('\n/* Outputs */')    
            for line, (i, output_layer, output) in  enumerate(zip (requested_layers, requested_layers_tensors , results)):
                print('Output idx: {:2d}    Layer: {:2d}: ({:40s}) \t  Output shape: {}'.format(line, i, output_layer.name, output.shape))
            print('\nNumber of layers generated: ', len(results), '\n')

            for line, (i, output_layer, output) in  enumerate(zip (requested_layers, requested_layers_tensors , results)):
                m = re.fullmatch(r"^.*/(.*):.*$", output_layer.name )
                varname  = m.group(1) if m else "<varname>"
                print('{:25s} = model_output[{:d}]          # layer: {:2d}   shape: {}' .format(varname, line, i, output.shape ))
        return results
    
    def get_layer_output_1(self, model_input, requested_layers, training_flag = True, verbose = True):
        ''' get model layer outputs using a list of layer indices 
        '''
        # _my_input = model_input + [training_flag]
        if verbose: 
            print('/* Inputs */')
            for i, (input_layer,input) in enumerate(zip(self.input, model_input)):
                print('Input {:2d}:  ({:40s}) \t  Input shape: {}'.format(i, input_layer.name, input.shape))
                
        requested_layers_tensors = [self.outputs[x] for x in requested_layers]
        
        get_output = KB.function(self.input, requested_layers_tensors)   
        results = get_output(model_input)                    
        
        if verbose:
            print('\n/* Outputs */')    
            for line, (i, output_layer, output) in  enumerate(zip (requested_layers, requested_layers_tensors , results)):
                print('Output idx: {:2d}    Layer: {:2d}: ({:40s}) \t  Output shape: {}'.format(line, i, output_layer.name, output.shape))
            print('\nNumber of layers generated: ', len(results), '\n')

            for line, (i, output_layer, output) in  enumerate(zip (requested_layers, requested_layers_tensors , results)):
                m = re.fullmatch(r"^.*/(.*):.*$", output_layer.name )
                varname  = m.group(1) if m else "<varname>"
                print('{:25s} = model_output[{:d}]          # layer: {:2d}   shape: {}' .format(varname, line, i, output.shape ))
        return results

        
        
    def get_layer_output_2(self, model_input, training_flag = True, verbose = True):
        if verbose: 
            print('/* Inputs */')
            for i, (input_layer,input) in enumerate(zip(self.input, model_input)):
                print('Input {:2d}:  ({:40s}) \t  Input shape: {}'.format(i, input_layer.name, input.shape))

        get_output = KB.function(self.input , self.outputs)
        results = get_output(model_input)                  
        
        if verbose:
            print('\n/* Outputs */')    
            for line, ( output_layer, output) in  enumerate(zip (self.outputs, results)):
                print('Output idx: {:2d}    Layer: {:2d}: ({:40s}) \t  Output shape: {}'.format(line, line, output_layer.name, output.shape))
            print('\nNumber of layers generated: ', len(results), '\n')

            for line, ( output_layer, output) in  enumerate(zip (self.outputs, results)):
                m = re.fullmatch(r"^.*/(.*):.*$", output_layer.name )
                varname  = m.group(1) if m else "<varname>"
                print('{:25s} = model_output[{:d}]          # layer: {:2d}   shape: {}' .format(varname, line, line, output.shape ))            
        return results    