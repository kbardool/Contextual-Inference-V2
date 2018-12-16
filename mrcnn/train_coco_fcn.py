# coding: utf-8
##-------------------------------------------------------------------------------------------
##
## Separated MRCNN-FCN Pipeline (import model_mrcnn)
## Train FCN head only
##
##  Pass predicitions from MRCNN to use as training data for FCN
##-------------------------------------------------------------------------------------------
import os, sys, math, io, time, gc, platform, pprint
import numpy as np
import tensorflow as tf
import keras
import keras.backend as KB
sys.path.append('../')

import mrcnn.model_mrcnn  as mrcnn_modellib
import mrcnn.model_fcn    as fcn_modellib
import mrcnn.visualize    as visualize

from datetime           import datetime   
from mrcnn.config       import Config
from mrcnn.dataset      import Dataset 
from mrcnn.utils        import log, stack_tensors, stack_tensors_3d, write_stdout
from mrcnn.datagen      import data_generator, load_image_gt
from mrcnn.coco         import CocoDataset, CocoConfig, CocoInferenceConfig, evaluate_coco, build_coco_results
from mrcnn.prep_notebook import mrcnn_coco_train, prep_coco_dataset
from mrcnn.utils        import command_line_parser, display_input_parms, Paths
                                
pp = pprint.PrettyPrinter(indent=2, width=100)
np.set_printoptions(linewidth=100,precision=4,threshold=1000, suppress = True)
os_platform = platform.system()

start_time = datetime.now().strftime("%m-%d-%Y @ %H:%M:%S")
print()
print('--> Execution started at:', start_time)
print("    Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))

##------------------------------------------------------------------------------------
## Parse command line arguments
##------------------------------------------------------------------------------------
parser = command_line_parser()
args = parser.parse_args()

##----------------------------------------------------------------------------------------------
## if debug is true set stdout destination to stringIO
##----------------------------------------------------------------------------------------------            
display_input_parms(args)

if args.sysout == 'FILE':
    print('    Output is written to file....')
    sys.stdout = io.StringIO()
    print()
    print('--> Execution started at:', start_time)
    print("    Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))
    display_input_parms(args)
    
##------------------------------------------------------------------------------------
## setup project directories
##   DIR_ROOT         : Root directory of the project 
##   MODEL_DIR        : Directory to save logs and trained model
##   COCO_MODEL_PATH  : Path to COCO trained weights
##---------------------------------------------------------------------------------
paths = Paths( mrcnn_training_folder = args.mrcnn_logs_dir, fcn_training_folder =  args.fcn_logs_dir)
# paths.display()

##------------------------------------------------------------------------------------
## Build configuration object 
##------------------------------------------------------------------------------------
mrcnn_config                      = CocoConfig()
mrcnn_config.NAME                 = 'mrcnn'              
mrcnn_config.TRAINING_PATH        = paths.MRCNN_TRAINING_PATH
mrcnn_config.COCO_DATASET_PATH    = paths.COCO_DATASET_PATH 
mrcnn_config.COCO_MODEL_PATH      = paths.COCO_MODEL_PATH   
mrcnn_config.RESNET_MODEL_PATH    = paths.RESNET_MODEL_PATH 
mrcnn_config.VGG16_MODEL_PATH     = paths.VGG16_MODEL_PATH  
mrcnn_config.COCO_CLASSES         = None 
mrcnn_config.DETECTION_PER_CLASS  = 200
mrcnn_config.HEATMAP_SCALE_FACTOR = int(args.scale_factor)

mrcnn_config.BATCH_SIZE           = int(args.batch_size)                  # Batch size is 2 (# GPUs * images/GPU).
mrcnn_config.IMAGES_PER_GPU       = int(args.batch_size)                  # Must match BATCH_SIZE                                  
mrcnn_config.STEPS_PER_EPOCH      = int(args.steps_in_epoch)
mrcnn_config.LEARNING_RATE        = float(args.lr)
mrcnn_config.EPOCHS_TO_RUN        = int(args.epochs)
mrcnn_config.FCN_INPUT_SHAPE      = mrcnn_config.IMAGE_SHAPE[0:2]
mrcnn_config.LAST_EPOCH_RAN       = int(args.last_epoch)
mrcnn_config.VERBOSE              = 1
mrcnn_config.NEW_LOG_FOLDER       = False
mrcnn_config.SYSOUT               = args.sysout
mrcnn_config.display() 

##------------------------------------------------------------------------------------
## Build Mask RCNN Model in TRAINFCN mode
##------------------------------------------------------------------------------------
try :
    del mrcnn_model
    print('delete model is successful')
    gc.collect()
except: 
    pass
KB.clear_session()
mrcnn_model = mrcnn_modellib.MaskRCNN(mode='trainfcn', config=mrcnn_config)


##------------------------------------------------------------------------------------
## Display model configuration information
##------------------------------------------------------------------------------------
paths.display()
mrcnn_config.display()  
mrcnn_model.display_layer_info()

##------------------------------------------------------------------------------------
## Build configuration for FCN model
##------------------------------------------------------------------------------------
fcn_config = CocoConfig()
fcn_config.NAME                 = 'fcn'              
fcn_config.TRAINING_PATH        = paths.FCN_TRAINING_PATH
fcn_config.VGG16_MODEL_PATH     = paths.FCN_VGG16_MODEL_PATH
fcn_config.FCN_INPUT_SHAPE      = mrcnn_config.IMAGE_SHAPE[0:2] // mrcnn_config.HEATMAP_SCALE_FACTOR 

fcn_config.BATCH_SIZE           = int(args.batch_size)                  # Batch size is 2 (# GPUs * images/GPU).
fcn_config.IMAGES_PER_GPU       = int(args.batch_size)                  # Must match BATCH_SIZE
fcn_config.EPOCHS_TO_RUN        = int(args.epochs)
fcn_config.STEPS_PER_EPOCH      = int(args.steps_in_epoch)
fcn_config.LEARNING_RATE        = float(args.lr)
fcn_config.LAST_EPOCH_RAN       = int(args.last_epoch)
fcn_config.VALIDATION_STEPS     = int(args.val_steps)
fcn_config.TRAINING_LAYERS      = args.fcn_layers

fcn_config.WEIGHT_DECAY         = 2.0e-4     ## FCN Weight decays are 5.0e-4 or 2.0e-4
fcn_config.BATCH_MOMENTUM       = 0.9

fcn_config.REDUCE_LR_FACTOR     = 0.5
fcn_config.REDUCE_LR_COOLDOWN   = 15
fcn_config.REDUCE_LR_PATIENCE   = 50
fcn_config.REDUCE_LR_MIN_DELTA  = 1e-6

fcn_config.EARLY_STOP_PATIENCE  = 150
fcn_config.EARLY_STOP_MIN_DELTA = 1.0e-7

fcn_config.MIN_LR               = 1.0e-10
fcn_config.CHECKPOINT_PERIOD    = 1
fcn_config.VERBOSE              = mrcnn_config.VERBOSE            
fcn_config.NEW_LOG_FOLDER       = args.new_log_folder
fcn_config.OPTIMIZER            = args.opt.upper()
fcn_config.SYSOUT               = args.sysout

##------------------------------------------------------------------------------------
## Build FCN Model in Training Mode
##------------------------------------------------------------------------------------
try :
    del fcn_model
    gc.collect()
except: 
    pass    
fcn_model = fcn_modellib.FCN(mode="training", arch = args.fcn_arch, config=fcn_config)


##------------------------------------------------------------------------------------
## Display model configuration information
##------------------------------------------------------------------------------------
paths.display()
fcn_config.display()  
fcn_model.display_layer_info()

##------------------------------------------------------------------------------------
## Load MRCNN Model Weight file
##------------------------------------------------------------------------------------
exclude_list = []
mrcnn_model.load_model_weights(init_with = args.mrcnn_model, exclude = exclude_list)   

##------------------------------------------------------------------------------------
## Load FCN Model weights  
##------------------------------------------------------------------------------------
if args.fcn_model != 'init':
    fcn_model.load_model_weights(init_with = args.fcn_model, verbose = 1)
else:
    print(' FCN Training starting from randomly initialized weights ...')

##------------------------------------------------------------------------------------
## Build & Load Training and Validation datasets
##------------------------------------------------------------------------------------
dataset_train = prep_coco_dataset(["train","val35k"], mrcnn_config, load_coco_classes =args.coco_classes)
dataset_val   = prep_coco_dataset(["minival"]       , mrcnn_config, load_coco_classes =args.coco_classes)

dataset_train.display_active_class_info()
dataset_val.display_active_class_info()
  
##----------------------------------------------------------------------------------------------
## Train the FCN only 
## Passing layers="heads" freezes all layers except the head
## layers. You can also pass a regular expression to select
## which layers to train by name pattern.
##----------------------------------------------------------------------------------------------            
train_layers = fcn_model.config.TRAINING_LAYERS
loss_names     = args.fcn_losses   ## ['fcn_BCE_loss']
# loss_names   = ['fcn_CE_loss']
# loss_names   = ['fcn_MSE_loss']
fcn_model.epoch                  = fcn_config.LAST_EPOCH_RAN

fcn_model.train_in_batches(
            mrcnn_model,    
            dataset_train,
            dataset_val, 
            layers = train_layers,
            losses = loss_names)


##------------------------------------------------------------------------------------
## Final save - only works with weights - Model is not JSON serializable
##------------------------------------------------------------------------------------
# final_save = 'fcn_{:04d}_final'.format(fcn_model.epoch) 
# file = fcn_model.save_model(filename = final_save)            
# print(' --> Final weights file saved: ', file)


print(' --> Execution ended at:',datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))
exit(' Execution terminated ' ) 
