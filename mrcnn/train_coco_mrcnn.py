# coding: utf-8
##-------------------------------------------------------------------------------------------
##
## Combined MRCNN-FCN Pipeline (import model_mrcnn) on COCO dataset
## Train MRCNN heads only
## MRCNN modeo (include model_mrcnn) does not include any mask related heads or losses 
##
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
from mrcnn.prep_notebook import mrcnn_coco_train, coco_dataset
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
mrcnn_config.HEATMAP_SCALE_FACTOR = 4

mrcnn_config.BATCH_SIZE           = int(args.batch_size)                  # Batch size is 2 (# GPUs * images/GPU).
mrcnn_config.IMAGES_PER_GPU       = int(args.batch_size)                  # Must match BATCH_SIZE
mrcnn_config.STEPS_PER_EPOCH      = int(args.steps_in_epoch)
mrcnn_config.LEARNING_RATE        = float(args.lr)
mrcnn_config.EPOCHS_TO_RUN        = int(args.epochs)
mrcnn_config.FCN_INPUT_SHAPE      = mrcnn_config.IMAGE_SHAPE[0:2]
mrcnn_config.LAST_EPOCH_RAN       = int(args.last_epoch)

mrcnn_config.WEIGHT_DECAY         = 2.0e-4
mrcnn_config.VALIDATION_STEPS     = int(args.val_steps)
mrcnn_config.REDUCE_LR_FACTOR     = 0.5
mrcnn_config.REDUCE_LR_COOLDOWN   = 30
mrcnn_config.REDUCE_LR_PATIENCE   = 40
mrcnn_config.EARLY_STOP_PATIENCE  = 80
mrcnn_config.EARLY_STOP_MIN_DELTA = 1.0e-4
mrcnn_config.MIN_LR               = 1.0e-10

mrcnn_config.NEW_LOG_FOLDER       = args.new_log_folder  
mrcnn_config.OPTIMIZER            = args.opt.upper()
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
mrcnn_model = mrcnn_modellib.MaskRCNN(mode='training', config=mrcnn_config)


##------------------------------------------------------------------------------------
## Display model configuration information
##------------------------------------------------------------------------------------
paths.display()
mrcnn_config.display()  
mrcnn_model.layer_info()


##------------------------------------------------------------------------------------
## Load Mask RCNN Model Weight file
##------------------------------------------------------------------------------------
# exclude_list = ["mrcnn_class_logits"]
#load_model(model, init_with = args.model)   
exclude_list = []
mrcnn_model.load_model_weights(init_with = args.model, exclude = exclude_list)   

##----------------------------------------------------------------------------------------------
## Build COCO Training and Validation Datasets
##----------------------------------------------------------------------------------------------
dataset_train = coco_dataset(["train","val35k"], mrcnn_config)
dataset_val   = coco_dataset(["minival"]       , mrcnn_config)


##----------------------------------------------------------------------------------------------
## Train the head branches
## Passing layers="heads" freezes all layers except the head
## layers. You can also pass a regular expression to select
## which layers to train by name pattern.
##----------------------------------------------------------------------------------------------
train_layers = [ 'mrcnn', 'fpn','rpn']
loss_names   = [ "rpn_class_loss", "rpn_bbox_loss" , "mrcnn_class_loss", "mrcnn_bbox_loss"]
# train_layers = [ 'mrcnn']
# loss_names   = [ "mrcnn_class_loss", "mrcnn_bbox_loss"]

mrcnn_model.epoch = mrcnn_model.config.LAST_EPOCH_RAN

mrcnn_model.train(dataset_train, 
            dataset_val, 
            learning_rate = mrcnn_model.config.LEARNING_RATE, 
            epochs_to_run = mrcnn_model.config.EPOCHS_TO_RUN,
            layers = train_layers,
            losses = loss_names
#             epochs = 25,            # total number of epochs to run (accross multiple trainings)
#             batch_size = 0
#             steps_per_epoch = 0 
			)
            

print(' --> Execution ended at:',datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))
exit(' Execution terminated ' ) 

            
"""
##----------------------------------------------------------------------------------------------
##  Training
## 
## Train in two stages:
## 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly 
##    initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). 
##    To train only the head layers, pass `layers='heads'` to the `train()` function.
## 
## 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to 
##    show the process. Simply pass `layers="all` to train all layers.
## ## Training head using  Keras.model.fit_generator()
##----------------------------------------------------------------------------------------------

##------------------------------------------------------------------------------------    
## Load and display random samples
##------------------------------------------------------------------------------------
# image_ids = np.random.choice(dataset_train.image_ids, 3)
# for image_id in [3]:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
"""


#------------------------------------------------------------------------------------
# setup tf session and debugging 
#------------------------------------------------------------------------------------
# keras_backend.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))
# if 'tensorflow' == KB.backend():
#     from tensorflow.python import debug as tf_debug
#
#    config = tf.ConfigProto(device_count = {'GPU': 0} )
#    tf_sess = tf.Session(config=config)    
#    tf_sess = tf_debug.LocalCLIDebugWrapperSession(tf_sess)
#    KB.set_session(tf_sess)
#
#
#   tfconfig = tf.ConfigProto(
#               gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5),
#               device_count = {'GPU': 1}
#              )    
#     tfconfig = tf.ConfigProto()
#     tfconfig.gpu_options.allow_growth=True
#     tfconfig.gpu_options.visible_device_list = "0"
#     tfconfig.gpu_options.per_process_gpu_memory_fraction=0.5
#     tf_sess = tf.Session(config=tfconfig)
#     set_session(tf_sess)
#------------------------------------------------------------------------------------
