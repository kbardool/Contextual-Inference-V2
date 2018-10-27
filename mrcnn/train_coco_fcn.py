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
from mrcnn.prep_notebook import mrcnn_coco_train, coco_dataset
from mrcnn.utils        import command_line_parser
                                
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
print("    MRCNN Model        : ", args.model)
print("    FCN Model          : ", args.fcn_model)
print("    MRCNN Log/Ckpt Dir : ", args.mrcnn_logs_dir)
print("    FCN Log/Ckpt  Dir  : ", args.fcn_logs_dir)
print("    FCN architecture   : ", args.fcn_arch)
print("    Last Epoch         : ", args.last_epoch)
print("    Epochs to run      : ", args.epochs)
print("    Steps in each epoch: ", args.steps_in_epoch)
print("    Validation steps   : ", args.val_steps)
print("    Batch Size         : ", args.batch_size)
print("    Optimizer          : ", type(args.opt), args.opt)
print("    New Log Folder     : ", type(args.new_log_folder), args.new_log_folder)
print("    Sysout             : ", type(args.sysout), args.sysout)
print("    OS Platform        : ", os_platform)

if args.sysout == 'FILE':
    print(' Output is written to file....')
    sys.stdout = io.StringIO()

##------------------------------------------------------------------------------------
## setup project directories
##   DIR_ROOT         : Root directory of the project 
##   MODEL_DIR        : Directory to save logs and trained model
##   COCO_MODEL_PATH  : Path to COCO trained weights
##---------------------------------------------------------------------------------
if os_platform == 'Windows':
    # Root directory of the project
    print(' windows ' , os_platform)
    # WINDOWS MACHINE ------------------------------------------------------------------
    DIR_ROOT          = "E:\\"
    DIR_TRAINING   = os.path.join(DIR_ROOT, "models")
    DIR_DATASET    = os.path.join(DIR_ROOT, 'MLDatasets')
    DIR_PRETRAINED = os.path.join(DIR_ROOT, 'PretrainedModels')
elif os_platform == 'Linux':
    print(' Linx ' , os_platform)
    # LINUX MACHINE ------------------------------------------------------------------
    DIR_ROOT       = os.getcwd()
    DIR_TRAINING   = os.path.expanduser('~/models')
    DIR_DATASET    = os.path.expanduser('~/MLDatasets')
    DIR_PRETRAINED = os.path.expanduser('~/PretrainedModels')
else :
    raise Error('unreconized system ')

MRCNN_TRAINING_PATH   = os.path.join(DIR_TRAINING  , args.mrcnn_logs_dir)
FCN_TRAINING_PATH     = os.path.join(DIR_TRAINING  , args.fcn_logs_dir)
COCO_DATASET_PATH     = os.path.join(DIR_DATASET   , "coco2014")
COCO_MODEL_PATH       = os.path.join(DIR_PRETRAINED, "mask_rcnn_coco.h5")
RESNET_MODEL_PATH     = os.path.join(DIR_PRETRAINED, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
VGG16_MODEL_PATH      = os.path.join(DIR_PRETRAINED, "vgg16_weights_tf_dim_ordering_tf_kernels.h5")
FCN_VGG16_MODEL_PATH  = os.path.join(DIR_PRETRAINED, "fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5")

##------------------------------------------------------------------------------------
## Build configuration object 
##------------------------------------------------------------------------------------
# mrcnn_config                   = CocoInferenceConfig()
mrcnn_config                    = CocoConfig()
mrcnn_config.NAME               = 'mrcnn'              
mrcnn_config.TRAINING_PATH      = MRCNN_TRAINING_PATH
mrcnn_config.COCO_DATASET_PATH  = COCO_DATASET_PATH 
mrcnn_config.COCO_MODEL_PATH    = COCO_MODEL_PATH   
mrcnn_config.RESNET_MODEL_PATH  = RESNET_MODEL_PATH 
mrcnn_config.VGG16_MODEL_PATH   = VGG16_MODEL_PATH  
mrcnn_config.COCO_CLASSES       = None 
mrcnn_config.DETECTION_PER_CLASS = 200
mrcnn_config.HEATMAP_SCALE_FACTOR = 4
mrcnn_config.BATCH_SIZE         = int(args.batch_size)                  # Batch size is 2 (# GPUs * images/GPU).
mrcnn_config.IMAGES_PER_GPU     = int(args.batch_size)                  # Must match BATCH_SIZE

mrcnn_config.STEPS_PER_EPOCH    = int(args.steps_in_epoch)
mrcnn_config.LEARNING_RATE      = float(args.lr)
mrcnn_config.EPOCHS_TO_RUN      = int(args.epochs)
mrcnn_config.FCN_INPUT_SHAPE    = mrcnn_config.IMAGE_SHAPE[0:2]
mrcnn_config.LAST_EPOCH_RAN     = int(args.last_epoch)

# mrcnn_config.WEIGHT_DECAY       = 2.0e-4
# mrcnn_config.VALIDATION_STEPS   = int(args.val_steps)
# mrcnn_config.REDUCE_LR_FACTOR   = 0.5
# mrcnn_config.REDUCE_LR_COOLDOWN = 30
# mrcnn_config.REDUCE_LR_PATIENCE = 40
# mrcnn_config.EARLY_STOP_PATIENCE= 80
# mrcnn_config.EARLY_STOP_MIN_DELTA = 1.0e-4
# mrcnn_config.MIN_LR             = 1.0e-10
# mrcnn_config.OPTIMIZER          = args.opt.upper()
# mrcnn_model.config.OPTIMIZER    = 'ADAGRAD'
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
# mrcnn_model, dataset_train, dataset_val, _ , _, mrcnn_config = mrcnn_coco_train(mode = 'trainfcn', mrcnn_config = mrcnn_config)

##------------------------------------------------------------------------------------
## Load Mask RCNN Model Weight file
##------------------------------------------------------------------------------------
exclude_list = []
mrcnn_model.load_model_weights(init_with = args.model, exclude = exclude_list)   


##------------------------------------------------------------------------------------
## Display model configuration information
##------------------------------------------------------------------------------------
mrcnn_config.display()  
print()
mrcnn_model.layer_info()
print()
print(' Training dir           : ', DIR_TRAINING)
print(' Dataset dir            : ', DIR_DATASET)
print(' Pretrained dir         : ', DIR_PRETRAINED)
print(' Checkpoint folder      : ', mrcnn_config.TRAINING_PATH)
print(' COCO   Dataset Path    : ', mrcnn_config.COCO_DATASET_PATH)
print(' COCO   Model Path      : ', mrcnn_config.COCO_MODEL_PATH)
print(' ResNet Model Path      : ', mrcnn_config.RESNET_MODEL_PATH)
print(' VGG16  Model Path      : ', mrcnn_config.VGG16_MODEL_PATH)
print(' mrcnn_config.BATCH_SIZE: ', mrcnn_config.BATCH_SIZE)



##------------------------------------------------------------------------------------
## Build configuration for FCN model
##------------------------------------------------------------------------------------
fcn_config = CocoConfig()
# fcn_config.IMAGE_MAX_DIM        = 600
# fcn_config.IMAGE_MIN_DIM        = 480      
fcn_config.NAME                 = 'fcn'              
fcn_config.TRAINING_PATH        = FCN_TRAINING_PATH
# mrcnn_config.COCO_DATASET_PATH  = COCO_DATASET_PATH 
# mrcnn_config.COCO_MODEL_PATH    = COCO_MODEL_PATH   
# mrcnn_config.RESNET_MODEL_PATH  = RESNET_MODEL_PATH 
fcn_config.VGG16_MODEL_PATH     = FCN_VGG16_MODEL_PATH
fcn_config.FCN_INPUT_SHAPE      = mrcnn_config.IMAGE_SHAPE[0:2] // mrcnn_config.HEATMAP_SCALE_FACTOR 

fcn_config.BATCH_SIZE           = int(args.batch_size)                  # Batch size is 2 (# GPUs * images/GPU).
fcn_config.IMAGES_PER_GPU       = int(args.batch_size)                  # Must match BATCH_SIZE
fcn_config.EPOCHS_TO_RUN        = int(args.epochs)
fcn_config.STEPS_PER_EPOCH      = int(args.steps_in_epoch)
fcn_config.LEARNING_RATE        = float(args.lr)
fcn_config.LAST_EPOCH_RAN       = int(args.last_epoch)
fcn_config.VALIDATION_STEPS     = int(args.val_steps)

fcn_config.WEIGHT_DECAY         = 2.0e-4
fcn_config.REDUCE_LR_FACTOR     = 0.5
fcn_config.REDUCE_LR_COOLDOWN   = 5
fcn_config.REDUCE_LR_PATIENCE   = 5
fcn_config.EARLY_STOP_PATIENCE  = 15
fcn_config.EARLY_STOP_MIN_DELTA = 1.0e-4
fcn_config.MIN_LR               = 1.0e-10

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
## Load FCN Model weights  
##------------------------------------------------------------------------------------
if args.fcn_model != 'init':
    fcn_model.load_model_weights(init_with = args.fcn_model)
else:
    print(' FCN Training starting from randomly initialized weights ...')


##------------------------------------------------------------------------------------
## Display model configuration information
##------------------------------------------------------------------------------------
fcn_config.display()  
print()
fcn_model.layer_info()
print()
print(' Training dir        : ', DIR_TRAINING)
print(' Dataset dir         : ', DIR_DATASET)
print(' Pretrained dir      : ', DIR_PRETRAINED)
print(' Checkpoint folder   : ', fcn_config.TRAINING_PATH)
# print(' COCO   Dataset Pat  : ', fcn_config.COCO_DATASET_PATH)
# print(' COCO   Model Path   : ', fcn_config.COCO_MODEL_PATH)
# print(' ResNet Model Path   : ', fcn_config.RESNET_MODEL_PATH)
print(' VGG16 Model Path    : ', fcn_config.VGG16_MODEL_PATH)
print(' BATCH_SIZE          : ', fcn_config.BATCH_SIZE)

##------------------------------------------------------------------------------------
## Build & Load Training and Validation datasets
##------------------------------------------------------------------------------------
dataset_train = coco_dataset(["train",  "val35k"], mrcnn_config)
dataset_val   = coco_dataset(["minival"]         , mrcnn_config)

  
##----------------------------------------------------------------------------------------------
## Train the FCN only 
## Passing layers="heads" freezes all layers except the head
## layers. You can also pass a regular expression to select
## which layers to train by name pattern.
##----------------------------------------------------------------------------------------------            
train_layers = ['block5+']
loss_names   = ['fcn_heatmap_loss']
fcn_model.epoch                  = fcn_config.LAST_EPOCH_RAN

fcn_model.train_in_batches(
            mrcnn_model,    
            dataset_train,
            dataset_val, 
            layers = train_layers,
            losses = loss_names,
            # learning_rate   = fcn_config.LEARNING_RATE,  
            # epochs          = 25,                             # total number of epochs to run (accross multiple trainings)
            # epochs_to_run   = fcn_config.EPOCHS_TO_RUN,
            # batch_size      = fcn_config.BATCH_SIZE,          # gets value from self.config.BATCH_SIZE
            # steps_per_epoch = fcn_config.STEPS_PER_EPOCH ,    # gets value form self.config.STEPS_PER_EPOCH
            # min_LR          = fcn_config.MIN_LR
            )


##------------------------------------------------------------------------------------
## Final save - not working properly - Model is not JSON serializable
##------------------------------------------------------------------------------------
# final_save = os.path.join(fcn_config.TRAINING_PATH, "fcn_train_final.h5")
# final_save  =  "/home/kbardool/models/train_fcn_alt/fcn_train_final.h5"
# file = fcn_model.save_model(final_save)            


print(' --> Execution ended at:',datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))
exit(' Execution terminated ' ) 



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
