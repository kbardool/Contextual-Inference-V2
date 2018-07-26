# coding: utf-8
# # Mask R-CNN - Train modified model on Shapes Dataset
# ### the modified model (include model_lib) does not include any mask related heads or losses 
##-------------------------------------------------------------------------------------------
##
## Separated MRCNN-FCN Pipeline (import model_mrcnn)
## Train FCN head only
##
##  Pass predicitions from MRCNN to use as training data for FCN
##-------------------------------------------------------------------------------------------

import os
import sys
import math
import gc
import time
import numpy as np
import argparse
import platform
import tensorflow as tf
import keras
import keras.backend as KB
import platform

sys.path.append('../')

import mrcnn.model_mrcnn  as mrcnn_modellib
import mrcnn.model_fcn    as fcn_modellib
import mrcnn.visualize    as visualize
import mrcnn.new_shapes   as shapes

from mrcnn.config       import Config
from mrcnn.dataset      import Dataset 
from mrcnn.utils        import log, stack_tensors, stack_tensors_3d
from mrcnn.datagen      import data_generator, load_image_gt
from mrcnn.callbacks    import get_layer_output_1,get_layer_output_2
import pprint
pp = pprint.PrettyPrinter(indent=2, width=100)
np.set_printoptions(linewidth=100,precision=4,threshold=1000, suppress = True)
syst = platform.system()

##------------------------------------------------------------------------------------
## process input arguments
##  example:
##           train-shapes_gpu --epochs 12 --steps-in-epoch 7 --last_epoch 1234 --logs_dir mrcnn_logs
##------------------------------------------------------------------------------------
# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Mask R-CNN on MS COCO.')
# parser.add_argument("command",
                    # metavar="<command>",
                    # help="'train' or 'evaluate' on MS COCO")
                    
# parser.add_argument('--dataset', required=True,
                    # metavar="/path/to/coco/",
                    # help='Directory of the MS-COCO dataset')
                    
# parser.add_argument('--limit', required=False,
                    # default=500,
                    # metavar="<image count>",
                    # help='Images to use for evaluation (defaults=500)')
                    
parser.add_argument('--model', required=False,
                    default='last',
                    metavar="/path/to/weights.h5",
                    help="MRCNN model weights file: 'coco' , 'init' , or Path to weights .h5 file ")

parser.add_argument('--fcn_model', required=False,
                    default='last',
                    metavar="/path/to/weights.h5",
                    help="FCN model weights file: 'init' , or Path to weights .h5 file ")

parser.add_argument('--logs_dir', required=True,
                    default='mrcnn_logs',
                    metavar="/path/to/logs/",
                    help='Logs and checkpoints directory (default=logs/)')

parser.add_argument('--last_epoch', required=False,
                    default=0,
                    metavar="<last epoch ran>",
                    help='Identify last completed epcoh for tensorboard continuation')
                    
parser.add_argument('--epochs', required=False,
                    default=3,
                    metavar="<epochs to run>",
                    help='Number of epochs to run (default=3)')
                    
parser.add_argument('--steps_in_epoch', required=False,
                    default=1,
                    metavar="<steps in each epoch>",
                    help='Number of batches to run in each epochs (default=5)')
                    
parser.add_argument('--batch_size', required=False,
                    default=5,
                    metavar="<batch size>",
                    help='Number of data samples in each batch (default=5)')                    

parser.add_argument('--lr', required=False,
                    default=0.001,
                    metavar="<learning rate>",
                    help='Learning Rate (default=0.001)')

                    
args = parser.parse_args()
# args = parser.parse_args("train --dataset E:\MLDatasets\coco2014 --model mask_rcnn_coco.h5 --limit 10".split())
pp.pprint(args)
print()
print("Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))
# print("Dataset: ", args.dataset)
# print("Logs:    ", args.logs)
# print("Limit:   ", args.limit)

print("   MRCNN Model        : ", args.model)
print("   FCN Model          : ", args.fcn_model)
print("   Log Dir            : ", args.logs_dir)
print("   Last Epoch         : ", args.last_epoch)
print("   Epochs to run      : ", args.epochs)
print("   Steps in each epoch: ", args.steps_in_epoch)
print("   Batch Size         : ", args.batch_size)
print("   OS Platform        : ", syst)


##------------------------------------------------------------------------------------
## setup project directories
#---------------------------------------------------------------------------------
# # Root directory of the project 
# MODEL_DIR    :    Directory to save logs and trained model
# COCO_MODEL_PATH  : Path to COCO trained weights
#---------------------------------------------------------------------------------
if syst == 'Windows':
    # WINDOWS MACHINE ------------------------------------------------------------------
    ROOT_DIR          = "E:\\"
    MODEL_PATH        = os.path.join(ROOT_DIR    , "models")
    DATASET_PATH      = os.path.join(ROOT_DIR    , 'MLDatasets')
    MODEL_DIR         = os.path.join(MODEL_PATH  , args.logs_dir)
    COCO_MODEL_PATH   = os.path.join(MODEL_PATH  , "mask_rcnn_coco.h5")
    DEFAULT_LOGS_DIR  = os.path.join(MODEL_PATH  , args.logs_dir) 
    COCO_DATASET_PATH = os.path.join(DATASET_PATH, "coco2014")
    RESNET_MODEL_PATH = os.path.join(MODEL_PATH  , "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
    VGG16_MODEL_PATH  = os.path.join(MODEL_PATH  , "fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5")
    
elif syst == 'Linux':
    # LINUX MACHINE ------------------------------------------------------------------
    ROOT_DIR          = os.getcwd()
    MODEL_PATH        = os.path.expanduser('~/models')
    DATASET_PATH      = os.path.expanduser('~/MLDatasets')
    MODEL_DIR         = os.path.join(MODEL_PATH  , args.logs_dir)
    COCO_MODEL_PATH   = os.path.join(MODEL_PATH  , "mask_rcnn_coco.h5")
    COCO_DATASET_PATH = os.path.join(DATASET_PATH, "coco2014")
    DEFAULT_LOGS_DIR  = os.path.join(MODEL_PATH  , args.logs_dir)
    RESNET_MODEL_PATH = os.path.join(MODEL_PATH  , "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
    VGG16_MODEL_PATH  = os.path.join(MODEL_PATH  , "fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5")
else :
    raise Error('unreconized system  '      )


    
# class InferenceConfig(CocoConfig):
    # # Set batch size to 1 since we'll be running inference on
    # # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 1
    # IMAGES_PER_GPU = 1
    # DETECTION_MIN_CONFIDENCE = 0
# config = InferenceConfig()
##------------------------------------------------------------------------------------
## Build configuration object 
##------------------------------------------------------------------------------------
mrcnn_config                    = shapes.NewShapesConfig()
mrcnn_config.BATCH_SIZE         = int(args.batch_size)                  # Batch size is 2 (# GPUs * images/GPU).
mrcnn_config.IMAGES_PER_GPU     = int(args.batch_size)                  # Must match BATCH_SIZE
mrcnn_config.STEPS_PER_EPOCH    = int(args.steps_in_epoch)
mrcnn_config.LEARNING_RATE      = float(args.lr)
                          
mrcnn_config.EPOCHS_TO_RUN      = int(args.epochs)
mrcnn_config.FCN_INPUT_SHAPE    = mrcnn_config.IMAGE_SHAPE[0:2]
mrcnn_config.LAST_EPOCH_RAN     = int(args.last_epoch)
mrcnn_config.WEIGHT_DECAY       = 2.0e-4
mrcnn_config.VALIDATION_STEPS   = 100
mrcnn_config.REDUCE_LR_FACTOR   = 0.5
mrcnn_config.REDUCE_LR_COOLDOWN = 30
mrcnn_config.REDUCE_LR_PATIENCE = 40
mrcnn_config.EARLY_STOP_PATIENCE= 80
mrcnn_config.MIN_LR             = 1.0e-10
# mrcnn_config.display() 

##------------------------------------------------------------------------------------
## Build shape dataset        
##------------------------------------------------------------------------------------
# Training dataset
# generate 500 shapes 
dataset_train = shapes.NewShapesDataset(mrcnn_config)
dataset_train.load_shapes(10000)
dataset_train.prepare()

# Validation dataset
dataset_val = shapes.NewShapesDataset(mrcnn_config)
dataset_val.load_shapes(2500)
dataset_val.prepare()


##------------------------------------------------------------------------------------
## Build Mask RCNN Model in TRAINFCN mode
##------------------------------------------------------------------------------------

try :
    del mrcnn_model
    gc.collect()
except: 
    pass
KB.clear_session()
mrcnn_model = mrcnn_modellib.MaskRCNN(mode="trainfcn", config=mrcnn_config, model_dir=MODEL_DIR)

print(' COCO Model Path       : ', COCO_MODEL_PATH)
print(' Checkpoint folder Path: ', MODEL_DIR)
print(' Model Parent Path     : ', MODEL_PATH)
# print(model.find_last())

## Load Mask RCNN Model Weight file
exclude_layers = \
       ['fcn_block1_conv1' 
       ,'fcn_block1_conv2' 
       ,'fcn_block1_pool' 
       ,'fcn_block2_conv1'
       ,'fcn_block2_conv2' 
       ,'fcn_block2_pool'  
       ,'fcn_block3_conv1' 
       ,'fcn_block3_conv2' 
       ,'fcn_block3_conv3' 
       ,'fcn_block3_pool'  
       ,'fcn_block4_conv1' 
       ,'fcn_block4_conv2' 
       ,'fcn_block4_conv3' 
       ,'fcn_block4_pool'  
       ,'fcn_block5_conv1' 
       ,'fcn_block5_conv2' 
       ,'fcn_block5_conv3' 
       ,'fcn_block5_pool'  
       ,'fcn_fc1'          
       ,'dropout_1'        
       ,'fcn_fc2'          
       ,'dropout_2'        
       ,'fcn_classify'     
       ,'fcn_bilinear'     
       ,'fcn_heatmap_norm' 
       ,'fcn_scoring'      
       ,'fcn_heatmap'      
       ,'fcn_norm_loss']
mrcnn_model.load_model_weights( init_with = args.model)   
print('==========================================')
print(" MRCNN MODEL Load weight file COMPLETE    ")
print('==========================================')

mrcnn_config.display()  
mrcnn_model.layer_info()

##------------------------------------------------------------------------------------
## Build configuration for FCN model
##------------------------------------------------------------------------------------
fcn_config                    = shapes.NewShapesConfig()
fcn_config.BATCH_SIZE         = mrcnn_config.BATCH_SIZE                 # Batch size is 2 (# GPUs * images/GPU).
fcn_config.IMAGES_PER_GPU     = mrcnn_config.BATCH_SIZE               # Must match BATCH_SIZE
fcn_config.STEPS_PER_EPOCH    = int(args.steps_in_epoch)

fcn_config.LEARNING_RATE      = float(args.lr)
# fcn_config.LEARNING_RATE      = 0.005
                          
fcn_config.EPOCHS_TO_RUN      = int(args.epochs)
fcn_config.FCN_INPUT_SHAPE    = mrcnn_config.IMAGE_SHAPE[0:2]
fcn_config.LAST_EPOCH_RAN     = int(args.last_epoch)
fcn_config.WEIGHT_DECAY       = 2.0e-4
fcn_config.VALIDATION_STEPS   = 5
fcn_config.REDUCE_LR_FACTOR   = 0.5
fcn_config.REDUCE_LR_COOLDOWN = 2
fcn_config.REDUCE_LR_PATIENCE = 3
fcn_config.EARLY_STOP_PATIENCE= 6
fcn_config.MIN_LR             = 1.0e-10
fcn_config.display() 

##------------------------------------------------------------------------------------
## Build FCN Model in Training Mode
##------------------------------------------------------------------------------------
try :
    del fcn_model
    gc.collect()
except: 
    pass
fcn_model = fcn_modellib.FCN(mode="training", config=fcn_config, model_dir=MODEL_DIR)

print(' COCO Model Path       : ', COCO_MODEL_PATH)
print(' Checkpoint folder Path: ', MODEL_DIR)
print(' Model Parent Path     : ', MODEL_PATH)

print('=====================================')
print(" Load second weight file  ")
print('=====================================')
fcn_model.load_model_weights(init_with = args.fcn_model)

print('=====================================')
print(" Load second weight file COMPLETE    ")
print('=====================================')
fcn_config.display()  
fcn_model.layer_info()

# exit(8)
##----------------------------------------------------------------------------------------------
## Train the FCN only 
## Passing layers="heads" freezes all layers except the head
## layers. You can also pass a regular expression to select
## which layers to train by name pattern.
##----------------------------------------------------------------------------------------------            

train_layers = ['fcn']
loss_names   = ["fcn_norm_loss"]

fcn_model.epoch                  = fcn_config.LAST_EPOCH_RAN
# fcn_model.config.LEARNING_RATE   = fcn_config.LEARNING_RATE
# fcn_model.config.STEPS_PER_EPOCH = fcn_config.STEPS_PER_EPOCH

fcn_model.train_in_batches(
            mrcnn_model,    
            dataset_train,
            dataset_val, 
            layers = train_layers,
            losses = loss_names
            # learning_rate   = fcn_config.LEARNING_RATE,  
            # epochs          = 25,                             # total number of epochs to run (accross multiple trainings)
            # epochs_to_run   = fcn_config.EPOCHS_TO_RUN,
            # batch_size      = fcn_config.BATCH_SIZE,          # gets value from self.config.BATCH_SIZE
            # steps_per_epoch = fcn_config.STEPS_PER_EPOCH ,    # gets value form self.config.STEPS_PER_EPOCH
            # min_LR          = fcn_config.MIN_LR
            )

final_save  =  "/home/kbardool/models/train_fcn_alt/fcn_train_final.h5"
file = fcn_model.save_model(final_save)            



##------------------------------------------------------------------------------------
## setup tf session and debugging 
##------------------------------------------------------------------------------------
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
##------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
#  main training routine 
#---------------------------------------------------------------------------------
# for epoch_num in range(num_epochs):
    
    # X, Y, img_data = next(data_gen_train)
    
    
    # progbar = generic_utils.Progbar(epoch_length)
    # print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

    # Call mrcnn in inference mode
    # results = mrcnn_model.predict()
    # Call FCN in training mode
    # fcn_loss = fcn_model.train_on_batch(mrcnn_output, mrcnn_gt)

    # print('len fcn_loss is :', len(fcn_loss))
    



            