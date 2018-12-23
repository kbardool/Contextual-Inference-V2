# coding: utf-8
# # Mask R-CNN - Train modified model on Shapes Dataset
# ### the modified model (include model_lib) does not include any mask related heads or losses 
##-------------------------------------------------------------------------------------------
##
## Combined MRCNN-FCN Pipeline (import model_mrcnn)
## Train all heads (MRCNN and FCN)  
##
##
##-------------------------------------------------------------------------------------------

import os, sys, math, io, time
import gc
import numpy as np
import argparse
import platform
import tensorflow as tf
import keras
import keras.backend as KB
import platform
sys.path.append('../')

import mrcnn.model_combined  as comb_model
import mrcnn.visualize    as visualize
import mrcnn.new_shapes   as shapes
from datetime import datetime   

from mrcnn.config       import Config
from mrcnn.dataset      import Dataset 
from mrcnn.utils        import log, stack_tensors, stack_tensors_3d, write_stdout
from mrcnn.datagen      import data_generator, load_image_gt
from mrcnn.callbacks    import get_layer_output_1,get_layer_output_2
import pprint
pp = pprint.PrettyPrinter(indent=2, width=100)
np.set_printoptions(linewidth=100,precision=4,threshold=1000, suppress = True)
syst = platform.system()


start_time = datetime.now().strftime("%m-%d-%Y @ %H:%M:%S")
print()
print('--> Execution started at:', start_time)
print("    Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))

##------------------------------------------------------------------------------------
## Parse command line arguments
##  
## Example:
##           train-shapes_gpu --epochs 12 --steps-in-epoch 7 --last_epoch 1234 --logs_dir mrcnn_logs
##------------------------------------------------------------------------------------
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

parser.add_argument('--val_steps', required=False,
                    default=1,
                    metavar="<val steps in each epoch>",
                    help='Number of validation batches to run at end of each epoch (default=1)')
                    
parser.add_argument('--batch_size', required=False,
                    default=5,
                    metavar="<batch size>",
                    help='Number of data samples in each batch (default=5)')                    

parser.add_argument('--lr', required=False,
                    default=0.001,
                    metavar="<learning rate>",
                    help='Learning Rate (default=0.001)')

parser.add_argument('--opt', required=False,
                    default='adagrad',
                    metavar="<optimizer>",
                    help='Optimizatoin Method: SGD, RMSPROP, ADAGRAD, ...')
                    
args = parser.parse_args()


##----------------------------------------------------------------------------------------------
## if debug is true set stdout destination to stringIO
##----------------------------------------------------------------------------------------------            
debug = True
if debug:
    sys.stdout = io.StringIO()

# args = parser.parse_args("train --dataset E:\MLDatasets\coco2014 --model mask_rcnn_coco.h5 --limit 10".split())
# pp.pprint(args)

print()
print('--> Execution started at:', start_time)
print("    Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))

# print("Dataset: ", args.dataset)
# print("Logs:    ", args.logs)
# print("Limit:   ", args.limit)

print("    MRCNN Model        : ", args.model)
print("    FCN Model          : ", args.fcn_model)
print("    Log Dir            : ", args.logs_dir)
print("    Last Epoch         : ", args.last_epoch)
print("    Epochs to run      : ", args.epochs)
print("    Steps in each epoch: ", args.steps_in_epoch)
print("    Validation steps   : ", args.val_steps)
print("    Batch Size         : ", args.batch_size)
print("    Optimizer          : ", args.opt.upper)
print("    OS Platform        : ", syst)




##------------------------------------------------------------------------------------
## setup project directories
## 
## ROOT_DIR        :    Root directory of the project 
## MODEL_DIR       :    Directory to save logs and trained model
## COCO_MODEL_PATH :    Path to COCO trained weights
##---------------------------------------------------------------------------------

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
 
##------------------------------------------------------------------------------------
## Build configuration object 
##------------------------------------------------------------------------------------
comb_config                    = shapes.NewShapesConfig()
comb_config.BATCH_SIZE         = int(args.batch_size)                  # Batch size is 2 (# GPUs * images/GPU).
comb_config.IMAGES_PER_GPU     = int(args.batch_size)                  # Must match BATCH_SIZE
comb_config.STEPS_PER_EPOCH    = int(args.steps_in_epoch)
comb_config.LEARNING_RATE      = float(args.lr)
                          
comb_config.EPOCHS_TO_RUN      = int(args.epochs)
comb_config.FCN_INPUT_SHAPE    = comb_config.IMAGE_SHAPE[0:2]
comb_config.LAST_EPOCH_RAN     = int(args.last_epoch)
comb_config.WEIGHT_DECAY       = 2.0e-4
comb_config.VALIDATION_STEPS   = 100
comb_config.REDUCE_LR_FACTOR   = 0.5
comb_config.REDUCE_LR_COOLDOWN = 30
comb_config.REDUCE_LR_PATIENCE = 40
comb_config.EARLY_STOP_PATIENCE= 80
comb_config.NEW_LOG_FOLDER     = True  
comb_config.EARLY_STOP_MIN_DELTA = 1.0e-4
comb_config.MIN_LR             = 1.0e-10


##------------------------------------------------------------------------------------
## Build shape dataset for Training and Validation       
##------------------------------------------------------------------------------------
dataset_train = shapes.NewShapesDataset(comb_config)
dataset_train.load_shapes(10000)
dataset_train.prepare()

dataset_val = shapes.NewShapesDataset(comb_config)
dataset_val.load_shapes(2500)
dataset_val.prepare()

##------------------------------------------------------------------------------------
## Build Model
##------------------------------------------------------------------------------------

try :
    del model
    gc.collect()
except: 
    pass
KB.clear_session()
comb_model = comb_model.MaskRCNN(mode="training", config=comb_config, model_dir=MODEL_DIR, FCN_layers = True)

##----------------------------------------------------------------------------------------------
## Load Model Weight file
##----------------------------------------------------------------------------------------------
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
load_model(model, init_with = args.model, exclude = exclude_layers)   

print('==========================================')
print(" MRCNN MODEL Load weight file COMPLETE    ")
print('==========================================')

"""   
print('=====================================')
print(" Load second weight file  ")
print('=====================================')

model.keras_model.load_weights('/home/kbardool/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name= True )

print('=====================================')
print(" Load second weight file COMPLETE    ")
print('=====================================')
"""

comb_config.display()  
comb_model.layer_info()

print(' COCO Model Path       : ', COCO_MODEL_PATH)
print(' Checkpoint folder Path: ', MODEL_DIR)
print(' Model Parent Path     : ', MODEL_PATH)


#----------------------------------------------------------------------------------------------
# If in debug mode write stdout intercepted IO to output file  
#----------------------------------------------------------------------------------------------            
# if debug:
    # write_stdout(fcn_model.log_dir, output_filename, sys.stdout )        
    # sys.stdout = sys.__stdout__
# print(' Run information written to ', fcn_model.log_dir+'_sysout.out')

##----------------------------------------------------------------------------------------------
## Setup optimizaion method 
##----------------------------------------------------------------------------------------------            
print('    learning rate : ', fcn_model.config.LEARNING_RATE)
print('    momentum      : ', fcn_model.config.LEARNING_MOMENTUM)
print()

opt = fcn_model.config.OPTIMIZER
if   opt == 'ADAGRAD':
    optimizer = keras.optimizers.Adagrad(lr=fcn_model.config.LEARNING_RATE, epsilon=None, decay=0.01)                                 
elif opt == 'SGD':
    optimizer = keras.optimizers.SGD(lr=fcn_model.config.LEARNING_RATE, 
                                 momentum=fcn_model.config.LEARNING_MOMENTUM, clipnorm=5.0)
elif opt == 'RMSPROP':                                 
    optimizer = keras.optimizers.RMSprop(lr=fcn_model.config.LEARNING_RATE, rho=0.9, epsilon=None, decay=0.0)
else:
    print('ERROR: Invalid optimizer specified:',opt)
    if debug:
        write_stdout(fcn_model.log_dir, '_sysout', sys.stdout )        
        sys.stdout = sys.__stdout__
    print('\n  Run information written to ', fcn_model.log_dir+'_sysout.out')
    print('  ERROR: Invalid optimizer specified:',opt)
    sys.exit('  Execution Terminated')
    
    
    
# optimizer= tf.train.GradientDescentOptimizer(learning_rate, momentum)
# optimizer = keras.optimizers.Adagrad(lr=learning_rate, epsilon=None, decay=0.0)
# optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# optimizer = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    
##----------------------------------------------------------------------------------------------
## Train the FCN only 
##
## Passing layers="heads" freezes all layers except the head
## layers. You can also pass a regular expression to select
## which layers to train by name pattern.
##----------------------------------------------------------------------------------------------            
train_layers = ['fcn']
loss_names   = ["fcn_norm_loss"]

comb_model.epoch                  = comb_config.LAST_EPOCH_RAN
comb_model.config.LEARNING_RATE   = comb_config.LEARNING_RATE
comb_model.config.STEPS_PER_EPOCH = comb_config.STEPS_PER_EPOCH

comb_model.train(
            optimizer,
            dataset_train, 
            dataset_val, 
            layers = train_layers,
            losses = loss_names
            # learning_rate = model.config.LEARNING_RATE, 
            # epochs_to_run = config.EPOCHS_TO_RUN,
            # epochs = 25,            # total number of epochs to run (accross multiple trainings)
            # batch_size = 0
            # steps_per_epoch = 0 
            # min_LR = 1.0e-9,
            )
            
            
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

            