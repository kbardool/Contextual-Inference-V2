# coding: utf-8
##-------------------------------------------------------------------------------------------
##
## Separated MRCNN-FCN Pipeline (import model_mrcnn)
## Train FCN head only
##
##  Pass predicitions from MRCNN to use as training data for FCN
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

import mrcnn.model_mrcnn  as mrcnn_modellib
import mrcnn.model_fcn    as fcn_modellib
import mrcnn.visualize    as visualize

from datetime           import datetime   
from mrcnn.config       import Config
from mrcnn.dataset      import Dataset 
from mrcnn.utils        import log, stack_tensors, stack_tensors_3d, write_stdout
from mrcnn.datagen      import data_generator, load_image_gt
from mrcnn.callbacks    import get_layer_output_1,get_layer_output_2
# from pycocotools.coco   import COCO
# from pycocotools.cocoeval import COCOeval
# from pycocotools        import mask as maskUtils
from mrcnn.coco         import CocoDataset, CocoConfig, CocoInferenceConfig, evaluate_coco, build_coco_results


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
                    default=1,
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
# args = parser.parse_args("train --dataset E:\MLDatasets\coco2014 --model mask_rcnn_coco.h5 --limit 10".split())
# pp.pprint(args)


##----------------------------------------------------------------------------------------------
## if debug is true set stdout destination to stringIO
##----------------------------------------------------------------------------------------------            
debug = False
if debug:
    sys.stdout = io.StringIO()

print()
print('--> Execution started at:', start_time)
print("    Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))

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
##   ROOT_DIR         : Root directory of the project 
##   MODEL_DIR        : Directory to save logs and trained model
##   COCO_MODEL_PATH  : Path to COCO trained weights
##---------------------------------------------------------------------------------
syst = platform.system()
print("    OS Platform        : ", syst)
if syst == 'Windows':
    # Root directory of the project
    # WINDOWS MACHINE ------------------------------------------------------------------
    ROOT_DIR          = "E:\\"
    TRAINING_DIR   = os.path.join(ROOT_DIR, "models")
    DATASET_DIR    = os.path.join(ROOT_DIR, 'MLDatasets')
    PRETRAINED_DIR = os.path.join(ROOT_DIR, 'PretrainedModels')
elif syst == 'Linux':
    # LINUX MACHINE ------------------------------------------------------------------
    ROOT_DIR       = os.getcwd()
    TRAINING_DIR   = os.path.expanduser('~/models')
    DATASET_DIR    = os.path.expanduser('~/MLDatasets')
    PRETRAINED_DIR = os.path.expanduser('~/PretrainedModels')
else :
    raise Error('unreconized system  '      )

# MODEL_DIR       = os.path.join(TRAINING_DIR  , "mrcnn_logs")
# DEFAULT_LOGS_DIR  = os.path.join(MODEL_PATH  , args.logs_dir)
TRAINING_PATH     = os.path.join(TRAINING_DIR  , args.logs_dir)
COCO_DATASET_PATH = os.path.join(DATASET_DIR   , "coco2014")
COCO_MODEL_PATH   = os.path.join(PRETRAINED_DIR, "mask_rcnn_coco.h5")
RESNET_MODEL_PATH = os.path.join(PRETRAINED_DIR, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
VGG16_MODEL_PATH  = os.path.join(PRETRAINED_DIR, "fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5")

##------------------------------------------------------------------------------------
## Build configuration object 
##------------------------------------------------------------------------------------
# mrcnn_config                    = shapes.NewShapesConfig()
# mrcnn_config.NAME               = 'mrcnn'              
# mrcnn_config.BATCH_SIZE         = int(args.batch_size)                  # Batch size is 2 (# GPUs * images/GPU).
# mrcnn_config.IMAGES_PER_GPU     = int(args.batch_size)                  # Must match BATCH_SIZE
# mrcnn_config.STEPS_PER_EPOCH    = int(args.steps_in_epoch)
# mrcnn_config.LEARNING_RATE      = float(args.lr)
                          
# mrcnn_config.EPOCHS_TO_RUN      = int(args.epochs)
# mrcnn_config.FCN_INPUT_SHAPE    = mrcnn_config.IMAGE_SHAPE[0:2]
# mrcnn_config.LAST_EPOCH_RAN     = int(args.last_epoch)
# mrcnn_config.WEIGHT_DECAY       = 2.0e-4
# mrcnn_config.VALIDATION_STEPS   = 100
# mrcnn_config.REDUCE_LR_FACTOR   = 0.5
# mrcnn_config.REDUCE_LR_COOLDOWN = 30
# mrcnn_config.REDUCE_LR_PATIENCE = 40
# mrcnn_config.EARLY_STOP_PATIENCE= 80
# mrcnn_config.NEW_LOG_FOLDER     = False
# mrcnn_config.EARLY_STOP_MIN_DELTA = 1.0e-4
# mrcnn_config.MIN_LR             = 1.0e-10
 
	
mrcnn_config                    = CocoInferenceConfig()
mrcnn_config.NAME               = 'mrcnn'              
mrcnn_config.COCO_MODEL_PATH    = COCO_MODEL_PATH
mrcnn_config.RESNET_MODEL_PATH  = RESNET_MODEL_PATH
mrcnn_config.VGG16_MODEL_PATH   = VGG16_MODEL_PATH
mrcnn_config.HEATMAP_SCALE_FACTOR = 4
##------------------------------------------------------------------------------------
## Build shape dataset for Training and Validation       
##------------------------------------------------------------------------------------
# if args.command == "train":
# Training dataset. Use the training set and 35K from the
# validation set, as as in the Mask RCNN paper.
# dataset_train = CocoDataset()
# dataset_train.load_coco(COCO_DATASET_PATH, "train", class_ids=[1,2,3,4,5,6,7,8,9,10])
# dataset_train.load_coco(COCO_DATASET_PATH, "val35k",class_ids=[1,2,3,4,5,6,7,8,9,10])
# dataset_train.prepare()

# Validation dataset
# dataset_val = CocoDataset()
# dataset_val.load_coco(COCO_DATASET_PATH, "minival",class_ids=[1,2,3,4,5,6,7,8,9,10])
# dataset_val.prepare()

##------------------------------------------------------------------------------------
## Build Mask RCNN Model in TRAINFCN mode
##------------------------------------------------------------------------------------
# try :
    # del mrcnn_model
    # gc.collect()
# except: 
    # pass
# KB.clear_session()
# mrcnn_model = mrcnn_modellib.MaskRCNN(mode="trainfcn", config=mrcnn_config, model_dir=MODEL_DIR)
mrcnn_model, dataset_train, dataset_val, _ , _, mrcnn_config = mrcnn_coco_train(mode = 'trainfcn', mrcnn_config = mrcnn_config)

##------------------------------------------------------------------------------------
## Load Mask RCNN Model Weight file
##------------------------------------------------------------------------------------
# exclude_list = ["mrcnn_class_logits"]
#load_model(model, init_with = args.model)   
exclude_list = []
mrcnn_model.load_model_weights(init_with = args.model, exclude = exclude_list)   

print('==========================================')
print(" MRCNN MODEL Load weight file COMPLETE    ")
print('==========================================')

mrcnn_config.display()  
mrcnn_model.layer_info()
# print(' Checkpoint directory  : ', mrcnn_config.TRAINING_DIR)
print(' Training dir           : ', TRAINING_DIR)
print(' Dataset dir            : ', DATASET_DIR)
print(' Pretrained dir         : ', PRETRAINED_DIR)
print(' Checkpoint folder      : ', mrcnn_config.TRAINING_PATH)
print(' COCO   Dataset Path    : ', mrcnn_config.COCO_DATASET_PATH)
print(' COCO   Model Path      : ', mrcnn_config.COCO_MODEL_PATH)
print(' ResNet Model Path      : ', mrcnn_config.RESNET_MODEL_PATH)
print(' VGG16  Model Path      : ', mrcnn_config.VGG16_MODEL_PATH)
print(' mrcnn_config.BATCH_SIZE: ', mrcnn_config.BATCH_SIZE)

# exit(8)


##------------------------------------------------------------------------------------
## Build configuration for FCN model
##------------------------------------------------------------------------------------
fcn_config = CocoConfig()

fcn_config.IMAGE_MAX_DIM        = 600
fcn_config.IMAGE_MIN_DIM        = 480      
fcn_config.NAME                 = 'fcn'              
# config.BATCH_SIZE      = 1                  # Batch size is 2 (# GPUs * images/GPU).
# config.IMAGES_PER_GPU  = 1                  # Must match BATCH_SIZE
fcn_config.BATCH_SIZE           = mrcnn_config.BATCH_SIZE                 # Batch size is 2 (# GPUs * images/GPU).
fcn_config.IMAGES_PER_GPU       = mrcnn_config.BATCH_SIZE               # Must match BATCH_SIZE
fcn_config.STEPS_PER_EPOCH      = int(args.steps_in_epoch)
fcn_config.LEARNING_RATE        = float(args.lr)
fcn_config.EPOCHS_TO_RUN        = int(args.epochs)
fcn_config.FCN_INPUT_SHAPE      = mrcnn_config.IMAGE_SHAPE[0:2] / mrcnn_config.HEATMAP_SCALE_FACTOR 
fcn_config.LAST_EPOCH_RAN       = int(args.last_epoch)
fcn_config.WEIGHT_DECAY         = 2.0e-4
fcn_config.VALIDATION_STEPS     = int(args.val_steps)
fcn_config.REDUCE_LR_FACTOR     = 0.5
fcn_config.REDUCE_LR_COOLDOWN   = 50
fcn_config.REDUCE_LR_PATIENCE   = 333
fcn_config.EARLY_STOP_PATIENCE  = 500
fcn_config.EARLY_STOP_MIN_DELTA = 1.0e-4
fcn_config.MIN_LR               = 1.0e-10
fcn_config.NEW_LOG_FOLDER       = True  
fcn_config.OPTIMIZER            = args.opt.upper()

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

##------------------------------------------------------------------------------------
## Load FCN Model weights  
##------------------------------------------------------------------------------------
# fcn_model.load_model_weights(init_with = args.fcn_model)

# print('=====================================')
# print(" Load second weight file COMPLETE    ")
# print('=====================================')
fcn_config.display()  
fcn_model.layer_info()

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
## Passing layers="heads" freezes all layers except the head
## layers. You can also pass a regular expression to select
## which layers to train by name pattern.
##----------------------------------------------------------------------------------------------            
train_layers = ['fcn']
loss_names   = ["fcn_norm_loss"]
fcn_model.epoch                  = fcn_config.LAST_EPOCH_RAN


fcn_model.train_in_batches(
            mrcnn_model,    
            optimizer,
            dataset_train,
            dataset_val, 
            layers = train_layers,
            losses = loss_names,
            debug = debug
            # learning_rate   = fcn_config.LEARNING_RATE,  
            # epochs          = 25,                             # total number of epochs to run (accross multiple trainings)
            # epochs_to_run   = fcn_config.EPOCHS_TO_RUN,
            # batch_size      = fcn_config.BATCH_SIZE,          # gets value from self.config.BATCH_SIZE
            # steps_per_epoch = fcn_config.STEPS_PER_EPOCH ,    # gets value form self.config.STEPS_PER_EPOCH
            # min_LR          = fcn_config.MIN_LR
            )

exit(' Execution terminated ' ) 

##------------------------------------------------------------------------------------
## Final save
##------------------------------------------------------------------------------------
final_save  =  "/home/kbardool/models/train_fcn_alt/fcn_train_final.h5"
file = fcn_model.save_model(final_save)            


print(' --> Execution ended at:',datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))

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

#------------------------------------------------------------------------------------
#  main training routine 
#------------------------------------------------------------------------------------
# for epoch_num in range(num_epochs):
    
    # X, Y, img_data = next(data_gen_train)
    
    
    # progbar = generic_utils.Progbar(epoch_length)
    # print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

    # Call mrcnn in inference mode
    # results = mrcnn_model.predict()
    # Call FCN in training mode
    # fcn_loss = fcn_model.train_on_batch(mrcnn_output, mrcnn_gt)

    # print('len fcn_loss is :', len(fcn_loss))
    



            
