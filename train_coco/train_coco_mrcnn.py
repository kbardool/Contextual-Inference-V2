# coding: utf-8
##-------------------------------------------------------------------------------------------
##
## Combined MRCNN-FCN Pipeline (import model_mrcnn) on COCO dataset
## Train MRCNN heads only
## MRCNN modeo (include model_mrcnn) does not include any mask related heads or losses 
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

import mrcnn.model_mrcnn  as mrcnn_modellib
import mrcnn.model_fcn    as fcn_modellib
import mrcnn.visualize    as visualize
import mrcnn.new_shapes   as shapes

from datetime           import datetime   
from mrcnn.config       import Config
from mrcnn.dataset      import Dataset 
from mrcnn.utils        import log, stack_tensors, stack_tensors_3d, write_stdout
from mrcnn.datagen_mod  import data_generator, load_image_gt
# from mrcnn.callbacks    import get_layer_output_1,get_layer_output_2
# from mrcnn.visualize    import plot_gaussian
# from mrcnn.prep_notebook import prep_oldshapes_train, load_model
from mrcnn.coco         import CocoDataset, CocoConfig, CocoInferenceConfig, evaluate_coco, build_coco_results
from mrcnn.prep_notebook import mrcnn_coco_train

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
                    help='Number of epochs to run (default=1)')
                    
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
debug = False
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
mrcnn_config                    = CocoConfig()
mrcnn_config.NAME               = 'mrcnn'              
mrcnn_config.TRAINING_PATH      = TRAINING_PATH
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
mrcnn_config.WEIGHT_DECAY       = 2.0e-4
mrcnn_config.VALIDATION_STEPS   = int(args.val_steps)
mrcnn_config.REDUCE_LR_FACTOR   = 0.5
mrcnn_config.REDUCE_LR_COOLDOWN = 30
mrcnn_config.REDUCE_LR_PATIENCE = 40
mrcnn_config.EARLY_STOP_PATIENCE= 80
mrcnn_config.EARLY_STOP_MIN_DELTA = 1.0e-4
mrcnn_config.MIN_LR             = 1.0e-10
mrcnn_config.NEW_LOG_FOLDER     = True  

mrcnn_config.display() 
    
##------------------------------------------------------------------------------------
## Build shape dataset for Training and Validation       
##------------------------------------------------------------------------------------

##------------------------------------------------------------------------------------
## Build Mask RCNN Model in TRAINING mode
##------------------------------------------------------------------------------------
# try :
    # del mrcnn_model
    # gc.collect()
# except: 
    # pass
# KB.clear_session()
# mrcnn_model = mrcnn_modellib.MaskRCNN(mode="training", config=mrcnn_config, model_dir=MODEL_DIR)
mrcnn_model, dataset_train, dataset_val, _ , _, mrcnn_config = mrcnn_coco_train(mode = 'training', mrcnn_config = mrcnn_config)

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


##----------------------------------------------------------------------------------------------
## Setup optimizaion method 
##----------------------------------------------------------------------------------------------            
print('    learning rate : ', mrcnn_model.config.LEARNING_RATE)
print('    momentum      : ', mrcnn_model.config.LEARNING_MOMENTUM)
# optimizer = keras.optimizers.SGD(lr=mrcnn_model.config.LEARNING_RATE, 
                                 # momentum=mrcnn_model.config.LEARNING_MOMENTUM, clipnorm=5.0)

# optimizer= tf.train.GradientDescentOptimizer(learning_rate, momentum)
# optimizer = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
optimizer = keras.optimizers.Adagrad(lr=mrcnn_model.config.LEARNING_RATE, epsilon=None, decay=0.0)
# optimizer = keras.optimizers.Adam(lr=mrcnn_model.config.LEARNING_RATE,beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# optimizer = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)


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
mrcnn_model.epoch                  = mrcnn_model.config.LAST_EPOCH_RAN
# mrcnn_model.config.LEARNING_RATE   = config.LEARNING_RATE
# mrcnn_model.config.STEPS_PER_EPOCH = config.STEPS_PER_EPOCH

mrcnn_model.train(optimizer, 
            dataset_train, 
            dataset_val, 
            learning_rate = mrcnn_model.config.LEARNING_RATE, 
            epochs_to_run = mrcnn_model.config.EPOCHS_TO_RUN,
            layers = train_layers,
            losses = loss_names
#             epochs = 25,            # total number of epochs to run (accross multiple trainings)
#             batch_size = 0
#             steps_per_epoch = 0 
			)
            

            
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