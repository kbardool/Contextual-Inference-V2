# coding: utf-8
# # Mask R-CNN - Train modified model on Shapes Dataset
# ### the modified model (include model_lib) does not include any mask related heads or losses 
##-------------------------------------------------------------------------------------------
##
## Combined MRCNN-FCN Pipeline (import model_mrcnn)
## Train MRCNN heads only
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

import mrcnn.model_mrcnn  as mrcnn_modellib
import mrcnn.model_fcn    as fcn_modellib
import mrcnn.visualize    as visualize
import mrcnn.new_shapes   as shapes
from datetime import datetime   

from mrcnn.config       import Config
from mrcnn.dataset      import Dataset 
from mrcnn.utils        import log, stack_tensors, stack_tensors_3d, write_stdout
from mrcnn.datagen      import data_generator, load_image_gt
from mrcnn.callbacks    import get_layer_output_1,get_layer_output_2
# from mrcnn.visualize    import plot_gaussian
# from mrcnn.prep_notebook import prep_oldshapes_train, load_model

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
print("    OS Platform        : ", syst)


##------------------------------------------------------------------------------------
## setup project directories
#---------------------------------------------------------------------------------
# # Root directory of the project 
# MODEL_DIR    :    Directory to save logs and trained model
# COCO_MODEL_PATH  : Path to COCO trained weights
#---------------------------------------------------------------------------------
import platform
syst = platform.system()
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
mrcnn_config.EARLY_STOP_MIN_DELTA = 1.0e-4
mrcnn_config.MIN_LR             = 1.0e-10
mrcnn_config.NEW_LOG_FOLDER     = True 

##------------------------------------------------------------------------------------
## Build shape dataset for Training and Validation       
##------------------------------------------------------------------------------------
dataset_train = shapes.NewShapesDataset(mrcnn_config)
dataset_train.load_shapes(10000)
dataset_train.prepare()

dataset_val = shapes.NewShapesDataset(mrcnn_config)
dataset_val.load_shapes(2500)
dataset_val.prepare()


##------------------------------------------------------------------------------------
## Build Mask RCNN Model in TRAINING mode
##------------------------------------------------------------------------------------
try :
    del mrcnn_model
    gc.collect()
except: 
    pass
KB.clear_session()
mrcnn_model = mrcnn_modellib.MaskRCNN(mode="training", config=mrcnn_config, model_dir=MODEL_DIR, FCN_layers = False)

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

print(' COCO Model Path       : ', COCO_MODEL_PATH)
print(' Checkpoint folder Path: ', MODEL_DIR)
print(' Model Parent Path     : ', MODEL_PATH)
print('config.BATCH_SIZE      : ', config.BATCH_SIZE)
print('model.config.BATCH_SIZE: ', model.config.BATCH_SIZE)
# exit(8)


##----------------------------------------------------------------------------------------------
## Setup optimizaion method 
##----------------------------------------------------------------------------------------------            
print('    learning rate : ', mrcnn_model.config.LEARNING_RATE)
print('    momentum      : ', mrcnn_model.config.LEARNING_MOMENTUM)
optimizer = keras.optimizers.SGD(lr=mrcnn_model.config.LEARNING_RATE, 
                                 momentum=mrcnn_model.config.LEARNING_MOMENTUM, clipnorm=5.0)
# optimizer= tf.train.GradientDescentOptimizer(learning_rate, momentum)
# optimizer = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
# optimizer = keras.optimizers.Adagrad(lr=learning_rate, epsilon=None, decay=0.0)
# optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
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
mrcnn_model.epoch                  = config.LAST_EPOCH_RAN
mrcnn_model.config.LEARNING_RATE   = config.LEARNING_RATE
mrcnn_model.config.STEPS_PER_EPOCH = config.STEPS_PER_EPOCH

mrcnn_model.train(optimizer, dataset_train, dataset_val, 
            learning_rate = model.config.LEARNING_RATE, 
            epochs_to_run = config.EPOCHS_TO_RUN,
#             epochs = 25,            # total number of epochs to run (accross multiple trainings)
#             batch_size = 0
#             steps_per_epoch = 0 
            layers = train_layers,
            losses = loss_names
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