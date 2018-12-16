'''
prep_dev_notebook:
pred_newshapes_dev: Runs against new_shapes
'''
import os, sys, math, io, time, gc, argparse, platform, pprint, time, random, re
from   datetime           import datetime   
import pprint
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.backend as KB

import mrcnn.model_mrcnn  as mrcnn_modellib
import mrcnn.model_fcn    as fcn_modellib
import mrcnn.visualize    as visualize
import mrcnn.utils        as utils

# import mrcnn.new_shapes   as shapes
from mrcnn.config      import Config
from mrcnn.dataset     import Dataset 
from mrcnn.utils       import stack_tensors, stack_tensors_3d, log, Paths,command_line_parser
from mrcnn.datagen     import data_generator, load_image_gt, data_gen_simulate
from mrcnn.coco        import CocoConfig, CocoInferenceConfig, prep_coco_dataset
from mrcnn.heatmap     import HeatmapDataset
from mrcnn.datagen_fcn import fcn_data_generator,fcn_data_gen_simulate
from mrcnn.datagen     import load_image_gt

import mrcnn.newshapes    as newshapes


pp = pprint.PrettyPrinter(indent=2, width=100)
np.set_printoptions(linewidth=100,precision=4,threshold=1000, suppress = True)

    

"""
    
##------------------------------------------------------------------------------------    
## mrcnn COCO TEST
##------------------------------------------------------------------------------------    
def mrcnn_coco_test(mode = 'inference' , 
                    batch_sz = 5, epoch_steps = 4, training_folder = "train_mrcnn_coco",
                    mrcnn_config = None , verbose = 0):

    if mrcnn_config is None:    
        paths = Paths()
        paths.display()
        mrcnn_config = CocoInferenceConfig()
        mrcnn_config.NAME                 = 'mrcnn'              
        mrcnn_config.DIR_DATASET        = paths.DIR_DATASET
        mrcnn_config.DIR_TRAINING       = paths.DIR_TRAINING
        mrcnn_config.DIR_PRETRAINED     = paths.DIR_PRETRAINED
        mrcnn_config.TRAINING_PATH      = os.path.join(paths.DIR_TRAINING, training_folder)
        mrcnn_config.COCO_DATASET_PATH  = paths.COCO_DATASET_PATH 
        mrcnn_config.COCO_MODEL_PATH    = paths.COCO_MODEL_PATH   
        mrcnn_config.RESNET_MODEL_PATH  = paths.RESNET_MODEL_PATH 
        mrcnn_config.VGG16_MODEL_PATH   = paths.VGG16_MODEL_PATH  

        mrcnn_config.DETECTION_PER_CLASS  = 200
        mrcnn_config.HEATMAP_SCALE_FACTOR = 4
        mrcnn_config.NEW_LOG_FOLDER     = False
        # mrcnn_config.COCO_CLASSES       = None
        # mrcnn_config.NUM_CLASSES      = len(mrcnn_config.COCO_CLASSES) + 1
        mrcnn_config.VERBOSE            = verbose
        
    # Recreate the model in inference mode
    try :
        del model
        print('delete model is successful')
        gc.collect()
    except: 
        pass
    KB.clear_session()
    mrcnn_model = mrcnn_modellib.MaskRCNN(mode=mode, config=mrcnn_config)

    ##------------------------------------------------------------------------------------
    ## Load Mask RCNN Model Weight file
    ##------------------------------------------------------------------------------------
    mrcnn_model.display_layer_info()
    mrcnn_config.display()     
    
    return [mrcnn_model, mrcnn_config]
"""
 

#######################################################################################    
## NEW SHAPES model preparation
#######################################################################################
# def mrcnn_newshape_train(mode = 'training' , mrcnn_config = None,
                     # batch_sz = 1, epoch_steps = 4, training_folder = "train_newshapes_coco",
                      # verbose = 0):
def mrcnn_newshape_train( mode = 'training', args = None, input_parms = None,  batch_size = 2, mrcnn_config = None, 
                          fcn_weight_file = 'last', fcn_train_dir = "train_fcn8_coco_adam", verbose = 0):
    
    import mrcnn.newshapes as newshapes

    #------------------------------------------------------------------------------------
    # Parse command line arguments
    #------------------------------------------------------------------------------------
    if args is None:
        parser = command_line_parser()
        if input_parms is None:
            input_parms = " --epochs 2 " 
            input_parms +=" --steps_in_epoch 100 "    
            input_parms +=" --val_steps        5 " 
            input_parms +=" --last_epoch       0 "
            input_parms +=" --batch_size "+str(batch_size)+ " " 
            input_parms +=" --lr 0.00001 "

            input_parms +=" --mrcnn_logs_dir train_mrcnn_newshapes "
            input_parms +=" --fcn_logs_dir   train_fcn8_newshapes "
            input_parms +=" --mrcnn_model    last "
            input_parms +=" --fcn_model      init "
            input_parms +=" --opt            adam "
            input_parms +=" --fcn_arch       fcn8 " 
            input_parms +=" --fcn_layers     all " 
            input_parms +=" --sysout        screen "
            input_parms +=" --scale_factor   1 " 
            input_parms +=" --new_log_folder   "        
            input_parms +=" --mrcnn_logs_dir train_mrcnn_newshapes "
            input_parms +=" --mrcnn_model    last "
            input_parms +=" --sysout        screen "
            print(input_parms)
        args = parser.parse_args(input_parms.split())

    utils.display_input_parms(args)    
    #------------------------------------------------------------------------------------
    # setup project directories
    #------------------------------------------------------------------------------------
    paths = Paths(fcn_training_folder = args.fcn_logs_dir, mrcnn_training_folder = args.mrcnn_logs_dir)
    paths.display()
    
    ##------------------------------------------------------------------------------------
    ## Build configuration object , if none has been passed
    ##------------------------------------------------------------------------------------
    if mrcnn_config is None:
        mrcnn_config = newshapes.NewShapesConfig()
        mrcnn_config.NAME                 = 'mrcnn'              
        # mrcnn_config.DIR_DATASET        = paths.DIR_DATASET
        # mrcnn_config.DIR_TRAINING       = paths.DIR_TRAINING
        # mrcnn_config.DIR_PRETRAINED     = paths.DIR_PRETRAINED
        # mrcnn_config.TRAINING_PATH      = os.path.join(paths.DIR_TRAINING, training_folder)
        mrcnn_config.TRAINING_PATH        = paths.MRCNN_TRAINING_PATH
        mrcnn_config.COCO_DATASET_PATH    = paths.COCO_DATASET_PATH 
        mrcnn_config.COCO_MODEL_PATH      = paths.COCO_MODEL_PATH   
        mrcnn_config.RESNET_MODEL_PATH    = paths.RESNET_MODEL_PATH 
        mrcnn_config.VGG16_MODEL_PATH     = paths.VGG16_MODEL_PATH  
        mrcnn_config.SHAPES_MODEL_PATH    = paths.SHAPES_MODEL_PATH   

        mrcnn_config.COCO_CLASSES         = None 
        mrcnn_config.DETECTION_PER_CLASS  = 200
        mrcnn_config.HEATMAP_SCALE_FACTOR = int(args.scale_factor)
        mrcnn_config.BATCH_SIZE           = int(args.batch_size)                  # Batch size is 2 (# GPUs * images/GPU).
        mrcnn_config.IMAGES_PER_GPU       = int(args.batch_size)                  # Must match BATCH_SIZE
        mrcnn_config.STEPS_PER_EPOCH      = int(args.steps_in_epoch)
        mrcnn_config.NEW_LOG_FOLDER       = True
        mrcnn_config.VERBOSE              = verbose
                                          
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
        mrcnn_config.OPTIMIZER            = args.opt.upper()
        mrcnn_config.SYSOUT               = args.sysout
        mrcnn_config.NEW_LOG_FOLDER       = args.new_log_folder
        mrcnn_config.VERBOSE              = verbose
    # create the model in training mode
    try :
        del model
        print('delete model is successful')
        gc.collect()
    except: 
        pass
    KB.clear_session()
    mrcnn_model = mrcnn_modellib.MaskRCNN(mode=mode, config=mrcnn_config)

    # display model layer info
    mrcnn_config.display()             
    mrcnn_model.display_layer_info()
    
    return mrcnn_model
    
 
##------------------------------------------------------------------------------------    
## New Shapes TESTING
##------------------------------------------------------------------------------------    
# def mrcnn_newshapes_test(init_with = 'last', FCN_layers = False, batch_sz = 5, epoch_steps = 4,training_folder= "mrcnn_newshape_test_logs"):

def mrcnn_newshape_test( mode = 'inference', args = None, input_parms = None,  batch_size = 2, mrcnn_config = None, 
                          verbose = 0):
    
    import mrcnn.newshapes as newshapes

    #------------------------------------------------------------------------------------
    # Parse command line arguments
    #------------------------------------------------------------------------------------
    if args is None:
        parser = command_line_parser()
        if input_parms is None:
            input_parms  =" --batch_size "+str(batch_size)+ " " 
            # input_parms +=" --lr 0.00001 "
            # input_parms = " --epochs 2 " 
            # input_parms +=" --steps_in_epoch 100 "    
            # input_parms +=" --val_steps        5 " 
            # input_parms +=" --last_epoch       0 "

            input_parms +=" --mrcnn_logs_dir train_mrcnn_newshapes "
            input_parms +=" --fcn_logs_dir   train_fcn8_newshapes "
            input_parms +=" --mrcnn_model    last "
            # input_parms +=" --fcn_model      init "
            # input_parms +=" --opt            adam "
            # input_parms +=" --fcn_arch       fcn8 " 
            # input_parms +=" --fcn_layers     all " 
            input_parms +=" --sysout        screen "
            input_parms +=" --scale_factor   1 " 
            # input_parms +=" --new_log_folder   "        
            input_parms +=" --mrcnn_logs_dir train_mrcnn_newshapes "
            input_parms +=" --mrcnn_model    last "
            input_parms +=" --sysout        screen "
            print(input_parms)
        args = parser.parse_args(input_parms.split())

    utils.display_input_parms(args)    
    #------------------------------------------------------------------------------------
    # setup project directories
    #------------------------------------------------------------------------------------
    paths = Paths(fcn_training_folder = args.fcn_logs_dir, mrcnn_training_folder = args.mrcnn_logs_dir)
    paths.display()
    
    ##------------------------------------------------------------------------------------
    ## Build configuration object , if none has been passed
    ##------------------------------------------------------------------------------------
    if mrcnn_config is None:
        mrcnn_config = newshapes.NewShapesConfig()
        mrcnn_config.NAME                 = 'mrcnn'              
        # mrcnn_config.DIR_DATASET        = paths.DIR_DATASET
        # mrcnn_config.DIR_TRAINING       = paths.DIR_TRAINING
        # mrcnn_config.DIR_PRETRAINED     = paths.DIR_PRETRAINED
        # mrcnn_config.TRAINING_PATH      = os.path.join(paths.DIR_TRAINING, training_folder)
        mrcnn_config.TRAINING_PATH        = paths.MRCNN_TRAINING_PATH
        mrcnn_config.COCO_DATASET_PATH    = paths.COCO_DATASET_PATH 
        mrcnn_config.COCO_MODEL_PATH      = paths.COCO_MODEL_PATH   
        mrcnn_config.RESNET_MODEL_PATH    = paths.RESNET_MODEL_PATH 
        mrcnn_config.VGG16_MODEL_PATH     = paths.VGG16_MODEL_PATH  
        mrcnn_config.SHAPES_MODEL_PATH    = paths.SHAPES_MODEL_PATH   

        mrcnn_config.COCO_CLASSES         = None 
        mrcnn_config.DETECTION_PER_CLASS  = 200
        mrcnn_config.HEATMAP_SCALE_FACTOR = int(args.scale_factor)
        mrcnn_config.BATCH_SIZE           = int(args.batch_size)                  # Batch size is 2 (# GPUs * images/GPU).
        mrcnn_config.IMAGES_PER_GPU       = int(args.batch_size)                  # Must match BATCH_SIZE
        mrcnn_config.STEPS_PER_EPOCH      = int(args.steps_in_epoch)
        mrcnn_config.NEW_LOG_FOLDER       = True
        mrcnn_config.VERBOSE              = verbose
                                          
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
        mrcnn_config.OPTIMIZER            = args.opt.upper()
        mrcnn_config.SYSOUT               = args.sysout
        mrcnn_config.NEW_LOG_FOLDER       = args.new_log_folder
        mrcnn_config.VERBOSE              = verbose

    # Recreate the model in inference mode
    try :
        del model
        print('delete model is successful')
        gc.collect()
    except: 
        pass
    KB.clear_session()
    mrcnn_model = modellib.MaskRCNN(mode="inference", config=mrcnn_config)
        
    print(' COCO Model Path       : ', COCO_DIR_TRAINING)
    print(' Checkpoint folder Path: ', MODEL_DIR)
    print(' Model Parent Path     : ', DIR_TRAINING)
    print(' Resent Model Path     : ', RESNET_DIR_TRAINING)

    # display model layer info
    mrcnn_config.display()             
    mrcnn_model.display_layer_info()
    
    return mrcnn_model
    

#######################################################################################    
## MRCNN Training pipeline
#######################################################################################
def mrcnn_coco_train(mode = 'training', 
                     batch_sz = 1, epoch_steps = 4, 
                     training_folder = 'train_mrcnn_coco',
                     mrcnn_config = None, verbose = 0):

    ##------------------------------------------------------------------------------------
    ## Build configuration object , if none has been passed
    ##------------------------------------------------------------------------------------
    if mrcnn_config is None:
        paths = Paths()
        paths.display()
        mrcnn_config = CocoConfig()
        mrcnn_config.NAME               = 'mrcnn'              
        mrcnn_config.DIR_DATASET        = paths.DIR_DATASET
        mrcnn_config.DIR_TRAINING       = paths.DIR_TRAINING
        mrcnn_config.DIR_PRETRAINED     = paths.DIR_PRETRAINED
        mrcnn_config.TRAINING_PATH      = paths.MRCNN_TRAINING_PATH
        # mrcnn_config.TRAINING_PATH      = os.path.join(paths.DIR_TRAINING, training_folder)
        mrcnn_config.COCO_DATASET_PATH  = paths.COCO_DATASET_PATH 
        mrcnn_config.COCO_MODEL_PATH    = paths.COCO_MODEL_PATH   
        mrcnn_config.RESNET_MODEL_PATH  = paths.RESNET_MODEL_PATH 
        mrcnn_config.VGG16_MODEL_PATH   = paths.VGG16_MODEL_PATH  

        mrcnn_config.COCO_CLASSES       = None 
        mrcnn_config.DETECTION_PER_CLASS = 200
        mrcnn_config.HEATMAP_SCALE_FACTOR = 4
        mrcnn_config.BATCH_SIZE         = batch_sz                  # Batch size is 2 (# GPUs * images/GPU).
        mrcnn_config.IMAGES_PER_GPU     = batch_sz                  # Must match BATCH_SIZE
        mrcnn_config.STEPS_PER_EPOCH    = epoch_steps
        mrcnn_config.NEW_LOG_FOLDER     = True
        mrcnn_config.VERBOSE            = verbose        

    # Recreate the model in training mode
    try :
        del model
        print('delete model is successful')
        gc.collect()
    except: 
        pass
    KB.clear_session()
    mrcnn_model = mrcnn_modellib.MaskRCNN(mode=mode, config=mrcnn_config)

    mrcnn_model.display_layer_info()
    
    return [mrcnn_model, mrcnn_config]
    
    
    
    
#######################################################################################    
## MRCNN inference pipeline
#######################################################################################
def build_mrcnn_inference_pipeline( args = None, input_parms = None,  batch_size = 2, fcn_weight_file = 'last', verbose = 0):
    '''
    sets up mrcnn model in inference mode
    '''
    start_time = datetime.now().strftime("%m-%d-%Y @ %H:%M:%S")
    print()
    print('--> Execution started at:', start_time)
    print("    Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))

    ##------------------------------------------------------------------------------------
    ## Parse command line arguments
    ##------------------------------------------------------------------------------------
    if args is None:
        parser = command_line_parser()
        if input_parms is None:
            input_parms = " --batch_size "+str(batch_size)+ " " 
            input_parms +=" --mrcnn_logs_dir train_mrcnn_coco "
            # input_parms +=" --fcn_logs_dir " + fcn_train_dir + " " 
            input_parms +=" --mrcnn_model    last "
            # input_parms +=" --fcn_model    " + fcn_weight_file + " "
            # input_parms +=" --fcn_arch       fcn8 " 
            # input_parms +=" --fcn_layers     all " 
            input_parms +=" --sysout        screen "
            input_parms +=" --coco_classes   62 63 67 78 79 80 81 82 72 73 74 75 76 77"
            print(input_parms)
        args = parser.parse_args(input_parms.split())

    utils.display_input_parms(args)    

    #------------------------------------------------------------------------------------
    # if debug is true set stdout destination to stringIO
    #------------------------------------------------------------------------------------
    if args.sysout == 'FILE':
        sys.stdout = io.StringIO()
    
    #------------------------------------------------------------------------------------
    # setup project directories
    #------------------------------------------------------------------------------------
    paths = Paths(fcn_training_folder = args.fcn_logs_dir, mrcnn_training_folder = args.mrcnn_logs_dir)
    paths.display()
    
    #------------------------------------------------------------------------------------
    # Build configuration object 
    #------------------------------------------------------------------------------------                          
    mrcnn_config                      = CocoConfig()
    mrcnn_config.NAME                 = 'mrcnn'              
    mrcnn_config.TRAINING_PATH        = paths.MRCNN_TRAINING_PATH
    mrcnn_config.COCO_DATASET_PATH    = paths.COCO_DATASET_PATH 
    mrcnn_config.COCO_MODEL_PATH      = paths.COCO_MODEL_PATH   
    mrcnn_config.RESNET_MODEL_PATH    = paths.RESNET_MODEL_PATH 
    mrcnn_config.VGG16_MODEL_PATH     = paths.VGG16_MODEL_PATH  
    # mrcnn_config.PRED_CLASS_INFO_PATH = paths.PRED_CLASS_INFO_PATH
    mrcnn_config.class_prediction_avg = utils.load_class_prediction_avg(paths.PRED_CLASS_INFO_PATH)
    mrcnn_config.COCO_CLASSES         = None 

    mrcnn_config.HEATMAP_SCALE_FACTOR = int(args.scale_factor)
    mrcnn_config.BATCH_SIZE           = int(args.batch_size)                  # Batch size is 2 (# GPUs * images/GPU).
    mrcnn_config.IMAGES_PER_GPU       = int(args.batch_size)                  # Must match BATCH_SIZE
    
    
    # mrcnn_config.STEPS_PER_EPOCH    = int(args.steps_in_epoch)
    # mrcnn_config.LEARNING_RATE      = float(args.lr)
    # mrcnn_config.EPOCHS_TO_RUN      = int(args.epochs)
    mrcnn_config.FCN_INPUT_SHAPE      = mrcnn_config.IMAGE_SHAPE[0:2]
    # mrcnn_config.LAST_EPOCH_RAN     = int(args.last_epoch)
    
    mrcnn_config.NEW_LOG_FOLDER       = args.new_log_folder
    # mrcnn_config.SYSOUT             = args.sysout
    mrcnn_config.VERBOSE              = verbose
    
    mrcnn_config.DETECTION_MAX_INSTANCES  = 200
    mrcnn_config.DETECTION_MIN_CONFIDENCE = 0.1
    mrcnn_config.DETECTION_PER_CLASS      = mrcnn_config.DETECTION_MAX_INSTANCES 

    #------------------------------------------------------------------------------------
    #  Build mrcnn Model
    #------------------------------------------------------------------------------------
    
    try :
        del mrcnn_model
        print('delete model is successful')
        gc.collect()
    except: 
        pass
    KB.clear_session()
    mrcnn_model = mrcnn_modellib.MaskRCNN(mode='inference', config=mrcnn_config)
    
    
    # Load MRCNN Model weights  
    # exclude=["mrcnn_class_logits"] # ,"mrcnn_bbox_fc"]   #, "mrcnn_bbox", "mrcnn_mask"])
    mrcnn_model.load_model_weights(init_with = args.mrcnn_model, exclude = None)  
    
    return mrcnn_model
    

    
#######################################################################################    
## MRCNN evaluate pipeline
#######################################################################################
def build_mrcnn_evaluate_pipeline( args = None, input_parms = None, batch_size = 2, verbose = 0):
    '''
    sets up mrcnn model in evaluate mode
    '''
    start_time = datetime.now().strftime("%m-%d-%Y @ %H:%M:%S")
    print()
    print('--> Execution started at:', start_time)
    print("    Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))

    ##------------------------------------------------------------------------------------
    ## Parse command line arguments
    ##------------------------------------------------------------------------------------
    if args is None:
        parser = command_line_parser()
        if input_parms is None:
            input_parms = " --batch_size "+str(batch_size)+ " " 
            input_parms +=" --mrcnn_logs_dir train_mrcnn_coco "
            input_parms +=" --mrcnn_model    last "
            input_parms +=" --sysout        screen "
            print(input_parms)
        args = parser.parse_args(input_parms.split())
    
    utils.display_input_parms(args)   

    #------------------------------------------------------------------------------------
    # if debug is true set stdout destination to stringIO
    #------------------------------------------------------------------------------------
    if args.sysout == 'FILE':
        sys.stdout = io.StringIO()
    
    #------------------------------------------------------------------------------------
    # setup project directories
    #------------------------------------------------------------------------------------
    paths = Paths(mrcnn_training_folder = args.mrcnn_logs_dir)
    paths.display()
    
    #------------------------------------------------------------------------------------
    # Build configuration object 
    #------------------------------------------------------------------------------------                          
    mrcnn_config                      = CocoConfig()
    mrcnn_config.NAME                 = 'mrcnn'              
    mrcnn_config.TRAINING_PATH        = paths.MRCNN_TRAINING_PATH
    mrcnn_config.COCO_DATASET_PATH    = paths.COCO_DATASET_PATH 
    mrcnn_config.COCO_MODEL_PATH      = paths.COCO_MODEL_PATH   
    mrcnn_config.RESNET_MODEL_PATH    = paths.RESNET_MODEL_PATH 
    mrcnn_config.VGG16_MODEL_PATH     = paths.VGG16_MODEL_PATH  
    # mrcnn_config.PRED_CLASS_INFO_PATH = paths.PRED_CLASS_INFO_PATH
    mrcnn_config.COCO_CLASSES         = None 
    mrcnn_config.class_prediction_avg = utils.load_class_prediction_avg(paths.PRED_CLASS_INFO_PATH)

    mrcnn_config.HEATMAP_SCALE_FACTOR = int(args.scale_factor)
    mrcnn_config.BATCH_SIZE           = int(args.batch_size)                  # Batch size is 2 (# GPUs * images/GPU).
    mrcnn_config.IMAGES_PER_GPU       = int(args.batch_size)                  # Must match BATCH_SIZE
    
    mrcnn_config.FCN_INPUT_SHAPE      = mrcnn_config.IMAGE_SHAPE[0:2]
    mrcnn_config.NEW_LOG_FOLDER       = args.new_log_folder
    # mrcnn_config.SYSOUT             = args.sysout
    mrcnn_config.VERBOSE              = verbose
    
    mrcnn_config.DETECTION_MAX_INSTANCES  = 200
    mrcnn_config.DETECTION_MIN_CONFIDENCE = 0.1
    mrcnn_config.DETECTION_PER_CLASS      = mrcnn_config.DETECTION_MAX_INSTANCES 

    #------------------------------------------------------------------------------------
    #  Build mrcnn Model
    #------------------------------------------------------------------------------------
    try :
        del mrcnn_model
        print('delete model is successful')
        gc.collect()
    except: 
        pass
    KB.clear_session()
    mrcnn_model = mrcnn_modellib.MaskRCNN(mode='evaluate', config=mrcnn_config)
    
    
    # Load MRCNN Model weights  
    # exclude=["mrcnn_class_logits"] # ,"mrcnn_bbox_fc"]   #, "mrcnn_bbox", "mrcnn_mask"])
    mrcnn_model.load_model_weights(init_with = args.mrcnn_model, exclude = None)  
    
    return mrcnn_model
    
    
    
    
#######################################################################################    
## FCN Training pipeline
#######################################################################################
def build_fcn_training_pipeline( args = None, input_parms = None,  batch_size = 2, 
                                 fcn_weight_file = 'last', fcn_train_dir = "train_fcn8_coco_adam", verbose = 0):
    start_time = datetime.now().strftime("%m-%d-%Y @ %H:%M:%S")
    print()
    print('--> Execution started at:', start_time)
    print("    Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))

    ##------------------------------------------------------------------------------------
    ## Parse command line arguments
    ##------------------------------------------------------------------------------------
    if args is None:
        parser = command_line_parser()
        if input_parms is None:
            input_parms = " --epochs 2 --steps_in_epoch 32  --last_epoch 0 "
            input_parms +=" --batch_size "+str(batch_size)+ " --lr 0.00001 --val_steps 8 " 
            input_parms +=" --mrcnn_logs_dir train_mrcnn_coco "
            input_parms +=" --fcn_logs_dir " + fcn_train_dir + " " 
            input_parms +=" --mrcnn_model    last "
            input_parms +=" --fcn_model    " + fcn_weight_file + " "
            input_parms +=" --opt            adam "
            input_parms +=" --fcn_arch       fcn8 " 
            input_parms +=" --fcn_layers     all " 
            input_parms +=" --sysout         screen "
            input_parms +=" --coco_classes   62 63 67 78 79 80 81 82 72 73 74 75 76 77"
            input_parms +=" --new_log_folder    "
            # input_parms +="--fcn_model /home/kbardool/models/train_fcn_adagrad/shapes20180709T1732/fcn_shapes_1167.h5"
            print(input_parms)
        args = parser.parse_args(input_parms.split())
    
    utils.display_input_parms(args)   
        
    # if debug is true set stdout destination to stringIO
    #----------------------------------------------------------------------------------------------            
    # debug = False
    if args.sysout == 'FILE':
        sys.stdout = io.StringIO()

    # setup project directories
    #---------------------------------------------------------------------------------
    paths = Paths(fcn_training_folder = args.fcn_logs_dir, mrcnn_training_folder = args.mrcnn_logs_dir)
    paths.display()

    # Build configuration object 
    #------------------------------------------------------------------------------------                          
    mrcnn_config                    = CocoConfig()
    # import mrcnn.new_shapes as new_shapes
    # mrcnn_config = new_shapes.NewShapesConfig()

    mrcnn_config.NAME                 = 'mrcnn'              
    mrcnn_config.TRAINING_PATH        = paths.MRCNN_TRAINING_PATH
    mrcnn_config.COCO_DATASET_PATH    = paths.COCO_DATASET_PATH 
    mrcnn_config.COCO_MODEL_PATH      = paths.COCO_MODEL_PATH   
    mrcnn_config.RESNET_MODEL_PATH    = paths.RESNET_MODEL_PATH 
    mrcnn_config.VGG16_MODEL_PATH     = paths.VGG16_MODEL_PATH  
    mrcnn_config.COCO_CLASSES         = None 
    mrcnn_config.DETECTION_PER_CLASS  = mrcnn_config.DETECTION_MAX_INSTANCES 
    
    mrcnn_config.HEATMAP_SCALE_FACTOR =  int(args.scale_factor)
    mrcnn_config.BATCH_SIZE           = int(args.batch_size)                  # Batch size is 2 (# GPUs * images/GPU).
    mrcnn_config.IMAGES_PER_GPU       = int(args.batch_size)                  # Must match BATCH_SIZE
                                      
    mrcnn_config.STEPS_PER_EPOCH      = int(args.steps_in_epoch)
    mrcnn_config.LEARNING_RATE        = float(args.lr)
    mrcnn_config.EPOCHS_TO_RUN        = int(args.epochs)
    mrcnn_config.FCN_INPUT_SHAPE      = mrcnn_config.IMAGE_SHAPE[0:2]
    mrcnn_config.LAST_EPOCH_RAN       = int(args.last_epoch)
    mrcnn_config.NEW_LOG_FOLDER       = False
    mrcnn_config.SYSOUT               = args.sysout
    mrcnn_config.VERBOSE              = verbose
    # mrcnn_config.WEIGHT_DECAY       = 2.0e-4
    # mrcnn_config.VALIDATION_STEPS   = int(args.val_steps)
    # mrcnn_config.REDUCE_LR_FACTOR   = 0.5
    # mrcnn_config.REDUCE_LR_COOLDOWN = 30
    # mrcnn_config.REDUCE_LR_PATIENCE = 40
    # mrcnn_config.EARLY_STOP_PATIENCE= 80
    # mrcnn_config.EARLY_STOP_MIN_DELTA = 1.0e-4
    # mrcnn_config.MIN_LR             = 1.0e-10
    # mrcnn_config.OPTIMIZER          = args.opt.upper()
    # mrcnn_config.display() 


    #  Build MRCNN Model
    #------------------------------------------------------------------------------------
    try :
        del mrcnn_model
        print('delete model is successful')
        gc.collect()
    except: 
        pass
    KB.clear_session()
    mrcnn_model = mrcnn_modellib.MaskRCNN(mode='trainfcn', config=mrcnn_config)
    
    #------------------------------------------------------------------------------------
    # Build configuration for FCN model
    #------------------------------------------------------------------------------------
    fcn_config = CocoConfig()
    fcn_config.COCO_DATASET_PATH      = paths.COCO_DATASET_PATH 
    fcn_config.COCO_HEATMAP_PATH      = paths.COCO_HEATMAP_PATH 
    # fcn_config.COCO_MODEL_PATH      = COCO_MODEL_PATH   
    # fcn_config.RESNET_MODEL_PATH    = RESNET_MODEL_PATH 
    fcn_config.NAME                   = 'fcn'              
    fcn_config.TRAINING_PATH          = paths.FCN_TRAINING_PATH
    fcn_config.VGG16_MODEL_PATH       = paths.FCN_VGG16_MODEL_PATH
    fcn_config.HEATMAP_SCALE_FACTOR   = mrcnn_config.HEATMAP_SCALE_FACTOR
    fcn_config.FCN_INPUT_SHAPE        = fcn_config.IMAGE_SHAPE[0:2] // mrcnn_config.HEATMAP_SCALE_FACTOR 
                                      
    fcn_config.BATCH_SIZE             = int(args.batch_size)                 # Batch size is 2 (# GPUs * images/GPU).
    fcn_config.IMAGES_PER_GPU         = int(args.batch_size)                   # Must match BATCH_SIZE
    fcn_config.EPOCHS_TO_RUN          = int(args.epochs)
    fcn_config.STEPS_PER_EPOCH        = int(args.steps_in_epoch)
    fcn_config.LAST_EPOCH_RAN         = int(args.last_epoch)
                                      
    fcn_config.LEARNING_RATE          = float(args.lr)
    fcn_config.VALIDATION_STEPS       = int(args.val_steps)
    fcn_config.BATCH_MOMENTUM         = 0.9
    fcn_config.WEIGHT_DECAY           = 2.0e-4     ## FCN Weight decays are 5.0e-4 or 2.0e-4
                                      
    fcn_config.REDUCE_LR_FACTOR       = 0.5
    fcn_config.REDUCE_LR_COOLDOWN     = 15
    fcn_config.REDUCE_LR_PATIENCE     = 50
    fcn_config.REDUCE_LR_MIN_DELTA    = 1e-6
                                      
    fcn_config.EARLY_STOP_PATIENCE    = 150
    fcn_config.EARLY_STOP_MIN_DELTA   = 1.0e-7
                                      
    fcn_config.MIN_LR                 = 1.0e-10
    fcn_config.CHECKPOINT_PERIOD      = 1     
                                      
    fcn_config.TRAINING_LAYERS        = args.fcn_layers
    fcn_config.NEW_LOG_FOLDER         = args.new_log_folder
    fcn_config.OPTIMIZER              = args.opt
    fcn_config.SYSOUT                 = args.sysout
    fcn_config.VERBOSE                = verbose


    ## Build FCN Model in Training Mode
    try :
        del fcn_model
        gc.collect()
    except: 
        pass    
    fcn_model = fcn_modellib.FCN(mode="training", arch = args.fcn_arch, config=fcn_config)

    ## Display model configuration information
    # paths.display()
    # fcn_config.display()  
    print()
    print(' MRCNN IO Layers ')
    print(' --------------- ')
    mrcnn_model.display_layer_info()
    print()
    print(' FCN IO Layers ')
    print(' ------------- ')
    fcn_model.display_layer_info()
    
    # exclude=["mrcnn_class_logits"] # ,"mrcnn_bbox_fc"]   #, "mrcnn_bbox", "mrcnn_mask"])
    mrcnn_model.load_model_weights(init_with = args.mrcnn_model , exclude = None, verbose = verbose)  
    
    # Load FCN Model weights  
    #------------------------------------------------------------------------------------
    if args.fcn_model != 'init':
        fcn_model.load_model_weights(init_with = args.fcn_model, verbose = verbose)
    else:
        print(' FCN Training starting from randomly initialized weights ...')
    
    return mrcnn_model, fcn_model

        

        
#######################################################################################    
## FCN inference pipeline
#######################################################################################
def build_fcn_inference_pipeline( args = None, input_parms = None,  batch_size = 2, fcn_weight_file = 'last', verbose = 0, mode = 'inference'):
    start_time = datetime.now().strftime("%m-%d-%Y @ %H:%M:%S")
    print()
    print('--> Execution started at:', start_time)
    print("    Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))


    ##------------------------------------------------------------------------------------
    ## Parse command line arguments
    ##------------------------------------------------------------------------------------
    if args is None:
        parser = command_line_parser()
        if input_parms is None:
            input_parms = " --batch_size "+str(batch_size)+ " " 
            input_parms +=" --mrcnn_logs_dir train_mrcnn_coco "
            input_parms +=" --fcn_logs_dir " + fcn_train_dir + " " 
            input_parms +=" --mrcnn_model    last "
            input_parms +=" --fcn_model    " + fcn_weight_file + " "
            input_parms +=" --fcn_arch       fcn8 " 
            input_parms +=" --fcn_layers     all " 
            input_parms +=" --sysout        screen "
            input_parms +=" --coco_classes   62 63 67 78 79 80 81 82 72 73 74 75 76 77"
            print(input_parms)
        args = parser.parse_args(input_parms.split())

    utils.display_input_parms(args)   

    print(' *** Keras Training mode:', KB.learning_phase())
    KB.set_learning_phase(0)
    print(' *** Keras Training mode after setting:', KB.learning_phase())

    # if debug is true set stdout destination to stringIO
    #------------------------------------------------------------------------------------
    # debug = False
    if args.sysout == 'FILE':
        sys.stdout = io.StringIO()

    ## setup project directories
    #------------------------------------------------------------------------------------
    paths = Paths(fcn_training_folder = args.fcn_logs_dir, mrcnn_training_folder = args.mrcnn_logs_dir)
    paths.display()

    # Build configuration object 
    #------------------------------------------------------------------------------------                          
    mrcnn_config                      = CocoConfig()
    mrcnn_config.NAME                 = 'mrcnn'              
    mrcnn_config.TRAINING_PATH        = paths.MRCNN_TRAINING_PATH
    mrcnn_config.COCO_DATASET_PATH    = paths.COCO_DATASET_PATH 
    mrcnn_config.COCO_MODEL_PATH      = paths.COCO_MODEL_PATH   
    mrcnn_config.RESNET_MODEL_PATH    = paths.RESNET_MODEL_PATH 
    mrcnn_config.VGG16_MODEL_PATH     = paths.VGG16_MODEL_PATH  
    # mrcnn_config.PRED_CLASS_INFO_PATH = paths.PRED_CLASS_INFO_PATH
    mrcnn_config.COCO_CLASSES         = None 
    mrcnn_config.class_prediction_avg = utils.load_class_prediction_avg(paths.PRED_CLASS_INFO_PATH)


    
    mrcnn_config.HEATMAP_SCALE_FACTOR =  int(args.scale_factor)
    mrcnn_config.BATCH_SIZE           = int(args.batch_size)                  # Batch size is 2 (# GPUs * images/GPU).
    mrcnn_config.IMAGES_PER_GPU       = int(args.batch_size)                  # Must match BATCH_SIZE

    mrcnn_config.DETECTION_MAX_INSTANCES = 200
    mrcnn_config.DETECTION_PER_CLASS  = mrcnn_config.DETECTION_MAX_INSTANCES 
    mrcnn_config.DETECTION_MIN_CONFIDENCE = 0.1
    
    mrcnn_config.FCN_INPUT_SHAPE      = mrcnn_config.IMAGE_SHAPE[0:2]
    mrcnn_config.NEW_LOG_FOLDER       = args.new_log_folder
    mrcnn_config.SYSOUT               = args.sysout
    mrcnn_config.VERBOSE              = verbose
    # mrcnn_config.STEPS_PER_EPOCH    = int(args.steps_in_epoch)
    # mrcnn_config.LEARNING_RATE      = float(args.lr)
    # mrcnn_config.EPOCHS_TO_RUN      = int(args.epochs)
    # mrcnn_config.LAST_EPOCH_RAN     = int(args.last_epoch)


    #------------------------------------------------------------------------------------
    #  Build mrcnn Model
    #------------------------------------------------------------------------------------    
    try :
        del mrcnn_model
        print('delete model is successful')
        gc.collect()
    except: 
        pass
    KB.clear_session()
    mrcnn_model = mrcnn_modellib.MaskRCNN(mode= mode, config=mrcnn_config)


    print(' *** Keras Training mode after setting:', KB.learning_phase())
    
    #------------------------------------------------------------------------------------
    # Build configuration for FCN model
    #------------------------------------------------------------------------------------
    fcn_config = CocoConfig()
    fcn_config.COCO_DATASET_PATH      = paths.COCO_DATASET_PATH 
    fcn_config.COCO_HEATMAP_PATH      = paths.COCO_HEATMAP_PATH 

    fcn_config.NAME                   = 'fcn'              
    fcn_config.TRAINING_PATH          = paths.FCN_TRAINING_PATH
    fcn_config.VGG16_MODEL_PATH       = paths.FCN_VGG16_MODEL_PATH
    fcn_config.HEATMAP_SCALE_FACTOR   = mrcnn_config.HEATMAP_SCALE_FACTOR
                                      
    fcn_config.FCN_INPUT_SHAPE        = fcn_config.IMAGE_SHAPE[0:2] // fcn_config.HEATMAP_SCALE_FACTOR 
    fcn_config.DETECTION_MIN_CONFIDENCE = mrcnn_config.DETECTION_MIN_CONFIDENCE
    fcn_config.DETECTION_MAX_INSTANCES  = mrcnn_config.DETECTION_MAX_INSTANCES 

    fcn_config.BATCH_SIZE             = int(args.batch_size)                 # Batch size is 2 (# GPUs * images/GPU).
    fcn_config.IMAGES_PER_GPU         = int(args.batch_size)                   # Must match BATCH_SIZE

    # fcn_config.COCO_MODEL_PATH      = COCO_MODEL_PATH   
    # fcn_config.RESNET_MODEL_PATH    = RESNET_MODEL_PATH 
    # fcn_config.EPOCHS_TO_RUN        = int(args.epochs)
    # fcn_config.STEPS_PER_EPOCH      = int(args.steps_in_epoch)
    # fcn_config.LAST_EPOCH_RAN       = int(args.last_epoch)
    # fcn_config.LEARNING_RATE        = float(args.lr)
    # fcn_config.VALIDATION_STEPS     = int(args.val_steps)
    
    fcn_config.BATCH_MOMENTUM         = 0.9
    
    # fcn_config.WEIGHT_DECAY         = 2.0e-4
    # fcn_config.REDUCE_LR_FACTOR     = 0.5
    # fcn_config.REDUCE_LR_COOLDOWN   = 5
    # fcn_config.REDUCE_LR_PATIENCE   = 5
    # fcn_config.EARLY_STOP_PATIENCE  = 15
    # fcn_config.EARLY_STOP_MIN_DELTA = 1.0e-4
    # fcn_config.MIN_LR               = 1.0e-10
    # fcn_config.OPTIMIZER            = args.opt     
    fcn_config.NEW_LOG_FOLDER         = args.new_log_folder
    fcn_config.SYSOUT                 = args.sysout
    fcn_config.VERBOSE                = verbose


    #------------------------------------------------------------------------------------
    # Build FCN Model
    #------------------------------------------------------------------------------------
    try :
        del fcn_model
        gc.collect()
    except: 
        pass    
    fcn_model = fcn_modellib.FCN(mode='inference', arch = args.fcn_arch, config=fcn_config)

    print(' *** Keras Training mode after setting:', KB.learning_phase())
    
    print()
    print(' MRCNN IO Layers ')
    print(' --------------- ')
    mrcnn_model.display_layer_info()
    print()
    print(' FCN IO Layers ')
    print(' ------------- ')
    fcn_model.display_layer_info()
    
    #------------------------------------------------------------------------------------
    # Load MRCNN Model weights  
    #------------------------------------------------------------------------------------
    mrcnn_model.load_model_weights(init_with = 'last', exclude = None, verbose = verbose)  
    
    #------------------------------------------------------------------------------------
    # Load FCN Model weights  
    #------------------------------------------------------------------------------------
    fcn_model.load_model_weights(init_with = args.fcn_model, verbose = verbose)
    
    print(' *** Keras Training mode after setting:', KB.learning_phase())

    return mrcnn_model, fcn_model

    
    
    
#######################################################################################    
## FCN evaluate pipeline
#######################################################################################
def build_fcn_evaluate_pipeline( args = None, input_parms = None,  batch_size = 2, fcn_weight_file = 'last', verbose = 0, mode = 'inference'):
    return build_fcn_inference_pipeline( args = args, input_parms = input_parms,  batch_size = batch_size, fcn_weight_file = fcn_weight_file, 
                                         verbose = verbose, mode = 'evaluate')

            


#######################################################################################    
## GET BATCH routines
#######################################################################################
##------------------------------------------------------------------------------------    
## get_image_batch() 
##------------------------------------------------------------------------------------    
def get_image_batch(dataset, image_ids = None, display = False):
    '''
    retrieves images for  a list of image ids, that can be passed to model detect() functions
    '''
    if image_ids is None:
        image_ids = random.choice(dataset.image_ids)    
    images = []
    if not isinstance(image_ids, list):
        image_ids = [image_ids]

    for image_id in image_ids:
        images.append(dataset.load_image(image_id))

    if display:
        log("Loading {} images".format(len(images)))
        for image in images:
            log("image", image)
        titles = ['id: '+str(i)+' ' for i in image_ids]
        visualize.display_images(images, titles = titles)
        
    return images
    


##------------------------------------------------------------------------------------    
## get_training_batch()
##------------------------------------------------------------------------------------    
def get_training_batch(dataset, config, image_ids, display = True, masks = False):
    '''
    retrieves a training batch from list of image ids, that can be passed to the MRCNN model train function
    or the run_pipeline() module
    
    '''
    if not isinstance(image_ids, list):
        image_ids = [image_ids]

    batch_x, _ = data_gen_simulate(dataset, config, image_ids)
    if display:
        visualize.display_training_batch(dataset, batch_x, masks = masks)

    return batch_x
        
        
##------------------------------------------------------------------------------------    
## get_inference_batch() : 
##------------------------------------------------------------------------------------    
def get_inference_batch(dataset, config, image_ids = None, generator = None, display = False):
    '''
    retrieves a list of image ids, that can be passed to model predict() functions
        
        images_ids:          List of images ids, potentially of different sizes.

    returns:
        images              List of images in raw format
        molded_images       [N, h, w, 3]. Images resized and normalized.
        image_metas         [N, length of meta data]. Details about each image.
        windows             [N, (y1, x1, y2, x2)]. The portion of the image that has the
                            original image (padding excluded).
    '''
    assert generator is not None or image_ids is not None, "  generator or image_ids must be passed to get_evaluate_batch()" 
    
    if image_ids is not None:
        if not isinstance(image_ids, list):
            image_ids = [image_ids]
        images = [dataset.load_image(image_id) for image_id in image_ids]       
        batch_x, _ = data_gen_simulate(dataset, config, image_ids)
    elif generator is not None:
        batch_x, _ = next(generator)
        images = [dataset.load_image(image_id) for image_id in batch_x[1][:,0]]       

    else:
        print('ERROR - dataset generator or list of image_ids must be passed to get_evaluate_batch()')
        return

    molded_images = batch_x[0]
    image_metas   = batch_x[1]
        
    if display:
        log("Processing {} images".format(len(images)))
        for image in images:
            log("image", image)
        log("molded_images", molded_images)
        log("image_metas"  , image_metas)
        titles = ['id: '+str(i)+' ' for i in image_metas]
        # visualize.display_images(images, titles = titles)        
        visualize.display_training_batch(dataset, [molded_images, image_metas], masks = True)
        
    return [images, molded_images, image_metas]


                
##------------------------------------------------------------------------------------    
## get_evaluation_batch()
##------------------------------------------------------------------------------------    
def get_evaluate_batch(dataset, config, image_ids = None, generator = None, display = True, masks = False):
    '''
    retrieves image batch from list of image ids or generator, that can be passed to MRCNN model in evaluate
    mode. or the run_pipeline() module
    
    returns
    -------
    List of     [images, molded_images, images_metas, gt_class_ids, gt_bboxes]
    '''
         
    if generator is not None:
        batch_x, _ = next(generator)
        images = [dataset.load_image(image_id) for image_id in batch_x[1][:,0]] 
        
    else: 
        if image_ids is not None:
            if not isinstance(image_ids, list):
                image_ids = [image_ids]
        else:
            image_ids = list(np.random.choice(dataset.image_ids, config.BATCH_SIZE))
            print(' Random selection of images: ' , image_ids)
            
        images = [dataset.load_image(image_id) for image_id in image_ids]   
        batch_x, _ = data_gen_simulate(dataset, config, image_ids)
        
        
    if display:
        visualize.display_training_batch(dataset, batch_x, masks = masks)
        
    #      [images, molded_images, images_metas, gt_class_ids, gt_bboxes]
    return [images, batch_x[0], batch_x[1], batch_x[4], batch_x[5]]
    
    
#######################################################################################    
## RUN PIPELINE  routines
#######################################################################################
##------------------------------------------------------------------------------------    
## Run Training batch through MRCNN model using get_layer_outputs()
##------------------------------------------------------------------------------------    
def run_mrcnn_training_pipeline(mrcnn_model, dataset, mrcnn_input = None, image_ids = None, verbose = 0):
    '''
    returns dictionary of input/outputs
    
    outputs['mrcnn_input']  = mrcnn_input
    outputs['mrcnn_output'] = mrcnn_output
    outputs['image_batch']  = image_batch    
    '''
        
    if mrcnn_input is not None:
        assert len(mrcnn_input) == 6, 'Length of mrcnn_input list must be 6'
        image_batch = mrcnn_input[0]     ## molded images
    elif image_ids is not None:
        if not isinstance(image_ids, list):
            image_ids = [image_ids]
        image_batch = get_image_batch(dataset, image_ids, display = True)
        mrcnn_input = get_training_batch(dataset, mrcnn_model.config, image_ids)
    else:
        print('ERROR - mrcnn_input or image_ids must be passed to inference pipeline')
        return

    print('** Pass through MRCNN model:')      
    mrcnn_output = mrcnn_model.get_layer_outputs( mrcnn_input, training_flag = True, verbose = verbose)

    if verbose :
        print(' mrcnn outputs: ')
        print('----------------')
        print(' length of mrcnn output : ', len(mrcnn_output))
        for i,j in enumerate(mrcnn_output):
            print('mrcnn output ', i, ' shape: ' ,  j.shape)

    outputs = {}
    outputs['mrcnn_input']  = mrcnn_input
    outputs['mrcnn_output'] = mrcnn_output
    outputs['image_batch']  = image_batch
    return outputs 
                

##------------------------------------------------------------------------------------    
## Run Training batch through FCN model using get_layer_outputs()
##------------------------------------------------------------------------------------    
def run_fcn_training_pipeline(fcn_model, fcn_input = None, verbose = 0):
    '''
    fcn_input  = [input_image_meta, pr_hm, pr_hm_scores, gt_hm, gt_hm_scores]
    outputs    = [fcn_hm, fcn_sm, fcn_MSE_loss, fcn_BCE_loss, fcn_scores]     [0,1,2,3,4]
    '''
    assert len(fcn_input) == 5, 'Length of Fcn_input list must be 5'
    
    print('\n** Pass through FCN model:')    
    fcn_output = fcn_model.get_layer_outputs(fcn_input, verbose = verbose)
    
    if verbose :
        print('\n fcn outputs: ')
        print('----------------')
        for i,j in enumerate(fcn_input):
            print('fcn input ', i, ' shape: ' ,  j.shape)
        for i,j in enumerate(fcn_output):
            print('fcn output ', i, ' shape: ' ,  j.shape)        
    
    outputs = {}
    # outputs['mrcnn_input']  = mrcnn_input
    # outputs['mrcnn_output'] = mrcnn_output
    # outputs['image_batch']  = image_batch
    outputs['fcn_input']    = fcn_input
    outputs['fcn_output']   = fcn_output
    return outputs 
                
    
##------------------------------------------------------------------------------------    
## Run Input batch or Image Ids through FCN inference pipeline using get_layer_outputs()
##------------------------------------------------------------------------------------    
def run_full_training_pipeline(mrcnn_model, fcn_model, dataset, mrcnn_input = None, image_ids = None, verbose = 0):
    '''
    Run a batch of training data through mrcnn-->fcn-->outputs
    Output is a dictionary of inputs and outputs:
    
    outputs['image_batch']  = image_batch       from mrcnn_training_pipeline
    outputs['mrcnn_input']  = mrcnn_input       from mrcnn_training_pipeline
    outputs['mrcnn_output'] = mrcnn_output      from mrcnn_training_pipeline
    outputs['fcn_input']                        input to fcn_training 
    outputs['fcn_output']                       output from fcn_training

    '''
 
    outputs = run_mrcnn_training_pipeline(mrcnn_model, dataset, mrcnn_input = mrcnn_input, image_ids = image_ids, verbose = verbose)  
    
    # build _fcn_input as:  
    #    inputs  = [input_image_meta, pr_hm, pr_hm_scores, gt_hm, gt_hm_scores]
    # outputs returned are:
    #    outputs = [fcn_hm, fcn_sm, fcn_MSE_loss, fcn_BCE_loss, fcn_scores]
    
    
    _fcn_input=  [mrcnn_input[1]]
    _fcn_input.extend(outputs['mrcnn_output'][:4])
    
    fcn_output = run_fcn_training_pipeline( fcn_model, _fcn_input, verbose = verbose)
    
    outputs.update(fcn_output)
    # Append addtional data to output
    # outputs['fcn_input']    = _fcn_input
    # outputs['fcn_output']   = fcn_output['fcn_output']
    return outputs 
    
    
##------------------------------------------------------------------------------------    
## Run Evaluate batch through MRCNN model using get_layer_outputs()
##------------------------------------------------------------------------------------    
def run_mrcnn_evaluate_pipeline(mrcnn_model, dataset, input = None, image_ids = None, verbose = 0):
    '''
    mrcnn_input       [ images, molded_images, image_metas, gt_class_ids, gt_bboxes]
    '''
    if verbose != mrcnn_model.config.VERBOSE:
        print('change mrcnn_model.config.VERBOSE from ', mrcnn_model.config.VERBOSE, ' to ', verbose)
        mrcnn_model.config.VERBOSE = verbose

    outputs = {}
    if input is not None:
       assert len(input) == 5, 'Length of mrcnn_input list must be 5 from get_evaluate_batch()'
    elif image_ids is not None:
        # image_batch = get_image_batch(dataset, image_ids, display = True)
        input = get_evaluate_batch(dataset, mrcnn_model.config, image_ids, display = verbose)
    else:
        print('ERROR - mrcnn_input or image_ids must be passed to inference pipeline')
        return
        
    image_batch = input[0]
    mrcnn_input = input[1:]
    
    if verbose:
        print('** Pass through MRCNN model:') 
    mrcnn_output = mrcnn_model.get_layer_outputs(mrcnn_input, training_flag = True, verbose = verbose)

    if verbose :
        print(' mrcnn outputs: ')
        print('----------------')
        print(' length of mrcnn output : ', len(mrcnn_output))
        for i,j in enumerate(mrcnn_output):
            print('mrcnn output ', i, ' shape: ' ,  j.shape)
    
    outputs['image_batch']  = image_batch
    outputs['mrcnn_input']  = mrcnn_input
    outputs['mrcnn_output'] = mrcnn_output

    return outputs 
                
    
##------------------------------------------------------------------------------------    
## Run Training batch through MRCNN model using get_layer_outputs()
##------------------------------------------------------------------------------------    
def run_mrcnn_inference_pipeline(mrcnn_model, dataset, input = None, image_ids = None, verbose = 0):
    '''
    input:       List of [ images, molded_images, images_meta]
                 images:  list of images in raw format
    
    '''
    assert input is not None or image_ids is not None, " input batch  or image_ids must be passed to inference pipeline"
    
    if input is not None:
        assert len(input) == 3, 'Length of mrcnn_input list must be 3 (output from get_inference_batch()'
        image_batch = input[0] 
        mrcnn_input = input[1:]
    elif image_ids is not None:
        if not isinstance(image_ids, list):
            image_ids = [image_ids]
        image_batch = get_image_batch(dataset, image_ids, display = (not verbose))
        mrcnn_input = get_inference_batch(dataset, mrcnn_model.config, image_ids)
        
    print('** Pass through MRCNN model:')  
    mrcnn_output = mrcnn_model.get_layer_outputs( mrcnn_input, training_flag = False , verbose = verbose)

    if verbose :
        print()
        print('mrcnn outputs: ')
        print('----------------')
        print(' length of mrcnn output : ', len(mrcnn_output))
        for i,j in enumerate(mrcnn_output):
            print('mrcnn output ', i, ' shape: ' ,  j.shape)

    outputs = {}
    outputs['image_batch']  = image_batch
    outputs['mrcnn_input']  = mrcnn_input
    outputs['mrcnn_output'] = mrcnn_output

    return outputs     
            
##------------------------------------------------------------------------------------    
## Run Input batch or Image Ids through FCN inference pipeline using get_layer_outputs()
##------------------------------------------------------------------------------------    
def run_fcn_inference_pipeline(fcn_model, input , verbose = 0):
    '''
    fcn_input:   List of [ pr_hm, pr_hm_scores]
    '''
    assert len(input) == 2, 'Length of input list must be 2 [pr_hm, pr_hm_scores]'

    print('** Pass through FCN model:')  
    
    fcn_output = fcn_model.get_layer_outputs( input, training_flag = True,  verbose = verbose)
    if verbose :
        print()
        print('fcn outputs: ')
        print('----------------')
        for i,j in enumerate(input):
            print('fcn input ', i, ' shape: ' ,  j.shape)
        for i,j in enumerate(fcn_output):
            print('fcn output ', i, ' shape: ' ,  j.shape)        
            
    outputs = {}
    outputs['fcn_input']    = input
    outputs['fcn_output']   = fcn_output
    return outputs 

    
    
        
##------------------------------------------------------------------------------------    
## Run Input batch or Image Ids through FCN inference pipeline using get_layer_outputs()
##------------------------------------------------------------------------------------    
def run_full_inference_pipeline(mrcnn_model, fcn_model, dataset, input = None, image_ids = None, verbose = 0):
    '''
    input:       List of [images, molded_images, images_meta]  produced by get_inference_batch()
                 images:  list of images in raw format
    '''
    assert len(input) == 3, 'Length of input list must be 3 [images, molded_images, images_meta] '
 
    outputs = run_mrcnn_inference_pipeline(mrcnn_model, dataset,input = input, image_ids = image_ids, verbose = verbose)  
    
    _fcn_input=  [outputs['mrcnn_output'][4], outputs['mrcnn_output'][5]]
    
    fcn_output = run_fcn_inference_pipeline( fcn_model, _fcn_input, verbose = verbose)
    
    # Append addtional data to output
    outputs['fcn_input']    = _fcn_input
    outputs['fcn_output']   = fcn_output['fcn_output']
    return outputs 
    
    
                
##------------------------------------------------------------------------------------    
## Run MRCNN detection on MRCNN Inference Pipeline
##------------------------------------------------------------------------------------            
def run_mrcnn_detection(mrcnn_model, dataset, image_ids = None, verbose = 0, display = False):
    '''
    Runs the fcn detection for on input list of image ids.

        images_ids:          List of images ids, potentially of different sizes.

    '''
    if image_ids is None:
        image_ids = random.choice(dataset.image_ids)

    if not isinstance(image_ids, list):
        image_ids = [image_ids]
        
    images = [dataset.load_image(image_id) for image_id in image_ids]
    
    if display:
        log("Display  {} images".format(len(images)))
        for image in images:
            log("image", image)
        titles = ['id: '+str(i)+' ' for i in image_ids]
        visualize.display_images(images, titles = titles)
    
    if verbose:
        for image_id in image_ids :
            print("Image Id  : {}     External Id: {}.{}     Image Reference: {}".format( image_id, 
                dataset.image_info[image_id]["source"], dataset.image_info[image_id]["id"], dataset.image_reference(image_id)))

    # Run object detection on list of images
    
    mrcnn_results = mrcnn_model.detect(images, verbose= verbose)
    
    if verbose:
        print('===> Return from mrcnn_model.detect() :  mrcnn_results', type(mrcnn_results), len(mrcnn_results), ' len(image_ids):', len(image_ids))

    for image_id, result in zip(image_ids, mrcnn_results):
        _, image_meta, gt_class_id, gt_bbox =\
            load_image_gt(dataset, mrcnn_model.config, image_id, use_mini_mask=False)
        result['orig_image_meta'] = image_meta
        result['gt_bboxes']       = gt_bbox
        result['gt_class_ids']    = gt_class_id

    # Display results   
    if verbose:
        np.set_printoptions(linewidth=180,precision=4,threshold=10000, suppress = True)
        print('Image meta: ', image_meta[:10])    
        print(' Length of results from MRCNN detect: ', len(mrcnn_results))
        r = mrcnn_results[0]
        print('mrcnn_results keys: ')
        print('--------------------')
        for i in sorted(r.keys()):
            print('   {:.<25s}  {}'.format(i , r[i].shape))        
        print()
        
        
    return mrcnn_results     
    

##------------------------------------------------------------------------------------    
## Run FCN detection on a list of image ids
##------------------------------------------------------------------------------------            
def run_fcn_detection(fcn_model, mrcnn_model, dataset, image_ids = None, verbose = 0):
    '''
    Runs the fcn detection for on input list of image ids.

        images_ids:          List of images ids, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
            N : NUMBER OF DETECTIONS 
            M : NUMBER OF GROUND TRUTH ANNOTATIONS (BBOXES/CLASSES)
            
           class_ids................  (N,)
           detections...............  (200, 6)
           fcn_hm...................  (256, 256, 81)
           fcn_scores...............  (N, 23)
           fcn_scores_by_class......  (81, 200, 23)
           fcn_sm...................  (256, 256, 81)
           gt_bbox..................  (M, 4)
           gt_class_id..............  (M,)
           image....................  (640, 480, 3)
           image_meta...............  (89,)
           molded_image.............  (1024, 1024, 3)
           molded_rois..............  (N, 4)
           orig_image_meta..........  (89,)
           pr_hm....................  (256, 256, 81)
           pr_hm_scores.............  (81, 200, 23)
           pr_scores................  (N, 23)
           pr_scores_by_class.......  (81, 200, 23)
           rois.....................  (N, 4)
           scores...................  (N,)            
            
            
    '''
    

    
    if image_ids is None:
        image_ids = random.choice(dataset.image_ids)
    else:
        if not isinstance(image_ids, list):
            image_ids = [image_ids]

    images       = [dataset.load_image(image_id) for image_id in image_ids]
    
    if verbose:
        for image_id in image_ids :
            print("Image Id  : {}     External Id: {}.{}     Image Reference: {}".format( image_id, 
                dataset.image_info[image_id]["source"], dataset.image_info[image_id]["id"], dataset.image_reference(image_id)))

    # Run object detection

    fcn_results = fcn_model.detect_from_images(mrcnn_model, images, verbose= verbose)

    # add GT information 
    
    print('===> Return from fcn_model.detect_from_images() : ', type(fcn_results), len(fcn_results), len(image_ids))

    for image_id, result in zip(image_ids, fcn_results):
        _, image_meta, gt_class_id, gt_bbox =\
            load_image_gt(dataset, mrcnn_model.config, image_id, use_mini_mask=False)
        print('Image meta: ', image_meta[:10])    
        
        result['orig_image_meta'] = image_meta
        result['gt_bboxes']         = gt_bbox
        result['gt_class_ids']     = gt_class_id

    # Display results
    if verbose:
        np.set_printoptions(linewidth=180,precision=4,threshold=10000, suppress = True)
        print(' Length of results from MRCNN detect: ', len(fcn_results))
        r = fcn_results[0]
        print('fcn_results keys: ')
        print('--------------------')
        for i in sorted(r.keys()):
            print('   {:.<25s}  {}'.format(i , r[i].shape))        
        print()
 
    return fcn_results

#######################################################################################    
## OLD SHAPES model preparation
#######################################################################################
def mrcnn_oldshapes_train(init_with = None, FCN_layers = False, batch_sz = 5, epoch_steps = 4, training_folder= "mrcnn_oldshape_training_logs"):
    import mrcnn.shapes    as shapes
    MODEL_DIR = os.path.join(DIR_TRAINING, training_folder)

    # Build configuration object -----------------------------------------------
    config = shapes.ShapesConfig()
    config.BATCH_SIZE      = batch_sz                  # Batch size is 2 (# GPUs * images/GPU).
    config.IMAGES_PER_GPU  = batch_sz                  # Must match BATCH_SIZE
    config.STEPS_PER_EPOCH = epoch_steps
    config.FCN_INPUT_SHAPE = config.IMAGE_SHAPE[0:2]

    try :
        del model
        print('delete model is successful')
        gc.collect()
    except: 
        pass
    KB.clear_session()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR, FCN_layers = FCN_layers)

    print(' COCO Model Path       : ', COCO_DIR_TRAINING)
    print(' Checkpoint folder Path: ', MODEL_DIR)
    print(' Model Parent Path     : ', DIR_TRAINING)
    print(' Resent Model Path     : ', RESNET_DIR_TRAINING)

    model.load_model_weights(init_with = init_with)

    # Build shape dataset        -----------------------------------------------
    dataset_train = shapes.ShapesDataset(config)
    dataset_train.load_shapes(3000) 
    dataset_train.prepare()

    # Validation dataset
    dataset_val  = shapes.ShapesDataset(config)
    dataset_val.load_shapes(500)
    dataset_val.prepare()
    
    train_generator = data_generator(dataset_train, model.config, shuffle=True,
                                     batch_size=model.config.BATCH_SIZE,
                                     augment = False)
    val_generator = data_generator(dataset_val, model.config, shuffle=True, 
                                    batch_size=model.config.BATCH_SIZE,
                                    augment=False)                                 
    model.config.display()     
    return [model, dataset_train, dataset_val, train_generator, val_generator, config]                                 


##------------------------------------------------------------------------------------    
## Old Shapes TESTING
##------------------------------------------------------------------------------------    
def mrcnn_oldshapes_test(init_with = None, FCN_layers = False, batch_sz = 5, epoch_steps = 4, training_folder= "mrcnn_oldshape_test_logs"):
    import mrcnn.shapes as shapes
    MODEL_DIR = os.path.join(DIR_TRAINING, training_folder)
    # MODEL_DIR = os.path.join(DIR_TRAINING, "mrcnn_development_logs")

    # Build configuration object -----------------------------------------------
    config = shapes.ShapesConfig()
    config.BATCH_SIZE      = batch_sz                  # Batch size is 2 (# GPUs * images/GPU).
    config.IMAGES_PER_GPU  = batch_sz                  # Must match BATCH_SIZE
    config.STEPS_PER_EPOCH = epoch_steps
    config.FCN_INPUT_SHAPE = config.IMAGE_SHAPE[0:2]

    # Build shape dataset        -----------------------------------------------
    dataset_test = shapes.ShapesDataset(config)
    dataset_test.load_shapes(500) 
    dataset_test.prepare()

    # Recreate the model in inference mode
    try :
        del model
        print('delete model is successful')
        gc.collect()
    except: 
        pass
    KB.clear_session()
    model = modellib.MaskRCNN(mode="inference", 
                              config=config,
                              model_dir=MODEL_DIR, 
                              FCN_layers = FCN_layers )
        
    print(' COCO Model Path       : ', COCO_DIR_TRAINING)
    print(' Checkpoint folder Path: ', MODEL_DIR)
    print(' Model Parent Path     : ', DIR_TRAINING)
    print(' Resent Model Path     : ', RESNET_DIR_TRAINING)

    model.load_model_weights(init_with = init_with)

    test_generator = data_generator(dataset_test, model.config, shuffle=True,
                                     batch_size=model.config.BATCH_SIZE,
                                     augment = False)
    model.config.display()     
    return [model, dataset_test, test_generator, config]                                 

    
                    