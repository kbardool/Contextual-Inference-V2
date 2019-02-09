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
import mrcnn.newshapes as newshapes

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

    
#######################################################################################    
## Build COCO configuration object
#######################################################################################

def build_coco_config( model = None, mode = 'training', args = None, verbose = 0):
    assert model in ['mrcnn', 'fcn']
    assert mode  in ['training' , 'trainfcn', 'inference', 'evaluate']
    assert int(args.scale_factor) == 4 , 'Scaling factor is not 4 for coco dataset: {}'.format(args.scale_factor)
    assert args.evaluate_method in [1,2,3],'Invalid evaluate_method : {} '.format(args.evaluate_method)
    
    paths = Paths(training_folder = 'models_coco',
                  fcn_training_folder = args.fcn_logs_dir, 
                  mrcnn_training_folder = args.mrcnn_logs_dir)
    
    if verbose:
        utils.display_input_parms(args)    
        paths.display()
    
    config                         = CocoConfig()
    config.NAME                    = model
    config.DIR_DATASET             = paths.DIR_DATASET
    config.DIR_TRAINING            = paths.DIR_TRAINING
    config.DIR_PRETRAINED          = paths.DIR_PRETRAINED
    config.COCO_DATASET_PATH       = paths.COCO_DATASET_PATH 
    config.COCO_MODEL_PATH         = paths.COCO_MODEL_PATH   
    config.RESNET_MODEL_PATH       = paths.RESNET_MODEL_PATH 

                                   
    config.HEATMAP_SCALE_FACTOR    = int(args.scale_factor)
    config.BATCH_SIZE              = int(args.batch_size)                  # Batch size is 2 (# GPUs * images/GPU).
    config.IMAGES_PER_GPU          = int(args.batch_size)                  # Must match BATCH_SIZE
    config.FCN_INPUT_SHAPE         = config.IMAGE_SHAPE[0:2] // config.HEATMAP_SCALE_FACTOR 
            
    config.DETECTION_MAX_INSTANCES = config.TRAIN_ROIS_PER_IMAGE
    config.DETECTION_MIN_CONFIDENCE = 0.1

    config.DETECTION_PER_CLASS     = config.DETECTION_MAX_INSTANCES 
    config.SYSOUT                  = args.sysout
    config.VERBOSE                 = verbose
    
    if mode =='evaluate':
        config.PRED_CLASS_INFO_PATH    = os.path.join(config.DIR_PRETRAINED  , "coco_class_stats_info.pkl")
        # config.PRED_CLASS_INFO_PATH = paths.PRED_CLASS_INFO_PATH
        config.EVALUATE_METHOD      = args.evaluate_method        

    if mode in ['training' , 'trainfcn'] :
        config.EPOCHS_TO_RUN      = int(args.epochs)
        config.LAST_EPOCH_RAN     = int(args.last_epoch)    
        config.STEPS_PER_EPOCH    = int(args.steps_in_epoch)
        config.VALIDATION_STEPS   = int(args.val_steps)
        config.LEARNING_RATE      = float(args.lr)
        config.NEW_LOG_FOLDER     = args.new_log_folder
    else:
        config.NEW_LOG_FOLDER     = False
        
    if model == "mrcnn":
        config.TRAINING_PATH      = paths.MRCNN_TRAINING_PATH
        config.VGG16_MODEL_PATH   = paths.VGG16_MODEL_PATH  
        
        if mode == 'training':
            config.WEIGHT_DECAY         = 2.0e-4
            config.REDUCE_LR_FACTOR     = 0.5
            config.REDUCE_LR_COOLDOWN   = 30
            config.REDUCE_LR_PATIENCE   = 60
            config.EARLY_STOP_PATIENCE  = 120
            config.EARLY_STOP_MIN_DELTA = 1.0e-4
            config.MIN_LR               = 1.0e-10
            config.OPTIMIZER            = args.opt
    
    elif model == 'fcn':
        config.TRAINING_PATH      = paths.FCN_TRAINING_PATH
        config.COCO_HEATMAP_PATH  = paths.COCO_HEATMAP_PATH 
        config.VGG16_MODEL_PATH   = paths.FCN_VGG16_MODEL_PATH

        if mode == 'training':
        
        
            config.WEIGHT_DECAY         = 1.0e-6     ## FCN Weight decays are 5.0e-4 or 2.0e-4 changed to 1e-6 15-12-2018
            config.BATCH_MOMENTUM       = 0.9

            config.REDUCE_LR_FACTOR     = 0.5
            config.REDUCE_LR_COOLDOWN   = 50
            config.REDUCE_LR_PATIENCE   = 350
            config.REDUCE_LR_MIN_DELTA  = 1e-6
            
            config.EARLY_STOP_PATIENCE  = 1000 
            config.EARLY_STOP_MIN_DELTA = 1.0e-7
            
            config.MIN_LR               = 1.0e-10
            config.CHECKPOINT_PERIOD    = 1
        
            config.TRAINING_LAYERS      = args.fcn_layers
            config.TRAINING_LOSSES      = args.fcn_losses
            config.OPTIMIZER            = args.opt
            config.FCN_BCE_LOSS_METHOD  = args.fcn_bce_loss_method
            config.FCN_BCE_LOSS_CLASS   = args.fcn_bce_loss_class

    if verbose: 
        config.display()
        
    return config

    
#######################################################################################    
## Build NEWSHAPES configuration object
#######################################################################################
def build_newshapes_config( model = None, mode = 'training', args = None, verbose = 0):
    assert model in ['mrcnn', 'fcn']
    assert mode  in ['training' , 'trainfcn', 'inference', 'evaluate']
    assert int(args.scale_factor) == 1 , 'Scaling factor is not 1 for newshapes dataset: {}'.format(args.scale_factor)
    assert args.evaluate_method in [1,2,3],'Invalid evaluate_method : {} '.format(args.evaluate_method)
    
    paths = Paths(training_folder = 'models_newshapes',
                  fcn_training_folder = args.fcn_logs_dir, 
                  mrcnn_training_folder = args.mrcnn_logs_dir)

    
    if verbose:
        utils.display_input_parms(args)    
        paths.display()
    
    config = newshapes.NewShapesConfig()
    config.NAME                    = model
    config.DIR_DATASET             = paths.DIR_DATASET
    config.DIR_TRAINING            = paths.DIR_TRAINING
    config.DIR_PRETRAINED          = paths.DIR_PRETRAINED
    config.RESNET_MODEL_PATH       = paths.RESNET_MODEL_PATH 

    config.HEATMAP_SCALE_FACTOR    = int(args.scale_factor)
    config.BATCH_SIZE              = int(args.batch_size)                  # Batch size is 2 (# GPUs * images/GPU).
    config.IMAGES_PER_GPU          = int(args.batch_size)                  # Must match BATCH_SIZE
    config.FCN_INPUT_SHAPE         = config.IMAGE_SHAPE[0:2] // config.HEATMAP_SCALE_FACTOR 
            

    config.DETECTION_MAX_INSTANCES  = 64
    config.DETECTION_PER_CLASS      = config.DETECTION_MAX_INSTANCES             
    config.DETECTION_MIN_CONFIDENCE = 0.1

    config.SYSOUT                  = args.sysout
    config.VERBOSE                 = verbose
    
    if mode =='evaluate':
        config.PRED_CLASS_INFO_PATH      = os.path.join(config.DIR_PRETRAINED  , "newshapes_class_stats_info.pkl")
        # config.PRED_CLASS_INFO_PATH = paths.PRED_CLASS_INFO_PATH
        config.EVALUATE_METHOD      = args.evaluate_method 
        
    if mode in ['training' , 'trainfcn'] :
        config.EPOCHS_TO_RUN      = int(args.epochs)
        config.LAST_EPOCH_RAN     = int(args.last_epoch)    
        config.STEPS_PER_EPOCH    = int(args.steps_in_epoch)
        config.VALIDATION_STEPS   = int(args.val_steps)
        config.LEARNING_RATE      = float(args.lr)
        config.NEW_LOG_FOLDER     = args.new_log_folder
    else:
        config.NEW_LOG_FOLDER     = False
        
    if model == "mrcnn":
        config.TRAINING_PATH      = paths.MRCNN_TRAINING_PATH
        config.VGG16_MODEL_PATH   = paths.VGG16_MODEL_PATH  
        config.RESNET_MODEL_PATH    = paths.RESNET_MODEL_PATH 
        config.VGG16_MODEL_PATH     = paths.VGG16_MODEL_PATH  
        config.SHAPES_MODEL_PATH    = paths.SHAPES_MODEL_PATH   
        
        if mode == 'training':
            
            config.WEIGHT_DECAY         = 1.0e-4
            config.REDUCE_LR_FACTOR     = 0.5
            config.REDUCE_LR_COOLDOWN   = 30
            config.REDUCE_LR_PATIENCE   = 60
            config.EARLY_STOP_PATIENCE  = 120
            config.EARLY_STOP_MIN_DELTA = 1.0e-4
            config.MIN_LR               = 1.0e-10
            config.OPTIMIZER            = args.opt
            # config.DETECTION_MAX_INSTANCES  = None
            # config.DETECTION_PER_CLASS      = None
            # config.DETECTION_MIN_CONFIDENCE = None
            
        elif mode == 'inference':
            pass
            # config.DETECTION_MAX_INSTANCES  = 64
            # config.DETECTION_PER_CLASS      = config.DETECTION_MAX_INSTANCES             
            # config.DETECTION_MIN_CONFIDENCE = 0.1
        
    elif model == 'fcn':

        config.TRAINING_PATH      = paths.FCN_TRAINING_PATH
        config.VGG16_MODEL_PATH   = paths.FCN_VGG16_MODEL_PATH

        if mode == 'training':
            config.WEIGHT_DECAY         = 1.0e-6     ## FCN Weight decays are 5.0e-4 or 2.0e-4 changed to 1e-6 15-12-2018
            config.BATCH_MOMENTUM       = 0.9

            config.REDUCE_LR_FACTOR     = 0.5
            config.REDUCE_LR_COOLDOWN   = 50
            config.REDUCE_LR_PATIENCE   = 350
            config.REDUCE_LR_MIN_DELTA  = 1e-6
            
            config.EARLY_STOP_PATIENCE  = 500
            config.EARLY_STOP_MIN_DELTA = 1.0e-7
            
            config.MIN_LR               = 1.0e-10
            config.CHECKPOINT_PERIOD    = 1
        
            config.TRAINING_LAYERS      = args.fcn_layers
            config.TRAINING_LOSSES      = args.fcn_losses
            config.OPTIMIZER            = args.opt
            config.FCN_BCE_LOSS_METHOD  = args.fcn_bce_loss_method
            config.FCN_BCE_LOSS_CLASS   = args.fcn_bce_loss_class


    return config

    
    
##------------------------------------------------------------------------------------    
## NEWSHAPES - MRCNN Training
##------------------------------------------------------------------------------------    
def  build_mrcnn_training_pipeline_newshapes( args = None, mrcnn_config = None,  mode = 'training',verbose = 0):
    
    if mrcnn_config is None :
        mrcnn_config = build_newshapes_config('mrcnn','training', args, verbose = verbose)
            
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
    
    # if args.mrcnn_model == 'coco':
        # exclude=["mrcnn_class_logits", "mrcnn_bbox_fc"]   #, "mrcnn_bbox", "mrcnn_mask"])
    # exclude = []
    # mrcnn_model.load_model_weights(init_with = args.mrcnn_model , exclude = exclude, verbose = verbose)  
    ##------------------------------------------------------------------------------------
    ## Load Mask RCNN Model Weight file
    ##------------------------------------------------------------------------------------
    # exclude_list = ["mrcnn_class_logits"]
    if args.mrcnn_model in [ 'coco', 'init']:
        args.mrcnn_model = 'coco'
        exclude_list = ["mrcnn_class_logits", "mrcnn_bbox_fc"]
        mrcnn_model.load_model_weights(init_with = args.mrcnn_model, exclude = exclude_list, verbose = 1)
    else:
        exclude_list = []
        mrcnn_model.load_model_weights(init_with = args.mrcnn_model, exclude = exclude_list, verbose = 1)

    return mrcnn_model

 
##------------------------------------------------------------------------------------    
## NEWSHAPES - MRCNN Inference
##------------------------------------------------------------------------------------    
def build_mrcnn_inference_pipeline_newshapes(args = None, mrcnn_config = None,  mode = 'inference', 
                                             verbose = 0):
    start_time = datetime.now().strftime("%m-%d-%Y @ %H:%M:%S")
    print()
    print('--> Execution started at:', start_time)
    print("    Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))
    print('    Build_mrcnn_inference_pipeline_newshapes MODE is :', mode)
    utils.display_input_parms(args)    
    
    if mrcnn_config is None :
        mrcnn_config = build_newshapes_config('mrcnn', mode , args, verbose = verbose)

    # Recreate the model in inference mode
    try :
        del model
        print('delete model is successful')
        gc.collect()
    except: 
        pass
    KB.clear_session()
    mrcnn_model = mrcnn_modellib.MaskRCNN(mode= mode, config=mrcnn_config)
        
    # display model layer info
    mrcnn_config.display()             
    mrcnn_model.display_layer_info()
    
    # Load MRCNN Model weights  
    # exclude=["mrcnn_class_logits"] # ,"mrcnn_bbox_fc"]   #, "mrcnn_bbox", "mrcnn_mask"])
    # exclude_list = ["mrcnn_class_logits"]
    if args.mrcnn_model in [ 'coco', 'init']:
        args.mrcnn_model = 'coco'
        exclude_list = ["mrcnn_class_logits", "mrcnn_bbox_fc"]
        mrcnn_model.load_model_weights(init_with = args.mrcnn_model, exclude = exclude_list, verbose = 1)
    else:
        exclude_list = []
        mrcnn_model.load_model_weights(init_with = args.mrcnn_model, exclude = exclude_list, verbose = 1)


    return mrcnn_model

    
##------------------------------------------------------------------------------------    
## NEWSHAPES - MRCNN Evaluate 
##------------------------------------------------------------------------------------    
def build_mrcnn_evaluate_pipeline_newshapes(args, mrcnn_config = None , mode = 'evaluate', 
                                            verbose = 0):      
   
    return build_mrcnn_inference_pipeline_newshapes( args = args, 
                                         mrcnn_config = mrcnn_config, 
                                         mode = mode, verbose = verbose)
   
##------------------------------------------------------------------------------------    
## NEWSHAPES - FCN Training 
##------------------------------------------------------------------------------------    
def build_fcn_training_pipeline_newshapes( args = None, mrcnn_config = None,  mode = 'training', 
                                           verbose = 0):

    print('MODE IS:' , mode)    
    ## Build Mask RCNN Model in TRAINFCN mode
    mrcnn_model  = build_mrcnn_training_pipeline_newshapes( mode = 'trainfcn', args = args, verbose = 0)

    
    fcn_config = build_newshapes_config('fcn','training', args, verbose = verbose)
    ## Build FCN Model in Training Mode
    try :
        del fcn_model
        gc.collect()
    except: 
        pass    
    fcn_model = fcn_modellib.FCN(mode="training", arch = args.fcn_arch, config=fcn_config)

    ## Display model configuration information
    fcn_config.display()             
    fcn_model.display_layer_info()
    
    ## Load FCN Model weights  
    if args.fcn_model != 'init':
        fcn_model.load_model_weights(init_with = args.fcn_model, verbose = verbose)
    else:
        print(' FCN Training starting from randomly initialized weights ...')
    
    return mrcnn_model, fcn_model


##------------------------------------------------------------------------------------    
## NEWSHAPES - FCN Inference
##------------------------------------------------------------------------------------    
def build_fcn_inference_pipeline_newshapes( args = None, mrcnn_config = None,  mode = 'inference', 
                                           verbose = 0):
    print('MODE IS:' , mode)    
    
    ## Build MRCNN Model in Inference mode
    mrcnn_model  = build_mrcnn_inference_pipeline_newshapes( mode = mode, args = args, verbose = verbose)
    
    fcn_config = build_newshapes_config("fcn","inference", args, verbose = verbose)

    ## Build FCN Model in Training Mode
    try :
        del fcn_model
        gc.collect()
    except: 
        pass    
    fcn_model = fcn_modellib.FCN(mode="inference", arch = args.fcn_arch, config=fcn_config)

    ## Display model configuration information
    print()
    print(' FCN Configuration Parameters ')
    print(' ------------------------------ ')
    fcn_config.display()             
    print()
    print(' FCN IO Layers ')
    print(' --------------- ')
    fcn_model.display_layer_info()
    
    ##------------------------------------------------------------------------------------
    ## Load FCN Model weights  
    ##------------------------------------------------------------------------------------
    if args.fcn_model != 'init':
        fcn_model.load_model_weights(init_with = args.fcn_model, verbose = verbose)
    else:
        print(' FCN Training starting from randomly initialized weights ...')
    
    return mrcnn_model, fcn_model


##------------------------------------------------------------------------------------    
## NEWSHAPES - FCN Evaluate
##------------------------------------------------------------------------------------    
def build_fcn_evaluate_pipeline_newshapes( args = None, mrcnn_config = None,  mode = 'evaluate', 
                                           verbose = 0):
        
    return build_fcn_inference_pipeline_newshapes( args = args, 
                                         mrcnn_config = mrcnn_config, 
                                         mode = mode, verbose = verbose)

    
#######################################################################################    
## MRCNN Training pipeline
#######################################################################################
def build_mrcnn_training_pipeline(args, mrcnn_config = None, verbose = 0):

    if mrcnn_config is None :
        mrcnn_config = build_coco_config('mrcnn','training', args, verbose = verbose)
    
    # Recreate the model in training mode
    try :
        del model
        print('delete model is successful')
        gc.collect()
    except: 
        pass
    KB.clear_session()
    mrcnn_model = mrcnn_modellib.MaskRCNN(mode='training', config=mrcnn_config)

    mrcnn_config.display()             
    mrcnn_model.display_layer_info()
    # Load MRCNN Model weights  
    # exclude=["mrcnn_class_logits"] # ,"mrcnn_bbox_fc"]   #, "mrcnn_bbox", "mrcnn_mask"])
    mrcnn_model.load_model_weights(init_with = args.mrcnn_model, exclude = None)      
    
    return mrcnn_model

mrcnn_coco_train = build_mrcnn_training_pipeline 
    
#######################################################################################    
## MRCNN inference pipeline
#######################################################################################
def build_mrcnn_inference_pipeline( args = None, mrcnn_config = None , verbose = 0):
    '''
    sets up mrcnn model in inference mode
    '''
    start_time = datetime.now().strftime("%m-%d-%Y @ %H:%M:%S")
    print()
    print('--> Execution started at:', start_time)
    print("    Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))


    utils.display_input_parms(args)    

    #------------------------------------------------------------------------------------
    # if debug is true set stdout destination to stringIO
    #------------------------------------------------------------------------------------
    # if args.sysout == 'FILE':
        # sys.stdout = io.StringIO()
    
    if mrcnn_config is None :
        mrcnn_config = build_coco_config('mrcnn','evaluate', args, verbose = verbose)
    
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
def build_mrcnn_evaluate_pipeline( args, mrcnn_config = None , verbose = 0):
    '''
    sets up mrcnn model in evaluate mode
    '''
    start_time = datetime.now().strftime("%m-%d-%Y @ %H:%M:%S")
    print()
    print('--> Execution started at:', start_time)
    print("    Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))

    if mrcnn_config is None :
        mrcnn_config = build_coco_config('mrcnn','evaluate', args, verbose = verbose)
        
    # if args.SYSOUT is FILE  set stdout destination to stringIO
    # if args.sysout == 'FILE':
        # sys.stdout = io.StringIO()

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
def build_fcn_training_pipeline( args = None, batch_size = 2, 
                                 fcn_weight_file = 'last', fcn_train_dir = "train_fcn8_coco_adam", verbose = 0):
    start_time = datetime.now().strftime("%m-%d-%Y @ %H:%M:%S")
    print()
    print('--> Execution started at:', start_time)
    print("    Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))

    ##------------------------------------------------------------------------------------
    ## Parse command line arguments
    ##------------------------------------------------------------------------------------
    if args is None:
        print(' args input missing')
        return 
         # parser = command_line_parser()
        # input_parms = " --epochs 2 --steps_in_epoch 32  --last_epoch 0 "
        # input_parms +=" --batch_size "+str(batch_size)+ " --lr 0.00001 --val_steps 8 " 
        # input_parms +=" --mrcnn_logs_dir train_mrcnn_coco "
        # input_parms +=" --fcn_logs_dir " + fcn_train_dir + " " 
        # input_parms +=" --mrcnn_model    last "
        # input_parms +=" --fcn_model    " + fcn_weight_file + " "
        # input_parms +=" --opt            adam "
        # input_parms +=" --fcn_arch       fcn8 " 
        # input_parms +=" --fcn_layers     all " 
        # input_parms +=" --new_log_folder    "
        # input_parms +="--fcn_model /home/kbardool/models/train_fcn_adagrad/shapes20180709T1732/fcn_shapes_1167.h5"
        # print(input_parms)
        # args = parser.parse_args(input_parms.split())
    
    utils.display_input_parms(args)   
        
    # if debug is true set stdout destination to stringIO
    #-------------------------------------------------------------------------------------
    # debug = False
    if args.sysout == 'FILE':
        sys.stdout = io.StringIO()

    if mrcnn_config is None :
        mrcnn_config = build_coco_config('mrcnn','trainfcn', args, verbose = verbose)

    #------------------------------------------------------------------------------------
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
    
    ### Build configuration for FCN model
    fcn_config = build_coco_config('fcn','training', args, verbose = verbose)

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
def build_fcn_inference_pipeline( args = None, mrcnn_config = None, fcn_config = None, 
                                   mode = 'inference', verbose = 0):
    
    start_time = datetime.now().strftime("%m-%d-%Y @ %H:%M:%S")
    print()
    print('--> Execution started at:', start_time)
    print("    Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))

    if mrcnn_config is None :
        mrcnn_config = build_coco_config('mrcnn', mode , args, verbose = verbose)
        
    print(' *** Keras Training mode:', KB.learning_phase())
    KB.set_learning_phase(0)
    print(' *** Keras Training mode after setting:', KB.learning_phase())

    # if debug is true set stdout destination to stringIO
    #------------------------------------------------------------------------------------
    if args.sysout == 'FILE':
        sys.stdout = io.StringIO()

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
    fcn_config = build_coco_config('fcn', 'inference'  , args, verbose = verbose)
    
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
    # Load MRCNN & FCN  Model weights  
    #------------------------------------------------------------------------------------
    mrcnn_model.load_model_weights(init_with = 'last', exclude = None, verbose = verbose)  
    fcn_model.load_model_weights(init_with = args.fcn_model, verbose = verbose)
    
    print(' *** Keras Training mode after setting:', KB.learning_phase())

    return mrcnn_model, fcn_model
    
#######################################################################################    
## FCN evaluate pipeline
#######################################################################################
def build_fcn_evaluate_pipeline( args = None, mrcnn_config = None, fcn_config = None, 
                                 mode = 'evaluate',  verbose = 0):
                                 
    return build_fcn_inference_pipeline( args = args, 
                                         mrcnn_config = mrcnn_config, 
                                         fcn_config = fcn_config, 
                                         mode = mode, verbose = verbose)

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
        visualize.display_training_batch(dataset, [molded_images, image_metas], masks = True, size = 8)
        
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
def run_fcn_only_detection(fcn_model, dataset, input, image_ids = None, verbose = 0):
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
    if not isinstance(image_ids, list):
        image_ids = [image_ids]

    images       = [dataset.load_image(image_id) for image_id in image_ids]
    
    if verbose:
        for image_id in image_ids :
            print("Image Id  : {}     External Id: {}.{}     Image Reference: {}".format( image_id, 
                dataset.image_info[image_id]["source"], dataset.image_info[image_id]["id"], dataset.image_reference(image_id)))

    # Run object detection

    fcn_results = fcn_model.detect(input, verbose= verbose)

    # add GT information 
    
    if verbose:
        print('===> Return from fcn_model.detect_from_images() : ', type(fcn_results), len(fcn_results), len(image_ids))

    for image_id, result in zip(image_ids, fcn_results):
        _, image_meta, gt_class_id, gt_bbox =\
            load_image_gt(dataset, fcn_model.config, image_id, use_mini_mask=False)
        if verbose:
            print('Image meta: ', image_meta[:10])    
        
        result['orig_image_meta'] = image_meta
        result['image_meta']      = image_meta
        result['gt_bboxes']       = gt_bbox
        result['gt_class_ids']    = gt_class_id

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
    
    if verbose:
        print('===> Return from fcn_model.detect_from_images() : ', type(fcn_results), len(fcn_results), len(image_ids))

    for image_id, result in zip(image_ids, fcn_results):
        _, image_meta, gt_class_id, gt_bbox =\
            load_image_gt(dataset, mrcnn_model.config, image_id, use_mini_mask=False)
        if verbose:
            print('Image meta: ', image_meta[:10])    
        
        result['orig_image_meta'] = image_meta
        result['image_meta']      = image_meta
        result['gt_bboxes']       = gt_bbox
        result['gt_class_ids']    = gt_class_id

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

##------------------------------------------------------------------------------------    
## Run MRCNN evaluation on an image ids
##------------------------------------------------------------------------------------            
def run_mrcnn_evaluation(mrcnn_model, dataset, image_ids = None, verbose = 0):
    
    evaluate_batch = get_evaluate_batch(dataset, mrcnn_model.config, image_ids, display = False)    

    assert mrcnn_model.mode  == "evaluate", "Create model in evaluate mode."
    assert len(evaluate_batch) == 5, " length of eval batch must be 4"
    sequence_column = 7
    
    results = mrcnn_model.evaluate(evaluate_batch, verbose = verbose)

    if verbose:
        print('===>   return from  MRCNN evaluate() : ', len(results))
        for i, r in enumerate(results):
            print('\n output ', i, '  ',sorted(r.keys()))
            for key in sorted(r):
                print(key.ljust(20), r[key].shape)        
    
    return results
    
##------------------------------------------------------------------------------------    
## Run FCN evaluation on an image ids
##------------------------------------------------------------------------------------            
def run_fcn_evaluation(fcn_model, mrcnn_model, dataset, image_ids = None, verbose = 0):
    
    eval_batch = get_evaluate_batch(dataset, mrcnn_model.config, image_ids, display = False)    
    fcn_results = fcn_model.evaluate(mrcnn_model, eval_batch, verbose = verbose)    

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

    
