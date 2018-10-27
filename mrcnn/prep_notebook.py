'''
prep_dev_notebook:
pred_newshapes_dev: Runs against new_shapes
'''
import os, sys, random, math, re, gc, time, platform
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
import mrcnn.new_shapes   as new_shapes

# import mrcnn.new_shapes   as shapes
from datetime import datetime   
from mrcnn.config      import Config
from mrcnn.dataset     import Dataset 
from mrcnn.utils       import stack_tensors, stack_tensors_3d, log, Paths
from mrcnn.datagen     import data_generator, load_image_gt
from mrcnn.coco        import CocoDataset, CocoConfig, CocoInferenceConfig, evaluate_coco, build_coco_results

# syst = platform.system()
# if syst == 'Windows':
    # # Root directory of the project
    # print(' windows ' , syst)
    # # WINDOWS MACHINE ------------------------------------------------------------------
    # DIR_ROOT          = "F:\\"
    # DIR_TRAINING   = os.path.join(DIR_ROOT, 'models')
    # DIR_DATASET    = os.path.join(DIR_ROOT, 'MLDatasets')
    # DIR_PRETRAINED = os.path.join(DIR_ROOT, 'PretrainedModels')
# elif syst == 'Linux':
    # print(' Linx ' , syst)
    # # LINUX MACHINE ------------------------------------------------------------------
    # DIR_ROOT       = os.getcwd()
    # DIR_TRAINING   = os.path.expanduser('~/models')
    # DIR_DATASET    = os.path.expanduser('~/MLDatasets')
    # DIR_PRETRAINED = os.path.expanduser('~/PretrainedModels')
# else :
    # raise Error('unreconized system ')

# TRAINING_PATH         = os.path.join(DIR_TRAINING  , "train_mrcnn_coco")
# COCO_DATASET_PATH     = os.path.join(DIR_DATASET   , "coco2014")
# COCO_MODEL_PATH       = os.path.join(DIR_PRETRAINED, "mask_rcnn_coco.h5")
# RESNET_MODEL_PATH     = os.path.join(DIR_PRETRAINED, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
# VGG16_MODEL_PATH      = os.path.join(DIR_PRETRAINED, "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
# FCN_VGG16_MODEL_PATH  = os.path.join(DIR_PRETRAINED, "fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5")

print("Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))
import pprint
pp = pprint.PrettyPrinter(indent=2, width=100)
np.set_printoptions(linewidth=100,precision=4,threshold=1000, suppress = True)


##------------------------------------------------------------------------------------
## Build  NewShapes Training and Validation datasets
##------------------------------------------------------------------------------------
def newshapes_dataset(type, config, image_count, shuffle = True, augment = False):
    '''
        type = { train, val, test}
    '''
    dataset = new_shapes.NewShapesDataset(config)
    dataset.load_shapes(image_count) 
    dataset.prepare()

    return dataset


##------------------------------------------------------------------------------------
## Build  COCO Training and Validation datasets
##------------------------------------------------------------------------------------
def coco_dataset(type, config):
    '''
        type = { train, val, test}
    '''
    # if args.command == "train":
    # Training dataset. Use the training set and 35K from the validation set, as as in the Mask RCNN paper.
    dataset = CocoDataset()
    
    # dataset_test.load_coco(COCO_DATASET_PATH,  "train", class_ids=mrcnn_config.COCO_CLASSES)
    for i in type:
        dataset.load_coco(config.COCO_DATASET_PATH, i )
    dataset.prepare()

    return dataset

    
##------------------------------------------------------------------------------------
## Build Training and Validation datasets
##------------------------------------------------------------------------------------
def prep_coco_dataset(type, config, generator = False):
    # dataset_train, train_generator = coco_dataset(["train",  "val35k"], mrcnn_config)

    # if args.command == "train":
    # Training dataset. Use the training set and 35K from the validation set, as as in the Mask RCNN paper.
    dataset = CocoDataset()
    
    # dataset_test.load_coco(COCO_DATASET_PATH,  "train", class_ids=mrcnn_config.COCO_CLASSES)
    for i in type:
        dataset.load_coco(config.COCO_DATASET_PATH, i )
    dataset.prepare()

    results =  dataset
    
    if generator:
        generator = data_generator(dataset, config, 
                                   batch_size=config.BATCH_SIZE,
                                   shuffle = True, augment = False) 
        results = [dataset, generator]
    return results
    
##------------------------------------------------------------------------------------    
## mrcnn COCO TRAIN
##------------------------------------------------------------------------------------    
def mrcnn_coco_train(mode = 'training', FCN_layers = False, 
                     batch_sz = 1, epoch_steps = 4, 
                     training_folder = 'train_mrcnn_coco',
                     mrcnn_config = None):

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
        mrcnn_config.TRAINING_PATH      = os.path.join(paths.DIR_TRAINING, training_folder)
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
        # mrcnn_config.COCO_CLASSES       = [1,2,3,4,5,6,7,8,9,10]
        # mrcnn_config.NUM_CLASSES        = len(mrcnn_config.COCO_CLASSES) + 1
        # mrcnn_config.FCN_INPUT_SHAPE        = config.IMAGE_SHAPE[0:2]
        # mrcnn_config.DETECTION_MIN_CONFIDENCE = 0.1
        

    # Recreate the model in training mode
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
    mrcnn_model.layer_info()
    
    return [mrcnn_model, mrcnn_config]
    


    
##------------------------------------------------------------------------------------    
## mrcnn COCO TEST
##------------------------------------------------------------------------------------    
def mrcnn_coco_test(mode = 'inference' , 
                    batch_sz = 5, epoch_steps = 4, training_folder = "mrcnn_coco_dev",
                    mrcnn_config = None ):

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
    mrcnn_model.layer_info()
    # print('\n Outputs: ') 
    # pp.pprint(mrcnn_model.keras_model.outputs)

    ##------------------------------------------------------------------------------------
    ## Build  Training and Validation datasets
    ##------------------------------------------------------------------------------------
    # dataset_train, train_generator = coco_dataset(["train",  "val35k"], mrcnn_config)
    dataset_test, test_generator = coco_dataset(["val"], mrcnn_config)
    test_generator = data_generator(dataset_test, mrcnn_config, 
                                   batch_size=mrcnn_config.BATCH_SIZE,
                                   shuffle = True, augment = False) 

    mrcnn_config.display()     
    
    return [mrcnn_model, dataset_test, test_generator, mrcnn_config]
    

    
##------------------------------------------------------------------------------------    
## FCN TRAIN
##------------------------------------------------------------------------------------    
def fcn_coco_train(mode = 'training', fcn_config = None, mrcnn_config = None, training_folder = 'train_mrcnn_coco'):
    
    FCN_TRAINING_PATH = os.path.join(DIR_TRAINING  , "train_fcn_coco")
    ##------------------------------------------------------------------------------------
    ## Build configuration object , if none has been passed
    ##------------------------------------------------------------------------------------
    if fcn_config is None:
        fcn_config = CocoConfig()
        fcn_config.IMAGE_MAX_DIM        = 600
        fcn_config.IMAGE_MIN_DIM        = 480      
        fcn_config.NAME                 = 'fcn'              
        fcn_config.BATCH_SIZE           = mrcnn_config.BATCH_SIZE                 # Batch size is 2 (# GPUs * images/GPU).
        fcn_config.IMAGES_PER_GPU       = mrcnn_config.BATCH_SIZE               # Must match BATCH_SIZE
        fcn_config.STEPS_PER_EPOCH      = mrcnn_config.steps_in_epoch
        fcn_config.LEARNING_RATE        = mrcnn_config.lr
        fcn_config.EPOCHS_TO_RUN        = mrcnn_config.epochs
        fcn_config.FCN_INPUT_SHAPE      = mrcnn_config.IMAGE_SHAPE[0:2] // mrcnn_config.HEATMAP_SCALE_FACTOR 
        fcn_config.LAST_EPOCH_RAN       = mrcnn_config.last_epoch
        fcn_config.WEIGHT_DECAY         = 2.0e-4
        fcn_config.VALIDATION_STEPS     = mrcnn_config.val_steps
        fcn_config.REDUCE_LR_FACTOR     = 0.5
        fcn_config.REDUCE_LR_COOLDOWN   = 50
        fcn_config.REDUCE_LR_PATIENCE   = 33
        fcn_config.EARLY_STOP_PATIENCE  = 50
        fcn_config.EARLY_STOP_MIN_DELTA = 1.0e-4
        fcn_config.MIN_LR               = 1.0e-10
        fcn_config.NEW_LOG_FOLDER       = True  
        fcn_config.OPTIMIZER            = args.opt.upper()
        fcn_config.FCN_TRAINING_PATH    = FCN_TRAINING_PATH

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
    ## Load Mask RCNN Model Weight file
    ##------------------------------------------------------------------------------------
    # mrcnn_config.display()  
    fcn_model.layer_info()
    # print('\n Outputs: ') 
    # pp.pprint(mrcnn_model.keras_model.outputs)
    
    return [fcn_model,fcn_config]
    
    

##------------------------------------------------------------------------------------    
## New Shapes TESTING
##------------------------------------------------------------------------------------    
def prep_newshapes_test(init_with = 'last', FCN_layers = False, batch_sz = 5, epoch_steps = 4,training_folder= "mrcnn_newshape_test_logs"):


    # Build configuration object -----------------------------------------------
    config = new_shapes.NewShapesConfig()
    config.TRAINING_PATH        = os.path.join(DIR_TRAINING, training_folder)
    config.BATCH_SIZE      = batch_sz                  # Batch size is 2 (# GPUs * images/GPU).
    config.IMAGES_PER_GPU  = batch_sz                  # Must match BATCH_SIZE
    config.STEPS_PER_EPOCH = epoch_steps
    config.FCN_INPUT_SHAPE = config.IMAGE_SHAPE[0:2]
    config.DETECTION_MIN_CONFIDENCE = 0.1
    # Build shape dataset        -----------------------------------------------
    # Training dataset
    dataset_test = new_shapes.NewShapesDataset(config)
    dataset_test.load_shapes(3000)
    dataset_test.prepare()


    # Recreate the model in inference mode
    try :
        del model
        print('delete model is successful')
        gc.collect()
    except: 
        pass
    KB.clear_session()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=config.TRAINING_PATH, 
                              FCN_layers = FCN_layers )
        
    print(' COCO Model Path       : ', COCO_DIR_TRAINING)
    print(' Checkpoint folder Path: ', MODEL_DIR)
    print(' Model Parent Path     : ', DIR_TRAINING)
    print(' Resent Model Path     : ', RESNET_DIR_TRAINING)
    # exclude_layers = \
           # ['fcn_block1_conv1' 
           # ,'fcn_block1_conv2' 
           # ,'fcn_block1_pool' 
           # ,'fcn_block2_conv1'
           # ,'fcn_block2_conv2' 
           # ,'fcn_block2_pool'  
           # ,'fcn_block3_conv1' 
           # ,'fcn_block3_conv2' 
           # ,'fcn_block3_conv3' 
           # ,'fcn_block3_pool'  
           # ,'fcn_block4_conv1' 
           # ,'fcn_block4_conv2' 
           # ,'fcn_block4_conv3' 
           # ,'fcn_block4_pool'  
           # ,'fcn_block5_conv1' 
           # ,'fcn_block5_conv2' 
           # ,'fcn_block5_conv3' 
           # ,'fcn_block5_pool'  
           # ,'fcn_fc1'          
           # ,'dropout_1'        
           # ,'fcn_fc2'          
           # ,'dropout_2'        
           # ,'fcn_classify'     
           # ,'fcn_bilinear'     
           # ,'fcn_heatmap_norm' 
           # ,'fcn_scoring'      
           # ,'fcn_heatmap'      
           # ,'fcn_norm_loss']
    
    # load_model(model, init_with = init_with, exclude = exclude_layers)
    model.load_model_weights(init_with = init_with) 

    # print('=====================================')
    # print(" Load second weight file ?? ")
    # model.keras_model.load_weights('E:/Models/vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name= True )
    
    test_generator = data_generator(dataset_test, model.config, shuffle=True,
                                     batch_size=model.config.BATCH_SIZE,
                                     augment = False)
    model.config.display()     
    return [model, dataset_test, test_generator, config]                                 
    

##------------------------------------------------------------------------------------    
## New Shapes TRAINING 
##------------------------------------------------------------------------------------            
def prep_newshapes_train(init_with = "last", FCN_layers= False, batch_sz =5, epoch_steps = 4, training_folder= None):

    import mrcnn.new_shapes as new_shapes
    config.CHECKPOINT_FOLDER = os.path.join(DIR_TRAINING, config.CHECKPOINT_FOLDER)
    MODEL_DIR = os.path.join(DIR_TRAINING, training_folder)

    # Build configuration object -----------------------------------------------
    config = new_shapes.NewShapesConfig()
    config.TRAINING_PATH        = os.path.join(DIR_TRAINING, training_folder)
    config.BATCH_SIZE      = batch_sz                  # Batch size is 2 (# GPUs * images/GPU).
    config.IMAGES_PER_GPU  = batch_sz                  # Must match BATCH_SIZE
    config.STEPS_PER_EPOCH = epoch_steps
    config.FCN_INPUT_SHAPE = config.IMAGE_SHAPE[0:2]

    # Build shape dataset        -----------------------------------------------
    # Training dataset
    dataset_train = new_shapes.NewShapesDataset(config)
    dataset_train.load_shapes(10000) 
    dataset_train.prepare()

    # Validation dataset
    dataset_val = new_shapes.NewShapesDataset(config)
    dataset_val.load_shapes(2500)
    dataset_val.prepare()

    try :
        del model
        print('delete model is successful')
        gc.collect()
    except: 
        pass
    KB.clear_session()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=config.TRAINING_PATH, FCN_layers = config.FCN_LAYERS)

    print('DIR_TRAINING        : ', DIR_TRAINING)
    print('COCO_DIR_TRAINING   : ', COCO_DIR_TRAINING)
    print('RESNET_DIR_TRAINING : ', RESNET_DIR_TRAINING)
    print('MODEL_DIR         : ', MODEL_DIR)
    print('Last Saved Model  : ', model.find_last())
    # exclude_layers = \
           # ['fcn_block1_conv1' 
           # ,'fcn_block1_conv2' 
           # ,'fcn_block1_pool' 
           # ,'fcn_block2_conv1'
           # ,'fcn_block2_conv2' 
           # ,'fcn_block2_pool'  
           # ,'fcn_block3_conv1' 
           # ,'fcn_block3_conv2' 
           # ,'fcn_block3_conv3' 
           # ,'fcn_block3_pool'  
           # ,'fcn_block4_conv1' 
           # ,'fcn_block4_conv2' 
           # ,'fcn_block4_conv3' 
           # ,'fcn_block4_pool'  
           # ,'fcn_block5_conv1' 
           # ,'fcn_block5_conv2' 
           # ,'fcn_block5_conv3' 
           # ,'fcn_block5_pool'  
           # ,'fcn_fc1'          
           # ,'dropout_1'        
           # ,'fcn_fc2'          
           # ,'dropout_2'        
           # ,'fcn_classify'     
           # ,'fcn_bilinear'     
           # ,'fcn_heatmap_norm' 
           # ,'fcn_scoring'      
           # ,'fcn_heatmap'      
           # ,'fcn_norm_loss']
    # load_model(model, init_with = 'last', exclude = exclude_layers)
    model.load_model_weights(init_with = init_with)
    
    # print('=====================================')
    # print(" Load second weight file ?? ")
    # model.keras_model.load_weights('E:/Models/vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name= True)
    
    dataset_train, train_generator = newshapes_dataset("train", mrcnn_config, image_count = 10000)
    dataset_val  , val_generator   = newshapes_dataset("val"  , mrcnn_config, image_count = 2500)
    
                                 
    config.display()     
    return [model, dataset_train, dataset_val, train_generator, val_generator, config]

    

##------------------------------------------------------------------------------------    
## Old Shapes TRAINING
##------------------------------------------------------------------------------------   
def prep_oldshapes_train(init_with = None, FCN_layers = False, batch_sz = 5, epoch_steps = 4, training_folder= "mrcnn_oldshape_training_logs"):
    import mrcnn.shapes    as shapes
    MODEL_DIR = os.path.join(DIR_TRAINING, training_folder)

    # Build configuration object -----------------------------------------------
    config = shapes.ShapesConfig()
    config.BATCH_SIZE      = batch_sz                  # Batch size is 2 (# GPUs * images/GPU).
    config.IMAGES_PER_GPU  = batch_sz                  # Must match BATCH_SIZE
    config.STEPS_PER_EPOCH = epoch_steps
    config.FCN_INPUT_SHAPE = config.IMAGE_SHAPE[0:2]

    # Build shape dataset        -----------------------------------------------
    dataset_train = shapes.ShapesDataset(config)
    dataset_train.load_shapes(3000) 
    dataset_train.prepare()

    # Validation dataset
    dataset_val  = shapes.ShapesDataset(config)
    dataset_val.load_shapes(500)
    dataset_val.prepare()
    
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
    
def prep_oldshapes_test(init_with = None, FCN_layers = False, batch_sz = 5, epoch_steps = 4, training_folder= "mrcnn_oldshape_test_logs"):
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


"""
##------------------------------------------------------------------------------------    
## Old Shapes DEVELOPMENT
##------------------------------------------------------------------------------------        
def prep_oldshapes_dev(init_with = None, FCN_layers = False, batch_sz = 5):
    import mrcnn.shapes    as shapes
    MODEL_DIR = os.path.join(DIR_TRAINING, "mrcnn_oldshape_dev_logs")

    config = build_config(batch_sz = batch_sz)

    dataset_train = shapes.ShapesDataset()
    dataset_train.load_shapes(150, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_train.prepare()
     
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

    load_model(model, init_with = init_with)

    train_generator = data_generator(dataset_train, model.config, shuffle=True,
                                 batch_size=model.config.BATCH_SIZE,
                                 augment = False)    
    model.config.display()

    return [model, dataset_train, train_generator, config]




    
##------------------------------------------------------------------------------------    
## New Shapes DEVELOPMENT
##------------------------------------------------------------------------------------            
def prep_newshapes_dev(init_with = "last", FCN_layers= False, batch_sz = 5):
    import mrcnn.new_shapes as new_shapes
    MODEL_DIR = os.path.join(DIR_TRAINING, "mrcnn_newshape_dev_logs")

    config = build_config(batch_sz = batch_sz, newshapes=True)

    # Build shape dataset        -----------------------------------------------
    # Training dataset 
    dataset_train = new_shapes.NewShapesDataset()
    dataset_train.load_shapes(3000, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_train.prepare()

    # Validation dataset
    dataset_val = new_shapes.NewShapesDataset()
    dataset_val.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_val.prepare()

    try :
        del model, train_generator, val_generator, mm
        gc.collect()
    except: 
        pass
    KB.clear_session()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR,FCN_layers = FCN_layers)

    print('DIR_TRAINING        : ', DIR_TRAINING)
    print('COCO_DIR_TRAINING   : ', COCO_DIR_TRAINING)
    print('RESNET_DIR_TRAINING : ', RESNET_DIR_TRAINING)
    print('MODEL_DIR         : ', MODEL_DIR)
    print('Last Saved Model  : ', model.find_last())

    load_model(model, init_with = 'last')

    train_generator = data_generator(dataset_train, model.config, shuffle=True,
                                 batch_size=model.config.BATCH_SIZE,
                                 augment = False)    
    config.display()     
    return [model, dataset_train, train_generator, config]
    
"""
