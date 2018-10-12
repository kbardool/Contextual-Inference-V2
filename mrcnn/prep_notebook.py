'''
prep_dev_notebook:
pred_newshapes_dev: Runs against new_shapes
'''
import os, sys, random, math, re, gc, time
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
from mrcnn.utils       import stack_tensors, stack_tensors_3d, log
from mrcnn.datagen     import data_generator, load_image_gt
from mrcnn.coco        import CocoDataset, CocoConfig, CocoInferenceConfig, evaluate_coco, build_coco_results

import platform

syst = platform.system()
if syst == 'Windows':
    # Root directory of the project
    print(' windows ' , syst)
    # WINDOWS MACHINE ------------------------------------------------------------------
    ROOT_DIR          = "E:\\"
    TRAINING_DIR   = os.path.join(ROOT_DIR, "models")
    DATASET_DIR    = os.path.join(ROOT_DIR, 'MLDatasets')
    PRETRAINED_DIR = os.path.join(ROOT_DIR, 'PretrainedModels')
elif syst == 'Linux':
    print(' Linx ' , syst)
    # LINUX MACHINE ------------------------------------------------------------------
    ROOT_DIR       = os.getcwd()
    TRAINING_DIR   = os.path.expanduser('~/models')
    DATASET_DIR    = os.path.expanduser('~/MLDatasets')
    PRETRAINED_DIR = os.path.expanduser('~/PretrainedModels')
else :
    raise Error('unreconized system  '      )

# MODEL_DIR    = os.path.join(TRAINING_DIR, "mrcnn_logs")

TRAINING_PATH     = os.path.join(TRAINING_DIR  , "train_mrcnn_coco")
COCO_DATASET_PATH = os.path.join(DATASET_DIR   , "coco2014")
COCO_MODEL_PATH   = os.path.join(PRETRAINED_DIR, "mask_rcnn_coco.h5")
RESNET_MODEL_PATH = os.path.join(PRETRAINED_DIR, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
VGG16_MODEL_PATH  = os.path.join(PRETRAINED_DIR, "fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5")

# print('\n\n\n')
# print(' Checkpoint directory  : ', TRAINING_DIR)
# print(' Checkpoint folder     : ', TRAINING_PATH)
# print(' COCO   Model Path     : ', COCO_MODEL_PATH)
# print(' ResNet Model Path     : ', RESNET_MODEL_PATH)
# print(' VGG16  Model Path     : ', COCO_MODEL_PATH)



print("Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))
import pprint
pp = pprint.PrettyPrinter(indent=2, width=100)
np.set_printoptions(linewidth=100,precision=4,threshold=1000, suppress = True)


##------------------------------------------------------------------------------------
## Build  NewShapes Training and Validation datasets
##------------------------------------------------------------------------------------
def newshapes_dataset(type, config, shuffle = True, augment = False):
    '''
        type = { train, val, test}
    '''
    dataset = new_shapes.NewShapesDataset(config)
    dataset.load_shapes(config.TRAINING_IMAGES) 
    dataset.prepare()

    generator = data_generator(dataset, config, batch_size=config.BATCH_SIZE,
                                shuffle=shuffle,
                                augment = augment)   

    return [dataset, generator]    


##------------------------------------------------------------------------------------
## Build  COCO Training and Validation datasets
##------------------------------------------------------------------------------------
def coco_dataset(type, config, shuffle = True, augment = False):
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

    generator = data_generator(dataset, config, batch_size = config.BATCH_SIZE,
                                     shuffle = shuffle,
                                     augment = augment)   

    return [dataset, generator]

    
    
##------------------------------------------------------------------------------------    
## mrcnn COCO TRAIN
##------------------------------------------------------------------------------------    
def mrcnn_coco_train(mode = 'training', FCN_layers = False, 
                     batch_sz = 1, epoch_steps = 4, 
                     mrcnn_config = None):

    ##------------------------------------------------------------------------------------
    ## Build configuration object , if none has been passed
    ##------------------------------------------------------------------------------------
    if mrcnn_config is None:
        mrcnn_config = CocoConfig()
        mrcnn_config.NAME               = 'mrcnn'              
        mrcnn_config.TRAINING_PATH      = os.path.join(TRAINING_DIR, training_folder)
        mrcnn_config.COCO_DATASET_PATH  = COCO_DATASET_PATH 
        mrcnn_config.COCO_MODEL_PATH    = COCO_MODEL_PATH   
        mrcnn_config.RESNET_MODEL_PATH  = RESNET_MODEL_PATH 
        mrcnn_config.VGG16_MODEL_PATH   = VGG16_MODEL_PATH  
        mrcnn_config.COCO_CLASSES       = None 
        mrcnn_config.DETECTION_PER_CLASS = 200
        mrcnn_config.HEATMAP_SCALE_FACTOR = 4
        mrcnn_config.BATCH_SIZE         = batch_sz                  # Batch size is 2 (# GPUs * images/GPU).
        mrcnn_config.IMAGES_PER_GPU     = batch_sz                  # Must match BATCH_SIZE
        mrcnn_config.STEPS_PER_EPOCH    = epoch_steps
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
    mrcnn_model = mrcnn_modellib.MaskRCNN(mode=mode, config=mrcnn_config, model_dir=TRAINING_PATH)

    ##------------------------------------------------------------------------------------
    ## Load Mask RCNN Model Weight file
    ##------------------------------------------------------------------------------------
    # mrcnn_model.load_model_weights( init_with = init_weights)   

    # print('==========================================')
    # print(" MRCNN MODEL Load weight file COMPLETE    ")
    # print('==========================================')
    # print('\n\n\n')
    # print(' Checkpoint directory  : ', TRAINING_DIR)
    # print(' Checkpoint folder     : ', TRAINING_PATH)
    # print(' COCO   Model Path     : ', COCO_MODEL_PATH)
    # print(' ResNet Model Path     : ', RESNET_MODEL_PATH)
    # print(' VGG16  Model Path     : ', COCO_MODEL_PATH)
    
    # mrcnn_config.display()  
    mrcnn_model.layer_info()
    # print('\n Outputs: ') 
    # pp.pprint(mrcnn_model.keras_model.outputs)
    
    ##------------------------------------------------------------------------------------
    ## Build Training and Validation datasets
    ##------------------------------------------------------------------------------------
    dataset_train, train_generator = coco_dataset(["train",  "val35k"], mrcnn_config)
    dataset_val  , val_generator   = coco_dataset(["minival"], mrcnn_config)
    mrcnn_config.display()     
    
    return [mrcnn_model, dataset_train, dataset_val, train_generator, val_generator, mrcnn_config]
    

##------------------------------------------------------------------------------------    
## mrcnn COCO TEST
##------------------------------------------------------------------------------------    
def mrcnn_coco_test(mode = 'inference' , batch_sz = 5, epoch_steps = 4, training_folder = "mrcnn_coco_dev"):

    TRAINING_PATH = os.path.join(TRAINING_DIR, training_folder)

    mrcnn_config = CocoInferenceConfig()
    mrcnn_config.NAME               = 'mrcnn'              
    mrcnn_config.TRAINING_PATH      = TRAINING_PATH
    mrcnn_config.COCO_MODEL_PATH    = COCO_MODEL_PATH   
    mrcnn_config.RESNET_MODEL_PATH  = RESNET_MODEL_PATH 
    mrcnn_config.VGG16_MODEL_PATH   = VGG16_MODEL_PATH  
    mrcnn_config.DETECTION_PER_CLASS= 200
    mrcnn_config.HEATMAP_SCALE_FACTOR = 4
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
    mrcnn_model = mrcnn_modellib.MaskRCNN(mode=mode, config=mrcnn_config, model_dir=TRAINING_PATH)

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
    
    mrcnn_config.display()     
    
    return [mrcnn_model, dataset_test, test_generator, mrcnn_config]
    


##------------------------------------------------------------------------------------    
## New Shapes TESTING
##------------------------------------------------------------------------------------    
def prep_newshapes_test(init_with = 'last', FCN_layers = False, batch_sz = 5, epoch_steps = 4,training_folder= "mrcnn_newshape_test_logs"):

    MODEL_DIR = os.path.join(TRAINING_DIR, training_folder)

    # Build configuration object -----------------------------------------------
    config = new_shapes.NewShapesConfig()
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
    model = modellib.MaskRCNN(mode="inference", 
                              config=config,
                              model_dir=MODEL_DIR, 
                              FCN_layers = FCN_layers )
        
    print(' COCO Model Path       : ', COCO_TRAINING_DIR)
    print(' Checkpoint folder Path: ', MODEL_DIR)
    print(' Model Parent Path     : ', TRAINING_DIR)
    print(' Resent Model Path     : ', RESNET_TRAINING_DIR)
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
def prep_newshapes_train2(init_with = "last",  config=None):

    import mrcnn.new_shapes as new_shapes
    config.CHECKPOINT_FOLDER = os.path.join(TRAINING_DIR, config.CHECKPOINT_FOLDER)

    try :
        del model
        print('delete model is successful')
        gc.collect()
    except: 
        pass
    KB.clear_session()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=config.CHECKPOINT_FOLDER, FCN_layers = config.FCN_LAYERS)

    print('TRAINING_DIR        : ', TRAINING_DIR)
    print('COCO_TRAINING_DIR   : ', COCO_TRAINING_DIR)
    print('RESNET_TRAINING_DIR : ', RESNET_TRAINING_DIR)
    print('CHECKPOINT_DIR      : ', config.CHECKPOINT_FOLDER)
    print('Last Saved Model    : ', model.find_last())
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
    dataset_train, train_generator = newshapes_dataset("train", mrcnn_config)
    dataset_val  , val_generator   = newshapes_dataset("val"  , mrcnn_config)    
    
                                 
    config.display()     
    return [model, dataset_train, dataset_val, train_generator, val_generator, config]

    
##------------------------------------------------------------------------------------    
## New Shapes TRAINING 
##------------------------------------------------------------------------------------            
def prep_newshapes_train(init_with = "last", FCN_layers= False, batch_sz =5, epoch_steps = 4, training_folder= None):

    MODEL_DIR = os.path.join(TRAINING_DIR, training_folder)

    # Build configuration object -----------------------------------------------
    config = new_shapes.NewShapesConfig()
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
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR,FCN_layers = FCN_layers)

    print('TRAINING_DIR        : ', TRAINING_DIR)
    print('COCO_TRAINING_DIR   : ', COCO_TRAINING_DIR)
    print('RESNET_TRAINING_DIR : ', RESNET_TRAINING_DIR)
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
    
    
    
    train_generator = data_generator(dataset_train, model.config, shuffle=True,
                                 batch_size=model.config.BATCH_SIZE,
                                 augment = False)   


    val_generator = data_generator(dataset_val, model.config, shuffle=True, 
                                    batch_size=model.config.BATCH_SIZE,
                                    augment=False)                                           
    config.display()     
    return [model, dataset_train, dataset_val, train_generator, val_generator, config]

    

    
##------------------------------------------------------------------------------------    
## Old Shapes TRAINING
##------------------------------------------------------------------------------------   
def prep_oldshapes_train(init_with = None, FCN_layers = False, batch_sz = 5, epoch_steps = 4, training_folder= "mrcnn_oldshape_training_logs"):
    import mrcnn.shapes    as shapes
    MODEL_DIR = os.path.join(TRAINING_DIR, training_folder)

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

    print(' COCO Model Path       : ', COCO_TRAINING_DIR)
    print(' Checkpoint folder Path: ', MODEL_DIR)
    print(' Model Parent Path     : ', TRAINING_DIR)
    print(' Resent Model Path     : ', RESNET_TRAINING_DIR)

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
    MODEL_DIR = os.path.join(TRAINING_DIR, training_folder)
    # MODEL_DIR = os.path.join(TRAINING_DIR, "mrcnn_development_logs")

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
        
    print(' COCO Model Path       : ', COCO_TRAINING_DIR)
    print(' Checkpoint folder Path: ', MODEL_DIR)
    print(' Model Parent Path     : ', TRAINING_DIR)
    print(' Resent Model Path     : ', RESNET_TRAINING_DIR)

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
    MODEL_DIR = os.path.join(TRAINING_DIR, "mrcnn_oldshape_dev_logs")

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

    print(' COCO Model Path       : ', COCO_TRAINING_DIR)
    print(' Checkpoint folder Path: ', MODEL_DIR)
    print(' Model Parent Path     : ', TRAINING_DIR)
    print(' Resent Model Path     : ', RESNET_TRAINING_DIR)

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
    MODEL_DIR = os.path.join(TRAINING_DIR, "mrcnn_newshape_dev_logs")

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

    print('TRAINING_DIR        : ', TRAINING_DIR)
    print('COCO_TRAINING_DIR   : ', COCO_TRAINING_DIR)
    print('RESNET_TRAINING_DIR : ', RESNET_TRAINING_DIR)
    print('MODEL_DIR         : ', MODEL_DIR)
    print('Last Saved Model  : ', model.find_last())

    load_model(model, init_with = 'last')

    train_generator = data_generator(dataset_train, model.config, shuffle=True,
                                 batch_size=model.config.BATCH_SIZE,
                                 augment = False)    
    config.display()     
    return [model, dataset_train, train_generator, config]
    
"""
