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
# import mrcnn.new_shapes   as shapes
from mrcnn.config      import Config
from mrcnn.dataset     import Dataset 
from mrcnn.utils       import stack_tensors, stack_tensors_3d, log, Paths,command_line_parser
from mrcnn.datagen     import data_generator, load_image_gt, data_gen_simulate
# from mrcnn.coco      import CocoDataset, CocoConfig, CocoInferenceConfig, evaluate_coco, build_coco_results,
from mrcnn.coco        import CocoConfig, CocoInferenceConfig, prep_coco_dataset
from mrcnn.heatmap     import HeatmapDataset
from mrcnn.datagen_fcn import fcn_data_generator,fcn_data_gen_simulate

import mrcnn.new_shapes   as shapes
import mrcnn.utils        as utils
# from mrcnn.callbacks    import get_layer_output_1,get_layer_output_2, get_layer_outputs


pp = pprint.PrettyPrinter(indent=2, width=100)
np.set_printoptions(linewidth=100,precision=4,threshold=1000, suppress = True)

##------------------------------------------------------------------------------------
## Build  NewShapes Training and Validation datasets
##------------------------------------------------------------------------------------
def prep_newshape_dataset(config, image_count, shuffle = True, augment = False, generator = False):
    '''
    '''
    import mrcnn.new_shapes   as new_shapes
    
    dataset = new_shapes.NewShapesDataset(config)
    dataset.load_shapes(image_count) 
    dataset.prepare()

    results = dataset
    
    if generator:
        generator = data_generator(dataset, config, 
                                   batch_size=config.BATCH_SIZE,
                                   shuffle = True, augment = False) 
        return [dataset, generator]
    else:
        return results
    
##------------------------------------------------------------------------------------
## Build Training and Validation datasets
##------------------------------------------------------------------------------------
def prep_heatmap_dataset(type, config, generator = False, shuffle = True, augment = False):
    # dataset_train, train_generator = coco_dataset(["train",  "val35k"], mrcnn_config)

    # if args.command == "train":
    # Training dataset. Use the training set and 35K from the validation set, as as in the Mask RCNN paper.
    dataset = HeatmapDataset()
    
    # dataset_test.load_coco(COCO_DATASET_PATH,  "train", class_ids=mrcnn_config.COCO_CLASSES)
    for i in type:
        dataset.load_heatmap(config.COCO_DATASET_PATH, config.COCO_HEATMAP_PATH, i )
    dataset.prepare()

    results =  dataset
    
    if generator:
        generator = fcn_data_generator(dataset, config, 
                                   batch_size=config.BATCH_SIZE,
                                   shuffle = shuffle, augment = augment) 
        results = [dataset, generator]
    return results
    
##------------------------------------------------------------------------------------    
## mrcnn COCO TRAIN
##------------------------------------------------------------------------------------    
def mrcnn_coco_train(mode = 'training', 
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
                    batch_sz = 5, epoch_steps = 4, training_folder = "train_mrcnn_coco",
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
    mrcnn_config.display()     
    
    return [mrcnn_model, mrcnn_config]

 

##------------------------------------------------------------------------------------    
## New Shapes TRAINING 
##------------------------------------------------------------------------------------            
def mrcnn_newshape_train(mode = 'training' ,
                     batch_sz = 1, epoch_steps = 4, training_folder = "train_newshapes_coco",
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

    # Recreate the model in training mode
    try :
        del model
        print('delete model is successful')
        gc.collect()
    except: 
        pass
    KB.clear_session()
    mrcnn_model = mrcnn_modellib.MaskRCNN(mode=mode, config=mrcnn_config)

    # display model layer info
    mrcnn_model.layer_info()
    
    return [mrcnn_model, mrcnn_config]
    
 
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
    
##------------------------------------------------------------------------------------    
## run fcn training pipeline on training batch or list of image ids 
##------------------------------------------------------------------------------------    
def run_mrcnn_training_pipeline(mrcnn_model, dataset, mrcnn_input = None, image_ids = None, verbose = 0):

    if image_ids is not None:
        image_batch = get_image_batch(dataset, image_ids, display = True)
        mrcnn_input = get_training_batch(dataset, mrcnn_model, image_ids)
    elif mrcnn_input is not None:
        image_batch = mrcnn_input[0]
    else:
        print('ERROR - mrcnn_input or image_ids must be passed to inference pipeline')
        assert len(mrcnn_input) == 2, 'Length of mrcnn_input list must be 2'
        return
    print('** Pass through MRCNN model:')  
    
    mrcnn_output = mrcnn_model.keras_model.get_layer_outputs( mrcnn_input, training_flag = True, verbose = verbose)

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
## build_trainfcn_pipeline()
##------------------------------------------------------------------------------------    
def build_fcn_training_pipeline( args = None, input_parms = None,  batch_size = 2, fcn_weight_file = 'last', fcn_train_dir = "train_fcn8_coco_adam"):
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

        
    # if debug is true set stdout destination to stringIO
    #----------------------------------------------------------------------------------------------            
    # debug = False
    if args.sysout == 'FILE':
        sys.stdout = io.StringIO()

    # print("    Dataset            : ", args.dataset)
    # print("    Logs               : ", args.logs)
    # print("    Limit              : ", args.limit)
    print("    MRCNN Model        : ", args.mrcnn_model)
    print("    FCN Model          : ", args.fcn_model)
    print("    MRCNN Log Dir      : ", args.mrcnn_logs_dir)
    print("    FCN Log Dir        : ", args.fcn_logs_dir)
    print("    FCN Arch           : ", args.fcn_arch)
    print("    FCN Log Dir        : ", args.fcn_layers)
    print("    Last Epoch         : ", args.last_epoch)
    print("    Epochs to run      : ", args.epochs)
    print("    Steps in each epoch: ", args.steps_in_epoch)
    print("    Validation steps   : ", args.val_steps)
    print("    Batch Size         : ", args.batch_size)
    print("    Optimizer          : ", args.opt)
    print("    sysout             : ", args.sysout)

    # setup project directories
    #---------------------------------------------------------------------------------
    paths = Paths(fcn_training_folder = args.fcn_logs_dir, mrcnn_training_folder = args.mrcnn_logs_dir)
    paths.display()

    # Build configuration object 
    #------------------------------------------------------------------------------------                          
    mrcnn_config                    = CocoConfig()
    # import mrcnn.new_shapes as new_shapes
    # mrcnn_config = new_shapes.NewShapesConfig()

    mrcnn_config.NAME               = 'mrcnn'              
    mrcnn_config.TRAINING_PATH      = paths.MRCNN_TRAINING_PATH
    mrcnn_config.COCO_DATASET_PATH  = paths.COCO_DATASET_PATH 
    mrcnn_config.COCO_MODEL_PATH    = paths.COCO_MODEL_PATH   
    mrcnn_config.RESNET_MODEL_PATH  = paths.RESNET_MODEL_PATH 
    mrcnn_config.VGG16_MODEL_PATH   = paths.VGG16_MODEL_PATH  
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
    mrcnn_config.NEW_LOG_FOLDER     = False
    mrcnn_config.SYSOUT             = args.sysout

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
    from mrcnn.prep_notebook import mrcnn_coco_train
    mrcnn_model, mrcnn_config = mrcnn_coco_train(mode = 'trainfcn', mrcnn_config = mrcnn_config)
    
    # Build configuration for FCN model
    #------------------------------------------------------------------------------------
    fcn_config = CocoConfig()
    fcn_config.COCO_DATASET_PATH  = paths.COCO_DATASET_PATH 
    fcn_config.COCO_HEATMAP_PATH  = paths.COCO_HEATMAP_PATH 

    # mrcnn_config.COCO_MODEL_PATH    = COCO_MODEL_PATH   
    # mrcnn_config.RESNET_MODEL_PATH  = RESNET_MODEL_PATH 
    fcn_config.NAME                 = 'fcn'              
    fcn_config.TRAINING_PATH        = paths.FCN_TRAINING_PATH
    fcn_config.VGG16_MODEL_PATH     = paths.FCN_VGG16_MODEL_PATH
    fcn_config.HEATMAP_SCALE_FACTOR = mrcnn_config.HEATMAP_SCALE_FACTOR
    fcn_config.FCN_INPUT_SHAPE      = fcn_config.IMAGE_SHAPE[0:2] // mrcnn_config.HEATMAP_SCALE_FACTOR 

    fcn_config.BATCH_SIZE           = int(args.batch_size)                 # Batch size is 2 (# GPUs * images/GPU).
    fcn_config.IMAGES_PER_GPU       = int(args.batch_size)                   # Must match BATCH_SIZE
    fcn_config.EPOCHS_TO_RUN        = int(args.epochs)
    fcn_config.STEPS_PER_EPOCH      = int(args.steps_in_epoch)
    fcn_config.LAST_EPOCH_RAN       = int(args.last_epoch)
    
    fcn_config.LEARNING_RATE        = float(args.lr)
    fcn_config.VALIDATION_STEPS     = int(args.val_steps)
    fcn_config.BATCH_MOMENTUM       = 0.9
    fcn_config.WEIGHT_DECAY         = 2.0e-4     ## FCN Weight decays are 5.0e-4 or 2.0e-4
    
    fcn_config.REDUCE_LR_FACTOR     = 0.5
    fcn_config.REDUCE_LR_COOLDOWN   = 15
    fcn_config.REDUCE_LR_PATIENCE   = 50
    fcn_config.REDUCE_LR_MIN_DELTA  = 1e-6

    fcn_config.EARLY_STOP_PATIENCE  = 150
    fcn_config.EARLY_STOP_MIN_DELTA = 1.0e-7

    fcn_config.MIN_LR               = 1.0e-10
    fcn_config.CHECKPOINT_PERIOD    = 1     

    fcn_config.TRAINING_LAYERS      = args.fcn_layers
    fcn_config.NEW_LOG_FOLDER       = args.new_log_folder
    fcn_config.OPTIMIZER            = args.opt
    fcn_config.SYSOUT               = args.sysout
    fcn_config.display()


    ## Build FCN Model in Training Mode
    try :
        del fcn_model
        gc.collect()
    except: 
        pass    
    fcn_model = fcn_modellib.FCN(mode="training", arch = args.fcn_arch, config=fcn_config)

    ## Display model configuration information
    paths.display()
    fcn_config.display()  
    fcn_model.layer_info()
    
    # exclude=["mrcnn_class_logits"] # ,"mrcnn_bbox_fc"]   #, "mrcnn_bbox", "mrcnn_mask"])
    mrcnn_model.load_model_weights(init_with = args.mrcnn_model , exclude = None)  
    
    # Load FCN Model weights  
    #------------------------------------------------------------------------------------
    if args.fcn_model != 'init':
        fcn_model.load_model_weights(init_with = args.fcn_model, verbose = 0)
    else:
        print(' FCN Training starting from randomly initialized weights ...')
    
    return mrcnn_model, fcn_model

##------------------------------------------------------------------------------------    
## run fcn training pipeline on training batch or list of image ids 
##------------------------------------------------------------------------------------    
def run_fcn_training_pipeline(mrcnn_model, fcn_model, dataset, mrcnn_input = None, image_ids = None, verbose = 0):

    if image_ids is not None:
        image_batch = get_image_batch(dataset, image_ids, display = True)
        mrcnn_input, _ = get_inference_batch(dataset, mrcnn_model, image_ids)
    elif mrcnn_input is not None:
        image_batch = mrcnn_input[0]
    else:
        print('ERROR - mrcnn_input or image_ids must be passed to inference pipeline')
        assert len(mrcnn_input) == 2, 'Length of mrcnn_input list must be 2'
        return
    print('** Pass through MRCNN model:')  
    
    mrcnn_output = mrcnn_model.keras_model.get_layer_outputs(mrcnn_input, training_flag = True, verbose = verbose)

    # if verbose :
        # print(' mrcnn outputs: ')
        # print('----------------')
        # print(' length of mrcnn output : ', len(mrcnn_output))
        # for i,j in enumerate(mrcnn_output):
            # print('mrcnn output ', i, ' shape: ' ,  j.shape)

    # build input to fcn model 
    # fcn input / outputs in inference mode:
    # inputs  = [input_image_meta, pr_hm, pr_hm_scores, gt_hm, gt_hm_scores]
    # outputs = [fcn_hm, fcn_sm, fcn_MSE_loss, fcn_CE_loss, fcn_scores]     [0,1,2,3,4]
    
    fcn_input = [mrcnn_input[1]]
    fcn_input.extend(mrcnn_output[:4])
    
    print('\n** Pass through FCN model:')    
    fcn_output = fcn_model.keras_model.get_layer_outputs(fcn_input, verbose = verbose)
    
    if verbose :
        print('\n fcn outputs: ')
        print('----------------')
        for i,j in enumerate(fcn_input):
            print('fcn input ', i, ' shape: ' ,  j.shape)
        for i,j in enumerate(fcn_output):
            print('fcn output ', i, ' shape: ' ,  j.shape)        
    
    outputs = {}
    outputs['mrcnn_input']  = mrcnn_input
    outputs['fcn_input']    = fcn_input
    outputs['mrcnn_output'] = mrcnn_output
    outputs['fcn_output']   = fcn_output
    outputs['image_batch']  = image_batch
    return outputs 
                
    
##------------------------------------------------------------------------------------    
## MRCNN build_inference_pipeline()
##------------------------------------------------------------------------------------    
def build_mrcnn_inference_pipeline( args = None, input_parms = None,  batch_size = 2, fcn_weight_file = 'last'):
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
            input_parms +=" --fcn_logs_dir " + fcn_train_dir + " " 
            input_parms +=" --mrcnn_model    last "
            input_parms +=" --fcn_model    " + fcn_weight_file + " "
            input_parms +=" --fcn_arch       fcn8 " 
            input_parms +=" --fcn_layers     all " 
            input_parms +=" --sysout        screen "
            input_parms +=" --coco_classes   62 63 67 78 79 80 81 82 72 73 74 75 76 77"
            print(input_parms)
        args = parser.parse_args(input_parms.split())


    # if debug is true set stdout destination to stringIO
    #------------------------------------------------------------------------------------
    # debug = False
    if args.sysout == 'FILE':
        sys.stdout = io.StringIO()

    # print("    Dataset            : ", args.dataset)
    # print("    Logs               : ", args.logs)
    # print("    Limit              : ", args.limit)
    print("    MRCNN Model        : ", args.mrcnn_model)
    print("    FCN Model          : ", args.fcn_model)
    print("    MRCNN Log Dir      : ", args.mrcnn_logs_dir)
    print("    FCN Log Dir        : ", args.fcn_logs_dir)
    print("    FCN Arch           : ", args.fcn_arch)
    print("    FCN Training Layers: ", args.fcn_layers)
    print("    sysout             : ", args.sysout)

    ## setup project directories
    #------------------------------------------------------------------------------------
    paths = Paths(fcn_training_folder = args.fcn_logs_dir, mrcnn_training_folder = args.mrcnn_logs_dir)
    paths.display()

    # Build configuration object 
    #------------------------------------------------------------------------------------                          
    mrcnn_config                    = CocoConfig()
    mrcnn_config.NAME               = 'mrcnn'              
    mrcnn_config.TRAINING_PATH      = paths.MRCNN_TRAINING_PATH
    mrcnn_config.COCO_DATASET_PATH  = paths.COCO_DATASET_PATH 
    mrcnn_config.COCO_MODEL_PATH    = paths.COCO_MODEL_PATH   
    mrcnn_config.RESNET_MODEL_PATH  = paths.RESNET_MODEL_PATH 
    mrcnn_config.VGG16_MODEL_PATH   = paths.VGG16_MODEL_PATH  
    mrcnn_config.COCO_CLASSES       = None 
    mrcnn_config.DETECTION_PER_CLASS = 200
    mrcnn_config.HEATMAP_SCALE_FACTOR = 4
    mrcnn_config.BATCH_SIZE         = int(args.batch_size)                  # Batch size is 2 (# GPUs * images/GPU).
    mrcnn_config.IMAGES_PER_GPU     = int(args.batch_size)                  # Must match BATCH_SIZE
    
    mrcnn_config.DETECTION_MIN_CONFIDENCE = 0.3
    mrcnn_config.DETECTION_MAX_INSTANCES =  100
    # mrcnn_config.STEPS_PER_EPOCH    = int(args.steps_in_epoch)
    # mrcnn_config.LEARNING_RATE      = float(args.lr)
    # mrcnn_config.EPOCHS_TO_RUN      = int(args.epochs)
    mrcnn_config.FCN_INPUT_SHAPE    = mrcnn_config.IMAGE_SHAPE[0:2]
    # mrcnn_config.LAST_EPOCH_RAN     = int(args.last_epoch)
    mrcnn_config.NEW_LOG_FOLDER       = args.new_log_folder
    # mrcnn_config.SYSOUT               = args.sysout

    #  Build mrcnn Model
    #------------------------------------------------------------------------------------
    
    from mrcnn.prep_notebook import mrcnn_coco_train
    mrcnn_model, mrcnn_config = mrcnn_coco_train(mode = 'inference', mrcnn_config = mrcnn_config)
    
    
    # Load MRCNN Model weights  
    # exclude=["mrcnn_class_logits"] # ,"mrcnn_bbox_fc"]   #, "mrcnn_bbox", "mrcnn_mask"])
    mrcnn_model.load_model_weights(init_with = 'last', exclude = None)  
    
    
    return mrcnn_model

##------------------------------------------------------------------------------------    
## MRCNN run_mrcnn_detection() on list of image ids
##------------------------------------------------------------------------------------            
def run_mrcnn_detection(mrcnn_model, dataset, image_id = None, verbose = 0):

    if image_id is None:
        image_id = random.choice(dataset.image_ids)
        
    image = dataset.load_image(image_id)
    gt_data = [{}]

    _, image_meta, gt_data[0]['gt_class_id'] , gt_data[0]['gt_bbox'] =\
            load_image_gt(dataset, mrcnn_model.config, image_id, use_mini_mask=False)
    print(gt_data[0]['gt_bbox'])
    info = dataset.image_info[image_id]
    print("Image Id  : {}     External Id: {}.{}     Image Reference: {}".format( image_id, info["source"], info["id"],
                                           dataset.image_reference(image_id)))
    print('Image meta: ', image_meta[:10])
    # Run object detection
    mrcnn_results = mrcnn_model.detect([image], verbose=1)
    
    mrcnn_results[0]['orig_image_meta']= image_meta

    # Display results
    if verbose:
        np.set_printoptions(linewidth=180,precision=4,threshold=10000, suppress = True)
        print(' Length of results from MRCNN detect: ', len(mrcnn_results))
        r = mrcnn_results[0]
        print('mrcnn_results keys: ')
        print('--------------------')
        for i in sorted(r.keys()):
            print('   {:.<25s}  {}'.format(i , r[i].shape))        
        print()
        r = gt_data[0]
        print('gt_data : ')
        print('----------')
        for i in sorted(r.keys()):
            print('   {:.<25s}  {}'.format(i , r[i].shape))        
        # print('image          : ', r["image"].shape )
        # print('image_meta     : ', r["image_meta"].shape, r["image_meta"][:11] )
        # print('class ids      : ', r['class_ids'].shape, r['class_ids'])
        # print('mrcnn_scores         : ', r['mrcnn_scores'].shape)
        # print(r['mrcnn_scores'])
        # print('pr_scores      : ', r['pr_scores'].shape)
        # print('pr_scores  : ')
        # print(r['pr_scores'])
        #            
            
            
    return mrcnn_results, gt_data            
    
    

##------------------------------------------------------------------------------------    
## FCN build_inference_pipeline()
##------------------------------------------------------------------------------------    
def build_fcn_inference_pipeline( args = None, input_parms = None,  batch_size = 2, fcn_weight_file = 'last'):
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


    # if debug is true set stdout destination to stringIO
    #------------------------------------------------------------------------------------
    # debug = False
    if args.sysout == 'FILE':
        sys.stdout = io.StringIO()
    utils.display_input_parms(args)

    ## setup project directories
    #------------------------------------------------------------------------------------
    paths = Paths(fcn_training_folder = args.fcn_logs_dir, mrcnn_training_folder = args.mrcnn_logs_dir)
    paths.display()

    # Build configuration object 
    #------------------------------------------------------------------------------------                          
    mrcnn_config                    = CocoConfig()
    mrcnn_config.NAME               = 'mrcnn'              
    mrcnn_config.TRAINING_PATH      = paths.MRCNN_TRAINING_PATH
    mrcnn_config.COCO_DATASET_PATH  = paths.COCO_DATASET_PATH 
    mrcnn_config.COCO_MODEL_PATH    = paths.COCO_MODEL_PATH   
    mrcnn_config.RESNET_MODEL_PATH  = paths.RESNET_MODEL_PATH 
    mrcnn_config.VGG16_MODEL_PATH   = paths.VGG16_MODEL_PATH  
    mrcnn_config.COCO_CLASSES       = None 
    mrcnn_config.DETECTION_PER_CLASS = 200
    mrcnn_config.HEATMAP_SCALE_FACTOR = 4
    mrcnn_config.BATCH_SIZE         = int(args.batch_size)                  # Batch size is 2 (# GPUs * images/GPU).
    mrcnn_config.IMAGES_PER_GPU     = int(args.batch_size)                  # Must match BATCH_SIZE
    
    mrcnn_config.DETECTION_MIN_CONFIDENCE = 0.3
    mrcnn_config.DETECTION_MAX_INSTANCES =  100
    # mrcnn_config.STEPS_PER_EPOCH    = int(args.steps_in_epoch)
    # mrcnn_config.LEARNING_RATE      = float(args.lr)
    # mrcnn_config.EPOCHS_TO_RUN      = int(args.epochs)
    mrcnn_config.FCN_INPUT_SHAPE    = mrcnn_config.IMAGE_SHAPE[0:2]
    # mrcnn_config.LAST_EPOCH_RAN     = int(args.last_epoch)
    mrcnn_config.NEW_LOG_FOLDER       = args.new_log_folder
    # mrcnn_config.SYSOUT               = args.sysout

    #  Build mrcnn Model
    #------------------------------------------------------------------------------------
    
    from mrcnn.prep_notebook import mrcnn_coco_train
    mrcnn_model, mrcnn_config = mrcnn_coco_train(mode = 'inference', mrcnn_config = mrcnn_config)
    
    # Build configuration for FCN model
    #------------------------------------------------------------------------------------
    fcn_config = CocoConfig()
    fcn_config.COCO_DATASET_PATH  = paths.COCO_DATASET_PATH 
    fcn_config.COCO_HEATMAP_PATH  = paths.COCO_HEATMAP_PATH 

    # mrcnn_config.COCO_MODEL_PATH    = COCO_MODEL_PATH   
    # mrcnn_config.RESNET_MODEL_PATH  = RESNET_MODEL_PATH 
    fcn_config.NAME                 = 'fcn'              
    fcn_config.TRAINING_PATH        = paths.FCN_TRAINING_PATH
    fcn_config.VGG16_MODEL_PATH     = paths.FCN_VGG16_MODEL_PATH
    fcn_config.HEATMAP_SCALE_FACTOR = mrcnn_config.HEATMAP_SCALE_FACTOR
    
    fcn_config.FCN_INPUT_SHAPE      = fcn_config.IMAGE_SHAPE[0:2] // fcn_config.HEATMAP_SCALE_FACTOR 
    fcn_config.DETECTION_MIN_CONFIDENCE = mrcnn_config.DETECTION_MIN_CONFIDENCE
    fcn_config.DETECTION_MAX_INSTANCES  = mrcnn_config.DETECTION_MAX_INSTANCES 

    fcn_config.BATCH_SIZE           = int(args.batch_size)                 # Batch size is 2 (# GPUs * images/GPU).
    fcn_config.IMAGES_PER_GPU       = int(args.batch_size)                   # Must match BATCH_SIZE
    # fcn_config.EPOCHS_TO_RUN        = int(args.epochs)
    # fcn_config.STEPS_PER_EPOCH      = int(args.steps_in_epoch)
    # fcn_config.LAST_EPOCH_RAN       = int(args.last_epoch)

    # fcn_config.LEARNING_RATE        = float(args.lr)

    # fcn_config.VALIDATION_STEPS     = int(args.val_steps)
    fcn_config.BATCH_MOMENTUM       = 0.9
    # fcn_config.WEIGHT_DECAY         = 2.0e-4

    # fcn_config.REDUCE_LR_FACTOR     = 0.5
    # fcn_config.REDUCE_LR_COOLDOWN   = 5
    # fcn_config.REDUCE_LR_PATIENCE   = 5
    # fcn_config.EARLY_STOP_PATIENCE  = 15
    # fcn_config.EARLY_STOP_MIN_DELTA = 1.0e-4
    # fcn_config.MIN_LR               = 1.0e-10
     
    fcn_config.NEW_LOG_FOLDER       = args.new_log_folder
    # fcn_config.OPTIMIZER            = args.opt
    fcn_config.SYSOUT               = args.sysout
    fcn_config.display()


    # Build FCN Model
    #------------------------------------------------------------------------------------
    try :
        del fcn_model
        gc.collect()
    except: 
        pass    
    fcn_model = fcn_modellib.FCN(mode="inference",  arch = args.fcn_arch, config=fcn_config)

    ####  Display FCN model info

    # fcn_model.config.display()  
    fcn_model.layer_info()
    
    # Load MRCNN Model weights  
    #------------------------------------------------------------------------------------
    # exclude=["mrcnn_class_logits"] # ,"mrcnn_bbox_fc"]   #, "mrcnn_bbox", "mrcnn_mask"])
    mrcnn_model.load_model_weights(init_with = 'last', exclude = None)  
    
    # Load FCN Model weights  
    #------------------------------------------------------------------------------------
    fcn_model.load_model_weights(init_with = fcn_weight_file)
    
    return mrcnn_model, fcn_model


from   mrcnn.datagen import load_image_gt


##------------------------------------------------------------------------------------    
## run fcn inference_pipeline() on data batch or list of image ids
##------------------------------------------------------------------------------------    
def run_fcn_inference_pipeline(mrcnn_model, fcn_model, dataset, mrcnn_input = None, image_ids = None, verbose = 0):

    if image_ids is not None:
        image_batch = get_image_batch(dataset, image_ids, display = True)
        mrcnn_input, _ = get_inference_batch(dataset, mrcnn_model, image_ids)
    elif mrcnn_input is None:
        print('ERROR - mrcnn_input or image_ids must be passed to inference pipeline')
        assert len(mrcnn_input) == 2, 'Length of mrcnn_input list must be 2'
        return
        
    #--------------------------------------------------------------------------------------------
    # run_pipeline_on_input(mrcnn_model, fcn_model, mrcnn_input, verbose = verbose)
    # batch_x, _ = data_gen_simulate(dataset, config, image_list)
    # mrcnn_model.layer_info()
    # mrcnn_output_layers = [4,5,6,7]
    # fcn_output_layers = [0,1,2,3,4]    
    mrcnn_output = mrcnn_model.keras_model.get_layer_output_1( mrcnn_input, [0,1,2,3,4,5], 1, verbose = verbose)

    ### Load input / output data
    if verbose :
        print(' mrcnn outputs: ')
        print('----------------')
        print(' length of mrcnn output : ', len(mrcnn_output))
        for i,j in enumerate(mrcnn_output):
            print('mrcnn output ', i, ' shape: ' ,  j.shape)

    # fcn_model.layer_info()
    # fcn input / outputs in inference mode:
    # inputs  = [ pr_hm , pr_hm_scores]                                        
    # outputs = [ fcn_hm, fcn_sm, fcn_scores]
    fcn_input = [mrcnn_inmrcnn_output[4], mrcnn_output[5]]
    fcn_output = fcn_model.keras_model.get_layer_output_1( fcn_input, [0,1,2], 1, verbose = verbose)
    if verbose :
        print(' fcn outputs: ')
        print('----------------')
        for i,j in enumerate(fcn_input):
            print('fcn input ', i, ' shape: ' ,  j.shape)
        for i,j in enumerate(fcn_output):
            print('fcn output ', i, ' shape: ' ,  j.shape)        
            
    outputs = {}
    outputs['mrcnn_input']  = mrcnn_input
    outputs['fcn_input']    = fcn_input
    outputs['mrcnn_output'] = mrcnn_output
    outputs['fcn_output']   = fcn_output
    outputs['image_batch']  = image_batch        
    return outputs 
                
            
##------------------------------------------------------------------------------------    
## get_image_batch() 
##------------------------------------------------------------------------------------    
def get_image_batch(dataset, image_list, display = False):
    '''
    retrieves a list of image ids, that can be passed to model predict() functions
    '''
    
    images = []
    if not isinstance(image_list, list):
        image_list = [image_list]

    for image_id in image_list:
        images.append(dataset.load_image(image_id))

            
    # Mold inputs to format expected by the neural network
        
    if display:
        log("Loading {} images".format(len(images)))
        for image in images:
            log("image", image)
        titles = ['id: '+str(i)+' ' for i in image_list]
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
def get_inference_batch(dataset, mrcnn_model, image_ids, display = False):
    '''
    retrieves a list of image ids, that can be passed to model predict() functions
    '''
    
    if not isinstance(image_ids, list):
        image_ids = [image_ids]

    images= get_image_batch(dataset, image_ids, display = True)            

    # Mold inputs to format expected by the neural network
    molded_images, image_metas, windows = mrcnn_model.mold_inputs(images)
    
    # if verbose:
    log("Processing {} images".format(len(images)))
    for image in images:
        log("image", image)
    log("molded_images", molded_images)
    log("image_metas"  , image_metas)

    for img_id, img_meta in zip(image_ids, image_metas):
        img_meta[0] = img_id
        
    if display:
        titles = ['id: '+str(i)+' ' for i in image_metas]
        # visualize.display_images(images, titles = titles)        
        visualize.display_training_batch(dataset, [molded_images, image_metas], masks = True)
    return [molded_images, image_metas], images
