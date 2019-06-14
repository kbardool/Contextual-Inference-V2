# coding: utf-8
##-------------------------------------------------------------------------------------------
##
## Separated MRCNN-FCN Pipeline (import model_mrcnn)
## Train FCN head only
##
##  Pass predicitions from MRCNN to use as training data for FCN
##-------------------------------------------------------------------------------------------
import os, sys, math, io, time, gc, platform, pprint, argparse
import numpy as np
import tensorflow as tf
import keras
import keras.backend as KB
sys.path.append('./')

import mrcnn.model_mrcnn  as mrcnn_modellib
import mrcnn.model_fcn    as fcn_modellib
import mrcnn.visualize    as visualize

from datetime             import datetime   
from mrcnn.config         import Config
from mrcnn.dataset        import Dataset 
from mrcnn.utils          import log, stack_tensors, stack_tensors_3d, write_stdout
from mrcnn.datagen        import data_generator, load_image_gt
from mrcnn.coco           import CocoDataset, CocoConfig, CocoInferenceConfig, evaluate_coco, build_coco_results
from mrcnn.prep_notebook  import mrcnn_coco_train, prep_coco_dataset
from mrcnn.utils          import Paths
                                
pp = pprint.PrettyPrinter(indent=2, width=100)
np.set_printoptions(linewidth=100,precision=4,threshold=1000, suppress = True)

##----------------------------------------------------------------------------------------------
## 
##---------------------------------------------------------------------------------------------- 
def build_heatmap_files(
          mrcnn_model,
          dataset, 
          iterations        = 5,
          start_from        = 0, 
          dest_path         = None):
   
    '''
    train_dataset:  Training Dataset objects.

    '''
    assert mrcnn_model.mode == "trainfcn", "Create model in training mode."
    log("Starting for  {} iterations - batch size of each iteration: {}".format(iterations, batch_size))
    log(" Output destination: {}".format(dest_path))
    tr_generator= data_generator(dataset, mrcnn_model.config, 
                                    shuffle=False,
                                    augment=False,
                                    batch_size= mrcnn_model.config.BATCH_SIZE,
                                    image_index = start_from)
    
    ## Start main loop
    epoch_idx = 0
    for  epoch_idx in range(iterations) :
        tm_start = time.time()
        
        train_batch_x, train_batch_y = next(tr_generator)
        print(' ==> mrcnn_model: step {} of {} iterations, image_id: {} '.format(epoch_idx, iterations, train_batch_x[1][:,0]))
        
        # print('   length of train_batch_x:', len(train_batch_x), ' number of things in batch x :', train_batch_x[1].shape)        
        # for i in train_batch_x:
            # print('       ', i.shape)
        # print('length of train_batch_y:', len(train_batch_y))

        # results = get_layer_output_1(mrcnn_model.keras_model, train_batch_x, [0,1,2,3], 1)
        
        results = mrcnn_model.keras_model.predict(train_batch_x) 
        
        # pr_hm_norm, gt_hm_norm, pr_hm_scores, gt_hm_scores = results[:4]              
        
        for i in range(batch_size):
            # print('  pr_hm_norm shape   :', results[0][i].shape)
            # print('  pr_hm_scores shape :', results[1][i].shape)
            # print('  gt_hm_norm shape   :', results[2][i].shape)
            # print('  gt_hm_scores shape :', results[3][i].shape)
            image_id = train_batch_x[1][i,0]

            coco_image_id = dataset.image_info[image_id]['id']
            coco_filename = os.path.basename(dataset.image_info[image_id]['path'])
            
            ## If we want to save the files with a sequence # 0,1,2,.... which is the index of dataset.image_info[index] use this:
            # filename = 'hm_{:012d}.npz'.format(image_id)
            
            ## If we want to use the coco_id as the file name, use the following:
            filename = 'hm_{:012d}.npz'.format(coco_image_id)
            
            print('  output: {}  image_id: {}  coco_image_id: {} coco_filename: {} output file: {}'.format(
                            i, image_id, coco_image_id, coco_filename, filename))
#             print('  output file: ',os.path.join(dest_path, filename))
            np.savez_compressed(os.path.join(dest_path, filename), 
                             input_image_meta=train_batch_x[1][i], 
                             pr_hm_norm   = results[0][i],
                             pr_hm_scores = results[1][i],
                             gt_hm_norm   = results[2][i],
                             gt_hm_scores = results[3][i],
                             coco_info    = np.array([coco_image_id, coco_filename])    )        
        tm_stop= time.time()            
        print(' ==> Elapsed time {:.4f}s #        of items in results: {} '.format(tm_stop - tm_start,len(train_batch_x)))

    print('Final : mrcnn_model epoch_idx{}   iterations {}'.format(epoch_idx, iterations))
    return    
##---------------------------------------------------------------------------
## main routine
##---------------------------------------------------------------------------

if __name__ == '__main__':

    start_time = datetime.now().strftime("%m-%d-%Y @ %H:%M:%S")
    print()
    print('--> Build heatmap npz files from MRCNN')
    print('--> Execution started at:', start_time)
    print("    Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))

    ##------------------------------------------------------------------------------------
    ## Parse command line arguments
    ##------------------------------------------------------------------------------------
    # parser = command_line_parser()
    parser = argparse.ArgumentParser(description='Train Mask R-CNN on MS COCO.')

                        
    parser.add_argument('--model', required=False,
                        default='last',
                        metavar="/path/to/weights.h5",
                        help="MRCNN model weights file: 'coco' , 'init' , or Path to weights .h5 file ")
    
    parser.add_argument('--datasets', required=False,
                        choices=['train', 'val35k', 'minival'],
                        nargs = '+',
                        type=str.lower, 
                        metavar="/path/to/weights.h5",
                        help="coco datasets to build: train, val35k, minival")
                        
    parser.add_argument('--output_dir', required=True,
                        default='train_heatmaps',
                        metavar="/MLDatasets/coco2014_heatmaps/train_heatmaps",
                        help='output directory (default=MLDatasets/coco2014_heatmaps/train_heatmaps)')

    parser.add_argument('--iterations', required=True,
                        default=1,  type = int,
                        metavar="<iterations to run>",
                        help='Number of iterations to run (default=1)')
                        
    parser.add_argument('--batch_size', required=False,
                        default=5, type = int,
                        metavar="<batch size>",
                        help='Number of data samples in each batch (default=5)')                    

    parser.add_argument('--start_from', required=False,
                        default=-1, type = int,
                        metavar="<last epoch ran>",
                        help='Starting image index -1 or n to start from image n+1')
                        
    parser.add_argument('--sysout', required=False,
                        choices=['SCREEN', 'FILE'],
                        default='screen', type=str.upper,
                        metavar="<sysout>",
                        help="sysout destination: 'screen' or 'file'")

    args = parser.parse_args()

    ##----------------------------------------------------------------------------------------------
    ## if debug is true set stdout destination to stringIO
    ##----------------------------------------------------------------------------------------------            
    print("    MRCNN Model        : ", args.model)
    print("    Output Directory   : ", args.output_dir)
    print("    Datasets           : ", args.datasets)
    print("    Iterations         : ", args.iterations)
    print("    Start from image # : ", args.start_from)
    print("    Batch Size         : ", args.batch_size)
    print("    Sysout             : ", args.sysout)
 
    if args.sysout == 'FILE':
        print(' Output is written to file....')
        sys.stdout = io.StringIO()

    ##------------------------------------------------------------------------------------
    ## setup project directories
    ##   DIR_ROOT         : Root directory of the project 
    ##   MODEL_DIR        : Directory to save logs and trained model
    ##   COCO_MODEL_PATH  : Path to COCO trained weights
    ##---------------------------------------------------------------------------------
    paths = Paths()
    paths.display()

    ##------------------------------------------------------------------------------------
    ## Build configuration object 
    ##------------------------------------------------------------------------------------
    mrcnn_config                    = CocoConfig()
    mrcnn_config.NAME               = 'mrcnn'              
    mrcnn_config.TRAINING_PATH      = paths.MRCNN_TRAINING_PATH
    mrcnn_config.COCO_DATASET_PATH  = paths.COCO_DATASET_PATH 
    mrcnn_config.COCO_MODEL_PATH    = paths.COCO_MODEL_PATH   
    mrcnn_config.RESNET_MODEL_PATH  = paths.RESNET_MODEL_PATH 
    mrcnn_config.VGG16_MODEL_PATH   = paths.VGG16_MODEL_PATH  
    mrcnn_config.COCO_CLASSES       = None 
    mrcnn_config.BATCH_SIZE         = int(args.batch_size)                  # Batch size is 2 (# GPUs * images/GPU).
    mrcnn_config.IMAGES_PER_GPU     = int(args.batch_size)                  # Must match BATCH_SIZE
    mrcnn_config.STEPS_PER_EPOCH    = 1     # int(args.steps_in_epoch)
    mrcnn_config.LEARNING_RATE      = 0.001 # float(args.lr)
    mrcnn_config.EPOCHS_TO_RUN      = 1     # int(args.epochs)
    mrcnn_config.LAST_EPOCH_RAN     = 0     # int(args.last_epoch)
    mrcnn_config.NEW_LOG_FOLDER     = False
    mrcnn_config.SYSOUT             = args.sysout
    mrcnn_config.display() 

    ##------------------------------------------------------------------------------------
    ## Build Mask RCNN Model in TRAINFCN mode
    ##------------------------------------------------------------------------------------
    try :
        del mrcnn_model
        print('delete model is successful')
        gc.collect()
    except: 
        pass
    KB.clear_session()
    mrcnn_model = mrcnn_modellib.MaskRCNN(mode='trainfcn', config=mrcnn_config)

    ##------------------------------------------------------------------------------------
    ## Load Mask RCNN Model Weight file
    ##------------------------------------------------------------------------------------
    exclude_list = []
    mrcnn_model.load_model_weights(init_with = args.model, exclude = exclude_list)   


    ##------------------------------------------------------------------------------------
    ## Display model configuration information
    ##------------------------------------------------------------------------------------
    mrcnn_config.display()  
    print()
    mrcnn_model.layer_info()
    print()

    ##------------------------------------------------------------------------------------
    ## Build & Load Training and Validation datasets
    ##------------------------------------------------------------------------------------
    dest_path = os.path.join(paths.DIR_DATASET, args.output_dir)
    print('Output destination folder : ', dest_path)
    iterations = args.iterations
    batch_size = args.batch_size
    start_from = args.start_from
    print(' Iterations: ', type(iterations), iterations)
    print(' batch Size: ', type(batch_size), batch_size)
    print(' StartFrom : ', type(start_from), start_from)
    # dataset = prep_coco_dataset(["train",  "val35k"], mrcnn_config)
    dataset = prep_coco_dataset(args.datasets, mrcnn_config)
    
    ##--------------------------------------------------------------------------------
    ## Call build routine
    ##--------------------------------------------------------------------------------
    build_heatmap_files(mrcnn_model, dataset, iterations= iterations, 
                        start_from = start_from, dest_path = dest_path)

    end_time = datetime.now().strftime("%m-%d-%Y @ %H:%M:%S")
    
    print()
    print('--> Build heatmap npz files from MRCNN')
    print('--> Execution ended at:', end_time)
    ##----------------------------------------------------------------------------------------------
    ## If in debug mode write stdout intercepted IO to output file  
    ##----------------------------------------------------------------------------------------------            
    if mrcnn_config.SYSOUT == 'FILE':
        sysout_file = os.path.join('./', "{:%Y%m%dT%H%M}".format(datetime.now()))
        write_stdout(sysout_file, '_sysout', sys.stdout )        
        sys.stdout = sys.__stdout__
        print(' Run information written to ', sysout_file+'_sysout.out')
        print('--> Build heatmap npz files from MRCNN')
        print('--> Execution ended at:', end_time)

    
    exit(0)