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
    return config

def build_newshapes_config( model = None, mode = 'training', args = None, verbose = 0):
    return config

    
    
#######################################################################################    
## NEW SHAPES model preparation
#######################################################################################
def build_mrcnn_training_pipeline_newshapes( args = None, mrcnn_config = None,  mode = 'training',verbose = 0):
    return mrcnn_model
 
def build_mrcnn_inference_pipeline_newshapes(args = None, mrcnn_config = None,  mode = 'inference', verbose = 0):
    return mrcnn_model

mrcnn_newshape_train = build_mrcnn_training_pipeline_newshapes
mrcnn_newshape_test  = build_mrcnn_inference_pipeline_newshapes

def build_fcn_training_pipeline_newshapes( args = None, mrcnn_config = None,  mode = 'training', verbose = 0):
    return mrcnn_model, fcn_model

def build_fcn_inference_pipeline_newshapes( args = None, mrcnn_config = None,  mode = 'inference', verbose = 0):
    return mrcnn_model, fcn_model

def build_fcn_evaluate_pipeline_newshapes( args = None, mrcnn_config = None,  mode = 'evaluate', verbose = 0):
    return mrcnn_model, fcn_model
    
#######################################################################################    
## MRCNN 
#######################################################################################
def build_mrcnn_training_pipeline(args, mrcnn_config = None, verbose = 0):
    return mrcnn_model

def build_mrcnn_inference_pipeline( args = None, mrcnn_config = None , verbose = 0):
    return mrcnn_model
    
def build_mrcnn_evaluate_pipeline( args, mrcnn_config = None , verbose = 0):
    return mrcnn_model

    
#######################################################################################    
## FCN
#######################################################################################
def build_fcn_training_pipeline( args = None, batch_size = 2, fcn_weight_file = 'last', fcn_train_dir = "train_fcn8_coco_adam", verbose = 0):
    return mrcnn_model, fcn_model
        
def build_fcn_inference_pipeline( args = None, mrcnn_config = None, fcn_config = None, mode = 'inference', verbose = 0):
    return mrcnn_model, fcn_model
    
def build_fcn_evaluate_pipeline( args = None, mrcnn_config = None, fcn_config = None,  mode = 'evaluate',  verbose = 0):
    return mrcnn_model, fcn_model
                                 

#######################################################################################    
## GET BATCH routines
#######################################################################################
def get_image_batch(dataset, image_ids = None, display = False):
    return images

def get_training_batch(dataset, config, image_ids, display = True, masks = False):
    return batch_x
        
def get_inference_batch(dataset, config, image_ids = None, generator = None, display = False):
    return [images, molded_images, image_metas]

def get_evaluate_batch(dataset, config, image_ids = None, generator = None, display = True, masks = False):
    return [images, batch_x[0], batch_x[1], batch_x[4], batch_x[5]]
    
    
#######################################################################################    
## RUN PIPELINE  routines
## RUNS Training/Evaluate/Inference batch through MRCNN or FCN 
## model using get_layer_outputs()
#######################################################################################
def run_mrcnn_training_pipeline(mrcnn_model, dataset, mrcnn_input = None, image_ids = None, verbose = 0):
    return outputs 
                    
def run_mrcnn_evaluate_pipeline(mrcnn_model, dataset, input = None, image_ids = None, verbose = 0):
    return outputs 
                
def run_mrcnn_inference_pipeline(mrcnn_model, dataset, input = None, image_ids = None, verbose = 0):
    return outputs     
            
def run_fcn_training_pipeline(fcn_model, fcn_input = None, verbose = 0):
    return outputs 
                            
def run_fcn_inference_pipeline(fcn_model, input , verbose = 0):
    return outputs 

       
##------------------------------------------------------------------------------------    
## Run Input batch or Image Ids through FCN inference pipeline using get_layer_outputs()
##------------------------------------------------------------------------------------    
def run_full_training_pipeline(mrcnn_model, fcn_model, dataset, mrcnn_input = None, image_ids = None, verbose = 0):
    return outputs 
    
        
##------------------------------------------------------------------------------------    
## Run Input batch or Image Ids through FCN inference pipeline using get_layer_outputs()
##------------------------------------------------------------------------------------    
def run_full_inference_pipeline(mrcnn_model, fcn_model, dataset, input = None, image_ids = None, verbose = 0):
    return outputs 
    
    
                
##------------------------------------------------------------------------------------    
## Run MRCNN detection on MRCNN Inference Pipeline
##------------------------------------------------------------------------------------            
def run_mrcnn_detection(mrcnn_model, dataset, image_ids = None, verbose = 0, display = False):
    return mrcnn_results     

##------------------------------------------------------------------------------------    
## Run FCN detection on a list of image ids
##------------------------------------------------------------------------------------            
def run_fcn_detection(fcn_model, mrcnn_model, dataset, image_ids = None, verbose = 0):
    return fcn_results

#######################################################################################    
## OLD SHAPES model preparation
#######################################################################################
def mrcnn_oldshapes_train(init_with = None, FCN_layers = False, batch_sz = 5, epoch_steps = 4, training_folder= "mrcnn_oldshape_training_logs"):
    return [model, dataset_train, dataset_val, train_generator, val_generator, config]                                 


##------------------------------------------------------------------------------------    
## Old Shapes TESTING
##------------------------------------------------------------------------------------    
def mrcnn_oldshapes_test(init_with = None, FCN_layers = False, batch_sz = 5, epoch_steps = 4, training_folder= "mrcnn_oldshape_test_logs"):
    return [model, dataset_test, test_generator, config]                                 
