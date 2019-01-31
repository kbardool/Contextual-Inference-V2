# coding: utf-8
##-------------------------------------------------------------------------------------------
##
## MRCNN Pipeline (import model_mrcnn) - MRCNN model is operating in training mode 
##
## 
##-------------------------------------------------------------------------------------------
import os, sys, math, io, time, gc, platform, pprint
import numpy as np
import tensorflow as tf
import keras
import keras.backend as KB
pp = pprint.PrettyPrinter(indent=2, width=100)
np.set_printoptions(linewidth=100,precision=4,threshold=1000, suppress = True)

sys.path.append('..')
print(sys.path)

import mrcnn.model_mrcnn  as mrcnn_modellib
import mrcnn.model_fcn    as fcn_modellib
import mrcnn.visualize    as visualize

from datetime             import datetime   
from mrcnn.utils          import command_line_parser, display_input_parms, Paths
from mrcnn.newshapes      import prep_newshape_dataset 
from mrcnn.prep_notebook  import mrcnn_newshape_train, build_mrcnn_training_pipeline_newshapes
                                
start_time = datetime.now()
start_time_disp = start_time.strftime("%m-%d-%Y @ %H:%M:%S")
print()
print('--> Execution started at:', start_time)
print("    Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))

##------------------------------------------------------------------------------------
## Parse command line arguments
##------------------------------------------------------------------------------------
parser = command_line_parser()
args = parser.parse_args()
display_input_parms(args)

##----------------------------------------------------------------------------------------------
## if debug is true set stdout destination to stringIO
##----------------------------------------------------------------------------------------------            
if args.sysout in [ 'FILE', 'HEADER', 'ALL'] :
    sysout_name = "{:%Y%m%dT%H%M}_sysout.out".format(start_time)
    print('    Output is written to file....', sysout_name)    
    sys.stdout = io.StringIO()
    print()
    print('--> Execution started at:', start_time)
    print("    Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))
    display_input_parms(args)
    
##------------------------------------------------------------------------------------
## Build Mask RCNN Model in TRAINING mode
##------------------------------------------------------------------------------------
mrcnn_config = build_newshapes_config('mrcnn','training', args, verbose = 1)

try :
    del mrcnn_model
    print('delete model is successful')
    gc.collect()
except: 
    pass
KB.clear_session()
mrcnn_model = mrcnn_modellib.MaskRCNN(mode='training', config=mrcnn_config)

if args.sysout in ['ALL']:
   sysout_path = fcn_model.log_dir
   f_obj = open(os.path.join(sysout_path , sysout_name),'w' , buffering = 1 )
   content = sys.stdout.getvalue()   #.encode('utf_8')
   f_obj.write(content)
   sys.stdout = f_obj
   sys.stdout.flush()

##------------------------------------------------------------------------------------
## Display model configuration information
##------------------------------------------------------------------------------------
mrcnn_model.config.display()  
mrcnn_model.display_layer_info()

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

##------------------------------------------------------------------------------------
## Build & Load Training and Validation datasets
##------------------------------------------------------------------------------------
dataset_train = prep_newshape_dataset(mrcnn_model.config, 10000)
dataset_val   = prep_newshape_dataset(mrcnn_model.config,  2500)


##----------------------------------------------------------------------------------------------
##  Training
## 
## Train in two stages:
## 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly 
##    initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). 
##    To train only the head layers, pass `layers='heads'` to the `train()` function.
##    Passing layers="heads" freezes all layers except the head layers. 
##    You can also pass a regular expression to select which layers to train by name pattern.
## 
## 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to 
##    show the process. Simply pass `layers="all` to train all layers.
## ## Training head using  Keras.model.fit_generator()
##----------------------------------------------------------------------------------------------
# train_layers = [ 'mrcnn', 'fpn','rpn']
# loss_names   = [ "mrcnn_class_loss", "mrcnn_bbox_loss"]
train_layers = args.mrcnn_layers
loss_names   = [ "rpn_class_loss", "rpn_bbox_loss" , "mrcnn_class_loss", "mrcnn_bbox_loss"]
mrcnn_model.epoch = mrcnn_model.config.LAST_EPOCH_RAN

mrcnn_model.train(dataset_train, 
            dataset_val, 
            learning_rate = mrcnn_model.config.LEARNING_RATE, 
            epochs_to_run = mrcnn_model.config.EPOCHS_TO_RUN,
            layers = train_layers,
            losses = loss_names,
            sysout_name = sysout_name)
            
##----------------------------------------------------------------------------------------------
## If in debug mode write stdout intercepted IO to output file
##----------------------------------------------------------------------------------------------            
end_time = datetime.now().strftime("%m-%d-%Y @ %H:%M:%S")
if args.sysout in  ['ALL']:
    print(' --> Execution ended at:', end_time)
    sys.stdout.flush()
    f_obj.close()    
    sys.stdout = sys.__stdout__
    print(' Run information written to ', sysout_name)    
 
print(' --> Execution ended at:',end_time)
exit(' Execution terminated ' ) 
            
