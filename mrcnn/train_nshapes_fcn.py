# coding: utf-8
##-------------------------------------------------------------------------------------------
##
## MRCNN-FCN Pipeline - Train FCN 
## Pass predicitions from MRCNN to use as training data for FCN
##
##-------------------------------------------------------------------------------------------
import os, sys, math, io, time, gc, platform, pprint, pickle
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
from mrcnn.prep_notebook  import mrcnn_newshape_train, build_newshapes_config

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
## Build Mask RCNN Model in TRAINFCN mode
##------------------------------------------------------------------------------------
mrcnn_config = build_newshapes_config('mrcnn','trainfcn', args, verbose = 1)

try :
    del mrcnn_model
    print('delete model is successful')
    gc.collect()
except: 
    pass
KB.clear_session()
mrcnn_model = mrcnn_modellib.MaskRCNN(mode='trainfcn', config=mrcnn_config)

##------------------------------------------------------------------------------------
## Display model configuration information
##------------------------------------------------------------------------------------
mrcnn_model.config.display()  
mrcnn_model.display_layer_info()

##------------------------------------------------------------------------------------
## Build configuration for FCN model
##------------------------------------------------------------------------------------
fcn_config = build_newshapes_config('fcn','training', args, verbose = 1)

##------------------------------------------------------------------------------------
## Build FCN Model in Training Mode
##------------------------------------------------------------------------------------
try :
    del fcn_model
    gc.collect()
except: 
    pass 
fcn_model = fcn_modellib.FCN(mode="training", arch = args.fcn_arch, config=fcn_config)

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
fcn_config.display()  
fcn_model.display_layer_info()

##------------------------------------------------------------------------------------
## Load MRCNN Model Weight file
##------------------------------------------------------------------------------------
exclude_list = []
mrcnn_model.load_model_weights(init_with = args.mrcnn_model, exclude = exclude_list)   

##------------------------------------------------------------------------------------
## Load FCN Model weights  
##------------------------------------------------------------------------------------
if args.fcn_model != 'init':
    fcn_model.load_model_weights(init_with = args.fcn_model, verbose = 1)
else:
    print(' FCN Training starting from randomly initialized weights ...')

##------------------------------------------------------------------------------------
## Build & Load Training and Validation datasets
##------------------------------------------------------------------------------------
# dataset_train = prep_newshape_dataset(mrcnn_model.config, 10000)
# dataset_val   = prep_newshape_dataset(mrcnn_model.config,  2500)
with open('E:\\git_projs\\MRCNN3\\train_newshapes\\newshapes_training_dataset_10000_A.pkl', 'rb') as outfile:
    dataset_train = pickle.load(outfile)
with open('E:\\git_projs\\MRCNN3\\train_newshapes\\newshapes_validation_dataset_2500_A.pkl', 'rb') as outfile:
    dataset_val = pickle.load(outfile)
    
print(' Training file size: ', len(dataset_train.image_ids), ' Validation file size: ', len(dataset_val.image_ids))    
dataset_train.display_active_class_info()
dataset_val.display_active_class_info()

##----------------------------------------------------------------------------------------------
## Train the FCN only 
## Passing layers="heads" freezes all layers except the head
## layers. You can also pass a regular expression to select
## which layers to train by name pattern.
##----------------------------------------------------------------------------------------------            
train_layers   = fcn_model.config.TRAINING_LAYERS
loss_names     = fcn_model.config.TRAINING_LOSSES  ## ['fcn_BCE_loss']
fcn_model.epoch = fcn_config.LAST_EPOCH_RAN

fcn_model.train_in_batches(
            mrcnn_model,    
            dataset_train,
            dataset_val, 
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

