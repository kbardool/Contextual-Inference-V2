# coding: utf-8
##-------------------------------------------------------------------------------------------
##
## Combined MRCNN-FCN Pipeline (import model_mrcnn) on COCO dataset
## Train MRCNN heads only
## MRCNN modeo (include model_mrcnn) does not include any mask related heads or losses 
##
##-------------------------------------------------------------------------------------------
import os, sys, math, io, time, gc, platform, pprint, json, pickle
import numpy as np
import tensorflow as tf
import keras
import keras.backend as KB
sys.path.append('../')

import mrcnn.model_mrcnn  as mrcnn_modellib
import mrcnn.model_fcn    as fcn_modellib
import mrcnn.visualize    as visualize

from datetime           import datetime   
from mrcnn.datagen      import data_generator, load_image_gt
from mrcnn.newshapes    import prep_newshape_dataset
from mrcnn.utils        import command_line_parser, display_input_parms, Paths
from mrcnn.prep_notebook import build_newshapes_config

pp = pprint.PrettyPrinter(indent=2, width=100)
np.set_printoptions(linewidth=100,precision=4,threshold=1000, suppress = True)
os_platform = platform.system()

start_time = datetime.now().strftime("%m-%d-%Y @ %H:%M:%S")
print()
print('args: ', sys.argv)
print('--> Execution started at:', start_time)
print("    Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))

##------------------------------------------------------------------------------------
## Parse command line arguments
##------------------------------------------------------------------------------------
parser = command_line_parser()

args = parser.parse_args()
verbose = 0

##----------------------------------------------------------------------------------------------
## if debug is true set stdout destination to stringIO
##----------------------------------------------------------------------------------------------            
display_input_parms(args)

if args.sysout == 'FILE':
    print('    Output is written to file....')
    sys.stdout = io.StringIO()
    print()
    print('--> Execution started at:', start_time)
    print("    Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))
    display_input_parms(args)

##------------------------------------------------------------------------------------
## Build configuration object 
##------------------------------------------------------------------------------------
mrcnn_config = build_newshapes_config('mrcnn', 'inference' , args, verbose = verbose)

##------------------------------------------------------------------------------------
## Build Mask RCNN Model in INFERENCE mode
##------------------------------------------------------------------------------------
try :
    del mrcnn_model
    print('delete model is successful')
    gc.collect()
except: 
    pass
KB.clear_session()
mrcnn_model = mrcnn_modellib.MaskRCNN(mode='inference', config=mrcnn_config)

##------------------------------------------------------------------------------------
## Display model configuration information
##------------------------------------------------------------------------------------
mrcnn_model.config.display()  
mrcnn_model.display_layer_info()

##------------------------------------------------------------------------------------
## Load Mask RCNN Model Weight file
##------------------------------------------------------------------------------------
exclude_list = []
mrcnn_model.load_model_weights(init_with = args.mrcnn_model, exclude = exclude_list)   

##----------------------------------------------------------------------------------------------
## Build Newshapes test Dataset
##----------------------------------------------------------------------------------------------
# dataset_test , test_generator   = prep_newshape_dataset( mrcnn_model.config,  1000, generator=True)
# with open(os.path.join(mrcnn_model.config.DIR_DATASET, 'newshapes_test_dataset_1000_B.pkl', 'rb') as outfile:
# with open(os.path.join(mrcnn_model.config.DIR_DATASET, 'newshapes_training_dataset_10000_A.pkl', 'rb') as outfile:
# with open(os.path.join(mrcnn_model.config.DIR_DATASET, 'newshapes2_training_dataset_15000_A.pkl'), 'rb') as outfile:
with open(os.path.join(mrcnn_model.config.DIR_DATASET, 'newshapes2_test_dataset_2500_A.pkl'), 'rb') as outfile:
    dataset_test = pickle.load(outfile)
    
class_names = dataset_test.class_names
print(len(dataset_test.image_ids), len(dataset_test.image_info))
dataset_test.display_active_class_info()

##----------------------------------------------------------------------------------------------
## setup data strutures for Class statistical info and GT/Prediction BBoxes
##----------------------------------------------------------------------------------------------
predicted_classes   = []
# ground_truth_bboxes = {}
# predicted_bboxes    = {}

for a,b in zip(dataset_test.class_ids, dataset_test.class_names):
    predicted_classes.append({'id'    : int(a),
                              'name'  : b,
                              'scores': [],
                              'bboxes': []})
                         
##----------------------------------------------------------------------------------------------
## Run detection process over all images 
##----------------------------------------------------------------------------------------------
num_images = len(dataset_test.image_ids)

print('Processing {:d} images ......'.format(num_images))

for image_id in range(num_images):
    keyname = 'newshapes_{:05d}'.format(image_id) 
    if image_id % 25 == 0:
        print('image id :', image_id, 'filename :', keyname)

    image = dataset_test.load_image(image_id)
    _, image_meta, gt_class_ids, gt_bboxes =\
            load_image_gt(dataset_test, mrcnn_model.config, image_id, use_mini_mask=False)

    results = mrcnn_model.detect([image], verbose= 0)
    r = results[0]    
    
    for cls, score, bbox in zip(r['class_ids'].tolist(), r['scores'].tolist(), r['molded_rois'].tolist()):
        predicted_classes[cls]['scores'].append(np.round(score,4))
        predicted_classes[cls]['bboxes'].append(bbox)
        

##----------------------------------------------------------------------------------------------
## add score average and quantiles to scores for each class         
##----------------------------------------------------------------------------------------------
for cls in predicted_classes:
    if (len(cls['scores']) == 0 ):
        cls['avg'] = 0.0000
        cls['percentiles'] = [0.0000, 0.0000, 0.0000] 
    else:
        cls['avg'] = np.round(np.mean(cls['scores']),4)
        cls['percentiles'] = np.round(np.percentile(cls['scores'],(25,50,75)),4).tolist()
        
##----------------------------------------------------------------------------------------------
## Write gt and prediction info to json files
##----------------------------------------------------------------------------------------------
filename = args.dataset+'_predicted_classes_info'
with open(filename+'.txt', 'w') as outfile:
    json.dump(predicted_classes, outfile)

with open(filename+'.pkl', 'wb') as outfile:
    pickle.dump(predicted_classes, outfile)
    
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
    
print(' --> Execution ended at:',datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))
exit(' Execution terminated ' ) 