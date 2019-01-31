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
from mrcnn.config       import Config
from mrcnn.dataset      import Dataset 
from mrcnn.utils        import log, stack_tensors, stack_tensors_3d, write_stdout
from mrcnn.datagen      import data_generator, load_image_gt
from mrcnn.coco         import prep_coco_dataset
from mrcnn.utils        import command_line_parser, display_input_parms, Paths
 
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
input_parms  =" --batch_size     1  "
input_parms +=" --mrcnn_logs_dir train_mrcnn_coco_subset "
# input_parms +=" --fcn_logs_dir   train_fcn8_subset " 
input_parms +=" --mrcnn_model    last "
# input_parms +=" --lr 0.00001     --val_steps 8 " 
# input_parms +=" --fcn_model      init "
# input_parms +=" --opt            adam "
# input_parms +=" --fcn_arch       fcn8 " 
# input_parms +=" --fcn_layers     all " 
input_parms +=" --sysout         screen "
input_parms +=" --coco_classes   78 79 80 81 82 44 46 47 48 49 50 51 34 35 36 37 38 39 40 41 42 43 10 11 13 14 15 "

# input_parms +=" --new_log_folder    "
args = parser.parse_args(input_parms.split())
verbose = 0

syst = platform.system()
if syst == 'Windows':
    save_path = "E:/git_projs/MRCNN3/train_coco/test_results"
    # test_dataset = "E:/git_projs/MRCNN3/train_newshapes/newshapes_test_dataset_1000_B.pkl"
elif syst == 'Linux':
    save_path = "/home/kbardool/mrcnn3/train_coco/test_results"
    # test_dataset = "/home/kbardool/mrcnn3/train_newshapes/newshapes_test_dataset_1000_B.pkl"
else :
    raise Error('unrecognized system ')

print(' OS ' , syst, ' : ', save_path)
# print(' OS ' , syst, ' : ', test_dataset)


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

mrcnn_config = build_coco_config(model = 'mrcnn', mode = 'inference', args = args)

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
mrcnn_config.display()  
mrcnn_model.display_layer_info()

##------------------------------------------------------------------------------------
## Load Mask RCNN Model Weight file
##------------------------------------------------------------------------------------
# exclude_list = ["mrcnn_class_logits"]
exclude_list = []
mrcnn_model.load_model_weights(init_with = args.mrcnn_model, exclude = exclude_list)   

##----------------------------------------------------------------------------------------------
## Build COCO Training and Validation Datasets
##----------------------------------------------------------------------------------------------
dataset_test  = prep_coco_dataset(["minival"], mrcnn_model.config, generator = False , 
                                   return_coco = False,load_coco_classes=args.coco_classes)

print(len(dataset_test.image_ids), len(dataset_test.image_info))
dataset_test.display_active_class_info()

predicted_classes = []
ground_truth_bboxes = {}
predicted_bboxes    = {}

for a,b in zip(dataset_test.class_ids, dataset_test.class_names):
    predicted_classes.append({'id'    : int(a),
                              'name'  : b,
                              'scores': [],
                              'bboxes': []})

                         
##----------------------------------------------------------------------------------------------
## Run detection process over all images 
##----------------------------------------------------------------------------------------------
num_images = min(len(dataset_test.image_ids), int(sys.argv[1]))

print('Processing {:d} images ......'.format(num_images))

for image_id in range(num_images):
    keyname = os.path.split(dataset_test.image_info[image_id]['path'])[1]
    # if image_id % 25 == 0:
    print('image id :', image_id, 'filename :', keyname)

    image = dataset_test.load_image(image_id)
    _, image_meta, gt_class_ids, gt_bboxes =\
            load_image_gt(dataset_test, mrcnn_model.config, image_id, use_mini_mask=False)

    results = mrcnn_model.detect([image], verbose= 0)
    r = results[0]    
    
    ground_truth_bboxes[keyname] = {"boxes" : gt_bboxes.tolist(),
                                    "class_ids" : gt_class_ids.tolist()}
    
    predicted_bboxes[keyname] =  {"boxes" : r['molded_rois'].tolist(),
                                    "scores" : r['scores'].tolist(),
                                    "class_ids" : r['class_ids'].tolist()}
                                    
    for cls, score, bbox in zip(r['class_ids'], r['scores'], r['molded_rois'].tolist()):
        predicted_classes[cls]['scores'].append(float(score))
        predicted_classes[cls]['bboxes'].append(bbox)

    # for cls, bbox in zip(gt_class_ids,gt_bboxes.tolist()):
        # print(cls, bbox)

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
with open('predicted_classes_info.txt', 'w') as outfile:
    json.dump(predicted_classes, outfile)
with open('ground_truth_bboxes.txt', 'w') as outfile:
    json.dump(ground_truth_bboxes, outfile)
with open('predicted_bboxes.txt', 'w') as outfile:
    json.dump(predicted_bboxes, outfile)

with open('predicted_classes_info.pkl', 'wb') as outfile:
    pickle.dump(predicted_classes, outfile)
with open('ground_truth_bboxes.pkl', 'wb') as outfile:
    pickle.dump(ground_truth_bboxes, outfile)
with open('predicted_bboxes.pkl', 'wb') as outfile:
    pickle.dump(predicted_bboxes, outfile)
    
print(' --> Execution ended at:',datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))
exit(' Execution terminated ' ) 

