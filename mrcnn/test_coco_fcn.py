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

import mrcnn.model_mrcnn as mrcnn_modellib
import mrcnn.model_fcn   as fcn_modellib
import mrcnn.visualize   as visualize
import mrcnn.utils       as utils 
import warnings

from datetime            import datetime   
from mrcnn.datagen       import data_generator, load_image_gt
from mrcnn.coco          import prep_coco_dataset
from mrcnn.utils         import command_line_parser, display_input_parms, Paths
from mrcnn.prep_notebook import build_coco_config, build_fcn_inference_pipeline, run_fcn_detection
from mrcnn.calculate_map import update_map_dictionaries

pp = pprint.PrettyPrinter(indent=2, width=100)
np.set_printoptions(linewidth=100,precision=4,threshold=1000, suppress = True)
os_platform = platform.system()
warnings.filterwarnings('ignore', '.*output shape of zoom.*')
start_time = datetime.now().strftime("%m-%d-%Y @ %H:%M:%S")
print()
print('args: ', sys.argv)
print('--> Execution started at:', start_time)
print("    Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))

##------------------------------------------------------------------------------------
## Parse command line arguments
##------------------------------------------------------------------------------------
parser = command_line_parser()
input_parms = " --batch_size 1  "
input_parms +=" --mrcnn_logs_dir train_mrcnn_coco_subset "
input_parms +=" --fcn_logs_dir   train_fcn8L2_BCE_subset "
input_parms +=" --fcn_model      last "
input_parms +=" --fcn_layer      all"
input_parms +=" --fcn_arch       fcn8L2 " 
input_parms +=" --sysout         screen "
input_parms +=" --scale_factor   4"
input_parms +=" --coco_classes   78 79 80 81 82 44 46 47 48 49 50 51 34 35 36 37 38 39 40 41 42 43 10 11 13 14 15 "
# input_parms +="--fcn_model /home/kbardool/models/train_fcn_adagrad/shapes20180709T1732/fcn_shapes_1167.h5"

args = parser.parse_args(input_parms.split())
verbose = 0

syst = platform.system()
if syst == 'Windows':
    save_path = "E:/git_projs/MRCNN3/train_coco/test_results"
    # test_dataset = "E:/git_projs/MRCNN3/train_newshapes/newshapes_test_dataset_1000_B.pkl"
    # DIR_WEIGHTS =  'F:/models_coco/train_fcn8L2_BCE_subset/fcn20190123T0000' 
    DIR_WEIGHTS =  'F:/models_coco/train_fcn8L2_BCE_subset/fcn20190120T0000' 
elif syst == 'Linux':
    save_path = "/home/kbardool/mrcnn3/train_coco/test_results"
    # test_dataset = "/home/kbardool/mrcnn3/train_newshapes/newshapes_test_dataset_1000_B.pkl"
    # DIR_WEIGHTS =  '/home/kbardool/models_coco/train_fcn8L2_BCE_subset/fcn20190123T0000' 
    DIR_WEIGHTS =  '/home/kbardool/models_coco/train_fcn8L2_BCE_subset/fcn20190120T0000' 
else :
    raise Error('unrecognized system ')

print(' OS ' , syst, ' SAVE_PATH   : ', save_path)
print(' OS ' , syst, ' DIR_WEIGHTS : ', DIR_WEIGHTS)

files       = ['fcn_1612.h5', 'fcn_1673.h5', 'fcn_2330.h5', 'fcn_3348.h5',
               'fcn_3742.h5', 'fcn_3816.h5', 'fcn_4345.h5']

# files       = ['fcn_1065.h5', 'fcn_1095.h5', 'fcn_1108.h5']

# files   = ['fcn_0001.h5', 'fcn_0026.h5', 'fcn_0162.h5', 'fcn_0350.h5',
           # 'fcn_0584.h5', 'fcn_0657.h5', 'fcn_0950.h5']

# files   = ['fcn_0001.h5', 'fcn_0150.h5', 'fcn_0346.h5', 'fcn_0421.h5',
           # 'fcn_0450.h5', 'fcn_0482.h5', 'fcn_0521.h5', 'fcn_0610.h5',
           # 'fcn_0687.h5', 'fcn_0793.h5', 'fcn_0821.h5', 'fcn_0940.h5',
           # 'fcn_1012.h5', 'fcn_1127.h5', 'fcn_1644.h5', 'fcn_1776.h5',


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

mrcnn_model, fcn_model = build_fcn_inference_pipeline(args = args,verbose = 1)

##----------------------------------------------------------------------------------------------
## Build Newshapes test Dataset
##----------------------------------------------------------------------------------------------
dataset_test = prep_coco_dataset(["minival"], mrcnn_model.config, generator = False , 
                                 shuffle = False, 
                                 load_coco_classes=args.coco_classes,
                                 loadAnns='active_only')
class_names = dataset_test.class_names
print(len(dataset_test.image_ids), len(dataset_test.image_info))
dataset_test.display_active_class_info()

##----------------------------------------------------------------------------------------------
## AP_Results file
##----------------------------------------------------------------------------------------------
file_prefix = 'test'
file_date   = datetime.now().strftime("_%Y_%m_%d")
old_AP_results_file = file_prefix+"_AP_results"+file_date
new_AP_results_file = file_prefix+"_AP_results"+file_date

print('Path:' ,save_path, '     Old Filename: ', old_AP_results_file , 'New Filename: ', new_AP_results_file)

# All_APResults = {}
##--OR --#
with open(os.path.join(save_path, old_AP_results_file+'.pkl'), 'rb') as outfile:
    All_APResults = pickle.load(outfile)
print('Loaded : ',old_AP_results_file)
print('-'*50)

print(len(All_APResults.keys()))
for i in sorted(All_APResults):
    print(i, All_APResults[i]['Epochs'])
    

##----------------------------------------------------------------------------------------------
##  Initialize data structures 
##----------------------------------------------------------------------------------------------
orig_score = 5
norm_score = 8
alt_scr_0  = 11
alt_scr_1  = 14   # in MRCNN alt_scr_1 ans alt_scr_2 are the same
alt_scr_2  = 20
IMGS = 500
# shuffled_image_ids = np.copy(dataset_test.image_ids)
# np.random.shuffle(shuffled_image_ids)
# image_ids = np.random.choice(dataset_test.image_ids, 300)
image_ids = dataset_test.image_ids[:IMGS]
print(len(image_ids))

##----------------------------------------------------------------------------------------------
## Run detection process over all images 
##----------------------------------------------------------------------------------------------
for FILE_IDX in range(len(files)):
    weights_file = os.path.join(DIR_WEIGHTS  , files[FILE_IDX])
    if os.path.isfile(weights_file):
        print("Loading weights ", weights_file)
        fcn_model.load_model_weights(weights_file)
    else:
        print(" weights file ", weights_file, " not found, going to next one...")
        continue
    
    class_dict = []
    gt_dict = {}
    pr_dict = {}

    for a,b in zip(dataset_test.class_ids, dataset_test.class_names):
        class_dict.append({'id'   : int(a),
                           'name' : b,
                           'scores': [],
                           'bboxes': [],
                           'mrcnn_score_orig' : [],
                           'mrcnn_score_norm' : [], 
                           'mrcnn_score_0' : [],
                           'mrcnn_score_1' : [],
                           'mrcnn_score_2' : [],
                           'fcn_score_0' : [],
                           'fcn_score_1' : [],
                           'fcn_score_2' : [],                      
                          })

    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy. 
    MRCNN_AP_Orig = []
    MRCNN_AP_0 = [] 
    MRCNN_AP_1 = [] 
    MRCNN_AP_2 = []
    FCN_AP_0   = []
    FCN_AP_1   = []
    FCN_AP_2   = []

    for image_id in image_ids:
        # Load image and ground truth data
        if image_id % 25 == 0:
            print('==> Calculate AP for image_id : ', image_id)
        # Run object detection
        try:
            fcn_results = run_fcn_detection(fcn_model, mrcnn_model, dataset_test, image_id, verbose = 0)  
        except Exception as e :
            print('\n failure on mrcnn predict image id: {} '.format(image_id))
            print('\n Exception information:')
            print(str(e))
            continue
 
        gt_dict, pr_dict, class_dict = update_map_dictionaries(fcn_results, gt_dict,pr_dict, class_dict)

        r = fcn_results[0] 

        #   Compute  AP, precisions, recalls, overlaps
        AP, _, _, _= utils.compute_ap(r['gt_bboxes'], r['gt_class_ids'], r["molded_rois"], r["class_ids"], r["pr_scores"][:,orig_score])
        MRCNN_AP_Orig.append(AP)

        AP, _, _, _= utils.compute_ap(r['gt_bboxes'], r['gt_class_ids'], r["molded_rois"], r["class_ids"], r["pr_scores"][:,alt_scr_0])
        MRCNN_AP_0.append(AP)

        AP, _, _, _= utils.compute_ap(r['gt_bboxes'], r['gt_class_ids'], r["molded_rois"], r["class_ids"], r["fcn_scores"][:,alt_scr_0])
        FCN_AP_0.append(AP)

        AP, _, _, _= utils.compute_ap(r['gt_bboxes'], r['gt_class_ids'], r["molded_rois"], r["class_ids"], r["pr_scores"][:,alt_scr_1])
        MRCNN_AP_1.append(AP)

        AP, _, _, _= utils.compute_ap(r['gt_bboxes'], r['gt_class_ids'], r["molded_rois"], r["class_ids"], r["fcn_scores"][:,alt_scr_1])
        FCN_AP_1.append(AP)

        AP, _, _, _= utils.compute_ap(r['gt_bboxes'], r['gt_class_ids'], r["molded_rois"], r["class_ids"], r["pr_scores"][:,alt_scr_2])
        MRCNN_AP_2.append(AP)

        AP, _, _, _= utils.compute_ap(r['gt_bboxes'], r['gt_class_ids'], r["molded_rois"], r["class_ids"], r["fcn_scores"][:,alt_scr_2])
        FCN_AP_2.append(AP)

    print('complete')

    epochs = files[FILE_IDX].split('_')[1].replace('.h5','')
    print('AP Calcs complete for epoch:', epochs , ' Weight file:', weights_file)
    APResult = {}
    APResult['Filename']      =  weights_file  
    APResult['Epochs']        =  epochs
    APResult['MRCNN_AP_Orig'] =  MRCNN_AP_Orig
    APResult['MRCNN_AP_0'   ] =  MRCNN_AP_0   
    APResult['MRCNN_AP_1'   ] =  MRCNN_AP_1   
    APResult['MRCNN_AP_2'   ] =  MRCNN_AP_2   
    APResult['FCN_AP_0'     ] =  FCN_AP_0     
    APResult['FCN_AP_1'     ] =  FCN_AP_1     
    APResult['FCN_AP_2'     ] =  FCN_AP_2     
    All_APResults[weights_file] = APResult

    print(len(All_APResults.keys()))
    for i in sorted(All_APResults):
        print(i, All_APResults[i]['Epochs'])        
        
    ## Save AP_Results
    ##------------------------------------------------------------------------
    with open(os.path.join(save_path, new_AP_results_file+'.pkl'), 'wb') as outfile:
        pickle.dump(All_APResults, outfile)
    print(' ***  Saved AP_results for epoch:',  All_APResults[i]['Epochs'], ' Weight file:', i)
    print('      to ----> ', save_path,'    Filename: ', new_AP_results_file)

    ## Save Cls_info, pr_bboxes dictionaries
    ##------------------------------------------------------------------------
    cls_info_file = file_prefix+'_cls_info_epoch'+epochs+'_'+str(len(image_ids))
    pr_boxes_file = file_prefix+'_pr_bboxes_epoch'+epochs+'_'+str(len(image_ids))
    gt_boxes_file = file_prefix+'_gt_bboxes_epoch'+epochs+'_'+str(len(image_ids))
    print(' ***  Save to :', cls_info_file,' -- ', pr_boxes_file,' -- ', gt_boxes_file)

    with open(os.path.join(save_path, cls_info_file+'.pkl'), 'wb') as outfile:
        pickle.dump(class_dict, outfile)
    with open(os.path.join(save_path, gt_boxes_file+'.pkl'), 'wb') as outfile:
        pickle.dump(gt_dict, outfile)
    with open(os.path.join(save_path, pr_boxes_file+'.pkl'), 'wb') as outfile:
        pickle.dump(pr_dict, outfile)    
    print(' ***  Saves complete')         

    ##
    ## Display recent calculated reslts
    ##--------------------------------------------------------------------------
    np_format = {}
    np_format['float'] = lambda x: "%10.4f" % x
    np_format['int']   = lambda x: "%10d" % x
    np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =10000, formatter = np_format)

    print()
    print('After {} training epochs.\nWeight file: {}'.format(epochs, weights_file))
    print()
    print("{:6s} {:^10s} {:>13s} {:>13s} {:>13s} {:>13s} {:>13s} {:>13s} {:>13s}".format("Images", "Epochs", "MRCNN_AP_Orig", "MRCNN_AP_0", "FCN_AP_0", "MRCNN_AP_1", "FCN_AP_1", "MRCNN_AP_2", "FCN_AP_2"))
    print('-'*116)
    for LIMIT in [10,50,100,250,500]:
        print("{:<6d} {:^10s} {:13.5f} {:13.5f} {:13.5f} {:13.5f} {:13.5f} {:13.5f} {:13.5f}".format(LIMIT, epochs,
                np.mean(MRCNN_AP_Orig[:LIMIT]), 
                np.mean(MRCNN_AP_0[:LIMIT]), np.mean(FCN_AP_0[:LIMIT]), 
                np.mean(MRCNN_AP_1[:LIMIT]), np.mean(FCN_AP_1[:LIMIT]), 
                np.mean(MRCNN_AP_2[:LIMIT]), np.mean(FCN_AP_2[:LIMIT]) ))            
                
                
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

