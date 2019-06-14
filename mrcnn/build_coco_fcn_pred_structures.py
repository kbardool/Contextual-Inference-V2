# coding: utf-8
"""
  
   Run FCN detection on test dataset in TEST or EVALUATION MODE and produce
   AP, Ground Truth and Predition data structures to pass to build_map_structures
   
   Input:
      -  Model weight files 
      -  test dataset
      
   Outputs:  
   
      -  eval*_AP_Results.pkl               results of mAPs over indivdual images 
      -  eval*_cls_info_epochxxxx_500.pkl   Dictionary of all scores grouped by Class and Score
      -  eval*_gt_boxes_epochxxxx_500.pkl   GT annotations 
      -  eval*_pr_boxes_epochxxxx_500.pkl   Predicted classes and bounding boxes
  
"""
import os, sys, math, io, time, gc, platform, pprint, json, pickle
import numpy as np
import tensorflow as tf
import keras
import keras.backend as KB
sys.path.append('../')

import mrcnn.visualize   as     visualize
import mrcnn.utils       as     utils 
import warnings
from datetime            import datetime 

from mrcnn.coco          import prep_coco_dataset
from mrcnn.utils         import command_line_parser, display_input_parms, Paths
from mrcnn.prep_notebook import build_coco_config
from mrcnn.prep_notebook import run_fcn_detection, run_fcn_evaluation
from mrcnn.prep_notebook import build_fcn_inference_pipeline_coco, build_fcn_evaluate_pipeline_coco
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
# input_parms = " --batch_size               1  "
# input_parms +=" --mrcnn_logs_dir           train_mrcnn_coco_subset "
# input_parms +=" --fcn_logs_dir             train_fcn8L2_BCE_subset "
# input_parms +=" --fcn_model                last "
# input_parms +=" --fcn_layer                all"
# input_parms +=" --fcn_arch                 fcn8L2 " 
# input_parms +=" --fcn_losses               fcn_BCE_loss "
# input_parms +=" --fcn_bce_loss_method      1 "
# input_parms +=" --sysout                   screen "
# input_parms +=" --scale_factor             4"
# input_parms +=" --coco_classes   78 79 80 81 82 44 46 47 48 49 50 51 34 35 36 37 38 39 40 41 42 43 10 11 13 14 15 "

parser = command_line_parser()
parser.add_argument('--mode', required=True,
                    choices=['test', 'eval'],
                    type=str, 
                    metavar="<Test eval mode>",
                    help="<Test/Eval mode selection" )
args = parser.parse_args()
display_input_parms(args)

verbose = 0
eval_method = str(args.evaluate_method)
syst = platform.system()

if syst == 'Windows':
    save_path     = "E:/git_projs/MRCNN3/"
    # DIR_WEIGHTS = "F:/models_newshapes/train_fcn8L2_BCE2/fcn20190131T0000"
    # DIR_WEIGHTS = "F:/models_newshapes/train_fcn8_l2_newshapes/fcn20181224T0000"
elif syst == 'Linux':
    save_path     = "/home/kbardool/mrcnn3/"
    DIR_WEIGHTS   = "fcn20190120T0000"
    # DIR_WEIGHTS = "/home/kbardool/models_newshapes/train_fcn8L2_BCE2/fcn20190131T0000"     
    # DIR_WEIGHTS = "/home/kbardool/models_newshapes/train_fcn8L2_BCE/fcn20181224T0000" 
else :
    raise Error('unrecognized system ')

    
if args.dataset == 'newshapes':
    save_path   += 'train_newshapes/'
    test_dataset = "newshapes_test_dataset_1000_A.pkl"
elif args.dataset == 'newshapes2':
    save_path   += 'train_newshapes_2/'    
    test_dataset = "newshapes2_test_dataset_1000_A.pkl"
else:
    save_path += 'train_coco/'
    
    
if args.mode == 'eval':
    save_path += "BCE_eval_method"+eval_method+"_results"
    file_prefix = 'eval'+eval_method
    run_inference  = run_fcn_evaluation
    build_pipeline = build_fcn_evaluate_pipeline_coco
else:
    save_path += "BCE_test_results"
    file_prefix = 'test' 
    build_pipeline = build_fcn_inference_pipeline_coco
    run_inference  = run_fcn_detection
        
old_AP_results_file = file_prefix+"_AP_results_2019_03_29"

print(' OS ' , syst, ' SAVE_PATH    : ', save_path)
print(' OS ' , syst, ' DIR_WEIGHTS  : ', DIR_WEIGHTS)


           
# fcn_files = ['fcn_0001.h5', 'fcn_0010.h5', 'fcn_0026.h5', 'fcn_0162.h5',
             # 'fcn_0350.h5', 'fcn_0584.h5', 'fcn_0657.h5', 'fcn_0950.h5', 
             # 'fcn_1065.h5', 'fcn_1095.h5', 'fcn_1108.h5']

fcn_files = ['fcn_1612.h5', 'fcn_1673.h5', 'fcn_2330.h5', 'fcn_3348.h5',
             'fcn_3742.h5', 'fcn_3816.h5', 'fcn_4345.h5']

# files       = ['fcn_1065.h5', 'fcn_1095.h5', 'fcn_1108.h5']

# files   = ['fcn_0001.h5', 'fcn_050.h5', 'fcn_0346.h5', 'fcn_0421.h5',
           # 'fcn_0450.h5', 'fcn_0482.h5', 'fcn_0521.h5', 'fcn_0610.h5',
           # 'fcn_0687.h5', 'fcn_0793.h5', 'fcn_0821.h5', 'fcn_0940.h5',
           # 'fcn_1012.h5', 'fcn_1127.h5', 'fcn_1644.h5', 'fcn_1776.h5',


##----------------------------------------------------------------------------------------------
## if sysout is 'FILE'  set stdout destination to stringIO
##----------------------------------------------------------------------------------------------            
if args.sysout in [ 'FILE', 'HEADER', 'ALL'] :
    sysout_name = "{:%Y%m%dT%H%M}_sysout.out".format(start_time)
    print('    Output is written to file....', sysout_name)    
    sys.stdout = io.StringIO()
    print()
    print('--> Execution started at:', start_time)
    print("    Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))
    display_input_parms(args)

##----------------------------------------------------------------------------------------------
## Build MRCNN and FCN models
##----------------------------------------------------------------------------------------------
mrcnn_model, fcn_model = build_pipeline(args = args,verbose = 0)

mrcnn_model.config.display()
fcn_model.config.display()

##----------------------------------------------------------------------------------------------
## Build COCO test Dataset
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
file_date           = datetime.now().strftime("_%Y_%m_%d")
# old_AP_results_file = file_prefix+"_AP_results"+file_date
new_AP_results_file = file_prefix+"_AP_results"+file_date

print('Path:' ,save_path, '     Old Filename: ', old_AP_results_file,
      'New Filename: ', new_AP_results_file)

All_APResults = {}
##--OR --#
## with open(os.path.join(save_path, old_AP_results_file+'.pkl'), 'rb') as outfile:
##     All_APResults = pickle.load(outfile)
## print('Loaded : ',old_AP_results_file)
## print('-'*50)
## print(len(All_APResults.keys()))
## for i in sorted(All_APResults):
##    print(i, All_APResults[i]['Epochs'])

##----------------------------------------------------------------------------------------------
##  Initialize data structures 
##----------------------------------------------------------------------------------------------
orig_score = 5
norm_score = 8
alt_scr_0  = 11
alt_scr_1  = 14   # in MRCNN alt_scr_1 ans alt_scr_2 are the same
alt_scr_2  = 20
IMGS       = 500
# shuffled_image_ids = np.copy(dataset_test.image_ids)
# np.random.shuffle(shuffled_image_ids)
# image_ids = np.random.choice(dataset_test.image_ids, 300)
# image_ids = dataset_test.image_ids[50:100]


image_ids = dataset_test.image_ids[:IMGS]
print(len(image_ids))

DIR_WEIGHTS = os.path.join(fcn_model.config.TRAINING_PATH, DIR_WEIGHTS)
print('FCN training path : ', fcn_model.config.TRAINING_PATH)
print('FCN weights  dir  : ', DIR_WEIGHTS)

##----------------------------------------------------------------------------------------------
## Run detection process over all images 
##----------------------------------------------------------------------------------------------
for FILE_IDX in range(len(fcn_files)):
# for FILE_IDX in [-2,-1]:

    weights_file = os.path.join(DIR_WEIGHTS, fcn_files[FILE_IDX])
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
        class_dict.append({'id'                  : int(a),
                           'name'                : b,
                           'scores'              : [],
                           'bboxes'              : [],
                           'mrcnn_score_orig'    : [],
                           'mrcnn_score_norm'    : [], 
                           'mrcnn_score_0'       : [],
                           'mrcnn_score_1'       : [],
                           'mrcnn_score_2'       : [],
                           'mrcnn_score_1_norm'  : [],
                           'mrcnn_score_2_norm'  : [],
                           'fcn_score_0'         : [],
                           'fcn_score_1'         : [],
                           'fcn_score_2'         : [],                      
                           'fcn_score_1_norm'    : [],
                           'fcn_score_2_norm'    : []
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
            fcn_results = run_inference(fcn_model, mrcnn_model, dataset_test, image_id, verbose = 0)  
            print('==> Calculate AP for image_id : ', image_id, ' image meta : ', fcn_results[0]['image_meta'][:10])
        except Exception as e :
            print('\n failure on mrcnn predict image id: {} '.format(image_id))
            print('\n Exception information:')
            print(str(e))
            continue
        else:
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

    epochs = fcn_files[FILE_IDX].split('_')[1].replace('.h5','')
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
        
    ##------------------------------------------------------------------------
    ## Save AP_Results, Cls_info, pr_bboxes dictionaries
    ##------------------------------------------------------------------------
    with open(os.path.join(save_path, new_AP_results_file+'.pkl'), 'wb') as outfile:
        pickle.dump(All_APResults, outfile)
    print(' ***  Saved AP_results for epoch:',  All_APResults[i]['Epochs'], ' Weight file:', i)
    print('      to ----> ', save_path,'    Filename: ', new_AP_results_file)

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

    ##--------------------------------------------------------------------------
    ## Display recent calculated reslts
    ##--------------------------------------------------------------------------
    np_format = {}
    np_format['float'] = lambda x: "%10.4f" % x
    np_format['int']   = lambda x: "%10d" % x
    np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =10000, formatter = np_format)

    print()
    print('After {} training epochs.\nWeight file: {}'.format(epochs, weights_file))
    print()
    print("{:6s} {:^10s} {:>13s} {:>13s} {:>13s} {:>13s} {:>13s} {:>13s} {:>13s}".
          format("Images", "Epochs", "MRCNN_AP_Orig", "MRCNN_AP_0", "FCN_AP_0", "MRCNN_AP_1", "FCN_AP_1", "MRCNN_AP_2", "FCN_AP_2"))
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
