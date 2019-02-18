import sys,os, pprint, pickle, math, time, platform
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from   scipy import stats
from datetime            import datetime   

print('Current working dir: ', os.getcwd())
if '..' not in sys.path:
    print("appending '..' to sys.path")
    sys.path.append('..')
import mrcnn.calculate_map_2 as cmap


pp = pprint.PrettyPrinter(indent=2, width=100)
np.set_printoptions(linewidth=100,precision=4,threshold=1000, suppress = True)
start_time = datetime.now().strftime("%m-%d-%Y @ %H:%M:%S")
print()
print('--> Execution started at:', start_time)
# print("    Tensorflow Version: {}   Keras Version : {} ".format(tf.__version__,keras.__version__))

eval_method = '2'
file_prefix = 'test'
#file_prefix = 'eval'+eval_method 

syst = platform.system()
if syst == 'Windows':
#    MAP_PATH = "E:/git_projs/MRCNN3/train_newshapes/BCE3_eval_method"+eval_method+"_results"
    MAP_PATH = "E:/git_projs/MRCNN3/train_newshapes/BCE3_test_results"
elif syst == 'Linux':
#    MAP_PATH = "/home/kbardool/mrcnn3/train_newshapes/BCE3_eval_method"+eval_method+"_results"
    MAP_PATH = "/home/kbardool/mrcnn3/train_newshapes/BCE3_test_results"
else :
    raise Error('unrecognized system ')

print(' OS ' , syst, ' : ', MAP_PATH)

# files = ['fcn_0001.h5', 'fcn_0150.h5', 'fcn_0346.h5', 'fcn_0421.h5', 'fcn_0450.h5', 
#          'fcn_0521.h5', 'fcn_0687.h5', 'fcn_0793.h5', 'fcn_0821.h5', 'fcn_0940.h5', 
#          'fcn_1012.h5', 'fcn_1127.h5', 'fcn_1644.h5', 'fcn_1776.h5', 'fcn_1848.h5', 
#          'fcn_2017.h5', 'fcn_2084.h5']   #idx 13,14,15,16,17

# files       = ['fcn_0500.h5']

# Files for BCE2 - Training with loss only on one class (Class 3 - Sun)
# files   = ['fcn_0001.h5', 'fcn_0022.h5', 'fcn_0057.h5', 'fcn_0092.h5',
           # 'fcn_0101.h5', 'fcn_0220.h5', 'fcn_0290.h5', 'fcn_0304.h5',
           # 'fcn_0372.h5', 'fcn_0423.h5', 'fcn_0500.h5', 'fcn_0530.h5',
           # 'fcn_0578.h5', 'fcn_0648.h5']        
           
# Files for BCE3 - Training with newly designed heatmap layer(adding FPs to the class channels)
files  = ['fcn_0001.h5', 'fcn_0003.h5', 'fcn_0005.h5', 'fcn_0009.h5', 
          'fcn_0012.h5', 'fcn_0020.h5', 'fcn_0023.h5', 'fcn_0027.h5', 
          'fcn_0033.h5', 'fcn_0047.h5', 'fcn_0070.h5', 'fcn_0080.h5', 
          'fcn_0101.h5', 'fcn_0106.h5', 'fcn_0112.h5', 'fcn_0124.h5', 
          'fcn_0138.h5', 'fcn_0144.h5', 'fcn_0161.h5', 'fcn_0171.h5', 'fcn_0181.h5']
           
           
CLASS_NAMES = ['ALL CLASSES', 'person', 'car', 'sun','building', 'tree', 'cloud']
CLASS_IDS   = [1,2,3,4,5,6] 
SCORES      = [ 'mrcnn_score_orig', 'mrcnn_score_norm', 
                'mrcnn_score_0', 'mrcnn_score_1', 'mrcnn_score_1_norm', 'mrcnn_score_2', 'mrcnn_score_2_norm',
                'fcn_score_0'  , 'fcn_score_1'  , 'fcn_score_1_norm'  , 'fcn_score_2'  , 'fcn_score_2_norm']


for FILE_IDX in range(len(files)):

    epochs = files[FILE_IDX].split('_')[1].replace('.h5','')
    
    cls_info_file = file_prefix+'_cls_info_epoch' +epochs+'_500.pkl'
    pr_boxes_file = file_prefix+'_pr_bboxes_epoch'+epochs+'_500.pkl'
    gt_boxes_file = file_prefix+'_gt_bboxes_epoch'+epochs+'_500.pkl'
    map_info_file = file_prefix+'_map_info_epoch'+epochs+'_FIXED.pkl'
    
    
    # cls_info_file = 'test_cls_info_epoch' +epochs+'_500.pkl'
    # pr_boxes_file = 'test_pr_bboxes_epoch'+epochs+'_500.pkl'
    # gt_boxes_file = 'test_gt_bboxes_epoch'+epochs+'_500.pkl'
    # map_info_file = 'test_map_info_epoch' +epochs+'.pkl'
 
    with open(os.path.join(MAP_PATH, cls_info_file), 'rb') as infile:
        cls_info = pickle.load(infile)            
    with open(os.path.join(MAP_PATH, pr_boxes_file), 'rb') as infile:
        pr_boxes = pickle.load(infile)
    with open(os.path.join(MAP_PATH, gt_boxes_file), 'rb') as infile:
        gt_boxes = pickle.load(infile)
    print('loaded :', cls_info_file, '   ', pr_boxes_file, '    ', gt_boxes_file)    

     ## build gt_boxes_class, pr_boxes_class which only containing info for each class
    iou_thresholds = np.arange(0.20, 0.95, 0.05)
    
    all_data = gt_boxes_class = pr_boxes_class = {}
    gt_boxes_class, pr_boxes_class  = cmap.filter_by_class(gt_boxes, pr_boxes, class_ids= CLASS_IDS)

    all_data    = cmap.build_mAP_data_structure_by_class(gt_boxes_class, pr_boxes_class, CLASS_IDS, SCORES, iou_thresholds)
    all_data[0] = cmap.build_mAP_data_structure_combined(gt_boxes, pr_boxes, SCORES, iou_thresholds)

    print(all_data.keys())

    ### Save mAP data to file `map_info_epochxxxx.pkl` for future reuse

    with open(os.path.join(MAP_PATH, map_info_file), 'wb') as outfile:
        pickle.dump(all_data, outfile)            
    print(' ===> Process of weight file :', files[FILE_IDX], ' complete')
    print(' ===> Datafile :', map_info_file , ' saved')


print(' --> Execution ended at:',datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))
exit(' Execution terminated ' ) 
