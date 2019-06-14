"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import math
import pickle
import random
import itertools
import colorsys
import numpy as np
import IPython.display
from scipy import interpolate
import tensorflow as tf
import keras.backend as KB
import matplotlib.pyplot as plt
from   mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import matplotlib.lines as lines
import skimage.util
from   skimage.measure import find_contours
from   PIL  import Image
from   matplotlib.patches import Polygon
from   matplotlib import cm
from   matplotlib.ticker import LinearLocator, FormatStrFormatter
import mrcnn.utils as utils
import mrcnn.visualize as visualize

CLASS_COLUMN        = 4
ORIG_SCORE_COLUMN   = 5
DT_TYPE_COLUMN      = 6
SEQUENCE_COLUMN     = 7
NORM_SCORE_COLUMN   = 8    
SCORE_0_SUM_COLUMN  = 9
SCORE_0_AREA_COLUMN = 10
SCORE_0_COLUMN      = 11
SCORE_1_SUM_COLUMN  = 12
SCORE_1_AREA_COLUMN = 13
SCORE_1_COLUMN      = 14 
SCORE_1_NORM_COLUMN = 17
SCORE_2_SUM_COLUMN  = 18
SCORE_2_AREA_COLUMN = 19
SCORE_2_COLUMN      = 20
SCORE_2_NORM_COLUMN = 23


##-----------------------------------------------------------------------------------------------------------    
##  DISPLAY xxxx_AP_results FILE and AP_Results data struct 
##-----------------------------------------------------------------------------------------------------------    
def display_AP_file(map_file):
    with open(map_file, 'rb') as outfile:
        APRes = pickle.load(outfile)
    print()
    for i in sorted(APRes):
        print(i, APRes[i]['Epochs'])
    display_AP_results(APRes)
    
def display_AP_results(APRes):    
    for key in sorted(APRes) :
        print('\n\nAfter {} training epochs.\nWeight file: {}'.format( APRes[key]['Epochs'],APRes[key]['Filename']))
        print("\n{:6s} {:^10s} {:>13s} {:>13s} {:>13s} {:>13s} {:>13s} {:>13s} {:>13s}".
             format("Images", "Epochs", "MRCNN_AP_Orig", "MRCNN_AP_0", "FCN_AP_0", "MRCNN_AP_1", "FCN_AP_1", "MRCNN_AP_2", "FCN_AP_2"))
        print('-'*116)
        for LIMIT in [10,50,100,250,500]:
            print("{:<6d} {:^10s} {:13.5f} {:13.5f} {:13.5f} {:13.5f} {:13.5f} {:13.5f} {:13.5f}".format(LIMIT, APRes[key]['Epochs'],
                    np.mean(APRes[key]['MRCNN_AP_Orig'][:LIMIT]), 
                    np.mean(APRes[key]['MRCNN_AP_0'][:LIMIT]),
                    np.mean(APRes[key]['FCN_AP_0'][:LIMIT]), 
                    np.mean(APRes[key]['MRCNN_AP_1'][:LIMIT]),
                    np.mean(APRes[key]['FCN_AP_1'][:LIMIT]), 
                    np.mean(APRes[key]['MRCNN_AP_2'][:LIMIT]), 
                    np.mean(APRes[key]['FCN_AP_2'][:LIMIT]) ))
        print('\n')                    
        
        
        
##-----------------------------------------------------------------------------------------------------------    
##  DISPLAY GROUND TRUTH BOUNDING BOXES 
##-----------------------------------------------------------------------------------------------------------    
def display_gt_bboxes(dataset, config, image_id=0, only = None, size = 8, verbose = False):
    from   mrcnn.datagen     import load_image_gt    

    p_original_image, p_image_meta, p_gt_class_ids, p_gt_bboxes =  \
                load_image_gt(dataset, config, image_id, augment=False, use_mini_mask=True)
    if only is None:
        only = np.unique(p_gt_class_ids).astype(np.int)
    print()
    print('GT_BOXES for image ', image_id)
    print('-'*80)
    print('        class             |                              ')
    print('seq  id name              |  Y1  X1  Y2  X2     CX     CY    AREA')
    print('-'*80)
    idx = 0 
    for cls , pre in zip(p_gt_class_ids, p_gt_bboxes):
        cx = pre[1] + (pre[3]-pre[1])/2
        cy = pre[0] + (pre[2]-pre[0])/2
        area = (pre[3]-pre[1]) * (pre[2]-pre[0])
        if  cls in only:
            print('{:3.0f} {:2d} {:18s} |'\
                  ' {:3.0f} {:3.0f} {:3.0f} {:3.0f}   {:5.1f}  {:5.1f}  {:7.2f}'.          
              format(idx, cls, dataset.class_names[cls],
                     pre[0],pre[1], pre[2], pre[3], cx, cy, area ))     #,  pre[4],pre[5],pre[6],roi))
        idx +=1 
    print()                
    ttl = "Ground Truth Boxes for image "+str(image_id)
    visualize.display_instances(p_original_image, p_gt_bboxes, p_gt_class_ids, dataset.class_names,
                      only_classes = only, title=ttl, size = size)
                      
    return



##-----------------------------------------------------------------------------------------------------------    
## DISPLAY MRCNN SCORES 
##-----------------------------------------------------------------------------------------------------------    
def display_pr_scores(f_input, class_names, only = None, display = True, size = 12):
    '''
    f_input  :    results from detection process (run_mrcnn_detection)
    pr_scores:    pr_scores returned from detection or evaluation process results['pr_scores']
    cn:   class names
    
    '''
    np_format = {}
    np_format['float'] = lambda x: "%10.4f" % x
    np_format['int']   = lambda x: "%10d" % x
    np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =10000, formatter = np_format)

    if isinstance (f_input, dict) :
        pr_scores = f_input['pr_scores']
    else:
        pr_scores = f_input    
    
    if only is None:
        only = np.unique(pr_scores[:,4]).astype(np.int)

    seq_start = pr_scores.shape[1]
    print('PR_SCORES from fcn/mrcnn_results:')
    print('-'*175)
    print('                              |   |                |        MRCNN score 0       |          MRCNN score 1             |           MRCNN score 2            |                         ')
    print('          class               |TP/| mrcnn  normlzd |  gaussian   bbox   nrm.scr*|  ga.sum    mask     score   norm   |  ga.sum    mask     score   norm   |                         ')
    print('    seq  id name              | FP| score   score  |    sum      area   gau.sum |  in mask   sum              score  |  in mask   sum              score  |  X1  Y1  X2  Y2   AREA  ')
    print('-'*175)

    for idx, pre in enumerate(pr_scores):
        cx = pre[1] + (pre[3]-pre[1])/2
        cy = pre[0] + (pre[2]-pre[0])/2
        area = (pre[3]-pre[1]) * (pre[2]-pre[0])
        int_cls = int(pre[CLASS_COLUMN])
        if  int_cls in only:
            print('{:3.0f} {:3.0f} {:2d} {:18s} |{:2.0f} | {:.4f}  {:.4f} |'\
                  '{:9.4f}  {:6.1f} {:9.4f} |'\
                  '{:9.4f}  {:6.1f} {:9.4f}  {:6.4f} |'\
                  '{:9.4f}  {:6.1f} {:9.4f}  {:6.4f} |'\
                  ' {:3.0f} {:3.0f} {:3.0f} {:3.0f} {:7.2f}'.          
              format(idx,    
                     pre[SEQUENCE_COLUMN],
                     int_cls, class_names[int_cls], pre[DT_TYPE_COLUMN], pre[ORIG_SCORE_COLUMN], pre[NORM_SCORE_COLUMN], 
                     pre[SCORE_0_SUM_COLUMN], pre[SCORE_0_AREA_COLUMN], pre[SCORE_0_COLUMN], 
                     pre[SCORE_1_SUM_COLUMN], pre[SCORE_1_AREA_COLUMN], pre[SCORE_1_COLUMN], pre[SCORE_1_NORM_COLUMN], 
                     pre[SCORE_2_SUM_COLUMN], pre[SCORE_2_AREA_COLUMN], pre[SCORE_2_COLUMN], pre[SCORE_2_NORM_COLUMN], 
                     pre[1],pre[0], pre[3], pre[2], area ))     #,  pre[4],pre[5],pre[6],roi))
                     
    if isinstance (f_input, dict)  and display:
        img_id = str(f_input['image_meta'][0])        
        visualize.display_instances(f_input['image'] , pr_scores[:,:CLASS_COLUMN], pr_scores[:,CLASS_COLUMN].astype(np.int32), 
                                    class_names, pr_scores[:,ORIG_SCORE_COLUMN], only_classes= only, size =size,
                                    title = 'MRCNN predictions for image id'+img_id)
                                    
                                    
display_mrcnn_scores = display_pr_scores

#-----------------------------------------------------------------------------------------------------------    
#
#-----------------------------------------------------------------------------------------------------------        
def display_pr_hm_scores(r, class_names, only = None):
    '''
    pr_hm_scores:   pr_hm_scores or pr_hm_scores_by_class ( [class, bbox, score info] ) results from mrcnn detect
    class_names :   class names
    
    '''
    pr_hm_scores = r['pr_hm_scores']
    
    print(pr_hm_scores.shape)
    np_format = {}
    float_formatter = lambda x: "%10.4f" % x
    int_formatter   = lambda x: "%10d" % x
    np_format['float'] = float_formatter
    np_format['int']   = int_formatter
    np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =10000, formatter = np_format)
    
    if only is None:
        only = range(pr_hm_scores.shape[0])
#     seq_start = r['pr_hm_scores'].shape[1]
    # print('  class ids  : ', r['class_ids'], type(r['class_ids']))
    print('  classes     : ', only)
    print('-'*182)
    print('                         |  |                 |           alt score 0      |              alt score 1              |                alt score 2            |')
    print('        class            |  | mrcnn   normlzd |  gauss     bbox   nrm.scr* |  ga.sum     mask    score      norm   |   ga.sum     mask     score   norm    |')
    print('seq  id     name         |  | score   score   |  sum       area   gau.sum  |  in mask    sum                score  |   in mask    sum              score   |    CX      CY      AREA')
    print('-'*182)
    
    for cls in range(pr_hm_scores.shape[0]):
        if cls in only:
            for pre in pr_hm_scores[cls]:
                width = pre[3]-pre[1]
                height = pre[2]-pre[0]
                if width == height == 0:
                    continue
                cx = pre[1] + width/2
                cy = pre[0] + height/2
                area = (pre[3]-pre[1])* (pre[2]-pre[0])
                print('{:3.0f} {:3.0f} {:15s}  |{:2.0f}| {:.4f}   {:.4f} |'\
                      ' {:6.4f}  {:9.4f}  {:7.4f} |'\
                      ' {:8.4f}  {:8.4f}  {:7.4f}   {:7.4f} |'\
                      ' {:8.4f}  {:8.4f}  {:7.4f}   {:7.4f} |'\
                      ' {:6.2f}  {:6.2f}  {:9.4f}'.          
                format(pre[SEQUENCE_COLUMN], pre[CLASS_COLUMN], class_names[cls], pre[DT_TYPE_COLUMN], pre[ORIG_SCORE_COLUMN], pre[NORM_SCORE_COLUMN], 
                       pre[SCORE_0_SUM_COLUMN], pre[SCORE_0_AREA_COLUMN], pre[SCORE_0_COLUMN], 
                       pre[SCORE_1_SUM_COLUMN], pre[SCORE_1_AREA_COLUMN], pre[SCORE_1_COLUMN], pre[SCORE_1_NORM_COLUMN], 
                       pre[SCORE_2_SUM_COLUMN], pre[SCORE_2_AREA_COLUMN], pre[SCORE_2_COLUMN], pre[SCORE_2_NORM_COLUMN], 
                       cx, cy , area))     #,  pre[4],pre[5],pre[6],roi))
            # print('-'*170)
    return 
    
#-----------------------------------------------------------------------------------------------------------    
#
#-----------------------------------------------------------------------------------------------------------        
def display_pr_hm_scores_box_info(r, class_names, only = None):
    '''
    r:    results from mrcnn detect
    cn:   class names
    
    '''
    pr_hm_scores = r['pr_hm_scores']
    
    print(pr_hm_scores.shape)
    
    np_format = {}
    float_formatter = lambda x: "%10.4f" % x
    int_formatter   = lambda x: "%10d" % x
    np_format['float'] = float_formatter
    np_format['int']   = int_formatter
    np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =10000, formatter = np_format)
    
    if only is None:
        only = range(pr_hm_scores.shape[0])
#     seq_start = r['pr_hm_scores'].shape[1]
    # print('  class ids  : ', r['class_ids'], type(r['class_ids']))
    print('  classes     : ', only)
    print('-'*175)
    print('                          ')
    print('        class                mrcnn   normlzd        ')
    print('seq  id     name             score    score         X1/Y1                  X2/Y2               CX / CY        WIDTH   HEIGHT      AREA     CV_X   CV_Y')
    print('-'*175)
    for cls in range(pr_hm_scores.shape[0]):
        if cls in only:
            for pre in pr_hm_scores[cls]:
        #     print(' {:4d}      {:12s}   {:.4f}   {:.4f}     {:.4f}  {:7.4f}  {:.4f}      {:.4f}   {}'.
                width = pre[3]-pre[1]
                height = pre[2]-pre[0]
                cx = pre[1] + width/2
                cy = pre[0] + height/2
                covar_x = math.sqrt(width * 0.5)
                covar_y = math.sqrt(height * 0.5)
                area = (pre[3]-pre[1])* (pre[2]-pre[0])
                if width == height == 0:
                    continue
                #    (' seq      id     name       5        8           X1/Y1                X2/Y2              CX / CY         WIDTH   HEIGHT       AREA      CV_X      CV_Y')
                print('{:3.0f} {:3.0f} {:15s}   {:8.4f}  {:8.4f}  ({:7.2f}, {:7.2f})   ({:7.2f}, {:7.2f})   {:7.2f}/{:7.2f}   {:7.2f}   {:7.2f}   {:12.2f}   {:8.4f}   {:8.4f}'\
                  ' '.format(pre[SEQUENCE_COLUMN], pre[CLASS_COLUMN],  class_names[cls], pre[ORIG_SCORE_COLUMN], pre[NORM_SCORE_COLUMN],
                         pre[1], pre[0],  pre[3], pre[2], 
                         cx,  cy,  width,  height, area, covar_x, covar_y))    
    print('-'*170)
    return
    


##-----------------------------------------------------------------------------------------------------------    
## DISPLAY FCN SCORES
##-----------------------------------------------------------------------------------------------------------    
def display_fcn_scores(f_input, class_names, only = None, display = True, size = 12):
    '''
    fcn_scores:    fcn_scores returned from detection or evaluation process
    cn:   class names3
    '''
    np_format = {}
    np_format['float'] = lambda x: "%10.4f" % x
    np_format['int']   = lambda x: "%10d" % x
    np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =10000, formatter = np_format)
    if isinstance (f_input, dict) :
        fcn_scores = f_input['fcn_scores']
    else:
        fcn_scores = f_input    
        
        
    if only is None:
        only = np.unique(fcn_scores[:,4]).astype(np.int)
        
    seq_start = fcn_scores.shape[1]
    print('\nFCN_SCORES:')
    print('-'*175)
    print('                              |   |                |         FCN score 0        |            FCN score 1             |            FCN score 2             |                         ')
    print('          class               |TP/| mrcnn  normlzd |  gaussian   bbox   nrm.scr*|  ga.sum    mask     score   norm   |  ga.sum    mask     score   norm   |                         ')
    print('    seq  id name              | FP| score   score  |    sum      area   gau.sum |  in mask   sum              score  |  in mask   sum              score  |  Y1  X1  Y2  X2   AREA  ')
    print('-'*175)

    for idx, pre in enumerate(fcn_scores):
        cx = pre[1] + (pre[3]-pre[1])/2
        cy = pre[0] + (pre[2]-pre[0])/2
        area = (pre[3]-pre[1]) * (pre[2]-pre[0])
        int_cls = int(pre[CLASS_COLUMN])
        if int_cls in only:
            print('{:3.0f} {:3.0f} {:2d} {:18s} |{:2.0f} | {:.4f}  {:.4f} |'\
                  '{:9.4f}  {:6.1f} {:9.4f} |'\
                  '{:9.4f}  {:6.1f} {:9.4f}  {:6.4f} |'\
                  '{:9.4f}  {:6.1f} {:9.4f}  {:6.4f} |'\
                  ' {:3.0f} {:3.0f} {:3.0f} {:3.0f} {:7.2f}'.          
                format(idx,
                     pre[SEQUENCE_COLUMN], 
                     int_cls, class_names[int_cls], pre[DT_TYPE_COLUMN], pre[ORIG_SCORE_COLUMN], pre[NORM_SCORE_COLUMN], 
                     pre[SCORE_0_SUM_COLUMN], pre[SCORE_0_AREA_COLUMN], pre[SCORE_0_COLUMN], 
                     pre[SCORE_1_SUM_COLUMN], pre[SCORE_1_AREA_COLUMN], pre[SCORE_1_COLUMN], pre[SCORE_1_NORM_COLUMN], 
                     pre[SCORE_2_SUM_COLUMN], pre[SCORE_2_AREA_COLUMN], pre[SCORE_2_COLUMN], pre[SCORE_2_NORM_COLUMN],                   
                     pre[0],pre[1], pre[2], pre[3], area ))     #,  pre[4],pre[5],pre[6],roi))
    print()
        
    if isinstance (f_input, dict)  and display:
        img_id = str(f_input['image_meta'][0])
        visualize.display_instances(f_input['image'] , fcn_scores[:,:CLASS_COLUMN ], fcn_scores[:,CLASS_COLUMN].astype(np.int32), 
                                    class_names, fcn_scores[:,SCORE_1_COLUMN], only_classes= only, size =size,
                                    title = 'FCN predictions for image id '+img_id)

def display_fcn_scores_box_info(fcn_scores, fcn_hm = None, class_names = None, only = None):
    '''
    r:    results from mrcnn detect
    cn:   class names
    
    '''
    np_format = {}
    np_format['float'] = lambda x: "%10.4f" % x
    np_format['int']   = lambda x: "%10d" % x
    np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =10000, formatter = np_format)
    
    if only is None:
        only = np.unique(fcn_scores[:,4])
        
    seq_start = fcn_scores.shape[1]
#     seq_start = r['pr_hm_scores'].shape[1]
    # print('  class ids  : ', r['class_ids'], type(r['class_ids']))
    print('  FCN BBox Information - classes   --  : ', only)
    print('-'*175)
    print('                                                         |                                                 |    (COVAR)    |               CLIP REGION   ')
    print('BOX     class                                            |                   Width   Height                |      SQRT     |      FROM/TO               FROM/TO ')
    print('seq  id     name              Y1/X1              Y2/X2   |    CX / CY         (W)  ~  (H)      AREA        |  W/2     H/2  |  X1/Y1    X2/Y2   A |  X1/Y1   X2/Y2   A')
    print('-'*175)                                                                                                                                          
    
    
    for pre in fcn_scores:
#     print(' {:4d}      {:12s}   {:.4f}   {:.4f}     {:.4f}  {:7.4f}  {:.4f}      {:.4f}   {}'.
        cls = int(pre[4])
        if cls in only:

            width = pre[3]-pre[1]
            height = pre[2]-pre[0]
            cx = pre[1] + (width / 2.0)
            cy = pre[0] + (height/ 2.0)
            covar_x = width * 0.5
            covar_y = height * 0.5
            sq_covar_x = np.sqrt(covar_x)
            sq_covar_y = np.sqrt(covar_y)
            area = (pre[3]-pre[1])* (pre[2]-pre[0])
            from_x = int(round(cx - sq_covar_x))
            to_x   = int(round(cx + sq_covar_x))
            from_y = int(round(cy - sq_covar_y))
            to_y   = int(round(cy + sq_covar_y))
            clip_area = (to_x - from_x) * (to_y - from_y)
            
            from_x_r  = round(cx) - round(sq_covar_x)
            to_x_r    = round(cx) + round(sq_covar_x)
            from_y_r  = round(cy) - math.ceil(sq_covar_y)
            to_y_r    = round(cy) + math.ceil(sq_covar_y)
            clip_area_r = (to_x_r - from_x_r) * (to_y_r - from_y_r)
        
            print('{:3.0f} {:3.0f} {:15s}'\
              ' ({:6.2f},{:6.2f})  ({:6.2f},{:6.2f}) |'\
              ' {:6.2f}/{:6.2f}  {:7.2f}~{:7.2f}  {:7.2f} {:7.2f} |'\
              ' {:6.2f} {:6.2f} |'\
              ' {:3d} {:3d} {:3d} {:3d} {:3d} |'\
              ' {:3.0f} {:3.0f} {:3.0f} {:3.0f} {:3.0f}'.
              format(pre[SEQUENCE_COLUMN], pre[CLASS_COLUMN],  class_names[cls], 
                     pre[0], pre[1],  pre[2], pre[3], 
                     cx, cy,  width,  height, area, pre[13],
                     sq_covar_x, sq_covar_y, 
                     from_x, from_y,  to_x, to_y , clip_area,
                     from_x_r, from_y_r, to_x_r, to_y_r, clip_area_r
                  ))    
            print(' {:2s}   mrcnn_scr: {:6.4f}  norm: {:6.4f}  scr0: {:6.4f}  scr1: {:6.4f}  scr2: {:6.4f}'.
                format('TP' if pre[DT_TYPE_COLUMN] == 1 else 'FP', 
                       pre[ORIG_SCORE_COLUMN], 
                       pre[NORM_SCORE_COLUMN], 
                       pre[SCORE_0_COLUMN], 
                       pre[SCORE_1_COLUMN], 
                       pre[SCORE_2_COLUMN]))
            for i in range(1,fcn_hm.shape[-1]):
                print('  '*10, 'cls: {:3d}-{:10s}' \
                      '  cx/cy: {:8.3f}   cx/cy+-1: {:8.3f}   cx/cy+-3: {:8.3f}' \
                      '  fr/to: {:8.3f}   fr/to+1: {:8.3f}    fr-1/to+1: {:8.3f}   full: {:8.3f}'.format( i, class_names[i],
                            np.sum(fcn_hm[int(cx),int(cy),i]), 
                            np.sum(fcn_hm[int(cx)-1:int(cx)+1,int(cy)-1:int(cy)+1,i]), 
                            np.sum(fcn_hm[int(cx)-3:int(cx)+3,int(cy)-3:int(cy)+3,i]), 
                            np.sum(fcn_hm[from_y:to_y, from_x:to_x,i]), 
                            np.sum(fcn_hm[from_y:to_y+1, from_x:to_x+1,i]), 
                            np.sum(fcn_hm[from_y-1:to_y+1, from_x-1:to_x+1,i]), 
                            np.sum(fcn_hm[int(pre[1]):int(pre[3]), int(pre[0]):int(pre[2]),i])))
            print()
    print('-'*175)
    return
    

#-----------------------------------------------------------------------------------------------------------    
#
#-----------------------------------------------------------------------------------------------------------        
def display_fcn_scores_box_info2(fcn_scores, fcn_hm , class_names, only = None):
    '''
    r:    results from mrcnn detect
    cn:   class names
    
    '''
    np_format = {}
    float_formatter = lambda x: "%10.4f" % x
    int_formatter   = lambda x: "%10d" % x
    np_format['float'] = float_formatter
    np_format['int']   = int_formatter
    np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =10000, formatter = np_format)
    
    if only is None:
        only = np.unique(fcn_scores[:,4])
        
    seq_start   = fcn_scores.shape[1]
    num_classes = fcn_hm.shape[-1]
    class_list  = np.arange(num_classes) 
    sub_title   = ''.join([ '{:>15s}'.format(i) for i in class_names])
    
#     seq_start = r['pr_hm_scores'].shape[1]
    # print('  class ids  : ', r['class_ids'], type(r['class_ids']))
    print('  classes     : ', only)
    # print('-'*175)
    # print('                                       |                                              |   (COVAR)     |    ')
    # print('BOX                                    |                   Width   Height             |     SQRT      |   FROM/TO  ')
    # print('seq       X1/Y1              X2/Y2     |    CX / CY         (W)  ~  (H)      AREA     |  W/2    H/2   |  X      Y ')
    # print('-'*175)   
    
    for pre in fcn_scores:
        cls = int(pre[4])
        if cls in only:
            print('-'*175)
            print('                                       |                                              |   (COVAR)     |    ')
            print('BOX                                    |                   Width   Height             |     SQRT      |   FROM/TO  ')
            print('seq       X1/Y1              X2/Y2     |    CX / CY         (W)  ~  (H)      AREA     |  W/2    H/2   |  X      Y ')
            # print('-'*175)   
      #     print(' {:4d}      {:12s}   {:.4f}   {:.4f}     {:.4f}  {:7.4f}  {:.4f}      {:.4f}   {}'.
            width = pre[3]-pre[1]
            height = pre[2]-pre[0]
            cx = pre[1] + (width / 2.0)
            cy = pre[0] + (height/ 2.0)
            covar_x = width * 0.5
            covar_y = height * 0.5
            sq_covar_x = np.sqrt(covar_x)
            sq_covar_y = np.sqrt(covar_y)
            area = (pre[3]-pre[1])* (pre[2]-pre[0])
            from_x = int(round(cx - sq_covar_x))
            to_x   = int(round(cx + sq_covar_x))
            from_y = int(round(cy - sq_covar_y))
            to_y   = int(round(cy + sq_covar_y))
            clip_area = (to_x - from_x) * (to_y - from_y)
            
            from_x_r  = round(cx) - round(sq_covar_x)
            to_x_r    = round(cx) + round(sq_covar_x)
            from_y_r  = round(cy) - math.ceil(sq_covar_y)
            to_y_r    = round(cy) + math.ceil(sq_covar_y)
            clip_area_r = (to_x_r - from_x_r) * (to_y_r - from_y_r)        
            
            pm1_area = ((int(cx)+1) - (int(cx) -1)) * ((int(cy)+1) - (int(cy)-1))
            pm3_area = ((int(cx)+3) - (int(cx) -3)) * ((int(cy)+3) - (int(cy)-3))
            fr_to_area     = (to_y - from_y) * (to_x - from_x)
            fr_to_p1_area  = ((to_y + 1) - from_y) * ((to_x +1) - from_x)
            fr_to_pm1_area = ((to_y+1) - (from_y-1)) * ((to_x+1) - (from_x-1))
            full_area      = (int(pre[3])-int(pre[1])) * (int(pre[2]) - int(pre[0]))
            print('-'*150)   
            print('{:3.0f}   ({:6.2f},{:6.2f})  ({:6.2f},{:6.2f}) |'\
              ' {:6.2f}/{:6.2f}  {:7.2f}~{:7.2f}  {:7.2f} {} |'\
              ' {:6.2f}  {:6.2f}|'\
              ' {} {} {} {} {} | {} {} {} {} {}'.format(pre[7], 
                     pre[1], pre[0],  pre[3], pre[2], 
                     cx, cy,  width,  height, area, pre[13],
                     sq_covar_x, sq_covar_y, 
                     from_x, to_x, from_y, to_y , clip_area,
                     from_x_r, to_x_r, from_y_r, to_y_r, clip_area_r
                  ))    
            
            print()
            
            print(' {:3s}  Original MRCNN Prediction for this box:  [{:2.0f}] - {:15s}  Orig Score: {:6.4f}  Norm Score:  {:6.4f} '.
                  format('TP' if pre[6] == 1 else 'FP', pre[4],  class_names[cls], pre[5], pre[8]))
            print('-'*150)   
            
            print()
            cx_cy_sum     = ''.join([ '{:15.4f}'.format(np.sum(fcn_hm[int(cx),int(cy),i])                                 )  for i in class_list])
            cx_cy_pm1_sum = ''.join([ '{:15.4f}'.format(np.sum(fcn_hm[int(cx)-1:int(cx)+1,int(cy)-1:int(cy)+1,i])         )  for i in class_list])
            cx_cy_pm3_sum = ''.join([ '{:15.4f}'.format(np.sum(fcn_hm[int(cx)-3:int(cx)+3,int(cy)-3:int(cy)+3,i])         )  for i in class_list])
            fr_to_sum     = ''.join([ '{:15.4f}'.format(np.sum(fcn_hm[from_y:to_y, from_x:to_x,i])                        )  for i in class_list])
            fr_to_p1_sum  = ''.join([ '{:15.4f}'.format(np.sum(fcn_hm[from_y:to_y+1, from_x:to_x+1,i])                    )  for i in class_list])
            fr_to_pm1_sum = ''.join([ '{:15.4f}'.format(np.sum(fcn_hm[from_y-1:to_y+1, from_x-1:to_x+1,i])                )  for i in class_list])
            full_box_sum  = ''.join([ '{:15.4f}'.format(np.sum(fcn_hm[int(pre[1]):int(pre[3]), int(pre[0]):int(pre[2]),i]))  for i in class_list])


            cx_cy_score     = ''.join([ '{:15.4f}'.format(np.sum(fcn_hm[int(cx),int(cy),i])                                 )  for i in class_list])
            cx_cy_pm1_score = ''.join([ '{:15.4f}'.format(np.sum(fcn_hm[int(cx)-1:int(cx)+1,int(cy)-1:int(cy)+1,i]) / pm1_area  )  for i in class_list])
            cx_cy_pm3_score = ''.join([ '{:15.4f}'.format(np.sum(fcn_hm[int(cx)-3:int(cx)+3,int(cy)-3:int(cy)+3,i]) / pm3_area  )  for i in class_list])
            fr_to_score     = ''.join([ '{:15.4f}'.format(np.sum(fcn_hm[from_y:to_y, from_x:to_x,i]) / fr_to_area               )  for i in class_list])
            fr_to_p1_score  = ''.join([ '{:15.4f}'.format(np.sum(fcn_hm[from_y:to_y+1, from_x:to_x+1,i]) / fr_to_p1_area        )  for i in class_list])
            fr_to_pm1_score = ''.join([ '{:15.4f}'.format(np.sum(fcn_hm[from_y-1:to_y+1, from_x-1:to_x+1,i]) / fr_to_pm1_area   )  for i in class_list])
            full_box_score  = ''.join([ '{:15.4f}'.format(np.sum(fcn_hm[int(pre[1]):int(pre[3]), int(pre[0]):int(pre[2]),i])/full_area )  for i in class_list])
            
            
            print(' '*10,'-'*155)
            print(' '*10, '                       ',sub_title)
            print(' '*10,'-'*155)
            print(' '*10, '       cx_cy score ({:6.1f}): {}'.format( 1             , cx_cy_sum))
            print(' '*10, '  cx_cy +/-1 score ({:6.1f}): {}'.format( pm1_area      , cx_cy_pm1_sum))
            print(' '*10, '  cx_cy +/-3 score ({:6.1f}): {}'.format( pm3_area      , cx_cy_pm3_sum))
            print(' '*10, '     from:to score ({:6.1f}): {}'.format( fr_to_area    , fr_to_sum))
            print(' '*10, '  from:to +1 score ({:6.1f}): {}'.format( fr_to_p1_area , fr_to_p1_sum))
            print(' '*10, 'from:to +/-1 score ({:6.1f}): {}'.format( fr_to_pm1_area, fr_to_pm1_sum))
            print(' '*10, '   full bbox score ({:6.1f}): {}'.format( full_area     , full_box_sum))
            print()   
                      
            print(' '*10,'-'*155)
            print(' '*10, '                       ',sub_title)
            print(' '*10,'-'*155)
            print(' '*10, '       cx_cy score ({:6.1f}): {}'.format( 1             , cx_cy_score))
            print(' '*10, '  cx_cy +/-1 score ({:6.1f}): {}'.format( pm1_area      , cx_cy_pm1_score))
            print(' '*10, '  cx_cy +/-3 score ({:6.1f}): {}'.format( pm3_area      , cx_cy_pm3_score))
            print(' '*10, '     from:to score ({:6.1f}): {}'.format( fr_to_area    , fr_to_score))
            print(' '*10, '  from:to +1 score ({:6.1f}): {}'.format( fr_to_p1_area , fr_to_p1_score))
            print(' '*10, 'from:to +/-1 score ({:6.1f}): {}'.format( fr_to_pm1_area, fr_to_pm1_score))
            print(' '*10, '   full bbox score ({:6.1f}): {}'.format( full_area     , full_box_score))
            print()
    print('-'*175)
    return

#-----------------------------------------------------------------------------------------------------------    
#
#-----------------------------------------------------------------------------------------------------------        
def display_fcn_hm_scores_by_class(fcn_hm_scores_by_class, class_names, only = None):
    '''
    fcn_hm_scores:    results from fcn_detection [num_classes, num_bboxes, columns] 
    cn:   class names
    
    '''
    np_format = {}
    float_formatter = lambda x: "%10.4f" % x
    int_formatter   = lambda x: "%10d" % x
    np_format['float'] = float_formatter
    np_format['int']   = int_formatter
    np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =10000, formatter = np_format)
    
    if only is None:
        only = range(fcn_hm_scores_by_class.shape[0])
#     seq_start = r['pr_hm_scores'].shape[1]
    # print('  class ids  : ', r['class_ids'], type(r['class_ids']))
    print('  classes     : ', only)
    print('-'*175)
    print('                                          |           alt score 0           |              alt score 1              |                alt score 2            |')
    print('        class             mrcnn   normlzd |   gauss      bbox    nrm.scr*   |  ga.sum     mask    score      norm   |   ga.sum     mask     score   norm    |')
    print('seq  id     name          score   score   |   sum        area    gau.sum    |  in mask    sum                score  |   in mask    sum              score   |')
    print('-'*175)
    
    for cls in range(fcn_hm_scores_by_class.shape[0]):
        if cls in only:
            for pre in fcn_hm_scores_by_class[cls]:
        #     print(' {:4d}      {:12s}   {:.4f}   {:.4f}     {:.4f}  {:7.4f}  {:.4f}      {:.4f}   {}'.
                width = pre[3]-pre[1]
                height = pre[2]-pre[0]
                if width == height == 0:
                    continue
                cx = pre[1] + width/2
                cy = pre[0] + height/2
                area = (pre[3]-pre[1])* (pre[2]-pre[0])
                print('{:3.0f} {:3.0f} {:15s}   {:.4f}   {:.4f} |'\
                      ' {:9.4f}  {:9.4f}  {:9.4f} |'\
                      ' {:8.4f}  {:8.4f}  {:7.4f}   {:7.4f} |'\
                      ' {:8.4f}  {:8.4f}  {:7.4f}   {:7.4f} |'\
                      ' {:6.2f}  {:6.2f}  {:9.4f}'.          
                  format(pre[7], pre[4], class_names[cls], pre[5], pre[8], 
                     pre[9], pre[10], pre[11], 
                     pre[12], pre[13], pre[14], pre[15], 
                     pre[18], pre[19], pre[20], pre[23],
                     cx, cy , area))     #,  pre[4],pre[5],pre[6],roi))
            print('-'*175)
    return 
                                    
                                    
                                    
##-----------------------------------------------------------------------------------------------------------    
## DISPLAY MRCNN AND FCN SCORES 
##-----------------------------------------------------------------------------------------------------------    
def display_pr_fcn_scores(f_input, class_names, only = None, display = True, size = 12):
    '''
    f_input  :    results from detection process (run_mrcnn_detection)
    pr_scores:    pr_scores returned from detection or evaluation process results['pr_scores']
    cn:   class names
    
    '''
    np_format = {}
    np_format['float'] = lambda x: "%10.4f" % x
    np_format['int']   = lambda x: "%10d" % x
    np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =10000, formatter = np_format)

    pr_scores = f_input['pr_scores']
    fcn_scores = f_input['fcn_scores']
    
    if only is None:
        only = np.unique(pr_scores[:,4]).astype(np.int)

    seq_start = pr_scores.shape[1]
    
    print('\n')
    print('PR_SCORES from fcn/mrcnn_results:   (top line MRCNN, bottom line FCN) ' )
    print('-'*175)
    print('                              |   |                |     MRCNN / FCN score 0    |         MRCNN / FCN score 1        |        MRCNN / FCN score 2         | ')
    print('          class               |TP/| mrcnn  normlzd |  gaussian   bbox   nrm.scr*|  ga.sum    mask     score   norm   |  ga.sum    mask     score   norm   | ')
    print('    seq  id name              | FP| score   score  |    sum      area   gau.sum |  in mask   sum              score  |  in mask   sum              score  |  Y1  X1  Y2  X2   AREA  ')
    print('-'*175)

    for idx, (pre, fcn) in enumerate(zip(pr_scores, fcn_scores)):
        int_cls = int(pre[CLASS_COLUMN])
        if  int_cls in only:
            cx = pre[1] + (pre[3]-pre[1])/2
            cy = pre[0] + (pre[2]-pre[0])/2
            area = (pre[3]-pre[1]) * (pre[2]-pre[0])
            print('{:3.0f} {:3.0f} {:2d} {:18s} |{:2.0f} | {:.4f}  {:.4f} |'\
                  '{:9.4f}  {:6.1f} {:9.4f} |'\
                  '{:9.4f}  {:6.1f} {:9.4f}  {:6.4f} |'\
                  '{:9.4f}  {:6.1f} {:9.4f}  {:6.4f} |'\
                  ' {:3.0f} {:3.0f} {:3.0f} {:3.0f} {:7.2f}'.          
              format(idx,    
                     pre[SEQUENCE_COLUMN],
                     int_cls, class_names[int_cls], pre[DT_TYPE_COLUMN], pre[ORIG_SCORE_COLUMN], pre[NORM_SCORE_COLUMN], 
                     pre[SCORE_0_SUM_COLUMN], pre[SCORE_0_AREA_COLUMN], pre[SCORE_0_COLUMN], 
                     pre[SCORE_1_SUM_COLUMN], pre[SCORE_1_AREA_COLUMN], pre[SCORE_1_COLUMN], pre[SCORE_1_NORM_COLUMN], 
                     pre[SCORE_2_SUM_COLUMN], pre[SCORE_2_AREA_COLUMN], pre[SCORE_2_COLUMN], pre[SCORE_2_NORM_COLUMN], 
                     pre[0],pre[1], pre[2], pre[3], area ))     #,  pre[4],pre[5],pre[6],roi))

        int_cls = int(fcn[CLASS_COLUMN])
        if int_cls in only:
            cx = fcn[1] + (fcn[3]-fcn[1])/2
            cy = fcn[0] + (fcn[2]-fcn[0])/2
            area = (fcn[3]-fcn[1]) * (fcn[2]-fcn[0])
            print('{:3.0f} {:3.0f} {:2d} {:18s} |{:2.0f} | {:.4f}  {:.4f} |'\
                  '{:9.4f}  {:6.1f} {:9.4f} |'\
                  '{:9.4f}  {:6.1f} {:9.4f}  {:6.4f} |'\
                  '{:9.4f}  {:6.1f} {:9.4f}  {:6.4f} |'\
                  ' {:3.0f} {:3.0f} {:3.0f} {:3.0f} {:7.2f}'.          
                format(idx,
                     fcn[SEQUENCE_COLUMN], 
                        int_cls, class_names[int_cls], fcn[DT_TYPE_COLUMN], fcn[ORIG_SCORE_COLUMN], fcn[NORM_SCORE_COLUMN], 
                     fcn[SCORE_0_SUM_COLUMN], fcn[SCORE_0_AREA_COLUMN], fcn[SCORE_0_COLUMN], 
                     fcn[SCORE_1_SUM_COLUMN], fcn[SCORE_1_AREA_COLUMN], fcn[SCORE_1_COLUMN], fcn[SCORE_1_NORM_COLUMN], 
                     fcn[SCORE_2_SUM_COLUMN], fcn[SCORE_2_AREA_COLUMN], fcn[SCORE_2_COLUMN], fcn[SCORE_2_NORM_COLUMN],                   
                     fcn[0],fcn[1], fcn[2], fcn[3], area ))     #,  fcn[4],fcn[5],fcn[6],roi))                     
            print()
        
    # if isinstance (f_input, dict)  and display:
        # img_id = str(f_input['image_meta'][0])        
        # visualize.display_instances(f_input['image'] , pr_scores[:,:CLASS_COLUMN], pr_scores[:,CLASS_COLUMN].astype(np.int32), 
                                    # class_names, pr_scores[:,ORIG_SCORE_COLUMN], only_classes= only, size =size,
                                    # title = 'MRCNN predictions for image id'+img_id)
                                    
                                    
#------------------------------------------------------------------------------------------------------------------                                    
# detections as returned from the model's `detect()` functon
#------------------------------------------------------------------------------------------------------------------
def display_mrcnn_style1(f, class_names,lmt =18):
    print('       classes :', f['class_ids'][:lmt])
    names  = " ".join([ '{:>10s}'.format(class_names[i][-10:]) for i in f['class_ids'][:lmt]])

    print('               : ', names)
    print('                ', f['detection_ind'][:lmt])
    print('   orig scores :', f['scores'][:lmt])
    print('   orig scores :', f['pr_scores'][:lmt, ORIG_SCORE_COLUMN])
    print('   norm scores :', f['pr_scores'][:lmt,8])
    print('            X1 :', f['pr_scores'][:lmt,1])
    print('            Y1 :', f['pr_scores'][:lmt,0])
    print('            X2 :', f['pr_scores'][:lmt,3])
    print('            Y2 :', f['pr_scores'][:lmt,2])

    # print('  pr_scores[5] :', f['pr_scores'][:,5])

    print('-'*185)
    print(' SCR 0    [11] :', f['pr_scores'][:lmt,11])
    # print(' fcn_scores[11] :', f['fcn_scores'][:lmt,11])
    print()
    print(' SCR 1    [14] :', f['pr_scores'][:lmt,14])
    # print('fcn_scores[14] :', f['fcn_scores'][:lmt,14])
    print()
    print(' SCR 2    [20] :', f['pr_scores'][:lmt,20])
    # print('fcn_scores[20] :', f['fcn_scores'][:lmt,20])
    
def display_pr_fcn_style1(f, class_names,lmt =18):
    names  = " ".join([ '{:>10s}'.format(class_names[i][-10:]) for i in f['pr_scores'][:lmt, 4].astype(np.int32)])

    print('         bbox seq id :', f['pr_scores'][:lmt,7].astype(int))
    print('  f[class_ids] class :', f['class_ids'][:lmt])
    print('   f[pr_score] class :', f['pr_scores'][:lmt,4].astype(np.int))
    print('  f[fcn_score] class :', f['fcn_scores'][:lmt,4].astype(np.int))
    print('                       ', names)
    print('           TP/FP Ind :', f['pr_scores'][:lmt,6].astype(np.int))
    print('      pr  orig score :', f['pr_scores'][:lmt,5])
    print('      fcn orig score :', f['fcn_scores'][:lmt,5])
    print('      cls norm score :', f['pr_scores'][:lmt,8])
    print('           bbox area :', f['pr_scores'][:lmt,10])
    print('           clip area :', f['pr_scores'][:lmt,13])
    print()
    print('-'*185)
    print('  pr norm_score [8]  :', f['pr_scores'][:lmt,8])
    print('  fcn norm_score[8]  :', f['fcn_scores'][:lmt,8])
    print()
    print('     pr_score_0 [11] :', f['pr_scores'][:lmt,11])
    print('    fcn_score_0 [11] :', f['fcn_scores'][:lmt,11])
    print()
    print('     pr_score_1 [14] :', f['pr_scores'][:lmt,14])
    print('    fcn_score_1 [14] :', f['fcn_scores'][:lmt,14])
    print()
    print('     pr_score_2 [20] :', f['pr_scores'][:lmt,20])
    print('    fcn_score_2 [20] :', f['fcn_scores'][:lmt,20])
    print()
    print('norm pr_score_1 [17] :', f['pr_scores'][:lmt,17])
    print('norm fcn_score  [17] :', f['fcn_scores'][:lmt,17])
    print()
    print('norm pr_score_2 [23] :', f['pr_scores'][:lmt,23])
    print('norm fcn_score  [23] :', f['fcn_scores'][:lmt,23])


def display_mrcnn_style2(f, class_names,lmt =18):
    for i, [molded_bbox, cls, scr, pr_scr] in enumerate(zip(f['molded_rois'].astype(np.int), f['class_ids'],  f['scores'], f['pr_scores'])):
        print('{} ({:2d})-  {:.<18s}  {:5.4f} {}  '.format(i, cls, class_names[cls], scr, pr_scr[[4,5,6,7,8]] ))
        print('{:>86s} {}'.format('  Bbox Coordinates - molded_rois: ', molded_bbox))
        print('{:>86s} {}'.format('  Bbox Coordinates - pr_scores  : ', pr_scr[:4]))
        print()    
        print('{:>86s} {}'.format('   Orig/Norm score : ',  pr_scr[[5,8]]))
        print('{:>86s} {}'.format('  mrcnn old scores : ',  pr_scr[[9,10,11]]))
        print()    
        print('{:>86s} {}'.format(' mrcnn alt scores1 : ', pr_scr[[12,13,14,15,16,17]]))
        print()    
        print('{:>86s} {}'.format(' mrcnn alt scores2 : ', pr_scr[[18,19,20,21,22,23]]))
        print()    
        

def display_pr_fcn_style2(f, class_names,lmt =18):
    for i, [molded_bbox, cls, scr, pr_scr, fcn_scr] in enumerate(zip(f['molded_rois'].astype(np.int), f['class_ids'],  f['scores'], f['pr_scores'], f['fcn_scores'])):
        
        print('{} ({:2d})-  {:.<18s}  {:5.4f} {}  '.format(i, cls, class_names[cls], scr, fcn_scr[[4,5,6,7,8]]))
        print('{:>40s} {}'.format('  Bbox Coordinates - molded_rois: ', molded_bbox))
        print('{:>40s} {}'.format('  Bbox Coordinates - pr_scores  : ', pr_scr[:4]))
        print('{:>40s} {}'.format('  Bbox Coordinates - fcn_scores : ', fcn_scr[:4]))
        print('{:>86s} {}'.format('      Orig / Norm score:  ',  pr_scr[[5,8]]))
        print('{:>86s} {}'.format(' mrcnn old style scores:  ',  pr_scr[[9,10,11]]))
        print('{:>86s} {}'.format('   fcn old style scores:  ', fcn_scr[[9,10,11]]))
        print()
        print('{:>86s} {}'.format('      mrcnn alt scores1:  ', pr_scr[[12,13,14,15,16,17]]))
        print('{:>86s} {}'.format('        fcn alt scores1:  ', fcn_scr[[12,13,14,15,16,17]]))
        print()
        print('{:>86s} {}'.format('          mrcnn_scores2:  ', pr_scr[[18,19,20,21,22,23]]))
        print('{:>86s} {}'.format('            fcn_scores2:  ', fcn_scr[[18,19,20,21,22,23]]))
        print()    
        
        
def display_pr_fcn_style3(f, class_names,lmt =18):
    print(f['detections'].shape)
    print('  alt_score  0: (gauss. sum over large bbox / bbox area/ gauss_sum * normlzd_score))')
    print('  alt_scores 1: (gauss. sum over small mask / mask area/ gauss_sum / mask_area):  ')
    sort_by_class_order = np.argsort(f['class_ids'])


    # for i in range(len( f['class_ids'])):
    for i in sort_by_class_order:
    #     print(i , f['rois'][i].astype(np.float), f['scores'][i], f['class_ids'][i], class_names[f['class_ids'][i]])
    #     print(i , f['detections'][i], f['class_ids'][i], class_names[f['class_ids'][i]])
        det_type = '       --> ADDED FP ' if f['pr_scores'][i,6] == -1 else '      Original detection'
        print(i , f['rois'][i])
        print(i , f['pr_scores'][i,:9], f['pr_scores'][i,4], class_names[ f['pr_scores'][i,4].astype(np.int)], det_type) 
        print()
        print(i , 'pr: alt_scores 0 [9,10,11]:  '.rjust(90), f['pr_scores'][i,9:12])    
        print(i , '  alt_scores 1 [12 - 17]:  '.rjust(90), f['pr_scores'][i,12:18])    
        print(i , '  alt_scores 2 [18 - 23]:  '.rjust(90), f['pr_scores'][i,18:24])    
        print(i)
    #     print(i , f['fcn_scores'][i,:8], f['fcn_scores'][i,4], class_names[ f['fcn_scores'][i,4].astype(np.int)])    
        print(i , 'fcn: alt_score 0 [9,10,11]:  '.rjust(90), f['fcn_scores'][i,9:12])    
        print(i ,  '  alt_scores 1 [12 - 17]:  '.rjust(90), f['fcn_scores'][i,12:18])    
        print(i ,  '  alt_scores 2 [18 - 23]:  '.rjust(90), f['fcn_scores'][i,18:24])    
        print()
        print(i , 'alt score 0 [11]:  '.rjust(90), ' from mrcnn:{:10.4f}  from FCN: {:10.4f} '.format(f['pr_scores'][i,11],f['fcn_scores'][i,11]))
        print(i , 'alt score 1 [14]:  '.rjust(90), ' from mrcnn:{:10.4f}  from FCN: {:10.4f} '.format(f['pr_scores'][i,14],f['fcn_scores'][i,14]))
        print(i , 'alt score 2 [20]:  '.rjust(90), ' from mrcnn:{:10.4f}  from FCN: {:10.4f} '.format(f['pr_scores'][i,20],f['fcn_scores'][i,20]))
        print()        
        
        
        

##-----------------------------------------------------------------------------------------------------------    
##
##-----------------------------------------------------------------------------------------------------------        
    

##-----------------------------------------------------------------------------------------------------------    
## DISPLAY INTERSECTION OVER UNION
##-----------------------------------------------------------------------------------------------------------    
def display_ious(f,class_names):
#    for i,j in zip(f['gt_class_ids'], f['gt_bboxes']):
#        print(i, ' ', j )
#     for i,j in zip(f['pr_scores'][:,4], f['pr_scores'][:,:4]):
#         print(i, ' ', j )    
    np_format = {}
    np_format['float'] = lambda x: "%10.4f" % x    
    np_format['int']   = lambda x: "%10d" % x      
    np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =10000, formatter = np_format)
    
        
    ovl = utils.compute_overlaps(f['gt_bboxes'], f['gt_bboxes'])
    hdr = ''.join(['  {:3d} /{:>3d} '.format(i,int(j)) for i,j in enumerate(f['gt_class_ids'])])
    print()
    print(' Overlap Matrix for Ground Truth Bounding Boxes ')
    print('------------------------------------------------')
    print(' '*25,  hdr)
    print(' Box/Cls :               ', '-'*120)
    for k,(i,j) in enumerate(zip(f['gt_class_ids'], ovl)):
        print('{:2d}/{:2d} - {:15s}  {}' .format(k, i, class_names[i],j))

        
    ovl = utils.compute_overlaps(f['pr_scores'][:,:4], f['pr_scores'][:,:4]) 
    hdr = ''.join(['  {:3d} /{:>3d} '.format(i,int(j)) for i,j in enumerate(f['pr_scores'][:,4])])
    print()
    print(' Overlap Matrix for MRCNN Detection Bounding Boxes')
    print('---------------------------------------------------')
    print(' '*25,  hdr)
    print(' Box/Cls :               ', '-'*120)
    for k,(i,j) in enumerate(zip(f['pr_scores'][:,4], ovl)):
        print('{:2d}/{:2d} - {:15s}  {}' .format(k, int(i), class_names[int(i)],j))
    
    ovl = utils.compute_overlaps(f['pr_scores'][:,:4], f['gt_bboxes']) 
    hdr = ''.join(['  {:3d} /{:>3d} '.format(i,int(j)) for i,j in enumerate(f['gt_class_ids'])])
    print()
    print(' Overlap Matrix for Detection vs GT Bounding Boxes')
    print('---------------------------------------------------')
    print('                GT Boxes:',  hdr)
    print(' Prediction Boxes:       ', '-'*120)
    for k,(i,j) in enumerate(zip(f['pr_scores'][:,4], ovl)):
        print('{:2d}/{:2d} - {:15s}  {}' .format(k, int(i), class_names[int(i)],j))
    
##-----------------------------------------------------------------------------------------------------------    
##
##-----------------------------------------------------------------------------------------------------------    
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    print(x.shape)
    out = x - x.mean()
    out /= (x.std() + 1e-5)
    out *= 0.1

    # clip to [0, 1]
    out += 0.5
    out = np.clip(out, 0, 1)

    # convert to RGB array
    out *= 255
    out = np.clip(out, 0, 255).astype('uint8')
    return out

def standardize_fcn_hm(x):
    # standardize tensor: center on 0., ensure std is 0.1
    print(x.shape)
    out = x - x.mean(axis=(0,1), keepdims = True)
    out /= (x.std(axis=(0,1), keepdims = True) + 1e-7)
    return out

def normalize_fcn_hm(x):
    # normalize FCN_HM across all classes 
    # print(x.shape)
    out = x - x.min(axis=(0,1), keepdims = True)
    out /= (x.max(axis=(0,1), keepdims = True) - x.min(axis=(0,1), keepdims = True))
    return out

normalize_by_class = normalize_fcn_hm
    
## used in experiment 6
def normalize_fcn_score(x):
    # normalize FCN_HM across all classes 
    print(x.min(axis=(1,2), keepdims = True).shape)
    out = x - x.min(axis=(1,2), keepdims = True)
    out /= (x.max(axis=(1,2), keepdims = True) - x.min(axis=(1,2), keepdims = True))
    return out

def standardize_fcn_score(x):
    # standardize tensor: center on 0., ensure std is 0.1
    print(x.shape)
    out = x - x.mean(axis=(1,2), keepdims = True)
    out /= (x.std(axis=(1,2), keepdims = True) + 1e-7)
    return out
    
normalize_by_class = normalize_fcn_hm
    
def normalize_all(x):
    # normalize FCN_HM across all classes 
    # print(x.shape)
    out = x - x.min(keepdims = True)
    out /= (x.max(keepdims = True) - x.min(keepdims = True))
    return out


    
##-----------------------------------------------------------------------------------------------------------    
## DISPLAY FCN HEATMAP
##-----------------------------------------------------------------------------------------------------------    
def display_fcn_hm(fcn_hm, class_names,  cmap = 'jet', columns = 8, size = 4 , classes = None):
    if fcn_hm.ndim == 4:
        fcn_hm = fcn_hm[0]
    if classes is not None:
        n_features = len(classes)
    else:
        n_features = fcn_hm.shape[-1]
        classes    = np.arange(n_features)
    # print('classes:',classes, 'n_features:', n_features)
    rows = math.ceil(n_features / columns )

#     print()
#     print('Layer:', LAYER, ' - ',layer_names[LAYER], '   Shape: ', activations[LAYER][0,:,:,:].shape, ' # features: ', n_features, rows, columns)
    fig = plt.figure(figsize=(size * columns, size * rows))
    vmin = fcn_hm.min()
    vmax = fcn_hm.max()
    for idx, cls  in enumerate(classes) :   ##  range(n_features): 
        row = idx // columns
        col = idx  % columns
        subplot = (row * columns) + col +1    
        # print('rows:', rows, 'cols:', columns, 'subplot: ', subplot)
        ax= fig.add_subplot(rows, columns, subplot)
        ax.set_title(class_names[cls], fontsize=15)
        # ax.tick_params(direction='out', length=6, width=1, colors='r', labelsize = 10)
        
        img = ax.imshow(fcn_hm[:,:,cls], origin = 'upper', cmap = cmap , interpolation='none', vmin = vmin, vmax = vmax)
        ax.tick_params(axis='both', bottom=True, left=True , top=False, right=False)
        cb = fig.colorbar(img, shrink=0.6, aspect=30, fraction=0.02)
        cb.ax.tick_params(labelsize=10)
    title = 'FCN Input - shape: {}'.format(str(fcn_hm.shape))        

    fig.suptitle(title, fontsize = 12, ha ='center' )        
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])   ## [left, bottom, right, top]
    # plt.subplots_adjust(top=0.95, bottom=0.02, left=0.02, right=0.98, hspace=0.25, wspace=0.15)
    plt.show()
    
    return

display_fcn_output  = display_fcn_hm
display_fcn_input   = display_fcn_hm

##-----------------------------------------------------------------------------------------------------------    
##
##-----------------------------------------------------------------------------------------------------------    
def display_fcn_input_2(fcn_hm, cmap = 'gray', columns = 8, size  = 4):
    if fcn_hm.ndim == 4:
        fcn_hm = fcn_hm[0]
    n_features = fcn_hm.shape[-1]   
    rows = math.ceil(n_features / columns )
    
    fig , ax = plt.subplots(rows, columns, figsize=(size*columns, size*rows))
    print(n_features, rows )
    # print('Layer:', LAYER, ' - ',layer_names[LAYER], '   Shape: ', activations[LAYER][0,:,:,:].shape, ' # features: ', n_features, rows, columns)
#     fig = plt.figure(figsize=(8 *columns, 12* rows))
    for idx in range(n_features): 
        row = idx // columns
        col = idx  % columns
        subplot = (row * columns) + col +1    
        print(row,col, subplot)
#         ax= fig.add_subplot(rows, columns, subplot)
        ax[row,col].set_title(class_names[idx], fontsize=20)
        ax[row,col].tick_params(axis='both', labelsize = 5)
        surf = ax[row,col].matshow(fcn_hm[:,:,idx], cmap = cmap , interpolation='none')
        cbar = ax[row,col].colorbar(cax = surf, shrink=0.6, aspect=30, fraction=0.05)
        cbar.ax.tick_params(labelsize=14) 
        
    ttl = title + ' {}'.format(str(fcn_hm.shape))        
    fig.suptitle(title, fontsize = 18, ha ='center' )      
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])   ## [left, bottom, right, top]
    plt.show()

    
##-----------------------------------------------------------------------------------------------------------    
##
##-----------------------------------------------------------------------------------------------------------    
def display_fcn_agg_heatmaps(fcn_hm, gt_cls_counts,  dt_cls_counts, class_names = None, classes = None, cmap = 'jet', 
                        columns= 4, title = None, norm = False):
    '''
    This routine is used in the display_aggreggate_heatmaps and exp2 notebooks 
    '''
    if classes is None :
        n_features = range(fcn_hm.shape[-1])
    else:
        n_features = classes

    rows = math.ceil(len(n_features) / columns )
    fig = plt.figure(figsize=(8 *columns, 8* rows))
    vmin = fcn_hm.min()
    vmax = fcn_hm.max()
    for idx, feat in enumerate(n_features): 
        row = idx // columns
        col = idx  % columns
        subplot = (row * columns) + col +1    
        ax= fig.add_subplot(rows, columns, subplot)
        subttl = '{} - {} - GT:{:4d}  Det:{:4d}'.format(feat, class_names[feat], gt_cls_counts[feat], dt_cls_counts[feat])
        ax.set_title(subttl, fontsize=20)
        ax.tick_params(axis='both', labelsize = 13, length = 6, width = 3)      
        surf = ax.matshow(fcn_hm[:,:,feat], cmap = cmap , interpolation='none', vmin=vmin, vmax = vmax)
        cbar = fig.colorbar(surf, shrink=0.6, aspect=30, fraction=0.05)
        cbar.ax.tick_params(labelsize=16) 
    
    if title is not None :
        ttl = title + ' {}'.format(str(fcn_hm.shape))           
        fig.suptitle(ttl, fontsize = 24, ha ='center' )        
    plt.subplots_adjust(top=0.9, bottom=0.02, left=0.02, right=0.98, hspace=0.25, wspace=0.15)
    # plt.tight_layout()
    plt.show()
    return fig
    
##-----------------------------------------------------------------------------------------------------------    
## Display score contours for experiment 6
##-----------------------------------------------------------------------------------------------------------    
def display_score_contours(scores, gt_cls_counts,  dt_cls_counts, class_names = None, classes = None, cmap = 'jet', 
                           columns= None, title = None, norm = False):
    '''
    This routine is used in the display_aggreggate_heatmaps and exp2 notebooks 
    '''
    if classes is None :
        classes = range(scores.shape[0])
    
    n_features = len(classes)
    
    if columns is None:
        columns = n_features
        
    rows = math.ceil(n_features / columns )
    # print( ' Num features: ', n_features , 'Rows / Columns: ', rows, columns)
    fig = plt.figure(figsize=(8*columns, 8*rows))
    vmin = scores.min()
    vmax = scores.max()
    norm = None
    for idx, feat in enumerate(classes): 
        row = idx // columns
        col = idx  % columns
        subplot = (row * columns) + col +1    
        ax= fig.add_subplot(rows, columns, subplot)
        subttl = '{} - {} - GT:{:4d}  Det:{:4d}'.format(feat, class_names[feat], gt_cls_counts[feat], dt_cls_counts[feat])
        ax.set_title(subttl, fontsize=20)
        ax.tick_params(axis='both', labelsize = 13, length = 6, width = 3)      
        ax.invert_yaxis()
        surf = ax.contourf(scores[feat],  cmap=cm.jet, norm= norm) 
        cbar = fig.colorbar(surf, shrink=0.6, aspect=30, fraction=0.05)
        cbar.ax.tick_params(labelsize=16) 
    
    if title is not None :
        ttl = title             
        fig.suptitle(ttl, fontsize = 24, ha ='center', va = 'top' )        
    
    plt.subplots_adjust(top=0.9, bottom=0.02, left=0.02, right=0.98, hspace=0.25, wspace=0.15)
    # plt.tight_layout()
    plt.show()
    return fig


def display_score_contours_compare(pr_scores, fcn_scores, class_names = None, classes = None, cmap = 'jet', 
                           columns= None, title = None, norm = False):
    '''
    This routine is used in experiment 6 to show scores from mrcnn and fcn side by side 
    '''
    if classes is None :
        classes = range(scores.shape[0])
    
    n_features = len(classes)
    
    rows = n_features
    columns = 3 
    
    # rows = math.ceil(n_features / columns )
    print( ' Num features: ', n_features , 'Rows / Columns: ', rows, columns)
    fig = plt.figure(figsize=(4*columns, 4*rows))
    # vmin = scores.min()
    # vmax = scores.max()
    norm = None
    title_fontsz  = 10
    subttl_fontsz = 10
    cbar_fontsz   = 9 
    label_fontsz  = 9 
    for idx, feat in enumerate(classes): 
        row = idx 
        
        subplot = (row * columns) + 1    
        ax= fig.add_subplot(rows, columns, subplot)
        subttl = '{} - {} -  MRCNN Score'.format(feat, class_names[feat])
        ax.set_title(subttl, fontsize=subttl_fontsz)
        ax.tick_params(axis='both', labelsize = label_fontsz, length = 6, width = 3)      
        ax.invert_yaxis()
        surf = ax.contourf(pr_scores[0,feat],  cmap=cm.jet, norm= norm) 
        cbar = fig.colorbar(surf, shrink=0.6, aspect=30, fraction=0.05)
        cbar.ax.tick_params(labelsize=cbar_fontsz) 

        subplot += 1    
        ax= fig.add_subplot(rows, columns, subplot)
        subttl = 'FCN Score 1'
        ax.set_title(subttl, fontsize=subttl_fontsz)
        ax.tick_params(axis='both', labelsize = label_fontsz, length = 6, width = 3)      
        ax.invert_yaxis()
        surf = ax.contourf(fcn_scores[0,feat],  cmap=cm.jet, norm= norm) 
        cbar = fig.colorbar(surf, shrink=0.6, aspect=30, fraction=0.05)
        cbar.ax.tick_params(labelsize=cbar_fontsz) 
    
        subplot += 1 
        ax= fig.add_subplot(rows, columns, subplot)
        subttl = 'FCN Score 2'
        ax.set_title(subttl, fontsize=subttl_fontsz)
        ax.tick_params(axis='both', labelsize = label_fontsz, length = 6, width = 3)      
        ax.invert_yaxis()
        surf = ax.contourf(fcn_scores[1,feat],  cmap=cm.jet, norm= norm) 
        cbar = fig.colorbar(surf, shrink=0.6, aspect=30, fraction=0.05)
        cbar.ax.tick_params(labelsize=cbar_fontsz) 
    
    if title is not None :
        ttl = title           
        fig.suptitle(ttl, fontsize = title_fontsz, ha ='center', va = 'top' )        
    
    plt.subplots_adjust(top=0.9, bottom=0.02, left=0.02, right=0.98, hspace=0.25, wspace=0.15)
    # plt.tight_layout()
    plt.show()
    return fig

##-----------------------------------------------------------------------------------------------------------    
##
##-----------------------------------------------------------------------------------------------------------    
def display_pairwise_heatmap(input_matrix, CLASS_IDS, CLASS_NAMES,title = '' , colormap = cm.coolwarm,
                             start_from = 1, columns = 100):
    # show_labels = labels[1:10]
#     num_classes = len(show_labels)
    #     print(' num classes: ', num_classes, 'matrix shape: ',show_matrix.shape)
    columns       = min(input_matrix.shape[1], columns)
    show_matrix   = input_matrix[CLASS_IDS, start_from:columns]
    n_classes     = show_matrix.shape[0] 
    n_instances   = show_matrix.shape[1]
    show_labels_x = [str(i) for i in range(start_from, n_instances+start_from)] 
    show_labels_y = ['{:2d}-{:s}'.format(i,CLASS_NAMES[i]) for i in CLASS_IDS] 

    fig, ax = plt.subplots(1,1,figsize=((n_classes)*1, (n_instances)*1))
    im = ax.imshow(show_matrix, cmap=colormap)
    # cmap=cm.bone, #  cmap=cm.Dark2 # cmap = cm.coolwarm   # cmap=cm.YlOrRd
    # We want to show all ticks...
    ax.set_xticks(np.arange(n_instances))
    ax.set_yticks(np.arange(n_classes))
    
    # ... and label them with the respective list entries
    ax.set_xticklabels(show_labels_x, size = 9)
    ax.set_yticklabels(show_labels_y, size = 9)
    ax.set_xlabel('Number of Instances ', size = 16)
    ax.set_ylabel('Class', size = 16 )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for cls in range(n_classes):
        for j in range(n_instances):
                text = ax.text(j, cls, show_matrix[cls, j], size=12, 
                           ha="center", va="center", color="w")
 
    ax.set_title(title, size = 14)
    # fig.tight_layout()
    plt.show()
    return  fig    
    

def display_pairwise_heatmap_rotated(input_matrix , CLASS_IDS, CLASS_NAMES,title = '' ,  colormap = cm.coolwarm, 
                             start_from = 1, columns = 100):
                             
    columns       = min(input_matrix.shape[1], columns)
    show_matrix = input_matrix[CLASS_IDS, start_from: columns]

    print(  ' -- matrix shape: ',show_matrix.shape)
    n_instances = show_matrix.shape[1] 
    n_classes   = show_matrix.shape[0]

    show_labels_y = [str(i) for i in range(start_from, n_instances+start_from)] 
    show_labels_x = ['{:2d}-{:s}'.format(i,CLASS_NAMES[i]) for i in CLASS_IDS] 
    sq_size = 1.0
    ## figsize = (width, height)
    # fig, ax = plt.subplots(1,1,figsize=((n_instances)* sq_size, (n_classes) * sq_size))
    fig, ax = plt.subplots(1,1,figsize=((n_classes)* sq_size, (n_instances) * sq_size))

    im = ax.imshow(show_matrix.T, cmap=colormap)
    
    # cmap=cm.bone, #  cmap=cm.Dark2 # cmap = cm.coolwarm   # cmap=cm.YlOrRd
    # We want to show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_instances))
    ax.set_xticklabels(show_labels_x, size = 9)
    ax.set_yticklabels(show_labels_y, size = 9)
    ax.set_xlabel('Class', size = 11 )
    ax.set_ylabel('Number of Instances ', size = 11)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for cls in range(n_classes):
        for i in range(n_instances):
            text = ax.text(cls, i, show_matrix[ cls, i], size=12, ha="center", va="center", color="w")
 
    ax.set_title(title, size = 14)
    # fig.tight_layout()

    ## Display total number of images per class 
    
    summ = show_matrix.sum(axis = 1, keepdims=True).T
    print('summshape : ', summ.shape)
    fig2, ax2 = plt.subplots(1,1,figsize=((n_classes)* sq_size, (1) * sq_size))
    im = ax2.imshow(summ, cmap=colormap)
    
    # cmap=cm.bone, #  cmap=cm.Dark2 # cmap = cm.coolwarm   # cmap=cm.YlOrRd
    # We want to show all ticks and label them with the respective list entries
    ax2.set_xticks(np.arange(n_classes))
    ax2.set_yticks(np.arange(1))
    ax2.set_xticklabels(show_labels_x, size = 9)
    ax2.set_yticklabels([' '], size = 9)
    ax2.set_xlabel('Class', size = 11 )
    ax2.set_ylabel('Images ', size = 11)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    plt.setp(ax2.get_yticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    ax2.set_title(title, size = 14)

    # Loop over data dimensions and create text annotations.
    for cls in range(n_classes):
        text = ax2.text(cls, 0, summ[0,cls], size=12, ha="center", va="center", color="w")
        
    plt.show()

    return  fig, fig2
    
    
##-----------------------------------------------------------------------------------------------------------    
##
##-----------------------------------------------------------------------------------------------------------    
def display_activations(activations, layer, layer_names, class_names, columns = 8, cmap = 'jet', normalize = True):
    LAYER = layer
    n_features = activations[LAYER].shape[-1]
    rows = math.ceil(n_features / columns )

    print()
    print('Layer:', LAYER, ' - ',layer_names[LAYER], '   Shape: ', activations[LAYER][0,:,:,:].shape, ' # features: ', n_features, rows, columns)
    fig = plt.figure(figsize=(8 *columns, 8 * rows))
    for idx in range(n_features): 
        row = idx // columns
        col = idx  % columns
        subplot = (row * columns) + col +1    
        ax= fig.add_subplot(rows, columns, subplot)
        ttl = 'Filter: {:4d}'.format(idx)
        ax.set_title(ttl, fontsize=20)
        
        channel_image = activations[LAYER][0,:,:, idx]
        if normalize:
            c_mean = channel_image.mean()
            c_std  = channel_image.std()

            ## Normalize to mean 0 , std_dev = 1
            channel_image -= c_mean
            channel_image /= c_std
#             channel_image *= 64
#             channel_image -= 128
        
        surf = ax.matshow(channel_image, cmap= cmap, interpolation='none')
        fig.colorbar(surf, shrink=0.6, aspect=30, fraction=0.05)
    title = 'Layer: {:3d} - {:25s} - shape: {}'.format(LAYER,layer_names[LAYER], str(activations[LAYER][0].shape))        
    fig.suptitle(title, fontsize = 24, ha ='center' )        
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])

    
##-----------------------------------------------------------------------------------------------------------    
##
##-----------------------------------------------------------------------------------------------------------        
def display_final_activations_2(activations, layers, layer_names, columns = 8, cmap = 'jet'):
    n_layers   = len(layers)
    n_features = activations[layers[0]].shape[-1]
    rows = math.ceil(n_features / columns )

    fig,ax = plt.subplots(n_layers,n_features, figsize=( n_layers * 10 , n_features * 10))  ### , gridspec_kw={"hspace":0.1, "wspace":0.1})
    
    fig_num = 1
    for row, layer in enumerate(layers):
        LAYER = layer
#         n_features = activations[LAYER].shape[-1]
#         rows = math.ceil(n_features / columns )

    #     print()
        print('Layer:', LAYER, ' - ',layer_names[LAYER], '   Shape: ', activations[LAYER][0,:,:,:].shape, ' # features: ', n_features, rows, columns)
    #     fig = plt.figure(fig_num, figsize=(8 *columns, 7* rows))
        for idx in range(n_features): 
#             row = idx // columns
            col = idx  % columns
            subplot = (row * columns) + col +1    
#             print('row, idx, col, subplot:', row, idx,col,subplot)
#             plt.subplot(rows, columns, subplot)
#             ax = plt.gca()
            ax[row, idx].set_title(class_names[idx], fontsize=18)
            surf = ax[row,idx].matshow(activations[LAYER][0,:,:,idx], cmap=cmap, interpolation='none', vmin=-1.0, vmax=1.0)
            plt.colorbar(surf,ax = ax[row,idx], shrink=0.6, aspect=30, fraction=0.05)
        title = 'Layer: {:3d} - {:25s} - shape: {}'.format(LAYER,layer_names[LAYER], str(activations[LAYER][0].shape))        
#         fig.suptitle(title, fontsize = 24, ha ='center' )            
#         fig.savfig
#         fig_num +=1
    plt.subplots_adjust(top=0.97, bottom=0.02, left=0.02, right=0.98, hspace=0.1, wspace=0.10)      
#     plt.subplots_adjust(top=0.97)      
    plt.show()
    


##-----------------------------------------------------------------------------------------------------------    
##
##-----------------------------------------------------------------------------------------------------------    
def display_final_activations(activations, layers, layer_names, class_names, columns = 8, cmap = 'jet'):
    n_layers   = len(layers)
    fig_num = 1
    
    for LAYER in layers:
        n_features = activations[LAYER].shape[-1]
        rows = math.ceil(n_features / columns )
        print('Layer:', LAYER, ' - ',layer_names[LAYER], '   Shape: ', activations[LAYER][0,:,:,:].shape, ' # features: ', n_features, columns, rows)
        fig = plt.figure(fig_num, figsize=(8 *columns, 7* rows))

        for idx in range(n_features): 
            row = idx // columns
            col = idx  % columns
            subplot = (row * columns) + col +1    
            plt.subplot(rows, columns, subplot)
            ax = plt.gca()
            ax.set_title(class_names[idx], fontsize=18)
            surf = ax.imshow(activations[LAYER][0,:,:,idx], cmap=cmap, interpolation='none', vmin=-1.0, vmax=1.0)
            
            fig.colorbar(surf, shrink=0.6, aspect=30, fraction=0.05)
        title = 'Layer: {:3d} - {:25s} - shape: {}'.format(LAYER, layer_names[LAYER], str(activations[LAYER][0].shape))        
        fig.suptitle(title, fontsize = 24, ha ='center' )            
#         fig.savfig
        fig_num +=1

    plt.show()


'''
##-----------------------------------------------------------------------------------------------------------    
##
##-----------------------------------------------------------------------------------------------------------     
def display_final_activations_pdf(activations, layers, columns = 8, cmap = 'jet'):
    import matplotlib.backends.backend_pdf
    # import matplotlib.backends.backend_pdf
    # pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
    # for fig in xrange(1, figure().number): ## will open an empty extra figure :(
    #     pdf.savefig( fig )
    # pdf.close()    
    pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")

    n_layers   = len(layers)
    
    fig = plt.figure(  figsize=(8 *columns, 8* n_layers))
    
    fig_num = 1
    for row, layer in enumerate(layers):
        LAYER = layer
        n_features = activations[LAYER].shape[-1]
        rows = math.ceil(n_features / columns )

    #     print()
    #     print('Layer:', LAYER, ' - ',layer_names[LAYER], '   Shape: ', activations[LAYER][0,:,:,:].shape, ' # features: ', n_features, rows, columns)
        for col in range(n_features): 
#             row = idx // columns
#             col = idx  % columns
            subplot = (row * columns) + col +1    
            print(n_layers, n_features, subplot)
            plt.subplot(rows, columns, subplot)
            ax = plt.gca()
            surf = ax.matshow(activations[LAYER][0,:,:,col], cmap=cmap, interpolation='none')
            fig.colorbar(surf, shrink=0.6, aspect=30, fraction=0.05)
        title = 'Layer: {:3d} - {:25s} - shape: {}'.format(LAYER,layer_names[LAYER], str(activations[LAYER][0].shape))        
        fig.suptitle(title, fontsize = 24, ha ='center' )            
        pdf.savefig( fig_num )

        fig_num +=1
        
    pdf.close()
    plt.show()
'''    
def display_fcn_agg_heatmaps_3d(fcn_hm, gt_cls_counts,  dt_cls_counts, class_names = None, classes = None, cmap = 'jet', 
                        columns= 4, title = None, norm = False):
    '''
    This routine is used in the display_aggreggate_heatmaps and exp2 notebooks 
    '''
    if classes is None :
        n_features = range(fcn_hm.shape[-1])
    else:
        n_features = classes

    image_height = fcn_hm.shape[0]
    image_width  = fcn_hm.shape[1]
    
    Y = np.arange(0, image_height, 1)
    X = np.arange(0, image_width, 1)
    X, Y = np.meshgrid(X, Y)        
    xnew, ynew = np.mgrid[-1:1:256j, -1:1:256j]

    rows = math.ceil(len(n_features) / columns )
    fig = plt.figure(figsize=(8 *columns, 8* rows))
    vmin = fcn_hm.min()
    vmax = fcn_hm.max()
    for idx, feat in enumerate(n_features): 
        row = idx // columns
        col = idx  % columns
        subplot = (row * columns) + col +1    
        ax = fig.add_subplot(rows, columns, subplot, projection='3d')
        max = fcn_hm[:,:,feat].max()
        subttl = '{} - {} - GT:{:4d}  Det:{:4d} max: {}'.format(feat, class_names[feat], gt_cls_counts[feat], dt_cls_counts[feat], max)
        ax.set_title(subttl, fontsize=20)
        ax.tick_params(axis='both', labelsize = 13, length = 6, width = 3)      
        ax.invert_yaxis()
        # ax.view_init( azim=-116,elev=40)            
        # surf = ax.plot_surface(X, Y, fcn_hm[:,:,feat], rstride = 1, cstride = 1, cmap=cm.jet, linewidth=0, antialiased=False)
        # surf = ax.plot_surface(X, Y, fcn_hm[:,:,feat], rstride = 20, cstride=20,  cmap=cm.jet, linewidth=0, antialiased=False)
        tck  = interpolate.bisplrep(X, Y, fcn_hm[:,:,feat], s=0)
        znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)
        surf = ax.plot_surface(xnew,ynew, znew, cmap='summer', alpha=None)
        
        # Add a color bar which maps values to colors.
        # plt.subplots_adjust(top=0.97, bottom=0.02, left=0.02, right=0.98, hspace=0.15, wspace=0.15)                
        cbar = fig.colorbar(surf, shrink=0.8, aspect=30, fraction=0.10)

        # cbar = fig.colorbar(surf, shrink=0.6, aspect=30, fraction=0.05)
        # cbar.ax.tick_params(labelsize=9) 
        cbar.ax.tick_params(labelsize=16) 
    
    if title is not None :
        ttl = title + ' {}'.format(str(fcn_hm.shape))           
        fig.suptitle(ttl, fontsize = 24, ha ='center' )        
    plt.subplots_adjust(top=0.9, bottom=0.02, left=0.02, right=0.98, hspace=0.25, wspace=0.15)
    # plt.tight_layout()
    plt.show()
    return fig
    
##-----------------------------------------------------------------------------------------------------------    
##  FUNCTIONS USED TO PLOT SCORE CURVES BASED ON OBJECT MOVEMENT
##  EXP-2 Aggregate Heatmap over out of context prediciton - Newshapes V2
##-----------------------------------------------------------------------------------------------------------     
    
    
def plot_fcn_score_curves(x_y_dim, fcn_scores, mrcnn_scores, class_names, ax = None , 
                          min_x = 0.0, title = None, mrcnn = False, fcn = True, classes = None):
    if ax is None:
        plt.figure(figsize=(10,5))
        ax = plt.gca()

    # scores is always passed ffom plot_mAP_by_scores, so it's nver None
    # so we loop on scores instead of sorted(class_data)
    # for idx, score_key in enumerate(sorted(class_data)):
    if fcn:
        for idx, cls  in enumerate(fcn_scores):
            print('cls: ', cls)
            if cls in classes:
                print('add to plot')
                ax.plot(x_y_dim, fcn_scores[cls], label= cls + ' - FCN score')
    if mrcnn:   
        for idx, cls  in enumerate(mrcnn_scores):
            if cls in classes:
                ax.plot(x_y_dim, mrcnn_scores[cls], label= cls + ' - MR-CNN score')
        
    print(' y limit:', plt.ylim(), ' xlimit : ', plt.xlim())

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Object displacement from origin axis', fontsize= 12)
    ax.set_ylabel('FCN Score', fontsize= 12)
    ax.tick_params(axis='both', labelsize = 10)
#     ax.set_xlim([min_x,1.05])
#     ax.set_ylim([all_scores.min()-0.05, all_scores.max()+0.05])
    leg = plt.legend(loc='lower left',frameon=True, fontsize = 10, markerscale = 6)
    leg.set_title(' Scores ',prop={'size':11})
    plt.grid(True)
#     for xval in np.linspace(0.0, 1.0, 11):
#         plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed', linewidth=1)

def plot_fcn_score_curves_1(x_y_dim, fcn_scores, mrcnn_scores, cls_name, ax = None ,
                            min_x = 0.0, title = None, mrcnn = True, fcn = True):
    if ax is None:
        plt.figure(figsize=(10,5))
        ax = plt.gca()
        
    if fcn:
        for idx, cls  in enumerate(fcn_scores):
            ax.plot(x_y_dim[cls], fcn_scores[cls], label= cls + ' - FCN score')

    if mrcnn:
        for idx, cls  in enumerate(mrcnn_scores):
            ax.plot(x_y_dim[cls], mrcnn_scores[cls], label= cls + ' - MR-CNN score')
        
    print(' y limit:', plt.ylim(), ' xlimit : ', plt.xlim())

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Object displacement from origin axis', fontsize= 12)
    ax.set_ylabel('FCN Score', fontsize= 12)
    ax.tick_params(axis='both', labelsize = 10)
    leg = plt.legend(loc='lower left',frameon=True, fontsize = 10, markerscale = 6)
    leg.set_title(' Scores ',prop={'size':11})
    plt.grid(True)

