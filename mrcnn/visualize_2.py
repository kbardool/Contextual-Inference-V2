"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import math
import random
import itertools
import colorsys
import numpy as np
import IPython.display

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



def display_gt_bboxes(dataset, config, image_id=0, only = None, size = 12):
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
##
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
    print('    seq  id name              | FP| score   score  |    sum      area   gau.sum |  in mask   sum              score  |  in mask   sum              score  |  Y1  X1  Y2  X2   AREA  ')
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
                     pre[0],pre[1], pre[2], pre[3], area ))     #,  pre[4],pre[5],pre[6],roi))
                     
    if isinstance (f_input, dict)  and display:
        img_id = str(f_input['image_meta'][0])        
        visualize.display_instances(f_input['image'] , pr_scores[:,:CLASS_COLUMN], pr_scores[:,CLASS_COLUMN].astype(np.int32), 
                                    class_names, pr_scores[:,ORIG_SCORE_COLUMN], only_classes= only, size =size,
                                    title = 'MRCNN predictions for image id'+img_id)
                                    
                                    
##-----------------------------------------------------------------------------------------------------------    
##
##-----------------------------------------------------------------------------------------------------------    
def display_fcn_scores(f_input, class_names, only = None, display = True, size = 12):
    '''
    fcn_scores:    fcn_scores returned from detection or evaluation process
    cn:   class names
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

#### detections as returned from the model's `detect()` functon
def display_pr_fcn_style1(f, class_names,lmt =18):
    lmt =18
    # f = fcn_results[0]
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



def display_pr_fcn_style2(f, class_names,lmt =18):
    # f = fcn_results[0]
    for i, [molded_bbox, cls, scr, pr_scr, fcn_scr] in enumerate(zip(f['molded_rois'].astype(np.int), f['class_ids'],  f['scores'], f['pr_scores'], f['fcn_scores'])):
        
        print('{} {} {:2d}  {:.<18s}  {:5.4f} {}  '.format(i, molded_bbox, cls, class_names[cls], scr, fcn_scr[[4,5,6,7,8]]))
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
    # r = results[0]
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
def display_mrcnn_scores(r, class_names, only = None, display = True, size = 8):
    return display_pr_scores(r, class_names, only = only, display = display, size = size)

##-----------------------------------------------------------------------------------------------------------    
##
##-----------------------------------------------------------------------------------------------------------        
def display_pr_hm_scores(pr_hm_scores, class_names, only = None):
    '''
    pr_hm_scores:   pr_hm_scores or pr_hm_scores_by_class ( [class, bbox, score info] ) results from mrcnn detect
    class_names :   class names
    
    '''
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
    
##-----------------------------------------------------------------------------------------------------------    
##
##-----------------------------------------------------------------------------------------------------------        
def display_pr_hm_scores_box_info(r, class_names, only = None):
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
        only = range(r.shape[0])
#     seq_start = r['pr_hm_scores'].shape[1]
    # print('  class ids  : ', r['class_ids'], type(r['class_ids']))
    print('  classes     : ', only)
    print('-'*175)
    print('                          ')
    print('        class             mrcnn   normlzd    gauss       bbox   nrm.scr*     ga.sum     mask    score    norm         ga.sum    mask   score   norm     ')
    print('seq  id     name              X1/Y1                  X2/Y2               CX / CY        WIDTH   HEIGHT      AREA                         CV_X   CV_Y')
    print('-'*175)
    for cls in range(r.shape[0]):
        if cls in only:
            for pre in r[cls]:
        #     print(' {:4d}      {:12s}   {:.4f}   {:.4f}     {:.4f}  {:7.4f}  {:.4f}      {:.4f}   {}'.
                width = pre[3]-pre[1]
                height = pre[2]-pre[0]
                cx = pre[1] + width/2
                cy = pre[0] + height/2
                covar_x = width * 0.5
                covar_y = height * 0.5
                area = (pre[3]-pre[1])* (pre[2]-pre[0])

                print('{:3.0f} {:3.0f} {:15s}   ({:7.2f}, {:7.2f})   ({:7.2f}, {:7.2f})   {:7.2f}/{:7.2f}   {:7.2f}   {:7.2f}   {:8.2f}  {:8.2f}  A{:8.4f}   {:8.4f}   {:8.4f}'\
                  ' '.format(pre[6], pre[4],  class_names[cls], 
                         pre[1], pre[0],  pre[3], pre[2], 
                         cx    ,     cy,  width,  height, area, 
                         pre[5], pre[8], covar_x, covar_y))    
            print('-'*170)
    return
    



##-----------------------------------------------------------------------------------------------------------    
##
##-----------------------------------------------------------------------------------------------------------        
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
    print('seq  id     name              Y1/X1              Y2/X2   |    CX / CY         (W)  ~  (H)      AREA        |  W/2     H/2  |  X       Y        A |  X       Y       A')
    print('-'*175)                                                                                                                                          
    
    
    for pre in fcn_scores:
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
        
        cls = int(pre[4])
        if cls in only:
            print('{:3.0f} {:3.0f} {:15s}'\
              ' ({:6.2f},{:6.2f})  ({:6.2f},{:6.2f}) |'\
              ' {:6.2f}/{:6.2f}  {:7.2f}~{:7.2f}  {:7.2f} {:7.2f} |'\
              ' {:6.2f} {:6.2f} |'\
              ' {:3d} {:3d} {:3d} {:3d} {:3d} |'\
              ' {:3.0f} {:3.0f} {:3.0f} {:3.0f} {:3.0f}'.
              format(pre[7], pre[4],  class_names[cls], 
                     pre[0], pre[1],  pre[2], pre[3], 
                     cx, cy,  width,  height, area, pre[13],
                     sq_covar_x, sq_covar_y, 
                     from_x, to_x, from_y, to_y , clip_area,
                     from_x_r, to_x_r, from_y_r, to_y_r, clip_area_r
                  ))    
            print(' {:2s}   mrcnn_scr: {:6.4f}'.format('TP' if pre[6] == 1 else 'FP', pre[5]))
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
    

##-----------------------------------------------------------------------------------------------------------    
##
##-----------------------------------------------------------------------------------------------------------        
def display_fcn_scores_box_info2(fcn_scores, class_names, fcn_hm = None, only = None):
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
    print('-'*175)
    print('                                       |                                              |   (COVAR)     |    ')
    print('BOX                                    |                   Width   Height             |     SQRT      |   FROM/TO  ')
    print('seq       X1/Y1              X2/Y2     |    CX / CY         (W)  ~  (H)      AREA     |  W/2    H/2   |  X      Y ')
    print('-'*175)   
    
    for pre in fcn_scores:
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
        
        cls = int(pre[4])
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
        
        print(' {:3s}  Original MRCNN Prediction for this box:  [{:2.0f}] - {:15s}  Orig Score: {:6.4f}  Norm Score:  {:6.4f} '.
              format('TP' if pre[6] == 1 else 'FP', pre[4],  class_names[cls], pre[5], pre[8]))
        print('-'*150)   
        print()
        cx_cy_score     = ''.join([ '{:15.4f}'.format(np.sum(fcn_hm[int(cx),int(cy),i])                                )  for i in class_list])
        cx_cy_pm1_score = ''.join([ '{:15.4f}'.format(np.sum(fcn_hm[int(cx)-1:int(cx)+1,int(cy)-1:int(cy)+1,i])         )  for i in class_list])
        cx_cy_pm3_score = ''.join([ '{:15.4f}'.format(np.sum(fcn_hm[int(cx)-3:int(cx)+3,int(cy)-3:int(cy)+3,i])         )  for i in class_list])
        fr_to_score     = ''.join([ '{:15.4f}'.format(np.sum(fcn_hm[from_y:to_y, from_x:to_x,i])                        )  for i in class_list])
        fr_to_p1_score  = ''.join([ '{:15.4f}'.format(np.sum(fcn_hm[from_y:to_y+1, from_x:to_x+1,i])                    )  for i in class_list])
        fr_to_pm1_score = ''.join([ '{:15.4f}'.format(np.sum(fcn_hm[from_y-1:to_y+1, from_x-1:to_x+1,i])                )  for i in class_list])
        full_box_score  = ''.join([ '{:15.4f}'.format(np.sum(fcn_hm[int(pre[1]):int(pre[3]), int(pre[0]):int(pre[2]),i]))  for i in class_list])
        
        
        print(' '*20,'-'*130)
        print(' '*20, '                    ',sub_title)
        print(' '*20,'-'*130)
        print(' '*20, '       cx_cy score :', cx_cy_score)
        print(' '*20, '  cx_cy +/-1 score :', cx_cy_pm1_score)
        print(' '*20, '  cx_cy +/-3 score :', cx_cy_pm3_score)
        print(' '*20, '     from:to score :', fr_to_score)
        print(' '*20, '  from:to +1 score :', fr_to_p1_score)
        print(' '*20, 'from:to +/-1 score :', fr_to_pm1_score)
        print(' '*20, '   full bbox score :', full_box_score)
        print()
    print('-'*175)
    return

##-----------------------------------------------------------------------------------------------------------    
##
##-----------------------------------------------------------------------------------------------------------        
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
##
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
    out = np.clip(x, 0, 1)

    # convert to RGB array
    out *= 255
    out = np.clip(out, 0, 255).astype('uint8')
    return out

def standardize_fcn_hm(x):
    # normalize tensor: center on 0., ensure std is 0.1
    print(x.shape)
    out = x - x.mean(axis=(0,1), keepdims = True)
    out /= (x.std(axis=(0,1), keepdims = True) + 1e-7)
    return out

def normalize_fcn_hm(x):
    # normalize tensor: center on 0., ensure std is 0.1
    print(x.shape)
    out = x - x.min(axis=(0,1), keepdims = True)
    out /= (x.max(axis=(0,1), keepdims = True) - x.min(axis=(0,1), keepdims = True))
    return out
    

##-----------------------------------------------------------------------------------------------------------    
##
##-----------------------------------------------------------------------------------------------------------    
def display_fcn_input(fcn_hm, class_names,  cmap = 'jet', columns = 8 ):
    if fcn_hm.ndim == 4:
        fcn_hm = fcn_hm[0]
    n_features = fcn_hm.shape[-1]
    rows = math.ceil(n_features / columns )

#     print()
#     print('Layer:', LAYER, ' - ',layer_names[LAYER], '   Shape: ', activations[LAYER][0,:,:,:].shape, ' # features: ', n_features, rows, columns)
    fig = plt.figure(figsize=(10 *columns, 10* rows))
    for idx in range(n_features): 
        row = idx // columns
        col = idx  % columns
        subplot = (row * columns) + col +1    
        ax= fig.add_subplot(rows, columns, subplot)
        ax.set_title(class_names[idx], fontsize=38)
        # ax.tick_params(direction='out', length=6, width=1, colors='r', labelsize = 10)
        
        img = ax.imshow(fcn_hm[:,:,idx], origin = 'upper', cmap = cmap , interpolation='none')
        ax.tick_params(axis='both', bottom=True, left=True , top=False, right=False)
        cb = fig.colorbar(img, shrink=0.6, aspect=30, fraction=0.02)
        cb.ax.tick_params(labelsize=25)
    title = 'FCN Input - shape: {}'.format(str(fcn_hm.shape))        

    fig.suptitle(title, fontsize = 18, ha ='center' )        
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])   ## [left, bottom, right, top]
    # plt.subplots_adjust(top=0.95, bottom=0.02, left=0.02, right=0.98, hspace=0.25, wspace=0.15)
    plt.show()
    
    return

display_fcn_output  = display_fcn_input


##-----------------------------------------------------------------------------------------------------------    
##
##-----------------------------------------------------------------------------------------------------------    
def display_fcn_input_2(fcn_hm, cmap = 'gray', columns = 8, size  = 4):
    if fcn_hm.ndim == 4:
        fcn_hm = fcn_hm[0]
    n_features = fcn_hm.shape[-1]
    
    rows = math.ceil(n_features / columns )
    fig , ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    print(n_features, rows )
#     print('Layer:', LAYER, ' - ',layer_names[LAYER], '   Shape: ', activations[LAYER][0,:,:,:].shape, ' # features: ', n_features, rows, columns)
#     fig = plt.figure(figsize=(8 *columns, 12* rows))
    for idx in range(n_features): 
        row = idx // cols
        col = idx  % cols
        subplot = (row * cols) + col +1    
        print(row,col, subplot)
#         ax= fig.add_subplot(rows, columns, subplot)
        ax[row,col].set_title(class_names[idx], fontsize=20)
        ax[row,col].tick_params(axis='both', labelsize = 5)

        surf = ax[row,col].matshow(fcn_hm[:,:,idx], cmap = cmap , interpolation='none')
        cbar = ax[row,col].colorbar(cax = surf, shrink=0.6, aspect=30, fraction=0.05)
        
        
        
    ttl = title + ' {}'.format(str(fcn_hm.shape))        
    fig.suptitle(title, fontsize = 18, ha ='center' )      
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])   ## [left, bottom, right, top]
    plt.show()

##-----------------------------------------------------------------------------------------------------------    
##
##-----------------------------------------------------------------------------------------------------------    
def display_fcn_input_3(fcn_hm, gt_cls_counts,  dt_cls_counts, features = None, class_names = None, cmap = 'gray', columns= 4, title = 'FCN Heatmaps'):
    if features is None :
        n_features = range(fcn_hm.shape[-1])
    else:
        n_features = features

    rows = math.ceil(len(n_features) / columns )

#     print()
#     print('Layer:', LAYER, ' - ',layer_names[LAYER], '   Shape: ', activations[LAYER][0,:,:,:].shape, ' # features: ', n_features, rows, columns)
    fig = plt.figure(figsize=(8 *columns, 8* rows))
    for idx, feat in enumerate(n_features): 
        row = idx // columns
        col = idx  % columns
        subplot = (row * columns) + col +1    
        ax= fig.add_subplot(rows, columns, subplot)
        subttl = '{} - {} - GT:{:4d}  Det:{:4d}'.format(feat, class_names[feat], gt_cls_counts[feat], dt_cls_counts[feat])
        ax.set_title(subttl, fontsize=20)
        ax.tick_params(axis='both', labelsize = 5)
        
        surf = ax.matshow(fcn_hm[:,:,feat], cmap = cmap , interpolation='none')
        cbar = fig.colorbar(surf, shrink=0.6, aspect=30, fraction=0.05)
        cbar.ax.tick_params(labelsize=14) 
        
    ttl = title + ' {}'.format(str(fcn_hm.shape))           
    fig.suptitle(ttl, fontsize = 24, ha ='center' )        
    plt.subplots_adjust(top=0.9, bottom=0.02, left=0.02, right=0.98, hspace=0.25, wspace=0.15)
    plt.show()

    
    

##-----------------------------------------------------------------------------------------------------------    
##
##-----------------------------------------------------------------------------------------------------------    
def display_pairwise_heatmap(input_matrix, CLASS_IDS, CLASS_NAMES,title = '' , colormap = cm.coolwarm):
    # show_labels = labels[1:10]
#     num_classes = len(show_labels)
    #     print(' num classes: ', num_classes, 'matrix shape: ',show_matrix.shape)
    show_matrix = input_matrix[CLASS_IDS, :20]
    n_rows = show_matrix.shape[0] 
    n_cols = show_matrix.shape[1]
    show_labels_x = [str(i) for i in range(n_cols)] 
    show_labels_y = ['{:2d}-{:s}'.format(i,CLASS_NAMES[i]) for i in CLASS_IDS] 

    fig, ax = plt.subplots(1,1,figsize=((n_rows)*1, (n_cols)*1))
    im = ax.imshow(show_matrix, cmap=colormap)
    # cmap=cm.bone, #  cmap=cm.Dark2 # cmap = cm.coolwarm   # cmap=cm.YlOrRd
    # We want to show all ticks...
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    
    # ... and label them with the respective list entries
    ax.set_xticklabels(show_labels_x, size = 9)
    ax.set_yticklabels(show_labels_y, size = 9)
    ax.set_xlabel('Number of Instances ', size = 16)
    ax.set_ylabel('Class', size = 16 )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(n_rows):
        for j in range(n_cols):
                text = ax.text(j, i, show_matrix[i, j], size=12, 
                           ha="center", va="center", color="w")
 
    ax.set_title("class instances in image"+ title.upper(), size = 14)
    # fig.tight_layout()
    plt.show()
    return  fig
    

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
    
    