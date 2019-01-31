"""
author: Timothy C. Arlen
date: 28 Feb 2018

Calculate Mean Average Precision (mAP) for a set of bounding boxes corresponding to specific
image Ids. Usage:

> python calculate_mean_ap.py

Will display a plot of precision vs recall curves at 10 distinct IoU thresholds as well as output
summary information regarding the average precision and mAP scores.

NOTE: Requires the files `ground_truth_boxes.json` and `predicted_boxes.json` which can be
downloaded fromt this gist.
"""

from __future__ import absolute_import, division, print_function

from copy import deepcopy
import json
import glob
import os
import time
import math
import pprint 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('white')
sns.set_context('poster')
pp = pprint.PrettyPrinter(indent=2, width=100)
COLORS = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']


def calc_iou_individual(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box

    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

    Returns:
        float: value of the IoU for the two boxes.

    Raises:
        AssertionError: if the box is obviously malformed
    """
    # x1_t, y1_t, x2_t, y2_t = gt_box
    # x1_p, y1_p, x2_p, y2_p = pred_box
    y1_t, x1_t, y2_t, x2_t = gt_box
    y1_p, x1_p, y2_p, x2_p = pred_box

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x  = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y  = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou


def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.

    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.

    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """

    all_pred_indices = range(len(pred_boxes))
    all_gt_indices   = range(len(gt_boxes))
    
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
        
    if len(all_gt_indices) == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

    gt_idx_thr   = []
    pred_idx_thr = []
    ious         = []
    
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_iou_individual(pred_box, gt_box)
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)

    return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}


def calc_precision_recall(img_results):
    """Calculates precision and recall from the set of images

    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }

    Returns:
        tuple: of floats of (precision, recall)
    """
    true_pos = 0; false_pos = 0; false_neg = 0
    for _, res in img_results.items():
        true_pos += res['true_pos']
        false_pos += res['false_pos']
        false_neg += res['false_neg']

    try:
        precision = true_pos/(true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = true_pos/(true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0.0

    return (precision, recall)

def get_model_scores_map(pred_boxes, score_key):
    """Creates a dictionary of from model_scores to image ids.

    Args:
        pred_boxes (dict): dict of dicts of 'boxes' and 'scores'

    Returns:
        dict: keys are model_scores and values are image ids (usually filenames)
        example:
            0.100929 : ['COCO_val2014_000000144798.jpg'],
            0.104556 : ['COCO_val2014_000000481573.jpg'],
    """
    model_scores_map = {}
    for img_id, val in pred_boxes.items():
        # for raw_score in val['scores']:
        for score in val[score_key]:
            # score = round(raw_score, 4)  <-- we are now writing all scores in rounded format
            if score not in model_scores_map.keys():
                model_scores_map[score] = [img_id]
            else:
                model_scores_map[score].append(img_id)
    return model_scores_map

    
def get_avg_precision_at_iou(gt_boxes, pr_boxes, iou_thr=0.5, score_key = 'scores'):
    """Calculates average precision at given IoU threshold.

    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (list of list of floats): list of locations of predicted
            objects as [xmin, ymin, xmax, ymax]
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.

    Returns:
        dict: avg precision as well as summary info about the PR curve

        Keys:
            'avg_prec' (float): average precision for this IoU threshold
            'precisions' (list of floats): precision value for the given
                model_threshold
            'recall' (list of floats): recall value for given
                model_threshold
            'models_thrs' (list of floats): model threshold value that
                precision and recall were computed for.
    """
    ## 01-05-19: added to prevent corruption of original data passed to function
    ## TODO: merge pred_boxes and pred_boxes_pruned to conserve memory
    pred_boxes = deepcopy(pr_boxes)
    
    model_scores_map    = get_model_scores_map(pred_boxes, score_key = score_key)
    sorted_model_scores = sorted(model_scores_map.keys())

    ## Sort the predicted boxes in ascending score order (lowest scoring boxes first):
    for img_id in pred_boxes.keys():
        # print()
        # print('image_id : ', img_id)
        # print('--------------------------')
        # print('scores:', pred_boxes[img_id]['scores'] )
        # print(score_key, ':' ,pred_boxes[img_id][score_key] )
        # print(pred_boxes[img_id]['boxes'] )
        
        arg_sort = np.argsort(pred_boxes[img_id][score_key])        
        pred_boxes[img_id]['scores'] = np.array(pred_boxes[img_id][score_key])[arg_sort].tolist()
        pred_boxes[img_id]['boxes']  = np.array(pred_boxes[img_id]['boxes'])[arg_sort].tolist()
        
        # print('after argsort:' , arg_sort)
        # print('--------------------------')
        # print('scores:', pred_boxes[img_id]['scores'] )
        # print(score_key, ':' ,pred_boxes[img_id][score_key] )
        # print(pred_boxes[img_id]['boxes'] )

    pred_boxes_pruned = deepcopy(pred_boxes)

    precisions  = []
    recalls     = []
    model_thrs  = []
    img_results = {}
    
    # Loop over model score thresholds and calculate precision, recall
    for ithr, model_score_thr in enumerate(sorted_model_scores[:-1]):
        # On first iteration, define img_results for the first time:
        # print('------------------------------------------------')
        # print('ithr ', ithr, 'model_scr_thr', model_score_thr)
        # print('------------------------------------------------')
        img_ids = gt_boxes.keys() if ithr == 0 else model_scores_map[model_score_thr]
        for img_id in img_ids:
            gt_boxes_img = gt_boxes[img_id]['boxes']
            box_scores = pred_boxes_pruned[img_id]['scores']
            start_idx = 0
            for score in box_scores:
                if score <= model_score_thr:
                    # pred_boxes_pruned[img_id]
                    start_idx += 1
                else:
                    break

            # Remove boxes, scores of lower than threshold scores:
            pred_boxes_pruned[img_id]['scores'] = pred_boxes_pruned[img_id]['scores'][start_idx:]
            pred_boxes_pruned[img_id]['boxes']  = pred_boxes_pruned[img_id]['boxes'][start_idx:]

            # Recalculate image results for this image
            img_results[img_id] = get_single_image_results(
                gt_boxes_img, pred_boxes_pruned[img_id]['boxes'], iou_thr)

            # print('Start Idx is ', start_idx)
            # print('image_id : ', img_id)
            # print('--------------------------')            
            # pp.pprint(gt_boxes_img)
            # pp.pprint(pred_boxes_pruned[img_id]['boxes'])
            # pp.pprint(img_results[img_id])
            # print()

        prec, rec = calc_precision_recall(img_results)
        # print('precision:', prec, 'Recall:', rec)
        
        precisions.append(prec)
        recalls.append(rec)
        model_thrs.append(model_score_thr)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    # print('final precsions:', precisions)
    # print('final recall   :', recalls)
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            args = np.argwhere(recalls >= recall_level).flatten()
            prec = max(precisions[args])
        except ValueError:
            prec = 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)

    return {
        'avg_prec'      : avg_prec,
        'precisions'    : precisions,
        'recalls'       : recalls,
        'model_thrs'    : model_thrs,
        'prec_at_rec'   : prec_at_rec }


def plot_pr_curve(
    precisions, recalls, category='Not Supplied', label=None, color=None, ax=None):
    """Simple plotting helper function"""

    if ax is None:
        plt.figure(figsize=(10,8))
        ax = plt.gca()

    if color is None:
        color = COLORS[0]
    ax.plot(recalls, precisions, label=label,  color=color)
    # ax.scatter(recalls, precisions, label=label, s=4, color=color)
    ax.set_xlabel(' recall ')
    ax.set_ylabel(' precision ')
    # ax.set_title('Precision-Recall curve for {}'.format(category))
    ax.set_xlim([0.0,1.2])
    ax.set_ylim([0.0,1.2])
    return ax


    
##------------------------------------------------------------------------------------------
##  Update mAP Dictionaries
##------------------------------------------------------------------------------------------    
def update_map_dictionaries(results, gt_dict, pr_dict, class_dict):
    orig_score = 5
    norm_score = 8
    alt_scr_0  = 11
    alt_scr_1  = 14   # in MRCNN alt_scr_1 ans alt_scr_2 are the same
    alt_scr_2  = 20    
    r = results[0]
    assert r['class_ids'].shape[0] == r['pr_scores'].shape[0] == r['fcn_scores'].shape[0], " {} {} {} {} ".format(
           r['class_ids'].shape, r['pr_scores'].shape,  r['fcn_scores'].shape, r['image_meta'])

    keyname = 'newshapes_{:05d}'.format(r['image_meta'][0])
    zero_ix = np.where(r['gt_bboxes'][:, 3] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else r['gt_bboxes'].shape[0]

#     for key in sorted(r):
#         print(key.ljust(20), r[key].shape) 
#     print(r['image_meta'])
#     print(' Keyname :' , keyname)
#     print('keyname:', keyname, 'zero_ix.shape: ', zero_ix.shape,'r[gt_bboxes].shape', r['gt_bboxes'].shape, ' N:', N)
#     print(r['gt_bboxes'][:N,:].tolist())
#     print(r['gt_class_ids'][:N].tolist())

    gt_dict[keyname] = {"boxes"     : r['gt_bboxes'][:N,:].tolist(),
                        "class_ids" : r['gt_class_ids'][:N].tolist()}
      
    pr_dict[keyname] =  {'scores': [], 'boxes':[], 'class_ids': [], 'det_ind' :[],
                        "mrcnn_score_orig": [],    
                        "mrcnn_score_norm": [],
                        "mrcnn_score_0"   : [],
                        "mrcnn_score_1"   : [],
                        "mrcnn_score_2"   : [],
                        "fcn_score_0"     : [],
                        "fcn_score_1"     : [],
                        "fcn_score_2"     : [] }
    
    for cls, score, bbox, pr_score, fcn_score, det_ind in zip(r['class_ids'].tolist(), r['scores'].tolist(), r['molded_rois'].tolist(), 
                                                              np.round(r['pr_scores'],4).tolist(), 
                                                              np.round(r['fcn_scores'],4).tolist(), 
                                                              r['detection_ind'].tolist()):
        pr_dict[keyname]['class_ids'].append(cls)
        pr_dict[keyname]['scores'].append(np.round(score,4))
        pr_dict[keyname]['boxes'].append(bbox)
        pr_dict[keyname]['det_ind'].append(np.rint(det_ind))
                         
        pr_dict[keyname]["mrcnn_score_orig"].append(pr_score[orig_score])
        pr_dict[keyname]["mrcnn_score_norm"].append(pr_score[norm_score])
        
        pr_dict[keyname]["mrcnn_score_0"   ].append(pr_score[alt_scr_0])
        pr_dict[keyname]["mrcnn_score_1"   ].append(pr_score[alt_scr_1])
        pr_dict[keyname]["mrcnn_score_2"   ].append(pr_score[alt_scr_2])
        
        pr_dict[keyname]["fcn_score_0"     ].append(fcn_score[alt_scr_0])
        pr_dict[keyname]["fcn_score_1"     ].append(fcn_score[alt_scr_1])
        pr_dict[keyname]["fcn_score_2"     ].append(fcn_score[alt_scr_2])

#         print('class_dict[cls]: ', cls, class_dict[cls]['scores'])
        class_dict[cls]['scores'].append(np.round(score,4))
        class_dict[cls]['bboxes'].append(bbox)
        class_dict[cls]["mrcnn_score_orig"].append(pr_score[orig_score])
        class_dict[cls]["mrcnn_score_norm"].append(pr_score[norm_score])
        class_dict[cls]["mrcnn_score_0"   ].append(pr_score[alt_scr_0])
        class_dict[cls]["mrcnn_score_1"   ].append(pr_score[alt_scr_1])
        class_dict[cls]["mrcnn_score_2"   ].append(pr_score[alt_scr_2])
        
        class_dict[cls]["fcn_score_0"     ].append(fcn_score[alt_scr_0])
        class_dict[cls]["fcn_score_1"     ].append(fcn_score[alt_scr_1])
        class_dict[cls]["fcn_score_2"     ].append(fcn_score[alt_scr_2])

    return gt_dict, pr_dict, class_dict    
    
##------------------------------------------------------------------------------------------
##  Plot Score Distribution
##------------------------------------------------------------------------------------------    
def plot_score_distribution(all_class_info, score, columns = 4, kde = True):
#     ext_class_ids = [1,2,3,4,5,6]
#     class_ids = [1,2,3,4,5,6]
    num_classes = len(all_class_info)
    rows     = math.ceil(num_classes/columns)
    fig = plt.figure(figsize=(columns*8, rows * 5))

#     for idx,cls in enumerate(class_ids):
    idx = 0
    for class_info in all_class_info:
        if class_info['id'] == 0: 
            continue
        cls = class_info['id']
        cls_name = class_info['name']

        row = idx // columns
        col = idx  % columns
        subplot = (row * columns) + col +1
#         lbl = "{:2d} - {}".format(cls, class_names[cls])
        mean = np.mean(class_info[score])
        median = np.median(class_info[score])
        std_dev = np.std(class_info[score])
        lbl = "{:2d} - {:s}  mean:{:.4f}  median:{:.4f}  std:{:.4f}".format(cls, cls_name, mean, median, std_dev)
        ax = fig.add_subplot(rows, columns, subplot)    
        ax.set_title(lbl, fontsize=16)
        x = class_info[score]
        sns.distplot(x, ax = ax, kde = kde, rug = True)
        idx += 1
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    
##------------------------------------------------------------------------------------------
##  filter  mAP data structure by class and return info only pertinent to class_id 
##------------------------------------------------------------------------------------------    
def filter_by_class(gt_boxes, pr_boxes, class_ids):
    assert class_ids is not None
    if not isinstance(class_ids, list):
        class_ids = [class_ids]
        
    pr_keys_len = len(pr_boxes.keys())
    gt_keys_len = len(gt_boxes.keys())
    assert pr_keys_len == gt_keys_len,  "Number of keys in two input dicts don't match"
    print(' # pr keys :', pr_keys_len, '# gt_keys: ', gt_keys_len)

    output_gt_boxes = {}
    output_pr_boxes = {}
    
    for class_id in class_ids:
        print(' Processing class : ', class_id)
        pr_boxes_class = {}
        gt_boxes_class = {}
        for key in gt_boxes.keys():
            kk = [ i  for i,j in enumerate(gt_boxes[key]['class_ids']) if j == class_id]
            jj = [ i  for i,j in enumerate(pr_boxes[key]['class_ids']) if j == class_id]
            if (len(kk) == len(jj) == 0 ):
    #             print(' Nothing found for this class_id, skip this entry')
                continue
            pr_boxes_class[key] = {}
            for sub_key in pr_boxes[key].keys():
    #             print('Key: ' , key, 'sub_key: ',sub_key)
                pr_boxes_class[key].setdefault(sub_key, [pr_boxes[key][sub_key][j]     for j in jj])

            gt_boxes_class[key] = {"boxes"    : [gt_boxes[key]['boxes'][k]     for k in kk],
                                   "class_ids" : [gt_boxes[key]['class_ids'][k] for k in kk] }
        output_gt_boxes[class_id] = gt_boxes_class
        output_pr_boxes[class_id] = pr_boxes_class
        # print(key)
        # print('indexes for gt_boxes: ', kk)
        # print('indexes for pr_boxes: ', jj)        
#         print('gt_boxes     : ',[gt_boxes[key]['boxes'][k] for k in kk])
#         print('gt_class_ids : ',[gt_boxes[key]['class_ids'][k] for k in kk])
#         print('pr_boxes     : ',[pr_boxes[key]['boxes'][j] for j in jj])
#         print('pr_scores    : ',[pr_boxes[key]['scores'][j] for j in jj])
#         print('pr_class_ids : ',[pr_boxes[key]['class_ids'][j] for j in jj])
                    
    return output_gt_boxes, output_pr_boxes 

    
##------------------------------------------------------------------------------------------
##  Build per-class mAP data structure 
##------------------------------------------------------------------------------------------

def build_mAP_data_structure_by_class(gt_boxes_class, pr_boxes_class, class_ids, scores, iou_thresholds = None):
    '''
    build AP info for each class at different thresholds 
    
    mAP_data                        is a dictionary keyed by class_id, e.g. mAP_data[1]. 
    
    Each CLASS DICTIONARY           (mAP_data[n]) dict keyed by the score name, e.g. 'mrcnn_score_orig', 'mrcnn_score_alt', etc....
    
    Each CLASS-SCORE DICTIONARY     (mAP_data[n]['score_name']) dict keyed by iou threshold. e.g. 0.5, 0.55,...,0.95
    
    Each CLASS-SCORE-IOU DICTIONARY (mAP_data[n]['score_name'][0.5]) dict to Precision/Recall information for that
                                    Score and given threshold and has the following keys: 
                                    {'iou', 'model_thrs', 'recalls', 'precisions', 'avg_prec'}
    
    iou :         indicates the iOU threshold of the dictionary entry
    model_thrs:   score thresholds
    recalls   :   recall at threshold
    precision :   precision at threshold
    
    mAP_data[1]:  {'score1': { 0.50: {'iou', 'model_thrs', 'recalls', 'precisions', 'avg_prec'}
                               0.55: {'iou', 'model_thrs', 'recalls', 'precisions', 'avg_prec'}
                               ...
                               ...
                               0.95: {'iou', 'model_thrs', 'recalls', 'precisions', 'avg_prec'}
                             }
                   'score2': { 0.50: {'iou', 'model_thrs', 'recalls', 'precisions', 'avg_prec'}
                               ...
                             }
                  }          
                    
    '''
    assert class_ids is not None
    assert scores is not None 
    
    print('Build mAP information for classes: ', class_ids, ' and scores ', scores)
    mAP_data = {}
    if iou_thresholds is None :
        iou_thresholds = np.arange(0.20, 0.95, 0.05)
#     class_ids = [1,2,3,4,5,6]
#     scores    = ['scores', 'mrcnn_score_orig', 'mrcnn_score_norm']
    
    for class_id in class_ids:
        # mAP_data[class_id] = {}
        class_by_score_data = {}
        print(  'class_id: {:3d}  '.format(class_id))
        for score_key in scores:
            mAP_by_iou_thr_data = {}
            for idx, thr in enumerate(iou_thresholds):
                iou_thr = np.round(thr, 2)
                # print(  'class_id: {:3d}   idx {:2d}   iou_thr: {:.2f}  score_key: {:20s}'.format(class_id, idx, iou_thr, score_key))
                outp = get_avg_precision_at_iou(gt_boxes_class[class_id], pr_boxes_class[class_id], iou_thr=iou_thr, score_key = score_key)
                outp['iou'] = iou_thr
                mAP_by_iou_thr_data[iou_thr] = outp
            class_by_score_data[score_key] = mAP_by_iou_thr_data

        mAP_data[class_id] = class_by_score_data
    return mAP_data

##------------------------------------------------------------------------------------------
##  Build mAP data structure (for all classes combined)
##------------------------------------------------------------------------------------------
def build_mAP_data_structure_combined(gt_boxes, pr_boxes, scores, iou_thresholds = None):
    '''
    build AP info at different thresholds (ALL CLASSES COMBINED)
    
    mAP_data                 : a dictionary keyed by the score name, e.g.  'mrcnn_score_orig', 'mrcnn_score_alt', etc....
    
    Each SCORE DICTIONARY    : (mAP_data['score_name']) is a dict keyed by iou threshold. e.g. 0.5, 0.55,...,0.95

    Each SCORE-IOU DICTIONARY: (mAP_data['score_name'][iou_threshold]) is a dict to Precision/Recall information for that 
                               Score and given threshold and has the following keys: 
                               {'iou', 'model_thrs', 'recalls', 'precisions', 'avg_prec'}
    
    iou :         indicates the iOU threshold of the dictionary entry
    model_thrs:   score thresholds
    recalls   :   recall at threshold
    precision :   precision at threshold
    
    mAP_data[1]:  {'score1': { 0.50: {'iou', 'model_thrs', 'recalls', 'precisions', 'avg_prec'}
                               0.55: {'iou', 'model_thrs', 'recalls', 'precisions', 'avg_prec'}
                               ...
                               ...
                               0.95: {'iou', 'model_thrs', 'recalls', 'precisions', 'avg_prec'}
                             }
                   'score2': { 0.50: {'iou', 'model_thrs', 'recalls', 'precisions', 'avg_prec'}
                               ...
                             }
                  }          
                    
    '''
    assert scores is not None 
    
    print('Build mAP (all classes combined) ', '\n For scores: ', scores)
    mAP_data = {}
    class_id = 0 
    
    if iou_thresholds is None :
        iou_thresholds = np.arange(0.20, 0.95, 0.05)

    for score_key in scores:
        mAP_by_iou_thr_data = {}
        # print(  ' score_key: {:20s} '.format(score_key))
        for idx, thr in enumerate(iou_thresholds):
            iou_thr = np.round(thr, 2)
            print(  ' score_key: {:20s}    iou_thr: {:.2f}  (idx {:2d})  '.format(score_key,iou_thr,idx))
            outp = get_avg_precision_at_iou(gt_boxes, pr_boxes, iou_thr=iou_thr, score_key = score_key)
            outp['iou'] = iou_thr
            mAP_by_iou_thr_data[iou_thr] = outp
        mAP_data[score_key] = mAP_by_iou_thr_data


    return mAP_data

    
    
##------------------------------------------------------------------------------------------
##  Plot plot_pr_curves_by_ious
##------------------------------------------------------------------------------------------

def plot_pr_curves_by_ious(class_data, class_id, class_name , score = None, ax = None ):
    avg_precs = []
    iou_thrs = []
    score_key = score
    
    for idx, iou_key in enumerate(sorted(class_data[score_key])):
#         pp.pprint(class_data[score_key][iou_key])
        iou_thr = class_data[score_key][iou_key]['iou']
        avg_precs.append(class_data[score_key][iou_key]['avg_prec'])
        iou_thrs.append(iou_thr)    
        precisions = class_data[score_key][iou_key]['precisions']
        recalls = class_data[score_key][iou_key]['recalls']
        ax = plot_pr_curve(precisions, recalls, label='{:.2f}'.format(iou_thr), color=COLORS[idx*2], ax=ax)


    # prettify for printing:
    avg_precs = [float('{:0.4f}'.format(ap)) for ap in avg_precs]
    iou_thrs  = [float('{:0.4f}'.format(thr)) for thr in iou_thrs]
    mAP = 100*np.mean(avg_precs)

    ax.set_xlabel('recall', fontsize= 16)
    ax.set_ylabel('precision', fontsize= 16)
    ax.tick_params(axis='both', labelsize = 15)
    ax.set_xlim([0.0,1.1])
    ax.set_ylim([0.0,1.1])
    if class_id == 0:
        ttl = 'PR curve for Score: {}   mAP: {:.2f}'.format(score.upper(),  mAP)
    else:
        ttl = 'PR curve for Score: {}  Class: {:2d} - {}  mAP: {:.2f}'.format(score.upper(), class_id, class_name, mAP)
    ax.set_title(ttl , fontsize=16)
    leg = plt.legend(loc='lower right',frameon=True, fontsize = 'xx-small', markerscale = 6)
    leg.set_title('IoU Thr',prop={'size':12})
    for xval in np.linspace(0.0, 1.0, 11):
        plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed', linewidth=2)

#     str_avg_precs = ''.join([" {:10.4f}".format(ap)  for ap in avg_precs])
#     str_iou_thrs  = ''.join([" {:10.4f}".format(thr) for thr in iou_thrs])
#     print('--------------------------------------------')
#     print(' class_id : ', class_id, ' - ', class_name)
#     print('--------------------------------------------')
#     print('iou_thrs : ', str_iou_thrs)
#     print('avg precs: ', str_avg_precs)
#     print('map      : %{:.2f}'.format(mAP))
#     print()
    return avg_precs, iou_thrs



##------------------------------------------------------------------------------------------
##  Plot mAPs by IoU PR Curves
##------------------------------------------------------------------------------------------
# _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
def plot_mAP_by_IOU(all_data, score , class_ids = None , class_names = None, columns = 3):
    print(class_names)
    if class_ids is None:
        disp_classes = all_data.keys()
    else:
        if not isinstance(class_ids, list):
            class_ids = [class_ids]
        disp_classes = class_ids       ## [36,37,38,39,40,41] #,42]
    all_precs = {}
    all_thrs  = []
    all_mAPs  = {}
    disp_score   = score
    num_disp_classes = len(disp_classes)
    columns = min(columns, num_disp_classes)
    rows    = math.ceil(num_disp_classes/columns)
    fig = plt.figure(figsize=(8 *columns, 5* rows))

    for idx, class_id in enumerate(disp_classes):
        #         print('idx:', idx, 'class_id: ',class_id)
        row = idx // columns
        col = idx  % columns
        subplot = (row * columns) + col +1

        ax= fig.add_subplot(rows, columns, subplot)
        avg_precs, iou_thrs = plot_pr_curves_by_ious(all_data[class_id], class_id, class_names[class_id], score = disp_score , ax = ax)    
        all_precs[class_id] = avg_precs
        all_mAPs[class_id]  = 100*np.mean(avg_precs)
        all_thrs.append(''.join([" {:10.4f}".format(thr) for thr in iou_thrs]))

    ## Print Summary 
    ttl = ' AP @ IoU Thresholds for Score Computation: {}'.format(score)
    sum = np.zeros((len(all_precs[0])))
    cnt = 0

    print()
    print(ttl.center(140))
    print()
    print('{:-^140}'.format('  IoU Thresholds  '))
    print('Id - ClassName{:15s}{}       mAP'.format(' ', all_thrs[0]))
    print('-'*140)
    for cls in sorted(all_precs):
        scores = ''.join([" {:10.4f}".format(ap)  for ap in all_precs[cls]])
        if cls != 0 :
            sum += np.asarray(all_precs[cls])
            cnt += 1
            print('{:3d} - {:20s}   {}      %{:.2f} '.format(cls , class_names[cls], scores,  all_mAPs[cls]))
        # print('cls: ', cls , ' avg_precs: ', all_precs[cls])
        # print('cls: ', cls , ' sum      : ', sum)
    # print('{:-^140}'.format('')) 
    
    ## print average of each IoU threshold    
    if len(disp_classes) > 1:
        print()
        sum /= cnt
        scores = ''.join([" {:>10.4f}".format(i)  for i in sum])
        print('{:28s} {} '.format('  average for IoU', scores ))
        print('{:-^140}'.format(''))
    
    ## print mAP accross all detections 
    # print()
    # print('{:-^140}'.format(''))    
    scores = ''.join([" {:10.2%}".format(ap)  for ap in all_precs[0]])
    print('  {:24s}   {}      %{:.2f} '.format( class_names[0], scores,  all_mAPs[0]))
    print('{:-^140}'.format(''))    
    
    
    plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98, hspace=0.30, wspace=0.20)   
    plt.show()  




##------------------------------------------------------------------------------------------
##  Plot mAP by Scores
##------------------------------------------------------------------------------------------
def plot_pr_curves_by_scores(class_data, class_id, class_name, scores = None, iou = None , ax = None ):
    avg_precs = {}
    iou_thrs = {}
    score_keys = []
    iou_key = np.round(iou,2)
    
    if ax is None:
        plt.figure(figsize=(10,10))
        ax = plt.gca()
    
    for idx, score_key in enumerate(sorted(class_data)):
        if  scores is not None and score_key not in  scores:
            continue        
#         print('score_key is: {:20s} iou: {:6.3f}  avg_prec: {:10.4f}'.format(score_key,  iou_key, class_data[score_key][iou_key]['avg_prec']))
        score_keys.append(score_key)
        avg_precs[score_key] = class_data[score_key][iou_key]['avg_prec']
        precisions = class_data[score_key][iou_key]['precisions']
        recalls    = class_data[score_key][iou_key]['recalls']
        label      = '{:15s}'.format(score_key)
        
        #### ax = plot_pr_curve(precisions, recalls, label= label, color=COLORS[idx*2], ax=ax)
        ax.plot(recalls, precisions, label=label,  color=COLORS[idx*2])


    ax.set_title(' Class: {:2d} - {} @IoU: {:4.2f} '.format(class_id, class_name, iou), fontsize=14)
    ax.set_xlabel('recall', fontsize= 12)
    ax.set_ylabel('precision', fontsize= 12)
    ax.tick_params(axis='both', labelsize = 10)
    ax.set_xlim([0.0,1.05])
    ax.set_ylim([0.0,1.05])
    leg = plt.legend(loc='lower right',frameon=True, fontsize = 10, markerscale = 6)
    leg.set_title('IoU Thr {:.2f}'.format(iou_key),prop={'size':11})
                  
    for xval in np.linspace(0.0, 1.0, 11):
        plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed', linewidth=2)
                         
    return avg_precs

    
##------------------------------------------------------------------------------------------
##  Plot mAPs by Scores PR Curves
##------------------------------------------------------------------------------------------
def plot_mAP_by_scores(all_data, scores = None, class_ids = None , iou = 0.5,  class_names = None, columns = 2):
    
    if class_ids is None:
        disp_classes = all_data.keys()
    else:
        disp_classes = class_ids
        
    if scores is None:
        disp_scores  = [ 'mrcnn_score_orig' , 'mrcnn_score_norm', 'mrcnn_score_0', 'mrcnn_score_1', 'mrcnn_score_2', 'fcn_score_0', 'fcn_score_1', 'fcn_score_2']
    else:
        disp_scores   = scores
        
    all_precs = {}
    all_mAPs  = {}
    
    num_disp_classes = len(disp_classes)
    columns = min(columns, num_disp_classes)
    rows    = math.ceil(num_disp_classes/columns)
    print('col/rows: ', columns, rows)
    fig = plt.figure(figsize=(8 *columns,6* rows))


    for idx, class_id in enumerate(disp_classes):
        row = idx // columns
        col = idx  % columns
        subplot = (row * columns) + col +1
        ax= fig.add_subplot(rows, columns, subplot)
        
        class_precs = plot_pr_curves_by_scores(all_data[class_id], class_id, class_names[class_id], scores = disp_scores, iou = iou, ax = ax)    
        all_precs[class_id] = class_precs
        # ax.autoscale_view()
        
    ## Print Summary 
    ttl = ' AP @ IoU {:.2f} Thresholds for Computed Scores '.format(iou)
    ttl_scores = ''.join([" {:>17s}".format(scr)  for scr in disp_scores])
    print()
    print('{:^150}'.format(ttl))
    print()
    print('{:-^150}'.format('  scores  '))
    print('{:2s} - {:17s} {}'.format('Id','ClassName',ttl_scores))
    print('{:-^150}'.format(''))
    for cls in disp_classes:
        if cls == 0:
            continue
        scores = ''.join([" {:>17.4f}".format(all_precs[cls][scr])  for scr in disp_scores])
        print('{:2d} - {:17s} {} '.format(cls , class_names[cls], scores ))

    ## print average of each score
    if len(disp_classes) > 1:
        for scr in disp_scores:
            all_mAPs[scr] = np.mean([float('{:6.4f}'.format(all_precs[cls][scr])) for cls in all_precs if cls != 0])

        #         print('scr', scr, 'map:', mAP[scr], np.mean(mAP[scr]))
        # print('{:-^170}'.format(''))  
        print()
        scores = ''.join([" {:>17.4f}".format(all_mAPs[scr])  for scr in disp_scores])
        print('{:22s} {} '.format(' average for score.', scores ))
        print('{:-^150}'.format(''))
    
    ## print mAP calculated across all detections
    if 0 in all_precs:
        scores = ''.join([" {:>17.2%}".format(all_precs[0][scr]) for scr in disp_scores])
        print('{:22s} {}'.format( class_names[0], scores))
        print('{:-^150}'.format(''))  
    
    plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98, hspace=0.35, wspace=0.15)   
    plt.show()         
    

##------------------------------------------------------------------------------------------
##  Plot mAPs vs.IOUs Bar Chart
##------------------------------------------------------------------------------------------
def plot_mAP_vs_IoUs_BarChart(all_data, scores = None, ious=None, class_ids = [0],  columns = 2):
    
    if class_ids is None:
        disp_classes = all_data.keys()
    else:
        disp_classes = class_ids
        
    if scores is None:
        disp_scores  = [ 'mrcnn_score_orig' , 'mrcnn_score_norm', 'mrcnn_score_0', 'mrcnn_score_1', 'mrcnn_score_2', 'fcn_score_0', 'fcn_score_1', 'fcn_score_2']
    else:
        disp_scores   = scores

    if ious is None :
        disp_ious = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    else:
        disp_ious = ious
        
    all_precs = {}
    all_mAPs  = {}
    all_IoUs  = {}
    score_keys = []
    
    num_disp_classes = len(disp_classes)
    columns = min(columns, num_disp_classes)
    rows    = math.ceil(num_disp_classes/columns)
    print(' Num disp classes', num_disp_classes, ' Columns: ', columns, ' Rows: ', rows)
    fig = plt.figure(figsize=(15 *columns,10* rows))
    ax = fig.gca()

    # # set width of bar
    barWidth = 0.125
    tick_list = np.arange(len(disp_ious))

    for idx, score_key in enumerate(disp_scores):
        # row = idx // columns
        # col = idx  % columns
        # subplot = (row * columns) + col +1
        # ax= fig.add_subplot(rows, columns, subplot)
        all_mAPs[score_key] = []
        all_IoUs[score_key] = []
        # print('Score key: ', score_key)
        score_keys.append(score_key)
        
        for j, iou_key in enumerate(disp_ious):
            if  scores is not None and score_key not in  scores:
                continue        
            # print('score_key is: {:20s} iou: {:6.3f}  avg_prec: {:10.4f}'.format(score_key,  iou_key, all_data[0][score_key][iou_key]['avg_prec']))
            all_mAPs[score_key].append(all_data[0][score_key][iou_key]['avg_prec'])
            all_IoUs[score_key].append(iou_key)
            # precisions = all_data[0][score_key][iou_key]['precisions']
            # recalls    = all_data[0]score_key][iou_key]['recalls']
            label      = '{:15s}'.format(score_key)
        

        r = [x + (barWidth*idx) for x in tick_list]
        # print(idx, 'r: ', r)
        # ax.plot(all_IoUs[score_key], all_mAPs[score_key], 's:', label=label,  color=COLORS[idx*2])
        ax.bar(r, all_mAPs[score_key], color=COLORS[idx*2], width=barWidth, edgecolor='white', label=label)

     

    ax.set_xlabel('IoU Threshold', fontsize= 16)
    ax.set_ylabel('AP', fontsize= 16)
    ax.tick_params(axis='both', labelsize = 15)
    ax.set_xlim([-0.15,8.8])
    ax.set_ylim([0.0,1.0])
    ax.set_title('mAP vs. IoU Thrshold for various scores', fontsize=16)
    leg = plt.legend(loc='upper right',frameon=True, fontsize = 'x-small', markerscale = 0.5)
    
    # leg.set_title('IoU Thr',prop={'size':12})
    # for xval in np.linspace(0.0, 1.0, 11):
        # plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed', linewidth=2)
    
    for yval in np.linspace(0.0, 1.0, 11):
        ax.hlines(yval, 0.0, 10, color='gray', alpha=0.4, linestyles='dashed', linewidth=0.5)
    
    # Add xticks on the middle of the group bars
    ax.set_xticks(tick_list + barWidth / 2)
    ax.set_xticklabels(disp_ious)
    ax.autoscale_view()

    # # Create legend & Show graphic
    # plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98, hspace=0.30, wspace=0.20)   
    plt.show()
        
    ## Print Summary 
    all_thrs =  ''.join([" {:10.4f}".format(thr) for thr in disp_ious])
    ttl = ' AP @ IoU Thresholds for computed scores '
    print()
    print(ttl.center(140))
    print()
    print('{:-^140}'.format('  IoU Thresholds  '))
    print('Score - {:20s} {}       mAP'.format(' ', all_thrs))
    print('-'*140)
    for scr in disp_scores:
        # print(all_mAPs[scr])
        scores = ''.join([" {:10.4f}".format(i) for i in all_mAPs[scr] ])        
        print('{:28s} {}     %{:.2f} '.format(scr, scores, 100*np.mean(all_mAPs[scr] )))
    print()
    
##------------------------------------------------------------------------------------------
##  Plot mAPs vs. Class Bar Chart
##------------------------------------------------------------------------------------------
def plot_mAP_vs_class_BarChart(all_data, scores = None, iou=0.5, class_ids = None, class_names = None):
    
    if class_ids is None:
        disp_classes = sorted(all_data.keys())
    else:
        disp_classes = sorted(class_ids)
        
    if scores is None:
        disp_scores  = [ 'mrcnn_score_orig', 'mrcnn_score_0', 'mrcnn_score_1', 'mrcnn_score_2', 'fcn_score_0', 'fcn_score_1', 'fcn_score_2']
    else:
        disp_scores   = scores

    iou_key    = iou        
    all_mAPs   = {}
    all_IoUs   = {}
    score_keys = []
    
    num_disp_ious  = 1
    margin         = 0.1
    bars_per_group = len(disp_scores)
    num_classes    = len(disp_classes)
    num_groups     = len(disp_classes)
    width          = max(15, num_groups )
    height         = 10  
    # tick_list    = np.linspace( 0.0 , width - (group_width+ group_margin+ 2*margin), num_classes)
    tick_list      = np.linspace( 0.0 , width - (2*margin), num_groups+1)[:-1]
    tick_list     += margin 
    group_spread   = tick_list[1]-tick_list[0]
    
    # # set width of bar   
    barWidth       = 0.125
    bar_width      = group_spread / (bars_per_group + 2)    
    barWidth       = min(0.4, bar_width)
    
    # print(' Num disp ious', num_disp_ious, 'classes ', num_groups, 'width: ', width,' width - (2*margin) :',  width - (2*margin))
    # print(' grp_spread: ',  group_spread, 'bar_width', barWidth, bar_width )
    # print(' tick-list: ', tick_list)

    fig = plt.figure(figsize=(width , height))
    ax = fig.gca()
    
    for idx, score_key in enumerate(disp_scores):
        
        all_mAPs[score_key] = []
        all_IoUs[score_key] = []
        score_keys.append(score_key)
        
        for j, class_key in enumerate(disp_classes):
            if  scores is not None and score_key not in  scores:
                continue        
            all_mAPs[score_key].append(all_data[class_key][score_key][iou_key]['avg_prec'])
            all_IoUs[score_key].append(iou_key)
            # print('score_key is: {:20s} class: {} iou: {:6.3f}  avg_prec: {:10.4f}'.format(score_key, class_key, iou_key, all_data[class_key][score_key][iou_key]['avg_prec']))
            # precisions = all_data[0][score_key][iou_key]['precisions']
            # recalls    = all_data[0]score_key][iou_key]['recalls']

        r = [x + (barWidth*idx) for x in tick_list]
        # print(idx, 'r: ', r)
        # print('label: ', label)
        # ax.plot(all_IoUs[score_key], all_mAPs[score_key], 's:', label=label,  color=COLORS[idx*2])
        
        label = 'cls {:15s}'.format(score_key)        
        ax.bar(r, all_mAPs[score_key], color=COLORS[idx*2], width=barWidth, edgecolor='white', label=label)

    ax.set_xlabel('Class', fontsize= 16)
    ax.set_ylabel('AP', fontsize= 16)
    ax.tick_params(axis='both', labelsize = 15)
    ax.set_xlim([0.0 - margin, width])
    ax.set_ylim([0.0,1.0])
    ax.set_title('mAP for various scores @ IoU {}'.format(iou_key), fontsize=16)
    leg = plt.legend(loc='upper left', frameon=True, fontsize = 10, markerscale = 0.5, framealpha = 1.0)    
    leg.set_title('Score',prop={'size':10})
    
    for yval in np.linspace(0.0, 1.0, 11):
        ax.hlines(yval, 0.0, width, color='black', alpha=0.5, linestyles='dashed', linewidth=0.5)
    
    # Add xticks on the middle of the group bars
    plt.xticks(rotation = 30)
    ax.set_xticks(tick_list + (group_spread/4))
    ax.set_xticklabels(['{:2d}-{}'.format(i,class_names[i]) for i in disp_classes ], size = 9)
    ax.autoscale_view()

    # # Create legend & Show graphic
    # plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98, hspace=0.30, wspace=0.20)   
    plt.show()
        
    #-------------------------------------------------------------------------------------
    # Print Summary 
    #-------------------------------------------------------------------------------------
    ttl = ' AP @ IoU {:.2f} Thresholds for Computed Scores '.format(iou_key)
    ttl_scores = ''.join([" {:>17s}".format(scr)  for scr in disp_scores])
    
    print()
    print('{:^140}'.format(ttl))
    print()
    print('{:-^140}'.format('  scores  '))
    print('{:2s} - {:17s} {}'.format('Id','ClassName',ttl_scores))
    print('{:-^140}'.format(''))
    
    for cls_idx, cls in enumerate(disp_classes):
        if cls == 0:
            continue
        # for scr in disp_scores:
            # print(cls, scr, len(all_mAPs[scr]))
        scores = ''.join([" {:>17.4f}".format(all_mAPs[scr][cls_idx])  for scr in disp_scores])
        print('{:2d} - {:17s} {} '.format(cls , class_names[cls], scores ))

    ## print average of each score
    if len(disp_classes) > 1:
        avg_mAP   = {}        
        for scr in disp_scores:
            avg_mAP[scr] = np.mean(all_mAPs[scr][1:])
#             print('scr', scr, 'map:',avg_mAP[scr])
        # print('{:-^170}'.format(''))  
        print()
        scores = ''.join([" {:>17.2%}".format(avg_mAP[scr])  for scr in disp_scores])
        print('{:22s} {} '.format(' average for score:', scores ))
        print('{:-^140}'.format(''))
    
#     ## print mAP calculated across all detections
#     scores = ''.join([" {:>17.2%}".format(all_mAPs[scr][0]) for scr in disp_scores])
#     print('{:22s} {}'.format( class_names[0], scores))
#     print('{:-^140}'.format(''))  
    return
    
####################################################################################
## Main Function 
####################################################################################
if __name__ == "__main__":

    with open('ground_truth_boxes.json') as infile:
        gt_boxes = json.load(infile)

    with open('predicted_boxes.json') as infile:
        pred_boxes = json.load(infile)

    # Runs it for one IoU threshold
    iou_thr = 0.7
    start_time = time.time()
    data = get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=iou_thr)
    end_time = time.time()
    print('Single IoU calculation took {:.4f} secs'.format(end_time - start_time))
    print('avg precision: {:.4f}'.format(data['avg_prec']))

    start_time = time.time()
    ax = None
    avg_precs = []
    iou_thrs = []
    for idx, iou_thr in enumerate(np.linspace(0.5, 0.95, 10)):
        data = get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=iou_thr)
        avg_precs.append(data['avg_prec'])
        iou_thrs.append(iou_thr)

        precisions = data['precisions']
        recalls    = data['recalls']
        ax = plot_pr_curve(
            precisions, recalls, label='{:.2f}'.format(iou_thr), color=COLORS[idx*2], ax=ax)

    # prettify for printing:
    avg_precs = [float('{:.4f}'.format(ap)) for ap in avg_precs]
    iou_thrs = [float('{:.4f}'.format(thr)) for thr in iou_thrs]
    print('map: {:.2f}'.format(100*np.mean(avg_precs)))
    print('avg precs: ', avg_precs)
    print('iou_thrs:  ', iou_thrs)
    plt.legend(loc='upper right', title='IOU Thr', frameon=True)
    for xval in np.linspace(0.0, 1.0, 11):
        plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed')
    end_time = time.time()
    print('\nPlotting and calculating mAP takes {:.4f} secs'.format(end_time - start_time))
    plt.show()
