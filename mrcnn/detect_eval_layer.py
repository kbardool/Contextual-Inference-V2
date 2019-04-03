"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

 
import numpy as np
import tensorflow as tf
import keras.backend as KB
import keras.engine as KE
from   mrcnn.utils        import apply_box_deltas_np, non_max_suppression, logt, flip_bbox, compute_2D_iou
from   mrcnn.detect_inf_layer import refine_detections
import pprint

pp = pprint.PrettyPrinter(indent=2, width=100)
np.set_printoptions(linewidth=100)


############################################################
##  Evaluate mode Detection Layer
############################################################

def add_evaluation_detections_1(gt_class_ids, gt_bboxes, config, class_pred_stats):
    '''
    Use GT bboxes and a series of False Postivies as detections to pass on to FCN
    Genertes list of detections using GT and adds FP bboxes for all GT bboxes 
    
    Scores are assigned using the AVERAGE CLASS SCORE + uniform random noise    
    
    Inputs:
    ------
        gt_class_ids: 
        gt_bboxes:  
        config
                
    Returns:
    --------
        detections      [M, (y1, x1, y2, x2, class_id, score)]  After inserting false positives 
                        M = N * 2 (N: number of GT annotations)
                        Number of FP insertions is equal to number of GT annotations
    '''
    # print('\n Add Evaluation Detections Method 1 (Evaluation Mode :{})'.format(config.EVALUATE_METHOD))
    verbose = config.VERBOSE
    verbose = False
    height, width   = config.IMAGE_SHAPE[:2]

    ##----------------------------------------------------------------------------
    ##  1.  Filter out boxes with class < 0 : background (0) and Crowds (neg classes)
    ##----------------------------------------------------------------------------
    gt_nz_idxs      = np.where(gt_class_ids > 0)[0]
    gt_nz_class_ids = gt_class_ids[gt_nz_idxs]
    gt_nz_bboxes    = gt_bboxes[gt_nz_idxs]

    ##----------------------------------------------------------------------------
    ##  2.  Generate False positives using ground truth info 
    ##----------------------------------------------------------------------------
    mod_detections = np.empty((0, 7))
    max_overlap = 0

    ## go through each class, gather the GTs for that class, and create their corrsponding FPs.

    for class_id in np.unique(gt_nz_class_ids) :
        # Pick detections of this class
        gt_class_ixs = np.where(gt_nz_class_ids == class_id)[0]
        tp_bboxes   =  gt_nz_bboxes[gt_class_ixs]

        ## Generate Flip gt_bboxes for this class 
        fp_bboxes   = flip_bbox(tp_bboxes, (height,width),   flip_x = True, flip_y = True)
        
 
        ## Get average score for the current class from predicted_classes information 
        ## Apply Average score for each class from "predictions_info"
        gt_bbox_count = gt_class_ixs.shape[0]
        score_sz = (gt_bbox_count,1)
        scale = 0.001
        
        cls_avg_score  = class_pred_stats['avg'][class_id]  ## np.average(dt_nz[dt_class_ixs,5])
        orig_tp_scores = np.full(score_sz, cls_avg_score)
        orig_fp_scores = np.full(score_sz, cls_avg_score)

        fp_noise = np.round(np.random.uniform(low = -scale, high= scale, size = score_sz) ,4)
        tp_noise = np.round(np.random.uniform(low = -scale, high= scale, size = score_sz) ,4)
        
        tp_scores  = orig_tp_scores + tp_noise
        fp_scores  = orig_fp_scores + fp_noise
#         print('Score size :', score_sz)
#         print('orginal tp_scores: \n', orig_tp_scores)
#         print('oringal fp_scores: \n', orig_fp_scores)
#         print('tp noise     : \n', tp_noise)
#         print('fp noise     : \n', fp_noise)
#         print('new tp_scores: \n', tp_scores)
#         print('new fp_scores: \n', fp_scores)
#         print('tp_bboxes', tp_bboxes.shape)
#         print(tp_bboxes)
#         print('fp_bboxes', fp_bboxes.shape)
#         print(fp_bboxes)
 
        tp_scores  = np.clip(tp_scores, 0.0, 1.0)
        fp_scores  = np.clip(fp_scores, 0.0, 1.0)            
            
            
        if verbose:
            print()
            print(' --------------------------------------------------------------------------------------')
            print(' class id : {:3d}     class avg score: {:.4f}    GT boxes : {:3d}'.format(class_id, cls_avg_score, gt_bbox_count))                
            print(' --------------------------------------------------------------------------------------')
            print('\n  TP Boxes: \t box \t\t\t orig_score \t noise \t\t new_score')
            print(' ','-'*80)
            for i, (box, orig_score, noise, new_score) in enumerate(zip(tp_bboxes, orig_tp_scores, tp_noise, tp_scores)):
                print(' ',i,' \t', box, '\t\t', orig_score, '\t', noise, '\t', new_score)
            print('\n  FP Boxes: \t box \t\t\t orig_score \t noise \t\t new_score')
            print(' ','-'*80)
            for i, (box, orig_score, noise, new_score) in enumerate(zip(fp_bboxes, orig_fp_scores, fp_noise, fp_scores)):
                print(' ',i,' \t', box, '\t\t', orig_score, '\t', noise, '\t', new_score)

        ##  Compute IoU between each GT box and its corresponding FP box. if their overlap > 0.80, reduce the
        ##  boundaries of the FP bbox by multiplying it's coordinates by 0.85, and use this modified FP box
        overlaps = compute_2D_iou(tp_bboxes, fp_bboxes)
        large_ious = np.where(overlaps > 0.80)[0]    
        
        # if there *are* ious > 0.8, modify the reduce the corresponing FP bboxes 
        if len(large_ious) > 0:
            if verbose :
                print('Large IOUS encountered!!!! :', large_ious)
                print('---------------------------------------------')
                print('class id :', class_id, 'class prediction avg:', cls_tp_score, cls_fp_score)
                print('---------------------------------------------')
                print(' pre adjustment overlaps ', overlaps.shape, ' max: ', overlaps.max() )
                print('---------------------------------------------')
                print(overlaps,'\n')            
                print('fp_bboxes before adjustment ', fp_bboxes.shape)
                print(fp_bboxes[large_ious])
            fp_bboxes[large_ious] = np.rint(fp_bboxes[large_ious].astype(np.float) * 0.85)
            overlaps = compute_2D_iou(tp_bboxes, fp_bboxes)
            if verbose :
                print('fp_bboxes after adjustment ', fp_bboxes.shape)
                print(fp_bboxes[large_ious])
                print(' post adjustment overlaps ', overlaps.shape, ' max: ', overlaps.max() )
                print(' ---------------------------------------------')
                print(overlaps,'\n')

        m_overlap = overlaps.max()
        max_overlap = m_overlap if m_overlap > max_overlap else max_overlap    

        classes   = np.expand_dims(gt_nz_class_ids[gt_class_ixs], axis = -1)
        tp_ind    = np.ones((gt_class_ixs.shape[0],1)) 

        class_tp  = np.concatenate([gt_nz_bboxes[gt_class_ixs], classes, tp_scores, tp_ind] , axis = -1)
        class_fp  = np.concatenate([fp_bboxes, classes, fp_scores, -1 * tp_ind] , axis = -1)
        mod_detections  = np.vstack([mod_detections, class_tp, class_fp])

    ##----------------------------------------------------------------------------
    ## 10.  Sort by decreasing score and Keep top DETECTION_MAX_INSTANCES detections
    ##----------------------------------------------------------------------------
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids   = np.argsort(mod_detections[:,5])[::-1][:roi_count]
    if verbose:
        print()
        print('GT boxes + fp boxes with noise, prior to sort:', mod_detections.shape)
        print('-'*75)
        print(mod_detections)    
        print()
        print('GT boxes + fp_boxes with noise - After Sort (Final):', mod_detections.shape)
        print('-'*75)
        print(mod_detections[top_ids])
        print("\n MAX OVERLAP {:.5f}".format(max_overlap))
    
    return mod_detections[top_ids], max_overlap    


def add_evaluation_detections_2(gt_class_ids, gt_bboxes, config, class_pred_stats):
    '''
    Use GT bboxes and a series of False Postivies as detections to pass on to FCN
    Generates list of detections using GT and adds FP bboxes for all GT bboxes 

    Scores are assigned using the 1st/3rd CLASS SCORE QUANTILES + uniform random noise    
    GT and FP boxes are evenly split between 1st/3rd quantiles, with the 3rd quantile 
    recieving the majority when there an odd number of GT boxes:
    
    Difference with add_evaluation_detections_3:
    
    In ODD number of bounding boxes, Q3 receives the lesser half of boxes
    ---------------------------------------------------------------------
    q3_count = gt_bbox_count // 2 
    q1_count = gt_bbox_count - q3_count    
    
    Inputs:
    ------
        gt_class_ids: 
        gt_bboxes:  
        config
                
    Returns:
    --------
        detections      [M, (y1, x1, y2, x2, class_id, score)]  After inserting false positives 
                        M = N * 2 (N: number of GT annotations)
                        Number of FP insertions is equal to number of GT annotations
    '''
    # print('\n Add Evaluation Detections Method 2 (Evaluation Mode :{})'.format(config.EVALUATE_METHOD))    
    verbose = config.VERBOSE
    height, width   = config.IMAGE_SHAPE[:2]

    ##----------------------------------------------------------------------------
    ##  1.  Filter out boxes with class < 0 : background (0) and Crowds (neg classes)
    ##----------------------------------------------------------------------------
    gt_nz_idxs      = np.where(gt_class_ids > 0)[0]
    gt_nz_class_ids = gt_class_ids[gt_nz_idxs]
    gt_nz_bboxes    = gt_bboxes[gt_nz_idxs]

    # print(' gt_nz_idxs     : ', gt_nz_idxs.shape, '\n', gt_nz_idxs)
    # print(' gt_nz_class_ids: ', gt_nz_class_ids.shape, '\n',  gt_nz_class_ids)
    # print(' gt_nz_bboxes   : ', gt_nz_bboxes.shape   , '\n', gt_nz_bboxes)

    ##----------------------------------------------------------------------------
    ##  2.  Generate False positives using ground truth info 
    ##----------------------------------------------------------------------------
    mod_detections = np.empty((0, 7))
    max_overlap = 0

    ## go through each class, gather the GTs for that class, and create their corrsponding FPs.

    for class_id in np.unique(gt_nz_class_ids) :
        # Pick detections of this class
        gt_class_ixs = np.where(gt_nz_class_ids == class_id)[0]
        tp_bboxes   =  gt_nz_bboxes[gt_class_ixs]

        ## Generate Flip gt_bboxes for this class 
        fp_bboxes   = flip_bbox(tp_bboxes, (height,width),   flip_x = True, flip_y = True)
        
 
        ## Get average score for the current class from predicted_classes information 
        ## Apply Average score for each class from "predictions_info"
        gt_bbox_count = gt_class_ixs.shape[0]
        score_sz = (gt_bbox_count,1)
        scale = 0.001
        
        q3_count = gt_bbox_count // 2 
        q1_count = gt_bbox_count - q3_count
        q1_score = class_pred_stats['pct'][class_id][0]   # 1st quantile 
        q3_score = class_pred_stats['pct'][class_id][2]   # 3rd quantile
        orig_tp_scores = np.zeros(score_sz)
        orig_fp_scores = np.zeros(score_sz)
        orig_tp_scores[:q3_count] = q3_score    
        orig_tp_scores[q3_count:] = q1_score
        orig_fp_scores[:q3_count] = q3_score    
        orig_fp_scores[q3_count:] = q1_score                        
            
        fp_noise = np.round(np.random.uniform(low = -scale, high= scale, size = score_sz) ,4)
        tp_noise = np.round(np.random.uniform(low = -scale, high= scale, size = score_sz) ,4)
        
        tp_scores  = orig_tp_scores + tp_noise
        fp_scores  = orig_fp_scores + fp_noise

        tp_scores  = np.clip(tp_scores, 0.0, 1.0)
        fp_scores  = np.clip(fp_scores, 0.0, 1.0)            
        
#         print('Score size :', score_sz)
#         print('orginal tp_scores: \n', orig_tp_scores)
#         print('oringal fp_scores: \n', orig_fp_scores)
#         print('tp noise     : \n', tp_noise)
#         print('fp noise     : \n', fp_noise)
#         print('new tp_scores: \n', tp_scores)
#         print('new fp_scores: \n', fp_scores)
#         print('tp_bboxes', tp_bboxes.shape)
#         print(tp_bboxes)
#         print('fp_bboxes', fp_bboxes.shape)
#         print(fp_bboxes)            
            
        if verbose:
            print()
            print(' --------------------------------------------------------------------------------------')
            print(' class id : {:3d}'.format(class_id))                
            print(' --------------------------------------------------------------------------------------')
            #  print(' class id : {:3d}     q3 score: {:.4f}  q1_score: {:.4f}'.format(class_id, q3_score, q1_score))                
            print('  GT boxes : {:3d}     q3 count/score: {:3d} / {:.4f}  q1_count/score: {:3d} / {:.4f} '.format(
                    gt_bbox_count, q3_count, q3_score, q1_count, q1_score))

            print('\n  TP Boxes: \t box \t\t\t orig_score \t noise \t\t new_score')
            print(' ','-'*80)
            for i, (box, orig_score, noise, new_score) in enumerate(zip(tp_bboxes, orig_tp_scores, tp_noise, tp_scores)):
                print(' ',i,' \t', box, '\t\t', orig_score, '\t', noise, '\t', new_score)
            print('\n  FP Boxes: \t box \t\t\t orig_score \t noise \t\t new_score')
            print(' ','-'*80)
            for i, (box, orig_score, noise, new_score) in enumerate(zip(fp_bboxes, orig_fp_scores, fp_noise, fp_scores)):
                print(' ',i,' \t', box, '\t\t', orig_score, '\t', noise, '\t', new_score)

        ##  Compute IoU between each GT box and its corresponding FP box. if their overlap > 0.80, reduce the
        ##  boundaries of the FP bbox by multiplying it's coordinates by 0.85, and use this modified FP box
        overlaps = compute_2D_iou(tp_bboxes, fp_bboxes)
        large_ious = np.where(overlaps > 0.80)[0]    
        
        # if there *are* ious > 0.8, modify the reduce the corresponing FP bboxes 
        if len(large_ious) > 0:
            if verbose :
                print('Large IOUS encountered!!!! :', large_ious)
                print('---------------------------------------------')
                print('class id :', class_id, 'class prediction avg:', cls_tp_score, cls_fp_score)
                print('---------------------------------------------')
                print(' pre adjustment overlaps ', overlaps.shape, ' max: ', overlaps.max() )
                print('---------------------------------------------')
                print(overlaps,'\n')            
                print('fp_bboxes before adjustment ', fp_bboxes.shape)
                print(fp_bboxes[large_ious])
            fp_bboxes[large_ious] = np.rint(fp_bboxes[large_ious].astype(np.float) * 0.85)
            overlaps = compute_2D_iou(tp_bboxes, fp_bboxes)
            if verbose :
                print('fp_bboxes after adjustment ', fp_bboxes.shape)
                print(fp_bboxes[large_ious])
                print(' post adjustment overlaps ', overlaps.shape, ' max: ', overlaps.max() )
                print(' ---------------------------------------------')
                print(overlaps,'\n')

        m_overlap = overlaps.max()
        max_overlap = m_overlap if m_overlap > max_overlap else max_overlap    

        classes   = np.expand_dims(gt_nz_class_ids[gt_class_ixs], axis = -1)
        tp_ind    = np.ones((gt_class_ixs.shape[0],1)) 

        class_tp  = np.concatenate([gt_nz_bboxes[gt_class_ixs], classes, tp_scores, tp_ind] , axis = -1)
        class_fp  = np.concatenate([fp_bboxes, classes, fp_scores, -1 * tp_ind] , axis = -1)
        mod_detections  = np.vstack([mod_detections, class_tp, class_fp])
        # print('classes :', classes.shape, '\n', classes, '\n')    
        # print('scores   :', scores.shape  , '\n', scores)
        # print('tp_ind :', tp_ind.shape, '\n', tp_ind)
        # print('fp_ind :', fp_ind.shape, '\n', fp_ind)
        # print('mod_detections:', mod_detections.shape)
        # print(mod_detections)

    ##----------------------------------------------------------------------------
    ## 10.  Sort by decreasing score and Keep top DETECTION_MAX_INSTANCES detections
    ##----------------------------------------------------------------------------
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids   = np.argsort(mod_detections[:,5])[::-1][:roi_count]
    if verbose:
        print()
        print('GT boxes + fp boxes with noise, prior to sort:', mod_detections.shape)
        print('-'*75)
        print(mod_detections)    
        print()
        print('GT boxes + fp_boxes with noise - After Sort (Final):', mod_detections.shape)
        print('-'*75)
        print(mod_detections[top_ids])
        print("\n MAX OVERLAP {:.5f}".format(max_overlap))
    
    return mod_detections[top_ids], max_overlap
    
    
def add_evaluation_detections_3(gt_class_ids, gt_bboxes, config, class_pred_stats):
    '''
    Use GT bboxes and a series of False Postivies as detections to pass on to FCN
    Generates list of detections using GT and adds FP bboxes for all GT bboxes 

    Scores are assigned using the 1st/3rd CLASS SCORE QUANTILES + uniform random noise    
    GT and FP boxes are evenly split between 1st/3rd quantiles, with the 3rd Q getting
    majority when there an odd number of GT boxes:
    
    Difference with add_evaluation_detections_2:
    
    In ODD number of bounding boxes, Q1 receives the lesser half of boxes
    ---------------------------------------------------------------------
    q1_count = gt_bbox_count // 2 
    q3_count = gt_bbox_count - q1_count    
    
    Inputs:
    ------
        gt_class_ids: 
        gt_bboxes:  
        config
                
    Returns:
    --------
        detections      [M, (y1, x1, y2, x2, class_id, score)]  After inserting false positives 
                        M = N * 2 (N: number of GT annotations)
                        Number of FP insertions is equal to number of GT annotations
    '''
    # print('\n Add Evaluation Detections Method 2 (Evaluation Mode :{})'.format(config.EVALUATE_METHOD))    
    verbose = config.VERBOSE
    height, width   = config.IMAGE_SHAPE[:2]

    ##----------------------------------------------------------------------------
    ##  1.  Filter out boxes with class < 0 : background (0) and Crowds (neg classes)
    ##----------------------------------------------------------------------------
    gt_nz_idxs      = np.where(gt_class_ids > 0)[0]
    gt_nz_class_ids = gt_class_ids[gt_nz_idxs]
    gt_nz_bboxes    = gt_bboxes[gt_nz_idxs]

    # print(' gt_nz_idxs     : ', gt_nz_idxs.shape, '\n', gt_nz_idxs)
    # print(' gt_nz_class_ids: ', gt_nz_class_ids.shape, '\n',  gt_nz_class_ids)
    # print(' gt_nz_bboxes   : ', gt_nz_bboxes.shape   , '\n', gt_nz_bboxes)

    ##----------------------------------------------------------------------------
    ##  2.  Generate False positives using ground truth info 
    ##----------------------------------------------------------------------------
    mod_detections = np.empty((0, 7))
    max_overlap = 0

    ## go through each class, gather the GTs for that class, and create their corrsponding FPs.

    for class_id in np.unique(gt_nz_class_ids) :
        # Pick detections of this class
        gt_class_ixs = np.where(gt_nz_class_ids == class_id)[0]
        tp_bboxes   =  gt_nz_bboxes[gt_class_ixs]

        ## Generate Flip gt_bboxes for this class 
        fp_bboxes   = flip_bbox(tp_bboxes, (height,width),   flip_x = True, flip_y = True)
        
 
        ## Get average score for the current class from predicted_classes information 
        ## Apply Average score for each class from "predictions_info"
        gt_bbox_count = gt_class_ixs.shape[0]
        score_sz = (gt_bbox_count,1)
        scale = 0.001
        
        q1_count = gt_bbox_count // 2 
        q3_count = gt_bbox_count - q1_count
        q1_score = class_pred_stats['pct'][class_id][0]   # 1st quantile 
        q3_score = class_pred_stats['pct'][class_id][2]   # 3rd quantile
        orig_tp_scores = np.zeros(score_sz)
        orig_fp_scores = np.zeros(score_sz)
        orig_tp_scores[:q3_count] = q3_score    
        orig_tp_scores[q3_count:] = q1_score
        orig_fp_scores[:q3_count] = q3_score    
        orig_fp_scores[q3_count:] = q1_score                        
            
        fp_noise = np.round(np.random.uniform(low = -scale, high= scale, size = score_sz) ,4)
        tp_noise = np.round(np.random.uniform(low = -scale, high= scale, size = score_sz) ,4)
        
        tp_scores  = orig_tp_scores + tp_noise
        fp_scores  = orig_fp_scores + fp_noise

        tp_scores  = np.clip(tp_scores, 0.0, 1.0)
        fp_scores  = np.clip(fp_scores, 0.0, 1.0)            
        
#         print('Score size :', score_sz)
#         print('orginal tp_scores: \n', orig_tp_scores)
#         print('oringal fp_scores: \n', orig_fp_scores)
#         print('tp noise     : \n', tp_noise)
#         print('fp noise     : \n', fp_noise)
#         print('new tp_scores: \n', tp_scores)
#         print('new fp_scores: \n', fp_scores)
#         print('tp_bboxes', tp_bboxes.shape)
#         print(tp_bboxes)
#         print('fp_bboxes', fp_bboxes.shape)
#         print(fp_bboxes)            
            
        if verbose:
            print()
            print(' --------------------------------------------------------------------------------------')
            print(' class id : {:3d}'.format(class_id))                
            print(' --------------------------------------------------------------------------------------')
            #  print(' class id : {:3d}     q3 score: {:.4f}  q1_score: {:.4f}'.format(class_id, q3_score, q1_score))                
            print('  GT boxes : {:3d}     q3 count/score: {:3d} / {:.4f}  q1_count/score: {:3d} / {:.4f} '.format(
                    gt_bbox_count, q3_count, q3_score, q1_count, q1_score))

            print('\n  TP Boxes: \t box \t\t\t orig_score \t noise \t\t new_score')
            print(' ','-'*80)
            for i, (box, orig_score, noise, new_score) in enumerate(zip(tp_bboxes, orig_tp_scores, tp_noise, tp_scores)):
                print(' ',i,' \t', box, '\t\t', orig_score, '\t', noise, '\t', new_score)
            print('\n  FP Boxes: \t box \t\t\t orig_score \t noise \t\t new_score')
            print(' ','-'*80)
            for i, (box, orig_score, noise, new_score) in enumerate(zip(fp_bboxes, orig_fp_scores, fp_noise, fp_scores)):
                print(' ',i,' \t', box, '\t\t', orig_score, '\t', noise, '\t', new_score)

        ##  Compute IoU between each GT box and its corresponding FP box. if their overlap > 0.80, reduce the
        ##  boundaries of the FP bbox by multiplying it's coordinates by 0.85, and use this modified FP box
        overlaps = compute_2D_iou(tp_bboxes, fp_bboxes)
        large_ious = np.where(overlaps > 0.80)[0]    
        
        # if there *are* ious > 0.8, modify the reduce the corresponing FP bboxes 
        if len(large_ious) > 0:
            if verbose :
                print('Large IOUS encountered!!!! :', large_ious)
                print('---------------------------------------------')
                print('class id :', class_id, 'class prediction avg:', cls_tp_score, cls_fp_score)
                print('---------------------------------------------')
                print(' pre adjustment overlaps ', overlaps.shape, ' max: ', overlaps.max() )
                print('---------------------------------------------')
                print(overlaps,'\n')            
                print('fp_bboxes before adjustment ', fp_bboxes.shape)
                print(fp_bboxes[large_ious])
            fp_bboxes[large_ious] = np.rint(fp_bboxes[large_ious].astype(np.float) * 0.85)
            overlaps = compute_2D_iou(tp_bboxes, fp_bboxes)
            if verbose :
                print('fp_bboxes after adjustment ', fp_bboxes.shape)
                print(fp_bboxes[large_ious])
                print(' post adjustment overlaps ', overlaps.shape, ' max: ', overlaps.max() )
                print(' ---------------------------------------------')
                print(overlaps,'\n')

        m_overlap = overlaps.max()
        max_overlap = m_overlap if m_overlap > max_overlap else max_overlap    

        classes   = np.expand_dims(gt_nz_class_ids[gt_class_ixs], axis = -1)
        tp_ind    = np.ones((gt_class_ixs.shape[0],1)) 

        class_tp  = np.concatenate([gt_nz_bboxes[gt_class_ixs], classes, tp_scores, tp_ind] , axis = -1)
        class_fp  = np.concatenate([fp_bboxes, classes, fp_scores, -1 * tp_ind] , axis = -1)
        mod_detections  = np.vstack([mod_detections, class_tp, class_fp])
        # print('classes :', classes.shape, '\n', classes, '\n')    
        # print('scores   :', scores.shape  , '\n', scores)
        # print('tp_ind :', tp_ind.shape, '\n', tp_ind)
        # print('fp_ind :', fp_ind.shape, '\n', fp_ind)
        # print('mod_detections:', mod_detections.shape)
        # print(mod_detections)

    ##----------------------------------------------------------------------------
    ## 10.  Sort by decreasing score and Keep top DETECTION_MAX_INSTANCES detections
    ##----------------------------------------------------------------------------
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids   = np.argsort(mod_detections[:,5])[::-1][:roi_count]
    if verbose:
        print()
        print('GT boxes + fp boxes with noise, prior to sort:', mod_detections.shape)
        print('-'*75)
        print(mod_detections)    
        print()
        print('GT boxes + fp_boxes with noise - After Sort (Final):', mod_detections.shape)
        print('-'*75)
        print(mod_detections[top_ids])
        print("\n MAX OVERLAP {:.5f}".format(max_overlap))
    
    return mod_detections[top_ids], max_overlap
    
    
        
class DetectionEvaluateLayer(KE.Layer):
    '''
    Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.
    Input:
    -------
    rpn_proposal_rois:     [batch, N, (y1, x1, y2, x2)] in normalized coordinates.
                           Proposal bboxes generated by RPN
                           Might be zero padded if there are not enough proposals.
    mrcnn_class : 
    mrcnn_bbox  :  
    input_image_meta:
    
    
    Returns:
    --------
    
    detections:         [batch, num_max_detections, 6 (y1, x1, y2, x2, class_id, class_score)] in pixels
    
    '''

    def __init__(self, config=None, class_pred_stats = None, **kwargs):
        # super(DetectionLayer, self).__init__(**kwargs)
        super().__init__(**kwargs)
        print('\n>>> Detection Layer (Evaluation Mode :{})'.format(config.EVALUATE_METHOD))
        self.config = config
        self.verbose = config.VERBOSE
        self.class_pred_stats = class_pred_stats
        if config.EVALUATE_METHOD == 1:
            self.build_evaluation_detections = add_evaluation_detections_1
        elif config.EVALUATE_METHOD == 2:
            self.build_evaluation_detections = add_evaluation_detections_2
        else:
            self.build_evaluation_detections = add_evaluation_detections_3
            
        
    def call(self, inputs):
        print('    Detection Layer : call() ', type(inputs), len(inputs))    
        # logt('rpn_proposals_roi ',  inputs[0], verbose = self.verbose)
        # logt('mrcnn_class.shape ',  inputs[1], verbose = self.verbose) 
        # logt('mrcnn_bboxes.shape',  inputs[2], verbose = self.verbose)
        # logt('input_image_meta  ',  inputs[0], verbose = self.verbose) 
        logt('input_gt_class_ids',  inputs[0], verbose = self.verbose) 
        logt('input_gt_bboxes   ',  inputs[1], verbose = self.verbose)
    
        def wrapper(gt_class_ids, gt_bboxes):
        # def wrapper(rois, mrcnn_class, mrcnn_bbox, image_meta, gt_class_ids, gt_bboxes):
            from mrcnn.utils import parse_image_meta
            mod_detections_batch = []
            
            for b in range(self.config.BATCH_SIZE):
                                
                # Run the regular detection graph, as we do in inference mode
                # 24-01-2019 : In add_evaluation_detections_1 & 2 we do not need the inference detections, 
                # So this has been commented out. 
                # _, _, window, _ =  parse_image_meta(image_meta)
                # detections      =  refine_detections(rois[b], mrcnn_class[b], mrcnn_bbox[b], window[b], self.config)
                
                ## Call routine to build the control file using GT annotations, adding false detections:
                
                # mod_detections  =  add_evaluation_detections_1(detections, image_meta[b], gt_class_ids[b], gt_bboxes[b], self.config)
                mod_detections, max_overlap  =  self.build_evaluation_detections( gt_class_ids[b], gt_bboxes[b], self.config, self.class_pred_stats)
                
                # if self.config.VERBOSE:
                    # print(' original detections (GT annotations) shape        :', gt_bboxes[b].shape)                
                    # print(' modified detections (after adding false positives):', mod_detections.shape)
                    # print(' Max Overlap: ', max_overlap)
                    # print(detections)
                    # pass
                    
                # Pad with zeros if detections < DETECTION_MAX_INSTANCES
                gap = self.config.DETECTION_MAX_INSTANCES - mod_detections.shape[0]
                assert gap >= 0
                if gap > 0:
                    mod_detections = np.pad(mod_detections, [(0, gap), (0, 0)], 'constant', constant_values=0)
                
                mod_detections_batch.append(mod_detections)

            # Stack detections and cast to float32
            # TODO: track where float64 is introduced
            mod_detections_batch = np.array(mod_detections_batch).astype(np.float32)
            num_columns = mod_detections_batch.shape[-1]

            # Reshape output
            # [batch, num_detections, (y1, x1, y2, x2, class_score, dt_ind)] in pixels
            return np.reshape(mod_detections_batch, [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, num_columns])

        # Return wrapped function
        return tf.py_func(wrapper, inputs, tf.float32, name = 'detections')

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 7)
        
        
        
    
"""
Archived on 24-01-2019

def add_evaluation_detections_2_old (gt_class_ids, gt_bboxes, config, class_pred_stats):
    '''
    Use GT bboxes and a series of False Postivies as detections to pass on to FCN
    Genertes list of detections using GT and adds FP bboxes for all GT bboxes 

    Inputs:
    ------
        gt_class_ids: 
        gt_bboxes:  
        config
                
    Returns:
    --------
        detections      [M, (y1, x1, y2, x2, class_id, score)]  After inserting false positives 
                        M = N * 2 (N: number of GT annotations)
                        Number of FP insertions is equal to number of GT annotations
    '''
    verbose = config.VERBOSE
    height, width   = config.IMAGE_SHAPE[:2]

    ##----------------------------------------------------------------------------
    ##  1.  Filter out boxes with class < 0 : background (0) and Crowds (neg classes)
    ##----------------------------------------------------------------------------
    gt_nz_idxs      = np.where(gt_class_ids > 0)[0]
    gt_nz_class_ids = gt_class_ids[gt_nz_idxs]
    gt_nz_bboxes    = gt_bboxes[gt_nz_idxs]

    # print(' gt_nz_idxs     : ', gt_nz_idxs.shape, '\n', gt_nz_idxs)
    # print(' gt_nz_class_ids: ', gt_nz_class_ids.shape, '\n',  gt_nz_class_ids)
    # print(' gt_nz_bboxes   : ', gt_nz_bboxes.shape   , '\n', gt_nz_bboxes)

    ##----------------------------------------------------------------------------
    ##  2.  Generate False positives using ground truth info 
    ##----------------------------------------------------------------------------
    mod_detections = np.empty((0, 7))
    max_overlap = 0

    ## go through each class, gather the GTs for that class, and create their corrsponding FPs.
     
    for class_id in np.unique(gt_nz_class_ids) :
        # Pick detections of this class
        gt_class_ixs = np.where(gt_nz_class_ids == class_id)[0]
        tp_bboxes   =  gt_nz_bboxes[gt_class_ixs]
        
        ## Generate Flip gt_bboxes for this class 
        fp_bboxes   = flip_bbox(tp_bboxes, (height,width),   flip_x = True, flip_y = True)
         
        ## Get average score for the current class from predicted_classes information 
        ## Apply Average score for each class from "predictions_info"
        score_sz = (gt_class_ixs.shape[0],1)
        # print('Score size :', score_sz)
        if config.EVALUATE_METHOD == 1:
            cls_fp_score = cls_tp_score = class_pred_stats['avg'][class_id] 
            tp_scores = fp_scores = np.full(score_sz, cls_tp_score)
        else:
            cls_fp_score = class_pred_stats['pct'][class_id][0]   # 1st quantile 
            cls_tp_score = class_pred_stats['pct'][class_id][2]   # 3rd quantile
            tp_scores    = np.random.choice([cls_tp_score, cls_fp_score], size= score_sz)
            fp_scores    = np.random.choice([cls_fp_score, cls_tp_score], size= score_sz)
        
        ## Add gaussian noise to the scores and clip to [0.0001, 0.9999]
        tp_scores += np.round(np.random.normal(scale = 0.0001, size = score_sz) ,4)
        fp_scores += np.round(np.random.normal(scale = 0.0001, size = score_sz) ,4)            
        tp_scores  = np.clip(tp_scores, 0.0001, 0.9999)
        fp_scores  = np.clip(fp_scores, 0.0001, 0.9999)         
        
        if verbose:
            print()
            print('---------------------------------------------')
            print('class id :', class_id, 'class prediction avg:', cls_tp_score, cls_fp_score)
            print('---------------------------------------------')
            print('gt_class_indexes : ', gt_class_ixs.shape, gt_class_ixs)
            print('tp_bboxes', tp_bboxes.shape)
            print(tp_bboxes)
            print('fp_bboxes', fp_bboxes.shape)
            print(fp_bboxes)

        ##  Compute IoU between each GT box and its corresponding FP box. if their overlap > 0.80, reduce the
        ##  boundaries of the FP bbox by multiplying it's coordinates by 0.85, and use this modified FP box
        overlaps = compute_2D_iou(tp_bboxes, fp_bboxes)

        # identify IoUs > 0.8    
        large_ious = np.where(overlaps > 0.80)[0]    
        # if there *are* ious > 0.8, modify the reduce the corresponing FP bboxes 
        if len(large_ious) > 0:
            if verbose :
                print('Large IOUS encountered!!!! :', large_ious)
                print('---------------------------------------------')
                print('class id :', class_id, 'class prediction avg:', cls_tp_score, cls_fp_score)
                print('---------------------------------------------')
                print(' pre adjustment overlaps ', overlaps.shape, ' max: ', overlaps.max() )
                print(' ---------------------------------------------')
                print(overlaps,'\n')            
                print('fp_bboxes before adjustment ', fp_bboxes.shape)
                print(fp_bboxes[large_ious])
            fp_bboxes[large_ious] = np.rint(fp_bboxes[large_ious].astype(np.float) * 0.85)
            overlaps = compute_1D_iou(tp_bboxes, fp_bboxes)
            if verbose :
                print('fp_bboxes after adjustment ', fp_bboxes.shape)
                print(fp_bboxes[large_ious])
                print(' post adjustment overlaps ', overlaps.shape, ' max: ', overlaps.max() )
                print(' ---------------------------------------------')
                print(overlaps,'\n')
            
        # keep track of largest overlap encountered
        m_overlap = overlaps.max()
        max_overlap = m_overlap if m_overlap > max_overlap else max_overlap    
        
        # build detections 
        classes   = np.expand_dims(gt_nz_class_ids[gt_class_ixs], axis = -1)
        tp_ind    = np.ones((gt_class_ixs.shape[0],1)) 
         
        class_tp  = np.concatenate([gt_nz_bboxes[gt_class_ixs], classes, tp_scores, tp_ind] , axis = -1)
        class_fp  = np.concatenate([fp_bboxes, classes, fp_scores, -1 * tp_ind] , axis = -1)
        mod_detections  = np.vstack([mod_detections, class_tp, class_fp])

        # print('classes :', classes.shape, '\n', classes, '\n')    
        # print('scores   :', scores.shape  , '\n', scores)
        # print('tp_ind :', tp_ind.shape, '\n', tp_ind)
        # print('fp_ind :', fp_ind.shape, '\n', fp_ind)
        # print('mod_detections:', mod_detections.shape)
        # print(mod_detections)
    
    ##----------------------------------------------------------------------------
    ## Sort by decreasing score and Keep top DETECTION_MAX_INSTANCES detections
    ##----------------------------------------------------------------------------
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids   = np.argsort(mod_detections[:,5])[::-1][:roi_count]
    if verbose:
        print('detections + fp boxes:', mod_detections.shape)
        print(mod_detections)    
        print('final detections: sorted tp_boxes + fp_boxes:', mod_detections.shape)
        print(mod_detections[top_ids])
        print("\n\n MAX OVERLAP", max_overlap)
    
    return mod_detections[top_ids], max_overlap

#---------------------------------------------------------------------------------------------    
#   Archived ~ 12-2018
#---------------------------------------------------------------------------------------------   
def add_evaluation_detections_old(detections, image_meta, gt_class_ids, gt_bboxes, config):
    '''
    Takes identifed detections, adds FP bboxes for all GT bboxes and adds them to the list of detections  
    False Positives are based off of Ground Truth bounding boxes

    Inputs:
    ------
        
        detections:     detections from previous layer  [N, (y1, x1, y2, x2, class_id, score, dt_indicator)] in normalized coordinates
        image_meta:
        gt_class_ids: 
        gt_bboxes:  
                
    Returns:
    --------
        detections      [M, (y1, x1, y2, x2, class_id, score)]  After inserting false positives 
                        M = N (number of original detections) + (number of FP insertions)
                        Number of FP insertions is equal to number of GT annotations
    '''

    # print(' MAX DETECTION INSTANCES: ', config.DETECTION_MAX_INSTANCES)
     
    height, width   = config.IMAGE_SHAPE[:2]

    ##----------------------------------------------------------------------------
    ##  1.  Filter out boxes with class == background (0)
    ##----------------------------------------------------------------------------
    dt_nz_idxs = np.where(detections[:,4] > 0)[0]
    gt_nz_idxs = np.where(gt_class_ids > 0)[0]

    dt_nz = detections[dt_nz_idxs]
    gt_nz_class_ids  = gt_class_ids[gt_nz_idxs]
    gt_nz_bboxes     = gt_bboxes[gt_nz_idxs]

    # print(' dt_nz_idxs     : ', dt_nz_idxs.shape, dt_nz_idxs)
    # print(' gt_nz_idxs     : ', gt_nz_idxs.shape, '\n', gt_nz_idxs)
    # print(' dt_nz          : ', dt_nz.shape     , '\n', dt_nz)
    # print(' gt_nz_class_ids: ', gt_nz_class_ids.shape, '\n',  gt_nz_class_ids)
    # print(' gt_nz_bboxes   : ', gt_nz_bboxes.shape   , '\n', gt_nz_bboxes)

    ##----------------------------------------------------------------------------
    ##  2.  Generate False positives using ground truth info 
    ##----------------------------------------------------------------------------
    ## we want to add false positivies for every GT 
    fp_additions = np.zeros((gt_nz_class_ids.shape[0],dt_nz.shape[-1]))
    # print(' fp_additions.shape: ', fp_additions.shape, ' gt_nz_class_ids.shape: ', gt_nz_class_ids.shape)

    ## go through each class, gather the GTs for that class, and create their corrsponding FPs.
    for class_id in np.unique(gt_nz_class_ids):
        # Pick detections of this class
        gt_class_ixs = np.where(gt_nz_class_ids == class_id)[0]
        dt_class_ixs = np.where(dt_nz[:,4] == class_id)[0]

        # flipped gt_bboxes for this class 
        fp_bboxes    = flip_bbox(gt_nz_bboxes[gt_class_ixs], (height,width),   flip_x = True, flip_y = True)
        dt_cls_avg_score = np.average(dt_nz[dt_class_ixs,5])
        
        #  DO WE NEED TO Apply NMS between the flipped bounding boxes and detected boudiing boxes??
        #  returns indices into the gt_class_ixs 
        #     class_keep = utils.non_max_suppression(pre_nms_rois[cla, 
        #                                            pre_nms_scores[class_ixs], config.DETECTION_NMS_THRESHOLD)
        if config.VERBOSE:
            print()
            print('class id :', class_id)
            print('---------------------------------------------')
            print('gt_class_indexes : ', gt_class_ixs.shape, gt_class_ixs)
            print('gt_bboxes')
            print(gt_nz_bboxes[gt_class_ixs])
            print('dt_class_indexes : ', dt_class_ixs.shape, dt_class_ixs)
            print('dt_bboxes')
            print(dt_nz[dt_class_ixs,:4].astype(np.int))
            print('dt_scores - avg: ', dt_cls_avg_score)
            print('dt_scores      : ', dt_nz[dt_class_ixs,5])
        
        fp_classes  = np.expand_dims(gt_nz_class_ids[gt_class_ixs], axis = -1)
        fp_scores   = np.ones((gt_class_ixs.shape[0],1)) * dt_cls_avg_score
        dt_type_ind = -np.ones((gt_class_ixs.shape[0],1)) 
        class_fp    = np.concatenate([fp_bboxes,fp_classes, fp_scores, dt_type_ind] , axis = -1)
        fp_additions[[gt_class_ixs]] = class_fp

        # if config.VERBOSE:
            # print('fp_scores   :', fp_scores.shape  , '\n', fp_scores)
            # print('dt_type_ind :', dt_type_ind.shape, '\n', dt_type_ind)
            # print(fp_bboxes.shape, fp_bboxes)
            # print(fp_classes.shape, fp_classes)
            # print('class_fp :', class_fp.shape, '\n', class_fp, '\n')
            # print('fp_additions:', fp_additions.shape, '\n', fp_additions, '\n')

    mod_detections  = np.vstack([dt_nz, fp_additions])

    ##----------------------------------------------------------------------------
    ## 3.  Sort by decreasing score and Keep top DETECTION_MAX_INSTANCES detections
    ##----------------------------------------------------------------------------

    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids   = np.argsort(mod_detections[:,5])[::-1][:roi_count]

    if config.VERBOSE:
        print(' Final fp_additons: ', fp_additions.shape)    
        # print(fp_additions) 
        print('detections + fp boxes:', mod_detections.shape)
        # print(mod_detections)
        print('final detections: sorted detections + fp boxes:', mod_detections.shape)
        # print(mod_detections[top_ids])
    
    
    return mod_detections[top_ids]
    
"""        