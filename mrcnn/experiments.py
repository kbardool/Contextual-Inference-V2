import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib          import cm
import mrcnn.score_columns as sc






def intialize_results(dim, NUM_CLASSES, X, Y):
    agg_hm_shape = (dim,dim,NUM_CLASSES)

    results = {}
    results['X']                  = X
    results['Y']                  = Y
    results['pr_agg_hm']          = np.zeros(agg_hm_shape)
    results['fcn_agg_hm']         = np.zeros(agg_hm_shape)
    results['pr_agg_hm_clipped']  = np.zeros(agg_hm_shape)
    results['fcn_agg_hm_clipped'] = np.zeros(agg_hm_shape)
#   results['fcn_hm_delta']       = np.zeros(agg_hm_shape)
    results['orig_scores']        = np.zeros((2, NUM_CLASSES, X.shape[0], X.shape[1]))
    results['fcn_scores']         = np.zeros((2, NUM_CLASSES, X.shape[0], X.shape[1]))
    results['pr_scores']          = np.zeros((2, NUM_CLASSES, X.shape[0], X.shape[1]))
    results['gt_cls_counts']      = np.zeros((NUM_CLASSES), dtype = np.int)
    results['dt_cls_counts']      = np.zeros((NUM_CLASSES), dtype = np.int)
    results['gt_ttl_img_by_inst'] = np.zeros((NUM_CLASSES, 200), dtype = np.int)  # 16 is config.MAX_SHAPES_PER_IMAGE
    results['dt_ttl_img_by_inst'] = np.zeros((NUM_CLASSES, 200), dtype = np.int)  # 16 is config.MAX_SHAPES_PER_IMAGE
   
    results['imgs_one_gt'] = 0
    results['imgs_one_dt'] = 0
    results['sav_pr_min']  = 0
    results['sav_pr_max']  = 0
    results['sav_fcn_min'] = 0
    results['sav_fcn_max'] = 0
    return results


def save_results(results, save_path, save_file):
    print(' Save to output file: ',save_path, save_file)
    np.savez_compressed(os.path.join(save_path, save_file),     
                    imgs_one_gt        = results['imgs_one_gt'], 
                    imgs_one_dt        = results['imgs_one_dt'],  
                    gt_cls_counts      = results['gt_cls_counts'], 
                    dt_cls_counts      = results['dt_cls_counts'],
                    gt_ttl_img_by_inst = results['gt_ttl_img_by_inst'],
                    dt_ttl_img_by_inst = results['dt_ttl_img_by_inst'],
                    sav_pr_min         = results['sav_pr_min']  ,
                    sav_pr_max         = results['sav_pr_max']  , 
                    sav_fcn_min        = results['sav_fcn_min'] , 
                    sav_fcn_max        = results['sav_fcn_max'] ,                     
                    pr_agg_hm          = results['pr_agg_hm']  ,
                    fcn_agg_hm         = results['fcn_agg_hm']  ,
                    pr_agg_hm_clipped  = results['pr_agg_hm_clipped'],
                    fcn_agg_hm_clipped = results['fcn_agg_hm_clipped'],
                    orig_scores        = results['orig_scores'],
                    fcn_scores         = results['fcn_scores'],
                    pr_scores          = results['pr_scores'])     
    return 0


def display_results(results):
    np_format = {}
    np_format['float']  = lambda x: "%10.4f" % x
    np_format['int']    = lambda x: "%6d" % x
    np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =10000, formatter = np_format)
    
    print()
    print(' Final Results:')
    print('---------------')
    print('  pr_hm MIN: {:12.5f}      MAX: {:12.5f}'.format(results['sav_pr_min'], results['sav_pr_max']))
    print(' fcn_hm MIN: {:12.5f}      MAX: {:12.5f}'.format(results['sav_fcn_min'], results['sav_fcn_max']))
    print()
    print(' images with one gt object:', results['imgs_one_gt'], '  one detection from MR-CNN:', results['imgs_one_dt'])
    print(' Ground Truth by class    :', results['gt_cls_counts'])
    print(' Detections by class      :', results['dt_cls_counts'])
    print()
    print(' Original Heatmap:')
    print(' ----------------')
    print('    pr_hm mins: ',np.min(results['pr_agg_hm'] , axis=(0,1)))
    print('          maxs: ',np.max(results['pr_agg_hm'] , axis=(0,1)))
    print()
    print('   fcn_hm mins: ',np.min(results['fcn_agg_hm'], axis=(0,1)))
    print('          maxs: ',np.max(results['fcn_agg_hm'], axis=(0,1)))
    print()
    print('   pr_hm sums : ',np.sum(results['pr_agg_hm'] , axis=(0,1)))
    print('   fcn_hm sums: ',np.sum(results['fcn_agg_hm'], axis=(0,1)))
    print()
    print(' Clipped Heatmap:')
    print(' ----------------')
    print('   pr_hm mins : ',np.min(results['pr_agg_hm_clipped'], axis=(0,1)))
    print('         maxs : ',np.max(results['pr_agg_hm_clipped'], axis=(0,1)))
    print('  fcn_hm mins : ',np.min(results['fcn_agg_hm_clipped'], axis=(0,1)))
    print('         maxs : ',np.max(results['fcn_agg_hm_clipped'], axis=(0,1)))
    print('   pr_hm sums : ',np.sum(results['pr_agg_hm_clipped'], axis=(0,1)))
    print('  fcn_hm sums : ',np.sum(results['fcn_agg_hm_clipped'], axis=(0,1)))

    print('\n gt_ttl_img_by_inst', results['gt_ttl_img_by_inst'].shape)
    print(' ------------------')
    print(results['gt_ttl_img_by_inst'][:,:16])
    print('\n dt_ttl_img_by_inst : ', results['dt_ttl_img_by_inst'].shape)
    print(' ------------------')
    print(results['dt_ttl_img_by_inst'][:,:16])
    return(0)

def aggregate_results(fcn_results, results, cx = None, cy= None):

    r = fcn_results[0]
    NUM_CLASSES = results['gt_cls_counts'].shape[0]

    gt_inst_per_class = np.bincount(np.abs(r['gt_class_ids']), minlength = NUM_CLASSES)
    dt_inst_per_class = np.bincount(r['class_ids']           , minlength = NUM_CLASSES)

    results['gt_cls_counts'] += gt_inst_per_class
    results['dt_cls_counts'] += dt_inst_per_class

    for i in range(NUM_CLASSES):
        results['gt_ttl_img_by_inst'][i,gt_inst_per_class[i]] += 1
        results['dt_ttl_img_by_inst'][i,dt_inst_per_class[i]] += 1


    if len(r['class_ids']) == 1: 
        results['imgs_one_dt'] += 1
    if len(r['gt_class_ids']) == 1:
        results['imgs_one_gt'] += 1

    fcn_hm_max   = np.max(r['fcn_hm'])
    fcn_hm_min   = np.min(r['fcn_hm'])
    pr_hm_max    = np.max(r['pr_hm'])
    pr_hm_min    = np.min(r['pr_hm'])
    
    if fcn_hm_max > results['sav_fcn_max'] :
        results['sav_fcn_max'] = fcn_hm_max

    if fcn_hm_min < results['sav_fcn_min'] :
        results['sav_fcn_min'] = fcn_hm_min

    if pr_hm_max > results['sav_pr_max']:
        results['sav_pr_max'] = pr_hm_max

    if pr_hm_min > results['sav_pr_min']:
        results['sav_pr_min'] = pr_hm_min
    
    results['pr_agg_hm']          += r['pr_hm']
    results['fcn_agg_hm']         += r['fcn_hm']
    results['pr_agg_hm_clipped']  += np.clip(r['pr_hm'], 0.0, 1.0)
    results['fcn_agg_hm_clipped'] += np.clip(r['fcn_hm'], 0.0, 1.0)
    
#     if cx is not None:
#         print(r['pr_scores'])
#         print(r['fcn_scores'])
    for row in r['fcn_scores'][::-1]:
        cls = int(row[sc.CLASS_COLUMN])
#             print(' Class: {:d}   fcn_score_1 {}  fcn_score_2 {}'.format(
#                 cls, row[SCORE_1_COLUMN], row[SCORE_2_COLUMN]))
        results['fcn_scores'][0, cls,cx,cy] = row[sc.SCORE_1_COLUMN]
        results['fcn_scores'][1, cls,cx,cy] = row[sc.SCORE_2_COLUMN]

    for row in r['pr_scores'][::-1]:
        cls = int(row[sc.CLASS_COLUMN])
#             print(' Class: {:d}  mrcnn_scr_1 {} mrcn_score_2 {}'.format(
#                 cls, row[SCORE_1_COLUMN], row[SCORE_2_COLUMN]))
        results['pr_scores'][0, cls,cx,cy] = row[sc.SCORE_1_COLUMN]
        results['pr_scores'][1, cls,cx,cy] = row[sc.SCORE_2_COLUMN]
    
    for row in r['pr_scores'][::-1]:
        cls = int(row[sc.CLASS_COLUMN])
#             print(' Class: {:d}  mrcnn_scr_1 {} mrcn_score_2 {}'.format(
#                 cls, row[SCORE_1_COLUMN], row[SCORE_2_COLUMN]))
        results['orig_scores'][0, cls,cx,cy] = row[sc.ORIG_SCORE_COLUMN]
        
    
    return results
    
    
    
def move_across_x_axis(IMAGE_ID, CLASS_ID, COLOR, TYPE = None, verbose = False, STEP_SIZE = 3):
    X_FROM  = 3
    X_TO    = 127

    axis_position = {}
    fcn_scores_1 = {}
    fcn_scores_2 = {}
    pr_scores = {}
    failed_predicts = 0 
    xy_movement = np.arange(X_FROM,X_TO,STEP_SIZE)
    

    CLASS_NAME = class_names[CLASS_ID]        
    HEIGHT = CY[TYPE][CLASS_NAME]
    if len(dataset_test.image_info[IMAGE_ID]['shapes']) > 0:
        dataset_test.image_info[IMAGE_ID]['shapes'] = []
#         print(' SHAPES IN IMAGE: ', IMAGE_ID , ' DELETED')
    
    new_obj = (CLASS_NAME, COLOR, (64, 64 , sizes[CLASS_NAME][0][0], sizes[CLASS_NAME][0][1]))
    dataset_test.image_info[IMAGE_ID]['shapes'].append(new_obj)
#     vis.display_image_gt(dataset_test, dataset_test.config, IMAGE_ID, size=6, verbose = False)          

    print(' Image Id   : ', IMAGE_ID, ' Class : ', CLASS_ID, ' - ', CLASS_NAME , ' Type: ' , TYPE, ' Height: ', HEIGHT)
    print(' Move from X: ', X_FROM, ' to: ', X_TO, ' step: ', STEP_SIZE)
    print(' Sizes      : ', sizes[CLASS_NAME])

    for i, (sx,sy) in enumerate(sizes[CLASS_NAME]):
        print(i, sx, sy)
        fcn_scores_1[i]  = []
        fcn_scores_2[i]  = []
        pr_scores[i]     = []
        axis_position[i] = []

        for cx in xy_movement:        
            new_obj = (CLASS_NAME, COLOR, (cx, HEIGHT , sx, sy))
            dataset_test.image_info[IMAGE_ID]['shapes'][0]= new_obj

            try:
                r = run_fcn_detection(fcn_model, mrcnn_model, dataset_test, IMAGE_ID, verbose = False)  
            except Exception as e :
                failed_predicts += 1
                print('    failure on mrcnn predict - image id: {} at cx:{} cy:{}    total failures: {}'.format(IMAGE_ID, cx, HEIGHT, failed_predicts))      
                continue

            if verbose & (cx % 30 == 0) :  
                vis.display_image_gt(dataset_test, dataset_test.config, IMAGE_ID, size=6, verbose = False)  
            found = False
            for idx, (pr_row, fcn_row) in enumerate(  zip(r[0]['pr_scores'][::-1],r[0]['fcn_scores'][::-1])): 
                pr_CLASS_ID = int(pr_row[sc.CLASS_COLUMN])
                if pr_CLASS_ID == CLASS_ID :
                    found = True
                    axis_position[i].append(cx)        
                    fcn_scores_1[i].append(round(fcn_row[sc.SCORE_1_COLUMN],4))  
                    fcn_scores_2[i].append(round(fcn_row[sc.SCORE_2_COLUMN],4))  
                    pr_scores[i].append(round(pr_row[sc.SCORE_1_COLUMN],4))

            if not found:
                axis_position[i].append(cx)        
                fcn_scores_1[i].append(0.0)  
                fcn_scores_2[i].append(0.0)  
                pr_scores[i].append(0.0)
                
    return (axis_position,fcn_scores_1, fcn_scores_2, pr_scores)    
    
    
    


def move_across_y_axis(IMAGE_ID, CLASS_ID, COLOR, TYPE = None, verbose = False, X_POS = None, STEP_SIZE = 3, SIZES = None, class_names = None):
    Y_FROM  = 3
    Y_TO    = 127

    axis_position = {}
    fcn_scores_1 = {}
    fcn_scores_2 = {}
    mrcnn_scores = {} 
    pr_scores = {}
    failed_predicts = 0 
    xy_movement = np.arange(Y_FROM,Y_TO,STEP_SIZE)
    

    CLASS_NAME = class_names[CLASS_ID]        
    
    if X_POS is None:
        X_POS = CY[TYPE][CLASS_NAME]
    
    if len(dataset_test.image_info[IMAGE_ID]['shapes']) > 0:
        dataset_test.image_info[IMAGE_ID]['shapes'] = []
#         print(' SHAPES IN IMAGE: ', IMAGE_ID , ' DELETED')
    
    new_obj = (CLASS_NAME, COLOR, (64, 64 , sizes[CLASS_NAME][0][0], sizes[CLASS_NAME][0][1]))
    dataset_test.image_info[IMAGE_ID]['shapes'].append(new_obj)
#     vis.display_image_gt(dataset_test, dataset_test.config, IMAGE_ID, size=6, verbose = False)          

    print(' Image Id   : ', IMAGE_ID, ' Class : ', CLASS_ID, ' - ', CLASS_NAME , ' Type: ' , TYPE, ' Height: ', HEIGHT)
    print(' Move from Y: ', Y_FROM, ' to: ', Y_TO, ' step: ', STEP_SIZE)
    print(' Sizes      : ', sizes[CLASS_NAME])

    for i, (sx,sy) in enumerate(SIZES):
        print(i, sx, sy)
        fcn_scores_1[i]  = []
        fcn_scores_2[i]  = []
        mrcnn_score[i]   = []
        pr_scores[i]     = []
        axis_position[i] = []

        for cy in xy_movement:        
            new_obj = (CLASS_NAME, COLOR, (X_POS, cy , sx, sy))
            dataset_test.image_info[IMAGE_ID]['shapes'][0]= new_obj

            try:
                r = run_fcn_detection(fcn_model, mrcnn_model, dataset_test, IMAGE_ID, verbose = False)  
            except Exception as e :
                failed_predicts += 1
                print('    failure on mrcnn predict - image id: {} at cx:{} cy:{}    total failures: {}'.format(IMAGE_ID, X_POS, cy, failed_predicts))      
                continue

            if verbose & (cx % 30 == 0) :  
                vis.display_image_gt(dataset_test, dataset_test.config, IMAGE_ID, size=6, verbose = False)  
            found = False
            for idx, (pr_row, fcn_row) in enumerate(  zip(r[0]['pr_scores'][::-1],r[0]['fcn_scores'][::-1])): 
                pr_CLASS_ID = int(pr_row[sc.CLASS_COLUMN])
                if pr_CLASS_ID == CLASS_ID :
                    found = True
                    axis_position[i].append(cx)        
                    fcn_scores_1[i].append(round(fcn_row[sc.SCORE_1_COLUMN],4))  
                    fcn_scores_2[i].append(round(fcn_row[sc.SCORE_2_COLUMN],4))  
                    mrcnn_scores[i].append(round(fcn_row[sc.ORIG_SCORE_COLUMN],4))  
                    pr_scores[i].append(round(pr_row[sc.SCORE_1_COLUMN],4))

            if not found:
                axis_position[i].append(cx)        
                fcn_scores_1[i].append(0.0)  
                fcn_scores_2[i].append(0.0)  
                mrcnn_scores[i].append(0.0)
                pr_scores[i].append(0.0)
                
    return (axis_position,fcn_scores_1, fcn_scores_2, pr_scores, mrcnn_scores)