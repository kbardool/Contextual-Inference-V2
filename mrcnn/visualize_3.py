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
from   mrcnn.visualize  import display_image, random_colors, get_ax
import mrcnn.utils as utils

MEAN_PIXEL = [123.7, 116.8, 103.9]
BLUE     = '#1f77b4'
LBLUE    = '#aec7e8'
ORANGE   = '#ff7f0e'
LORANGE  = '#ffbb78'
GREEN    = '#2ca02c'
LGREEN   = '#98df8a'
RED      = '#d62728'
LRED     = '#ff9896'
PURPLE   = '#9467bd'
LPURPLE  = '#c5b0d5'
BROWN    = '#8c564b'
LBROWN   = '#c49c94'
PINK     = '#e377c2'
LPINK    = '#f7b6d2'
GRAY     = '#7f7f7f'
LGRAY    = '#c7c7c7'
GOLD     = '#bcbd22'
LGOLD    = '#dbdb8d'
AQUA     = '#17becf'
LAQUA    = '#9edae5'

SCORE_COLORS = {  'mrcnn_score_orig':  BLUE
                , 'mrcnn_score_0'   :  LORANGE
                , 'mrcnn_score_1'   :  LRED
                , 'mrcnn_score_2'   :  LGREEN
                
                , 'fcn_score_0'     :  ORANGE 
                , 'fcn_score_1'     :  RED
                , 'fcn_score_2'     :  GREEN
                , 'fcn_score_1_norm':  BROWN
                , 'fcn_score_2_norm':  PINK
               }
                 
COLORS      = [ AQUA , RED,   GREEN  , PURPLE , BLUE , ORANGE ,  GOLD  ,  PINK, BROWN ]
LT_COLORS   = [ LAQUA, LRED,  LGREEN , LPURPLE, LBLUE, LORANGE,  LGOLD , LPINK, LBROWN ]





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


##----------------------------------------------------------------------
## inference_heatmaps_display()
##----------------------------------------------------------------------     
# def inference_heatmaps_display( input, image_id = 0 , hm = 'fcn_hm' ,  
                                # heatmaps = None, 
                                # class_ids = None, 
                                # class_names = None,
                                # size = (8,8), 
                                # columns = 3, 
                                # config = None, 
                                # scaling = 'clip',
                                # cbar = True) :
##----------------------------------------------------------------------
## plot 2d heatmap for one image with bboxes
##----------------------------------------------------------------------        
def plot_2d_heatmaps( input, image_id = 0, hm = 'fcn_hm', 
                            class_ids = None,  
                            class_names=None, 
                            size = (9,9),
                            columns = None,  
                            num_bboxes = 999,
                            title = '2d heatmap w/ bboxes', 
                            scale = 1, 
                            scaling = 'none'):

    '''
    Z:             Gaussian heatmap [ BatchSize, height, width, Num_classes]
    boxes:         [BatchSz, NumClasses, NumBBoxes, {y2,x2, y1,x1,...}] or
                   [NumClasses, NumBBoxes, {y2,x2, y1,x1,...}] 
    image_id :    index to image 
    class_ids :    Lists of class ids to display
    '''
    scaling    = scaling.lower()
    print(" Scaling options are 'all', 'class'/'each', 'clip', or 'none' ")
    assert hm in ['fcn_hm', 'fcn_sm', 'pr_hm'], "hm must be 'fcn_hm', 'fcn_sm', or 'pr_hm'"

    results    = input[image_id]
    
    image      = results['image']
    image_meta = results['image_meta']

    if hm in ['fcn_hm', 'fcn_sm']:
        boxes =  results['fcn_scores_by_class'] 
    else:
        boxes =  results['pr_scores_by_class'] 
    
    Z = results[hm]
    if hm == 'fcn_hm':
        # Z1    =  results['fcn_hm']
        title = 'Image: {:2d} - FCN Heatmaps '.format(image_id)         
    elif hm == 'fcn_sm':
        # Z1    =  results['fcn_sm']
        title = 'Image: {:2d} - FCN Softmax '.format(image_id)
    else :
        # Z1    =  results['pr_hm']
        title = 'Image: {:2d} - MRCNN Heatmaps '.format(image_id)
    print(" Shape of Z: ", Z.shape, " boxes: ", boxes.shape)

    # if Z.ndim == 4:
        # Z = Z[image_id]
        
    # if boxes.ndim == 4:
        # boxes = boxes[image_id]

    if class_ids is None :
        print(' Image Id: ', image_id , ' Display all classes...')
        num_classes = Z.shape[-1]
        class_ids   = np.arange(num_classes)
    else:
        print(' Image Id: ', image_id , ' Display classes:', class_ids)
        num_classes = len(class_ids)
    class_ids.sort()
    
    if columns is None:
        columns = num_classes

    
    rows   = math.ceil(num_classes/columns)
    print(' rows  ',rows, ' columns :', columns, 'boxes.shape : ',boxes.shape)
    # Z = Z[image_id]
    # boxes = boxes[image_id]
    width  = size[0] * columns
    height = size[1] * rows
    fig = plt.figure(figsize=(width, height))
    
    image_height = Z.shape[0]
    image_width  = Z.shape[1]
    
    # print(image_height, image_width)
    Y = np.arange(0, image_height, 1)
    X = np.arange(0, image_width, 1)
    X, Y = np.meshgrid(X, Y)        

    if num_bboxes == 999 :
        num_bboxes  = boxes.shape[1]  
    
    if num_bboxes > 0:
        x1    = boxes[:,:,1] // scale
        x2    = boxes[:,:,3] // scale
        y1    = boxes[:,:,0] // scale
        y2    = boxes[:,:,2] // scale
        box_w = x2 - x1   # x2 - x1
        box_h = y2 - y1 
        # cx    = (x1 + ( box_w / 2.0)).astype(int)
        # cy    = (y1 + ( box_h / 2.0)).astype(int)
        # print('x1, x2...shapes:', x1.shape, x2.shape, y1.shape, y2.shape, box_h.shape, box_w.shape)

    min_z_all = np.amin(Z)
    max_z_all = np.amax(Z)
    min_z_cls = np.amin(Z, axis = (0,1), keepdims = True)
    max_z_cls = np.amax(Z, axis = (0,1), keepdims = True)
    avg_z_cls = np.mean(Z, axis = (0,1), keepdims = True)
    sum_z_cls = np.sum(Z, axis = (0,1), keepdims = True)
    print(' min_z_all shape:', min_z_all.shape,' min_z_all:', min_z_all,' max_z_all:', max_z_all.shape,'max_z_all:', max_z_all)
    print(' min_z_cls shape:', min_z_cls.shape,' max_z_cls shape:', max_z_cls.shape)
    
    if  scaling == 'clip':
        print(' SCALING == clip to [-1, +1])')    
        title += ' - Clip output to [-1, +1]'
        YY = np.clip(Z, -1.0, 1.0) 
    elif scaling == 'all':
        print(' SCALING == all')
        title += ' - NORMALIZED to 1 Over all classes'
        YY = (Z - min_z_all)/(max_z_all - min_z_all + 1.0e-9)
    elif scaling in ['class', 'each']:
        print(' SCALING == class')
        title += ' - NORMALIZED to 1 in each class'
        YY = (Z - min_z_cls)/(max_z_cls - min_z_cls + 1.0e-9)            
    elif scaling == 'none':
        print(' SCALING == none ')
        title += ' - No Normalization'
        YY = Z
    else: 
        print(" ERROR - scaling must be 'all', 'class'/'each', 'clip', or 'none' ")
        return
                

    colors = random_colors(num_classes)
    style = "dotted"
    alpha = 1
    color = colors[0]
    # color = (0.5, 0.0, 1.0)
    color = 'xkcd:neon green'
    color = 'black'

    for idx, cls in enumerate(class_ids):
        # color = colors[idx]
        row = idx // columns
        col = idx  % columns
        subplot = (row * columns) + col +1
        # print('idx ', idx,  ' class:', cls, 'row:', row,'col:', col, 'subplot: ', subplot, 'cls:', cls)
        if class_names is None:
            ttl = 'Cls: {:2d} '.format(cls)
        else:
            ttl = 'Cls: {:2d}/{:s}'.format(cls, class_names[cls])
        
        # ttl += '- min: {:5.4f}  max: {:5.4f} avg: {:5.4f} sum: {:6.3f}'.format(min_z_cls[0,0,cls], max_z_cls[0,0,cls],
                                                                            # avg_z_cls[0,0,cls], sum_z_cls[0,0,cls])
        
        ttl += '- min: {:5.4f}  max: {:5.4f} avg: {:5.4f} '.format(min_z_cls[0,0,cls], max_z_cls[0,0,cls], avg_z_cls[0,0,cls])
        ax = fig.add_subplot(rows, columns, subplot)
        ax.set_title(ttl, fontsize=15)
        ax.tick_params(axis='both', labelsize = 5)
        ax.set_ylim(0, image_height)
        ax.set_xlim(0, image_width )
        ax.set_xlabel(' X axis', fontsize=8)
        ax.set_ylabel(' Y axis', fontsize=8)
        ax.invert_yaxis()
        
        if scaling == 'clip':
            vmin = np.amin(YY[:,:,cls])
            vmax = np.amax(YY[:,:,cls])
        elif scaling in ['all','class']:            
            vmin = 0
            vmax = 1
        else:
            vmin = min_z_cls[0,0,cls]
            vmax = max_z_cls[0,0,cls]
            # vmin = np.amin(YY[:,:,cls])
            # vmax = np.amax(YY[:,:,cls])
            
        surf = plt.matshow(YY[:,:, cls], fignum = 0, cmap = cm.jet,vmin = vmin, vmax = vmax )
        
        for bbox in range(num_bboxes):
            if boxes[cls, bbox, 7 ] == 0:
                break
            # print(' num boxes: ', num_bboxes, bbox, cls, boxes[cls,bbox])
            # print(' ttl : ',  ttl,x1[cls,bbox], y1[cls,bbox],x2[cls,bbox],y2[cls,bbox])
            p = patches.Rectangle( (x1[cls,bbox],y1[cls,bbox]), box_w[cls,bbox], box_h[cls,bbox], 
                                   linewidth=1.5, alpha=alpha, linestyle=style, edgecolor=color, facecolor='none')
            ax.add_patch(p)
            
        # plt.subplots_adjust(top=0.93, bottom=0.02, left=0.02, right=0.98, hspace=0.15, wspace=0.15)      
        fig.colorbar(surf, shrink=0.8, aspect=30, fraction=0.05)
        
        
    # plt.tight_layout()
    fig.tight_layout(rect=[0, 0.0, 1, 1.0])
    # fig.suptitle(title, fontsize = 15, ha ='center' )
    plt.show()
    
    return fig 

##----------------------------------------------------------------------
## plot 2d heatmap for one image with bboxes
##----------------------------------------------------------------------        
def plot_2d_heatmaps_poster( input, image_id = 0, hm = 'fcn_hm', 
                            class_ids = None,  
                            class_names=None, 
                            size = (9,9),
                            columns = None,  
                            num_bboxes = 999,
                            title = '2d heatmap w/ bboxes', 
                            config = None,
                            scale = 1, 
                            scaling = 'none',
                            fontsize = 24, overlay_image = True):

    '''
    Z:             Gaussian heatmap [ BatchSize, height, width, Num_classes]
    boxes:         [BatchSz, NumClasses, NumBBoxes, {y2,x2, y1,x1,...}] or
                   [NumClasses, NumBBoxes, {y2,x2, y1,x1,...}] 
    image_id :    index to image 
    class_ids :    Lists of class ids to display
    '''
    
    scaling    = scaling.lower()
    print(" Scaling options are 'all', 'class'/'each', 'clip', or 'none' ")
    assert hm in ['fcn_hm', 'fcn_sm', 'pr_hm'], "hm must be 'fcn_hm', 'fcn_sm', or 'pr_hm'"

    results    = input[image_id]
    
    image      = utils.unmold_image(results['molded_image'], config)
    image_meta = results['image_meta']
    # print('Image shape :',image.shape)
    # display_image(image)
    ## Convert to grayscale np array   
    if overlay_image:
        print('0')
        image_bw = np.asarray(Image.fromarray(image).convert(mode='L'))
        downscaled_image_bw = utils.scale_image(image_bw, scale = 1.0/config.HEATMAP_SCALE_FACTOR)
        alpha = 0.6
        grid  = False
    else:
        alpha = 1.0 
        grid = True
    
    # print(' image_bw shape: ', image_bw.shape, image_bw.dtype)
    # print(' downscaled_image_bw shape: ', downscaled_image_bw.shape, downscaled_image_bw.dtype)
    
    if hm in ['fcn_hm', 'fcn_sm']:
        boxes =  results['fcn_scores_by_class'] 
    else:
        boxes =  results['pr_scores_by_class'] 
    
    Z = results[hm]
    if hm == 'fcn_hm':
        # Z1    =  results['fcn_hm']
        title = 'Image: {:2d} - FCN Heatmaps '.format(image_id)         
    elif hm == 'fcn_sm':
        # Z1    =  results['fcn_sm']
        title = 'Image: {:2d} - FCN Softmax '.format(image_id)
    else :
        # Z1    =  results['pr_hm']
        title = 'Image: {:2d} - MRCNN Heatmaps '.format(image_id)
    print(" Shape of Z: ", Z.shape, " boxes: ", boxes.shape)

    # if Z.ndim == 4:
        # Z = Z[image_id]
        
    # if boxes.ndim == 4:
        # boxes = boxes[image_id]

    if class_ids is None :
        print(' Image Id: ', image_id , ' Display all classes...')
        num_classes = Z.shape[-1]
        class_ids   = np.arange(num_classes)
    else:
        print(' Image Id: ', image_id , ' Display classes:', class_ids)
        num_classes = len(class_ids)
    class_ids.sort()
    
    if columns is None:
        columns = num_classes + 1
    
    image_height = Z.shape[0]
    image_width  = Z.shape[1]
    
    # print(image_height, image_width)
    Y = np.arange(0, image_height, 1)
    X = np.arange(0, image_width, 1)
    X, Y = np.meshgrid(X, Y)        

    if num_bboxes == 999 :
        num_bboxes  = boxes.shape[1]  
    
    if num_bboxes > 0:
        x1    = boxes[:,:,1] // scale
        x2    = boxes[:,:,3] // scale
        y1    = boxes[:,:,0] // scale
        y2    = boxes[:,:,2] // scale
        box_w = x2 - x1   # x2 - x1
        box_h = y2 - y1 
        # cx    = (x1 + ( box_w / 2.0)).astype(int)
        # cy    = (y1 + ( box_h / 2.0)).astype(int)
        # print('x1, x2...shapes:', x1.shape, x2.shape, y1.shape, y2.shape, box_h.shape, box_w.shape)

    min_z_all = np.amin(Z)
    max_z_all = np.amax(Z)
    min_z_cls = np.amin(Z, axis = (0,1), keepdims = True)
    max_z_cls = np.amax(Z, axis = (0,1), keepdims = True)
    avg_z_cls = np.mean(Z, axis = (0,1), keepdims = True)
    sum_z_cls = np.sum(Z, axis = (0,1), keepdims = True)
    print(' min_z_all shape:', min_z_all.shape,' min_z_all:', min_z_all,' max_z_all:', max_z_all.shape,'max_z_all:', max_z_all)
    print(' min_z_cls shape:', min_z_cls.shape,' max_z_cls shape:', max_z_cls.shape)
    
    if  scaling == 'clip':
        print(' SCALING == clip to [-1, +1])')    
        title += ' - Clip output to [-1, +1]'
        YY = np.clip(Z, -1.0, 1.0) 
    elif scaling == 'all':
        print(' SCALING == all')
        title += ' - NORMALIZED to 1 Over all classes'
        YY = (Z - min_z_all)/(max_z_all - min_z_all + 1.0e-9)
    elif scaling in ['class', 'each']:
        print(' SCALING == class')
        title += ' - NORMALIZED to 1 in each class'
        YY = (Z - min_z_cls)/(max_z_cls - min_z_cls + 1.0e-9)            
    elif scaling == 'none':
        print(' SCALING == none ')
        title += ' - No Normalization'
        YY = Z
    else: 
        print(" ERROR - scaling must be 'all', 'class'/'each', 'clip', or 'none' ")
        return
                
    
    colors = random_colors(num_classes)
    style = "dotted"
    color = colors[0]
    # color = (0.5, 0.0, 1.0)
    color = 'xkcd:neon green'
    color = 'black'
    
    # columns  = columns + 1  ## min(columns, num_classes)
    rows     = math.ceil(num_classes+1/columns)
    width  = size[0] * columns
    height = size[1] * rows
    fig = plt.figure(figsize=(width, height))
    print(' rows  ',rows, ' columns :', columns, 'boxes.shape : ',boxes.shape)

    subplot = 1
    ax = fig.add_subplot(rows, columns, subplot)
    
    ax.imshow(image) 
    ttl = ' Image Id: {} \n'.format(image_meta[0])
    ax.text(0.5, 1.0, ttl, fontsize=fontsize, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    # ax.set_title(ttl, fontsize=18)
    plt.tick_params(bottom = False, left=False, labelbottom =False, labelleft=False)
    
    for idx, cls in enumerate(class_ids):
        # color = colors[idx]
        # row = idx // columns
        # col = idx  % columns
        # subplot = (row * columns) + col +1
        subplot += 1
        # print('idx ', idx,  ' class:', cls, 'row:', row,'col:', col, 'subplot: ', subplot, 'cls:', cls)
        if class_names is None:
            ttl = 'Cls: {:2d} \n'.format(cls)
        else:
            ttl = 'Cls: {:2d} - {:s} \n'.format(cls, class_names[cls])
        
        # ttl += '- min: {:5.4f}  max: {:5.4f} avg: {:5.4f} sum: {:6.3f}'.format(min_z_cls[0,0,cls], max_z_cls[0,0,cls],
                                                                            # avg_z_cls[0,0,cls], sum_z_cls[0,0,cls])        
        # ttl += '- min: {:5.4f}  max: {:5.4f} avg: {:5.4f} '.format(min_z_cls[0,0,cls], max_z_cls[0,0,cls], avg_z_cls[0,0,cls])

        ax = fig.add_subplot(rows, columns, subplot)
        ax.text(0.5, 1.0, ttl, fontsize=fontsize, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        
        if scaling == 'clip':
            vmin = np.amin(YY[:,:,cls])
            vmax = np.amax(YY[:,:,cls])
        elif scaling in ['all','class']:            
            vmin = 0
            vmax = 1
        else:
            vmin = min_z_cls[0,0,cls]
            vmax = max_z_cls[0,0,cls]

        if overlay_image:
            print('1 alpha :', alpha)
            ax.imshow(downscaled_image_bw , cmap='gray')        
            
        ax.imshow(YY[:,:,cls], alpha = alpha , cmap=cm.jet, vmin = vmin, vmax = vmax )              
        # ax.grid(grid)
        
        for bbox in range(num_bboxes):
            if boxes[cls, bbox, 7 ] == 0:
                break
            # print(' num boxes: ', num_bboxes, bbox, cls, boxes[cls,bbox])
            # print(' ttl : ',  ttl,x1[cls,bbox], y1[cls,bbox],x2[cls,bbox],y2[cls,bbox])
            p = patches.Rectangle( (x1[cls,bbox],y1[cls,bbox]), box_w[cls,bbox], box_h[cls,bbox], 
                                   linewidth=1.5, alpha=alpha, linestyle=style, edgecolor=color, facecolor='none')
            ax.add_patch(p)
        # plt.subplots_adjust(top=0.93, bottom=0.02, left=0.02, right=0.98, hspace=0.15, wspace=0.15)      
        plt.tick_params(bottom = False, left=False, labelbottom =False, labelleft=False)
        # fig.colorbar(surf, shrink=0.8, aspect=30, fraction=0.05)

    for i in range(subplot+1, columns):
        print('add_subplot :', i)
        ax = fig.add_subplot(rows, columns, i)
        plt.tick_params(bottom = False, left=False, labelbottom =False, labelleft=False)
        
    # plt.tight_layout()
    # fig.tight_layout(rect=[0.2, 0.0, 1, 1.0])
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # fig.suptitle(title, fontsize = 15, ha ='center' )
    plt.show()
    
    return fig 

##----------------------------------------------------------------------
## inference_heatmaps_display()
##----------------------------------------------------------------------     
def inference_heatmaps_display( input, image_id = 0 , hm = 'fcn_hm' ,  
                                heatmaps = None, 
                                class_ids = None, 
                                class_names = None,
                                size = (8,8), 
                                columns = 3, 
                                config = None, 
                                scaling = 'clip',
                                cbar = True) :
    '''
    input
    -----
        Z       Gaussian distribution (Batch Sz, Class_sz, Img Height, Img Width)
        boxes   array of bounding boxes 
    '''
    scaling    = scaling.lower()
    hm         = hm.lower()
    print(" Scaling options are:  'all', 'class'/'each' , or  'clip' ")
    assert hm in ['fcn_hm', 'fcn_sm', 'pr_hm'], "hm must be 'fcn_hm', 'fcn_sm', or 'pr_hm'"

    results    = input[image_id]
    
    image      = results['image']
    image_meta = results['image_meta']

    if hm in ['fcn_hm', 'fcn_sm']:
        boxes =  results['fcn_scores_by_class'] 
    else:
        boxes =  results['pr_scores_by_class'] 
    
    Z1 = results[hm]
    if hm == 'fcn_hm':
        # Z1    =  results['fcn_hm']
        title = 'Image: {:2d} - FCN Heatmaps '.format(image_id)         
    elif hm == 'fcn_sm':
        # Z1    =  results['fcn_sm']
        title = 'Image: {:2d} - FCN Softmax '.format(image_id)
    else :
        # Z1    =  results['pr_hm']
        title = 'Image: {:2d} - MRCNN Heatmaps '.format(image_id)

    print(' heatmap shape: ', Z1.shape,' Bounding boxes shape: ', boxes.shape)
    scale = config.HEATMAP_SCALE_FACTOR

    if class_ids is None :
        print('Display all classes...')
        num_classes = Z1.shape[-1]
        class_ids   = np.arange(num_classes)
    else:
        print('Display classes:', class_ids)
        num_classes = len(class_ids)    

    # if class_ids is None :
        # class_ids = np.unique(results['class_ids'])
    class_ids = np.sort(class_ids)
    num_classes = len(class_ids)
    
    # print('Image shape :',image.shape)

    display_image(image)
    ## Convert to grayscale np array   
    image_bw = np.asarray(Image.fromarray(image).convert(mode='L'))
    
    columns  = min(columns, num_classes)
    rows     = math.ceil(num_classes/columns)
    width  = size[0] * columns
    height = size[1] * rows
    fig = plt.figure(figsize=(width, height))
    
    num_bboxes  = boxes.shape[2]  
    if num_bboxes > 0:
        x1    = boxes[:,:,1] 
        x2    = boxes[:,:,3] 
        y1    = boxes[:,:,0] 
        y2    = boxes[:,:,2] 
        box_w = x2 - x1    
        box_h = y2 - y1 
        # print('x1, x2...shapes:', x1.shape, x2.shape, y1.shape, y2.shape, box_h.shape, box_w.shape)    
    
    min_z1_all = np.amin(Z1)
    max_z1_all = np.amax(Z1
    )    
    min_z1_cls = np.amin(Z1, axis = (0,1), keepdims = True)
    max_z1_cls = np.amax(Z1, axis = (0,1), keepdims = True)
    avg_z1_cls = np.mean(Z1, axis = (0,1), keepdims = True)
    
    if scaling == 'all':   
        Z1 = (Z1 - min_z1_all)/(max_z1_all - min_z1_all + 1.0e-9)
        title += ' - NORMALIZED to [0, 1] across ALL classes (jointly)'
        zlim = 'one'
    elif scaling in [ 'class', 'each']:
        Z1 = (Z1 - min_z1_cls)/(max_z1_cls - min_z1_cls + 1.0e-9)
        title += ' - NORMALIZED to [0, 1] over each class '
    elif scaling == 'clip':    
        print(' SCALING == clip (clip to [-1, +1])')    
        Z1 = np.clip(Z1, -1.0, 1.0) 
        title += ' - Clip output to [-1, +1]'
    else: 
        print(" ERROR - scaling must be 'all', 'class'/'each' , or  'clip' : ", scaling)
        return        
    
    colors = random_colors(num_classes)
    style = "dotted"
    linewidth = 1.0
    alpha = 1
    color = (0.5, 0.0, 1.0)

    for idx, cls in enumerate(class_ids):
        row = idx // columns
        col = idx  % columns
        subplot = (row * columns) + col +1
        # print('idx ', idx,  ' class:', cls, 'row:', row,'col:', col, 'subplot: ', subplot, 'clor:', color)
        if class_names is None:
            ttl = 'Cls: {:2d} '.format(cls)
        else:
            ttl = 'Cls: {:2d}/{:s}'.format(cls, class_names[cls])

        ax = fig.add_subplot(rows, columns, subplot)
        ttl = ttl +'  -  min: {:6.5f}  max: {:6.5f}  avg: {:6.5f}'.format(min_z1_cls[0,0,cls], max_z1_cls[0,0,cls], avg_z1_cls[0,0,cls])
        ax.set_title(ttl, fontsize=12)
        ax.tick_params(axis='both', labelsize = 5)
        ax.tick_params(direction='out', length=6, width=1, colors='r', labelsize = 10)
        ax.set_xlabel(' X axis', fontsize=10)
        ax.set_ylabel(' Y axis', fontsize=10)

        if scaling == 'clip':
            vmin = min_z1_cls[0,0,cls]
            vmax = max_z1_cls[0,0,cls]
        else:
            vmin = 0
            vmax = 1

        unmolded_heatmap = utils.unmold_heatmap(Z1[:,:,cls],image_meta, upscale = config.HEATMAP_SCALE_FACTOR)
        # print(' unmolded_heatmap: shape:', unmolded_heatmap.shape, unmolded_heatmap.dtype, np.amin(unmolded_heatmap), np.amax(unmolded_heatmap))
        ax.imshow(image_bw , cmap=plt.cm.gray)        

        for bbox in range(num_bboxes):
            # print(ttl,x1[cls,bbox], y1[cls,bbox],x2[cls,bbox],y2[cls,bbox])
            p = patches.Rectangle( (x1[cls,bbox],y1[cls,bbox]), box_w[cls,bbox], box_h[cls,bbox], 
                               linewidth=1, alpha=alpha, linestyle=style, edgecolor=color, facecolor='none')
            ax.add_patch(p)
        surf = ax.imshow(unmolded_heatmap, alpha = 0.6, cmap=cm.jet, vmin = vmin, vmax = vmax )              
        if cbar:
            fig.colorbar(surf, shrink=0.7, aspect=30, fraction=0.05)
        plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98, hspace=0.10, wspace=0.10)                
    
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.suptitle(title, fontsize = 13) ## , ha ='center' )
    plt.show()
    
    # plt.close()
    return fig
    

##----------------------------------------------------------------------
## inference_heatmaps_display()
##----------------------------------------------------------------------     
def inference_heatmaps_display_poster( input, image_id = 0 , hm = 'fcn_hm' ,  
                                heatmaps = None, 
                                class_ids = None, 
                                class_names = None,
                                size = (8,8), 
                                columns = 3, 
                                config = None, 
                                scaling = 'clip') :
    '''
    input
    -----
        Z       Gaussian distribution (Batch Sz, Class_sz, Img Height, Img Width)
        boxes   array of bounding boxes 
    '''
    scaling    = scaling.lower()
    hm         = hm.lower()
    print(" Scaling options are:  'all', 'class'/'each' , or  'clip' ")
    assert hm in ['fcn_hm', 'fcn_sm', 'pr_hm'], "hm must be 'fcn_hm', 'fcn_sm', or 'pr_hm'"

    results    = input[image_id]
    
    image      = results['image']
    image_meta = results['image_meta']
    # print('Image shape :',image.shape)
    # display_image(image)
    ## Convert to grayscale np array   
    image_bw = np.asarray(Image.fromarray(image).convert(mode='L'))
        
    if hm in ['fcn_hm', 'fcn_sm']:
        boxes =  results['fcn_scores_by_class'] 
    else:
        boxes =  results['pr_scores_by_class'] 
    
    Z1 = results[hm]
    if hm == 'fcn_hm':
        # Z1    =  results['fcn_hm']
        title = 'Image: {:2d} - FCN Heatmaps '.format(image_id)         
    elif hm == 'fcn_sm':
        # Z1    =  results['fcn_sm']
        title = 'Image: {:2d} - FCN Softmax '.format(image_id)
    else :
        # Z1    =  results['pr_hm']
        title = 'Image: {:2d} - MRCNN Heatmaps '.format(image_id)

    print(' heatmap shape: ', Z1.shape,' Bounding boxes shape: ', boxes.shape)
    scale = config.HEATMAP_SCALE_FACTOR

    if class_ids is None :
        print('Display all classes...')
        num_classes = Z1.shape[-1]
        class_ids   = np.arange(num_classes)
    else:
        print('Display classes:', class_ids)
        num_classes = len(class_ids)    

    # if class_ids is None :
        # class_ids = np.unique(results['class_ids'])
    class_ids = np.sort(class_ids)
    num_classes = len(class_ids)
    

    
    num_bboxes  = boxes.shape[2]  
    if num_bboxes > 0:
        x1    = boxes[:,:,1] 
        x2    = boxes[:,:,3] 
        y1    = boxes[:,:,0] 
        y2    = boxes[:,:,2] 
        box_w = x2 - x1    
        box_h = y2 - y1 
        # print('x1, x2...shapes:', x1.shape, x2.shape, y1.shape, y2.shape, box_h.shape, box_w.shape)    
    
    min_z1_all = np.amin(Z1)
    max_z1_all = np.amax(Z1)    
    min_z1_cls = np.amin(Z1, axis = (0,1), keepdims = True)
    max_z1_cls = np.amax(Z1, axis = (0,1), keepdims = True)
    avg_z1_cls = np.mean(Z1, axis = (0,1), keepdims = True)
    
    if scaling == 'all':   
        Z1 = (Z1 - min_z1_all)/(max_z1_all - min_z1_all + 1.0e-9)
        title += ' - NORMALIZED to [0, 1] across ALL classes (jointly)'
        zlim = 'one'
    elif scaling in [ 'class', 'each']:
        Z1 = (Z1 - min_z1_cls)/(max_z1_cls - min_z1_cls + 1.0e-9)
        title += ' - NORMALIZED to [0, 1] over each class '
    elif scaling == 'clip':    
        print(' SCALING == clip (clip to [-1, +1])')    
        Z1 = np.clip(Z1, -1.0, 1.0) 
        title += ' - Clip output to [-1, +1]'
    else: 
        print(" ERROR - scaling must be 'all', 'class'/'each' , or  'clip' : ", scaling)
        return        
    
    colors = random_colors(num_classes)
    style = "dotted"
    linewidth = 1.0
    alpha = 1
    color = (0.5, 0.0, 1.0)

    columns  = num_classes + 1  ## min(columns, num_classes)
    rows     = math.ceil(num_classes/columns)
    width  = size[0] * columns
    height = size[1] * rows
    fig = plt.figure(figsize=(width, height))

    subplot = 1
    ax = fig.add_subplot(rows, columns, subplot)
    ax.imshow(image) 
    ttl = ' Image Id: {} '.format(image_meta[0])
    ax.set_title(ttl, fontsize=18)
    plt.tick_params(bottom = False, left=False, labelbottom =False, labelleft=False)
    
    for idx, cls in enumerate(class_ids):
        # row = idx // columns
        # col = idx  % columns
        # subplot = (row * columns) + col +1
        subplot += 1
        # print('idx ', idx,  ' class:', cls, 'row:', row,'col:', col, 'subplot: ', subplot, 'clor:', color)
        if class_names is None:
            ttl = 'Cls: {:2d} '.format(cls)
        else:
            ttl = 'Cls: {:2d}/{:s}'.format(cls, class_names[cls])

        ax = fig.add_subplot(rows, columns, subplot)
        ttl = ttl +'  -  min: {:6.5f}  max: {:6.5f}  avg: {:6.5f}'.format(min_z1_cls[0,0,cls], max_z1_cls[0,0,cls], avg_z1_cls[0,0,cls])
        ax.set_title(ttl, fontsize=15)
        # ax.tick_params(axis='both', labelsize = 5)
        # ax.tick_params(direction='out', length=6, width=1, colors='r', labelsize = 10)
        # ax.set_xlabel(' X axis', fontsize=10)
        # ax.set_ylabel(' Y axis', fontsize=10)

        if scaling == 'clip':
            vmin = min_z1_cls[0,0,cls]
            vmax = max_z1_cls[0,0,cls]
        else:
            vmin = 0
            vmax = 1

        unmolded_heatmap = utils.unmold_heatmap(Z1[:,:,cls],image_meta, upscale = config.HEATMAP_SCALE_FACTOR)
        # print(' unmolded_heatmap: shape:', unmolded_heatmap.shape, unmolded_heatmap.dtype, np.amin(unmolded_heatmap), np.amax(unmolded_heatmap))
        ax.imshow(image_bw , cmap=plt.cm.gray)        

        for bbox in range(num_bboxes):
            # print(ttl,x1[cls,bbox], y1[cls,bbox],x2[cls,bbox],y2[cls,bbox])
            p = patches.Rectangle( (x1[cls,bbox],y1[cls,bbox]), box_w[cls,bbox], box_h[cls,bbox], 
                               linewidth=1, alpha=alpha, linestyle=style, edgecolor=color, facecolor='none')
            ax.add_patch(p)
            
        surf = ax.imshow(unmolded_heatmap, alpha = 0.6, cmap=cm.jet, vmin = vmin, vmax = vmax )              
        
        plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98, hspace=0.10, wspace=0.10)                
        plt.tick_params(bottom = False, left=False, labelbottom =False, labelleft=False)
        
        
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.show()
    
    # plt.close()
    return fig
    

##----------------------------------------------------------------------
## display_instances from pr_scores
##----------------------------------------------------------------------
def display_instances_w_scores(input, score_column, 
                      class_names,
                      title="", only_classes = None, 
                      size = 12, 
                      ax=None, score_range = (0.0, 1.0)):
    """
    boxes:                  [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    pr_scores :             Condensed score array (scores_by_image)
    class_names:            list of class names of the dataset
    figsize:                (optional) the size of the image.
    max_score:              show instances with score less than this 
    """
    # Number of instances
    image     = input['image']
    pr_scores = input['pr_scores']
    boxes     = pr_scores[:,:4]
    class_ids = pr_scores[:,4].astype(int)
    scores    = pr_scores[:,score_column]
    det_ind   = pr_scores[:,6].astype(int)
    sequences = pr_scores[:,7].astype(int)

    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == class_ids.shape[0], " boxes.shape[0]: {:d} must be ==class_ids.shape[0]: {:d}".format(boxes.shape[0], class_ids.shape[0])
    print(' display_instances() : Image shape: ', image.shape)

    if not ax:
        ax = get_ax(rows =1, cols = 1, size= size)
        # _, ax = plt.subplots(rows = 1, cols = 1, figsize = (size,size))

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    # ax.axis('off')
    ax.set_title(title)
 
    masked_image = image.astype(np.uint32).copy()

    for i in range(N):
        class_id = class_ids[i]
        
        if only_classes is not None:
            if class_id not in only_classes:
                continue
    
        if scores is not None:
            if scores[i] <= score_range[0] or scores[i] >= score_range[1]:
                continue
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)

        # Label
        
        score = scores[i] if scores is not None else None
        if det_ind[i] == -1:
            det_ttl = ' ADDED FP'
        else:
            det_ttl = ''
            
        if class_id >= 0 :
            label = class_names[class_id] + det_ttl
        else:
            label = class_names[-class_id] + ' (CROWD)'
            
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{:2d}-{} {:.4f}".format(class_id, label, score) if score else label
        ax.text(x1, y1 - 2, caption, color='k', size=9, backgroundcolor="w")

    ax.imshow(masked_image.astype(np.uint8))
    plt.show()
    return
    
    
    
