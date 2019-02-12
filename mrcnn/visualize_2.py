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
    




##-----------------------------------------------------------------------------------------------------------    
##
##-----------------------------------------------------------------------------------------------------------    
def display_mrcnn_scores(r, class_names, only = None, display = True):
    '''
    r:    results from mrcnn detect
    cn:   class names
    
    '''
    np_format = {}
    np_format['float'] = lambda x: "%10.4f" % x
    np_format['int']   = lambda x: "%10d" % x
    np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =10000, formatter = np_format)
    print('pr_hm_scores: ', r['pr_hm_scores'].shape)
    print('pr_scores: ', r['pr_scores'].shape)
    
    if only is None:
        only = np.unique(r['class_ids'])
    seq_start = r['pr_hm_scores'].shape[1]
    # print('  class ids  : ', r['class_ids'], type(r['class_ids']))
    # print('  scores     : ', r['scores'], type(r['scores']))
    print('-'*175)
    print('                          |   |                 |         alt score 0         |            alt score 1              |            alt score 2              |')
    print('        class             |TP/| mrcnn   normlzd |  gaussian   bbox   nrm.scr* |  ga.sum    mask     score   norm    |  ga.sum    mask     score   norm    |')
    print('seq  id name              | FP| score    score  |    sum      area   gau.sum  |  in mask   sum              score   |  in mask   sum              score   |')
    print('-'*175)

    for cls,scr,pre, roi in zip(r['class_ids'], r['scores'], r['pr_scores'],r['rois']):
    #     print(' {:4d}      {:12s}   {:.4f}   {:.4f}     {:.4f}  {:7.4f}  {:.4f}      {:.4f}   {}'.
        cx = pre[1] + (pre[3]-pre[1])/2
        cy = pre[0] + (pre[2]-pre[0])/2
        if cls in only:

            print('{:3.0f} {:3d} {:15s}   |{:2.0f} | {:.4f}   {:.4f} |'\
                  '  {:.4f}  {:9.4f}  {:.4f}  |  {:7.4f}  {:8.4f}  {:.4f}  {:.4f}  |  {:7.4f}  {:8.4f}  {:.4f}  {:.4f}  |'\
                  '  {:6.2f}  {:6.2f}'.          
              format(pre[7], cls, class_names[cls], pre[6], scr, pre[8], 
                     pre[9], pre[10], pre[11], 
                     pre[12], pre[13], pre[14], pre[17], 
                     pre[18], pre[19], pre[20], pre[23],
                     cx, cy ))     #,  pre[4],pre[5],pre[6],roi))
    if display:
        visualize.display_instances(r['image'], r['rois'], r['class_ids'], class_names, r['scores'], only_classes= only, size =8)

##-----------------------------------------------------------------------------------------------------------    
##
##-----------------------------------------------------------------------------------------------------------    
def display_pr_scores(pr_scores, class_names, only = None, display = True):
    '''
    pr_scores:    pr_scores returned from detection or evaluation process
    cn:   class names
    
    '''
    np_format = {}
    np_format['float'] = lambda x: "%10.4f" % x
    np_format['int']   = lambda x: "%10d" % x
    np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =10000, formatter = np_format)
    
    if only is None:
        only = np.unique(pr_scores[:,4])

    seq_start = pr_scores.shape[1]
    # print('  class ids  : ', r['class_ids'], type(r['class_ids']))
    # print('  scores     : ', r['scores'], type(r['scores']))
    print('\nPR_SCORES:')
    print('-'*150)
    print('                                            |         alt score 0         |            alt score 1              |            alt score 2              |')
    print('        class            TP mrcnn   normlzd |  gaussian   bbox   nrm.scr* |  ga.sum    mask     score   norm    |  ga.sum    mask     score   norm    |')
    print('seq  id name             FP score    score  |    sum      area   gau.sum  |  in mask   sum              score   |  in mask   sum              score   |')
    print('-'*150)

    for pre in pr_scores:
    #     print(' {:4d}      {:12s}   {:.4f}   {:.4f}     {:.4f}  {:7.4f}  {:.4f}      {:.4f}   {}'.
        cx = pre[1] + (pre[3]-pre[1])/2
        cy = pre[0] + (pre[2]-pre[0])/2
        # cls = int(pre[4])
        if pre[4] in only:
            cls = int(pre[4])
            print('{:3.0f} {:3.0f} {:13s}   {:2.0f}  {:.4f}   {:.4f} |  {:.4f}  {:9.4f}  {:.4f}  |  {:7.4f}  {:8.4f}  {:.4f}  {:.4f}  |  {:7.4f}  {:8.4f}  {:.4f}  {:.4f}  |'\
                  '  {:6.2f}  {:6.2f}'.          
              format(pre[7], pre[4], class_names[cls], pre[6], pre[5], pre[8], 
                     pre[9], pre[10], pre[11], 
                     pre[12], pre[13], pre[14], pre[17], 
                     pre[18], pre[19], pre[20], pre[23],
                     cx, cy ))     #,  pre[4],pre[5],pre[6],roi))
    # if display:
        # visualize.display_instances(r['image'], r['rois'], r['class_ids'], class_names, r['scores'], only_classes= only, size =8)

##-----------------------------------------------------------------------------------------------------------    
##
##-----------------------------------------------------------------------------------------------------------    
def display_fcn_scores(fcn_scores, class_names, only = None, display = True):
    '''
    fcn_scores:    fcn_scores returned from detection or evaluation process
    cn:   class names
    
    '''
    np_format = {}
    np_format['float'] = lambda x: "%10.4f" % x
    np_format['int']   = lambda x: "%10d" % x
    np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =10000, formatter = np_format)
    
    if only is None:
        only = np.unique(fcn_scores[:,4])
        print(' Dispaly classes : ', only)
    seq_start = fcn_scores.shape[1]
    # print('  class ids  : ', r['class_ids'], type(r['class_ids']))
    # print('  scores     : ', r['scores'], type(r['scores']))
    print('\nFCN_SCORES:')
    print('-'*175)
    print('                                             |         FCN score 0          |            FCN score 1                 |            FCN score 2              |')
    print('        class            TP  mrcnn   normlzd |  gaussian   bbox     nrm.scr*|  ga.sum    mask       score     norm   |  ga.sum    mask     score   norm    |')
    print('seq  id name             FP  score    score  |    sum      area     gau.sum |  in mask   sum                  score  |  in mask   sum              score   |')
    print('-'*175)

    for pre in fcn_scores:
    #     print(' {:4d}      {:12s}   {:.4f}   {:.4f}     {:.4f}  {:7.4f}  {:.4f}      {:.4f}   {}'.
        cx = pre[1] + (pre[3]-pre[1])/2
        cy = pre[0] + (pre[2]-pre[0])/2
        # cls = int(pre[4])
        if pre[4] in only:
            cls = int(pre[4])
            print('{:3.0f} {:3.0f} {:13s}   {:2.0f}  {:.4f}   {:.4f}  |'\
                  ' {:9.4f}  {:6.0f}  {:9.4f} |'\
                  ' {:9.4f}  {:6.0f}  {:9.4f}  {:7.4f} |'\
                  ' {:9.4f}  {:6.0f}  {:8.4f}  {:7.4f} |'\
                  ' {:6.2f}  {:6.2f}'.          
              format(pre[7], pre[4], class_names[cls], pre[6], pre[5], pre[8], 
                     pre[9], pre[10], pre[11], 
                     pre[12], pre[13], pre[14], pre[17], 
                     pre[18], pre[19], pre[20], pre[23],
                     cx, cy ))     #,  pre[4],pre[5],pre[6],roi))
    print()
    # if display:
        # visualize.display_instances(r['image'], r['rois'], r['class_ids'], class_names, r['scores'], only_classes= only, size =8)



##-----------------------------------------------------------------------------------------------------------    
##
##-----------------------------------------------------------------------------------------------------------        
def display_fcn_scores_box_info(fcn_scores, class_names, fcn_hm = None, only = None):
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
    print('  classes     : ', only)
    print('-'*175)
    print('                                                         |                                          |               |   (COVAR)    |   ')
    print('BOX     class                                            |                   Width   Height         | MRCNN CLS NRM |     SQRT     |   FROM/TO ')
    print('seq  id     name              X1/Y1              X2/Y2   |    CX / CY         (W)  ~  (H)      AREA | SCORE  SCORE  |  W/2    H/2  |  X      Y')
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
        print('{:3.0f} {:3.0f} {:15s}'\
          ' ({:6.2f},{:6.2f})  ({:6.2f},{:6.2f}) |'\
          ' {:6.2f}/{:6.2f}  {:7.2f}~{:7.2f}  {:7.2f} {} |'\
          ' {:6.4f}  {:6.4f}|{:6.2f}  {:6.2f}|'\
          ' {} {} {} {} {} | {} {} {} {} {}'.format(pre[7], pre[4],  class_names[cls], 
                 pre[1], pre[0],  pre[3], pre[2], 
                 cx, cy,  width,  height, area, pre[13],
                 pre[5], pre[8], sq_covar_x, sq_covar_y, 
                 from_x, to_x, from_y, to_y , clip_area,
                 from_x_r, to_x_r, from_y_r, to_y_r, clip_area_r
              ))    
        print()
        for i in range(1,fcn_hm.shape[-1]):
            print('  cls: {:3d}' \
                  '  cx/cy: {:8.3f}   cx/cy+-1: {:8.3f}   cx/cy+-3: {:8.3f}' \
                  '  fr/to: {:8.3f}   fr/to+1: {:8.3f}    fr-1/to+1: {:8.3f}   full: {:8.3f}'.format( i, 
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
def display_pr_hm_scores(pr_hm_scores, class_names, only = None):
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
        only = range(pr_hm_scores.shape[0])
#     seq_start = r['pr_hm_scores'].shape[1]
    # print('  class ids  : ', r['class_ids'], type(r['class_ids']))
    print('  classes     : ', only)
    print('-'*175)
    print('                                          |           alt score 0      |              alt score 1              |                alt score 2            |')
    print('        class             mrcnn   normlzd |  gauss     bbox   nrm.scr* |  ga.sum     mask    score      norm   |   ga.sum     mask     score   norm    |')
    print('seq  id     name          score   score   |  sum       area   gau.sum  |  in mask    sum                score  |   in mask    sum              score   |')
    print('-'*175)
    
    for cls in range(pr_hm_scores.shape[0]):
        if cls in only:
            for pre in pr_hm_scores[cls]:
        #     print(' {:4d}      {:12s}   {:.4f}   {:.4f}     {:.4f}  {:7.4f}  {:.4f}      {:.4f}   {}'.
                width = pre[3]-pre[1]
                height = pre[2]-pre[0]
                if width == height == 0:
                    continue
                cx = pre[1] + width/2
                cy = pre[0] + height/2
                area = (pre[3]-pre[1])* (pre[2]-pre[0])
                print('{:3.0f} {:3.0f} {:15s}   {:.4f}   {:.4f} |'\
                      ' {:6.4f}  {:9.4f}  {:7.4f} |'\
                      ' {:8.4f}  {:8.4f}  {:7.4f}   {:7.4f} |'\
                      ' {:8.4f}  {:8.4f}  {:7.4f}   {:7.4f} |'\
                      ' {:6.2f}  {:6.2f}  {:9.4f}'.          
                  format(pre[6], pre[4], class_names[cls], pre[5], pre[7], 
                     pre[8], pre[9], pre[10], 
                     pre[11], pre[12], pre[13], pre[14], 
                     pre[17], pre[18], pre[19], pre[22],
                     cx, cy , area))     #,  pre[4],pre[5],pre[6],roi))
            print('-'*170)
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
    
    
    