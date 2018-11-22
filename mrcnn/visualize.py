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
import matplotlib.patches as patches
import matplotlib.lines as lines
import skimage.util
from   skimage.measure import find_contours
from   PIL  import Image
from   matplotlib.patches import Polygon
from   matplotlib import cm
from   mpl_toolkits.mplot3d import Axes3D
from   matplotlib.ticker import LinearLocator, FormatStrFormatter
import mrcnn.utils as utils
from   mrcnn.datagen     import load_image_gt    


############################################################
#  Visualization
############################################################
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
    

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

##----------------------------------------------------------------------
## display_images
##----------------------------------------------------------------------
def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None, width=14):
    """
    Display the given set of images, optionally with titles.
    
    images:             list or array of image tensors in HWC format.
    titles:             optional. A list of titles to display with each image.
    cols:               number of images per row
    cmap:               Optional. Color map to use. For example, "Blues".
    norm:               Optional. A Normalize instance to map values to colors.
    interpolation:      Optional. Image interporlation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
     
    plt.figure(figsize=(width, width * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        title += "H x W={}x{}".format(image.shape[0], image.shape[1])
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()

##----------------------------------------------------------------------
## display_image
##      figsize : tuple of integers, optional: (width, height) in inches
##                default: None
##                If not provided, defaults to rc figure.figsize.
##----------------------------------------------------------------------
def display_image(image, title='', cmap=None, norm=None,
                   interpolation=None, figsize=(10,10), ax=None):
    """
    Display one image, optionally with titles.
    
    image:             list or array of image tensors in HWC format.
    title:             optional. A list of titles to display with each image.
    cols:               number of images per row
    cmap:               Optional. Color map to use. For example, "Blues".
    norm:               Optional. A Normalize instance to map values to colors.
    interpolation:      Optional. Image interporlation to use for display.
    """
    plt.figure(figsize=figsize)
    # if title is None:
    title += "H x W={}x{}".format(image.shape[0], image.shape[1])
    plt.title(title, fontsize=12)
    plt.imshow(image, cmap=cmap,
               norm=norm, interpolation=interpolation)
        
        
##----------------------------------------------------------------------
## display_image
##      figsize : tuple of integers, optional: (width, height) in inches
##                default: None
##                If not provided, defaults to rc figure.figsize.
##----------------------------------------------------------------------
def display_image_bw(image, title="B/W Display" , cmap=None, norm=None,
                   interpolation=None, figsize=(10,10), ax=None):
    """
    Display one image, optionally with titles.
    
    image:             list or array of image tensors in HWC format.
    title:             optional. A list of titles to display with each image.
    cols:               number of images per row
    cmap:               Optional. Color map to use. For example, "Blues".
    norm:               Optional. A Normalize instance to map values to colors.
    interpolation:      Optional. Image interporlation to use for display.
    """
    plt.figure(figsize=figsize)
    plt.title(title, fontsize=12)
    arr = np.asarray(image)
    # print(type(image), image.shape)
    # print(type(arr), arr.shape)
    # plt.imshow(image.astype(np.uint8), cmap=cmap,
               # norm=norm, interpolation=interpolation)
    plt.imshow(arr, cmap='gray')

    
##----------------------------------------------------------------------
## display_instances
##----------------------------------------------------------------------
def display_instances(image, boxes, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None, score_range = (0.0, 1.0)):
    """
    boxes:                  [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks:                  [num_instances, height, width]
    class_ids:              [num_instances]
    class_names:            list of class names of the dataset
    scores:                 (optional) confidence scores for each box
    figsize:                (optional) the size of the image.
    max_score:              show instances with score less than this 
    """
    # Number of instances
    N = boxes.shape[0]   
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == class_ids.shape[0]
        
    # print(' display_instances() : Image shape: ', image.shape)
   
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

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

        if scores is not None:
            # print(' boxes ' ,i,'   ' , boxes[i], 'score: ', scores[i], '    ', score_range)
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
        
        class_id = class_ids[i]
        if class_id >= 0 :
            label = class_names[class_id]
        else:
            label = class_names[-class_id] + ' (CROWD)'
            
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{} {:.3f}".format(label, score) if score else label
        ax.text(x1, y1 + 8, caption, color='k', size=9, backgroundcolor="w")

        # Mask
        # mask = masks[:, :, i]
        # masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        # padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        # padded_mask[1:-1, 1:-1] = mask
        # contours = find_contours(padded_mask, 0.5)
        # for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            # verts = np.fliplr(verts) - 1
            # p = Polygon(verts, facecolor="none", edgecolor=color)
            # ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    plt.show()
    return 
    
##----------------------------------------------------------------------
## display_instances_with_mask
##----------------------------------------------------------------------
def display_instances_with_mask(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None):
    """
    boxes:                  [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks:                  [num_instances, height, width]
    class_ids:              [num_instances]
    class_names:            list of class names of the dataset
    scores:                 (optional) confidence scores for each box
    figsize:                (optional) the size of the image.
    max_score:              show instances with score less than this 
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    
    # print(' display_instances WITH MASK() : Image shape: ', image.shape)
    
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

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
        
        class_id = class_ids[i]
        
        # label = class_names[class_id]
        if class_id >= 0 :
            label = class_names[class_id]
        else:
            label = class_names[-class_id] + ' (CROWD)'

        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{} {:.3f}".format(label, score) if score else label
        ax.text(x1, y1 + 8, caption, color='k', size=11, backgroundcolor="w")
        
        
        # Mask
        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    plt.show()
    return

    
##----------------------------------------------------------------------
## display_instances from pr_scores
##----------------------------------------------------------------------
def display_instances_from_prscores(image, pr_scores, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None, score_range = (0.0, 1.0)):
    """
    boxes:                  [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks:                  [num_instances, height, width]
    class_ids:              [num_instances]
    class_names:            list of class names of the dataset
    scores:                 (optional) confidence scores for each box
    figsize:                (optional) the size of the image.
    max_score:              show instances with score less than this 
    """
    # Number of instances
    
    boxes = pr_scores[:,:4]
    class_ids = pr_scores[:,4].astype(int)
    scores = pr_scores[:,5]
    sequences = pr_scores[:,6].astype(int)

    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == class_ids.shape[0]
    print(' display_instances() : Image shape: ', image.shape)

    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

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
        
        class_id = class_ids[i]
        if class_id >= 0 :
            label = class_names[class_id]
        else:
            label = class_names[-class_id] + ' (CROWD)'
            
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{:2d}-{} {:.4f}".format(class_id, label, score) if score else label
        ax.text(x1, y1 - 8, caption, color='k', size=9, backgroundcolor="w")

    ax.imshow(masked_image.astype(np.uint8))
    plt.show()
    return
    
    

##----------------------------------------------------------------------
## draw_rois (along with the refined_rois)
##----------------------------------------------------------------------
# def draw_rois_with_refinements(image, rois, refined_rois, mask, class_ids, class_names, limit=10):
def draw_rois_with_refinements(image, rois, refined_rois, class_ids, class_names, limit=0, random = False):
    """
    rois:           [n, 4 : {y1, x1, y2, x2}] list of anchors in image coordinates.
    refined_rois:   [n, 4 : {y1, x1, y2, x2}] the same anchors but refined to fit objects better.
    """
    masked_image = image.copy()

    # Pick random anchors in case there are too many.
    print('    rois.shape[0]:  ',rois.shape[0], ' limit = ', limit)
    if limit == 0 :
        limit = max(rois.shape[0], limit)
    print('    limit : ', limit)
    
    ids = np.arange(limit, dtype=np.int32)
    if random:
        ids = np.random.choice(ids, limit, replace=False) if ids.shape[0] > limit else ids
    print(' ids : ', ids)
    
    
    fig, ax = plt.subplots(1, figsize=(12, 12))
    if rois.shape[0] > limit:
        plt.title("Showing {} random ROIs out of {}".format(
            len(ids), rois.shape[0]))
    else:
        plt.title("{} ROIs".format(len(ids)))

    # Show area outside image boundaries.
    ax.set_ylim(image.shape[0] + 20, -20)
    ax.set_xlim(-50, image.shape[1] + 20)
    ax.axis('off')

    for i, id in enumerate(ids):
        print('i: ', i, 'id :', id)
        color = np.random.rand(3)
        class_id = class_ids[id]
        # ROI
        y1, x1, y2, x2 = rois[id]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              edgecolor=color if class_id else "gray",
                              facecolor='none', linestyle="dashed")
        ax.add_patch(p)
        # Refined ROI
        if class_id:
            ry1, rx1, ry2, rx2 = refined_rois[id]
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal for easy visualization
            ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

            # Label
            label = class_names[class_id]
            ax.text(rx1, ry1 + 8, "{}".format(label),
                    color='w', size=11, backgroundcolor="none")

            # Mask
            # m = utils.unmold_mask(mask[id], rois[id]
                                  # [:4].astype(np.int32), image.shape)
            # masked_image = apply_mask(masked_image, m, color)

    ax.imshow(masked_image)

    # Print stats
    print("Positive ROIs: ", class_ids[class_ids > 0].shape[0])
    print("Negative ROIs: ", class_ids[class_ids == 0].shape[0])
    print("Positive Ratio: {:.2f}".format(
        class_ids[class_ids > 0].shape[0] / class_ids.shape[0]))

        
##----------------------------------------------------------------------
## draw rois proposals  (w/o refinements)
##----------------------------------------------------------------------
def draw_rois(image, rois, class_ids, class_names, bbox_ids = None , limit=0, random = False):
    """
    anchors: [n, (y1, x1, y2, x2)] list of anchors in image coordinates.
    proposals: [n, 4] the same anchors but refined to fit objects better.
    bbox_ids : list of bbox ids that will be displayed. If not specified will use limit
    """
    masked_image = image.copy()

    # Pick random anchors in case there are too many.
    print('    rois.shape[0]:  ',rois.shape[0], ' limit = ', limit)
    
    if bbox_ids:
        pass
    else:
        if limit == 0 :
            limit = max(rois.shape[0], limit)
        print('    limit : ', limit)
        
        bbox_ids = np.arange(limit, dtype=np.int32)
        if random:
            bbox_ids = np.random.choice(bbox_ids, limit, replace=False) if ids.shape[0] > limit else bbox_ids
        print(' bbox_ids : ', bbox_ids)
        
    fig, ax = plt.subplots(1, figsize=(12, 12))
    if rois.shape[0] > limit:
        plt.title("Showing {} random ROIs out of {}".format(
            len(bbox_ids), rois.shape[0]))
    else:
        plt.title("{} ROIs".format(len(bbox_ids)))

    # Show area outside image boundaries.
    ax.set_ylim(image.shape[0] + 20, -20)
    ax.set_xlim(-50, image.shape[1] + 20)
    ax.axis('off')

    for i, id in enumerate(bbox_ids):
        color = np.random.rand(3)
        class_id = class_ids[id]
        # ROI

        if not class_id:
            # Skip this instance. Has no class id 
            print('index: ', i, 'box_id :', id, 'class_id: ', class_id,' Skipping box ',i)
            continue
         
        # print('index: ', i, 'box_id :', id, 'class_id: ', class_id)
            
        y1, x1, y2, x2 = rois[id]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              edgecolor=color if class_id else "gray",
                              facecolor='none', linestyle="dashed")
        ax.add_patch(p)

        # Refined ROI
        # if not class_id:
            # Skip this instance. Has no class id 
            # print(' Skipping box ',i)
            # continue

            # ry1, rx1, ry2, rx2 = refined_rois[id]
            # p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  # edgecolor=color, facecolor='none')
            # ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal for easy visualization
            # ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

            # Label
        label = class_names[class_id]
        # ax.text(rx1, ry1 + 8, "{}".format(label),
                # color='w', size=11, backgroundcolor="none")
        ax.text(x1, y1 + 8, "{}".format(label),
                color='w', size=11, backgroundcolor="none")

    ax.imshow(masked_image)

    # Print stats
    print("Positive ROIs: ", class_ids[class_ids > 0].shape[0])
    print("Negative ROIs: ", class_ids[class_ids == 0].shape[0])
    print("Positive Ratio: {:.2f}".format(
        class_ids[class_ids > 0].shape[0] / class_ids.shape[0]))

'''
similar to draw_roi_proposals
        
##----------------------------------------------------------------------
## draw_output_rois
##----------------------------------------------------------------------
def draw_output_rois(image, rois, class_ids, class_names, limit=10):
    """
    rois :     [n, (y1, x1, y2, x2)] list bounding boxes in NN image format (resized and padded).
    proposals: [n, 4] the same anchors but refined to fit objects better.
    """
    masked_image = image.copy()

    # Pick random anchors in case there are too many.
    print('rois.shape[0]:  ',rois.shape[0])
    ids = np.arange(rois.shape[0], dtype=np.int32)
    # ids = np.random.choice(ids, limit, replace=False) if ids.shape[0] > limit else ids

    fig, ax = plt.subplots(1, figsize=(12, 12))
    if rois.shape[0] > limit:
        plt.title("Showing {} random ROIs out of {}".format(
            len(ids), rois.shape[0]))
    else:
        plt.title("{} ROIs".format(len(ids)))

    # Show area outside image boundaries.
    ax.set_ylim(image.shape[0] + 20, -20)
    ax.set_xlim(-50, image.shape[1] + 20)
    ax.axis('off')

    for i, id in enumerate(ids):
        color = np.random.rand(3)
        class_id = class_ids[id]
        # ROI
        print('i: ', i, 'id :', id, 'class_id: ', class_id)
        if not class_id:
            # Skip this instance. Has no class id 
            print(' Skipping box ',i)
            continue

        y1, x1, y2, x2 = rois[id]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              edgecolor=color if class_id else "gray",
                              facecolor='none', linestyle="dashed")
        ax.add_patch(p)

        label = class_names[class_id]
        ax.text(x1, y1 + 8, "{}".format(label),
                color='w', size=11, backgroundcolor="none")

    ax.imshow(masked_image)

    # Print stats
    print("Positive ROIs: ", class_ids[class_ids > 0].shape[0])
    print("Negative ROIs: ", class_ids[class_ids == 0].shape[0])
    print("Positive Ratio: {:.2f}".format(class_ids[class_ids > 0].shape[0] / class_ids.shape[0]))
'''

##----------------------------------------------------------------------
## draw_box
##----------------------------------------------------------------------
# TODO: Replace with matplotlib equivalent?
def draw_box(image, box, color):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    y1, x1, y2, x2 = box
    image[y1:y1 + 2, x1:x2] = color
    image[y2:y2 + 2, x1:x2] = color
    image[y1:y2, x1:x1 + 2] = color
    image[y1:y2, x2:x2 + 2] = color
    return image


##----------------------------------------------------------------------
## display_top_masks
##----------------------------------------------------------------------    
def display_top_masks(image, mask, class_ids, class_names, limit=4):
    """Display the given image and the top few class masks."""
    to_display = []
    titles = []
    to_display.append(image)
    titles.append("H x W={}x{}".format(image.shape[0], image.shape[1]))
    # Pick top prominent classes in this image
    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]
    # Generate images and titles
    for i in range(limit):
        class_id = top_ids[i] if i < len(top_ids) else -1
        # Pull masks of instances belonging to the same class.
        m = mask[:, :, np.where(class_ids == class_id)[0]]
        m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
        to_display.append(m)
        titles.append(class_names[class_id] if class_id != -1 else "-")
    display_images(to_display, titles=titles, cols=limit + 1, cmap="Blues_r")

##----------------------------------------------------------------------
## plot_precision_recall
##----------------------------------------------------------------------
def plot_precision_recall(AP, precisions, recalls):
    """Draw the precision-recall curve.

    AP: Average precision at IoU >= 0.5
    precisions: list of precision values
    recalls: list of recall values
    """
    # Plot the Precision-Recall curve
    _, ax = plt.subplots(1)
    ax.set_title("Precision-Recall Curve. AP@50 = {:.3f}".format(AP))
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 1.1)
    _ = ax.plot(recalls, precisions)

##----------------------------------------------------------------------
## plot_overlaps
##----------------------------------------------------------------------
def plot_overlaps(gt_class_ids, pred_class_ids, pred_scores,
                  overlaps, class_names, threshold=0.5):
    """
    Draw a grid showing how ground truth objects are classified.
    gt_class_ids: [N] int. Ground truth class IDs
    pred_class_id: [N] int. Predicted class IDs
    pred_scores: [N] float. The probability scores of predicted classes
    overlaps: [pred_boxes, gt_boxes] IoU overlaps of predictins and GT boxes.
    class_names: list of all class names in the dataset
    threshold: Float. The prediction probability required to predict a class
    """
    gt_class_ids = gt_class_ids[gt_class_ids != 0]
    pred_class_ids = pred_class_ids[pred_class_ids != 0]

    plt.figure(figsize=(12, 10))
    plt.imshow(overlaps, interpolation='nearest', cmap=plt.cm.Blues)
    plt.yticks(np.arange(len(pred_class_ids)),
               ["{} ({:.2f})".format(class_names[int(id)], pred_scores[i])
                for i, id in enumerate(pred_class_ids)])
    plt.xticks(np.arange(len(gt_class_ids)),
               [class_names[int(id)] for id in gt_class_ids], rotation=90)

    thresh = overlaps.max() / 2.
    for i, j in itertools.product(range(overlaps.shape[0]),
                                  range(overlaps.shape[1])):
        text = ""
        if overlaps[i, j] > threshold:
            text = "match" if gt_class_ids[j] == pred_class_ids[i] else "wrong"
        color = ("white" if overlaps[i, j] > thresh
                 else "black" if overlaps[i, j] > 0
                 else "grey")
        plt.text(j, i, "{:.3f}\n{}".format(overlaps[i, j], text),
                 horizontalalignment="center", verticalalignment="center",
                 fontsize=9, color=color)

    plt.tight_layout()
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")

##----------------------------------------------------------------------
## draw_boxes
##----------------------------------------------------------------------
def draw_boxes(image, boxes=None, refined_boxes=None,
               masks=None, captions=None, visibilities=None,
               title="", ax=None, width=12):
    """Draw bounding boxes and segmentation masks with differnt
    customizations.

    boxes:                  [N, (y1, x1, y2, x2, class_id)] in image coordinates.
    refined_boxes:          Like boxes, but draw with solid lines to show
                            that they're the result of refining 'boxes'.
    masks:                  [N, height, width]
    captions:               List of N titles to display on each box
    visibilities:           (optional) List of values of 0, 1, or 2. Determine how
                            prominant each bounding box should be.
    title:                  An optional title to show over the image
    ax:                     (optional) Matplotlib axis to draw on.
    """
    # Number of boxes
    assert boxes is not None or refined_boxes is not None
    N = boxes.shape[0] if boxes is not None else refined_boxes.shape[0]

    # Matplotlib Axis
    if not ax:
        _, ax = plt.subplots(1, figsize=(width, width))

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.   
    margin = image.shape[0] // 10
    ax.set_ylim(image.shape[0] + margin, -margin)
    ax.set_xlim(-margin, image.shape[1] + margin)
    ax.axis('off')

    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        # Box visibility
        visibility = visibilities[i] if visibilities is not None else 1
        if visibility == 0:
            color = "gray"
            style = "dotted"
            alpha = 0.5
        elif visibility == 1:
            color = colors[i]
            style = "dotted"
            alpha = 1
        elif visibility == 2:
            color = colors[i]
            style = "solid"
            alpha = 1

        # Boxes
        if boxes is not None:
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=alpha, linestyle=style,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Refined boxes
        if refined_boxes is not None and visibility > 0:
            ry1, rx1, ry2, rx2 = refined_boxes[i].astype(np.int32)
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal
            if boxes is not None:
                ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

        # Captions
        if captions is not None:
            caption = captions[i]
            # If there are refined boxes, display captions on them
            if refined_boxes is not None:
                y1, x1, y2, x2 = ry1, rx1, ry2, rx2
 
##             x = random.randint(x1, (x1 + x2) // 2)
## replaced x1 with x1 // 1 to avoid failure in randint (13-03-2018)
            x = random.randint(x1 //1 , (x1 + x2) // 2)
            ax.text(x1, y1, caption, size=11, verticalalignment='top',
                    color='w', backgroundcolor="none",
                    bbox={'facecolor': color, 'alpha': 0.5,
                          'pad': 2, 'edgecolor': 'none'})

        # Masks
        if masks is not None:
            mask = masks[:, :, i]
            masked_image = apply_mask(masked_image, mask, color)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))

##----------------------------------------------------------------------
## display_table
##----------------------------------------------------------------------
def display_table(table):
    """Display values in a table format.
    table: an iterable of rows, and each row is an iterable of values.
    """
    style = "<style> @page{ size: a4 landscape;} </style>"
    html = ""
    for row in table:
        row_html = ""
        for col in row:
            row_html += "<td>{}</td>".format(str(col))
        html += "<tr>" + row_html + "</tr>"
    html = "<table>" + html + "</table>"
    IPython.display.display(IPython.display.HTML(html))
    return html

##----------------------------------------------------------------------
## display_weight_stats
##----------------------------------------------------------------------    
def display_weight_stats(model):
    """Scans all the weights in the model and returns a list of tuples
    that contain stats about each weight.
    """
    layers = model.get_trainable_layers()
    table = [["LAYER"," WEIGHT NAME", "LAYER TYPE", "SHAPE", "MIN", "MAX", "STD"]]
    for l_idx , l in enumerate(layers):
        weight_values  = l.get_weights()  # list of Numpy arrays
        weight_tensors = l.weights        # list of TF tensors
        
        for i, w in enumerate(weight_values):
            weight_name = weight_tensors[i].name
            
            # Detect problematic layers. Exclude biases of conv layers.
            alert = ""
            if w.min() == w.max() and not (l.__class__.__name__ in [ "Conv2D","Conv2DTranspose"] and i == 1):
                alert += "  <span style='color:red'>*** dead?</span>"
            if np.abs(w.min()) > 1000 or np.abs(w.max()) > 1000:
                alert += "  <span style='color:red'>*** Overflow?</span>"
            # Add row
            table.append(["{:3d}".format(l_idx),
                "{:40s}".format(weight_name + alert),
                l.__class__.__name__,
                str(w.shape),
                "{:+12.10f}".format(w.min()),
                "{:+13.10f}".format(w.max()),
                "{:+12.10f}".format(w.std()),
            ])
    html = display_table(table)
    return html


##----------------------------------------------------------------------
## display_weight_histograms
##----------------------------------------------------------------------    
def display_weight_histograms(model, width= 15, height=4, bins =50, filename = ''):
    LAYER_TYPES = ['Conv2D', 'Dense', 'Conv2DTranspose']
    
    title = ' Trainable weights distribution - weight file: ' + filename
    # Get layers
    layers = model.get_trainable_layers()
    layers = list(filter(lambda l: l.__class__.__name__ in LAYER_TYPES, layers))
    
    # Display Histograms
    fig, ax = plt.subplots(len(layers), 2, 
                           figsize=( width, height * len(layers)),
                           gridspec_kw={"hspace":1})
                           
    for l, layer in enumerate(layers):
        weights = layer.get_weights()
        for w, weight in enumerate(weights):
            tensor = layer.weights[w]
            ax[l, w].set_title(tensor.name)
            _ = ax[l, w].hist(weight[w].flatten(), bins)   # second parm is number of bins
    fig.suptitle(title, fontsize =12 )
    # plt.subplots_adjust(top=0.97, bottom=0.02, left=0.02, right=0.98, hspace=0.0, wspace=0.20)      
    plt.subplots_adjust(top=0.97) 
    return fig

    
##----------------------------------------------------------------------
## display_gt_bboxes
##----------------------------------------------------------------------    
    
def display_gt_bboxes(model_info, input_image_meta, image_idx=0):

    dataset_train = model_info[2]
    config = model_info[1]
    image_idx = 0
    image_id = input_image_meta[image_idx,0]
    print('Image id: ',image_id)
    p_original_image, p_image_meta, p_gt_class_id, p_gt_bbox, p_gt_mask =  \
                load_image_gt(dataset_train, config, image_id, augment=False, use_mini_mask=True)
    # print(p_gt_class_id.shape, p_gt_bbox.shape, p_gt_mask.shape)
    print(p_gt_bbox[0:3,:])
    print(p_gt_class_id)
    draw_boxes(p_original_image, p_gt_bbox[0:3])    
    return


##----------------------------------------------------------------------
## display_roi_proposals
##----------------------------------------------------------------------       
def display_roi_proposals(model_info, input_image_meta, pred_tensor, classes, image_idx = 0) :

    dataset_train = model_info[2]
    config = model_info[1]
    image_id = input_image_meta[image_idx,0]

    p_image, p_image_meta, p_gt_class_id, p_gt_bbox, p_gt_mask =  \
                load_image_gt(dataset_train, config, image_id, augment=False, use_mini_mask=True)
    print('Image id      : ',image_id)
    print('Image metadata: ', p_image_meta)
    for cls in classes:
        ttl = 'FR-CNN (pred_tensor) refined ROI bounding boxes - img:{} (img_id {}) class id: {} '.format(image_idx,image_id, cls)
        caps = [str(i)+'-'+str(np.around(x[1],decimals = 3))  for i,x in enumerate(pred_tensor[image_idx,cls,:].tolist()) ]
        draw_boxes(p_image, pred_tensor[image_idx,cls,:,0:4], captions = caps, title = ttl, width =10)
    



##------------------------------------------------------------------------------------    
## display_training_batch()
##------------------------------------------------------------------------------------    
def display_training_batch(dataset, batch_x, masks= False):
    ''' 
    display images in a mrcnn train_batch 
    '''
    # replaced following two lines with next line to avoid the need to pass model to this fuction
    # imgmeta_idx = mrcnn_model.keras_model.input_names.index('input_image_meta')
    # img_meta    = train_batch_x[imgmeta_idx]

    img_meta    = batch_x[1]

    for img_idx in range(img_meta.shape[0]):
        image_id = img_meta[img_idx,0]
        image    = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        bbox     = utils.extract_bboxes(mask)
        class_names = [str(dataset.class_names[class_id]) for class_id in class_ids]
        print(' Image_id    : ', image_id, ' Reference: ', dataset.image_reference(image_id) , 'Coco Id:', dataset.image_info[image_id]['id'])
        print(' Image meta', img_meta[img_idx, :10])
        print(' Class ids   : ', class_ids.shape, '  ' , class_ids)
        print(' Class Names : ', class_names)    #     print('Classes (1: circle, 2: square, 3: triangle ): ',class_ids)
        display_top_masks(image, mask, class_ids, dataset.class_names)
        if masks:
            display_instances_with_mask(image, bbox, mask, class_ids, dataset.class_names) 
        else:
            display_instances(image, bbox, class_ids, dataset.class_names)
    return    
        
##----------------------------------------------------------------------
## plot 2d heatmaps form gauss_scatter with bouding boxes for one image (all classes)
##----------------------------------------------------------------------     
def display_heatmaps( input_list, image_list, image_id, hm = 'pr' ,  heatmaps = None, 
                      class_ids = None, class_names = None,
                      title = '', size = (8,8), columns = 3, config = None) :
    '''
    input
    -----
        Z       Gaussian distribution (Batch Sz, Class_sz, Img Height, Img Width)
        boxes   array of bounding boxes 
    '''
    # image is in molded format - unmold it:
    image      = image_list[image_id]
    image_meta = input_list[0][image_id]


    # print(' image: ', image.shape, image.dtype, np.min(image), np.max(image))
    # image = utils.unresize_image(input_list[0][image_id])
    # image = skimage.util.img_as_ubyte(image)
    # print(' image: ', image.shape, image.dtype, np.min(image), np.max(image))
    hm = hm.upper()
    if heatmaps is None:
        if hm == 'PR':
            heatmaps  =  input_list[1][image_id]
        elif hm == 'GT':
            heatmaps  =  input_list[3][image_id]
        else:
            print(' ERROR - Invalid hm type specified when heatmap parm not passed :', hm )
            return
      
    if hm == 'PR':
        title = 'Mask RCNN Predicted Heatmaps'         
    elif hm == 'GT':
        title = 'Ground Truth Heatmaps' 
    elif hm == 'FCN':
        title = 'FCN Prediction Heatmaps' 
    else:
        print(' ERROR - Invalid hm type specified : ',hm )
        return
  
    if class_ids is None :
        num_classes = heatmaps.shape[-1]
        class_ids   = np.arange(num_classes)
    else:
        num_classes = len(class_ids)
        
    print('Image shape : ',image.shape,' class_ids:', class_ids)

    ## image as passed to function  
    print('1- image    : ', image.shape, image.dtype, np.min(image), np.max(image))
    
    ## add back pixel mean values
    image      = utils.unmold_image(image, config)
    print('2- image    : ', image.shape, image.dtype, np.min(image), np.max(image))
    # display_image(image)

    ## convert from resized to original size 
    image = utils.unresize_image(image, image_meta)
    print('3- image    : ', image.shape, image.dtype, np.min(image), np.max(image))    
    # display_image(image)
     
    ## Convert to grayscale np array   
    image_bw = Image.fromarray(image).convert(mode='L')
    image_bw = np.asarray(image_bw)
    
    print('5- image_bw: ', image_bw.shape, image_bw.dtype, np.min(image_bw), np.max(image_bw))    
    display_image(image_bw, cmap=plt.cm.gray)

    columns  = min(columns, num_classes)
    rows     = math.ceil(num_classes/columns)
    width  = size[0] * columns
    height = size[1] * rows
    fig = plt.figure(figsize=(width, height))
    

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
            ttl = 'Image: {:2d} Cls: {:2d} '.format( image_id,cls)
        else:
            ttl = 'Image: {:2d} Cls: {:2d}/{:s}'.format(image_id, cls, class_names[cls])

        ax = fig.add_subplot(rows, columns, subplot)
        ax.set_title(ttl, fontsize=14)
        ax.tick_params(axis='both', labelsize = 5)
        ax.tick_params(direction='out', length=6, width=1, colors='r', labelsize = 10)
        ax.set_xlabel(' X axis', fontsize=10)
        ax.set_ylabel(' Y axis', fontsize=10)

        unmolded_heatmap = utils.unresize_image(heatmaps[:,:,cls],image_meta, upscale = config.HEATMAP_SCALE_FACTOR)
        
        ax.imshow(image_bw , cmap=plt.cm.gray)
        ax.imshow(unmolded_heatmap, alpha = 0.6,cmap=cm.YlOrRd)              

        # print(ttl,x1[cls,bbox], y1[cls,bbox],x2[cls,bbox],y2[cls,bbox])
        # p = patches.Rectangle( (x1[cls,bbox],y1[cls,bbox]), box_w[cls,bbox], box_h[cls,bbox], 
                               # linewidth=1, alpha=alpha, linestyle=style, edgecolor=color, facecolor='none')
        # ax.add_patch(p)
        plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98, hspace=0.10, wspace=0.10)                
    fig.suptitle(title, fontsize = 15, ha ='center' )
    plt.show()
    
    # plt.close()
    return

    
##----------------------------------------------------------------------
## Display heatmaps overlaying image from fcn_input_batch - separate entry for 'heatmap'
##----------------------------------------------------------------------     
def display_heatmaps_fcn(fcn_input_batch, fcn_input_image, image_id, hm = 'pr' ,  
                      class_ids = None, class_names = None, num_bboxes = 999, flip = False,
                      title = '', size = (8,8), columns = 2, config = None, scaling = False) :
    '''
    input
    -----
        fcn_input_batch: 
        [0]     Image Metadata
        [1]     
        Z       Gaussian distribution (Batch Sz, Class_sz, Img Height, Img Width)
        boxes   array of bounding boxes 
    '''
    image      = fcn_input_image[image_id]
    image_meta = fcn_input_batch[0][image_id]
    display_list = [] 
    title_list   = []
    
    
    if hm == 'pr':
        Z          = fcn_input_batch[1][image_id]
        boxes      = fcn_input_batch[2][image_id]        
        title      = 'FCN Produced Heatmaps - (w/ Predicted Bounding Boxes) '         
    elif hm == 'gt':
        Z          = fcn_input_batch[3][image_id]
        boxes     =  fcn_input_batch[4][image_id]        
        title      = 'FCN Produced Heatmaps - (w/ Ground Truth Bounding Boxes) '         

    scale = config.HEATMAP_SCALE_FACTOR

    if class_ids is None :
        num_classes = Z.shape[-1]
        class_ids   = np.arange(num_classes)
    else:
        num_classes = len(class_ids)
    print(' class_ids:', class_ids)
     
    print('1- Image    : ', image.shape, image.dtype, np.min(image), np.max(image))
    ## add back pixel mean values
    image      = utils.unmold_image(image, config)
    display_list.append(image)
    title_list.append('image pre-flipping')
    # display_image(image, title = 'image pre-flipping')
    
    if flip:
        image = np.fliplr(image)
        display_list.append(image)
        title_list.append('flipped image')
        # display_image(image, title = 'flipped image')
    
    print('2- image    : ', image.shape, image.dtype, np.min(image), np.max(image))

    ## convert from resized to original size 
    image = utils.unresize_image(image, image_meta)
    print('3- image    : ', image.shape, image.dtype, np.min(image), np.max(image))    
    display_list.append(image)
    title_list.append('Reverted to original size')    
    # display_image(image)
 
    ## Convert to grayscale np array   
    image_bw = Image.fromarray(image).convert(mode='L')
    image_bw = np.asarray(image_bw)
    
    print('4- image_bw: ', image_bw.shape, image_bw.dtype, np.min(image_bw), np.max(image_bw))    

    display_images(display_list, title_list, cols = len(title_list), width =18)
    display_image(image_bw, cmap=plt.cm.gray)

    
    columns = min(columns, num_classes)
    rows   = math.ceil(num_classes/columns)
    width  = size[0] * columns
    height = size[1] * rows
    fig = plt.figure(figsize=(width, height))
    
    if num_bboxes == 999 :
        num_bboxes  = boxes.shape[2]  

    if num_bboxes > 0:
        boxes = utils.boxes_to_image_domain(boxes[...,:4], image_meta)
        x1    = boxes[:,:,1] 
        x2    = boxes[:,:,3] 
        y1    = boxes[:,:,0] 
        y2    = boxes[:,:,2] 
        box_w = x2 - x1   # x2 - x1
        box_h = y2 - y1 
        
    colors = random_colors(num_classes)
    style = "dotted"
    linewidth = 1.0
    alpha = 1
    color = (0.5, 0.0, 1.0)

    for idx, cls in enumerate(class_ids):
        # color = colors[idx]
        row = idx // columns
        col = idx  % columns
        subplot = (row * columns) + col +1
        # print('idx ', idx,  ' class:', cls, 'row:', row,'col:', col, 'subplot: ', subplot, 'Z dims:', Z[:,:,cls].shape)
        if class_names is None:
            ttl = 'Image: {:2d} Cls: {:2d} '.format( image_id,cls)
        else:
            ttl = 'Image: {:2d} Cls: {:2d}/{:s}'.format(image_id, cls, class_names[cls])

        ax = fig.add_subplot(rows, columns, subplot)
        ax.set_title(ttl, fontsize=14)
        ax.tick_params(axis='both', labelsize = 5)
        ax.tick_params(direction='out', length=6, width=1, colors='r', labelsize = 10)
        ax.set_xlabel(' X axis', fontsize=10)
        ax.set_ylabel(' Y axis', fontsize=10)
        ax.imshow(image_bw , cmap=plt.cm.gray)
        
        ## unmolded heatmap
        
        ZZ = utils.unresize_image(Z[:,:,cls],image_meta, upscale = config.HEATMAP_SCALE_FACTOR)
        if scaling :
            min_z = np.amin(ZZ, axis = (0,1), keepdims = True)
            max_z = np.amax(ZZ, axis = (0,1), keepdims = True)
            # print(' unmolded heatmap min:{}, max:{}'.format(min_z,max_z))
            ZZ = (ZZ - min_z)/(max_z - min_z + 1.0e-6)
        # ax.invert_xaxis()
        surf = ax.imshow(ZZ, alpha = 0.6,cmap=cm.YlOrRd)              
        
        for bbox in range(num_bboxes):
            p = patches.Rectangle( (x1[cls,bbox],y1[cls,bbox]), box_w[cls,bbox], box_h[cls,bbox], 
                               linewidth=linewidth, alpha=alpha, linestyle=style, edgecolor=color, facecolor='none')
            ax.add_patch(p)
        
        plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98, hspace=0.10, wspace=0.10)   
        # fig.colorbar(surf, shrink=0.8, aspect=30, fraction=0.05)
    
    if scaling :
        title += ' - NORMALIZED to 1'
        
    fig.suptitle(title, fontsize = 18, ha ='center' )
    plt.show()
    
    # plt.close()
    return

    
    
##----------------------------------------------------------------------
## Display heatmaps overlaying image for mrcnn input/output  
##----------------------------------------------------------------------     
def display_heatmaps_mrcnn(mrcnn_input_batch, mrcnn_output_batch, image_id, hm = 'pr' ,  
                      class_ids = None, class_names = None, num_bboxes = 999,
                      title = '', size = (7,7), columns = 2, config = None) :
    '''
    input
    -----
        Z       Gaussian distribution (Batch Sz, Class_sz, Img Height, Img Width)
        boxes   array of bounding boxes 
    '''
    
    image      = mrcnn_input_batch[0][image_id]
    image_meta = mrcnn_input_batch[1][image_id]

    if hm == 'pr':
        heatmaps   = mrcnn_output_batch[0][image_id]
        boxes      = mrcnn_output_batch[1][image_id,:,:,:4]        
        title      = 'MRCNN Heatmaps - Predictions '         
    elif hm == 'gt':
        heatmaps   = mrcnn_output_batch[2][image_id]
        boxes      = mrcnn_output_batch[3][image_id,:,:,:4]        
        title      = 'MRCNN Heatmaps - Ground Truth '         
    else:
        print(' Error - invalid hm parameter....')
        return
        
    if class_ids is None :
        num_classes = heatmaps.shape[-1]
        class_ids   = np.arange(num_classes)
    else:
        class_ids  = np.unique(class_ids)
        num_classes = len(class_ids)
    
    print('Image shape :',image.shape,' class_ids:', class_ids)
    print(' heatmap shape: ', heatmaps.shape, 'boxes shape: ',boxes.shape)    
    scale  = config.HEATMAP_SCALE_FACTOR
    
    # print('heatmaps shape:', heatmaps.shape, 'bboxes shape :', boxes.shape)
    # print('1- image    : ', image.shape, image.dtype, np.min(image), np.max(image))
    ## add back pixel mean values
    image      = utils.unmold_image(mrcnn_input_batch[0][image_id], config)
    # print('2- image    : ', image.shape, image.dtype, np.min(image), np.max(image))
    # display_image(image)

    ## convert from resized to original size 
    image = utils.unresize_image(image, image_meta)
    # print('3- image    : ', image.shape, image.dtype, np.min(image), np.max(image))    
    display_image(image)
     
    ## Convert to grayscale np array   
    image_bw = Image.fromarray(image).convert(mode='L')
    image_bw = np.asarray(image_bw)
    
    # print('5- image_bw: ', image_bw.shape, image_bw.dtype, np.min(image_bw), np.max(image_bw))    
    # display_image(image_bw, cmap=plt.cm.gray)


    columns = min(columns, num_classes)
    rows   = math.ceil(num_classes/columns)
    width  = size[0] * columns
    height = size[1] * rows
    fig = plt.figure(figsize=(width, height))
    
    if num_bboxes == 999 :
        num_bboxes  = boxes.shape[1]  

    if num_bboxes > 0:
        boxes = utils.boxes_to_image_domain(boxes[...,:4], image_meta)
        x1    = boxes[:,:,1] # // scale
        x2    = boxes[:,:,3] # // scale
        y1    = boxes[:,:,0] # // scale
        y2    = boxes[:,:,2] # // scale
        box_w = x2 - x1   # x2 - x1
        box_h = y2 - y1 
        # cx    = (x1 + ( box_w / 2.0)).astype(int)
        # cy    = (y1 + ( box_h / 2.0)).astype(int)
        # print('x1, x2...shapes:', x1.shape, x2.shape, y1.shape, y2.shape, box_h.shape, box_w.shape)    
        
    colors = random_colors(num_classes)
    style = "dotted"
    alpha = 1
    linewidth = 1.0
    color = (0.5, 0.0, 1.0)

    for idx, cls in enumerate(class_ids):
        # color = colors[idx]
        row = idx // columns
        col = idx  % columns
        subplot = (row * columns) + col +1
        # print('idx ', idx,  ' class:', cls, 'row:', row,'col:', col, 'subplot: ', subplot)
        if class_names is None:
            ttl = 'Image: {:2d} Cls: {:2d} '.format( image_id,cls)
        else:
            ttl = 'Image: {:2d} Cls: {:2d}/{:s}'.format(image_id, cls, class_names[cls])

        ax = fig.add_subplot(rows, columns, subplot)
        ax.set_title(ttl, fontsize=14)
        ax.tick_params(axis='both', labelsize = 5)
        ax.tick_params(direction='out', length=6, width=1, colors='r', labelsize = 10)
        ax.set_xlabel(' X axis', fontsize=10)
        ax.set_ylabel(' Y axis', fontsize=10)
    
        ## heatmaps here are already in the range [0,1.0]
        unmolded_heatmap = utils.unresize_heatmap(heatmaps[:,:,cls],image_meta, upscale = config.HEATMAP_SCALE_FACTOR)
        
        ax.imshow(image_bw , cmap=plt.cm.gray)
        ax.imshow(unmolded_heatmap, alpha = 0.6,cmap=cm.YlOrRd)              
        
        for bbox in range(num_bboxes):
            p = patches.Rectangle( (x1[cls,bbox],y1[cls,bbox]), box_w[cls,bbox], box_h[cls,bbox], 
                               linewidth= linewidth, alpha=alpha, linestyle=style, edgecolor=color, facecolor='none')
            ax.add_patch(p)
        
        plt.subplots_adjust(top=0.94, bottom=0.02, left=0.02, right=0.98, hspace=0.10, wspace=0.10)                
    
    fig.suptitle(title, fontsize = 15, ha ='center' )
    plt.show()
    
    # plt.close()
    return

    

##----------------------------------------------------------------------
## Display heatmaps overlaying image - separate entry for 'heatmap'
##----------------------------------------------------------------------     
def display_heatmaps_mrcnn_fcn(mrcnn_input_batch, mrcnn_output_batch, image_id, heatmap, hm = 'pr' ,  
                      class_ids = None, class_names = None, num_bboxes = 999,
                      size = (7,7), columns = 2, config = None, scaling = 'clip') :
    '''
    input
    -----
        Z             Gaussian distribution (Batch Sz, Class_sz, Img Height, Img Width)
        boxes         array of bounding boxes 
        scaling       'all'    : Normalize to 1 over all classes
                      'class'  : Normalize to 1 over each class indiviually 
                      None     : No Scaling - clip to [-1 , 1]
                      
        hm            'pr'     : use predicted bouding boxes 
                      'gt'       use ground truth bounding boxes
                      
    '''
    ## image is in molded format - unmold it:
 
    scaling    = scaling.lower()
    image      = mrcnn_input_batch[0][image_id]
    image_meta = mrcnn_input_batch[1][image_id]
    Z          = heatmap[image_id]
    
    if hm == 'pr':
        # boxes =  fcn_input_batch[2][image_id]        
        boxes =  mrcnn_output_batch[1][image_id]        
        title = 'FCN Produced Heatmaps - (w/ Predicted Bounding Boxes) '         
    elif hm == 'gt':
        # boxes =  fcn_input_batch[4][image_id]        
        boxes =  mrcnn_output_batch[3][image_id]        
        title = 'FCN Produced Heatmaps - (w/ Ground Truth Bounding Boxes) '         
        
    scale = config.HEATMAP_SCALE_FACTOR
    print(' Bounding boxes shape: ', boxes.shape)
    if class_ids is None :
        num_classes = Z.shape[-1]
        class_ids   = np.arange(num_classes)
    else:
        num_classes = len(class_ids)
    
    print('Image shape :',image.shape,' class_ids:', class_ids)
    
    # image here is float32 between -127 and +128
    # print('1- image    : ', image.shape, image.dtype, np.min(image), np.max(image))   
    
    ## add back pixel mean values - uint8 between 0, 255
    image      = utils.unmold_image(mrcnn_input_batch[0][image_id], config)
    # print('2- image    : ', image.shape, image.dtype, np.min(image), np.max(image))
    # display_image(image)

    ## convert from resized to original size 
    image = utils.unresize_image(image, image_meta)
    # print('3- image    : ', image.shape, image.dtype, np.min(image), np.max(image))    
    display_image(image)
     
    # image = skimage.util.img_as_ubyte(image)
    ## Convert to grayscale np array   
    image_bw = Image.fromarray(image).convert(mode='L')
    image_bw = np.asarray(image_bw)
    
    # print('4- image_bw: ', image_bw.shape, image_bw.dtype, np.min(image_bw), np.max(image_bw))    
    # display_image(image_bw, cmap=plt.cm.gray)

    columns = min(columns, num_classes)
    rows   = math.ceil(num_classes/columns)
    width  = size[0] * columns
    height = size[1] * rows
    fig = plt.figure(figsize=(width, height))
    
    if num_bboxes == 999 :
        num_bboxes  = boxes.shape[2]  

    if num_bboxes > 0:
        boxes = utils.boxes_to_image_domain(boxes[...,:4], image_meta)
        x1    = boxes[:,:,1] 
        x2    = boxes[:,:,3] 
        y1    = boxes[:,:,0] 
        y2    = boxes[:,:,2] 
        box_w = x2 - x1    
        box_h = y2 - y1 
        # print('x1, x2...shapes:', x1.shape, x2.shape, y1.shape, y2.shape, box_h.shape, box_w.shape)    

    
        
    colors = random_colors(num_classes)
    style = "dotted"
    linewidth = 1.0
    alpha = 1
    color = (0.5, 0.0, 1.0)
    
    min_z_all = np.amin(Z)
    max_z_all = np.amax(Z)
    min_z_cls = np.amin(Z, axis=(0,1), keepdims= True)
    max_z_cls = np.amax(Z, axis=(0,1), keepdims= True)
    avg_z_cls = np.mean(Z, axis=(0,1), keepdims= True)
    print(' min_z_all: ',  min_z_all.shape, min_z_all , ' max_z_all :', max_z_all.shape, max_z_all)
    # print(' min_z_cls: ',  min_z_cls.shape,  ' max_z_cls :',max_z_cls.shape)
    
    if  scaling == 'clip':
        print(' SCALING == none (clip to [-1, +1])')    
        title += ' - Clip output to [-1, +1]'
        YY = np.clip(Z, -1.0, 1.0) 
    elif scaling == 'all':
        print(' SCALING == all')
        title += ' - NORMALIZED to 1 Over all classes'
        YY = (Z - min_z_all)/(max_z_all - min_z_all + 1.0e-9)
    elif scaling == 'class':
        print(' SCALING == class')
        title += ' - NORMALIZED to 1 in each class'
        YY = (Z - min_z_cls)/(max_z_cls - min_z_cls + 1.0e-9)            
    else: 
        print(" ERROR - scaling must be 'all', 'class', or 'clip' ")
        return

    for idx, cls in enumerate(class_ids):
        # color = colors[idx]
        row = idx // columns
        col = idx  % columns
        subplot = (row * columns) + col +1
        # print('idx ', idx,  ' class:', cls, 'row:', row,'col:', col, 'subplot: ', subplot)
        if class_names is None:
            ttl = 'Cls: {:2d} '.format(cls)
        else:      
            ttl = 'Cls: {:2d}/{:s}'.format(cls, class_names[cls])

        ax = fig.add_subplot(rows, columns, subplot)
        ax.tick_params(axis='both', labelsize = 5)
        ax.tick_params(direction='out', length=6, width=1, colors='r', labelsize = 10)
        ax.set_xlabel(' X axis', fontsize=10)
        ax.set_ylabel(' Y axis', fontsize=10)
        ax.imshow(image_bw , cmap=plt.cm.gray)
        
        if scaling == 'all':
            vmin = 0
            vmax = 1
        elif scaling == 'class':
            vmin = 0
            vmax = 1
        elif scaling == 'clip':
            vmin = np.amin(YY[:,:,cls])
            vmax = np.amax(YY[:,:,cls])

        ttl += ' - min: {:6.5f}  max: {:6.5f}  avg: {:6.5f}'.format(min_z_cls[0,0,cls], max_z_cls[0,0,cls], avg_z_cls[0,0,cls])
        print(' Title: ', ttl)

        unmolded_heatmap = utils.unresize_heatmap(YY[:,:,cls],image_meta, upscale = config.HEATMAP_SCALE_FACTOR)
        # print(' unmolded_heatmap: shape:', unmolded_heatmap.shape, unmolded_heatmap.dtype, np.amin(unmolded_heatmap), np.amax(unmolded_heatmap))
        surf = ax.imshow(unmolded_heatmap, alpha = 0.6, cmap=cm.YlOrRd, vmin = vmin, vmax = vmax )              
        
        for bbox in range(num_bboxes):
            p = patches.Rectangle( (x1[cls,bbox],y1[cls,bbox]), box_w[cls,bbox], box_h[cls,bbox], 
                               linewidth=linewidth, alpha=alpha, linestyle=style, edgecolor=color, facecolor='none')
            ax.add_patch(p)
        
        ax.set_title(ttl, fontsize=12)
        plt.subplots_adjust(top=0.93, bottom=0.02, left=0.02, right=0.98, hspace=0.1, wspace=0.10)   
        fig.colorbar(surf, shrink=0.6, aspect=30, fraction=0.05)
        

    fig.suptitle(title, fontsize = 18, ha ='center' )
    plt.show()
    
    # plt.close()
    return

    
    
##----------------------------------------------------------------------
## Display heatmaps overlaying image - separate entry for 'heatmap'
##----------------------------------------------------------------------     
def display_heatmaps_compare(mrcnn_input_batch, mrcnn_output_batch,  heatmap, image_id = 0, hm = 'pr' ,  
                      class_ids = None, class_names = None, num_bboxes = 999,
                      size = (7,7), columns = 2, config = None, scaling = 'clip') :
    '''
    input
    -----
        Z             Gaussian distribution (Batch Sz, Class_sz, Img Height, Img Width)
        boxes         array of bounding boxes 
        scaling       'all'    : Normalize to 1 over all classes
                      'class'  : Normalize to 1 over each class indiviually 
                      None     : No Scaling - clip to [-1 , 1]
                      
        hm            'pr'     : use predicted bouding boxes 
                      'gt'       use ground truth bounding boxes
                      
    '''
    ## image is in molded format - unmold it:
 
    scaling    = scaling.lower()
    image      = mrcnn_input_batch[0][image_id]
    image_meta = mrcnn_input_batch[1][image_id]
    
    if hm == 'pr':
        Z1    =  mrcnn_output_batch[0][image_id]        
        boxes =  mrcnn_output_batch[1][image_id]        
        title = 'MRCNN Predicted vs. FCN Heatmaps '         
    elif hm == 'gt':
        # boxes =  fcn_input_batch[4][image_id]        
        Z1    =  mrcnn_output_batch[2][image_id]        
        boxes =  mrcnn_output_batch[3][image_id]        
        title = ' MRCNN Ground Truth vs. FCN Heatmaps '         
    Z2 = heatmap[image_id]
    assert Z1.shape == Z2.shape
    
    scale = config.HEATMAP_SCALE_FACTOR

    print(' Bounding boxes shape: ', boxes.shape)
    if class_ids is None :
        num_classes = Z1.shape[-1]
        class_ids   = np.arange(num_classes)
    else:
        num_classes = len(class_ids)
    
    print('Image shape :',image.shape,' class_ids:', class_ids)
    
    ## 0 - image here is float32 between -127 and +128
    ## 1 - add back pixel mean values -> uint8 between 0, 255
    ## 2 - convert from resized to original size 
    ## 3 - Convert to grayscale np array   
    # print('0 - image    : ', image.shape, image.dtype, np.min(image), np.max(image))   
    image      = utils.unmold_image(mrcnn_input_batch[0][image_id], config)
    # print('1- image    : ', image.shape, image.dtype, np.min(image), np.max(image))
    image = utils.unresize_image(image, image_meta)
    # print('2- image    : ', image.shape, image.dtype, np.min(image), np.max(image))    
    display_image(image)
    # image = skimage.util.img_as_ubyte(image)
    image_bw = Image.fromarray(image).convert(mode='L')
    image_bw = np.asarray(image_bw)
    # print('3- image_bw: ', image_bw.shape, image_bw.dtype, np.min(image_bw), np.max(image_bw))    
    # display_image(image_bw, cmap=plt.cm.gray)

    rows        = num_classes 
    columns     = 2
    width  = size[0] * columns
    height = size[1] * num_classes
    fig = plt.figure(figsize=(width, height))
    
    if num_bboxes == 999 :
        num_bboxes  = boxes.shape[2]  

    if num_bboxes > 0:
        boxes = utils.boxes_to_image_domain(boxes[...,:4], image_meta)
        x1    = boxes[:,:,1] 
        x2    = boxes[:,:,3] 
        y1    = boxes[:,:,0] 
        y2    = boxes[:,:,2] 
        box_w = x2 - x1    
        box_h = y2 - y1 
        # print('x1, x2...shapes:', x1.shape, x2.shape, y1.shape, y2.shape, box_h.shape, box_w.shape)    
        
    colors = random_colors(num_classes)
    style = "dotted"
    linewidth = 1.0
    alpha = 1
    color = (0.5, 0.0, 1.0)
    
    min_z1_all = np.amin(Z1)
    min_z2_all = np.amin(Z2)    
    max_z1_all = np.amax(Z1)
    max_z2_all = np.amax(Z2)
    min_z_all  = np.minimum(min_z1_all, min_z2_all)
    max_z_all  = np.maximum(max_z1_all, max_z2_all)
    
    min_z1_cls = np.amin(Z1, axis = (0,1), keepdims = True)
    max_z1_cls = np.amax(Z1, axis = (0,1), keepdims = True)
    min_z2_cls = np.amin(Z2, axis = (0,1), keepdims = True)
    max_z2_cls = np.amax(Z2, axis = (0,1), keepdims = True)
    avg_z1_cls = np.mean(Z1, axis = (0,1), keepdims = True)
    avg_z2_cls = np.mean(Z2, axis = (0,1), keepdims = True)
    
    min_z_cls  = np.minimum(min_z1_cls, min_z2_cls)
    max_z_cls  = np.maximum(max_z1_cls, max_z2_cls)
    
    # print(' min_z1_all shape:', min_z1_all.shape,' min_z1_all:', min_z1_all,' max_z1_all:', max_z1_all.shape,'max_z1_all:', max_z1_all)
    # print(' min_z2_all shape:', min_z2_all.shape,' min_z2_all:', min_z2_all,' max_z2_all:', max_z2_all.shape,'max_z2_all:', max_z2_all)
    # print(' min_z_all shape:', min_z_all.shape,' min_z_all:', min_z_all,' max_z_all:', max_z_all.shape,'max_z_all:', max_z_all)
    # print(' min_z1_cls shape:', min_z1_cls.shape,' max_z1_cls shape:', max_z1_cls.shape)        
    # print(' min_z2_cls shape:', min_z2_cls.shape,' max_z2_cls shape:', max_z2_cls.shape)    
    # print(' min_z_cls shape:', min_z_cls.shape,' max_z_cls shape:', max_z_cls.shape)    



    if scaling == 'all':   
        Z1 = (Z1 - min_z_all)/(max_z_all - min_z_all + 1.0e-9)
        Z2 = (Z2 - min_z_all)/(max_z_all - min_z_all + 1.0e-9)
        title += ' - NORMALIZED to [0, 1] over all classes (jointly)'
        zlim = 'one'
    elif scaling == 'class':
        Z1 = (Z1 - min_z_cls)/(max_z_cls - min_z_cls + 1.0e-9)
        Z2 = (Z2 - min_z_cls)/(max_z_cls - min_z_cls + 1.0e-9)
        title += ' - NORMALIZED to [0, 1] over each class (jointly) '
    elif scaling == 'each':
        Z1 = (Z1 - min_z1_cls)/(max_z1_cls - min_z1_cls + 1.0e-9)
        Z2 = (Z2 - min_z2_cls)/(max_z2_cls - min_z2_cls + 1.0e-9)
        title += ' - NORMALIZED to [0, 1] over each class (separately)'
    elif scaling == 'clip':    
        print(' SCALING == clip (clip to [-1, +1])')    
        Z1 = np.clip(Z1, -1.0, 1.0) 
        Z2 = np.clip(Z2, -1.0, 1.0) 
        title += ' - Clip output to [-1, +1]'
    else: 
        print(" ERROR - scaling must be 'all', 'class', 'each' , or  'clip' : ", scaling)
        return        

    
    for idx, cls in enumerate(class_ids):
        # color = colors[idx]
        row = idx // columns
        col = idx  % columns
        
        ## Plot 1st columm 
        ##--------------------
        subplot = (idx * columns) + 1
        if class_names is None:
            ttl = 'Img:{:2d} Cls: {:2d} '.format( image_id,cls)
        else:
            ttl = 'Img:{:2d} Cls: {:2d}/{:s}'.format(image_id, cls, class_names[cls])
        # print('idx ', idx,  ' class:', cls, 'row:', row,'col:', col, 'subplot: ', subplot)
        ax = fig.add_subplot(rows, columns, subplot)
        ax.tick_params(axis='both', labelsize = 5)
        ax.tick_params(direction='out', length=6, width=1, colors='r', labelsize = 10)
        ax.set_xlabel(' X axis', fontsize=10)
        ax.set_ylabel(' Y axis', fontsize=10)
        ax.imshow(image_bw , cmap=plt.cm.gray)
        
        if scaling == 'clip':
            vmin = min_z1_cls[0,0,cls]
            vmax = max_z1_cls[0,0,cls]
        else:
            vmin = 0
            vmax = 1

        ttl = ttl +' - min: {:6.5f}  max: {:6.5f}  avg: {:6.5f}'.format(min_z1_cls[0,0,cls], max_z1_cls[0,0,cls], avg_z1_cls[0,0,cls])
        unmolded_heatmap = utils.unresize_heatmap(Z1[:,:,cls],image_meta, upscale = config.HEATMAP_SCALE_FACTOR)
        # print(' unmolded_heatmap: shape:', unmolded_heatmap.shape, unmolded_heatmap.dtype, np.amin(unmolded_heatmap), np.amax(unmolded_heatmap))
        surf = ax.imshow(unmolded_heatmap, alpha = 0.6, cmap=cm.YlOrRd, vmin = vmin, vmax = vmax )              
        
        for bbox in range(num_bboxes):
            p = patches.Rectangle( (x1[cls,bbox],y1[cls,bbox]), box_w[cls,bbox], box_h[cls,bbox], 
                               linewidth=linewidth, alpha=alpha, linestyle=style, edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # adjustments and colorbar
        #-------------------------
        plt.subplots_adjust(top=0.94, bottom=0.02, left=0.02, right=0.98, hspace=0.25, wspace=0.15)      
        fig.colorbar(surf, shrink=0.8, aspect=30, fraction=0.05)
        ax.set_title(ttl, fontsize=12)

        ## Plot 2nd columm 
        ##--------------------
        subplot = (idx * columns) + 2
        if class_names is None:
            ttl = 'FCN Cls: {:2d} '.format(cls)
        else:         
            ttl = 'FCN Cls: {:2d}-{:s}'.format(cls, class_names[cls])
        ax = fig.add_subplot(rows, columns, subplot)
        ax.tick_params(axis='both', labelsize = 5)
        ax.tick_params(direction='out', length=6, width=1, colors='r', labelsize = 10)
        ax.set_xlabel(' X axis', fontsize=10)
        ax.set_ylabel(' Y axis', fontsize=10)
        ax.imshow(image_bw , cmap=plt.cm.gray)

        if scaling == 'clip':
            vmin = min_z2_cls[0,0,cls]
            vmax = max_z2_cls[0,0,cls]
        else:
            vmin = 0
            vmax = 1

        ttl = ttl +' - min: {:6.5f}  max: {:6.5f}  avg: {:6.5f}'.format(min_z2_cls[0,0,cls], max_z2_cls[0,0,cls], avg_z2_cls[0,0,cls])
        # print(' Title: ', ttl)
        unmolded_heatmap = utils.unresize_heatmap(Z2[:,:,cls],image_meta, upscale = config.HEATMAP_SCALE_FACTOR)
        # print(' unmolded_heatmap: shape:', unmolded_heatmap.shape, unmolded_heatmap.dtype, np.amin(unmolded_heatmap), np.amax(unmolded_heatmap))
        surf = ax.imshow(unmolded_heatmap, alpha = 0.6, cmap=cm.YlOrRd, vmin = vmin, vmax = vmax )              
        
        for bbox in range(num_bboxes):
            p = patches.Rectangle( (x1[cls,bbox],y1[cls,bbox]), box_w[cls,bbox], box_h[cls,bbox], 
                               linewidth=linewidth, alpha=alpha, linestyle=style, edgecolor=color, facecolor='none')
            ax.add_patch(p)
        ax.set_title(ttl, fontsize=12)

        # adjustments and colorbar
        #-------------------------
        plt.subplots_adjust(top=0.94, bottom=0.02, left=0.02, right=0.98, hspace=0.25, wspace=0.15)      
        fig.colorbar(surf, shrink=0.8, aspect=30, fraction=0.05)
        
        # plt.subplots_adjust(top=0.93, bottom=0.02, left=0.02, right=0.98, hspace=0.1, wspace=0.10)   
        # fig.colorbar(surf, shrink=0.6, aspect=30, fraction=0.05)
        

    fig.suptitle(title, fontsize = 16, ha ='center' )
    plt.show()
    
    return




##----------------------------------------------------------------------
## plot 2d heatmap for one image with bboxes
##----------------------------------------------------------------------        
def plot_2d_heatmap( Z, boxes, image_idx, class_ids = None,  
                     columns = 2, size = (7,7), num_bboxes = 999,
                     title = '2d heatmap w/ bboxes', class_names=None, scale = 1, scaling = 'none'):

    '''
    Z:             Gaussian heatmap [ BatchSize, height, width, Num_classes]
    image_idx :    index to image 
    class_ids :    Lists of class ids to display
    '''
    scaling    = scaling.lower()
    if Z.ndim == 4:
        Z = Z[image_idx]
        boxes = boxes[image_idx]

    if class_ids is None :
        print(' Image Id: ', image_idx , ' Display all classes...')
        num_classes = Z.shape[-1]
        class_ids   = np.arange(num_classes)
    else:
        print(' Image Id: ', image_idx , ' Display classes:', class_ids)
        num_classes = len(class_ids)
    class_ids.sort()
    
    rows   = math.ceil(num_classes/columns)
    print(' rows  ',rows, ' columns :', columns, 'boxes.shape : ',boxes.shape)
    # Z = Z[image_idx]
    # boxes = boxes[image_idx]
    
    image_height = Z.shape[0]
    image_width  = Z.shape[1]
    
    # print(image_height, image_width)
    Y = np.arange(0, image_height, 1)
    X = np.arange(0, image_width, 1)
    X, Y = np.meshgrid(X, Y)        

    if num_bboxes == 999 :
        num_bboxes  = boxes.shape[2]  
    
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
    print(' min_z_all shape:', min_z_all.shape,' min_z_all:', min_z_all,' max_z_all:', max_z_all.shape,'max_z_all:', max_z_all)
    print(' min_z_cls shape:', min_z_cls.shape,' max_z_cls shape:', max_z_cls.shape)
    
    # if scaling :
        # Z = (Z - min_z)/(max_z - min_z + 1.0e-6)
        # title += ' - NORMALIZED to 1'
        # zlim = 'one'
        
    if  scaling == 'clip':
        print(' SCALING == clip to [-1, +1])')    
        title += ' - Clip output to [-1, +1]'
        YY = np.clip(Z, -1.0, 1.0) 
    elif scaling == 'all':
        print(' SCALING == all')
        title += ' - NORMALIZED to 1 Over all classes'
        YY = (Z - min_z_all)/(max_z_all - min_z_all + 1.0e-9)
    elif scaling == 'class':
        print(' SCALING == class')
        title += ' - NORMALIZED to 1 in each class'
        YY = (Z - min_z_cls)/(max_z_cls - min_z_cls + 1.0e-9)            
    elif scaling == 'none':
        print(' SCALING == none ')
        title += ' - No Normalization'
        YY = Z
    else: 
        print(" ERROR - scaling must be 'all', 'class', 'clip', or 'none' ")
        return
        
    width  = size[0] * columns
    height = size[1] * rows
    fig = plt.figure(figsize=(width, height))

    colors = random_colors(num_classes)
    style = "dotted"
    alpha = 1
    color = colors[0]
    # color = (0.5, 0.0, 1.0)
    color = 'xkcd:neon green'

    for idx, cls in enumerate(class_ids):
        # color = colors[idx]
        row = idx // columns
        col = idx  % columns
        subplot = (row * columns) + col +1
        # print('idx ', idx,  ' class:', cls, 'row:', row,'col:', col, 'subplot: ', subplot, 'cls:', cls)
        if class_names is None:
            ttl = 'Image: {:2d} Cls: {:2d} '.format( image_idx,cls)
        else:
            ttl = 'Image: {:2d} Cls: {:2d}/{:s}'.format(image_idx, cls, class_names[cls])

        ttl += '  -  min: {:6.5f}  max: {:6.5f}'.format(min_z_cls[0,0,cls], max_z_cls[0,0,cls])
        
        ax = fig.add_subplot(rows, columns, subplot)
        ax.set_title(ttl, fontsize=12)
        ax.tick_params(axis='both', labelsize = 5)
        ax.set_ylim(0, image_height)
        ax.set_xlim(0, image_width )
        ax.set_xlabel(' X axis', fontsize=8)
        ax.set_ylabel(' Y axis', fontsize=8)
        ax.invert_yaxis()
        
        if scaling == 'clip':
            vmin = -1
            vmax = 1
        elif scaling in ['all','class']:            
            vmin = 0
            vmax = 1
        else:
            vmin = np.amin(YY[:,:,cls])
            vmax = np.amax(YY[:,:,cls])
            
        surf = plt.matshow(YY[:,:, cls], fignum = 0, cmap = cm.coolwarm,vmin = vmin, vmax = vmax )
        
        for bbox in range(num_bboxes):
            # print(num_bboxes, bbox, cls, boxes[image_idx, cls,bbox])
            # print(ttl,x1[cls,bbox], y1[cls,bbox],x2[cls,bbox],y2[cls,bbox])
            p = patches.Rectangle( (x1[cls,bbox],y1[cls,bbox]), box_w[cls,bbox], box_h[cls,bbox], 
                                   linewidth=1.5, alpha=alpha, linestyle=style, edgecolor=color, facecolor='none')
            ax.add_patch(p)
        
        plt.subplots_adjust(top=0.93, bottom=0.02, left=0.02, right=0.98, hspace=0.15, wspace=0.15)      
        fig.colorbar(surf, shrink=0.8, aspect=30, fraction=0.05)
        
        
    # plt.tight_layout()
    fig.suptitle(title, fontsize = 15, ha ='center' )
    plt.show()
    
    return fig 

plot_2d_heatmap_with_bboxes = plot_2d_heatmap


##----------------------------------------------------------------------
## plot 3D gauss_distribution for one image, for a list of classes
## 
## 19-09-2018 - previously named plot_gaussian
##----------------------------------------------------------------------       
def plot_3d_heatmap( Z, image_idx, class_ids = None,  columns = 2,
                      title = '3d heatmap', 
                      size = (8,8), class_names=None, zlim = 'all', scaling =  'none'):
    '''
    
    Z:             Gaussian heatmap [ BatchSize, height, width, Num_classes]
    image_idx :    index to image 
    class_ids :    Lists of class ids to display
    
    zlim      :    all  - normalize across all heatmaps
                   each - normalize each heatmap separately
                   one  - set zlim to [0, 1.005]
    '''
    scaling    = scaling.lower()

    if class_ids is None :
        print('Display all classes...')
        num_classes = Z.shape[-1]
        class_ids   = np.arange(num_classes)
    else:
        print('Display classes:', class_ids)
        num_classes = len(class_ids)
        
    rows   = math.ceil(num_classes/columns)    
    print('rows  ',rows, ' columns :', columns)
    # num_classes = len(class_ids)
    # rows        = num_classes 
    # columns     = 1
    Z = Z[image_idx]
    
    # fig.set_figheight(width-1)
    image_height = Z.shape[0]
    image_width  = Z.shape[1]
    
    Y = np.arange(0, image_height, 1)
    X = np.arange(0, image_width, 1)
    X, Y = np.meshgrid(X, Y)        

    pos = np.empty(X.shape+(2,))   # concatinate shape of x to make ( x.rows, x.cols, 2)
    pos[:,:,0] = X;
    pos[:,:,1] = Y;
    

    min_z_all = np.amin(Z)
    max_z_all = np.amax(Z)
    min_z_cls = np.amin(Z, axis = (0,1), keepdims = True)
    max_z_cls = np.amax(Z, axis = (0,1), keepdims = True)
    print(' min_z_all shape:', min_z_all.shape,' min_z_all:', min_z_all,' max_z_all:', max_z_all.shape,'max_z_all:', max_z_all)
    print(' min_z_cls shape:', min_z_cls.shape    ,' max_z_cls shape:', max_z_cls.shape)
    
    if  scaling == 'clip':
        print(' SCALING == clip to [-1, +1])')    
        title += ' - Clip output to [-1, +1]'
        YY = np.clip(Z, -1.0, 1.0) 
        zlim_min = -1.0
        zlim_max = +1.0 
        print('zlim = clip   zlim_min : {:10.8f} zlim_max: {:10.8f} '.format(zlim_min, zlim_max))
    elif scaling == 'all':
        print(' SCALING == all')
        title += ' - NORMALIZED to 1 Over all classes'
        YY = (Z - min_z_all)/(max_z_all - min_z_all + 1.0e-9)
        zlim_min = min_z_all 
        zlim_max = max_z_all
        print('zlim = all   zlim_min : {:10.8f} zlim_max: {:10.8f} '.format(zlim_min, zlim_max))
    elif scaling == 'class':
        print(' SCALING == class')
        title += ' - NORMALIZED to 1 in each class'
        YY = (Z - min_z_cls)/(max_z_cls - min_z_cls + 1.0e-9)            
        zlim_min = min_z_all 
        zlim_max = max_z_all
        print('zlim = class(== all)   zlim_min : {:10.8f} zlim_max: {:10.8f} '.format(zlim_min, zlim_max))
    elif scaling == 'none':
        print(' SCALING == none ')
        title += ' - No Normalization'
        YY = Z
    else: 
        print(" ERROR - scaling must be 'all', 'class', or 'clip'/ None ")
        return
        
        
    width  = size[0] * columns
    height = size[1] * rows
    fig = plt.figure(figsize=(width, height))
        
    for idx,cls in enumerate(class_ids):
        row = idx // columns
        col = idx  % columns
        subplot = (row * columns) + col +1
        # print('idx ', idx,  ' class:', cls, 'row:', row,'col:', col, 'subplot: ', subplot)
        # for col  in range(2):
        if class_names is None:
            ttl = 'Image:{:2d} Cls: {:2d} '.format( image_idx,cls)
        else:
            ttl = 'Image:{:2d} Cls: {:2d}-{:s}'.format(image_idx, cls, class_names[cls])

        # plt.subplot(rows, columns, col+1)
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(rows, columns, subplot, projection='3d')
        
        ## set z axis boundaries
        if scaling == 'none': 
            zlim_min = min_z_cls[0,0,cls] 
            zlim_max = max_z_cls[0,0,cls]
            print('1 zlim = each  zlim_min : {:10.8f} zlim_max: {:10.8f} '.format(min_z_cls[0,0,cls], max_z_cls[0,0,cls]))

        ttl += '  -  min: {:6.5f}  max: {:6.5f}'.format(np.amin(Z[:,:,cls]), np.amax(Z[:,:,cls]))
        
        ax.set_title(ttl, fontsize = 10, ha= 'center')            
        ax.set_zlim(zlim_min , zlim_max)            
        ax.tick_params(axis='both', labelsize = 6)
        ax.set_ylim(0, image_height + image_height //10)
        ax.set_xlim(0, image_width  + image_width  //10)
        ax.set_xlabel(' X axis', fontsize=8)
        ax.set_ylabel(' Y axis', fontsize=8)
        ax.invert_yaxis()
        ax.view_init( azim=-116,elev=40)            
        # ax.view_init(azim=-37, elev=43)            
        
        surf = ax.plot_surface(X, Y, YY[:,:,cls],cmap=cm.coolwarm, linewidth=0, antialiased=False)
        
        # Add a color bar which maps values to colors.
        plt.subplots_adjust(top=0.97, bottom=0.02, left=0.02, right=0.98, hspace=0.15, wspace=0.15)                
        cbar = fig.colorbar(surf, shrink=0.8, aspect=30, fraction=0.10)
        cbar.ax.tick_params(labelsize=9) 

    fig.suptitle(title, fontsize = 14 )
    plt.show()
    return
    


##----------------------------------------------------------------------
## comparative 2D plot of two gauss_distributions for one image, for a list of classes
##----------------------------------------------------------------------       
def plot_2d_heatmap_compare( Z1, Z2, boxes, image_idx, class_ids = None,   
                             size = (7,7), num_bboxes = 0, class_names=None, scale = 1, scaling = 'none', 
                             title = '2D Comparison between 2d heatmaps w/ bboxes'):
    '''    
    Z:             Gaussian heatmap [ BatchSize, height, width, Num_classes]
    image_idx :    index to image 
    class_ids :    Lists of class ids to display
    '''
    assert Z1.shape == Z2.shape
    if class_ids is None :
        print('Display all classes...')
        num_classes = Z1.shape[-1]
        class_ids   = np.arange(num_classes)
    else:
        print('Display classes:', class_ids)
        num_classes = len(class_ids)
    
    
    Z1 = Z1[image_idx]
    Z2 = Z2[image_idx]
    boxes = boxes[image_idx]
    image_height = Z1.shape[0]
    image_width  = Z1.shape[1]

    ## if num_bbox not specified, display all bboxes
    if num_bboxes == 0 :
        num_bboxes  = boxes.shape[2]  

    # print(image_height, image_width)
    Y = np.arange(0, image_height, 1)
    X = np.arange(0, image_width, 1)
    X, Y = np.meshgrid(X, Y)        
    
    x1    = boxes[:,:,1] // scale
    x2    = boxes[:,:,3] // scale
    y1    = boxes[:,:,0] // scale
    y2    = boxes[:,:,2] // scale
    box_w = x2 - x1   # x2 - x1
    box_h = y2 - y1 
    
    min_z1_all = np.amin(Z1)
    min_z2_all = np.amin(Z2)    
    max_z1_all = np.amax(Z1)
    max_z2_all = np.amax(Z2)
    min_z_all  = np.minimum(min_z1_all, min_z2_all)
    max_z_all  = np.maximum(max_z1_all, max_z2_all)
    
    min_z1_cls = np.amin(Z1, axis = (0,1), keepdims = True)
    max_z1_cls = np.amax(Z1, axis = (0,1), keepdims = True)
    min_z2_cls = np.amin(Z2, axis = (0,1), keepdims = True)
    max_z2_cls = np.amax(Z2, axis = (0,1), keepdims = True)
    
    min_z_cls  = np.minimum(min_z1_cls, min_z2_cls)
    max_z_cls  = np.maximum(max_z1_cls, max_z2_cls)
    
    print(' min_z1_all shape:', min_z1_all.shape,' min_z1_all:', min_z1_all,' max_z1_all:', max_z1_all.shape,'max_z1_all:', max_z1_all)
    print(' min_z2_all shape:', min_z2_all.shape,' min_z2_all:', min_z2_all,' max_z2_all:', max_z2_all.shape,'max_z2_all:', max_z2_all)
    # print(' min_z_all shape:', min_z_all.shape,' min_z_all:', min_z_all,' max_z_all:', max_z_all.shape,'max_z_all:', max_z_all)
    # print(' min_z1_cls shape:', min_z1_cls.shape,' max_z1_cls shape:', max_z1_cls.shape)        
    # print(' min_z2_cls shape:', min_z2_cls.shape,' max_z2_cls shape:', max_z2_cls.shape)    
    # print(' min_z_cls shape:', min_z_cls.shape,' max_z_cls shape:', max_z_cls.shape)    
        
    if scaling == 'all':
        Z1 = (Z1 - min_z_all)/(max_z_all - min_z_all + 1.0e-9)
        Z2 = (Z2 - min_z_all)/(max_z_all - min_z_all + 1.0e-9)
        title += ' - NORMALIZED to 1 over all classes (jointly)'
    elif scaling == 'class':
        Z1 = (Z1 - min_z_cls)/(max_z_cls - min_z_cls + 1.0e-9)
        Z2 = (Z2 - min_z_cls)/(max_z_cls - min_z_cls + 1.0e-9)
        title += ' - NORMALIZED to 1 over each classes (jointly) '
    elif scaling == 'each':
        Z1 = (Z1 - min_z1_cls)/(max_z1_cls - min_z1_cls + 1.0e-9)
        Z2 = (Z2 - min_z2_cls)/(max_z2_cls - min_z2_cls + 1.0e-9)
        title += ' - NORMALIZED to 1 over each classes (separately)'
    elif scaling == 'none':    
        title += ' - No Normalization'
    else: 
        print(" ERROR - scaling must be 'all', 'class', 'each' , or  'none' : ", scaling)
        return        
    
    if scaling :
        Z1 = (Z1 - min_z1_cls)/(max_z1_cls - min_z1_cls + 1.0e-6)
        Z2 = (Z2 - min_z2_cls)/(max_z2_cls - min_z2_cls + 1.0e-6)
        title += ' - NORMALIZED to 1'
        zlim = 'one'

    rows    = num_classes 
    columns = 2    
    width   = size[0] * columns
    height  = size[1] * num_classes
    fig     = plt.figure(figsize=(width, height))
   
    colors = random_colors(num_classes)
    style = "dotted"
    alpha = 1
    color = colors[0]
    # color = (0.5, 0.0, 1.0)
    color = 'xkcd:neon green'

    for idx, cls in enumerate(class_ids):
        # color = colors[idx]
        row = cls // columns
        col = cls  % columns
        # print('Image: ', img, 'class:', cls, 'row:', row,'col:', col)
        if class_names is None:
            ttl = 'HM1 Img: {:2d} Cls: {:2d} '.format( image_idx,cls)
        else:
            ttl = 'HM1 Img: {:2d} Cls: {:2d}/{:s}'.format(image_idx, cls, class_names[cls])

        ## Plot 1st columm 
        ##--------------------
        subplot = (idx * columns) + 1
        ax = fig.add_subplot(rows, columns, subplot)
        ax.set_title(ttl, fontsize=9)
        ax.tick_params(axis='both', labelsize = 5)
        ax.set_ylim(0, image_height)
        ax.set_xlim(0, image_width )
        ax.set_xlabel(' X axis', fontsize=8)
        ax.set_ylabel(' Y axis', fontsize=8)
        ax.invert_yaxis()
        surf = plt.matshow(Z1[:,:, cls], fignum = 0, cmap = cm.coolwarm)

        for bbox in range(num_bboxes):
            # print(num_bboxes, bbox, cls, boxes[image_idx, cls,bbox])
            # print(ttl,x1[cls,bbox], y1[cls,bbox],x2[cls,bbox],y2[cls,bbox])
            p = patches.Rectangle( (x1[cls,bbox],y1[cls,bbox]), box_w[cls,bbox], box_h[cls,bbox], 
                                   linewidth=1, alpha=alpha, linestyle=style, edgecolor=color, facecolor='none')
            ax.add_patch(p)

        fig.colorbar(surf, shrink=0.8, aspect=30, fraction=0.05)

        
        ## Plot 2nd columm 
        ##--------------------
        if class_names is None:
            ttl = 'HM2 Cls: {:2d} '.format(cls)
        else:          
            ttl = 'HM2 Cls: {:2d}/{:s}'.format(cls, class_names[cls])
        subplot = (idx * columns) + 2
        ax = fig.add_subplot(rows, columns, subplot)
        
        ttl += ' - min: {:6.5f}  max: {:6.5f}'.format(min_z2_cls[0,0,cls], max_z2_cls[0,0,cls])
        
        ax.set_title(ttl, fontsize=9)
        surf = plt.matshow(Z2[:,:, cls], fignum = 0, cmap = cm.coolwarm)
        for bbox in range(num_bboxes):
            p = patches.Rectangle( (x1[cls,bbox],y1[cls,bbox]), box_w[cls,bbox], box_h[cls,bbox], 
                                   linewidth=1, alpha=alpha, linestyle=style, edgecolor=color, facecolor='none')
            ax.add_patch(p)
        plt.subplots_adjust(top=0.94, bottom=0.02, left=0.02, right=0.98, hspace=0.25, wspace=0.15)      
        fig.colorbar(surf, shrink=0.8, aspect=30, fraction=0.05)
        
    # plt.subplots_adjust(top=0.90, bottom=0.02, left=0.02, right=0.98, hspace=0.10, wspace=0.10)      
    # plt.tight_layout()
    fig.suptitle(title, fontsize = 10 )
    plt.show()
    # plt.savefig('sample.png')
    
    return fig 


    
##----------------------------------------------------------------------
## comparative 3D plot of two gauss_distributions for one image, for a list of classes
##----------------------------------------------------------------------       
def plot_3d_heatmap_compare( Z1, Z2, image_idx, class_ids = None, 
                             title = '3d heatmap comparison',
                             size = (8,8), class_names=None, zlim = 'all' , scaling = 'none'):
    '''
    
    Z:             Gaussian heatmap [ BatchSize, height, width, Num_classes]
    image_idx :    index to image 
    class_ids :    Lists of class ids to display
    '''
    assert Z1.shape == Z2.shape
    scaling    = scaling.lower()
    if Z1.ndim == 4:
        Z1 = Z1[image_idx]
        Z2 = Z2[image_idx]
    
    if class_ids is None :
        print('Display all classes...')
        num_classes = Z1.shape[-1]
        class_ids   = np.arange(num_classes)
    else:
        print('Display classes:', class_ids)
        num_classes = len(class_ids)    

    
    # fig.set_figheight(width-1)
    image_height = Z1.shape[0]
    image_width  = Z1.shape[1]
    print(' image height/width ',image_height, image_width)
    Y = np.arange(0, image_height, 1)
    X = np.arange(0, image_width, 1)
    X, Y = np.meshgrid(X, Y)        
    pos = np.empty(X.shape+(2,))   # concatinate shape of x to make ( x.rows, x.cols, 2)
    pos[:,:,0] = X;
    pos[:,:,1] = Y;
    
    print('shape pos ', pos.shape, 'shape Z1: ' ,Z1.shape, 'shape Z2:', Z2.shape)
    
    min_z1_all = np.amin(Z1)
    min_z2_all = np.amin(Z2)    
    max_z1_all = np.amax(Z1)
    max_z2_all = np.amax(Z2)
    min_z_all  = np.minimum(min_z1_all, min_z2_all)
    max_z_all  = np.maximum(max_z1_all, max_z2_all)
    
    min_z1_cls = np.amin(Z1, axis = (0,1), keepdims = True)
    max_z1_cls = np.amax(Z1, axis = (0,1), keepdims = True)
    min_z2_cls = np.amin(Z2, axis = (0,1), keepdims = True)
    max_z2_cls = np.amax(Z2, axis = (0,1), keepdims = True)
    
    min_z_cls  = np.minimum(min_z1_cls, min_z2_cls)
    max_z_cls  = np.maximum(max_z1_cls, max_z2_cls)
    
    print(' min_z1_all shape:', min_z1_all.shape,' min_z1_all:', min_z1_all,' max_z1_all:', max_z1_all.shape,'max_z1_all:', max_z1_all)
    print(' min_z2_all shape:', min_z2_all.shape,' min_z2_all:', min_z2_all,' max_z2_all:', max_z2_all.shape,'max_z2_all:', max_z2_all)
    # print(' min_z_all shape:', min_z_all.shape,' min_z_all:', min_z_all,' max_z_all:', max_z_all.shape,'max_z_all:', max_z_all)
    # print(' min_z1_cls shape:', min_z1_cls.shape,' max_z1_cls shape:', max_z1_cls.shape)        
    # print(' min_z2_cls shape:', min_z2_cls.shape,' max_z2_cls shape:', max_z2_cls.shape)    
    # print(' min_z_cls shape:', min_z_cls.shape,' max_z_cls shape:', max_z_cls.shape)    
        
    if scaling == 'all':
        Z1 = (Z1 - min_z_all)/(max_z_all - min_z_all + 1.0e-9)
        Z2 = (Z2 - min_z_all)/(max_z_all - min_z_all + 1.0e-9)
        title += ' - NORMALIZED to 1 over all classes (jointly)'
        zlim = 'one'
    elif scaling == 'class':
        Z1 = (Z1 - min_z_cls)/(max_z_cls - min_z_cls + 1.0e-9)
        Z2 = (Z2 - min_z_cls)/(max_z_cls - min_z_cls + 1.0e-9)
        title += ' - NORMALIZED to 1 over each classes (jointly) '
    elif scaling == 'each':
        Z1 = (Z1 - min_z1_cls)/(max_z1_cls - min_z1_cls + 1.0e-9)
        Z2 = (Z2 - min_z2_cls)/(max_z2_cls - min_z2_cls + 1.0e-9)
        title += ' - NORMALIZED to 1 over each classes (separately)'
    elif scaling == 'none':    
        title += ' - No Normalization'
    else: 
        print(" ERROR - scaling must be 'all', 'class', 'each' , or  'none' : ", scaling)
        return        
        
    rows        = num_classes 
    columns     = 2
    width  = size[0] * columns
    height = size[1] * num_classes
    fig = plt.figure(figsize=(width, height))
    fig.suptitle(title, fontsize = 14 )
                
    subplt = 1
    for idx,cls in enumerate(class_ids):
        row = idx 
        col = cls  % columns
        
        if class_names is None:
            ttl = 'HM1 Img: {:2d} Cls: {:2d} '.format( image_idx,cls)
        else:
            ttl = 'HM1 Img: {:2d} Cls: {:2d}/{:s}'.format(image_idx, cls, class_names[cls])

        ax = fig.add_subplot(rows, columns, subplt , projection='3d')
        ax.tick_params(axis='both', labelsize = 5)
        
        if scaling == 'all':
            zlim_min = min_z_all
            zlim_max = max_z_all 
        elif scaling == 'class': 
            zlim_min = 0.0   #zlim_min = min_z_cls[0,0,cls]
            zlim_max = 1.005 #zlim_max = max_z_cls[0,0,cls]
            print(' scaling = class : plot 1 min_z : {:10.8f} max_z: {:10.8f} '.format(zlim_min, zlim_max))
        elif scaling == 'each': 
            zlim_min = 0.0   # min_z1_cls[0,0,cls]
            zlim_max = 1.005 # max_z1_cls[0,0,cls]
            print(' scaling = each  : plot 1 min_z : {:10.8f} max_z: {:10.8f} '.format(zlim_min, zlim_max))
        else:
            zlim_min = 0.0
            zlim_max = 1.005

        ax.set_zlim(zlim_min , zlim_max)  
        ttl += '  -  min: {:6.5f}  max: {:6.5f}'.format(min_z1_cls[0,0,cls], max_z1_cls[0,0,cls])
        ax.set_title(ttl, fontsize=12)
        print( 'ttl: ', ttl)        
        # print( 'class: {}  subplt: {} row: {}  col: {}   min_z : {:10.8f} max_z: {:10.8f} '.format(cls, subplt, row, col, min_z,max_z))

        ax.set_ylim(0, image_height + image_height //10)
        ax.set_xlim(0, image_width  + image_width  //10)
        ax.set_xlabel(' X axis', fontsize=8)
        ax.set_ylabel(' Y axis', fontsize=8)
        ax.invert_yaxis()
        
        # ax.view_init( azim=-110,elev=60)            
        surf = ax.plot_surface(X, Y, Z1[:,:,cls],cmap=cm.viridis, linewidth=0, antialiased=False)
        ax.view_init(azim=-37, elev=43)     
        fig.colorbar(surf, shrink=0.8, aspect=30, fraction=0.05)

        ## Plot 2nd columm 
        ##--------------------
        ax = fig.add_subplot(rows, columns, subplt + 1, projection='3d')
        if class_names is None:
            ttl = 'HM2 Img: {:2d} Cls: {:2d} '.format( image_idx,cls)
        else:
            ttl = 'HM2 Img: {:2d} Cls: {:2d}/{:s}'.format(image_idx, cls, class_names[cls])


        ax.tick_params(axis='both', labelsize = 5)
        
        # if scaling == 'all':
            # zlim_min = min_z_all
            # zlim_max = max_z_all 
        # elif scaling == 'class': 
            # zlim_min = 0.0   #zlim_min = min_z_cls[0,0,cls]
            # zlim_max = 1.005 #zlim_max = max_z_cls[0,0,cls]
            # print(' scaling = class : plot 1 min_z : {:10.8f} max_z: {:10.8f} '.format(zlim_min, zlim_max))
        # elif scaling == 'each': 
            # zlim_min = 0.0   # min_z1_cls[0,0,cls]
            # zlim_max = 1.005 # max_z1_cls[0,0,cls]
            # print(' scaling = each  : plot 1 min_z : {:10.8f} max_z: {:10.8f} '.format(zlim_min, zlim_max))
        # else:
            # zlim_min = 0.0
            # zlim_max = 1.005

        ax.invert_yaxis()
        ax.set_zlim(zlim_min , zlim_max)          
        ttl += '  -  min: {:6.5f}  max: {:6.5f}'.format(min_z2_cls[0,0,cls], max_z2_cls[0,0,cls])
        ax.set_title(ttl, fontsize=12)
        print( 'ttl: ', ttl)
        # print( 'class: {}  subplt: {} row: {}  col: {}   min_z : {:10.8f} max_z: {:10.8f} '.format(cls, subplt+1, row, col, min_z,max_z))
        surf = ax.plot_surface(X, Y, Z2[:,:,cls],cmap=cm.viridis, linewidth=0, antialiased=False)
        ax.view_init(azim=-37, elev=43)     
                
        plt.subplots_adjust(top=0.97, bottom=0.02, left=0.02, right=0.98, hspace=0.25, wspace=0.15)      
        fig.colorbar(surf, shrink=0.8, aspect=30, fraction=0.05)
        
        subplt +=2

    plt.show()
    return
    


 
##----------------------------------------------------------------------
## plot one gaussian distribution heatmap - 2D
##----------------------------------------------------------------------       
def plot_2d_gaussian( heatmap, title = 'My figure', width = 10, height = 10, zlim = 1.05 ):
    columns     = 1
    rows        = 1 
    height      = width 
    image_height, image_width = heatmap.shape
    
    fig = plt.figure(figsize=(width, height))
    fig.suptitle(title, fontsize = 10 )
    fig.set_figheight(width-1)

    X = np.arange(0, image_width, 1)
    Y = np.arange(0, image_height, 1)
    X, Y = np.meshgrid(X, Y)        
    pos = np.empty(X.shape+(2,))   # concatinate shape of x to make ( x.rows, x.cols, 2)
    pos[:,:,0] = X;
    pos[:,:,1] = Y;

    col = 0    
    cls = 0
    # for col  in range(2):
    subplot = (cls * columns) + col + 1
    ttl = 'Heatmap {} '.format(col+1)
    # plt.subplot(rows, columns, col+1)
    # ax = fig.gca(projection='3d')
    
    ax = fig.add_subplot(rows, columns, subplot )
    ax.set_title(ttl)
    ax.set_ylim(0,image_height )
    ax.set_xlim(0,image_width)
    ax.set_xlabel(' X axis', fontsize = 8)
    ax.set_ylabel(' Y axis', fontsize = 8)
    # ax.invert_yaxis()
    surf = ax.matshow(heatmap,  cmap=cm.coolwarm)
    # # Customize the z axis.
    # plt.plot()
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.8, aspect=30, fraction=0.05)
    
    plt.show()

 
##----------------------------------------------------------------------
## plot one gaussian distribution heatmap - 3D
##----------------------------------------------------------------------       

def plot_3d_gaussian( heatmap, title = 'My figure', width = 10, height = 10, zlim = 1.05 ):
    columns     = 1
    rows        = 1 
    image_height, image_width = heatmap.shape
    print(' height: {}  width: {} '.format(image_height,image_width))
    fig = plt.figure(figsize=(width, height))
    fig.suptitle(title, fontsize = 10 )
    fig.set_figheight(width-1)

    X = np.arange(0, image_width, 1)
    Y = np.arange(0, image_height, 1)
    X, Y = np.meshgrid(X, Y)        
    pos = np.empty(X.shape+(2,))   # concatinate shape of x to make ( x.rows, x.cols, 2)
    pos[:,:,0] = X;
    pos[:,:,1] = Y;

    col = 0    
    cls = 0
    # for col  in range(2):
    subplot = (cls * columns) + col + 1
    ttl = 'Heatmap {} '.format(col+1)
    # plt.subplot(rows, columns, col+1)
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(rows, columns, subplot, projection='3d')
    ax.set_title(ttl)
    zlim = np.max(heatmap)
    print('Zlim is : ', zlim)
    ax.set_zlim(0.0 , zlim)
    ax.set_ylim(0,image_height )
    ax.set_xlim(0,image_width)
    ax.set_xlabel(' X axis')
    ax.set_ylabel(' Y axis')
    ax.invert_yaxis()
    # ax.view_init( azim=-110,elev=60)            
    ax.view_init(azim=-37, elev=43)            
    surf = ax.plot_surface(X, Y, heatmap[X,Y],cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # # Customize the z axis.
    # plt.plot()
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.  
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
"""    
##----------------------------------------------------------------------
## plot one gauss_scatter for one image and ALL CLASSES
##----------------------------------------------------------------------       

def plot_gaussian2( Zlist, image_idx, title = 'My figure', width = 7 ):
    columns     = len(Zlist)
    num_classes = Zlist[0].shape[-1]
    rows        = num_classes 
    height      = rows * width /2 
    
    fig = plt.figure(figsize=(width, width))
    fig.suptitle(title, fontsize =12 )
    fig.set_figheight(width-1)

    X = np.arange(0, 128, 1)
    Y = np.arange(0, 128, 1)
    X, Y = np.meshgrid(X, Y)        
    pos = np.empty(X.shape+(2,))   # concatinate shape of x to make ( x.rows, x.cols, 2)
    pos[:,:,0] = X;
    pos[:,:,1] = Y;

    for cls in range(num_classes):
    
        for col  in range(2):
            subplot = (cls * columns) + col + 1
            ttl = 'Heatmap {} - image :  {} class: {} '.format(col+1, image_idx,cls)
            # plt.subplot(rows, columns, col+1)
            # ax = fig.gca(projection='3d')
            ax = fig.add_subplot(rows, columns, subplot, projection='3d')
            ax.set_title(ttl)
            ax.set_zlim(0.0 , 1.05)
            ax.set_ylim(0,130)
            ax.set_xlim(0,130)
            ax.set_xlabel(' X axis')
            ax.set_ylabel(' Y axis')
            ax.invert_yaxis()
            # ax.view_init( azim=-110,elev=60)            
            ax.view_init(azim=-37, elev=43)            
            surf = ax.plot_surface(X, Y, Zlist[col][image_idx,:,:,cls],cmap=cm.coolwarm, linewidth=0, antialiased=False)
            # # Customize the z axis.
            # plt.plot()
            # ax.zaxis.set_major_locator(LinearLocator(10))
            # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

            # Add a color bar which maps values to colors.
   
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()
"""    

"""
##----------------------------------------------------------------------
## plot 2d heatmap for one image with bboxes
##----------------------------------------------------------------------        
def plot_2d_heatmap_all_classes( Z, boxes, image_idx,  
                                 columns = 4, size = (7,7), 
                                 num_bboxes = 999, title = '2d heatmap w/ bboxes', class_names=None, scale = 1):

    '''
    Z:             Gaussian heatmap [ BatchSize, height, width, Num_classes]
    image_idx :    index to image 
    class_ids :    Lists of class ids to display
    '''
    num_classes = Z.shape[-1] 
    rows   = math.ceil(num_classes/columns)
    print('rows  ',rows, ' columns :', columns)
    
    image_height = Z.shape[1]
    image_width  = Z.shape[2]


    print(image_height, image_width)
    Y = np.arange(0, image_height, 1)
    X = np.arange(0, image_width, 1)
    X, Y = np.meshgrid(X, Y)        
    
    width  = size[0] * columns
    height = size[1] * rows
    fig = plt.figure(figsize=(width, height))
    fig.suptitle(title, fontsize = 10 )

    # colors = random_colors(num_classes)
    # color = colors[0]
    style = "dotted"
    alpha = 1

    if num_bboxes == 999 :
        num_bboxes  = boxes.shape[2]  
        
    if num_bboxes > 0 :
        x1    = boxes[image_idx,:,:,1] // scale
        x2    = boxes[image_idx,:,:,3] // scale
        y1    = boxes[image_idx,:,:,0] // scale
        y2    = boxes[image_idx,:,:,2] // scale
        box_w = x2 - x1   # x2 - x1
        box_h = y2 - y1 
        # cx    = (x1 + ( box_w / 2.0)).astype(int)
        # cy    = (y1 + ( box_h / 2.0)).astype(int)
        # print('x1, x2...shapes:', x1.shape, x2.shape, y1.shape, y2.shape, box_h.shape, box_w.shape)


    if size is None:
        width  *= columns
        height *= rows
    else: 
        width  = size[0] * columns
        height = size[1] * rows
    fig = plt.figure(figsize=(width, height))  #width , height
    fig.suptitle(title, fontsize = 10 )
   
    for cls in range(num_classes):
        # color = colors[cls]
        color = (0.5, 0.0, 1.0)
        row = cls // columns
        col = cls  % columns
        subplot = (row * columns) + col +1
        # print('idx ', idx,  ' class:', cls, 'row:', row,'col:', col, 'subplot: ', subplot, 'clor:', color)
        if class_names is None:
            ttl = 'Image: {:2d} Cls: {:2d} '.format( image_idx,cls)
        else:
            ttl = 'Image: {:2d} Cls: {:2d}/{:s}'.format(image_idx, cls, class_names[cls])

        
        ax = fig.add_subplot(rows, columns, subplot)
        ax.set_title(ttl, fontsize=9)
        ax.tick_params(axis='both', labelsize = 5)
        ax.set_ylim(0, image_height)
        ax.set_xlim(0, image_width )
        ax.set_xlabel(' X axis', fontsize=8)
        ax.set_ylabel(' Y axis', fontsize=8)
        ax.invert_yaxis()
        surf = plt.matshow(Z[image_idx, :,:, cls], fignum = 0, cmap = cm.coolwarm)
        for bbox in range(num_bboxes):
            # print(num_bboxes, bbox, cls, boxes[image_idx, cls,bbox])
            # print(ttl,x1[cls,bbox], y1[cls,bbox],x2[cls,bbox],y2[cls,bbox])
            p = patches.Rectangle( (x1[cls,bbox],y1[cls,bbox]), box_w[cls,bbox], box_h[cls,bbox], 
                                   linewidth=1, alpha=alpha, linestyle=style, edgecolor=color, facecolor='none')
            ax.add_patch(p)
        plt.subplots_adjust(top=0.94, bottom=0.02, left=0.02, right=0.98, hspace=0.25, wspace=0.15)      
        fig.colorbar(surf, shrink=0.8, aspect=30, fraction=0.05)
    # plt.subplots_adjust(top=0.90, bottom=0.02, left=0.02, right=0.98, hspace=0.10, wspace=0.10)          
    # plt.tight_layout()
    plt.show()
    # plt.savefig('sample.png')
    
    return    
"""

"""
##----------------------------------------------------------------------
## plot 3d heatmap for one image (all classes)
## 19-09-2018
##----------------------------------------------------------------------    
def plot_3d_heatmap_all_classes( Z, image_idx, columns = 3,
                                 title = '3d heatmap - all classes',  
                                 size = (8,8), class_names=None , zlim=1.05):
    '''
    
    Z:             Gaussian heatmap [ BatchSize, height, width, Num_classes]
    image_idx      index to image to display
    
    '''
    image_height = Z.shape[1]
    image_width  = Z.shape[2]
    num_classes  = Z.shape[3]
    print('shape of z', Z.shape , 'num_classes: ', num_classes, 'columns: ',columns )
    
    Y = np.arange(0, image_height, 1)
    X = np.arange(0, image_width, 1)
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape+(2,))   # concatinate shape of x to make ( x.rows, x.cols, 2)
    pos[:,:,0] = X;
    pos[:,:,1] = Y;
    
    # ax = fig.gca(projection='3d')
    # fig.set_figheight(width-1)
    rows   = math.ceil(num_classes/columns)
    # height = math.ceil((width / columns) * rows )
    
    if size is None:
        width  *= columns
        height *= rows
    else: 
        width  = size[0] * columns
        height = size[1] * rows
    # print('height: ',height, 'width: ', width)
    fig = plt.figure(figsize=(width, height))  #width , height
    
    for cls in range(num_classes):
        row = cls // columns
        col = cls  % columns
        
        if class_names is None:
            ttl = 'Image: {:2d} Cls: {:2d} '.format( image_idx,cls)
        else:
            ttl = 'Image: {:2d} Cls: {:2d}/{:s}'.format(image_idx, cls, class_names[cls])

        ax = fig.add_subplot(rows, columns, cls+1, projection='3d')
        ax.set_title(ttl, fontsize=12)
        ax.tick_params(axis='both', labelsize = 5)
        
        if zlim == 0.0: 
            min_z = np.min(Z[image_idx,:,:,cls])
            max_z = np.max(Z[image_idx,:,:,cls])
        else:
            min_z = 0.0
            max_z = zlim
        ax.set_zlim(min_z , max_z)  
        # print( 'class: {}  row: {}  col: {}   min_z : {:10.8f} max_z: {:10.8f} '.format(cls, row,col, min_z,max_z))
        ax.set_ylim(0, image_height + image_height //10)
        ax.set_xlim(0, image_width  + image_width  //10)
        ax.set_xlabel(' X axis', fontsize=8)
        ax.set_ylabel(' Y axis', fontsize=8)
        ax.invert_yaxis()
        # ax.view_init( azim=-110,elev=60)            
        surf = ax.plot_surface(X, Y, Z[image_idx,:,:,cls],cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.view_init(azim=-37, elev=43)     
        
        plt.subplots_adjust(top=0.97, bottom=0.02, left=0.02, right=0.98, hspace=0.25, wspace=0.15)      
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.8, aspect=30, fraction=0.05)
    
    fig.suptitle(title, fontsize = 9 )
    plt.show()
    # plt.savefig('sample.png')
    return    
"""

##----------------------------------------------------------------------
## plot gauss_heatmap for one bbox instance
##----------------------------------------------------------------------        
def plot_one_bbox_heatmap( Z, boxes, title = 'My figure', width = 7, height =12 ):
    N = boxes.shape[0]
    colors = random_colors(N)

    style = "dotted"
    alpha = 1
    color = colors[0]
    
    fig = plt.figure(figsize=(width, height))
    fig.suptitle(title, fontsize =12 )
    ax = fig.gca()
    fig.set_figheight(width-1)
    # surf = ax.plot_surface(X, Y, Z,cmap=cm.coolwarm, linewidth=0, antialiased=False)
    cax  = ax.matshow(Z, cmap=cm.coolwarm )
    cbar = fig.colorbar(cax, ticks=[ 0, 0.5, 1])
    cbar.ax.set_yticklabels(['< 0', '0.5', '> 1'])  # vertically oriented colorbar    
    
    y1, x1, y2, x2 = boxes[:4]
    p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
              alpha=alpha, linestyle=style,
              edgecolor=color, facecolor='none')
    ax.add_patch(p)
    ax.set_ylim(0,130)
    ax.set_xlim(0,130)
    ax.set_xlabel(' X axis')
    ax.set_ylabel(' Y axis')
    ax.invert_yaxis()    
    plt.show()

    
    
"""
##----------------------------------------------------------------------
## plot 2D gauss_distribution for one image, for a list of classes
##----------------------------------------------------------------------        
def plot_2d_heatmap_no_bboxes( Z, image_idx, class_ids = None,  
                               columns = 1, size = (7,7), 
                               title = '2d heatmap', class_names=None, scale = 1):
    '''
    
    Z:             Gaussian heatmap [ BatchSize, height, width, Num_classes]
    image_idx :    index to image 
    class_ids :    Lists of class ids to display
    '''
    if class_ids is None :
        num_classes = Z.shape[-1]
        class_ids   = np.arange(num_classes)
    else:
        num_classes = len(class_ids)
    class_ids.sort()
    print('class_ids:', class_ids)
    
    image_height = Z.shape[1]
    image_width  = Z.shape[2]
    # print(image_height, image_width)
    Y = np.arange(0, image_height, 1)
    X = np.arange(0, image_width, 1)
    X, Y = np.meshgrid(X, Y)        

    rows   = math.ceil(num_classes/columns)
    width  = size[0] * columns
    height = size[1] * rows
    fig = plt.figure(figsize=(width, height))
    fig.suptitle(title, fontsize = 10 )
    # print( 'columns : ', columns , ' width : ', width, ' rows : ',rows, ' height : ',height)
    
    for idx,cls in enumerate(class_ids):
        row = idx // columns
        col = idx  % columns
        subplot = (row * columns) + col +1
        if class_names is None:
            ttl = 'Image: {:2d} Cls: {:2d} '.format( image_idx,cls)
        else:
            ttl = 'Image: {:2d} Cls: {:2d} - {:s}'.format(image_idx, cls, class_names[cls])
        
        ax = fig.add_subplot(rows, columns, subplot)
        ax.set_title(ttl, fontsize = 12, ha= 'center')
        ax.tick_params(axis='both', labelsize = 6)
        ax.set_ylim(0, image_height)
        ax.set_xlim(0, image_width )
        ax.set_xlabel(' X axis', fontsize=8)
        ax.set_ylabel(' Y axis', fontsize=8)
        ax.invert_yaxis()
        surf = plt.matshow(Z[image_idx, :,:, cls], fignum = 0, cmap = cm.coolwarm)


        # ax.plot(Z[image_idx, :,:, cls], cmap = cm.coolwarm)
        plt.subplots_adjust(top=0.97, bottom=0.02, left=0.02, right=0.98, hspace=0.15, wspace=0.15)      
        # Add a color bar which maps values to colors.
        # cbar = fig.colorbar(surf, shrink=0.8, aspect=30, fraction=0.05)
        cbar = fig.colorbar(surf, shrink=0.8, aspect=30, fraction = 0.1)
        cbar.ax.tick_params(labelsize=5) 

    # plt.subplots_adjust(top=0.94, bottom=0.02, left=0.02, right=0.98, hspace=0.25, wspace=0.15)      
    # fig.suptitle(title, fontsize = 10 )
    plt.show()
    return    
"""

##----------------------------------------------------------------------
## inference_heatmaps_display()
##----------------------------------------------------------------------     
def inference_heatmaps_display( input, image_id, hm = 'hm' ,  heatmaps = None, 
                      class_ids = None, 
                      class_names = None,
                      size = (8,8), columns = 3, config = None, scaling = 'clip') :
    '''
    input
    -----
        Z       Gaussian distribution (Batch Sz, Class_sz, Img Height, Img Width)
        boxes   array of bounding boxes 
    '''

    scaling    = scaling.lower()
    results    = input[image_id]
    
    image      = results['image']
    image_meta = results['image_meta']
    
    boxes =  results['fcn_scores_by_class'] 

    if hm == 'hm':
        Z1    =  results['fcn_hm']
        title = 'Image: {:2d} - FCN Heatmaps '.format(image_id)         
    elif hm == 'sm':
        Z1    =  results['fcn_sm']
        title = 'Image: {:2d} - FCN Softmax '.format(image_id)

    print(' heatmap shape: ', Z1.shape)
    scale = config.HEATMAP_SCALE_FACTOR

    print(' Bounding boxes shape: ', boxes.shape)

    if class_ids is None :
        class_ids = np.unique(results['class_ids'])
    class_ids = np.sort(class_ids)
    num_classes = len(class_ids)
    
    print('Image shape :',image.shape,' class_ids:', class_ids)

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

        unmolded_heatmap = utils.unresize_heatmap(Z1[:,:,cls],image_meta, upscale = config.HEATMAP_SCALE_FACTOR)
        # print(' unmolded_heatmap: shape:', unmolded_heatmap.shape, unmolded_heatmap.dtype, np.amin(unmolded_heatmap), np.amax(unmolded_heatmap))
        ax.imshow(image_bw , cmap=plt.cm.gray)        
        surf = ax.imshow(unmolded_heatmap, alpha = 0.6, cmap=cm.YlOrRd, vmin = vmin, vmax = vmax )              

        for bbox in range(num_bboxes):
        # print(ttl,x1[cls,bbox], y1[cls,bbox],x2[cls,bbox],y2[cls,bbox])
            p = patches.Rectangle( (x1[cls,bbox],y1[cls,bbox]), box_w[cls,bbox], box_h[cls,bbox], 
                               linewidth=1, alpha=alpha, linestyle=style, edgecolor=color, facecolor='none')
            ax.add_patch(p)
        plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98, hspace=0.10, wspace=0.10)                
        fig.colorbar(surf, shrink=0.7, aspect=30, fraction=0.05)
    
    fig.suptitle(title, fontsize = 15, ha ='center' )
    plt.show()
    
    # plt.close()
    return
    
##----------------------------------------------------------------------
## inference_heatmaps_compare()
##----------------------------------------------------------------------     
def inference_heatmaps_compare(input, image_id = 0, hm = 'hm' ,  
                      class_ids = None, 
                      class_names = None, 
                      # num_bboxes = 999,
                      size = (8,8), columns = 2, config = None, scaling = 'clip') :
    '''
    input
    -----
        Z             Gaussian distribution (Batch Sz, Class_sz, Img Height, Img Width)
        boxes         array of bounding boxes 
        scaling       'all'    : Normalize to 1 over all classes
                      'class'  : Normalize to 1 over each class indiviually 
                      None     : No Scaling - clip to [-1 , 1]
                      
        hm            'pr'     : use predicted bouding boxes 
                      'gt'       use ground truth bounding boxes
                      
    '''
    ## image is in molded format - unmold it:
 
    scaling    = scaling.lower()
    results    = input[image_id]
    
    image      = results['image']
    image_meta = results['image_meta']
    
    Z1    =  results['pr_hm']        
    boxes =  results['fcn_scores_by_class'] 

    if hm == 'hm':
        Z2    =  results['fcn_hm']
        title = 'MRCNN Predicted vs. FCN Heatmaps '         
    elif hm == 'sm':
        Z2    =  results['fcn_sm']
        title = 'MRCNN Predicted vs. FCN Softmax '         
    assert Z1.shape == Z2.shape
    print(' heatmap shape: ', Z1.shape)
    scale = config.HEATMAP_SCALE_FACTOR

    print(' Bounding boxes shape: ', boxes.shape)
    
    if class_ids is None :
        class_ids = np.unique(results['class_ids'])
    # else:
        # num_classes = Z1.shape[-1]
        # class_ids   = np.arange(num_classes)
    class_ids = np.sort(class_ids)
    num_classes = len(class_ids)
    
    print('Image shape :',image.shape,' class_ids:', class_ids)
    
    # 0 - image here is float32 between -127 and +128
    # 1 - add back pixel mean values -> uint8 between 0, 255
    # 2 - convert from resized to original size 
    # 3 - Convert to grayscale np array   
    # print('0 - image    : ', image.shape, image.dtype, np.min(image), np.max(image))   
    # image      = utils.unmold_image(mrcnn_input_batch[0][image_id], config)
    # print('1- image    : ', image.shape, image.dtype, np.min(image), np.max(image))
    # image = utils.unresize_image(image, image_meta)
    # print('2- image    : ', image.shape, image.dtype, np.min(image), np.max(image))    
    display_image(image)
    image_bw = np.asarray(Image.fromarray(image).convert(mode='L'))

    rows        = num_classes 
    columns     = 2
    width  = size[0] * columns
    height = size[1] * num_classes
    fig = plt.figure(figsize=(width, height))
    
    # if num_bboxes == 999 :
        # num_bboxes  = boxes.shape[2]  
    num_bboxes  = boxes.shape[2]  
    if num_bboxes > 0:
        x1    = boxes[:,:,1] 
        x2    = boxes[:,:,3] 
        y1    = boxes[:,:,0] 
        y2    = boxes[:,:,2] 
        box_w = x2 - x1    
        box_h = y2 - y1 
        # print('x1, x2...shapes:', x1.shape, x2.shape, y1.shape, y2.shape, box_h.shape, box_w.shape)    
        
    colors = random_colors(num_classes)
    style = "dotted"
    linewidth = 1.0
    alpha = 1
    color = (0.5, 0.0, 1.0)
    
    min_z1_all = np.amin(Z1)
    max_z1_all = np.amax(Z1)
    min_z1_cls = np.amin(Z1, axis = (0,1), keepdims = True)
    max_z1_cls = np.amax(Z1, axis = (0,1), keepdims = True)
    avg_z1_cls = np.mean(Z1, axis = (0,1), keepdims = True)
    
    min_z2_all = np.amin(Z2)    
    max_z2_all = np.amax(Z2)
    min_z2_cls = np.amin(Z2, axis = (0,1), keepdims = True)
    max_z2_cls = np.amax(Z2, axis = (0,1), keepdims = True)
    avg_z2_cls = np.mean(Z2, axis = (0,1), keepdims = True)
    
    min_z_all  = np.minimum(min_z1_all, min_z2_all)
    max_z_all  = np.maximum(max_z1_all, max_z2_all)
    
    min_z_cls  = np.minimum(min_z1_cls, min_z2_cls)
    max_z_cls  = np.maximum(max_z1_cls, max_z2_cls)
    
    # print(' min_z1_all shape:', min_z1_all.shape,' min_z1_all:', min_z1_all,' max_z1_all:', max_z1_all.shape,'max_z1_all:', max_z1_all)
    # print(' min_z2_all shape:', min_z2_all.shape,' min_z2_all:', min_z2_all,' max_z2_all:', max_z2_all.shape,'max_z2_all:', max_z2_all)
    # print(' min_z_all shape:', min_z_all.shape,' min_z_all:', min_z_all,' max_z_all:', max_z_all.shape,'max_z_all:', max_z_all)
    # print(' min_z1_cls shape:', min_z1_cls.shape,' max_z1_cls shape:', max_z1_cls.shape)        
    # print(' min_z2_cls shape:', min_z2_cls.shape,' max_z2_cls shape:', max_z2_cls.shape)    
    # print(' min_z_cls shape:', min_z_cls.shape,' max_z_cls shape:', max_z_cls.shape)    
        
    if scaling == 'all':   
        Z1 = (Z1 - min_z_all)/(max_z_all - min_z_all + 1.0e-9)
        Z2 = (Z2 - min_z_all)/(max_z_all - min_z_all + 1.0e-9)
        title += ' - NORMALIZED to [0, 1] over all classes (jointly)'
        zlim = 'one'
    elif scaling == 'class':
        Z1 = (Z1 - min_z_cls)/(max_z_cls - min_z_cls + 1.0e-9)
        Z2 = (Z2 - min_z_cls)/(max_z_cls - min_z_cls + 1.0e-9)
        title += ' - NORMALIZED to [0, 1] over each class (jointly) '
    elif scaling == 'each':
        Z1 = (Z1 - min_z1_cls)/(max_z1_cls - min_z1_cls + 1.0e-9)
        Z2 = (Z2 - min_z2_cls)/(max_z2_cls - min_z2_cls + 1.0e-9)
        title += ' - NORMALIZED to [0, 1] over each class (separately)'
    elif scaling == 'clip':    
        print(' SCALING == clip (clip to [-1, +1])')    
        Z1 = np.clip(Z1, -1.0, 1.0) 
        Z2 = np.clip(Z2, -1.0, 1.0) 
        title += ' - Clip output to [-1, +1]'
    else: 
        print(" ERROR - scaling must be 'all', 'class', 'each' , or  'clip' : ", scaling)
        return        
        

    for idx, cls in enumerate(class_ids):
        # color = colors[idx]
        row = idx // columns
        col = idx  % columns
        
        ## Plot 1st columm 
        ##--------------------
        subplot = (idx * columns) + 1
        if class_names is None:
            ttl = 'Cls: {:2d} '.format(cls)
        else:
            ttl = 'Cls: {:2d}/{:s}'.format(cls, class_names[cls])
        # print('idx ', idx,  ' class:', cls, 'row:', row,'col:', col, 'subplot: ', subplot)
        ax = fig.add_subplot(rows, columns, subplot)
        ax.tick_params(axis='both', labelsize = 5)
        ax.tick_params(direction='out', length=6, width=1, colors='r', labelsize = 10)
        ax.set_xlabel(' X axis', fontsize=10)
        ax.set_ylabel(' Y axis', fontsize=10)

        
        if scaling == 'clip':
            vmin = np.amin(Z1[:,:,cls])
            vmax = np.amax(Z1[:,:,cls])
        else:
            vmin = 0
            vmax = 1

        # ttl = ttl +'  -  min: {:6.5f}  max: {:6.5f}'.format(min_z1_cls[0,0,cls], max_z1_cls[0,0,cls])
        ttl = ttl +'  -  min: {:6.5f}  max: {:6.5f}  avg: {:6.5f}'.format(min_z1_cls[0,0,cls], max_z1_cls[0,0,cls], avg_z1_cls[0,0,cls])
        
        
        unmolded_heatmap = utils.unresize_heatmap(Z1[:,:,cls],image_meta, upscale = config.HEATMAP_SCALE_FACTOR)
        # print(' unmolded_heatmap: shape:', unmolded_heatmap.shape, unmolded_heatmap.dtype, np.amin(unmolded_heatmap), np.amax(unmolded_heatmap))
        ax.imshow(image_bw , cmap=plt.cm.gray)        
        surf = ax.imshow(unmolded_heatmap, alpha = 0.6, cmap=cm.YlOrRd, vmin = vmin, vmax = vmax )              
        
        for bbox in range(num_bboxes):
            p = patches.Rectangle( (x1[cls,bbox],y1[cls,bbox]), box_w[cls,bbox], box_h[cls,bbox], 
                               linewidth=linewidth, alpha=alpha, linestyle=style, edgecolor=color, facecolor='none')
            ax.add_patch(p)
        
        ax.set_title(ttl, fontsize=12)
        plt.subplots_adjust(top=0.94, bottom=0.02, left=0.02, right=0.98,  wspace=0.15)      
        fig.colorbar(surf, shrink=0.7, aspect=30, fraction=0.05)

        ## Plot 2nd columm 
        ##--------------------
        subplot = (idx * columns) + 2
        # ttl = .format(cls)
        ax = fig.add_subplot(rows, columns, subplot)
        ax.tick_params(axis='both', labelsize = 5)
        ax.tick_params(direction='out', length=6, width=1, colors='r', labelsize = 10)
        ax.set_xlabel(' X axis', fontsize=10)
        ax.set_ylabel(' Y axis', fontsize=10)
        ax.imshow(image_bw , cmap=plt.cm.gray)

        if scaling == 'clip':
            vmin = np.amin(Z1[:,:,cls])
            vmax = np.amax(Z1[:,:,cls])
        else:
            vmin = 0
            vmax = 1

        # ttl = ttl +'  -  min: {:6.5f}  max: {:6.5f}'.format(np.amin(Z2[:,:,cls]), np.amax(Z2[:,:,cls]))
        ttl = 'FCN Cls: {:2d} - min: {:6.5f}  max: {:6.5f}  mean: {:6.5f}'.format(cls, min_z2_cls[0,0,cls], max_z2_cls[0,0,cls], avg_z2_cls[0,0,cls])
        # print(' Title: ', ttl)
        unmolded_heatmap = utils.unresize_heatmap(Z2[:,:,cls],image_meta, upscale = config.HEATMAP_SCALE_FACTOR)
        # print(' unmolded_heatmap: shape:', unmolded_heatmap.shape, unmolded_heatmap.dtype, np.amin(unmolded_heatmap), np.amax(unmolded_heatmap))
        surf = ax.imshow(unmolded_heatmap, alpha = 0.6, cmap=cm.YlOrRd, vmin = vmin, vmax = vmax )              
        
        for bbox in range(num_bboxes):
            p = patches.Rectangle( (x1[cls,bbox],y1[cls,bbox]), box_w[cls,bbox], box_h[cls,bbox], 
                               linewidth=linewidth, alpha=alpha, linestyle=style, edgecolor=color, facecolor='none')
            ax.add_patch(p)
        ax.set_title(ttl, fontsize=12)

        ## adjustments and colorbar
        ##-------------------------
#hspace=0.25,
        plt.subplots_adjust(top=0.94, bottom=0.02, left=0.02, right=0.98,  wspace=0.15)      
        fig.colorbar(surf, shrink=0.7, aspect=30, fraction=0.05)
        
        # plt.subplots_adjust(top=0.93, bottom=0.02, left=0.02, right=0.98, hspace=0.1, wspace=0.10)   
        # fig.colorbar(surf, shrink=0.6, aspect=30, fraction=0.05)
        

    fig.suptitle(title, fontsize = 16, ha ='center' )
    plt.show()
    
    return

    