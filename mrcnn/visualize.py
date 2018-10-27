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

import tensorflow as tf
import keras.backend as KB
from   skimage.measure import find_contours
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from   matplotlib.patches import Polygon
from   matplotlib import cm
from   mpl_toolkits.mplot3d import Axes3D
from   matplotlib.ticker import LinearLocator, FormatStrFormatter


import IPython.display

import mrcnn.utils as utils
from mrcnn.datagen     import load_image_gt    


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
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()

##----------------------------------------------------------------------
## display_images
##      figsize : tuple of integers, optional: (width, height) in inches
##                default: None
##                If not provided, defaults to rc figure.figsize.
##----------------------------------------------------------------------
def display_image(image, title=None, cmap=None, norm=None,
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
    height, width = image.shape[:2]
    plt.title(title, fontsize=12)
    plt.imshow(image.astype(np.uint8), cmap=cmap,
               norm=norm, interpolation=interpolation)
    
        
##----------------------------------------------------------------------
## display_instances
##----------------------------------------------------------------------
def display_instances(image, boxes, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None):
    """
    boxes:                  [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks:                  [num_instances, height, width]
    class_ids:              [num_instances]
    class_names:            list of class names of the dataset
    scores:                 (optional) confidence scores for each box
    figsize:                (optional) the size of the image.
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == class_ids.shape[0]
    print(' display_instances() : Image shape: ', image.shape)
    # plt.figure(figsize=(12, 10))        
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
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{} {:.3f}".format(label, score) if score else label
        ax.text(x1, y1 + 8, caption, color='k', size=11, backgroundcolor="w")

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
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
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
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{} {:.3f}".format(label, score) if score else label
        ax.text(x1, y1 + 8, caption,
                color='k', size=11, backgroundcolor="none")

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
    html = ""
    for row in table:
        row_html = ""
        for col in row:
            row_html += "<td>{:40}</td>".format(str(col))
        html += "<tr>" + row_html + "</tr>"
    html = "<table>" + html + "</table>"
    IPython.display.display(IPython.display.HTML(html))

##----------------------------------------------------------------------
## display_weight_stats
##----------------------------------------------------------------------    
def display_weight_stats(model):
    """Scans all the weights in the model and returns a list of tuples
    that contain stats about each weight.
    """
    layers = model.get_trainable_layers()
    table = [["WEIGHT NAME", "SHAPE", "MIN", "MAX", "STD"]]
    for l in layers:
        weight_values  = l.get_weights()  # list of Numpy arrays
        weight_tensors = l.weights        # list of TF tensors
        
        for i, w in enumerate(weight_values):
            weight_name = weight_tensors[i].name
            
            # Detect problematic layers. Exclude biases of conv layers.
            alert = ""
            if w.min() == w.max() and not (l.__class__.__name__ == "Conv2D" and i == 1):
                alert += "  <span style='color:red'>*** dead?</span>"
            if np.abs(w.min()) > 1000 or np.abs(w.max()) > 1000:
                alert += "  <span style='color:red'>*** Overflow?</span>"
            # Add row
            table.append([
                weight_name + alert,
                str(w.shape),
                "{:+9.7f}".format(w.min()),
                "{:+10.7f}".format(w.max()),
                "{:+9.7f}".format(w.std()),
            ])
    display_table(table)



##----------------------------------------------------------------------
## display_weight_histograms
##----------------------------------------------------------------------    
def display_weight_histograms(model, width= 10, height=3, bins =50):
    LAYER_TYPES = ['Conv2D', 'Dense', 'Conv2DTranspose']
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

    
    
##----------------------------------------------------------------------
## plot 2d heatmaps form gauss_scatter with bouding boxes for one image (all classes)
##----------------------------------------------------------------------     
def plot_bbox_heatmaps_all_classes( Z, boxes, title = 'My figure', width = 7, height= 10, columns = 4, num_bboxes=0 ):
    '''
    input
    -----
        Z       Gaussian distribution (Batch Sz, Class_sz, Img Height, Img Width)
        boxes   array of bounding boxes 
    '''
    print('shape of z', Z.shape, 'shape of boxes', boxes.shape)
    num_classes = Z.shape[0]

    if num_bboxes == 0 :
        num_bboxes  = Z.shape[1]
    
    rows   = math.ceil(num_bboxes/columns)
    height = math.ceil((width / columns) * rows )
    print('Number of classes is :', num_classes, 'num_boxes:', num_bboxes, 'rows :', rows, 'columns: ', columns)
    colors = random_colors(num_classes)
    style = "dotted"
    alpha = 1

    x1    = boxes[:,:,1]
    x2    = boxes[:,:,3]
    y1    = boxes[:,:,0]
    y2    = boxes[:,:,2]
    box_w = x2 - x1   # x2 - x1
    box_h = y2 - y1 
    cx    = (x1 + ( box_w / 2.0)).astype(int)
    cy    = (y1 + ( box_h / 2.0)).astype(int)


    for cls in range(num_classes):
        fig = plt.figure(figsize=(width, height))  #width , height
        color = colors[cls]
        for bbox in range(num_bboxes):
            row = bbox // columns
            col = bbox % columns
            # print('bbox:',bbox, 'row:', row,'col:', col)
            ttl = 'Cls:{:2d} BB{:2d}  r/c:{:1d}/{:1d}  - {:4d}/{:4d} '.format( cls,bbox, row,col, cx[cls,bbox], cy[cls,bbox])
            ax = fig.add_subplot(rows, columns, bbox+1)
            ax.set_title(ttl, fontsize=11)
            ax.tick_params(axis='both', labelsize = 5)
            ax.set_ylim(0,130)
            ax.set_xlim(0,130)
            ax.set_xlabel(' X axis', fontsize=6)
            ax.set_ylabel(' Y axis', fontsize=6)
            ax.invert_yaxis()
            ax.matshow(Z[cls,bbox])
            
            # print(ttl,x1[cls,bbox], y1[cls,bbox],x2[cls,bbox],y2[cls,bbox])
            p = patches.Rectangle( (x1[cls,bbox],y1[cls,bbox]), box_w[cls,bbox], box_h[cls,bbox], 
                                   linewidth=1, alpha=alpha, linestyle=style, edgecolor=color, facecolor='none')
            ax.add_patch(p)
        fig_title = 'class: {:2d} - {:2d} boxes '.format(cls, num_bboxes)
        fig.suptitle(fig_title, fontsize =16 )
        plt.subplots_adjust(top=0.90, bottom=0.02, left=0.02, right=0.98, hspace=0.10, wspace=0.10)                
        # plt.tight_layout()
        plt.show()
    # plt.savefig('sample.png')
    
    # plt.close()
    return



    
##----------------------------------------------------------------------
## plot  gaussian distribution heatmap 
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
    ax.invert_yaxis()
    surf = ax.matshow(heatmap,  cmap=cm.coolwarm)
    # # Customize the z axis.
    # plt.plot()
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.8, aspect=30, fraction=0.05)
    
    plt.show()

 

##----------------------------------------------------------------------
## plot 2d heatmap for one image from Gaussian heatmap tensor
##----------------------------------------------------------------------        
def plot_2d_heatmap( Z, image_idx, class_ids,  title = '2d heatmap', width = 6, height = 6, class_names=None):
    '''
    
    Z:             Gaussian heatmap [ BatchSize, height, width, Num_classes]
    image_idx :    index to image 
    class_ids :    Lists of class ids to display
    '''
    num_classes = len(class_ids)
    columns     = 1
    rows        = num_classes 
    height      = rows * max(height, 3)
    print('height is ',height)
    
    image_height = Z.shape[1]
    image_width  = Z.shape[2]
    print(image_height, image_width)
    Y = np.arange(0, image_height, 1)
    X = np.arange(0, image_width, 1)
    X, Y = np.meshgrid(X, Y)        

    fig = plt.figure(figsize=(width, height))  #width , height
    
    for idx,cls in enumerate(class_ids):
        # for col  in range(2):

        if class_names == None:
            ttl = 'Image: {:2d} Cls: {:2d} '.format( image_idx,cls)
        else:
            ttl = 'Image: {:2d} Cls: {:2d}/{:s}'.format(image_idx, cls, class_names[cls])
        
        
        subplot = (idx * columns) + 1
        ax = fig.add_subplot(rows, columns, subplot)
        ax.set_title(ttl, fontsize = 9, ha= 'center')
        ax.tick_params(axis='both', labelsize = 6)
        ax.set_ylim(0, image_height)
        ax.set_xlim(0, image_width )
        ax.set_xlabel(' X axis', fontsize=8)
        ax.set_ylabel(' Y axis', fontsize=8)
        ax.invert_yaxis()
        surf = plt.matshow(Z[image_idx, :,:, cls], fignum = 0, cmap = cm.coolwarm)
        # Add a color bar which maps values to colors.
        cbar = fig.colorbar(surf, shrink=0.8, aspect=30, fraction = 0.1)
        cbar.ax.tick_params(labelsize=5) 

        # ax.plot(Z[image_idx, :,:, cls], cmap = cm.coolwarm)

    plt.subplots_adjust(top=0.94, bottom=0.02, left=0.02, right=0.98, hspace=0.25, wspace=0.15)      
    fig.suptitle(title, fontsize = 10 )
    plt.show()
    return    
            

##----------------------------------------------------------------------
## plot gauss_distribution for one or more classes
## previously named plot_gaussian
## 19-09-2018
##----------------------------------------------------------------------       
def plot_3d_heatmap( Z, image_idx, class_ids, title = '3d heatmap', width = 7, height = 7, class_names=None):
    '''
    
    Z:             Gaussian heatmap [ BatchSize, height, width, Num_classes]
    image_idx :    index to image 
    class_ids :    Lists of class ids to display
    '''
    columns     = 1
    num_classes = len(class_ids)
    rows        = num_classes 
    height      = rows * max(height, 3)
    print('height is ',height)
    
    # fig.set_figheight(width-1)
    image_height = Z.shape[1]
    image_width  = Z.shape[2]
    
    Y = np.arange(0, image_height, 1)
    X = np.arange(0, image_width, 1)
    X, Y = np.meshgrid(X, Y)        

    pos = np.empty(X.shape+(2,))   # concatinate shape of x to make ( x.rows, x.cols, 2)
    pos[:,:,0] = X;
    pos[:,:,1] = Y;
    
    print('shape pos ', pos.shape, 'shape Z: ' ,Z[image_idx,:,:,1].shape)
    
    fig = plt.figure(figsize=(width, height))

    for idx,cls in enumerate(class_ids):

        # for col  in range(2):
        if class_names == None:
            ttl = 'Image: {:2d} Cls: {:2d} '.format( image_idx,cls)
        else:
            ttl = 'Image: {:2d} Cls: {:2d}/{:s}'.format(image_idx, cls, class_names[cls])

        subplot = (idx * columns) + 1
        # plt.subplot(rows, columns, col+1)
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(rows, columns, subplot, projection='3d')
        ax.set_title(ttl, fontsize = 9, ha= 'center')
        ax.set_zlim(0.0 , 1.05)
        ax.tick_params(axis='both', labelsize = 6)
        ax.set_ylim(0, image_height + image_height //10)
        ax.set_xlim(0, image_width  + image_width  //10)
        ax.set_xlabel(' X axis', fontsize=8)
        ax.set_ylabel(' Y axis', fontsize=8)
        ax.invert_yaxis()
        ax.view_init( azim=-116,elev=40)            
        # ax.view_init(azim=-37, elev=43)            
        surf = ax.plot_surface(X, Y, Z[image_idx,:,:,cls],cmap=cm.coolwarm, linewidth=0, antialiased=False)
        # Add a color bar which maps values to colors.
        cbar = fig.colorbar(surf, shrink=0.8, aspect=30, fraction=0.10)
        cbar.ax.tick_params(labelsize=5) 

        # # Customize the z axis.
        # plt.plot()
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    plt.subplots_adjust(top=0.960, bottom=0.02, left=0.02, right=0.98, hspace=0.25, wspace=0.515)                
    fig.suptitle(title, fontsize = 10 )
    plt.show()
    return
    

##----------------------------------------------------------------------
## plot 2d heatmap for one image with bboxes
##----------------------------------------------------------------------        
def plot_2d_heatmap_with_bboxes( Z, boxes, image_idx, class_ids,  width = 7, height = 7, 
                                 num_bboxes = 0, title = '2d heatmap w/ bboxes', class_names=None, scale = 1):

    '''
    
    Z:             Gaussian heatmap [ BatchSize, height, width, Num_classes]
    image_idx :    index to image 
    class_ids :    Lists of class ids to display
    '''
    num_classes = len(class_ids)
    columns     = 1
    rows        = num_classes 
    height      = rows * max(height, 3)
    print('height is ',height)
    
    image_height = Z.shape[1]
    image_width  = Z.shape[2]
    if num_bboxes == 0 :
        num_bboxes  = boxes.shape[2]  

    print(image_height, image_width)
    Y = np.arange(0, image_height, 1)
    X = np.arange(0, image_width, 1)
    X, Y = np.meshgrid(X, Y)        

    
    # Z = np.transpose(Z, [2,0,1])
    # num_classes = Z.shape[0]    
    colors = random_colors(num_classes)
    rows   = math.ceil(num_classes/columns)
    height = math.ceil((width / columns) * rows )
    style = "dotted"
    alpha = 1
    color = colors[0]

    x1    = boxes[image_idx,:,:,1] // scale
    x2    = boxes[image_idx,:,:,3] // scale
    y1    = boxes[image_idx,:,:,0] // scale
    y2    = boxes[image_idx,:,:,2] // scale
    box_w = x2 - x1   # x2 - x1
    box_h = y2 - y1 
    cx    = (x1 + ( box_w / 2.0)).astype(int)
    cy    = (y1 + ( box_h / 2.0)).astype(int)
    print('x1, x2...shapes:', x1.shape, x2.shape, y1.shape, y2.shape, box_h.shape, box_w.shape)
    fig = plt.figure(figsize=(width, height))  #width , height
    fig.suptitle(title, fontsize = 10 )
   
    for idx, cls in enumerate(class_ids):
        color = colors[idx]
        row = cls // columns
        col = cls  % columns
        # print('Image: ', img, 'class:', cls, 'row:', row,'col:', col)
        if class_names == None:
            ttl = 'Image: {:2d} Cls: {:2d} '.format( image_idx,cls)
        else:
            ttl = 'Image: {:2d} Cls: {:2d}/{:s}'.format(image_idx, cls, class_names[cls])

        subplot = (idx * columns) + 1
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

        fig.colorbar(surf, shrink=0.8, aspect=30, fraction=0.05)

    # plt.subplots_adjust(top=0.90, bottom=0.02, left=0.02, right=0.98, hspace=0.10, wspace=0.10)      
    plt.subplots_adjust(top=0.94, bottom=0.02, left=0.02, right=0.98, hspace=0.25, wspace=0.15)      
    # plt.tight_layout()
    plt.show()
    # plt.savefig('sample.png')
    
    return    
            

##----------------------------------------------------------------------
## plot 3d heatmap for one image (all classes)
## 19-09-2018
##----------------------------------------------------------------------    
def plot_3d_heatmap_all_classes( Z, image_idx, width = 12, columns = 6, title = '3d heatmap - all classes', class_names=None):
    '''
    
    Z:             Gaussian heatmap [ BatchSize, height, width, Num_classes]
    image_idx      index to image to display
    
    '''
    # Z = np.transpose(Z, [2,0,1])
    print('shape of z', Z.shape )
    num_classes = 12
    
    image_height = Z.shape[1]
    image_width  = Z.shape[2]

    Y = np.arange(0, image_height, 1)
    X = np.arange(0, image_width, 1)

    X, Y = np.meshgrid(X, Y)
    print(X.shape, Y.shape)
    pos = np.empty(X.shape+(2,))   # concatinate shape of x to make ( x.rows, x.cols, 2)
    pos[:,:,0] = X;
    pos[:,:,1] = Y;
    
    # ax = fig.gca(projection='3d')
    # fig.set_figheight(width-1)
    rows   = math.ceil(num_classes/columns)
    height = math.ceil((width / columns) * rows )

    fig = plt.figure(figsize=(width, height))  #width , height
    for cls in range(num_classes):
        row = cls // columns
        col = cls  % columns
        print( 'class:', cls, 'row:', row,'col:', col)
        if class_names == None:
            ttl = 'Cls:  {:2d} '.format( cls)
        else:
            ttl = 'Cls:  {:s}'.format(class_names[cls])

        ax = fig.add_subplot(rows, columns, cls+1, projection='3d')
        ax.set_title(ttl, fontsize=12)
        ax.tick_params(axis='both', labelsize = 5)
        ax.set_zlim(0.0 , 1.1)
        ax.set_ylim(0, image_height + image_height //10)
        ax.set_xlim(0, image_width  + image_width  //10)
        ax.set_xlabel(' X axis', fontsize=8)
        ax.set_ylabel(' Y axis', fontsize=8)
        ax.invert_yaxis()
        # ax.view_init( azim=-110,elev=60)            
        surf = ax.plot_surface(X, Y, Z[image_idx,:,:,cls],cmap=cm.viridis, linewidth=0, antialiased=False)
        ax.view_init(azim=-37, elev=43)     
        
    plt.subplots_adjust(top=0.90, bottom=0.02, left=0.02, right=0.98, hspace=0.55, wspace=0.55)                
    fig.suptitle(title, fontsize = 9 )
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=2, aspect=20, fraction=0.05)
    plt.show()
    # plt.savefig('sample.png')
    return    



##----------------------------------------------------------------------
## plot one gauss heatmap scatter for one image and ALL CLASSES
##----------------------------------------------------------------------       

def plot_3d_gaussian( heatmap, title = 'My figure', width = 10, height = 10, zlim = 1.05 ):
    columns     = 1
    rows        = 1 
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
    ax = fig.add_subplot(rows, columns, subplot, projection='3d')
    ax.set_title(ttl)
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