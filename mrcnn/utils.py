"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os, sys, math, zlib, argparse, random, platform, pprint, datetime
from   sys      import stdout    
import numpy as np
import tensorflow as tf
import scipy.misc
import skimage.color
import skimage.io
import skimage.transform
import keras.backend as KB
import keras
from distutils.version import LooseVersion
if LooseVersion(keras.__version__) >= LooseVersion('2.2.0'):
    from keras.engine.saving import load_attributes_from_hdf5_group as _load_attributes_from_hdf5_group
    from keras.engine.saving import preprocess_weights_for_loading
else:
    from keras.engine.topology import _load_attributes_from_hdf5_group, preprocess_weights_for_loading

try:
    from xhtml2pdf import pisa
except:
    pass

pp = pprint.PrettyPrinter(indent=2, width=100)

### Batch Slicing -------------------------------------------------------------------
##   Some custom layers support a batch size of 1 only, and require a lot of work
##   to support batches greater than 1. This function slices an input tensor
##   across the batch dimension and feeds batches of size 1. Effectively,
##   an easy way to support batches > 1 quickly with little code modification.
##   In the long run, it's more efficient to modify the code to support large
##   batches and getting rid of this function. Consider this a temporary solution
##   batch dimension size:
##       DetectionTargetLayer    IMAGES_PER_GPU  * # GPUs (batch size)
##-----------------------------------------------------------------------------------

def batch_slice(inputs, graph_fn, batch_size, names=None):
    '''
    Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs:     list of tensors. All must have the same first dimension length
    graph_fn:   A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names:      If provided, assigns names to the resulting tensors.
    '''
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]    # inputs is a list eg. [sc, ix] => input_slice = [sc[0], ix[0],...]
        output_slice = graph_fn(*inputs_slice)   # pass list of inputs_slices through function => graph_fn(sc[0], ix[0],...)
    
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)

    # Change outputs from:
    #    a list of slices where each is a list of outputs, e.g.  [ [out1[0],out2[0]], [out1[1], out2[1]],.....
    # to 
    #    a list of outputs and each has a list of slices ==>    [ [out1[0],out1[1],...] , [out2[0], out2[1],....],.....    
    
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    # for o,n in zip(outputs,names):
        # print(' outputs shape: ', len(o), 'name: ',n)
        # for i in range(len(o)):
            # print(' shape of item ',i, 'in tuple', o[i].shape)
        
    result = [tf.stack(o, axis=0, name=n) for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


def trim_zeros(x):
    '''
    It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    '''
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]

def stack_tensors(x):
    ''' 
    stack an [Batch x Class x Row x Col] tensor into Row x Cols
    originally written for pred_tensor
    '''
    print(' input shape is : ', x.shape)
    lst2 = [ np.squeeze(item) for item in np.split( x, x.shape[0], axis = 0 )]
    lst2 = [ np.squeeze(np.concatenate(np.split(item, item.shape[0], axis = 0 ), axis = 1)) for item in lst2]
    result = [ item[~np.all(item[:,0:4] == 0, axis=1)] for item in lst2]
    print(' length of output list is : ', len(result))
    return (result)

def stack_tensors_3d(x):
    ''' 
    stack an  [Class x Row x Col] tensor into Row x Cols
    originally written for pred_tensor[img_id]
    ''' 
    print(' input shape is : ', x.shape)
    lst2   = [np.squeeze(item) for item in np.split( x, x.shape[0], axis = 0 )]
    result = np.concatenate( [ i[~np.all(i[:,0:4] == 0, axis=1)] for i in lst2] , axis = 0)
    print(' output shape is : ', result.shape)
    # print(result)
    return (result)


###############################################################################
## Image Data Formatting
###############################################################################

def compose_image_meta(image_id, image_shape, window, active_class_ids):
    '''
    Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.

    image_id:               An int ID of the image. Useful for debugging.
    image_shape:            [height, width, channels]
    window:                 (y1, x1, y2, x2) in pixels. The area of the image where the real
                            image is (excluding the padding)
    active_class_ids:       List of class_ids available in the dataset from which
                            the image came. Useful if training on images from multiple datasets
                            where not all classes are present in all datasets.
    '''
    meta = np.array(
        [image_id] +            # size=1
        list(image_shape) +     # size=3
        list(window) +          # size=4 (y1, x1, y2, x2) in image cooredinates
        list(active_class_ids)  # size=num_classes
    )
    return meta


# Two functions (for Numpy and TF) to parse image_meta tensors.
def parse_image_meta(meta):
    '''
    Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    '''
    image_id    = meta[:, 0]
    image_shape = meta[:, 1:4]
    window      = meta[:, 4:8]   # (y1, x1, y2, x2) window of image in in pixels
    active_class_ids = meta[:, 8:]
    return image_id, image_shape, window, active_class_ids


def parse_image_meta_graph(meta):
    '''
    Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta:       [batch, meta length] where meta length depends on NUM_CLASSES
    '''
    # print('    Parse Image Meta Graph ')
    # print('        meta : ' , type(meta), KB.int_shape(meta))
    image_id         = meta[:, 0:1]
    image_shape      = meta[:, 1:4]
    window           = meta[:, 4:8]
    active_class_ids = meta[:, 8:]
    return [image_id, image_shape, window, active_class_ids]

def parse_active_class_ids_graph(meta):
    '''
    Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta:       [batch, meta length] where meta length depends on NUM_CLASSES
    '''
    print(' Parse Image Meta Graph ')
    print('     meta : ' , type(meta), KB.int_shape(meta))
    active_class_ids = meta[:, 8:]
    return active_class_ids

def mold_image(images, config):
    '''
    Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    '''
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    '''
    Takes a image normalized with mold() and returns the original.
    '''
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


##------------------------------------------------------------------------------------------
## Resize image such that it complies with the shape expected by the NN (config.IMAGE_SHAPE)
##------------------------------------------------------------------------------------------
def resize_image(image, min_dim=None, max_dim=None, padding=False):
    '''
    Resizes an image keeping the aspect ratio.

    min_dim:        if provided, resizes the image such that it's smaller
                    dimension == min_dim
    max_dim:        if provided, ensures that the image longest side doesn't
                    exceed this value.
    padding:        If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    --------
    image:          the resized image (as uint8)
    
    window:         (y1, x1, y2, x2). If max_dim is provided, padding might
                    be inserted in the returned image. If so, this window is the
                    coordinates of the image part of the full image (excluding
                    the padding). The x2, y2 pixels are not included.
    scale:          The scale factor used to resize the image
    padding:        Padding added to the image [(top, bottom), (left, right), (0, 0)]
    '''
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    # print(' Resize Image: h: {}, w: {}, min_dim: {}, max_dim: {}'.format(h,w,min_dim,max_dim))
    
    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
        # print(' Scale based on min_dim is :', scale)
        
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        # print(' Round(image_max * scale) > max_dim :  {} *{} = {} >? {}'.format(image_max,scale,image_max*scale, max_dim)) 
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
            # print(' Scale based on max_dim is :', scale)
            
    # Resize image and mask
    if scale != 1:
        image = scipy.misc.imresize(image, (round(h * scale), round(w * scale)))

    # print('scipy.misc.imresize datatype ', image.dtype)
    # print(' Finalized scale is : ', scale)
    # print(' Resized image shape is :', image.shape)
    # Need padding?
    
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad    = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad   = (max_dim - w) // 2
        right_pad  = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)

        # print(' Padding:  top: {}  bottom: {} left: {} right:{} '.format(top_pad, bottom_pad,left_pad,right_pad))
            
    return image, window, scale, padding


##------------------------------------------------------------------------------------------
## Resize image such that it complies with the shape expected by the NN (config.IMAGE_SHAPE)
##------------------------------------------------------------------------------------------
def unresize_image(image, image_meta, upscale = None):
    '''
    Reversed the resize_image function 


    image:          Heatmap to resize 
    image_meta:     Image meta contains information about padding applied to image 
    upscale:        Necessary upscale if heatmap was downscaled in MRCNN 

    Returns:
    --------
    image:          the resized image

    '''
    # Default window (y1, x1, y2, x2) and default scale == 1.
    # print('unresize_image() : input image datatype ', image.dtype)

    h, w = image.shape[:2]

    # Get new height and width
    to_h, to_w = image_meta[1:3]
    window     = image_meta[4:8]
    top_pad, left_pad , bottom_pad, right_pad   = image_meta[4:8]
    scale = 1
    
    if upscale is not None:
        print('0.1 - unresize_image(): shape before upscale: {}  datatype {}  Upscale to h/w: {}/{} '.format(image.shape,image.dtype, (h * upscale),(w * upscale)))
        print('     min: {} , max: {} '.format(np.amin(image), np.amax(image)))
        image = skimage.util.img_as_ubyte(skimage.transform.resize (image, (h*upscale, w * upscale)))
        # print('0.2 - unresize_image(): shape after  upscale: {}  datatype {} '.format(image.shape, image.dtype))
        # print('     min: {} , max: {} '.format(np.amin(image), np.amax(image)))
        
    # print('1 - unresize_image(): Resize Image from: h/w: {}/{}  To:  h/w: {}/{}  , Padding: top: {}  bot: {}  left: {}   right: {} '.
                # format(h, w, to_h, to_w, top_pad, bottom_pad, left_pad , right_pad))
        
    # format applied for np.pad:  [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)] 
    # print('2- unresize_image(): shape before padding removal: {}  datatype {} '.format(image.shape, image.dtype))
    image = image[top_pad: bottom_pad, left_pad: right_pad]
    # print('3-    unresize_image(): shape after padding removal: {}  datatype {} '.format(image.shape, image.dtype))

    # to_image_max = max(to_h, to_w)    
    # from_image_max = max(h, w)
    # scale = to_image_max / from_image_max
    # print('unresize_image(): scale: {}    h-scale: {}    w-scale: {} '.format(scale,round(h * scale), round(w * scale)))

    # print('4 -  unresize_image(): shape r  resize: {}  datatype {} '.format(image.shape, image.dtype))
    # print('     min: {} , max: {} '.format(np.amin(image), np.amax(image)))

    image = skimage.util.img_as_ubyte(skimage.transform.resize (image, (to_h, to_w)))

    # print('5 -  unresize_image(): shape after  resize: {}  datatype {} '.format(image.shape, image.dtype))
    # print('     min: {} , max: {} '.format(np.amin(image), np.amax(image)))


    return image
     


##------------------------------------------------------------------------------------------
## Resize image such that it complies with the shape expected by the NN (config.IMAGE_SHAPE)
##------------------------------------------------------------------------------------------
def unresize_heatmap(image, image_meta, upscale = None):
    '''
    Similar to unresize_image, but doesnt convert to uint. When converting heatmap to uint many values 
    are driven to 0.

    image:          Heatmap to resize 
    image_meta:     Image meta contains information about padding applied to image 
    upscale:        Necessary upscale if heatmap was downscaled in MRCNN 

    Returns:
    --------
    image:          the resized image
    '''
    # Default window (y1, x1, y2, x2) and default scale == 1.
    # print('unresize_image() : input image datatype ', image.dtype)

    h, w = image.shape[:2]

    # Get new height and width
    to_h, to_w = image_meta[1:3]
    window     = image_meta[4:8]
    top_pad, left_pad , bottom_pad, right_pad  = image_meta[4:8]
    scale = 1
    
    if upscale is not None:
        # print('0.1 - unresize_heatmap(): shape before upscale: {}  dtype {}  Upscale to h/w: {}/{} '.format(image.shape,image.dtype, (h * upscale),(w * upscale)))
        # print('      min: {} , max: {} '.format(np.amin(image), np.amax(image)))
        image = skimage.transform.resize (image, (h*upscale, w * upscale))
        # print('0.2 - unresize_heatmap(): shape after  upscale: {}  dtype {} '.format(image.shape, image.dtype))
        # print('      min: {} , max: {} '.format(np.amin(image), np.amax(image)))
    
    # print('1 - unresize_heatmap(): Resize Image from: h/w: {}/{}  To:  h/w: {}/{}  , Padding: top: {}  bot: {}  left: {}   right: {} '.
                # format(h, w, to_h, to_w, top_pad, bottom_pad, left_pad , right_pad))
        
    ## format applied for np.pad is :  [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)] 
    # print('2 - unresize_image(): shape before padding removal: {}  datatype {} '.format(image.shape, image.dtype))
    image = image[top_pad: bottom_pad, left_pad: right_pad]
    # print('3 - unresize_heatmap(): shape after padding removal: {}  datatype {} '.format(image.shape, image.dtype))

    # print('4 - unresize_heatmap(): shape before resize: {}  dtype: {} '.format(image.shape, image.dtype)) 
    image = skimage.transform.resize (image, (to_h, to_w))

    # print('5 - unresize_heatmap(): shape  after resize: {}  dtype: {} '.format(image.shape, image.dtype)) 

    return image



################################################################################################
##  Bounding Box utility functions
################################################################################################

##------------------------------------------------------------------------------------------
##  clip_to_window - - numpy version 
##------------------------------------------------------------------------------------------
def clip_to_window_np(window, boxes):
    '''
    window: (y1, x1, y2, x2). The window in the image we want to clip to.
    boxes : [N, (y1, x1, y2, x2)]
    '''
    new_boxes = np.zeros_like(boxes)
    # print('clip_to_window_np()')
    # print('     boxes.shape: ', boxes.shape)
    # print('     window     : ', window)
    new_boxes[..., 0] = np.maximum(np.minimum(boxes[..., 0], window[2]), window[0])
    new_boxes[..., 1] = np.maximum(np.minimum(boxes[..., 1], window[3]), window[1])
    new_boxes[..., 2] = np.maximum(np.minimum(boxes[..., 2], window[2]), window[0])
    new_boxes[..., 3] = np.maximum(np.minimum(boxes[..., 3], window[3]), window[1])
    return new_boxes
    
##------------------------------------------------------------------------------------------
##  clip_to_window - - tensorflow version 
##------------------------------------------------------------------------------------------
def clip_to_window_tf(window, boxes):
    '''
    window: (y1, x1, y2, x2). The window in the image we want to clip to.
    boxes:  [N, (y1, x1, y2, x2)]
    '''
    window = tf.cast(window, tf.float32) 

    num_images = tf.shape(boxes)[0]
    num_rois   = tf.shape(boxes)[1]
    
    low_vals   = tf.stack([window[:,0],window[:,1], window[:,0], window[:,1]], axis = -1)
    high_vals  = tf.stack([window[:,2],window[:,3], window[:,2], window[:,3]], axis = -1)
    
    low_vals   = tf.expand_dims(low_vals, axis = 1)
    high_vals  = tf.expand_dims(high_vals, axis = 1)
    
    low_vals   = tf.tile(low_vals ,[num_images, num_rois,1])    
    high_vals  = tf.tile(high_vals,[num_images, num_rois,1])    
    
    tmp1 = tf.where(boxes < high_vals , boxes,  high_vals)
    tmp2 = tf.where( tmp1 > low_vals  , tmp1 ,  low_vals )
    return tmp2
        
        
##----------------------------------------------------------------------------------------------
## byclass_to_byimage_np/tf
##----------------------------------------------------------------------------------------------
def byclass_to_byimage_np(in_array, seqid_column = 7):    
    ''' 
    convert a per-class tensor, shaped as  [ num_classes, num_bboxes, columns ]
         to                                  
            a per-image tensor shaped  as  [ num_bboxes, columns]
    '''
    assert in_array.ndim == 3, "number of dimensions must be 3  is : given {}".format(in_array.ndim)
    if in_array.ndim == 3:
        p_sum = np.sum(np.abs(in_array[:,:,0:4]), axis=-1)
        class_idxs, bbox_idxs = np.where(p_sum > 0)
        output = in_array[class_idxs, bbox_idxs, :]
    # elif in_array.ndim == 4:
        # p_sum = np.sum(np.abs(in_array[:,:,:,0:4]), axis=-1, keepdims = True)
        # img_idxs, class_idxs, bbox_idxs, _ = np.where(p_sum > 0)
        # output = in_array[img_idxs, class_idxs, bbox_idxs, :]
    else:
        raise ValueError("number of dimensions must be 3 or 4, is : given {}".format(in_array.ndim)) 
    #p_sum = np.sum(np.abs(in_array[:,:,0:4]), axis=-1)

    # class_idxs contains the indices of the first dimension (class)
    #  bbox_idxs contains the indices of the second dim (bboxes)
    # class_idxs, bbox_idxs = np.where(p_sum > 0)
    
    # max_bboxes      = in_array.shape[-2]
    # non_zero_bboxes = class_idxs.shape[0]
    # output = np.zeros((non_zero_bboxes, in_array.shape[-1]))
    
    # print(' class_idxs: {} , {} '.format(class_idxs.shape, class_idxs))
    # print(' bbox_idxs : {} , {} '.format(bbox_idxs.shape, bbox_idxs))
    # print(' max_bboxes: {} , non_zero bboxes: {}'.format(max_bboxes, non_zero_bboxes))
    # print(' output    : {}'.format(output.shape))
    
    ## 29-01-2019 -- modified to simpler form below 
    # for cls , box in zip(class_idxs, bbox_idxs):
        # print( ' building output: ', cls, box, max_bboxes  - in_array[cls, box, seqid_column].astype(int))
        # output[max_bboxes - in_array[cls, box, seqid_column].astype(int) ] = in_array[cls, box]
    # output = in_array[class_idxs, bbox_idxs, :]
    
    return output

    
    
def byclass_to_byimage_tf(in_array, seqid_column):    
    ''' 
    convert a by class tensor shaped  [batch_size, num_classes, num_bboxes, columns ] to 
            a by image tensor shaped  [batch_size, num_bboxes, columns]
    '''
    aa = tf.reshape(in_array, [in_array.shape[0], -1, in_array.shape[-1]])
    _ , sort_inds = tf.nn.top_k(tf.abs(aa[:,:,seqid_column]), k= in_array.shape[2])
    batch_grid, bbox_grid = tf.meshgrid(tf.range(in_array.shape[0]), tf.range(in_array.shape[2]),indexing='ij')
    gather_inds = tf.stack([batch_grid, sort_inds],axis = -1)
    output = tf.gather_nd(aa, gather_inds )
    return output    
    

##------------------------------------------------------------------------------------------
##  Convert bounding box coordinates from NN shape back to original image coordinates
##------------------------------------------------------------------------------------------
def boxes_to_image_domain(boxes, image_meta):
    '''
    convert the coordinates of a bounding box from resized back to original img coordinates
    
    Input:  
    ------
       boxes        box in NN input coordiantes
       image_meta   image meta-data structure 
       
    Return:
    -------
        boxes       bounding box in image domain coordinates
       
    '''
    # print('   boxes_to_image_domain(): image_meta: ', type(image_meta), image_meta.shape)
    # image_id    = image_meta[0]
    image_shape = image_meta[1:4]
    window      = image_meta[4:8]
    
    # Compute scale and shift to translate coordinates to image domain.
    h_scale = image_shape[0] / (window[2] - window[0])
    w_scale = image_shape[1] / (window[3] - window[1])
    scale   = min(h_scale, w_scale)
    shift   = window[:2]  # y, x
    scales = np.array([scale, scale, scale, scale])
    shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

    # print(' scales shape: ', scales.shape, '  shifts shape: ', shifts.shape)
    # print(' scales: ', scales, 'shifts: ', shifts)
    # Translate bounding boxes to image domain
    boxes = np.multiply(boxes - shifts, scales).astype(np.int32)
    boxes = clip_to_window_np([0,0,image_shape[0], image_shape[1]], boxes)    
    # print(' boxes_to_image_domain() ')
    # print('    Original image shape : {}   Image window info: {} '.format(image_shape, window))
    # print('    Adjustment scale     : {}   Adjustment shift : {} '.format(scales, shift))
    # print('    Adjusted boxes shape : {} '.format(boxes.shape))
    
    return boxes


##------------------------------------------------------------------------------------------
##  Compute bounding boxes from masks.
##------------------------------------------------------------------------------------------
def extract_bboxes(mask):
    '''
    Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    '''
    # print(' ===> Extract_bboxes() ')
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    
    
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies   = np.where(np.any(m, axis=1))[0]
        
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)

    
def flip_bbox(bbox, size, flip_x=False, flip_y=False):
    """Flip bounding boxes according to image flipping directions.

    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(y1_{min}, x1_{min}, y2_{max}, x2_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
        
    size : tuple
        Tuple of length 2: (width, height).
    flip_x : bool
        Whether flip horizontally.
    flip_y : bool
        Whether flip vertically.

    Returns
    -------
    numpy.ndarray
        Flipped bounding boxes with original shape.
    """
    if not len(size) == 2:
        raise ValueError("size requires length 2 tuple, given {}".format(len(size)))
    width, height = size
    bbox = bbox.copy()
    if flip_y:
        ymax = height - bbox[:, 0]
        ymin = height - bbox[:, 2]
        bbox[:, 0] = ymin
        bbox[:, 2] = ymax
    if flip_x:
        xmax = width - bbox[:, 1]
        xmin = width - bbox[:, 3]
        bbox[:, 1] = xmin
        bbox[:, 3] = xmax
    return bbox
    

############################################################################################
##  Computation of IoU, AP,  Precision / Recall calculations
############################################################################################
##------------------------------------------------------------------------------------------
##  Computes IoU overlaps between two sets of boxes.in normalized coordinates
##------------------------------------------------------------------------------------------    
def overlaps_graph_np(proposals, gt_boxes):
    '''
    Computes IoU overlaps between two sets of boxes.in normalized coordinates
    
    boxes1 - proposals :  [batch_size,  proposal_counts, 4 (y1, x1, y2, x2)] <-- Region proposals
    boxes2 - gt_boxes  :  [batch_size, max_gt_instances, 4 (y1, x1, y2, x2)] <-- input_normlzd_gt_boxes
    
    proposal_counts : 2000 (training) or 1000 (inference)
    max_gt_instances: 100
    
    returns :
    ---------
    overlaps :          [ proposal_counts, max_gt_instances] 
                        IoU of all proposal box / gt_box pairs
                        The dimensionality :
                            row:  number of non_zero proposals 
                            cols: number of non_zero gt_bboxes
    '''
    # print(proposals.shape)
    # print(gt_boxes.shape)
    ##------------------------------------------------------------------------------------------------------------
    ## 1. Tile boxes2 and repeat boxes1. This allows us to compare every boxes1 against every boxes2 without loops.
    ##    TF doesn't have an equivalent to np.repeat() so simulate it using tf.tile() and tf.reshape.
    ##  b1: duplicate each row of boxes1 <boxes2.shape[0]> times 
    ##      R1,R2, R3 --> R1,R1,R1,..,R2,R2,R2,...,R3,R3,R3
    ##  b2: duplicate the set of rows in boxes2 <boxes1.shape[0]> times 
    ##      R1,R2,R3 --> R1,R2,R3,R1,R2,R3,....,R1,R2,R3
    ##------------------------------------------------------------------------------------------------------------
    b1 = np.repeat(proposals, gt_boxes.shape[0],axis =0)
    b2 = np.tile(gt_boxes, (proposals.shape[0],1))    # repeat number of times of r_gt_boxes
    
    ##------------------------------------------------------------------------------------------------------------
    ## 2. Compute intersections
    ## b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    ## b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    ##------------------------------------------------------------------------------------------------------------
    y1 = np.maximum(b1[:,0], b2[:,0])
    x1 = np.maximum(b1[:,1], b2[:,1])
    y2 = np.minimum(b1[:,2], b2[:,2])
    x2 = np.minimum(b1[:,3], b2[:,3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    ##------------------------------------------------------------------------------------------------------------
    ## 3. Compute unions
    ##------------------------------------------------------------------------------------------------------------
    b1_area = (b1[:,2] - b1[:,0]) * (b1[:,3] - b1[:,1])
    b2_area = (b2[:,2] - b2[:,0]) * (b2[:,3] - b2[:,1])
    union = b1_area + b2_area - intersection
    iou = intersection / union

    overlaps = np.reshape(iou, (proposals.shape[0], gt_boxes.shape[0]))
    return  overlaps, np.expand_dims(intersection, axis =1), np.expand_dims(union,axis = 1), np.expand_dims(iou, axis = 1)



##------------------------------------------------------------------------------------------
##  Compute IoU between a box and an array of boxes  
##------------------------------------------------------------------------------------------
def compute_2D_iou(box1, box2):
    """
    Calculates IoU between two 2D bounding box arrays
    box1:                1D vector [boxes_count, 4 (y1, x1, y2, x2)]
    box2:                          [boxes_count, 4 (y1, x1, y2, x2)]
    """
    # Calculate intersection areas
    y1 = np.maximum(box1[:,0], box2[:,0])
    y2 = np.minimum(box1[:,2], box2[:,2])
    x1 = np.maximum(box1[:,1], box2[:,1])
    x2 = np.minimum(box1[:,3], box2[:,3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    area1 = (box1[:,2] - box1[:,0]) * (box1[:,3] - box1[:,1])
    area2 = (box2[:,2] - box2[:,0]) * (box2[:,3] - box2[:,1])

    union = area1 + area2 - intersection
    
    return intersection / union  

##------------------------------------------------------------------------------------------
##  Compute IoU between a box and an array of boxes  
##------------------------------------------------------------------------------------------
def compute_one_iou(box1, box2):
    """
    Calculates IoU between two indiviual bounding boxes 
    box1:                1D vector [y1, x1, y2, x2]
    box2:                          [y1, x1, y2, x2]
    """
    # Calculate intersection areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    y1 = np.maximum(box1[0], box2[0])
    y2 = np.minimum(box1[2], box2[2])
    x1 = np.maximum(box1[1], box2[1])
    x2 = np.minimum(box1[3], box2[3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = area1 + area2 - intersection
    print(' Intersection: {}   Union: {}   IoU: {} '.format(intersection , union , intersection /union ))
    return intersection / union      
    
    
##------------------------------------------------------------------------------------------
##  Compute IoU between a box and an array of boxes  
##------------------------------------------------------------------------------------------
def compute_iou(box, boxes, box_area, boxes_area):
    """
    Calculates IoU of the given box with the array of the given boxes.
    box:                1D vector [y1, x1, y2, x2]
    boxes:              [boxes_count, (y1, x1, y2, x2)]
    box_area:           float. the area of 'box'
    boxes_area:         array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    
    # print(' Intersection: \n', intersection)
    # print(' Union       : \n', union)
    # if union == 0 :
        # print('Warning!! Union is 0')
    
    iou = intersection / union  
    return iou


##------------------------------------------------------------------------------------------
##  Compute IoU between two arrays of bounding boxes 
##------------------------------------------------------------------------------------------
def compute_overlaps(boxes1, boxes2):
    """
    Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    
    ## loop over boxes in boxes2
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


##------------------------------------------------------------------------------------------
##  Compute Average Precision 
##------------------------------------------------------------------------------------------
def compute_ap(gt_boxes, gt_class_ids,
               pred_boxes, pred_class_ids, pred_scores,
               iou_threshold=0.5, verbose = 0):
    '''
    Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP:            Mean Average Precision
    precisions:     List of precisions at different class score thresholds.
    recalls:        List of recall values at different class score thresholds.
    overlaps:       [pred_boxes, gt_boxes] IoU overlaps.
    '''
    # Trim zero padding and sort predictions by score from high to low
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes   = trim_zeros(gt_boxes)
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores= pred_scores[:pred_boxes.shape[0]]
    indices    = np.argsort(pred_scores)[::-1]   ## sort indices from largest to smallest

    pred_boxes     = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores    = pred_scores[indices]

    # Compute IoU overlaps [pred_boxes, gt_boxes]
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    
    # Loop through ground truth boxes and find matching predictions
    match_count = 0
    pred_match = np.zeros([pred_boxes.shape[0]])
    gt_match   = np.zeros([gt_boxes.shape[0]])
    
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # print('\n', i, ' sorted overlaps:',overlaps[i, sorted_ixs])
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] == 1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            # print('overlaps[',i,',',j,'] :', overlaps[i,j])
            if iou < iou_threshold:
                if verbose:
                    print(' i:', i, ' pred_box[i]:', pred_boxes[i], 'class[i]:', pred_class_ids[i],' gt_bx j',j, gt_boxes[j], 'class: ', gt_class_ids[j], ', iou:', round(iou,4), 'not meeting IoU threshold')
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                if verbose:
                    print(' i:', i, ' pred_box[i]:', pred_boxes[i], 'class[i]:', pred_class_ids[i],' gt_bx j:',j, gt_boxes[j], 'class: ', gt_class_ids[j], ', iou:', round(iou,4))
                match_count  += 1
                gt_match[j]   = 1
                pred_match[i] = 1
                break

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match) / (np.arange(len(pred_match)) + 1)
    recalls    = np.cumsum(pred_match).astype(np.float32) / len(gt_match)
    if verbose:    
        print(' Cummulatvie sum precision/recalls')
        print('   predictions: ', (np.arange(len(pred_match)) + 1))
        print('   matches(TP): ', np.cumsum(pred_match))
        print('    precisions: ', precisions)
        print('       recalls: ', recalls)
        print('       recalls= predictions /',len(gt_match)) 
    
    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls    = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])
    if verbose:
        print('    precisions: ', precisions)
        print('       recalls: ', recalls)

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP     = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])
    if verbose:
        print('                ', np.where(recalls[:-1] != recalls[1:])[0])
        print('       indices: ', indices)
        print('   recall diff: ', (recalls[indices] - recalls[indices - 1]))
        print('        * PREC: ', (recalls[indices] - recalls[indices - 1])*precisions[indices])
        print('           mAP: ', mAP)
    return mAP, precisions, recalls, overlaps


def compute_recall(pred_boxes, gt_boxes, iou):
    '''
    Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.

    pred_boxes:     [N, (y1, x1, y2, x2)] in image coordinates
    gt_boxes:       [N, (y1, x1, y2, x2)] in image coordinates
    '''
    # Measure overlaps
    overlaps     = compute_overlaps(pred_boxes, gt_boxes)
    iou_max      = np.max(overlaps, axis=1)
    iou_argmax   = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids

    
##------------------------------------------------------------------------------------------
##  Apply non maximal suppression on a set of bounding boxes 
##------------------------------------------------------------------------------------------
def non_max_suppression(boxes, scores, threshold):
    '''
    Identify bboxes with an IoU > Threshold for suppression    
    Performs non-maximum supression and returns indicies of kept boxes.
    Input:
    ------
    boxes:          [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores:         1-D array of box scores.
    threshold:      Float. IoU threshold to use for filtering.
    '''
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)
    # print(' non_max_suppression ')
    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    # print('====> Initial Ixs: ', ixs)
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        # print(' starting ixs : ', ixs,' compare ',i, ' with ', ixs[1:])
        pick.append(i)
        
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        
        # Identify boxes with IoU over the threshold. This
        # returns indicies into ixs[1:], so add 1 to get
        # indicies into ixs.
        #tst =  np.where(iou>threshold)
        remove_ixs = np.where(iou > threshold)[0] + 1
        # print(' np.where( iou > threshold) : ' ,tst, 'tst[0] (index into ixs[1:]: ', tst[0], 
         # ' remove_ixs (index into ixs) : ',remove_ixs)
        
        # Remove indicies of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
        # print(' edning ixs (after deleting ixs[0]): ', ixs, ' picked so far: ',pick)
    # print('====> Final Picks: ', pick)
    return np.array(pick, dtype=np.int32)


    
############################################################################################
##  Bounding box refinement - Compute Refinements/ Apply Refinements
############################################################################################
   
##------------------------------------------------------------------------------------------
##  Bounding box refinement - apply delta bbox refinement - single bbox
##------------------------------------------------------------------------------------------
def apply_box_delta(box, delta):
    """
    13-09-2018
    Applies the given delta to the given box.
    
    boxes:          [y1, x1, y2, x2]. 
                    Note that (y2, x2) is outside the box.
    deltas:         [dy, dx, log(dh), log(dw)]
    """
    box    = box.astype(np.float32)
    
    # Convert to y, x, h, w
    height   = box[2] - box[0]
    width    = box[3] - box[1]
    center_y = box[0] + 0.5 * height
    center_x = box[1] + 0.5 * width
    # print(' first   height: {}  width: {} center x/y : {}/{}'.format(height, width, center_x, center_y))

    # Apply deltas
    center_y += delta[0] * height
    center_x += delta[1] * width
    height   *= np.exp(delta[2])
    width    *= np.exp(delta[3])
    # print(' second   height: {}  width: {} center x/y : {}/{}'.format(height, width, center_x, center_y))
    
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width

    # print(' third   y1/x1: {}/{}  y2/x2 : {}/{}'.format(y1,x1,y2,x2))
    
    return np.array([y1, x1, y2, x2])


##------------------------------------------------------------------------------------------
##  Bounding box refinement - apply delta bbox refinement - numpy version
##------------------------------------------------------------------------------------------
def apply_box_deltas_np(boxes, deltas):
    """
    Applies the given deltas to the given boxes.
    boxes:          [N, (y1, x1, y2, x2)]. 
                    Note that (y2, x2) is outside the box.
    deltas:         [N, (dy, dx, log(dh), log(dw))]
    """
    
    boxes    = boxes.astype(np.float32)
    
    # Convert to y, x, h, w
    height   = boxes[:, 2] - boxes[:, 0]
    width    = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height   *= np.exp(deltas[:, 2])
    width    *= np.exp(deltas[:, 3])
    
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    
    return np.stack([y1, x1, y2, x2], axis=1)

    
##------------------------------------------------------------------------------------------
##  Bounding box refinement - apply delta bbox refinement - tensorflow version
##------------------------------------------------------------------------------------------
def apply_box_deltas_tf(boxes, deltas):
    """
    Applies the given deltas to the given boxes.
    boxes:          [BS, N, (y1, x1, y2, x2)]. 
                    Note that (y2, x2) is outside the box.
    deltas:         [BS, N, (dy, dx, log(dh), log(dw))]
    """
    
    boxes    = tf.cast(boxes, tf.float32)
    
    # Convert to y, x, h, w
    height   = boxes[:,:, 2] - boxes[:,:, 0]
    width    = boxes[:,:, 3] - boxes[:,:, 1]
    center_y = boxes[:,:, 0] + 0.5 * height
    center_x = boxes[:,:, 1] + 0.5 * width
    
    # Apply deltas
    center_y += deltas[:,:, 0] * height
    center_x += deltas[:,:, 1] * width
    height   *= tf.exp(deltas[:,:, 2])
    width    *= tf.exp(deltas[:,:, 3])
    
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return tf.stack([y1, x1, y2, x2], axis=-1)


##------------------------------------------------------------------------------------------
##  box_refinement - Compute Bbox delta refinement - single box
##------------------------------------------------------------------------------------------
def box_refinement(box, gt_box):
    """
    Compute refinement needed to transform ONE bounding box to gt_box.
     
    box   :     [y1, x1, y2, x2]
    gt_box:     [y1, x1, y2, x2]
          
                (y2, x2) is  assumed to be outside the box
    """
    box         = box.astype(np.float32)
    gt_box      = gt_box.astype(np.float32)

    height      = box[2] - box[0]
    width       = box[3] - box[1]
    center_y    = box[0] + 0.5 * height
    center_x    = box[1] + 0.5 * width

    gt_height   = gt_box[2] - gt_box[0]
    gt_width    = gt_box[3] - gt_box[1]
    gt_center_y = gt_box[0] + 0.5 * gt_height
    gt_center_x = gt_box[1] + 0.5 * gt_width

    dy          = (gt_center_y - center_y) / height
    dx          = (gt_center_x - center_x) / width
    dh          = np.log(gt_height / height)
    dw          = np.log(gt_width / width)

    return np.array([dy, dx, dh, dw])


##------------------------------------------------------------------------------------------
##  box_refinement - Compute Bbox delta refinement - numpy version 
##------------------------------------------------------------------------------------------
def box_refinement_np(box, gt_box):
    """
    Compute refinement needed to transform box to gt_box.
    (Non tensorflow version)
    box and gt_box:     [N, (y1, x1, y2, x2)]
                        (y2, x2) is  assumed to be outside the box.
    """
    box         = box.astype(np.float32)
    gt_box      = gt_box.astype(np.float32)

    height      = box[:, 2] - box[:, 0]
    width       = box[:, 3] - box[:, 1]
    center_y    = box[:, 0] + 0.5 * height
    center_x    = box[:, 1] + 0.5 * width

    gt_height   = gt_box[:, 2] - gt_box[:, 0]
    gt_width    = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy          = (gt_center_y - center_y) / height
    dx          = (gt_center_x - center_x) / width
    dh          = np.log(gt_height / height)
    dw          = np.log(gt_width / width)

    return np.stack([dy, dx, dh, dw], axis=1)

    
##------------------------------------------------------------------------------------------
##  box_refinement - Compute Bbox delta refinement - tensorflow version 
##------------------------------------------------------------------------------------------
def box_refinement_graph(box, gt_box):
    """
    Compute refinement needed to transform box to gt_box.  (tensorflow version)
    
    box   :     [N, (y1, x1, y2, x2)]
    gt_box:     [N, (y1, x1, y2, x2)]
    """
    box         = tf.cast(box, tf.float32)
    gt_box      = tf.cast(gt_box, tf.float32)

    height      = box[:, 2] - box[:, 0]
    width       = box[:, 3] - box[:, 1]
    center_y    = box[:, 0] + 0.5 * height
    center_x    = box[:, 1] + 0.5 * width

    gt_height   = gt_box[:, 2] - gt_box[:, 0]
    gt_width    = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy          = (gt_center_y - center_y) / height
    dx          = (gt_center_x - center_x) / width
    dh          = tf.log(gt_height / height)
    dw          = tf.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result



###############################################################################
## Mask Operations 
###############################################################################
def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask

    
def minimize_mask(bbox, mask, mini_shape):
    '''
    Resize masks to a smaller version to cut memory load.
    Mini-masks can then resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    '''
    # print(' ===> minimize_mask() ')
    # print('   bbox :' )
    # pp.pprint( bbox)
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    # print('    mask :   {}     mini_mask shape: {} '.format( mask.shape, mini_mask.shape))
    
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        # print('    bbox: {}      m[{}:{} , {}:{}]    m.shape: {} '.format(i, y1,y2,x1,x2, m.shape))
        m = m[y1:y2, x1:x2]
        # print(' m.size is ', m.size)
        # added else clause below, commented raise exception for invalid bounding box to avoid 
        # abend of process when an 0 size bounding box is encountered   7-6-2018
        if m.size == 0:
            print('      ######  m.size is zero for bbox ',i)
            # raise Exception("Invalid bounding box with area of zero")
        else:
            m = scipy.misc.imresize(m.astype(float), mini_shape, interp='bilinear')
            mini_mask[:, :, i] = np.where(m >= 128, 1, 0)
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().
    calls scipy resize to resize mask to the height and width of its corresponding bbox, 
    
    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        m = scipy.misc.imresize(m.astype(float), (h, w), interp='bilinear')
        mask[y1:y2, x1:x2, i] = np.where(m >= 128, 1, 0)
    return mask


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
    pass


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = scipy.misc.imresize(
        mask, (y2 - y1, x2 - x1), interp='bilinear').astype(np.float32) / 255.0
    mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask

	

###############################################################################
##  Miscellenous Graph Functions
###############################################################################

def trim_zeros_graph(boxes, name=None):
    """
    Often boxes are represented with matricies of shape [N, 4] and
    are padded with zeros. This removes zero boxes by summing the coordinates
    of boxes, converting 0 to False and <> 0 ti True and creating a boolean mask
    
    boxes:      [N, 4] matrix of boxes.
    non_zeros:  [N] a 1D boolean mask identifying the rows to keep
    """
    # sum tf.abs(boxes) across axis 1 (sum all cols for each row) and cast to boolean.
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    
    # extract non-zero rows from boxes
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros

def batch_pack_graph(x, counts, num_rows):
    """
    Picks different number of values from each row in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)
    

###############################################################################
## Pyramid Anchors
###############################################################################

def generate_anchors(scales, ratios, feature_shape, feature_stride, anchor_stride):
    '''
    scales:             1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios:             1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    feature_shape:      [height, width] spatial shape of the feature map over which
                        to generate anchors.
    feature_stride:     Stride of the feature map relative to the image in pixels.
    anchor_stride:      Stride of anchors on the feature map. For example, if the
                        value is 2 then generate anchors for every other feature map pixel.
    Returns
    -------
           Array of anchor box cooridnates in the format (y1,x1, y2,x2)
    '''
    
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    
    scales = scales.flatten()
    ratios = ratios.flatten()
    
    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)  # 3x1
    widths  = scales * np.sqrt(ratios)  # 3x1
    # print('     - generate_anchors()   Scale(s): ', scales, 'Ratios: ', ratios, ' Heights: ' ,heights, 'Widths: ' ,widths)
    
    
    # Enumerate x,y shifts in feature space - which depends on the feature stride
    # for feature_stride 3 - shifts_x/y is 32
    # 
    shifts_y = np.arange(0, feature_shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, feature_shape[1], anchor_stride) * feature_stride
    # print(' Strides shift_x, shift_y:\n ' ,shifts_x,'\n', shifts_y)

    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
    # print(' meshgrid shift_x, shift_y: ' ,shifts_x.shape, shifts_y.shape)
    
    # Enumerate combinations of shifts, widths, and heights
    # shape of each is [ shape[0] * shape[1] * size of (width/height)] 
    box_widths , box_centers_x = np.meshgrid(widths, shifts_x)    
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    
    # Reshape to get a list of (y, x) and a list of (h, w)
    # print(' box_widths  ', box_widths.shape ,' box_cneterss: ' , box_centers_x.shape)
    # print(' box_heights ', box_heights.shape,' box_cneters_y: ' , box_centers_y.shape)
    # print(' box_centers stack   :' , np.stack([box_centers_y, box_centers_x], axis=2).shape)
    # print(' box_centers reshape :' , np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1,2]).shape)
    # print(' box_sizes   stack   :' , np.stack([box_heights, box_widths], axis=2).shape)
    # print(' box_sizes   reshape :' , np.stack([box_heights, box_widths], axis=2).reshape([-1,2]).shape)
    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes   = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    # print(' Anchor boxes shape is : ' ,boxes.shape)
    return boxes


def generate_pyramid_anchors(anchor_scales, anchor_ratios, feature_shapes, feature_strides, anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    # print('\n>>> Generate pyramid anchors ')
    # print('      Anchor  scales:  ', anchor_scales)
    # print('      Anchor  ratios:  ', anchor_ratios)
    # print('      Anchor  stride:  ', anchor_stride)
    # print('      Feature shapes:  ', feature_shapes)
    # print('      Feature strides: ', feature_strides)

    anchors = []
    for i in range(len(anchor_scales)):
        anchors.append(generate_anchors(anchor_scales[i], anchor_ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    # anchors is a list of 5 np.arrays (one for each anchor scale)
    # concatenate these arrays on axis 0
    
    pp = np.concatenate(anchors, axis=0)
    # print('    Size of anchor array is :',pp.shape)
   
    return pp
   

###############################################################################
##  Utility Functions
###############################################################################
# def tensor_info(x):
    # print('     {:25s} : {}- {}  KerasTensor: {} '.format(x.name, x.shape, KB.int_shape(x), KB.is_keras_tensor(X)))

  
def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)

def logt(text = '', tensor=None, indent=1, verbose = 1):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if not verbose:
        return
        
    text = '    '*indent+ text.strip()
    text = text.ljust(35)
    
    if tensor is None : 
        pass
    elif isinstance(tensor, list):
        text += (":  list length: {}".format(len(tensor)))
    elif isinstance(tensor, str):
        text += (":  value: {}".format(tensor))
    elif isinstance(tensor, int):
        text += (":  value: {}".format(tensor))
    elif isinstance(tensor, float):
        text += (":  value: {}".format(tensor))
    elif isinstance(tensor, np.ndarray):
        text += (":  shape: {}".format(tensor.shape))
    elif isinstance(tensor, tuple):
        text += (":  tuple length: {}".format(len(tensor)))
    else:
        text = '    '*indent+ text.strip()
        text = text.ljust(35)
        text += (":  shape: {:20}  KB.shape:{:20}  Keras Tensor: {}".format(
            str(tensor.shape),
            str(KB.int_shape(tensor)),
            KB.is_keras_tensor(tensor)))
    print(text)

def mask_string(mask):
    return np.array2string(np.where(mask,mask,0),max_line_width=134, separator = '')    


    
def write_stdout(filepath=None,filenm=None,stdout=stdout):
    
    with open(os.path.join(filepath,filenm),'wb'  ) as f_obj :
        content = stdout.getvalue().encode('utf_8')
        f_obj.write(content)
        f_obj.close()    
    
    return


def write_sysout(directory):
    sys.stdout.flush()
    sysout_file = "{:%Y%m%dT%H%M}_sysout.out".format(datetime.datetime.now())
    write_stdout( directory, sysout_file , sys.stdout )        
    sys.stdout = sys.__stdout__
    print(' Run information written to ', sysout_file+'.out')
    
    return

    
def write_zip_stdout(filepath=None,filenm=None,stdout=stdout):
    with open(filepath+filenm+'_xtrct_out.zip','wb'  ) as f_obj :
        stdout = stdout.getvalue().encode('utf_8')
        comp_stdout = zlib.compress(stdout)   
        f_obj.write(comp_stdout)
        f_obj.close()    
    return

def convertHtmlToPdf(sourceHtml, outputFilename):

    outputFile = open(outputFilename, "w+b")
    pisaStatus = pisa.CreatePDF(sourceHtml, dest = outputFile)
    outputFile.close()
    return pisaStatus.err
    
def load_class_prediction_avg(filename, verbose = 0):
    '''
    Load class predictions avergaes for evaluation mode
    '''
    import pickle
    print(' load class+predcition_info from :', filename)
    
    with open(filename, 'rb') as infile:
        class_prediction_info = pickle.load(infile)
     
    class_pred_stats  = {}
    class_pred_stats.setdefault('avg', [cls['avg'] for cls in class_prediction_info])
    class_pred_stats.setdefault('pct', [cls['percentiles'] for cls in class_prediction_info])    
    
    for stat, pct, cls in zip(class_pred_stats['avg'], class_pred_stats['pct'], class_prediction_info):
        if verbose:
            print('  ', cls['id'], cls['name'], ' bboxes:', len(cls['bboxes']) , 'avg: ' , cls['avg'], ' -  ', stat, ' pctile:', pct)
            
        assert stat == cls['avg'], 'mismtach between class_prediction_avg and class_prediction_info.'
        assert pct  == cls['percentiles'], 'mismtach between class_prediction_avg and class_prediction_info.'
        
    return class_pred_stats

    
##----------------------------------------------------------------------------------------------
## get_predicted_mrcnn_deltas
##----------------------------------------------------------------------------------------------
def get_predicted_mrcnn_deltas(m_class, m_bbox, verbose= False):
    '''
    
    return  MRCNN_BBOX deltas corresponding to the max predicted MRCNN_CLASS
    extract predicted refinements for highest predicted class
    
    m_class:      [Batch_Size, #predictions, # classes ]
                    Predicted  scores for each class  (returned in mrcnn_class)
    m_bbox :      [Batch_Size, #predictions, # classes, 4 (dy, dx, log(dh), log(dw))]
                    Predicted refinements for each class (returned in mrcnn_bbox)
                    
    Returns:
    --------
    
    predicted_classes  [ Batch_size, # Predictions] : class_id predidcted

    predicted_deltas   [Batch_Size, #predictions, 4 (dy, dx, log(dh), log(dw))]
                        delta matching predicted class_id
    '''
    
    predicted_classes = np.argmax(m_class,axis = -1)
    row,col = np.indices(predicted_classes.shape)
    predicted_deltas = m_bbox[row,col,predicted_classes]

    if verbose:
        print('mrcnn_class shape:', m_class.shape)
        print('mrcnn_bbox shape :', m_bbox.shape)
        print('predicted_classes:', predicted_classes.shape)
#         print('predicted_classes[1]:\n',predicted_classes[1])
        print('predicted_deltas :', predicted_deltas.shape)
#     print('row: ',row.shape, row)
#     print('col: ',col.shape, col)
#     print(mrcnn_bbox[row,col,predicted_classes].shape)
    
    return predicted_classes, predicted_deltas    
    
    
##----------------------------------------------------------------------------------------------
## Load weights from hdf5 file
##----------------------------------------------------------------------------------------------

def load_weights_from_hdf5_group(f, layers, reshape=False, verbose = 0):
    """Implements topological (order-based) weight loading.

    # Arguments
        f: A pointer to a HDF5 group.
        layers: a list of target layers.
        reshape: Reshape weights to fit the layer when the correct number
            of values are present but the shape does not match.

    # Raises
        ValueError: in case of mismatch between provided layers
            and weights file.
    """
    ## from keras.engine.topology import _load_attributes_from_hdf5_group
    
    if 'keras_version' in f.attrs:
        original_keras_version = f.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in f.attrs:
        original_backend = f.attrs['backend'].decode('utf8')
    else:
        original_backend = None

    filtered_layers = []
    for layer in layers:
        weights = layer.weights
        if weights:
            filtered_layers.append(layer)

    layer_names = _load_attributes_from_hdf5_group(f, 'layer_names')
    filtered_layer_names = []
    for name in layer_names:
        g = f[name]
        weight_names = _load_attributes_from_hdf5_group(g, 'weight_names')
        if weight_names:
            filtered_layer_names.append(name)
    layer_names = filtered_layer_names
    if len(layer_names) != len(filtered_layers):
        raise ValueError('You are trying to load a weight file '
                         'containing ' + str(len(layer_names)) +
                         ' layers into a model with ' +
                         str(len(filtered_layers)) + ' layers.')

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = _load_attributes_from_hdf5_group(g, 'weight_names')
        weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]
        layer = filtered_layers[k]
        symbolic_weights = layer.weights
        weight_values = preprocess_weights_for_loading(layer,
                                                       weight_values,
                                                       original_keras_version,
                                                       original_backend,
                                                       reshape=reshape)
        if len(weight_values) != len(symbolic_weights):
            raise ValueError('Layer #' + str(k) +
                             ' (named "' + layer.name +
                             '" in the current model) was found to '
                             'correspond to layer ' + name +
                             ' in the save file. '
                             'However the new layer ' + layer.name +
                             ' expects ' + str(len(symbolic_weights)) +
                             ' weights, but the saved weights have ' +
                             str(len(weight_values)) +
                             ' elements.')
        weight_value_tuples += zip(symbolic_weights, weight_values)
    KB.batch_set_value(weight_value_tuples)


##----------------------------------------------------------------------------------------------
## Load weights from hdf5 file
##----------------------------------------------------------------------------------------------
def load_weights_from_hdf5_group_by_name(f, layers, skip_mismatch=False, reshape=False, verbose = 0):
    """Implements name-based weight loading.

    (instead of topological weight loading).

    Layers that have no matching name are skipped.

    # Arguments
        f: A pointer to a HDF5 group.
        layers: A list of target layers.
        skip_mismatch: Boolean, whether to skip loading of layers
            where there is a mismatch in the number of weights,
            or a mismatch in the shape of the weights.
        reshape: Reshape weights to fit the layer when the correct number
            of values are present but the shape does not match.

    # Raises
        ValueError: in case of mismatch between provided layers
            and weights file and skip_mismatch=False.
    """
    ## import keras.backend as K
    ## from keras.engine.topology import _load_attributes_from_hdf5_group, preprocess_weights_for_loading
   
        
    if 'keras_version' in f.attrs:
        original_keras_version = f.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in f.attrs:
        original_backend = f.attrs['backend'].decode('utf8')
    else:
        original_backend = None

    # New file format.
    layer_names = _load_attributes_from_hdf5_group(f, 'layer_names')
    
    # Reverse index of layer name to list of layers with name.    
    index = {}
    for layer in layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    if verbose > 1:        
        print('    load_weights_from_hdf5_group_by_name()  model layers: type: ', type(layers), 'length: ', len(layers))    
        print('    load_weights_from_hdf5_group_by_name()  HDF5  layers: type: ', type(layer_names), 'length: ', len(layer_names))
            
    if verbose > 1:
        print("=========================================")
        print(" Layer names loaded from hd5 file:       ")
        print("=========================================")
        for idx,layer in enumerate(layer_names):
            print(' {:4d}  {} '.format(idx, layer ))

    if verbose > 2:
        # print(" Load weights from hd5 by name ")
        print("=====================================================================")
        print(" Model Layer names (passed to load_weights_from_hd5_group) ")
        print("=====================================================================")
        for idx,layer in enumerate(layers):
            print(' {:4d}   {:25s}   {} '.format(idx,layer.name, layer))
        print("=====================================================================")
        print(" Model Reverese Index:")
        print("=====================================================================")
        for key  in index.keys():
            print(' {:.<30s}     {}'.format(key, index[key]))

    if verbose > 2:
        print("=====================================================================")
        print(" List Defined Model  and matching hd5 layers, if present       ")
        print("       Model Layer                     HD5 Layer ")
        print("=====================================================================")
        for idx, layer in enumerate(layers):
            hdf5_layer_count = layer_names.count(layer.name) 
            print(' {:4d}  {:.<30s}  {} '.format(idx, layer.name, hdf5_layer_count  if hdf5_layer_count  else '!!! Not Found in HDF5 file !!!')) 
        print("=====================================================================")
        print(" List hd5 layers and matching layers from Defined Model, if present  ")
        print("       HDF5 Layer                      Model Layer")
        print("=====================================================================")
        for idx, name in enumerate(layer_names):
            mdl_layer_name = index.get(name, []) 
            print(' {:4d}  {:.<30s}  {} '.format(idx, name, mdl_layer_name[0].name  if mdl_layer_name else '!!! NotFound in Defined Model !!!'))
            
    if verbose > 1:
        print("=====================================================================")
        print(" Weight Matchup between model and HDF5 weights ")
        print("=====================================================================")

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []    
    ## layer_names  : layers names from hdf5 file
    ## weight_names : weight names from hdf5 file
    
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = _load_attributes_from_hdf5_group(g, 'weight_names')
        weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]
        
        model_layers = index.get(name,[])
        if verbose > 1:
            if not model_layers:
                print('\n{:3d} {:25s} *** No corresponding layers found in model ***'.format(k,name))
                print('    HDF5 Weights  : {} \n'.format(weight_names))
                for i in range(len(weight_values)):
                    print('{:5d} {:35s}  hdf5 Weights: {}'.format( i, weight_names[i], weight_values[i].shape))
            else:
                print('\n{:3d} {:25s} Model Layer Name/Type : {} '.format(k,name, [ [i.name, i.__class__.__name__] for i in model_layers]))
                print('    HDF5 Weights  : {} \n'.format(weight_names))
        
        
        for layer in index.get(name, []):
            symbolic_weights = layer.weights
            weight_values = preprocess_weights_for_loading(
                layer,
                weight_values,
                original_keras_version,
                original_backend,
                reshape=reshape)
                
            # print('{} len weight_values: {}   len symbolic_weights: {}'.format(' '*30,len(weight_values), len(symbolic_weights)))
            
            if len(weight_values) != len(symbolic_weights):
                print('      Skipping loading of weights for layer {}'.format(layer.name) +
                              ' due to mismatch in number of weights' +
                              ' ({} vs {}).'.format(len(symbolic_weights), len(weight_values)))

                if skip_mismatch:
                    warnings.warn('Skipping loading of weights for layer {}'.format(layer.name) +
                                  ' due to mismatch in number of weights' +
                                  ' ({} vs {}).'.format(len(symbolic_weights), len(weight_values)))
                    continue
                else:
                    raise ValueError('Layer #' + str(k) +
                                     ' (named "' + layer.name +
                                     '") expects ' +
                                     str(len(symbolic_weights)) +
                                     ' weight(s), but the saved weights' +
                                     ' have ' + str(len(weight_values)) +
                                     ' element(s).')
            
            # Set values.
            for i in range(len(weight_values)):
                if verbose > 1:
                    status = " " if weight_values[i].shape == symbolic_weights[i].shape  else  " ***** MISMATCH ***** "
                    print('{:5d} {:35s}  hdf5 Weights: {}  \t\t Symbolic Wghts: {}  {}'.format(
                                     i, weight_names[i], weight_values[i].shape, symbolic_weights[i].shape, status))
                
                if skip_mismatch:
                    if KB.int_shape(symbolic_weights[i]) != weight_values[i].shape:
                        print('{}  {} {:35s}  Weight MISMATCH {}'.format(' '*30, i, weight_names[i],weight_values[i].shape))
                        warnings.warn('Skipping loading of weights for layer {}'.format(layer.name) +
                                      ' due to mismatch in shape' +
                                      ' ({} vs {}).'.format(
                                          symbolic_weights[i].shape,
                                          weight_values[i].shape))
                        continue
                        
                weight_value_tuples.append((symbolic_weights[i], weight_values[i]))
    KB.batch_set_value(weight_value_tuples)

    
    
    
##------------------------------------------------------------------------------------
## FILE PATHS CLASS  
##------------------------------------------------------------------------------------
class Paths(object):
    """
    Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    
    def __init__(self, dataset               , 
                       training_folder       , 
                       fcn_training_folder   ,  
                       mrcnn_training_folder ):
        print(">>> Initialize Paths")
        syst = platform.system()
        if syst == 'Windows':
            # Root directory of the project
            print(' System Platform: ' , syst)
            # WINDOWS MACHINE ------------------------------------------------------------------
            self.DIR_ROOT          = "F:\\"
            self.DIR_TRAINING   = os.path.join(self.DIR_ROOT, training_folder)
            self.DIR_DATASET    = os.path.join(self.DIR_ROOT, 'MLDatasets', dataset)
            self.DIR_PRETRAINED = os.path.join(self.DIR_ROOT, 'PretrainedModels')
        elif syst == 'Linux':
            print(' Linx ' , syst)
            # LINUX MACHINE ------------------------------------------------------------------
            self.DIR_ROOT       = os.getcwd()
            self.DIR_TRAINING   = os.path.expanduser('~/'+training_folder)
            self.DIR_DATASET    = os.path.expanduser(os.path.join('~/MLDatasets', dataset))
            self.DIR_PRETRAINED = os.path.expanduser('~/PretrainedModels')
        else :
            raise Error('unrecognized system ')

        self.MRCNN_TRAINING_PATH   = os.path.join(self.DIR_TRAINING    , mrcnn_training_folder)
        self.FCN_TRAINING_PATH     = os.path.join(self.DIR_TRAINING    , fcn_training_folder)
        self.COCO_DATASET_PATH     = self.DIR_DATASET
        # self.COCO_HEATMAP_PATH     = os.path.join(self.DIR_DATASET     , "coco2014_heatmaps")
        self.COCO_MODEL_PATH       = os.path.join(self.DIR_PRETRAINED  , "mask_rcnn_coco.h5")
        self.SHAPES_MODEL_PATH     = os.path.join(self.DIR_PRETRAINED  , "mask_rcnn_shapes.h5")
        self.RESNET_MODEL_PATH     = os.path.join(self.DIR_PRETRAINED  , "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
        self.VGG16_MODEL_PATH      = os.path.join(self.DIR_PRETRAINED  , "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
        self.FCN_VGG16_MODEL_PATH  = os.path.join(self.DIR_PRETRAINED  , "fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5")
        # self.PRED_CLASS_INFO_PATH  = os.path.join(self.DIR_PRETRAINED  , "predicted_classes_info.pkl")
        return 
        
    def display(self):
        """Display Paths values."""
        print()
        print("   Paths:")
        print("   -------------------------")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("   {:30} {}".format(a, getattr(self, a)))
        print()
    
##------------------------------------------------------------------------------------
## Parse command line arguments
##  
## Example:
## train-shapes_gpu --epochs 12 --steps-in-epoch 7 --last_epoch 1234 --logs_dir mrcnn_logs
## args = parser.parse_args("train --dataset E:\MLDatasets\coco2014 --model mask_rcnn_coco.h5 --limit 10".split())
##------------------------------------------------------------------------------------
def command_line_parser():
    parser = argparse.ArgumentParser(description='Train Mask R-CNN on MS COCO.')

    # parser.add_argument("command",
                        # metavar="<command>",
                        # help="'train' or 'evaluate' on MS COCO")

    # parser.add_argument('--dataset', required=True,
                        # metavar="/path/to/coco/",
                        # help='Directory of the MS-COCO dataset')
    
    # parser.add_argument('--limit', required=False,
                        # default=500,
                        # metavar="<image count>",
                        # help='Images to use for evaluation (defaults=500)')
                        
    parser.add_argument('--mrcnn_model', required=False,
                        default='last',
                        metavar="/path/to/weights.h5",
                        help="MRCNN model weights file: 'coco' , 'init' , or Path to weights .h5 file ")

    parser.add_argument('--mrcnn_exclude_layers', required=False,
                        nargs = '+',
                        type=str.lower, 
                        metavar="/path/to/weights.h5",
                        help="layers to exclude from loading from weight file" )
                        
    parser.add_argument('--mrcnn_logs_dir', required=True,
                        default='train_mrcnn',
                        metavar="/path/to/logs/",
                        help="MRCNN Logs and checkpoints directory (default=logs/)")

    parser.add_argument('--mrcnn_layers', required=False,
                        nargs = '+',
                        default=['mrcnn', 'fpn', 'rpn'], type=str.lower, 
                        metavar="/path/to/weights.h5",
                        help="MRCNN layers to train" )
                        
    parser.add_argument('--evaluate_method', required=False,
                        choices = [1,2,3],
                        default=1, type = int, 
                        metavar="<evaluation method>",
                        help="Detection Evaluation method : [1,2,3]")
                        
    parser.add_argument('--fcn_model', required=False,
                        default='last',
                        metavar="/path/to/weights.h5",
                        help="FCN model weights file: 'init' , or Path to weights .h5 file ")

    parser.add_argument('--fcn_logs_dir', required=False,
                        default='train_fcn',
                        metavar="/path/to/logs/",
                        help="FCN Logs and checkpoints directory (default=logs/)")

    parser.add_argument('--fcn_arch', required=False,
                        choices=['FCN32', 'FCN16', 'FCN8', 'FCN8L2', 'FCN32L2'],
                        default='FCN32', type=str.upper, 
                        metavar="/path/to/weights.h5",
                        help="FCN Architecture : fcn32, fcn16, or fcn8")

    parser.add_argument('--fcn_layers', required=False,
                        nargs = '+',
                        default=['fcn32+'], type=str.lower, 
                        metavar="/path/to/weights.h5",
                        help="FCN layers to train" )

    parser.add_argument('--fcn_losses', required=False,
                        nargs = '+',
                        default='fcn_BCE_loss', 
                        metavar="/path/to/weights.h5",
                        help="FCN Losses: fcn_CE_loss, fcn_BCE_loss, fcn_MSE_loss" )
                        
    parser.add_argument('--fcn_bce_loss_method', required=False,
                        choices = [1,2],
                        default=1, type = int, 
                        metavar="<BCE Loss evaluation method>",
                        help="Evaluation method : [1: Loss on all classes ,2: Loss on one class only]")
                        
    parser.add_argument('--fcn_bce_loss_class', required=False,
                        default=0, type = int, 
                        metavar="<BCE Loss evaluation class>",
                        help="Evaluation class")
                        

    parser.add_argument('--last_epoch', required=False,
                        default=0,
                        metavar="<last epoch ran>",
                        help="Identify last completed epcoh for tensorboard continuation")
                        
    parser.add_argument('--epochs', required=False,
                        default=1,
                        metavar="<epochs to run>",
                        help="Number of epochs to run (default=3)")
                        
    parser.add_argument('--steps_in_epoch', required=False,
                        default=1,
                        metavar="<steps in each epoch>",
                        help="Number of batches to run in each epochs (default=5)")

    parser.add_argument('--val_steps', required=False,
                        default=1,
                        metavar="<val steps in each epoch>",
                        help="Number of validation batches to run at end of each epoch (default=1)")
                        
    parser.add_argument('--batch_size', required=False,
                        default=5,
                        metavar="<batch size>",
                        help="Number of data samples in each batch (default=5)")                    

    parser.add_argument('--scale_factor', required=False,
                        default=4,
                        metavar="<heatmap scale>",
                        help="Heatmap scale factor")                    

    parser.add_argument('--lr', required=False,
                        default=0.001,
                        metavar="<learning rate>",
                        help="Learning Rate (default=0.001)")

    parser.add_argument('--opt', required=False,
                        default='adagrad', type = str.upper,
                        metavar="<optimizer>",
                        help="Optimization Method: SGD, RMSPROP, ADAGRAD, ...")
                        
    parser.add_argument('--sysout', required=False,
                        choices=['SCREEN', 'HEADER', 'ALL'],
                        default='screen', type=str.upper,
                        metavar="<sysout>",
                        help="sysout destination: 'screen', 'header' , 'all' (header == file) ")

    parser.add_argument('--new_log_folder', required=False,
                        default=False, action='store_true',
                        help="put logging/weights files in new folder: True or False")

    parser.add_argument('--coco_classes', required=False,
                        nargs = '+',
                        default=None, type=int, 
                        metavar="<active coco classes>",
                        help="<identifies active coco classes" )

    parser.add_argument('--dataset', required=False,
                        choices=['newshapes', 'newshapes2', 'coco2014'],
                        default='newshapes', type=str, 
                        metavar="<Toy dataset type>",
                        help="<identifies toy dataset: newshapes or newshapes2" )
    
    return parser
    
def display_input_parms(args):
    """Display Configuration values."""
    print("\n   Arguments passed :")
    print("   --------------------")
    for a in dir(args):
        if not a.startswith("__") and not callable(getattr(args, a)):
            print("   {:30} {}".format(a, getattr(args, a)))
    print("\n")
 
