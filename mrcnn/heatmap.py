"""
Mask R-CNN
Configurations and data loading code for MS COCO HEATMAP FILES.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

"""

import os, re
import time
import numpy as np
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from   mrcnn.config import Config
import mrcnn.utils as utils
# import mrcnn.model as modellib
import mrcnn.dataset as dataset


# ############################################################
# #  Configurations
# ############################################################

# class CocoConfig(Config):
    # """Configuration for training on MS COCO.
    # Derives from the base Config class and overrides values specific
    # to the COCO dataset.
    # """
    # # Give the configuration a recognizable name
    # NAME = "coco"

    # # We use a GPU with 12GB memory, which can fit two images.
    # # Adjust down if you use a smaller GPU.
    # IMAGES_PER_GPU = 2

    # # Uncomment to train on 8 GPUs (default is 1)
    # # GPU_COUNT = 8

    # # Number of classes (including background)
    # NUM_CLASSES = 1 + 80  # COCO has 80 classes

    
# class CocoInferenceConfig(CocoConfig):
    # # Set batch size to 1 since we'll be running inference on
    # # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 1
    # IMAGES_PER_GPU = 1
    # DETECTION_MIN_CONFIDENCE = 0
	
##------------------------------------------------------------------------------------
## Build Training and Validation datasets
##------------------------------------------------------------------------------------
def prep_heatmap_dataset(type, config, generator = False, shuffle = True, augment = False):
    # dataset_train, train_generator = coco_dataset(["train",  "val35k"], mrcnn_config)

    # if args.command == "train":
    # Training dataset. Use the training set and 35K from the validation set, as as in the Mask RCNN paper.
    dataset = HeatmapDataset()
    
    # dataset_test.load_coco(COCO_DATASET_PATH,  "train", class_ids=mrcnn_config.COCO_CLASSES)
    for i in type:
        dataset.load_heatmap(config.COCO_DATASET_PATH, config.COCO_HEATMAP_PATH, i )
    dataset.prepare()

    results =  dataset
    
    if generator:
        generator = fcn_data_generator(dataset, config, 
                                   batch_size=config.BATCH_SIZE,
                                   shuffle = shuffle, augment = augment) 
        results = [dataset, generator]
    return results



############################################################
##  heatmap Dataset Class extension
############################################################

class HeatmapDataset(dataset.Dataset):
    
    def load_heatmap(self, dataset_dir, heatmap_dir, subset, class_ids=None,
                  class_map=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir:    The root directory of the COCO heatmap dataset (coco2014_hetmap).
        subset:         What to load (train, val, minival, val35k)
        class_ids:      If provided, only loads images that have the given classes.
        class_map:      TODO: Not implemented yet. Supports maping classes from
                              different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        """
        # Path
        image_dir = os.path.join(dataset_dir, "train2014" if subset == "train" else "val2014")
        heatmap_dir = os.path.join(heatmap_dir, "train2014" if subset == "train" else "val2014")
#       image_dir = os.path.join(dataset_dir, "train2017" if subset == "train" lse "val2017")
        print('image_dir : ', image_dir,'\n heatmap_dir: ', heatmap_dir)
        # Create COCO object
        json_path_dict = {
            "train"  :  "annotations/instances_train2014.json",
            "val"    :  "annotations/instances_val2014.json",
            "minival":  "annotations/instances_minival2014.json",
            "val35k" :  "annotations/instances_valminusminival2014.json",
            "test"   :  "annotations/image_info_test2014.json"
        }
        print('subset: ', subset, 'json_path_dir: ', json_path_dict[subset])
        coco = COCO(os.path.join(dataset_dir, json_path_dict[subset]))
        
        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())
  
        ##--------------------------------------------------------------
        ## Get image ids - using COCO
        ##--------------------------------------------------------------
        
        # #All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())
        # Add classes to dataset.class_info structure
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        print(' image dir        : ', image_dir) 
        print(' json_path_dir    : ', os.path.join(dataset_dir, json_path_dict[subset]))
        print(' number of images : ', len(image_ids))
        print(' ImageIds[:10]    : ', image_ids[:10])
        
        ##--------------------------------------------------------------
        ## Add images to dataset.image_info structure
        ##-------------------------------------------------------------- 
        heatmap_notfound=  heatmap_found = 0
        print(heatmap_notfound, heatmap_found)
        for i in image_ids:
            # print('image id: ',i)
            heatmap_filename = 'hm_{:012d}.npz'.format(i)
            heatmap_path = os.path.join(heatmap_dir, heatmap_filename) 
            
            ## Only load image_info data structure for images where the corrsponding 
            ## heatmap .npz file exist
            if not os.path.isfile(heatmap_path):
                # print('file not found:::',heatmap_filename)
                heatmap_notfound += 1
            else:
                self.add_image(
                    "coco", image_id=i,
                    path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                    width=coco.imgs[i]["width"],
                    height=coco.imgs[i]["height"],
                    heatmap_path=heatmap_path,
                    annotations=coco.loadAnns(coco.getAnnIds(
                                                imgIds=[i], catIds=class_ids, iscrowd=None))
                    
                  )
                heatmap_found += 1
                
        print(' Images ids :', len(image_ids))
        print('    Corresponding heatmap found     :' , heatmap_found)
        print('    Corresponding heatmap not found :' , heatmap_notfound)
        print(' Total      :', heatmap_found + heatmap_notfound)
        #--------------------------------------------------------------
        # Get image ids - using walk on HEATMAP_PATH
        #--------------------------------------------------------------
        # print(' image dir        : ', image_dir) 
        # print(' json_path_dir    : ', os.path.join(dataset_dir, json_path_dict[subset]))
        # regex = re.compile(".*/\w+(\d{12})\.jpg")
        

        # image_ids = [] 
        # heatmap_files = next(os.walk(heatmap_dir))[2]
        # print('heat ap dir :' , heatmap_dir)
        
        # for hm_file in heatmap_files:
            # print(' Processing file: ', hm_file)
            # heatmap_path=os.path.join(heatmap_dir, hm_file) 
            # i = int(os.path.splitext(hm_file.lstrip('hm_'))[0])
            # loaddata = np.load(heatmap_path)
            # input_image_meta = loaddata['input_image_meta']
            # input_filename   = str(loaddata['dataset_name']) 
            # loaddata.close()

            # print(input_filename, type(input_filename), len(input_filename))
            # coco_filename = input_filename.replace('\\' , "/")
            # print(coco_filename)
            # regex_match  = regex.match(input_filename)            
            # # Add images to dataset.image_info structure
            # if regex_match:
                # coco_id = int(regex_match.group(1))
            # print(i, input_image_meta[:8],' ', input_filename, ' coco_id : ',coco_id)
                
            # self.add_image(
                # "coco", 
                # image_id=i,
                # path = input_filename,
                # height=input_image_meta[1],
                # width= input_image_meta[2],
                # heatmap_path=heatmap_path
              # )
            # image_ids.append(i)
                # # annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None)))
        # print(' number of images : ', len(image_ids))

        if return_coco:
            return coco

        
    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(self.__class__).image_reference(self, image_id)

            
    def load_image_heatmap(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        coco_file    = self.image_info[image_id]['path']
        heatmap_file = self.image_info[image_id]['heatmap_path']
        # print('Read from : ', coco_file) 
        # print('Read from : ', heatmap_file)
        loaddata = np.load(heatmap_file)
        # print(loaddata.keys())
        # for i in loaddata.keys():
            # print(i, loaddata[i].shape)
        # image = super(self.__class__,self).load_image(image_id) # skimage.io.imread(coco_file)
        # If grayscale. Convert to RGB for consistency.
        # if image.ndim != 3:
            # image = skimage.color.gray2rgb(image)
        return loaddata

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(self.__class__).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(self.__class__).load_mask(image_id)

        

    # The following two functions are from pycocotools with a few changes.

    
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m
        