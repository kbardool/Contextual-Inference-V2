"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import time
import numpy as np
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco     import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools          import mask as maskUtils
import mrcnn.dataset  as dataset
import mrcnn.utils    as utils

from   mrcnn.config   import Config
from   mrcnn.datagen  import data_generator




##------------------------------------------------------------------------------------
## Build Training and Validation datasets
##------------------------------------------------------------------------------------
def prep_coco_dataset(type, config, load_coco_classes = None, class_ids = None, loadAnns = 'active_only',
                      generator = False, shuffle = True, augment = False, return_coco = False):
    # dataset_train, train_generator = coco_dataset(["train",  "val35k"], mrcnn_config)

    # if args.command == "train":
    # Training dataset. Use the training set and 35K from the validation set, as as in the Mask RCNN paper.
    dataset = CocoDataset()
    
    # dataset_test.load_coco(COCO_DATASET_PATH,  "train", class_ids=mrcnn_config.COCO_CLASSES)
    for i in type:
        dataset.load_coco(config.COCO_DATASET_PATH, i , class_ids = class_ids, loadAnns= loadAnns, 
                            load_coco_classes = load_coco_classes, return_coco = return_coco)
    
    # all datasets loaded, now prep the final dataset
    dataset.prepare()

    results =  dataset
    
    if generator:
        generator = data_generator(dataset, config, 
                                   batch_size=config.BATCH_SIZE,
                                   shuffle = shuffle, augment = augment) 
        results = [dataset, generator]
    return results


############################################################
#  Configurations
############################################################

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

    
class CocoInferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0
	
    
############################################################
#  COCO Dataset Class extension
############################################################

class CocoDataset(dataset.Dataset):
    
    def load_coco(self, dataset_dir, subset, load_coco_classes=None,
                  class_ids=None, class_map=None, return_coco=False, loadAnns = None):
        """Load a subset of the COCO dataset.
        dataset_dir:    The root directory of the COCO dataset.
        subset:         What to load (train, val, minival, val35k)
        class_ids:      If provided, only loads images that have the given classes.
        class_map:      TODO: Not implemented yet. Supports maping classes from
                              different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        """
        assert loadAnns in ['all_classes', 'active_only'], "loadAnns must be 'all_classes' or 'active_only' "
        if loadAnns == 'active_only':
            print('=====================================================================')
            print('          !!! Loading annotations for ACTIVE CLASSES ONLY !!!')
            print(' Dataset dir : ', dataset_dir, ' subset: ', subset)
            print('=====================================================================')
        else:
            print('=====================================================================')
            print('             Loading annotations for ALL Coco classes ...    ')
            print(' Dataset dir : ', dataset_dir, ' subset: ', subset)
            print('=====================================================================')

        # Path
        image_dir = os.path.join(dataset_dir, "train2014" if subset == "train" else "val2014")
        # image_dir = os.path.join(dataset_dir, "train2017" if subset == "train" lse "val2017")
        
        # Create COCO object
        json_path_dict = {
            "train"  :  "annotations/instances_train2014.json",
            "val"    :  "annotations/instances_val2014.json",
            "minival":  "annotations/instances_minival2014.json",
            "val35k" :  "annotations/instances_valminusminival2014.json",
            "test"   :  "annotations/image_info_test2014.json"
        }
        coco = COCO(os.path.join(dataset_dir, json_path_dict[subset]))
        
        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        ## Add classes to the class_info dictionary - load classes using the Coco internal class id
        for i in class_ids:
            cocoClassInfo = coco.loadCats(i)[0]
            img_count = len(coco.getImgIds(catIds=i))
            # print('num images: ', img_count)
            self.add_class("coco", i, cocoClassInfo["name"], cocoClassInfo["supercategory"], img_count = img_count)

            
        # All images or a subset?
        # if class_ids:
            # print(' load subset of classes: ', class_ids)
            # image_ids = []
            # for id in class_ids:
                # image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # # Remove duplicates
            # image_ids = list(set(image_ids))
        # else:
            # # load All image ids
            # print(' load all classes: ', class_ids)            
            # image_ids = list(coco.imgs.keys())
            
        if load_coco_classes is not None:
            print(' load subset of classes: ', load_coco_classes)
            image_ids = []
            for id in load_coco_classes:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # load All image ids
            print(' loading  all classes')            
            load_coco_classes = class_ids
            image_ids = list(coco.imgs.keys())

        self.active_ext_class_ids = sorted(load_coco_classes)
            
        print(' image dir            : ', image_dir) 
        print(' json_path_dir        : ', os.path.join(dataset_dir, json_path_dict[subset]))
        print(' number of images     : ', len(image_ids))
        print(' image_ids[:10]       : ', image_ids[:10])
        print(' image_ids[1000:1010] : ', image_ids[1000:1010])
        # print(' ClassIds     :', class_ids)

        ## determine which annotations to load based on the loadAnns parameter
        annotation_classes = class_ids if loadAnns == 'all_classes' else self.active_ext_class_ids 
        
        ## Add images to the image_info dictionary
        for i in image_ids:
            self.add_image(
                "coco", 
                image_id    = i,
                path        = os.path.join(image_dir, coco.imgs[i]['file_name']),
                width       = coco.imgs[i]["width"],
                height      = coco.imgs[i]["height"],               
                annotations = coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=annotation_classes, iscrowd=None))) 
        
        if return_coco:
            self.source_objs[subset] = coco
            return coco

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
            return super().load_mask(image_id)

        instance_masks = []
        class_ids = []
        
        ## get image annotations
        annotations = self.image_info[image_id]["annotations"]
        
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        
        for annotation in annotations:
            class_id = self.map_source_class_id( "coco.{}".format(annotation['category_id']))
            # print("coco.id: {} --> class_id : {}  - {} ".format(annotation['category_id'],class_id, self.class_names[class_id]))
            if class_id:
                m = self.annToMask(annotation, image_info["height"], image_info["width"])
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
            # super(self.__class__, self) is equivalent to super() 
            return super().load_mask(image_id)

    def display_annotation_info(self, image_ids):
        if not isinstance(image_ids, list):
            image_ids = [image_ids]
    
        for image_id in image_ids:
            print()
            print('IMAGE_ID : ', image_id)   
            annotations = self.image_info[image_id]["annotations"]
            for annotation in annotations:
                class_id = self.map_source_class_id( "coco.{}".format(annotation['category_id']))
                print("ext.id: {} --> {} - {} ".format(annotation['category_id'],class_id, self.class_names[class_id]))
                
    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super().image_reference(self, image_id)

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


    
############################################################
#  COCO Evaluation - Build Results
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results

############################################################
#  Evaluate Coco
############################################################

def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset:    A Dataset object with valiadtion data
    eval_type:  "bbox" or "segm" for bounding box or segmentation evaluation
    limit:      if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"], r["masks"])
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


