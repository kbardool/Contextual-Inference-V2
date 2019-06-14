"""
Mask R-CNN
Dataset functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import numpy as np
import skimage.color
import skimage.io

############################################################
#  Dataset
############################################################

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG", "category": "background", "img_count" : 0}]
        self.source_class_ids = {}
        self.source_objs = {}

    def add_class(self, source, class_id, class_name, category = None, img_count = 0):
        '''
        Add class to dataset obj class_info attribute
        '''
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                info["img_count"]+= img_count
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id"    : class_id,
            "name"  : class_name,
            "category" : category if category is not None else '',
            "img_count": img_count
        })

    def add_image(self, source, image_id, path, **kwargs):
        """
        Add image info to dataset obj image_info attribute
        """

        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
            **kwargs    # added -7-05
        }
        image_info.update(kwargs)
  
        self.image_info.append(image_info)
         
        
    def image_reference(self, image_id):
        """
        Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """
        Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """
        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids   = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images  = len(self.image_info)
        self._image_ids  = np.arange(self.num_images)
        
        
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

        # add internal class id to class_info dictionary, build external-internal lookups
        self.ext_to_int_id = {}
        self.int_to_ext_id = {}
        # self.active_class_info = {}
        
        for cls_info in self.class_info:
            source_key = cls_info['source']+'.'+str(cls_info['id'])
            internal_id =  self.class_from_source_map[source_key]
            cls_info['internal_id'] = internal_id
            self.ext_to_int_id.setdefault(cls_info['id'], internal_id) 
            self.int_to_ext_id.setdefault(internal_id, cls_info['id'])
            # if cls_info.

         
        self.active_class_info = {}    
        
        for cls in  self.active_ext_class_ids:
            int_id = self.ext_to_int_id[cls]
            self.active_class_info.setdefault(int_id, {'name':self.class_names[int_id], 'ext_id':cls})
        
        self.active_class_ids  = [i for i in sorted(self.active_class_info)]
        
        self.build_category_to_class_map()
        self.build_category_to_external_class_map()
        
        print('Prepares complete')
        
    def build_category_to_class_map(self):
        self.category_to_class_map = {}
        for i in self.class_info:
            self.category_to_class_map.setdefault(i["category"],[]).append(i["internal_id"])
            #     print(i["category"], '   ',category_to_class_map[i["category"]])
        # ttl = 0 
        # for i in self.category_to_class_map:
            # print('{:15s} {:4d}  {} '.format(i, len(self.category_to_class_map[i]) , self.category_to_class_map[i]))
            # ttl += len(self.category_to_class_map[i])
        # print('Total classes: ', ttl)          
        return 
        
        
    def build_category_to_external_class_map(self):
        self.category_to_external_class_map = {}
        for i in self.class_info:
            self.category_to_external_class_map.setdefault(i["category"],[]).append(i["id"])
        
        # ttl = 0 
        # for i in self.category_to_external_class_map:
            # print('{:15s}   external ids {:4d}  {}'.format(i, len(self.category_to_external_class_map[i]) , self.category_to_external_class_map[i]))
            # ttl += len(self.category_to_class_map[i])
        # print('Total classes: ', ttl)          
        return 
        
    def display_active_class_info(self):
        print(' Active Class Information ')
        print('--------------------------')
        print(self.active_class_ids)
        for cls in self.active_class_ids:
            # class_id = self.map_source_class_id( "coco.{}".format(ext_cls))
            ext_cls = self.int_to_ext_id[cls]
            print( 'internal_class: ', cls,'ext_cls:',ext_cls, 'category-name:', self.class_info[cls]['category'],'-',self.class_info[cls]['name'])        

    def display_class_info(self):
        print(' Class Information ')
        print('-------------------')
        for cls_info in self.class_info:
            print('{:25s}  source: {:10s}   ext_id: {:3d}   internal_id: {:3d}  category: {:20s}  img_count: {:6d}' .format(
                    cls_info['name'], 
                    cls_info['source'], 
                    cls_info['id'], 
                    cls_info['internal_id'] , 
                    cls_info['category'] if cls_info['category'] is not None else 'n/a', 
                    cls_info["img_count"]))


                
    def map_source_class_id(self, source_class_id):
        """
        Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

        
    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

        
    def append_data(self, class_info, image_info):
        self.external_to_class_id = {}
        for i, c in enumerate(self.class_info):
            for ds, id in c["map"]:
                self.external_to_class_id[ds + str(id)] = i

        # Map external image IDs to internal ones.
        self.external_to_image_id = {}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info["ds"] + str(info["id"])] = i

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's availble online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        # print(' loading image id: ', image_id, ' file:',self.image_info[image_id]['path'], '   ', self.image_info[image_id]['id'])
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids
    