import math
import random
import copy
import numpy as np
import cv2
from matplotlib      import colors 
from mrcnn.visualize import display_images
from mrcnn.dataset   import Dataset
# from mrcnn.shapes    import ShapesConfig
from mrcnn.datagen   import load_image_gt, data_generator
from mrcnn.visualize import draw_boxes
from mrcnn.config    import Config
from mrcnn.dataset   import Dataset
from mrcnn.utils     import non_max_suppression, mask_string
from mrcnn.Image     import Image, draw_object, order_shapes_by_bottom_edge, display_shapes
from importlib       import reload
# import mrcnn.utils as utils
import pprint
p4 = pprint.PrettyPrinter(indent=4, width=100)
p8 = pprint.PrettyPrinter(indent=8, width=100)
pp = p4

class ObjectClass(object):
    '''
    Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    '''
    def __init__(self, Dataset, name, min_x, max_x, min_y, max_y ):
        print(' Generate new object type :', name, ' for ', Dataset.config.NAME)
        self.name   = name 
        
        
 
##------------------------------------------------------------------------------------
## Build  NewShapes Training and Validation datasets
##------------------------------------------------------------------------------------
def prep_newimage_dataset(config, image_count, shuffle = True, augment = False, generator = False):
    '''
    '''
    dataset = NewImagesDataset(config)
    dataset.load_images(image_count) 
    dataset.prepare()

    results = dataset
    
    if generator:
        generator = data_generator(dataset, config, 
                                   batch_size=config.BATCH_SIZE,
                                   shuffle = True, augment = False) 
        return [dataset, generator]
    else:
        return results
    


class NewImagesConfig(Config):
    '''
    Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    '''
    # Give the configuration a recognizable name
    NAME = "newshapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # background + 6 shapes
    MAX_SHAPES_PER_IMAGE = 15
    MIN_SHAPES_PER_IMAGE = 1

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    IMAGE_BUFFER  = 20
    
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


    
class NewImagesDataset(Dataset):
    '''
    Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    '''
    def __init__(self, config ):
        super().__init__()
        self.config = config
        self.config.HEIGHT = config.IMAGE_SHAPE[0]
        self.config.WIDTH  = config.IMAGE_SHAPE[1]
#        self.config.min_shapes_per_image = config.MIN_SHAPES_PER_IMAGE
#        self.config.max_shapes_per_image = config.MAX_SHAPES_PER_IMAGE                                                  
        
        buffer = self.config.IMAGE_BUFFER  
        height = self.config.HEIGHT        
        width  = self.config.WIDTH  

        self.add_class("newshapes", 1, "person")
        self.add_class("newshapes", 2, "car")
        self.add_class("newshapes", 3, "sun")
        self.add_class("newshapes", 4, "building")
        self.add_class("newshapes", 5, "tree")
        self.add_class("newshapes", 6, "cloud")
        self.add_class("newshapes", 7, "airplane")
        self.add_class("newshapes", 8, "truck")
        self.active_ext_class_ids=[1,2,3,4,5,6,7,8]
        self.draw_priority_list = ['sun', 'cloud', 'airplane']

        self.config.max_dim = {}
        self.config.min_dim = {}
        self.config.Min_Y   = {}
        self.config.Max_Y   = {}
        self.config.Min_X   = {}
        self.config.Max_X   = {}
        self.config.Min_X['_default'] =  buffer
        self.config.Max_X['_default'] =  height - buffer - 1
        self.config.Min_Y['_default'] =  buffer
        self.config.Max_Y['_default'] =  height - buffer - 1

        self.config.min_dim['building'] =  15
        self.config.max_dim['building'] =  25
        self.config.Max_X  ['building'] =  width - buffer
        self.config.Min_Y  ['building'] =  height //3 
        self.config.Min_X  ['building'] =  buffer
        self.config.Max_Y  ['building'] =  2 * height //3   ##* min_range_y

        self.config.min_dim['person'] =  8     ## 10
        self.config.max_dim['person'] =  16    ## 20
        self.config.Min_Y  ['person'] =  height //2 
        self.config.Max_Y  ['person'] =  height - buffer - 1
        self.config.Min_X  ['person'] =  0
        self.config.Max_X  ['person'] =  height - buffer - 1

        self.config.min_dim['car' ]   =  6
        self.config.max_dim['car' ]   =  13
        self.config.Min_X  ['car' ]   =  buffer
        self.config.Max_X  ['car' ]   =  height - buffer - 1
        self.config.Min_Y  ['car' ]   =  height //2
        self.config.Max_Y  ['car' ]   =  height - buffer - 1
        
        self.config.min_dim['sun']    =  4
        self.config.max_dim['sun']    =  10
        self.config.Min_X  ['sun']    =  buffer //3                
        self.config.Max_X  ['sun']    =  width - (buffer//3) - 1  
        self.config.Min_Y  ['sun']    =  buffer //3
        self.config.Max_Y  ['sun']    =  height //5    ##* min_range_y
        
        self.config.max_dim['tree']   =  30    ## 36
        self.config.min_dim['tree']   =  9     ## 9
        self.config.Min_X  ['tree']   =  buffer
        self.config.Max_X  ['tree']   =  height - buffer - 1
        self.config.Min_Y  ['tree']   =  height // 3
        self.config.Max_Y  ['tree']   =  width - (buffer) - 1    ##* min_range_y

        self.config.min_dim['cloud']  =  3
        self.config.max_dim['cloud']  =  13
        self.config.Min_X  ['cloud']  =  buffer//2                 
        self.config.Max_X  ['cloud']  =  width - (buffer//2) - 1    
        self.config.Min_Y  ['cloud']  =  buffer
        self.config.Max_Y  ['cloud']  =  height //4
        
        self.config.min_dim['airplane']  =  5   ## 4
        self.config.max_dim['airplane']  =  11   ## 10
        self.config.Min_X  ['airplane']  =  buffer//2                 
        self.config.Max_X  ['airplane']  =  width - (buffer//2) - 1    
        self.config.Min_Y  ['airplane']  =  buffer
        self.config.Max_Y  ['airplane']  =  height //4

        self.config.min_dim['truck']  =  7
        self.config.max_dim['truck']  =  14
        self.config.Min_X  ['truck']  =  buffer//2                 
        self.config.Max_X  ['truck']  =  width - (buffer//2) - 1    
        self.config.Min_Y  ['truck']  =  buffer //4
        self.config.Max_Y  ['truck']  =  height - height //4

        # buffer = self.config.IMAGE_BUFFER
        # height = self.config.HEIGHT
        # width  = self.config.WIDTH

        ## Types of objects in this Dataset.....                          
        # person = ObjectClass(self, 'person', buffer, height - buffer, height//2, height - buffer)
        
        print('Active Class Info in ', self.config.NAME)
        print('------------------------------------')
        pp.pprint(self.class_info)

    
    ##---------------------------------------------------------------------------------------------
    ## load_images
    ##---------------------------------------------------------------------------------------------
    def load_images(self, num_new_images, verbose = False):
        '''
        Generate the requested number of synthetic images.
        num_new_images: number of images to generate.
        height, width: the size of the generated images.
        '''
        
        self.config.object_choices = [ cls_inf['name'] for cls_inf in self.class_info if cls_inf['id'] > 0]
        if verbose:
            print('Class Object Choices')
            print('--------------------')
            print(self.config.object_choices)
       




        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        curr_img_count = len(self.image_info)
        print(' Dataset test currently has :', curr_img_count, ' images defined...')
        print(' Adding ', num_new_images , 'images')
        for image_id in range(curr_img_count, curr_img_count + num_new_images ):
            if verbose:
                print()
                print(' Add image ---> ',image_id )
                print('-----------------------------')
            else:
                if image_id % 25 == 0:
                    print(' Add image ---> ',image_id )
                    print('-----------------------------')
                    
            image = Image(image_id, self.config, verbose)    
            self.add_image("newshapes", image_id = image_id, path = None, **image.image_data)    

    ##---------------------------------------------------------------------------------------------
    ## build_image
    ##---------------------------------------------------------------------------------------------
    def display_image(self, image_ids = None, display = False):
        '''
        retrieves images for  a list of image ids, that can be passed to model detect() functions
        '''
        images = []
        if not isinstance(image_ids, list):
            image_ids = [image_ids]

        for image_id in image_ids:
            images.append(self.load_image(image_id))

        display_images(images, titles = ['id: '+str(i)+' ' for i in image_ids], cols = 5, width = 25)
        return 


    ##---------------------------------------------------------------------------------------------
    ## image_reference
    ##---------------------------------------------------------------------------------------------
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)    

    ##---------------------------------------------------------------------------------------------
    ## load_mask
    ##---------------------------------------------------------------------------------------------
    def load_mask(self, image_id):
        '''
        Generate instance masks for shapes of the given image ID.
        '''
        # print(' ===> Loading mask info for image_id : ',image_id)
        info   = self.image_info[image_id]
        shapes = info['shapes']
        
        # print('\n Load Mask information (shape, (color rgb), (x_ctr, y_ctr, size) ): ')
        # p4.pprint(info['shapes'])
        count  = len(shapes)
        mask   = np.zeros([info['height'], info['width'], count], dtype=np.uint8)

        # print(' Shapes obj mask shape is :',mask.shape)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i + 1] = draw_object(mask[:, :, i:i + 1].copy(), shape, dims, 1)
        
        #----------------------------------------------------------------------------------
        ## Handle occlusions 
        #   Occlusion starts with the last object an list and in each iteration of the loop 
        #   adds an additional  object. Pixes assigned to objects are 0. Non assigned pixels 
        #   are 1
        #-----------------------------------------------------------------------------------
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            # print('------------------------------------')
            # print(' i is :', i, 'BEFORE Mask - shape: ', mask[:, :, i:i + 1].shape, ' Mask all zeros: ', ~np.any(mask[:, :, i:i + 1]))
            # print('------------------------------------')            
            # print(mask_string(mask[:,:,i]))
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        
        # Assign class Ids to each shape --- Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask, class_ids.astype(np.int32)
    
    ##---------------------------------------------------------------------------------------------
    ## load_image
    ##---------------------------------------------------------------------------------------------
    def load_image(self, image_id, verbose = False):
        '''
        Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but in this case it
        generates the image on the fly from the specs in image_info.
        '''
        if verbose:
            print(' ===> Loading image * image_id : ',image_id)

        info = self.image_info[image_id]
        # pp.pprint(info)
        # bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        
        horizon_color = np.array(info['horizon'][2]).reshape([1, 1, 3])
        horizon_line  = info['horizon'][0]
        ground_color  = np.array(info['ground'][2]).reshape([1, 1, 3])
        ground_line   = info['ground'][0]
        image         = np.zeros([info['height'], info['width'], 3], dtype=np.uint8)

        image[:horizon_line,:,:] =  horizon_color
        image[ground_line:,:,:]  =  ground_color
        # print(' image shape ', image.shape)            
        # print(" Load Image : Shapes ")
        # p4.pprint(info['shapes'])
        
        for shape, color, dims in info['shapes']:
        # for shape, color, dims in draw_list:
           image = draw_object(image, shape, dims, color, verbose)        

        return image
