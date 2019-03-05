import math
import random
import copy
import numpy as np
import cv2
from matplotlib      import colors
from mrcnn.visualize import display_images
from mrcnn.dataset   import Dataset
from mrcnn.datagen   import load_image_gt, data_generator
from mrcnn.visualize import draw_boxes
from mrcnn.config    import Config
from mrcnn.dataset   import Dataset
from mrcnn.utils     import non_max_suppression, mask_string
# import mrcnn.utils as utils
import pprint
p4 = pprint.PrettyPrinter(indent=4, width=100)
p8 = pprint.PrettyPrinter(indent=8, width=100)
pp = p4



##---------------------------------------------------------------------------------------------
## draw_shape
##---------------------------------------------------------------------------------------------
def draw_object(image, shape, dims, color, verbose = False):
    """Draws a shape from the given specs."""

    # Get the center x, y and the size s
    cx, cy, sx, sy = dims

    if verbose:
        print(' draw_image() Shape : {:20s}   Cntr (x,y): ({:3d} , {:3d})    Size_x: {:3d}   Size_y: {:3d} {}'.format(shape,cx,cy,sx, sy,color))
        print('  Draw ', shape, ' Color:', color, ' shape', type(color))
        print('    CX :', cx, 'CY:', cy , 'sx: ',sx , 'sy: ', sy)

    if shape == "building":
        x1 = cx - sx
        y1 = cy - sy
        x2 = cx + sx
        y2 = cy + sy
        image = cv2.rectangle(image, (x1,y1), (x2, y2), color, -1)
         
#             print('X :', x, 'y:', y , '     sx: ',sx , 'sy: ', sy, 'hs:', hs)

    elif shape == "car":
        body_y  = sy //3
        wheel_r = 2* sy // 5
        body_y  = int(2 * sy /5)
        wheel_r = int(2 * sy /5)
        
        wheel_x = sx //2
        top_x   = sx //4
        bot_x   = 3*sx //4
        if verbose: 
            print('    CX :', cx, ' CY:', cy , '     SX: ',sx , 'sy: ', sy)
            print('    Car Top(y) : ', cy - sy   , '  Bottom(y)   : ', cy + body_y + wheel_r, ' Left(x):', cx - sx, ' Right(x) : ', cx+sx) 
            print('    Cab top    : ', cy - sy   , '  Cab bottom  : ', cy - body_y , ' Cab Height: ', sy - body_y)
            print('    Car height : ', 2*sy      , '  Half Car hgt: ', sy,  ' Half Body height (body_y): ', body_y , ' Half body width : ', sx)
            print('    wheel_x    : ', wheel_x   , '  wheel_r     : ', wheel_r)
            print('    Color:', color)
        
        
        image = cv2.rectangle(image, (cx - sx, cy - body_y), (cx + sx, cy + body_y), color, -1)
        image = cv2.circle(image, (cx - wheel_x , cy + body_y), wheel_r, color, -1)
        image = cv2.circle(image, (cx + wheel_x , cy + body_y), wheel_r, color, -1)
        # Top cab    
        points = np.array([[(cx - top_x,  cy - sy    ), (cx + top_x, cy - sy),
                            (cx + bot_x,  cy - body_y), (cx - bot_x, cy - body_y), ]], dtype=np.int32)
        image = cv2.fillPoly(image, points, color)

    elif shape == "oldcar":
        body_y  = sy //3
        wheel_x = sx //2
        wheel_r = sx //5
        top_x   = sx //4
        bot_x   = 3*sx //4
        if verbose:
            print('    Car Top(y): ', cy - sy , '  Bottom(y) : ', cy + body_y + wheel_r, ' Left(x):', cx - sx, ' Right(x) : ', cx+sx) 
            print('    Half Car hgt: ', sy,  ' Half Body height: ', body_y , ' Half body width : ', sx)
        
        image = cv2.rectangle(image, (cx - sx, cy - body_y), (cx + sx, cy + body_y), color, -1)
        image = cv2.circle(image, (cx - wheel_x , cy + body_y), wheel_r, color, -1)
        image = cv2.circle(image, (cx + wheel_x , cy + body_y), wheel_r, color, -1)
        # Top cab
        points = np.array([[(cx - top_x , cy - sy),   (cx + top_x, cy - sy),
                            (cx + bot_x,  cy - body_y),(cx - bot_x, cy - body_y), ]], dtype=np.int32)
        image = cv2.fillPoly(image, points, color)

    elif shape == "person":
#             print('X :', x, 'y:', y , 'sx: ',sx , 'sy: ', sy, 'hs:', hs)

        hy = sy // 4   # head height
        by = sy - hy   # body height
        # torso
        image = cv2.rectangle(image, (cx - sx, cy - by), (cx + sx, cy + by//4), color, -1)
        # legs
        image = cv2.rectangle(image, (cx - sx, cy + by//4), (cx - sx +sx//4, cy + by), color, -1)
        image = cv2.rectangle(image, (cx + sx - sx//4, cy + by//4), (cx + sx, cy + by), color, -1)
        #head
        image = cv2.circle(image, (cx , cy -(by+hy) ), sx, color, -1)
        if verbose:
            print('    Person  Top(y) : ', cy -(by+hy)+sx , '  Bottom(y) : ', cy+by, ' Left(x):', cx - sx, ' Right(x) : ', cx+sx)

    elif shape == "tree":
        sin_t = math.sin(math.radians(60))

        full_height = 2 * sy
        ty = full_height //5                # trunk length
        by = (full_height - ty) // 2        # half body length
        bx = int(by / sin_t) // 2  # half body width 
        tx = bx//5                 # trunk width
        # orde of points: top, left, right
        points = np.array([[(cx, cy - by),                    ## top 
                            (cx - bx, cy + by),     ## left
                            (cx + bx, cy + by),     ## right 
                            ]], dtype=np.int32)
        image = cv2.fillPoly(image, points, color)
        image = cv2.rectangle(image,(cx-tx,cy+by), (cx+tx, cy+by+ty),color, -1)
        if verbose:
            print('    Tree  Top(y) : ', cy - by , '  Bottom(y) : ', cy+by+ty, ' Left(x):', cx - bx, ' Right(x) : ', cx+bx)
            print('    Trunk Length : ', ty, '  Body Length :', by, '    Half Body Width: ', bx, '  Half Trunk Width: ', tx)

    elif shape == 'truck' :
        body_y  = sy //5
        wheel_x = sx //2
        top_x   = sx //4
        bot_x   = 3*sx //4
        wheel_r = 2 * sy //3
        cab_top_x = (3 * sx)//4
        xy_ratio = sx/sy
        if verbose:
            print('CX :', cx, 'CY:', cy , 'X to Y Ratio:', xy_ratio, '     sx: ',sx , 'sy: ', sy, 'body_y ', body_y,' wheel_r :', wheel_r)

        ## cab
        image = cv2.rectangle(image, (cx + cab_top_x, cy -sy), (cx + sx, cy ), color, -1)  


        ## Rear Wheel 
        image = cv2.circle(image, (cx - (3*sx//4) , cy + body_y), wheel_r, color, -1)     
        image = cv2.circle(image, (cx - (  sx//4) , cy + body_y), wheel_r, color, -1)     
        ## Front Wheel
        image = cv2.circle(image, (cx + (2*sx//3) , cy + body_y ), wheel_r, color, -1)     
        ## Wheels - Older method
        #     image = cv2.circle(image, (cx - wheel_x , cy ), wheel_r, color, -1)     
        #     image = cv2.circle(image, (cx - wheel_x , cy + body_y), wheel_r, color, -1)     
        ## Front Wheel
        #     image = cv2.circle(image, (x + wheel_x , y + body_y), wheel_r, color, -1)     

        ## lower bed
        image = cv2.rectangle(image, (cx - sx, cy - body_y), (cx + sx, cy + body_y), color, -1)    
        ## Upper bed
        if isinstance(color, int):
            image = cv2.rectangle(image, (cx - sx+1, cy - (2*sy//3))  , (cx + cab_top_x -1, cy -body_y), 1, -1)
        else:
            image = cv2.rectangle(image, (cx - sx+1, cy - (2*sy//3))  , (cx + cab_top_x -1, cy -body_y), (181,185,189), -1)

        


    elif shape =='airplane':
        point_list_16_by_8 = [(4,0), (2,2), (2,6), (0,10), (0,12), (2,10), (2,12), (0,14), (0,16), (4,14), (4,10), (5,12), (5,10), (4,6) ]        
        step_sz = sy / 4
        x_rng = np.arange( -sx , (sx + 1), step_sz)
        y_rng = np.arange( -sy , (sy + 1), step_sz)
        x_values, y_values = np.meshgrid(x_rng, y_rng)
        x_values += cx
        y_values += cy
        if verbose:
            print('            x_rng: ', x_rng)
            print('  adjusted for CX: ', x_rng+cx)
            print('            y_rng: ', y_rng)
            print('  adjusted for CY: ', y_rng+cy)
        points = [(x_values[i,j], y_values[i,j]) for (i,j) in point_list_16_by_8]
        points = np.array([points], dtype=np.int32)
        # print(' points list ')
        # print(points)
        # print(' points np.array ', points.shape)
        # print(points)
        image = cv2.fillPoly(image, points, color)             

    elif shape == "sun":
        image = cv2.circle(image, (cx, cy), sx, color, -1)

    elif shape == "cloud":
        image = cv2.ellipse(image,(cx,cy),(sx, sy),0,0,360,color,-1)

    if shape == "square":
        image = cv2.rectangle(image, (cx - sx, cy - sy), (cx + sx, cy + sy), color, -1)

    elif shape == "rectangle":
        image = cv2.rectangle(image, (cx - sx, cy - sy), (cx + sx, cy + sy), color, -1)
#             print('X :', x, 'y:', y , '     sx: ',sx , 'sy: ', sy, 'hs:', hs)

    elif shape == "circle":
        image = cv2.circle(image, (cx, cy), sx, color, -1)

    elif shape == "ellipse":
        image = cv2.ellipse(image,(cx,cy),(sx, sy),0,0,360,color,-1)

    elif shape == "triangle":
        sin60 = math.sin(math.radians(60))
        # orde of points: top, left, right
        points = np.array([[(cx, cy - sx),
                            (cx - (sx / sin60), cy + sx),
                            (cx + (sx / sin60), cy + sx),
                            ]], dtype=np.int32)
        image = cv2.fillPoly(image, points, color)

    return image



class Image():
    '''
    Creates random specifications of an image with multiple objects.
    Returns the background color of the image and a list of shape
    specifications that can be used to draw the image.

    returns
    -------
    bg_color:    general back ground color
    shapes  :    list of objects
    '''
    custom_colors = {}
    custom_colors['sun'] = [ (colors.to_rgba_array(i)*255)[0,:3] for i in 
                                                                 [colors.CSS4_COLORS['lightyellow'],
                                                                  colors.CSS4_COLORS['yellow'],
                                                                  colors.CSS4_COLORS['gold'],
                                                                  colors.CSS4_COLORS['orange'],
                                                                  colors.CSS4_COLORS['darkorange']
                                                                  ]]

    custom_colors['cloud']  = [ (colors.to_rgba_array(i)*255)[0,:3] for i in 
                                                                 [colors.CSS4_COLORS['silver'],
                                                                  colors.CSS4_COLORS['lightslategray'],
                                                                  colors.CSS4_COLORS['lightgray'],
                                                                  colors.CSS4_COLORS['gainsboro'],
                                                                  colors.CSS4_COLORS['whitesmoke'],
                                                                  colors.CSS4_COLORS['white'],
                                                                  colors.CSS4_COLORS['snow'],
                                                                  colors.CSS4_COLORS['ghostwhite'],
                                                                  ]]

    custom_colors['ground'] = [ (colors.to_rgba_array(i)*255)[0,:3] for i in 
                                                                 [colors.CSS4_COLORS['sandybrown'],
                                                                  colors.CSS4_COLORS['peachpuff'],
                                                                  colors.CSS4_COLORS['peru'],
                                                                  colors.CSS4_COLORS['linen'],
                                                                  colors.CSS4_COLORS['burlywood'],
                                                                  colors.CSS4_COLORS['antiquewhite'],
                                                                  colors.CSS4_COLORS['moccasin'],
                                                                  ]]

    custom_colors['sky']   = [ (colors.to_rgba_array(i)*255)[0,:3] for i in 
                                                                 [colors.CSS4_COLORS['lightblue'],
                                                                  colors.CSS4_COLORS['deepskyblue'],
                                                                  colors.CSS4_COLORS['skyblue'],
                                                                  colors.CSS4_COLORS['lightskyblue'],
                                                                  colors.CSS4_COLORS['steelblue'],
                                                                  colors.CSS4_COLORS['aliceblue']
                                                                  ]]

    custom_colors['person'] = [ (colors.to_rgba_array(i)*255)[0,:3] for i in 
                                                                  [colors.CSS4_COLORS['black'],
                                                                   colors.CSS4_COLORS['tan'],
                                                                   colors.CSS4_COLORS['sienna'],
                                                                   colors.CSS4_COLORS['saddlebrown'],
                                                                   colors.CSS4_COLORS['khaki'],
                                                                   colors.CSS4_COLORS['chocolate'],
                                                                  ]]



    custom_colors['tree'] = [ (colors.to_rgba_array(i)*255)[0,:3] for i in  
                                                                 [colors.CSS4_COLORS['forestgreen'],
                                                                  colors.CSS4_COLORS['limegreen'],
                                                                  colors.CSS4_COLORS['darkgreen'],
                                                                  colors.CSS4_COLORS['green'],
                                                                  colors.CSS4_COLORS['seagreen'],
                                                                  colors.CSS4_COLORS['olive'],
                                                                  colors.CSS4_COLORS['olivedrab'],
                                                                  colors.CSS4_COLORS['yellowgreen'],
                                                                  colors.CSS4_COLORS['darkolivegreen']
                                                                 ]]

    # possible_choices = ['sun':1,'car':3, 'building':3, 'person':1, 'cloud':2, 'tree':4]
    # possible_choices = {'sun':1,'car':2, 'tree':5}
    custom_colors_keys   = list(custom_colors.keys())
    possible_choices     = {'sun':1,   'car':3 , 'tree':5, 'person':5, 'cloud':3, 'building':3, 'airplane':3, 'truck':3} 
    object_priority_list = ['building','tree','car']
    BUILD_MAX_TRIES      = 7
    person_car_gap       = 10   # fixed spread between car and person

    print(' Init Image Class - Possible Object Choices: ', possible_choices)
    print(' Init Image Class - Custom Color Keys      : ', custom_colors_keys)
    print(' Init Image Class - Object Priority List   : ', object_priority_list)
    print(' Init Image Class - BUILD_MAX_TRIES        : ', BUILD_MAX_TRIES)
    print(' Init Image Class - person_car_gap         : ', person_car_gap)

    def __init__(self, image_id,  datasetConfig ,verbose = False):

        super().__init__()
        self.image_id           = image_id
        self.config             = datasetConfig
        self.height             = self.config.HEIGHT
        self.width              = self.config.WIDTH
        self.buffer             = self.config.IMAGE_BUFFER
        self.max_range_y        = self.config.HEIGHT - self.config.IMAGE_BUFFER
        self.horizon            = self.build_horizon()
        self.ground             = self.build_ground()
        self.rightmost_building = 0
        self.rightmost_vehicle  = 0 
        self.rightmost_tree     = 0 
        self.leftmost_building  = self.width
        self.leftmost_vehicle   = self.width
        self.leftmost_tree      = self.width
        self.lowest_building    = self.horizon[0]
        self.lowest_vehicle     = self.horizon[0] 
        self.lowest_tree        = self.horizon[0] 
        self.highest_building   = self.height
        self.highest_vehicle    = self.height
        self.highest_tree       = self.height
        self.first_tree         = (0,0)
        self.bg_color           = np.array([random.randint(0, 255) for _ in range(3)])
        self.shapes             = []
        self.selected_counts    = {}
        self.allowed_counts     = {}
        self.built_counts       = {}
        self.possible_choices   = []   
        self.vehicles           = []
        self.object_list        = []
        self.occlusion_mask     = np.ones([self.config.HEIGHT, self.config.WIDTH], dtype =np.uint8)
        for shape in Image.possible_choices:
            self.possible_choices.append(shape)
            self.allowed_counts[shape]  = Image.possible_choices[shape]
            self.selected_counts[shape] = 0
            self.built_counts[shape]    = 0

        # Generate a few random shapes and record their bounding boxes
        N = random.randint(self.config.MIN_SHAPES_PER_IMAGE, self.config.MAX_SHAPES_PER_IMAGE)    # number to shapes in image

        
        for _ in range(N):
            shape       = random.choice(self.possible_choices)
            self.object_list.append(shape)
            self.selected_counts[shape] += 1
            
            if self.selected_counts[shape] == 1:
                self.remove_conflicting_choices(shape)

            if self.selected_counts[shape] == self.allowed_counts[shape]:
                # print(' Max number of ',shape, ' reached - remove from possible choices ')
                self.possible_choices.remove(shape)
                # print(' Possible choices now: ', self.possible_choices)
            
            # If no more possible choices, stop selection
            if not self.possible_choices :
                break


        # if verbose:        
        print(' Initial Number of objects for image: ', N)
        print(' Initial list of selected objects for image: ', self.object_list)

        ## Build prioritized objects in object list first
        for priority_obj in Image.object_priority_list:
            while True:
                try:
                    pos = self.object_list.index(priority_obj)
                except ValueError:
                    break
                else:
                    object = self.object_list.pop(pos)
                    # print('Found ', object ' in position : ',pos,'  remaining list: ', object_list)
                    self.build_test_add_object(object, verbose = True)
        
        ## Build remaining objects in object list
        if verbose:
            print('remaining list: ', self.object_list)
        for object in self.object_list:
            self.build_test_add_object(object, verbose = True)

        print(' Completed list of objects:')
        display_shapes(self.shapes)
        ##--------------------------------------------------------------------------------
        ## Reorder shapes to simulate overlay (nearer shapes cover farther away shapes)
        ## order shape objects based on closeness to bottom of image (-1) or top (+1)
        ## this will result in items closer to the viewer have higher priority in NMS
        ##--------------------------------------------------------------------------------
        # tmp_shapes = order_shapes_by_bottom_edge(self.shapes)
        tmp_shapes = copy.copy(self.shapes)
        print(' List of objects after 1st Sort:')
        display_shapes(tmp_shapes)

        ##-------------------------------------------------------------------------------
        ## find and remove shapes completely covered by other shapes
        ##-------------------------------------------------------------------------------
        hidden_shape_ixs = self.find_hidden_shapes(tmp_shapes, verbose = True)
        if len(hidden_shape_ixs) > 0:
            non_hidden_shapes = [s for i, s in enumerate(tmp_shapes) if i not in hidden_shape_ixs]
            print('    ===> Image Id : (',image_id, ')   ---- Zero Mask Encountered ')
            # print('    ------ Original Shapes ------' )
            # p8.pprint(tmp_shapes)
            # print('    ------ shapes after removal of totally hidden shapes ------' )
            # p8.pprint(non_hidden_shapes)
            # print('    Number of shapes now is : ', len(non_hidden_shapes))
        else:
            non_hidden_shapes = tmp_shapes

        print(' List of objects suppresion of completely hidden shapes:')
        display_shapes(non_hidden_shapes)

        ##--------------------------------------------------------------------------------
        ## Non Maximal Suppression
        ## - build boxes for to pass to non_max_suppression
        ## - Suppress occulsions more than 0.3 IoU
        ##   Apply non-max suppression with 0.3 threshold to avoid shapes covering each other
        ##   object scores (which dictate the priority) are assigned in the order they were created
        ##--------------------------------------------------------------------------------
        # keep_ixs =  debug_non_max_suppression(np.array(boxes), np.arange(N), 0.29, verbose)
        keep_ixs =  custom_non_max_suppression(non_hidden_shapes, 0.29, verbose)

        tmp_shapes = [s for i, s in enumerate(non_hidden_shapes) if i in keep_ixs]
        if verbose:
            print('===> Original number of shapes {} '.format(N))
            for i in non_hidden_shapes:
                print('     ', i)
            print('     Number of shapes after NMS {}'.format(len(tmp_shapes)))
            for i in tmp_shapes:
                print('     ', i)
        
        print(' List of objects after NMS:')
        display_shapes(tmp_shapes)


        ##--------------------------------------------------------------------------------
        ## Reorder shapes to simulate overlay (nearer shapes cover farther away shapes)
        ## order shape objects based on closeness to bottom of image (-1) or top (+1)
        ## this will result in items closer to the viewer have higher priority in NMS
        ##--------------------------------------------------------------------------------
        # self.shapes = order_shapes_by_bottom_edge(tmp_shapes)
        # self.shapes = tmp_shapes


        ##--------------------------------------------------------------------------------
        ## Reorder shapes to simulate overlay (nearer shapes cover farther away shapes)
        ## order shape objects based on closeness to bottom of image (-1) or top (+1)
        ## this will result in items closer to the viewer have higher priority in NMS
        ##--------------------------------------------------------------------------------
        ## Build prioritized objects in object list first
        # object_list = []
        # object_list = copy.copy(self.image_info[image_id]['shapes'])
        shape_list = [i[0] for i in tmp_shapes]
        print('original list')
        display_shapes(tmp_shapes)
        draw_list = [] 

        for priority_obj in ['sun', 'cloud', 'airplane']:
        # for priority_obj in self.draw_priority_list:
            while True:
                try:
                    pos = shape_list.index(priority_obj)
                except ValueError:
                    break
                else:
                    draw_list.append(tmp_shapes.pop(pos))
                    shape_list.pop(pos)
                    # print('Found ', priority_obj, ' in position : ',pos,'  remaining list: ',shape_list)
        
        print(' Priority draw list')
        display_shapes(draw_list)

        # self.shapes = order_shapes_by_bottom_edge(draw_list)
        self.shapes = draw_list

        print(' Sorted Priority draw list')
        display_shapes(self.shapes)
        
        ## Build remaining objects in object list
        self.shapes.extend(order_shapes_by_bottom_edge(tmp_shapes))
        # self.shapes.extend(tmp_shapes)

        print(' Final list - after sort by bottom edge of priority objects')
        display_shapes(self.shapes)

        self.image_data =  { 'bg_color' : self.bg_color,    # Tuple of RGB color
                 'horizon'  : self.horizon,     # Horizon Info - Tuple
                 'ground'   : self.ground,      # Ground Info - Tuple
                 'width'    : self.width,
                 'height'   : self.height,
                 'shapes'   : self.shapes       # List of objects
               }
        return

    ##---------------------------------------------------------------------------------------------
    ## build_horizon
    ##---------------------------------------------------------------------------------------------
    def build_horizon(self, verbose = False):
        ## Horizon between 1/3*height and 4/5*height
        y1 = random.randint(self.height // 3,   2 * self.height//3)
        y2 = y1
        color = random.choice(Image.custom_colors['sky'])
        if verbose:
            print(' Horizon between ', self.height // 3,   2 * self.height//3, ' is: ', y1)
            print( ' Horizon : ', y1, ' Color:', color, type(color), color.dtype)
        return (y1, y2, color)

    ##---------------------------------------------------------------------------------------------
    ## build_ground
    ##---------------------------------------------------------------------------------------------
    def build_ground(self, verbose = False):
        y2 = y1 = self.horizon[0]
        color = random.choice(Image.custom_colors['ground'])

        if verbose:
            print( '   Ground : ', y1, ' Color:', color, type(color), color.dtype)
        return (y1, y2, color)

    ##---------------------------------------------------------------------------------------------
    ## build_ground
    ##---------------------------------------------------------------------------------------------
    def get_random_color(self, shape):
        while True:
            if shape in Image.custom_colors_keys:
                color = random.choice(Image.custom_colors[shape])
            else:
                color = np.random.randint(0, 255, (3,), dtype = np.int32).astype(np.float32)

            if np.any(color != self.horizon[2]) and np.any(color != self.ground[2]):
                color = (np.asscalar(color[0]), np.asscalar(color[1]), np.asscalar(color[2]))
                break
        return color

    ##---------------------------------------------------------------------------------------------
    ## remove possible choices based on previous selected option 
    ##---------------------------------------------------------------------------------------------
    def remove_conflicting_choices(self, shape):
        if shape in [ 'airplane', 'truck']:
            for i in ['building', 'car', 'tree']:
                try:
                    self.possible_choices.remove(i)
                except:
                    continue
            # print(' Image is: ', self.image_id, ' Shape: ', shape, '  REMOVE BUILDING/CAR/TREE FROM POSSIBLE CHOICES')
            # print(' Possible choices:', self.possible_choices)
        elif shape in ['building', 'car', 'tree']:
            for i in ['airplane', 'truck']:
                try:
                    self.possible_choices.remove(i)
                except:
                    continue
            # print(' Image is: ', self.image_id,  ' Shape: ', shape, '  REMOVE AIRPLANE/ TRUCK FROM POSSIBLE CHOICES')
            # print(' Possible choices:', self.possible_choices)
        return 

    ##---------------------------------------------------------------------------------------------
    ## build, test, and add object
    ##---------------------------------------------------------------------------------------------
    def build_test_add_object(self, shape, verbose = False):
        if verbose:
            print()
            print('===> Image: ', self.image_id, ' build_test_add_object() - ', shape.upper(),'  Image currently has ', len(self.shapes), '  shapes')

        occ_ratio_list= []
        
        for i in range(Image.BUILD_MAX_TRIES):
            # print('   - Build object, try # ', i)

            new_object  = self.build_object(shape, verbose = False)
            
            # print('===> call get_max_occlusion()')
            # occlusions  = get_max_occlusion(new_object, self.shapes, verbose = True)
            occlusions  = self.get_pairwise_occlusion_ratio(new_object, verbose)

            occ_ratio   = occlusions.max()
            
            ## OR 
            # print('===> call get_occlusion_ratio()')
            # occ_ratio, object_mask   = self.get_occlusion_ratio(new_object, verbose)

            ## if occlusion rate is acceptable, add object 
            if occ_ratio < 0.75:
                self.shapes.append(new_object)
                self.built_counts[shape] += 1
                
                if verbose:
                    print(' Build succeeded - max_occlusion encounted on try # {:2d}  is: {:6.4f}'.format(i,occ_ratio))
                    _ , _, dims = new_object
                    print('        Shape : {:15s}   Top:{:3d}  Bot: {:3d} Left: {:3d} Right: {:3d}  Occ_Ratio: {:8.4f}    dims:{}'.format(
                                shape.upper(), dims[1]-dims[3], dims[1]+dims[3], dims[0]-dims[2], dims[0]+dims[2], occ_ratio, dims))
                    print('        Occlusion list:', occlusions)
                
                ## get_occlusion_ratio() work...
                # self.occlusion_mask = np.logical_and(self.occlusion_mask, np.logical_not(object_mask))                
                # if verbose:
                #     print('     New Occlusion = Occlusion && NOT(Mask[i]):   sum of occluded areas so far: ',np.logical_not(self.occlusion_mask).sum())
                #     # print(mask_string(self.occlusion_mask))
                #     print()
                return
            else:
                if verbose:
                    print(' Build failed - max_occlusion encounted on try # {:2d}  is: {:6.4f}  ... Retry building object'.format(i,occ_ratio))
                occ_ratio_list.append(occ_ratio)

        print()
        print('  -------------------------------------------------------------------------------------------')
        print('  Problem in building image ', self.image_id)
        print('  Selected objects : ', self.object_list)
        print('  Cannot build ', shape, ' object due to occlusion...')
        print('  Occlusions encountered:  ',  np.around(occ_ratio_list, 4))
        print('  lowest car       : ', self.lowest_vehicle       , ' highest car       :', self.highest_vehicle)
        print('  leftmost car     : ', self.leftmost_vehicle     , ' rightmost car     :', self.rightmost_vehicle)            
        print('  lowest building  : ', self.lowest_building  , ' highest building  :', self.highest_building)
        print('  leftmost building: ', self.leftmost_building, ' rightmost building:', self.rightmost_building)            
        for i, shp in enumerate(self.shapes):
            print('  {:2d}  {:15s}  cx: {:4d}  cy:{:4d}  sx:{:4d}  sy:{:4d}    '.format(i,shp[0],shp[2][0],shp[2][1],shp[2][2],shp[2][3]))
        print('  -------------------------------------------------------------------------------------------')
        print()
        return 

    ##---------------------------------------------------------------------------------------------
    ## build_object
    ##---------------------------------------------------------------------------------------------
    def build_object(self, shape, verbose = False):
        """Generates specifications of an object that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        buffer      = self.config.IMAGE_BUFFER
        min_y_dim   = self.config.min_dim[shape]
        max_y_dim   = self.config.max_dim[shape]        
        min_range_x = self.config.Min_X[shape]
        max_range_x = self.config.Max_X[shape]
        min_range_y = self.config.Min_Y[shape]
        max_range_y = self.config.Max_Y[shape]

        if verbose:
            print(' Build ',shape)            
            self.display_layout_info()

        color = self.get_random_color(shape)  

        if shape == "person":
            # color = random.choice(Image.personcolors)    

            ## 2 - get CX, CY between allowable limits
            dflt_min_range_y = self.horizon[0] - (min_y_dim //2)
            dflt_max_range_y = self.height     - (max_y_dim //2)
            dflt_min_range_x = 0   # self.config.Min_X[shape]
            dflt_max_range_x = self.width  #    self.config.Max_X[shape]
            found_coordiantes = False
            ## place person on left hand side of a car that doesn't have a person
            for i, (car_cx, car_cy, car_sx, _, person_placed) in enumerate(self.vehicles):
                if verbose:
                    print('car: @ CX/CY: ', car_cx, car_cy, 'Person next to it? ', person_placed)
                if not person_placed:
                    cx =  max(car_cx - car_sx - Image.person_car_gap ,0)
                    cy =  car_cy
                    self.vehicles[i][4] = True
                    found_coordiantes = True
                    break

            if not found_coordiantes:                    
                if verbose:
                    print(' Car not found')
                # min_range_x = self.config.Min_X[shape]
                min_range_x = 0
                max_range_x = self.leftmost_vehicle 
                # min_range_y = self.lowest_building - (min_y_dim //2)
                # max_range_y = self.highest_car

                min_range_y = min(self.lowest_building + min_y_dim//2 , self.height) 
                max_range_y = max(self.height - max_y_dim//2          , min_range_y)

                if verbose:
                    print('   Build between Y: [', min_range_y ,max_range_y, ']   X: [', min_range_x, max_range_x, ']' )

                if min_range_x +3  > max_range_x:
                    print('   Problem in building image ', self.image_id)
                    print('   Cannot build \'person\' object due to space limitations...')
                    print('   Condition: min_range_x +3  > max_range_x ')    
                    print('   Building range  Y: [', min_range_y ,max_range_y, ']   X: [', min_range_x, max_range_x, ']' )
                    print('   lowest car       : ', self.lowest_vehicle       , ' highest car       :', self.highest_vehicle)
                    print('   leftmost car     : ', self.leftmost_vehicle     , ' rightmost car     :', self.rightmost_vehicle)            
                    print('   lowest building  : ', self.lowest_building  , ' highest building  :', self.highest_building)
                    print('   leftmost building: ', self.leftmost_building, ' rightmost building:', self.rightmost_building)            
                    for i, shp in enumerate(self.shapes):
                        print('   {:2d}  {:15s}  cx: {}  cy:{}  sx:{}  sy:{}    '.format(i,shp[0],shp[2][0],shp[2][1],shp[2][2],shp[2][3]))
                    return
                cx = random.randint(min_range_x, max_range_x)
                cy = random.randint(min_range_y, max_range_y)
                if verbose:
                    print('   Build between Y: [', min_range_y ,max_range_y, ']   X: [', min_range_x, max_range_x, ']' )
                    print('   CX: ', cx, 'CY: ', cy)

            sy = int(np.interp([cy],[dflt_min_range_y, dflt_max_range_y], [min_y_dim   , max_y_dim]))
            sx = int(np.interp([cy],[dflt_min_range_y, dflt_max_range_y], [min_y_dim//4, max_y_dim//4] ))
            if verbose:
                print('   interpolation range Y: [',dflt_min_range_y,  dflt_max_range_y,' ] Min / Max Dim: [ ' , min_y_dim, max_y_dim, ']  CY:', cy, 'SY: ', sy)
                print('   interpolation range Y: [',dflt_min_range_x,  dflt_max_range_x,' ] Min / Max Dim: [ ' , min_y_dim//5, max_y_dim//5, '] Cx:', cx, 'SY: ', sx)
                print('   Final (cx,cy,sx,sy): ', cx,cy,sx,sy)

        elif shape == "building":

            sy = random.randint(min_y_dim  , max_y_dim)
            sx = random.randint(min_y_dim-5, max_y_dim+5)
            if verbose:
                print('   Y Dim between Y: [', min_y_dim ,max_y_dim, ']' )
                print('   SX: ', sx, 'SY: ', sy)

            # determine size of the building
            min_range_y = self.horizon[0] +  2 - sy
            max_range_y = max(self.horizon[0] + 15 - sy, min_range_y)                     
            cx = random.randint(min_range_x, max_range_x)
            cy = random.randint(min_range_y, max_range_y)
            if verbose:
                print('   Build between Y: [', min_range_y ,max_range_y, ']   X: [', min_range_x, max_range_x, ']' )
                print('   CX: ', cx, 'CY: ', cy)
            
            # max_dim = min(self.height - cy, max_dim)
            # based on size determine location realtive to horizon 
            #  sy = int(np.interp([cy],[min_range_y, max_range_y], [min_dim, max_dim]))
            #  sx = random.randint(5,15)

            self.leftmost_building  = min( cx-sx, self.leftmost_building )
            self.rightmost_building = max( cx+sx, self.rightmost_building)
            self.highest_building   = min( cy-sy, self.highest_building)
            self.lowest_building    = max( cy+sy, self.lowest_building)
            if verbose:
                print('   cy:', cy, ' sy: ', sy, '  lowest   :', self.lowest_building  , ' highest   :', self.highest_building)
                print('   cx:', cx, ' sx: ', sx, '  leftmost :', self.leftmost_building, ' rightmost :', self.rightmost_building)            

        elif shape == "car":
            ## 1 - get CX, CY between allowable limits
            dflt_min_range_y = self.horizon[0] - (min_y_dim //2)
            dflt_max_range_y = self.height     - (max_y_dim //2)
            # dflt_min_range_x = min_y_dim *2 + self.person_car_gap + 5   # min_x_dim == min_y_dim*2  self.config.Min_X[shape]
            dflt_min_range_x = min_y_dim *2.4     
            dflt_max_range_x = self.width      #    self.config.Max_X[shape]
            
            min_range_y = min(self.lowest_building + min_y_dim//2 , self.height) 
            # min_range_y = min(self.lowest_tree + min_y_dim//2 , self.height) 
            max_range_y = max(self.height - max_y_dim//2          , min_range_y)
            cy = random.randint(min_range_y, max_range_y)

            if verbose:
                print('   Build between Y: [', min_range_y ,max_range_y, ']   X: [', min_range_x, max_range_x, ']' )

            ## 3 - interpolate SX, SY based on loaction of CY
            # scale width based on location on the image. Images closer to the bottom will be larger
            sy = int(np.interp([cy],[dflt_min_range_y, dflt_max_range_y], [min_y_dim   , max_y_dim]  ))
            sx = int(np.interp([cy],[dflt_min_range_y, dflt_max_range_y], [min_y_dim*2 , max_y_dim*2] ))
            
            ## 4 - determine range of possible locations for CX based on interpolated SX, and pick a CX
            min_range_x = sx + Image.person_car_gap
            max_range_x = dflt_max_range_x            
            cx = random.randint(min_range_x, max_range_x)

            self.leftmost_vehicle  = min( cx-sx, self.leftmost_vehicle)
            self.rightmost_vehicle = max( cx+sx, self.rightmost_vehicle)
            self.highest_vehicle   = min( cy-sy, self.highest_vehicle)
            self.lowest_vehicle    = max( cy+sy, self.lowest_vehicle)
            self.vehicles.append([cx,cy,sx,sy,False])

            if verbose:
                print('   interpolation range Y: [',dflt_min_range_y,  dflt_max_range_y,' ] Min / Max Dim: [ ' , min_y_dim, max_y_dim    , '] CY:', cy, 'SY: ', sy)
                print('   interpolation range X: [',dflt_min_range_x,  dflt_max_range_x,' ] Min / Max Dim: [ ' , min_y_dim*2.4, max_y_dim*2.4, '] Cx:', cx, 'SY: ', sx)
                print('   cy:', cy, ' sy: ', sy, '  lowest car  :',   self.lowest_vehicle, ' highest car :',   self.highest_vehicle)
                print('   cx:', cx, ' sx: ', sx, '  leftmost car:', self.leftmost_vehicle,  ' rightmost car :', self.rightmost_vehicle)            
                print('   Final (cx,cy,sx,sy): ', cx,cy,sx,sy)

        elif shape == 'truck' :

            max_y_dim        = self.config.max_dim[shape]
            min_y_dim        = self.config.min_dim[shape]
            xy_ratio = random.randint(3,4)
            
            ## 2 - get CX, CY between allowable limits
            dflt_min_range_y = self.horizon[0] - (min_y_dim //2)
            dflt_max_range_y = self.height     - (max_y_dim //2)
            # dflt_min_range_x = 0
            dflt_max_range_x = self.width      #    self.config.Max_X[shape]

            min_range_y = dflt_min_range_y
            max_range_y = dflt_max_range_y
            cy = random.randint(min_range_y, max_range_y)

            ## 3 - interpolate SY based on loaction of CY
            # scale width based on location on the image. Images closer to the bottom will be larger
            sy = int(np.interp([cy],[dflt_min_range_y, dflt_max_range_y], [min_y_dim, max_y_dim]))
            sx = int(np.interp([cy],[dflt_min_range_y, dflt_max_range_y], [min_y_dim*xy_ratio , max_y_dim*xy_ratio] ))
            # sx = sy * xy_ratio

            ## 4 - determine range of possible locations for CX based on interpolated SX, and pick a CX
            min_range_x = sx + Image.person_car_gap
            max_range_x = dflt_max_range_x
            # max_range_x = self.width - sx
            cx = random.randint(min_range_x, max_range_x)

            # self.leftmost_vehicle  = min( cx-sx, self.leftmost_vehicle)
            # self.rightmost_vehicle = max( cx+sx, self.rightmost_vehicle)
            # self.highest_vehicle   = min( cy-sy, self.highest_vehicle)
            # self.lowest_vehicle    = max( cy+sy, self.lowest_vehicle)
            # self.vehicles.append([cx,cy,sx,sy,False])

            if verbose:
                print('   Build between Y: [', min_range_y ,max_range_y, ']   X: [', min_range_x, max_range_x, ']' )
                print('   CY: ', cy, '   SY: ', sy,'   CX: ', cx,'   SX: ', sx, )

        elif shape == "tree":
            ver_save = verbose
            verbose = True
            if verbose:
                print(' Build Tree')
                self.display_layout_info()

            # color = random.choice(Image.treecolors)    
            group_range_y = 25
            group_range_x = 25

            ## 2 - get CX, CY between allowable limits
            dflt_min_range_y = self.horizon[0] - (min_y_dim)
            dflt_max_range_y = min(4*self.horizon[0]//3, self.height - (max_y_dim)) ## + (max_y_dim//4)
            dflt_min_range_x = 0               # self.config.Min_X[shape]
            dflt_max_range_x = self.width      # self.config.Max_X[shape]
            if verbose:
                print('   First Tree       : ', self.first_tree)
                print('   dflt min rage    : ', dflt_min_range_y      , ' dflt max range    : ', dflt_max_range_y)
            
            if self.built_counts[shape] == 0 :
                min_range_x = dflt_min_range_x
                max_range_x = dflt_max_range_x
                min_range_y = dflt_min_range_y
                max_range_y = min(self.highest_vehicle + 10, dflt_max_range_y) 
                if verbose:
                    print('   First Tree - CY range: [',min_range_y,  max_range_y,' ]     CX Range: [ ' , min_range_x, max_range_x, '] '  )
            else:
                min_range_y = max(self.first_tree[1] - group_range_y, dflt_min_range_y)
                max_range_y = min(self.first_tree[1] + group_range_y, dflt_max_range_y)
                min_range_x = max(self.first_tree[0] - group_range_x, dflt_min_range_x)
                max_range_x = min(self.first_tree[0] + group_range_x, dflt_max_range_x)
                if verbose:
                    print('   Next Tree  - CY range: [',min_range_y,  max_range_y,' ]     CX Range: [ ' , min_range_x, max_range_x, '] '  )

            cx = random.randint(min_range_x, max_range_x)
            cy = random.randint(min_range_y, max_range_y)
            if verbose:
                print('   CY between    :[', min_range_y, max_range_y, ']  CX between : [',min_range_x, max_range_x,' ]') 
                print('   CX: ', cx, 'CY: ', cy)

            ## 3 - interpolate SX, SY based on loaction of CY
            sy = int(np.interp([cy],[dflt_min_range_y, dflt_max_range_y], [min_y_dim   , max_y_dim]   ))
            sx = int(np.interp([cy],[dflt_min_range_y, dflt_max_range_y], [min_y_dim//4, max_y_dim//4]))
            if verbose:
                print('   After Interpolation SX: ', sx, 'SY: ', sy)
            if self.built_counts[shape] == 0 :
                self.first_tree = (cx,cy)
            verbose = ver_save
            
            self.leftmost_tree  = min( cx-sx, self.leftmost_tree)
            self.rightmost_tree = max( cx+sx, self.rightmost_tree)
            self.highest_tree   = min( cy-sy, self.highest_tree)
            self.lowest_tree    = max( cy+sy, self.lowest_tree)

        elif shape == "airplane":
         
            max_y_dim        = self.config.max_dim[shape]
            min_y_dim        = self.config.min_dim[shape]

            dflt_min_range_y = min_y_dim
            dflt_max_range_y = self.horizon[0] - 10 
            dflt_min_range_x = 0      
            dflt_max_range_x = self.width   
            #     cx = 64
            cy = random.randint(min_range_y, max_range_y)
            cx = random.randint(min_range_x, max_range_x)
            sy = int(np.interp([cy],[dflt_min_range_y, dflt_max_range_y], [min_y_dim   , max_y_dim]  ))
            sx = sy * 8//3
            step_sz = sy / 4
            if verbose:            
                print('   Build between Y      : [',      min_range_y,     max_range_y, ']               X: [', min_range_x, max_range_x, ']' )
                print('   interpolation range Y: [', dflt_min_range_y, dflt_max_range_y,']   Min / Max Dim: [ ' , min_y_dim, max_y_dim, ']  CY:', cy, 'SY: ', sy)
                print('   Step Size            : ', step_sz, '    sy: ', sy , '  sx:', sx)
                print('   Final (cx,cy,sx,sy)  : ', cx,cy,sx,sy)

        elif shape == "sun":
            # color = random.choice(Image.suncolors)    
            if verbose:
                print(' Build Sun')
                print('  Sun Colors is :', color, type(color), color.dtype)
            cx = random.randint(min_range_x, max_range_x)
            cy = random.randint(min_range_y, max_range_y)

            sy = int(np.interp([cy],[min_range_y, max_range_y], [min_y_dim, max_y_dim]))
            sx = sy

        elif shape == "cloud":
            # color = random.choice(Image.cloudcolors)            

            ## 1 - Get SX , SY between limits. 
            max_y_dim        = self.config.max_dim[shape]
            min_y_dim        = self.config.min_dim[shape]

            ## 2 - get CX, CY between allowable limits
            dflt_min_range_y = (max_y_dim //2)
            dflt_max_range_y = self.config.Max_Y[shape] - (min_y_dim //2)
            dflt_min_range_x = 0   # self.config.Min_X[shape]
            dflt_max_range_x = self.width  #    self.config.Max_X[shape]

            min_range_x = dflt_min_range_x
            max_range_x = dflt_max_range_x
            min_range_y = dflt_min_range_y 
            max_range_y = dflt_max_range_y
            # min_range_y = min(self.lowest_building + min_y_dim//2 , self.height) 
            # max_range_y = max(self.height - max_y_dim//2          , min_range_y)
            cx = random.randint(min_range_x, max_range_x)
            cy = random.randint(min_range_y, max_range_y)
            if verbose:
                print('   Build between Y: [', min_range_y ,max_range_y, ']   X: [', min_range_x, max_range_x, ']' )
                print('   CX: ', cx, 'CY: ', cy)
            ## 3 - interpolate SX, SY based on loaction of CY
            ratio = random.randint(3, 5)
            sy = int(np.interp([cy],[dflt_min_range_y, dflt_max_range_y], [min_y_dim   , max_y_dim]   ))
            sx = int(np.interp([cy],[dflt_min_range_y, dflt_max_range_y], [min_y_dim*ratio, max_y_dim*ratio]))
            if verbose:
                print('   interpolation range Y: [',dflt_min_range_y,  dflt_max_range_y,' ] Min / Max Dim: [ ' , min_y_dim, max_y_dim    , '] CY:', cy, 'SY: ', sy)
                print('   interpolation range X: [',dflt_min_range_x,  dflt_max_range_x,' ] Min / Max Dim: [ ' , min_y_dim*ratio, max_y_dim*ratio, '] Cx:', cx, 'SY: ', sx)

        # elif shape == "old cloud":
        #     print(' Build Cloud')
        #     color = random.choice(Image.cloudcolors)            
        #     cx = random.randint(min_range_x, max_range_x)
        #     cy = random.randint(min_range_y, max_range_y)

        #     sx = int(np.interp([cy],[min_range_y, max_range_y], [min_y_dim, max_y_dim]))
        # #     min_height ,max_height = 10, 20
        # #     sy = random.randint(min_height, max_height)
        #     sx = sy *  random.randint(3, 5)

        # elif shape == "new building":
        #     if verbose:
        #         print(' Build Building :')
        #         print('   Horizion         : ', self.horizon[0], ' Color: ', self.horizon[2])
        #         print('   lowest building  :', self.lowest_building  , ' highest building  :', self.highest_building)
        #         print('   leftmost building:', self.leftmost_building, ' rightmost building :', self.rightmost_building)            
        #     color = self.get_random_color(shape)
            
        #     ## 1 - Get a random Building size (SX,SY) between limits. 
        #     max_y_dim        = self.config.max_dim[shape]
        #     min_y_dim        = self.config.min_dim[shape]
        #     ratio = random.choice([0.5, 0.75, 1.25, 1.5, 1.75, 2])
        #     dim1 = np.array([random.randint(min_y_dim  , max_y_dim), min_y_dim, max_y_dim, 1], dtype = np.float)
        #     dim2 = np.array([dim1[0] * ratio, min_y_dim * ratio, max_y_dim * ratio, ratio], dtype = np.float)
        #     dims = np.vstack([dim1, dim2])
        #     np.random.shuffle(dims)
        #     print(' Ratio: ', ratio,' Randomized SX :', dims[0], ' SY: ', dims[1])
        #     sx, min_x_dim, max_x_dim = dims[0,:3]
        #     sy, min_y_dim, max_y_dim = dims[1,:3]

        #     ## 2 - get CX, CY between allowable limits
        #     dflt_min_range_y = self.horizon[0] - (min_y_dim //2)
        #     dflt_max_range_y = self.height     - (max_y_dim //2)
        #     min_range_y = dflt_min_range_y 
        #     max_range_y = dflt_max_range_y
        #     cy = random.randint(min_range_y, max_range_y)
        #     sy = int(np.interp([cy],[dflt_min_range_y, dflt_max_range_y], [min_y_dim , max_y_dim] ))
        #     sx = int(np.interp([cy],[dflt_min_range_y, dflt_max_range_y], [min_x_dim , max_x_dim] ))
            
        #     ## 4 - determine range of possible locations for CX based on interpolated SX, and pick a CX
        #     # dflt_min_range_x = max_x_dim
        #     # dflt_max_range_x = self.width - max_x_dim     #    self.config.Max_X[shape]
        #     min_range_x = sx
        #     max_range_x = self.width - sx
        #     cx = random.randint(min_range_x, max_range_x)


        #     # min_range_y = min(self.lowest_building + min_y_dim//2 , self.height) 
        #     # max_range_y = max(self.height - max_y_dim//2          , min_range_y)
        #     cx = random.randint(min_range_x, max_range_x)

        #     self.leftmost_building  = min( cx-sx, self.leftmost_building )
        #     self.rightmost_building = max( cx+sx, self.rightmost_building)
        #     self.highest_building   = min( cy-sy, self.highest_building)
        #     self.lowest_building    = max( cy+sy, self.lowest_building)

        #     if verbose:
        #         print('   Build between Y: [', min_range_y ,max_range_y, ']   X: [', min_range_x, max_range_x, ']' )
        #         print('   CX: ', cx, 'CY: ', cy)
        #         print('   After Interpolation SX: ', sx, 'SY: ', sy)
        #         print('   interpolation range Y: [',dflt_min_range_y,  dflt_max_range_y,' ] Min / Max Dim: [ ' , min_y_dim, max_y_dim, '] CY:', cy, 'SY: ', sy)
        #         print('   interpolation range X: [',dflt_min_range_x,  dflt_max_range_x,' ] Min / Max Dim: [ ' , min_x_dim, max_x_dim, '] Cx:', cx, 'SY: ', sx)
        #         print('   cy:', cy, ' sy: ', sy, '  lowest   bldg :', self.lowest_building  , ' highest   bldg:', self.highest_building)
        #         print('   cx:', cx, ' sx: ', sx, '  leftmost bldg :', self.leftmost_building, ' rightmost bldg:', self.rightmost_building)            
        #         print('                             lowest car    :', self.lowest_car       , ' highest car   :', self.highest_car)
        #         print('                             leftmost car  :', self.leftmost_car     , ' rightmost car :', self.rightmost_car)            
        #         print('   Final (cx,cy,sx,sy): ', cx,cy,sx,sy)


        else :
            print(' Build miscellaneous object')
            min_y_dim   = buffer
            max_y_dim   = self.width // 4
            color = self.get_random_color(shape)
            cx = random.randint(self.config.Min_X['_default'], self.config.Max_X['_default'])
            cy = random.randint(self.config.Min_Y['_default'], self.config.Max_Y['_default'])

            sy = int(np.interp([cy],[self.config.Min_Y['_default'], self.config.Max_Y['_default']], [min_y_dim, max_y_dim]))

            # sx = random.randint(min_size, max_size)

            if shape == "rectangle":
                sx = random.randint(min_y_dim, max_y_dim)
            else:
                ## other shapes have same sx and sy
                sx = sy


        return (shape, color, (cx, cy, sx,sy))

    ##---------------------------------------------------------------------------------------------
    ## find_hidden_shapes
    ##---------------------------------------------------------------------------------------------
    def find_hidden_shapes(self, shapes, verbose = False):
        '''
        a variation of load_masks customized to find objects that
        are completely hidden by other shapes
        '''

        # print('\n load mask information (shape, (color rgb), (x_ctr, y_ctr, size) ): ')
        # p4.pprint(info['shapes'])
        hidden_shapes = []
        count  = len(shapes)
        mask   = np.zeros( [self.height, self.width, count], dtype=np.uint8)
        
        ## get masks for each shape
        for i, (shape, _, dims) in enumerate(shapes):
            mask[:, :, i:i + 1] = draw_object(mask[:, :, i:i + 1].copy(), shape, dims, 1)

        #----------------------------------------------------------------------------------
        #  start with last shape as the occlusion mask
        #   occlusion starts with the last object an list and in each iteration of the loop
        #   adds an additional  object. pixes assigned to objects are 0. non assigned pixels
        #   are 1
        #-----------------------------------------------------------------------------------
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)

        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

            ##-------------------------------------------------------------------------------------
            ## if the shape has been completely occluded by other shapes, it's mask is all zeros.
            ## in this case np.any(mask) will return false.
            ## for these completely hidden objects, we record their id in hidden []
            ## and later remove them from the  list of shapes
            ##-------------------------------------------------------------------------------------
            if ( ~np.any(mask[:,:,i]) ) :
                # print(' !!!!!!  zero mask found !!!!!!' )
                hidden_shapes.append(i)

        if verbose and len(hidden_shapes) > 0:
            print(' ===> find hidden shapes() found hidden objects ')
            p8.pprint(shapes)
            print(' ****** objects completely hidden are : ', hidden_shapes)
            for i in hidden_shapes:
                p8.pprint(shapes[i])
        return hidden_shapes

    ##---------------------------------------------------------------------------------------------
    ## get_occlusion_ratio
    ##---------------------------------------------------------------------------------------------
    def get_occlusion_ratio(self, object, verbose = False):
        '''
        Compute occlusion between new object and mask of previously added objects
        a variation of load_masks customized to find objects that
        are completely hidden by other shapes
        '''
        shape, _, dims = object
        object_mask   = np.zeros( [self.height, self.width], dtype=np.uint8)

        ## get object_mask for shape
        object_mask = draw_object(object_mask, shape, dims, 1)
        shape_area = object_mask.sum()

        if verbose:
            np_format = {'int': lambda x: "%1d" % x}
            np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =20000, formatter = np_format)
            top  = dims[1]-dims[3]
            bot  = dims[1]+dims[3]
            left = dims[0]-dims[2]
            right= dims[0]+dims[2]
            print('     get_occ_ratio():  Shape: {:15s}   Top:{:3d}  Bot: {:3d} Left: {:3d} Right: {:3d}  shape_area: {:8.2f}    dims:{}'.format(
                        shape.upper(), top, bot, left, right, shape_area, dims))
            # print(mask_string(object_mask))        

        #----------------------------------------------------------------------------------
        #  apply occlusion_mask on object_mask and determine remaining area of object
        #-----------------------------------------------------------------------------------

        occluded_obj = object_mask * self.occlusion_mask
        occluded_obj_area = occluded_obj.sum()
        non_occ_ratio = (occluded_obj_area / shape_area)
        occ_ratio = 1.0 - non_occ_ratio
        
        # if verbose:
        #     print('     Shape: {:15s} occluded_object (= Mask[i] * Occlusion)  Orig area:{}  Area considering occlusions: {}'.format(shape,shape_area, occluded_obj_area))
        #     print('     Object Occlusion Ratio is : {:8.4f}   NonOcclusion ratio: {:8.4f}'.format(occ_ratio, non_occ_ratio))
        #     # print(mask_string(occluded_obj))
        #     print()

        return occ_ratio, object_mask

    ##---------------------------------------------------------------------------------------------
    ## get_pairwise occlusion_ratio
    ##---------------------------------------------------------------------------------------------
    def get_pairwise_occlusion_ratio(self, new_object, verbose = False):
        '''
        a variation of load_masks customized to find objects that
        are completely hidden by other shapes
        '''
        np_format = {'int': lambda x: "%1d" % x}
        np.set_printoptions(linewidth=195, precision=4, floatmode='fixed', threshold =20000, formatter = np_format)
        # if verbose:
        print('   ====> get_pairwise_occlusion_ratio()  - len(other_shapes) :', len(self.shapes))
        if len(self.shapes) == 0:
            print('   Other objects is empty -- return')
            return np.array([0], dtype=np.float)

        height   = 128
        width    = 128
        new_shape, _, new_dims = new_object
        new_top  = new_dims[1] - new_dims[3]
        new_bot  = new_dims[1] + new_dims[3]
        new_left = new_dims[0] - new_dims[2]
        new_right= new_dims[0] + new_dims[2]
        
        ## get object_mask for shape
        new_object_mask   = np.zeros( [height, width], dtype=np.uint8)
        new_object_mask = np.expand_dims(draw_object(new_object_mask, new_shape, new_dims, 1), axis = -1)
        neg_new_object_mask = np.logical_not(new_object_mask)
        new_object_area = new_object_mask.sum()

        # print(utils.mask_string(new_object_mask[:,:,0]))        
        # old_tp    = np.array([shp[2][1]-shp[2][3] for shp in other_objects])
        # old_bt    = np.array([shp[2][1]+shp[2][3] for shp in other_objects])
        # print('old_tp ', old_tp)
        # print('old_bt ', old_bt)
        # above_indexes = np.where(old_bt < new_top)
        # below_indexes = np.where(old_tp > new_bot)
        # print(' Aboves: ', above_indexes)
        # print(' Belows: ', below_indexes)
        
        old_count = len(self.shapes)
        old_dims  = np.array([shp[2] for shp in self.shapes] )
        old_shapes= np.array([shp[0] for shp in self.shapes])
        
        old_top   = old_dims[:,1:2] - old_dims[:,3:]
        old_bot   = old_dims[:,1:2] + old_dims[:,3:]
        old_left  = old_dims[:,0:1] - old_dims[:,2:3]
        old_right = old_dims[:,0:1] + old_dims[:,2:3]
        old_boxes = np.hstack([old_top,old_left,old_bot,old_right])
        old_count = old_shapes.shape[0]
        
        ## get masks for existing shapes
        
        mask      = np.zeros( [height, width, old_count], dtype=np.uint8)
        for i, (shp, dim) in enumerate(zip(old_shapes, old_dims)):
            mask[:, :, i:i + 1] = draw_object(mask[:, :, i:i + 1].copy(), shp, dim, 1)
        neg_mask =np.logical_not(mask)
        old_object_area = mask.sum(axis = (0,1))    
        
        if verbose:
            print()
            # print(' new_obj_mask_shape', new_object_mask.shape, ' New object area', new_object_area)
            print('     Newshape: ')
            print('           Shape: {:15s}   CY: {:3d}   CX: {:3d}  Top:{:3d}  Bot: {:3d} Left: {:3d} Right: {:3d}  area: {:8.2f}    dims:{}'.format(
                        new_shape.upper(), new_dims[1], new_dims[0], new_top, new_bot, new_left, new_right, new_object_area, new_dims))
            print()
            print('     Previous objects :', old_count)
            for i in range(old_count):
                print('       {:2d}  Shape: {:15s}   CY: {:3d}   CX: {:3d}  Top:{:3d}  Bot: {:3d} Left: {:3d} Right: {:3d}  area: {:8.2f}    dims:{}'.format(
                    i, old_shapes[i].upper(), old_dims[i,1],old_dims[i,0], 
                        old_boxes[i,0], old_boxes[i,2], old_boxes[i,1], old_boxes[i,3], old_object_area[i], old_dims[i]))
        
        # identify objects not affected by new object and drop them from occlusion calculations
        
        above_indexes = np.where(old_bot  < new_top)
        below_indexes = np.where(old_top  > new_bot)
        drop_indexes  = np.union1d(above_indexes[0], below_indexes[0]).astype(np.int)

        old_shapes = np.delete(old_shapes, drop_indexes, axis =0)
        old_boxes  = np.delete(old_boxes, drop_indexes, axis =0)
        old_dims   = np.delete(old_dims, drop_indexes, axis =0)
        mask       = np.delete(mask, drop_indexes, axis = -1)
        neg_mask   = np.logical_not(mask)
        old_object_area = mask.sum(axis = (0,1))    
        old_count  = old_shapes.shape[0]
        if verbose:
            print('           Aboves: ', above_indexes[0])
            print('           Belows: ', below_indexes[0])
            print('           drop_indexes: ', drop_indexes)
            print('           Trimmed old shaps info:', old_shapes.shape)
            print('           Trimmed old dims  info:', old_dims.shape)
            print('           Trimmed old boxes info:', old_boxes.shape)    
            print('           Trimmed old masks info:', mask.shape)    
            
        if verbose:
            print('     Newshape: ')
            print('            Shape: {:15s}   CY: {:3d}   CX: {:3d}  Top:{:3d}  Bot: {:3d} Left: {:3d} Right: {:3d}  area: {:8.2f}    dims:{}'.format(
                        new_shape.upper(), new_dims[1], new_dims[0], new_top, new_bot, new_left, new_right, new_object_area, new_dims))
            print()
            print('     Previous objects after removing irrelevant boxes :', old_count)
            for i in range(old_count):
                print('        {:2d}  Shape: {:15s}   CY: {:3d}   CX: {:3d}  Top:{:3d}  Bot: {:3d} Left: {:3d} Right: {:3d}  area: {:8.2f}    dims:{}'.format(
                    i, old_shapes[i].upper(), old_dims[i,1], old_dims[i,0],
                    old_boxes[i,0], old_boxes[i,2], old_boxes[i,1], old_boxes[i,3], old_object_area[i], old_dims[i]))
        if old_count == 0:
            print('     After Trimming Other objects is empty -- return')
            return np.array([0], dtype=np.float)

        #----------------------------------------------------------------------------------
        #  apply occlusion_mask on object_mask and determine remaining area of object
        #-----------------------------------------------------------------------------------
        occluded1 = np.logical_and(neg_mask, new_object_mask)
        t_new_obj_non_occ_area = occluded1.sum(axis =(0,1))
        t_new_obj_non_occ_ratio= t_new_obj_non_occ_area / new_object_area
        t_new_obj_occ_ratio    = 1 - t_new_obj_non_occ_ratio
        
        occluded2 = np.logical_and(mask, neg_new_object_mask)
        t_old_obj_non_occ_area = occluded2.sum(axis =(0,1))
        t_old_obj_non_occ_ratio= t_old_obj_non_occ_area / old_object_area
        t_old_obj_occ_ratio    = 1 - t_old_obj_non_occ_ratio
        abv_objs = np.where(old_boxes[:,2]<=new_bot)[0]
        bel_objs = np.where(old_boxes[:,2] >new_bot)[0]
        t_ttl_obj_occ_ratio =  np.maximum(t_new_obj_occ_ratio, t_old_obj_occ_ratio)

        print()
        print('           PREVIOUSLY DEFINED OBJECTS    : ',''.join([i+' ' for i in old_shapes]))
        print('           NEW OBJECT- NON OCCLUDED AREAS: ', t_new_obj_non_occ_area)
        print('           NEW OBJECT- TOTAL AREA        : ', new_object_area)
        print('           NEW OBJECT- NON OCCLUDED RATIO: ', t_new_obj_non_occ_ratio)
        print('           NEW OBJECT- OCCLUDED RATIO    : ', t_new_obj_occ_ratio)
        print()      
        print('           OLD OBJECT- NON OCCLUDED AREAS: ', t_old_obj_non_occ_area)
        print('           OLD OBJECT- TOTAL AREA        : ', old_object_area)
        print('           OLD OBJECT- NON OCCLUDED RATIO: ', t_old_obj_non_occ_ratio)
        print('           OLD OBJECT- OCCLUDED RATIO    : ', t_old_obj_occ_ratio)        
        print('           Objects above this new object : ', abv_objs )
        print('           Objects below this new object : ', bel_objs )
        print('           Occlusions by this object on above objs   :  ', t_old_obj_occ_ratio[abv_objs])
        print('           Occlusions by below objects on this object:  ', t_new_obj_occ_ratio[bel_objs])
        print()          
        print('           Max occlusion b/w new and old objects     :  ', t_ttl_obj_occ_ratio)
        
        return t_ttl_obj_occ_ratio

    def display_layout_info(self):
        print('   Horizon           : ', self.horizon[0])
        print('   lowest   building : ', self.lowest_building  , ' highest   building :', self.highest_building)
        print('   leftmost building : ', self.leftmost_building, ' rightmost building :', self.rightmost_building)            
        print('   lowest   tree     : ', self.lowest_tree      , ' highest   tree     :', self.highest_tree)
        print('   leftmost tree     : ', self.leftmost_tree    , ' rightmost tree     :', self.rightmost_tree)            
        print('   lowest   car      : ', self.lowest_vehicle       , ' highest   car  :', self.highest_vehicle)
        print('   leftmost car      : ', self.leftmost_vehicle     , ' rightmost car  :', self.rightmost_vehicle) 


def order_shapes_by_bottom_edge(shapes, verbose = False):
    if verbose:
        print(" ===== Objects before sorting on cy =====  ")
        p4.pprint(shapes)

    sort_lst = [itm[2][1]+itm[2][3] for itm in shapes]
    if verbose:
        print(' sort list (cy + sy) : ', sort_lst)

    sorted_shapes_ind = np.argsort(np.array(sort_lst))[::+1]
    sorted_shapes = [shapes[i] for i in sorted_shapes_ind]

    # print(sort_lst)
    # print(sorted_shape_ind)
    if verbose:
        print(' ===== Objects after sorting on cy+sy ===== ')
        p4.pprint(sorted_shapes)
    return sorted_shapes

def get_max_occlusion(new_shape, other_shapes, verbose = False ):
    '''
    Determined occlusion between new_shape and existing shapes  
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    '''
    if verbose:
        print('   ====> get_max_occlusion()  - len(other_shapes) :', len(other_shapes))
    if len(other_shapes) == 0:
        return np.array([0], dtype=np.float)

    new_x, new_y, new_sx, new_sy = new_shape[2]
    new_box = np.array([new_y - new_sy, new_x - new_sx, new_y + new_sy, new_x + new_sx])
    new_class = new_shape[0]
    new_area = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1]) 
    
    boxes   = []
    classes = []
    cy = []
    cx = [] 
    for shp in other_shapes:
        # print('   class:', shp[0], '     box:', shp[2])
        x, y, sx, sy = shp[2]
        boxes.append([y - sy, x - sx, y + sy, x + sx])
        classes.append(shp[0])
        cy.append(y)
        cx.append(x)
    boxes   = np.array(boxes)
    classes = np.array(classes)
    cy      = np.array(cy)
    cx      = np.array(cx)
    scores  = np.arange(len(boxes))
    areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1]) 
    # cy = boxes[:,0] + (boxes[:,2] - boxes[:,0]) //2
    # cx = boxes[:,1] + (boxes[:,3] - boxes[:,1]) //2
    
    if verbose:
        print('     Test:    Area: {:5d}  {:15s}  CY/CX:{:3d}/{:3d}  {}'.format(new_area,  new_class, new_y, new_x, new_box))
        print('       against:')
        for box, cls, scr ,ar, y,x  in zip(boxes, classes, scores, areas, cy,cx):
            print('     scr: {:2d}  Area: {:5d}  {:15s}  CY/CX:{:3d}/{:3d}  {}'.format(scr, ar,  cls, y,x, box))

    # Compute clipped box areas
    clipped_boxes = np.zeros_like(boxes)
    clp_y1 = clipped_boxes[:,0] = np.maximum(boxes[:, 0], 0)    ## y1
    clp_x1 = clipped_boxes[:,1] = np.maximum(boxes[:, 1], 0)    ## x1
    clp_y2 = clipped_boxes[:,2] = np.minimum(boxes[:, 2], 128)  ## y2
    clp_x2 = clipped_boxes[:,3] = np.minimum(boxes[:, 3], 128)  ## x2
    clipped_areas = (clp_y2 - clp_y1) * (clp_x2 - clp_x1) 

    if verbose:
        print('   ====> After Clipping ')
        for box,cls,scr, ar, y, x in zip(clipped_boxes, classes, scores, clipped_areas, cy,cx):
            print('     scr: {:2d}  Area: {:5d}  {:15s}  CY/CX:{:3d}/{:3d}  {}'.format(scr, ar,  cls, y,x, box))
    
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)
             
    # Compute IoU of the picked box with the rest
    # iou, inter, union, occlusion = debug_compute_iou(new_box, boxes, new_area, areas)
    iou, inter, union, occlusion = debug_compute_iou(new_box, clipped_boxes, new_area, clipped_areas)
    
    if verbose:
        print()
        print('   IOU- new shape: ', new_class,'     box:', new_box, '     area: ', new_area)
        print('           clsses: ', ''.join( [i.rjust(11) for i in classes]))
        print('            areas: ', areas )   
        print('              iou: ', iou)
        print('     intersection: ', inter)
        print('            union: ', union)
        print('        occlusion: ', occlusion)

    return occlusion

def debug_compute_iou(box, boxes, box_area, boxes_area, verbose = False):
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
    iou = intersection / union  
    occlusion = intersection / box_area

    if verbose:
        print('      box is      : ', box)
        # print('      box[0]      : ', box[0],'  boxes[:,0] : ', boxes[:,0], ' y1 - np.max ', y1)
        # print('      box[2]      : ', box[2],'  boxes[:,2] : ', boxes[:,2], ' y2 - np.min ', y2)
        # print('      box[1]      : ', box[1],'  boxes[:,1] : ', boxes[:,1], ' x1 - np.max ', x1)
        # print('      box[3]      : ', box[3],'  boxes[:,3] : ', boxes[:,3], ' x2 - np.min ', x2)
        print('      intersection: ', intersection)
        print('      union       : ', union)
        print('      ious        : ', iou)

    return iou, intersection, union, occlusion

def custom_non_max_suppression(shapes, threshold, verbose = False ):
    '''
    identical to debug_non_max_suppresion - just recevies the shapes and does
    all processing inside this function.

    Performs non-maximum supression and returns indicies of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    '''
    boxes = []
    for shp in shapes:
        x, y, sx, sy = shp[2]
        boxes.append([y - sy, x - sx, y + sy, x + sx])

    assert len(boxes) == len(shapes), "Problem with the shape and box sizes matching"
    boxes = np.array(boxes)
    # N = len(boxes)
    scores = np.arange(len(boxes))

    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)
    if verbose:      
        print('====> non_max_suppression ')

    # for box, scr, cls in zip(boxes, scores, classes):
    #     print( '{}   {:15s}    {}   '.format(scr, cls, box))
    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)
    
    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]
  
    # Get indicies of boxes sorted by scores (lowest first)
    ixs = scores.argsort()

    pick = []
    if verbose:
        print('====> Initial Ixs: ', ixs)
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        cy = y1[i] + (y2[i]-y1[i])//2
        cx = x1[i] + (x2[i]-x1[i])//2
        pick.append(i)
        if verbose:
            print(' **  ix : ', i, 'ctr (x,y)', cx,' ',cy,)
            print('     box    : ', boxes[i], ' compare ',i, ' with ', ixs[1:])
            print('     area[i]: ', area[i] , ' area[ixs[1:]] :',area[ixs[1:]] )   

        # Compute IoU of the picked box with the rest
        iou, _,_,_ = debug_compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        
        # Identify boxes with IoU over the threshold. This
        # returns indicies into ixs[1:], so add 1 to get
        # indicies into ixs.
        tst =  np.where(iou>threshold)
        remove_ixs = np.where(iou > threshold)[0] + 1
        
        # Remove indicies of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
        if verbose:
            print('     np.where( iou > threshold) : ' ,tst, 'tst[0] (index into ixs[1:]: ', tst[0], 
                    ' remove_ixs (index into ixs) : ',remove_ixs)
            print('     ending ixs (after deleting ixs[0]): ', ixs, ' picked so far: ',pick)
    
    if verbose:
        print('====> Final Picks: ', pick)
    return np.array(pick, dtype=np.int32)

def debug_non_max_suppression(boxes, scores, threshold, verbose = False ):
    '''
    Performs non-maximum supression and returns indicies of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    '''
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)
    if verbose:      
        print('====> non_max_suppression ')
    # for box, scr, cls in zip(boxes, scores, classes):
    #     print( '{}   {:15s}    {}   '.format(scr, cls, box))
    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)
    
    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]
  
    pick = []
    if verbose:
        print('====> Initial Ixs: ', ixs)
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        cy = y1[i] + (y2[i]-y1[i])//2
        cx = x1[i] + (x2[i]-x1[i])//2
        pick.append(i)
        if verbose:
            print(' **  ix : ', i, 'ctr (x,y)', cx,' ',cy,)
            print('     box    : ', boxes[i], ' compare ',i, ' with ', ixs[1:])
            print('     area[i]: ', area[i] , ' area[ixs[1:]] :',area[ixs[1:]] )   

        # Compute IoU of the picked box with the rest
        iou, _,_,_ = debug_compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        
        # Identify boxes with IoU over the threshold. This
        # returns indicies into ixs[1:], so add 1 to get
        # indicies into ixs.
        tst =  np.where(iou>threshold)
        remove_ixs = np.where(iou > threshold)[0] + 1
        
        # Remove indicies of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
        if verbose:
            print('     np.where( iou > threshold) : ' ,tst, 'tst[0] (index into ixs[1:]: ', tst[0], 
                    ' remove_ixs (index into ixs) : ',remove_ixs)
            print('     ending ixs (after deleting ixs[0]): ', ixs, ' picked so far: ',pick)
    
    if verbose:
        print('====> Final Picks: ', pick)
    return np.array(pick, dtype=np.int32)

def debug_non_max_suppression_2(shapes, threshold, verbose = False ):
    '''
    Performs non-maximum supression and returns indicies of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    '''
    if verbose:
        print('====> non_max_suppression ')

    boxes = []
    classes = []
    for shp in shapes:
        x, y, sx, sy = shp[2]
        boxes.append([y - sy, x - sx, y + sy, x + sx])
        classes.append(shp[0])
    boxes   = np.array(boxes)
    classes = np.array(classes)
    scores  = np.arange(len(boxes))
    area = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1]) 
    cy = boxes[:,0] + (boxes[:,2] - boxes[:,0]) //2
    cx = boxes[:,1] + (boxes[:,3] - boxes[:,1]) //2
    
    if verbose:
        for box, cls, scr ,ar, y,x  in zip(boxes, classes, scores, area, cy,cx):
            print( 'scr:', scr, '  ', box, '   ', ar,  '    ', cls,'   CY/CX:   ',y,x)

    # Compute cliiped box areas
    clipped_boxes = np.zeros_like(boxes)
    clp_y1 = clipped_boxes[:,0] = np.maximum(boxes[:, 0], 0)    ## y1
    clp_x1 = clipped_boxes[:,1] = np.maximum(boxes[:, 1], 0)    ## x1
    clp_y2 = clipped_boxes[:,2] = np.minimum(boxes[:, 2], 128)  ## y2
    clp_x2 = clipped_boxes[:,3] = np.minimum(boxes[:, 3], 128)  ## x2
    clp_area = (clp_y2 - clp_y1) * (clp_x2 - clp_x1) 

    if verbose:
        print('====> After Clipping ')
        for box,cls,scr, ar in zip(clipped_boxes, classes, scores, clp_area):
            print( 'scr:', scr, '  ', box, '   ', ar, '    ', cls)
    
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)
      
    # Get indicies of boxes sorted by scores (highest first)
    # ixs = scores.argsort()[::-1]
    # Get indicies of boxes sorted by Area (LOWEST first)
    # ixs = area.argsort()
    # Get indicies of boxes sorted by highest in image (Top objects  first)
    ixs = boxes[:,0].argsort()

    pick = []
    if verbose:
        print('====> After Sorting - sort indices: ', ixs)
        for i in ixs : 
            print( 'scr:', scores[i], '  ', clipped_boxes[i], '   ', area[i], '    ', classes[i])


    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        cls = classes[i]
        pick.append(i)
        
        # Compute IoU of the picked box with the rest
        iou, inter, union, occlusion = debug_compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        if verbose:
            print()
            print(' **  ix : ', i, ' shape : ',classes[i],'     box:', boxes[i], '     ctr (x,y):   (', cx[i],',',cy[i],')', ' area: ', area[i])
            print('              ixs: ', ixs[1:])
            print('           clsses: ', ''.join( [i.rjust(11) for i in classes[ixs[1:]] ]))
            print('            areas: ', area[ixs[1:]] )   
            print('              iou: ', iou)
            print('     intersection: ', inter)
            print('            union: ', union)
            print('        occlusion: ', occlusion)

        # Identify boxes with IoU over the threshold. This
        # returns indicies into ixs[1:], so add 1 to get
        # indicies into ixs.
        
        tst =  np.where(iou>threshold)
        remove_ixs = np.where(iou > threshold)[0] + 1
        
        # Remove indicies of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
        if verbose:
            print('     np.where( iou > threshold) : ' ,tst, 'tst[0] (index into ixs[1:]: ', tst[0], 
                    ' remove_ixs (index into ixs) : ',remove_ixs)
            print('     ending ixs (after deleting ixs[0]): ', ixs, ' picked so far: ',pick)
    if verbose:    
        print('====> Final Picks: ', pick)

    return np.array(pick, dtype=np.int32)

def display_shapes(shapes):
    print('{:2s} {:15s}    {:3s}  {:3s}  {:3s}  {:3s}    {:3s} {:3s} {:3s} {:3s}'.format('seq', 'class_name', 'Y1', 'X1' ,
                                                                                        'Y2' , 'X2', ' CX ', 'CY','SX','SY'))
    print('-'*65)
    for idx, shp in enumerate(shapes):
        color = shp[1]
        x, y, sx, sy = shp[2]
        print('{:2} {:15s}    {:3d}  {:3d}  {:3d}  {:3d}     {:3d} {:3d} {:3d} {:3d}    {}'.format(idx, 
                shp[0], y - sy, x - sx, y + sy, x + sx, shp[2][0],shp[2][1],shp[2][2],shp[2][3], color))
    print()                                                                                        
    return

"""     
def debug_compute_iou_old(box, boxes, box_area, boxes_area, verbose = False):
    '''
    Calculates IoU of the given box with the array of the given boxes.
    box:                1D vector [y1, x1, y2, x2]
    boxes:              [boxes_count, (y1, x1, y2, x2)]
    box_area:           float. the area of 'box'
    boxes_area:         array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    '''
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])

    
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union  

    if verbose:
        print('      box is      : ', box)
        # print('      box[0]      : ', box[0],'  boxes[:,0] : ', boxes[:,0], ' y1 - np.max ', y1)
        # print('      box[2]      : ', box[2],'  boxes[:,2] : ', boxes[:,2], ' y2 - np.min ', y2)
        # print('      box[1]      : ', box[1],'  boxes[:,1] : ', boxes[:,1], ' x1 - np.max ', x1)
        # print('      box[3]      : ', box[3],'  boxes[:,3] : ', boxes[:,3], ' x2 - np.min ', x2)
        print('      intersection: ', intersection)
        print('      union       : ', union)
        print('      ious        : ', iou)

    return iou
"""

"""
elif shape == "old person":
    print(' Build Person')
    print('   Horizion         : ', self.horizon[0])
    print('   lowest car  :', self.lowest_car  , ' highest car  :',   self.highest_car)
    print('   leftmost car:', self.leftmost_car, ' rightmost car:', self.rightmost_car)            
    min_range_x = self.config.Min_X[shape]
    max_range_x = self.leftmost_car
    min_range_y = self.highest_car
    max_range_y = self.config.Max_Y[shape]
    print('   Build between Y: [', min_range_y ,max_range_y, ']   X: [', min_range_x, max_range_x, ']' )
    if min_range_x +3  > max_range_x:
        print('cannot build object due to space limitations...')
        return
    cx = random.randint(min_range_x, max_range_x)
    cy = random.randint(min_range_y, max_range_y)

    ## 3 - interpolate SX, SY based on loaction of CY
    # scale width based on location on the image. Images closer to the bottom will be larger
    # old method :  sx = random.randint(min_width , max_width)
    sy = int(np.interp([cy],[dflt_min_range_y, dflt_max_range_y], [min_y_dim   , max_y_dim]  ))
    sx = int(np.interp([cy],[dflt_min_range_y, dflt_max_range_y], [min_y_dim*2, max_y_dim*2] ))
    print('   interpolation range Y: [',min_range_y,  max_range_y,' ] Min / Max Dim: [ ' , min_dim, max_dim, '] CY:', cy, 'SY: ', sy)
    print('   After Interpolation SX: ', sx, 'SY: ', sy)


#           sy = random.randint(min_height, max_height)
    sy = int(np.interp([cy],[min_range_y,  max_range_y], [min_dim, max_dim]))
    sx = sy //5    # body width
    print('   Final (cx,cy,sx,sy): ', cx,cy,sx,sy)
"""
