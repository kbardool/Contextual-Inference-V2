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
    x, y, sx, sy = dims
    # print(' draw_image() Shape : {:20s}   Cntr (x,y): ({:3d} , {:3d})    Size_x: {:3d}   Size_y: {:3d} {}'.format(shape,x,y,sx, sy,color))
    if verbose:
        print('  Draw ', shape, ' Color:', color, ' shape', type(color))
        print('    CX :', x, 'CY:', y , 'sx: ',sx , 'sy: ', sy)

    if shape == "building":
        x1 = x - sx
        y1 = y - sy
        x2 = x + sx
        y2 = y + sy
        image = cv2.rectangle(image, (x1,y1), (x2, y2), color, -1)
         
#             print('X :', x, 'y:', y , '     sx: ',sx , 'sy: ', sy, 'hs:', hs)

    elif shape == "car":
        body_y  = sy //3
        wheel_x = sx //2
        wheel_r = sx //5
        top_x   = sx //4
        bot_x   = 3*sx //4
        if verbose:
            print('    Car Top(y): ', y - sy , '  Bottom(y) : ', y + body_y + wheel_r, ' Left(x):', x - sx, ' Right(x) : ', x+sx) 
            print('    Half Car hgt: ', sy,  ' Half Body height: ', body_y , ' Half body width : ', sx)
        
        image = cv2.rectangle(image, (x - sx, y - body_y), (x + sx, y + body_y), color, -1)
        image = cv2.circle(image, (x - wheel_x , y + body_y), wheel_r, color, -1)
        image = cv2.circle(image, (x + wheel_x , y + body_y), wheel_r, color, -1)
        # Top cab
        points = np.array([[(x - top_x , y - sy),   (x + top_x, y - sy),
                            (x + bot_x,  y - body_y),(x - bot_x, y - body_y), ]], dtype=np.int32)
        image = cv2.fillPoly(image, points, color)

    elif shape == "person":
#             hy = sy // 4   # head height
#             by = sy - hy   # body height
#             print('X :', x, 'y:', y , 'sx: ',sx , 'sy: ', sy, 'hs:', hs)
#             image = cv2.rectangle(image, (x - sx, y - by), (x + sx, y + by), color, -1)
#             image = cv2.circle(image, (x , y -(by+hy) ), sx, color, -1)

        hy = sy // 4   # head height
        by = sy - hy   # body height
        # torso
        image = cv2.rectangle(image, (x - sx, y - by), (x + sx, y + by//4), color, -1)
        # legs
        image = cv2.rectangle(image, (x - sx, y + by//4), (x - sx +sx//4, y + by), color, -1)
        image = cv2.rectangle(image, (x + sx - sx//4, y + by//4), (x + sx, y + by), color, -1)
        #head
        image = cv2.circle(image, (x , y -(by+hy) ), sx, color, -1)
        if verbose:
            print('    Person  Top(y) : ', y -(by+hy)+sx , '  Bottom(y) : ', y+by, ' Left(x):', x - sx, ' Right(x) : ', x+sx)

    elif shape == "tree":
        sin_t = math.sin(math.radians(60))
        full_height = 2 * sy
        ty = full_height //5                # trunk length
        by = (full_height - ty) // 2        # half body length
        bx = int(by / sin_t) // 2  # half body width 
        tx = bx//5                 # trunk width
        # orde of points: top, left, right
        points = np.array([[(x, y - by),                    ## top 
                            (x - bx, y + by),     ## left
                            (x + bx, y + by),     ## right 
                            ]], dtype=np.int32)
        image = cv2.fillPoly(image, points, color)
        image = cv2.rectangle(image,(x-tx,y+by), (x+tx, y+by+ty),color, -1)
        if verbose:
            print('    Tree  Top(y) : ', y - by , '  Bottom(y) : ', y+by+ty, ' Left(x):', x - bx, ' Right(x) : ', x+bx)
            print('    Trunk Length : ', ty, '  Body Length :', by, '    Half Body Width: ', bx, '  Half Trunk Width: ', tx)

    elif shape == "sun":
        image = cv2.circle(image, (x, y), sx, color, -1)

    elif shape == "cloud":
        image = cv2.ellipse(image,(x,y),(sx, sy),0,0,360,color,-1)

    if shape == "square":
        image = cv2.rectangle(image, (x - sx, y - sy), (x + sx, y + sy), color, -1)

    elif shape == "rectangle":
        image = cv2.rectangle(image, (x - sx, y - sy), (x + sx, y + sy), color, -1)
#             print('X :', x, 'y:', y , '     sx: ',sx , 'sy: ', sy, 'hs:', hs)

    elif shape == "circle":
        image = cv2.circle(image, (x, y), sx, color, -1)

    elif shape == "ellipse":
        image = cv2.ellipse(image,(x,y),(sx, sy),0,0,360,color,-1)

    elif shape == "triangle":
        sin60 = math.sin(math.radians(60))
        # orde of points: top, left, right
        points = np.array([[(x, y - sx),
                            (x - (sx / sin60), y + sx),
                            (x + (sx / sin60), y + sx),
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
    suncolors   = [ (colors.to_rgba_array(i)*255)[0,:3] for i in [colors.CSS4_COLORS['lightyellow'],
                                                                  colors.CSS4_COLORS['yellow'],
                                                                  colors.CSS4_COLORS['gold'],
                                                                  colors.CSS4_COLORS['orange'],
                                                                  colors.CSS4_COLORS['darkorange']
                                                                  ]]

    cloudcolors = [ (colors.to_rgba_array(i)*255)[0,:3] for i in [colors.CSS4_COLORS['silver'],
                                                                  colors.CSS4_COLORS['lightslategray'],
                                                                  colors.CSS4_COLORS['lightgray'],
                                                                  colors.CSS4_COLORS['gainsboro'],
                                                                  colors.CSS4_COLORS['whitesmoke'],
                                                                  colors.CSS4_COLORS['white'],
                                                                  colors.CSS4_COLORS['snow'],
                                                                  colors.CSS4_COLORS['ghostwhite'],
                                                                  ]]

    skycolors   = [ (colors.to_rgba_array(i)*255)[0,:3] for i in [colors.CSS4_COLORS['lightblue'],
                                                                  colors.CSS4_COLORS['deepskyblue'],
                                                                  colors.CSS4_COLORS['skyblue'],
                                                                  colors.CSS4_COLORS['lightskyblue'],
                                                                  colors.CSS4_COLORS['steelblue'],
                                                                  colors.CSS4_COLORS['aliceblue']
                                                                  ]]

    personcolors = [ (colors.to_rgba_array(i)*255)[0,:3] for i in [colors.CSS4_COLORS['black'],
                                                                   colors.CSS4_COLORS['tan'],
                                                                   colors.CSS4_COLORS['sienna'],
                                                                   colors.CSS4_COLORS['saddlebrown'],
                                                                   colors.CSS4_COLORS['khaki'],
                                                                   colors.CSS4_COLORS['oldlace'],
                                                                  ]]

    groundcolors = [ (colors.to_rgba_array(i)*255)[0,:3] for i in [
                                                                   colors.CSS4_COLORS['sandybrown'],
                                                                   colors.CSS4_COLORS['peachpuff'],
                                                                   colors.CSS4_COLORS['peru'],
                                                                   colors.CSS4_COLORS['linen'],
                                                                   colors.CSS4_COLORS['burlywood'],
                                                                   colors.CSS4_COLORS['antiquewhite'],
                                                                   colors.CSS4_COLORS['seagreen'],
                                                                  ]]


    treecolors = [ (colors.to_rgba_array(i)*255)[0,:3] for i in  [colors.CSS4_COLORS['forestgreen'],
                                                                  colors.CSS4_COLORS['limegreen'],
                                                                  colors.CSS4_COLORS['darkgreen'],
                                                                  colors.CSS4_COLORS['darkolivegreen'],
                                                                  colors.CSS4_COLORS['lightgreen'],
                                                                  colors.CSS4_COLORS['olive'],
                                                                  colors.CSS4_COLORS['blanchedalmond'],
                                                                  ]]

    # possible_choices = ['sun':1,'car':3, 'building':3, 'person':1, 'cloud':2, 'tree':4]
    # possible_choices = {'sun':1,'car':2, 'tree':5}
    possible_choices = {'sun':1,   'car':3 , 'tree':5, 'person':5, 'cloud':3, 'building':3}  ## tree':3, 'car':3}
    object_priority_list = ['building','tree','car']
    BUILD_MAX_TRIES = 7


    def __init__(self, image_id,  datasetConfig ,verbose = False):

        super().__init__()
        if verbose:
            print(' Init Object -Possible Object Choices: ', Image.possible_choices)
            # print('             Max Choices: ', Image.max_choices)
        self.image_id           = image_id
        self.config             = datasetConfig
        self.height             = self.config.HEIGHT
        self.width              = self.config.WIDTH
        self.buffer             = self.config.IMAGE_BUFFER
        self.max_range_y        = self.config.HEIGHT - self.config.IMAGE_BUFFER
        self.horizon            = self.build_horizon()
        self.ground             = self.build_ground()
        self.rightmost_building = 0
        self.rightmost_car      = 0 
        self.leftmost_building  = self.width
        self.leftmost_car       = self.width
        self.lowest_building    = self.horizon[0]
        self.lowest_car         = self.horizon[0] 
        self.highest_building   = self.height
        self.highest_car        = self.height
        self.first_tree         = (0,0)
        self.person_car_gap     = 10   # fixed spread between car and person
        self.bg_color           = np.array([random.randint(0, 255) for _ in range(3)])
        self.shapes             = []
        self.selected_counts    = {}
        self.allowed_counts     = {}
        self.built_counts       = {}
        self.possible_choices   = []   
        self.cars               = []
        self.object_list    = []
        for shape in Image.possible_choices:
            self.possible_choices.append(shape)
            self.allowed_counts[shape]  = Image.possible_choices[shape]
            self.selected_counts[shape] = 0
            self.built_counts[shape]    = 0


        # Generate a few random shapes and record their bounding boxes
        N = random.randint(self.config.MIN_SHAPES_PER_IMAGE, self.config.MAX_SHAPES_PER_IMAGE)    # number to shapes in image

        
        for _ in range(N):
            shape       = random.choice(self.possible_choices)
            self.selected_counts[shape] += 1
            self.object_list.append(shape)
            if self.selected_counts[shape] == self.allowed_counts[shape]:
                # print(' Max number of ',shape, ' reached - remove from possible choices ')
                self.possible_choices.remove(shape)
                # print(' Possible choices now: ', self.possible_choices)
            if not self.possible_choices :
                break
        if verbose:        
            print(' Number of objects to pick ', N)
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
                    self.build_test_add_object(object, verbose)
        
        ## Build remaining objects in object list
        if verbose:
            print('remaining list: ', self.object_list)
        for object in self.object_list:
            self.build_test_add_object(object, verbose)

        ##--------------------------------------------------------------------------------
        ## Reorder shapes by increasing cy to simulate overlay
        ## (nearer shapes cover farther away shapes)
        ## order shape objects based on closeness to bottom of image (-1) or top (+1)
        ## this will result in items closer to the viewer have higher priority in NMS
        ##--------------------------------------------------------------------------------
        if verbose:
            print(" ===== Objects before sorting on cy =====  ")
            p4.pprint(self.shapes)

        sort_lst = [itm[2][1]+itm[2][3] for itm in self.shapes]
        if verbose:
            print(' sort list (cy + sy) : ', sort_lst)
        
        sorted_shape_ind = np.argsort(np.array(sort_lst))[::+1]
        tmp_shapes = [self.shapes[i] for i in sorted_shape_ind]

        # print(sort_lst)
        # print(sorted_shape_ind)
        if verbose:
            print(' ===== Objects after sorting on cy+sy ===== ')
            p4.pprint(tmp_shapes)


        ##-------------------------------------------------------------------------------
        ## find and remove shapes completely covered by other shapes
        ##-------------------------------------------------------------------------------
        hidden_shape_ixs = self.find_hidden_shapes(tmp_shapes)
        if len(hidden_shape_ixs) > 0:
            non_hidden_shapes = [s for i, s in enumerate(tmp_shapes) if i not in hidden_shape_ixs]
            # print('    ===> Image Id : (',image_id, ')   ---- Zero Mask Encountered ')
            # print('    ------ Original Shapes ------' )
            # p8.pprint(tmp_shapes)
            # print('    ------ shapes after removal of totally hidden shapes ------' )
            # p8.pprint(non_hidden_shapes)
            # print('    Number of shapes now is : ', len(non_hidden_shapes))
        else:
            non_hidden_shapes = tmp_shapes

        ##-------------------------------------------------------------------------------
        ## build boxes for to pass to non_max_suppression
        ##-------------------------------------------------------------------------------
        boxes = []
        for shp in non_hidden_shapes:
            x, y, sx, sy = shp[2]
            boxes.append([y - sy, x - sx, y + sy, x + sx])

        ##--------------------------------------------------------------------------------
        ## Non Maximal Suppression
        ## Suppress occulsions more than 0.3 IoU
        ## Apply non-max suppression with 0.3 threshold to avoid shapes covering each other
        ## object scores (which dictate the priority) are assigned in the order they were created
        ##--------------------------------------------------------------------------------
        assert len(boxes) == len(non_hidden_shapes), "Problem with the shape and box sizes matching"
        N = len(boxes)


        keep_ixs =  debug_non_max_suppression(np.array(boxes), np.arange(N), 0.29, verbose)
        tmp_shapes = [s for i, s in enumerate(non_hidden_shapes) if i in keep_ixs]
        if verbose:
            print('===> Original number of shapes {} '.format(N))
            for i in non_hidden_shapes:
                print('     ', i)
            print('     Number of shapes after NMS {}'.format(len(tmp_shapes)))
            for i in tmp_shapes:
                print('     ', i)


        ##--------------------------------------------------------------------------------
        ## Reorder shapes to simulate overlay (nearer shapes cover farther away shapes)
        ## order shape objects based on closeness to bottom of image (-1) or top (+1)
        ## this will result in items closer to the viewer have higher priority in NMS
        ##--------------------------------------------------------------------------------
        sort_lst = [itm[2][1]+itm[2][3] for itm in tmp_shapes]
        sorted_shape_ind = np.argsort(np.array(sort_lst))[::+1]

        # print(" =====  Before final sort =====  ")
        # p4.pprint(shapes)
        # print(sort_lst)
        # print(sorted_shape_ind)

#        tmp_shapes = []
#        for i in sorted_shape_ind:
#            tmp_shapes.append(shapes[i])

        self.shapes = [tmp_shapes[i] for i in sorted_shape_ind]
        
        if verbose:
            print(' ===== Shapes after Final Sorting ===== ')
            p4.pprint(self.shapes)

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
        color = random.choice(Image.skycolors)
        if verbose:
            print(' Horizon between ', self.height // 3,   2 * self.height//3, ' is: ', y1)
            print( ' Horizon : ', y1, ' Color:', color, type(color), color.dtype)
        return (y1, y2, color)

    ##---------------------------------------------------------------------------------------------
    ## build_ground
    ##---------------------------------------------------------------------------------------------
    def build_ground(self, verbose = False):
#        y1 = random.randint(horizon[0], self.max_range_y )
        y2 = y1 = self.horizon[0]
        # color = np.array([random.randint(0, 255) for _ in range(3)])
        color = random.choice(Image.groundcolors)
        if verbose:
            print( '   Ground : ', y1, ' Color:', color, type(color), color.dtype)
        return (y1, y2, color)

    ##---------------------------------------------------------------------------------------------
    ## build_ground
    ##---------------------------------------------------------------------------------------------
    def get_random_color(self, shape):
        while True:
            color = np.random.randint(0, 255, (3,), dtype = np.int32).astype(np.float32)
            if np.any(color != self.horizon[2]) and np.any(color != self.ground[2]):
                color = (np.asscalar(color[0]), np.asscalar(color[1]), np.asscalar(color[2]))
                break
        return color


    ##---------------------------------------------------------------------------------------------
    ## build_object
    ##---------------------------------------------------------------------------------------------
    def build_test_add_object(self, shape, verbose = False):
        print()
        print('===> build_test_add_object():  Image currently has ', len(self.shapes), '  shapes')

        max_occlusion_list= []
        
        for i in range(Image.BUILD_MAX_TRIES):
            print('   - Build ',shape.upper(),' object, try # ', i)

            new_object = self.build_object(shape, verbose = False)
            # print('===> call get_max_occlusion()')
            occlusions  = get_max_occlusion(new_object, self.shapes, verbose)
            max_occlusion = occlusions.max()
            if max_occlusion < 0.5:
                print('     Build succeeded - max_occlusion encounted on try # ',i, ' is: ', max_occlusion, max_occlusion.shape)
                self.shapes.append(new_object)
                self.built_counts[shape] += 1
                return
            else:
                print('     Build failed - max_occlusion encounted is: ', max_occlusion, max_occlusion.shape,' ... Retry building object')
                max_occlusion_list.append(max_occlusion)

        # if i == MAX_TRIES:
        print()
        print('     Problem in building image ', self.image_id)
        print('     Cannot build ', shape, ' object due to occlusion...')
        print('     Occlusions encountered:  ',  max_occlusion_list)
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

        ## get random center of object (which is constrainted by min/mx ranges above)

        if shape == "person":
            if verbose:
                print(' Build Person')
                print('   Horizion         : ', self.horizon[0], ' Color: ', self.horizon[2])
                print('   lowest car       : ', self.lowest_car       , ' highest car       :', self.highest_car)
                print('   leftmost car     : ', self.leftmost_car     , ' rightmost car     :', self.rightmost_car)            
                print('   lowest building  : ', self.lowest_building  , ' highest building  :', self.highest_building)
                print('   leftmost building: ', self.leftmost_building, ' rightmost building:', self.rightmost_building)            
            color = random.choice(Image.personcolors)    

            ## 1 - Get SX , SY between limits. 
            # max_y_dim        = self.config.max_dim[shape]
            # min_y_dim        = self.config.min_dim[shape]

            ## 2 - get CX, CY between allowable limits
            dflt_min_range_y = self.horizon[0] - (min_y_dim //2)
            dflt_max_range_y = self.height     - (max_y_dim //2)
            dflt_min_range_x = 0   # self.config.Min_X[shape]
            dflt_max_range_x = self.width  #    self.config.Max_X[shape]
            found_coordiantes = False
            ## place person on left hand side of a car that doesn't have a person
            for i, (car_cx, car_cy, car_sx, car_sy, person_placed) in enumerate(self.cars):
                if verbose:
                    print('car: @ CX/CY: ', car_cx, car_cy, 'Person next to it? ', person_placed)
                if not person_placed:
                    cx =  max(car_cx - car_sx - self.person_car_gap ,0)
                    cy =  car_cy
                    self.cars[i][4] = True
                    found_coordiantes = True
                    break

            if not found_coordiantes:                    
                if verbose:
                    print(' Car not found')
                # min_range_x = self.config.Min_X[shape]
                min_range_x = 0
                max_range_x = self.leftmost_car 
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
                    print('   lowest car       : ', self.lowest_car       , ' highest car       :', self.highest_car)
                    print('   leftmost car     : ', self.leftmost_car     , ' rightmost car     :', self.rightmost_car)            
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


#           sy = random.randint(min_height, max_height)
            sy = int(np.interp([cy],[dflt_min_range_y, dflt_max_range_y], [min_y_dim   , max_y_dim]))
            sx = int(np.interp([cy],[dflt_min_range_y, dflt_max_range_y], [min_y_dim//5, max_y_dim//5] ))
            if verbose:
                print('   interpolation range Y: [',dflt_min_range_y,  dflt_max_range_y,' ] Min / Max Dim: [ ' , min_y_dim, max_y_dim, ']  CY:', cy, 'SY: ', sy)
                print('   interpolation range Y: [',dflt_min_range_x,  dflt_max_range_x,' ] Min / Max Dim: [ ' , min_y_dim//5, max_y_dim//5, '] Cx:', cx, 'SY: ', sx)
                print('   Final (cx,cy,sx,sy): ', cx,cy,sx,sy)

        elif shape == "new building":
            if verbose:
                print(' Build Building :')
                print('   Horizion         : ', self.horizon[0], ' Color: ', self.horizon[2])
                print('   lowest building  :', self.lowest_building  , ' highest building  :', self.highest_building)
                print('   leftmost building:', self.leftmost_building, ' rightmost building :', self.rightmost_building)            
            color = self.get_random_color(shape)
            
            ## 1 - Get SX , SY between limits. 
            # max_y_dim        = self.config.max_dim[shape]
            # min_y_dim        = self.config.min_dim[shape]

            # sy = random.randint(min_dim, max_dim)
            # sx = sy //2 + 2
            # print('   Selected   SX: ', sx, 'SY: ', sy)

            ## 2 - get CX, CY between allowable limits
            dflt_min_range_y = self.horizon[0] - (min_y_dim //2)
            dflt_max_range_y = self.height     - (max_y_dim //2)
            dflt_min_range_x = 0   # self.config.Min_X[shape]
            dflt_max_range_x = self.width  #    self.config.Max_X[shape]

            min_range_x = 0
            max_range_x = self.width
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
            sy = int(np.interp([cy],[dflt_min_range_y, dflt_max_range_y], [min_y_dim   , max_y_dim]   ))
            sx = int(np.interp([cy],[dflt_min_range_y, dflt_max_range_y], [min_y_dim//2, max_y_dim//2]))
            if verbose:
                print('   After Interpolation SX: ', sx, 'SY: ', sy)
                print('   interpolation range Y: [',dflt_min_range_y,  dflt_max_range_y,' ] Min / Max Dim: [ ' , min_y_dim, max_y_dim, '] CY:', cy, 'SY: ', sy)
                print('   interpolation range Y: [',dflt_min_range_x,  dflt_max_range_x,' ] Min / Max Dim: [ ' , min_y_dim*2, max_y_dim*2, '] Cx:', cx, 'SY: ', sx)
                print('   cy:', cy, ' sy: ', sy, '  lowest car  :',   self.lowest_car, ' highest car :',   self.highest_car)
                print('   cx:', cx, ' sx: ', sx, '  leftmost car:', self.leftmost_car,  ' rightmost car :', self.rightmost_car)            
                print('   Final (cx,cy,sx,sy): ', cx,cy,sx,sy)

            self.leftmost_building  = min( cx-sx, self.leftmost_building )
            self.rightmost_building = max( cx+sx, self.rightmost_building)
            self.highest_building   = min( cy-sy, self.highest_building)
            self.lowest_building    = max( cy+sy, self.lowest_building)
            if verbose:            
                print('   cy:', cy, ' sy: ', sy, '  lowest   :', self.lowest_building  , ' highest   :', self.highest_building)
                print('   cx:', cx, ' sx: ', sx, '  leftmost :', self.leftmost_building, ' rightmost :', self.rightmost_building)            

        elif shape == "building":
            if verbose:
                print(' Build Building :')
                print('   Horizion         : ', self.horizon[0], ' Color: ', self.horizon[2])
                print('   lowest car       : ', self.lowest_car       , ' highest car       :', self.highest_car)
                print('   leftmost car     : ', self.leftmost_car     , ' rightmost car     :', self.rightmost_car)            
                print('   lowest building  : ', self.lowest_building  , ' highest building  :', self.highest_building)
                print('   leftmost building: ', self.leftmost_building, ' rightmost building:', self.rightmost_building)            
            color = self.get_random_color(shape)            
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
            if verbose:
                print(' Build Car')            
                print('   Horizion         : ', self.horizon[0])
                print('   lowest building  : ', self.lowest_building  , ' highest building  :', self.highest_building)
                print('   leftmost building: ', self.leftmost_building, ' rightmost building :', self.rightmost_building)            
            color = self.get_random_color(shape)
            
            ## 1 - Get SX , SY between limits. 
            # max_y_dim        = self.config.max_dim[shape]
            # min_y_dim        = self.config.min_dim[shape]

            ## 2 - get CX, CY between allowable limits
            dflt_min_range_y = self.horizon[0] - (min_y_dim //2)
            dflt_max_range_y = self.height     - (max_y_dim //2)
            # dflt_min_range_x = min_y_dim *2 + self.person_car_gap + 5   # min_x_dim == min_y_dim*2  self.config.Min_X[shape]
            dflt_min_range_x = min_y_dim *2     
            dflt_max_range_x = self.width      #    self.config.Max_X[shape]
            
            min_range_y = min(self.lowest_building + min_y_dim//2 , self.height) 
            max_range_y = max(self.height - max_y_dim//2          , min_range_y)
            cy = random.randint(min_range_y, max_range_y)

            if verbose:
                print('   Build between Y: [', min_range_y ,max_range_y, ']   X: [', min_range_x, max_range_x, ']' )
                print('   CX: ', cx, 'CY: ', cy)


            ## 3 - interpolate SX, SY based on loaction of CY
            # scale width based on location on the image. Images closer to the bottom will be larger
            # old method :  sx = random.randint(min_width , max_width)
            sy = int(np.interp([cy],[dflt_min_range_y, dflt_max_range_y], [min_y_dim   , max_y_dim]  ))

            sx = int(np.interp([cy],[dflt_min_range_y, dflt_max_range_y], [min_y_dim*2, max_y_dim*2] ))
            min_range_x = sx + self.person_car_gap
            max_range_x = dflt_max_range_x
            cx = random.randint(min_range_x, max_range_x)

            self.leftmost_car  = min( cx-sx, self.leftmost_car)
            self.rightmost_car = max( cx+sx, self.rightmost_car)
            self.highest_car   = min( cy-sy, self.highest_car)
            self.lowest_car    = max( cy+sy, self.lowest_car)

            if verbose:
                print('   interpolation range Y: [',dflt_min_range_y,  dflt_max_range_y,' ] Min / Max Dim: [ ' , min_y_dim, max_y_dim    , '] CY:', cy, 'SY: ', sy)
                print('   interpolation range X: [',dflt_min_range_x,  dflt_max_range_x,' ] Min / Max Dim: [ ' , min_y_dim*2, max_y_dim*2, '] Cx:', cx, 'SY: ', sx)
                print('   cy:', cy, ' sy: ', sy, '  lowest car  :',   self.lowest_car, ' highest car :',   self.highest_car)
                print('   cx:', cx, ' sx: ', sx, '  leftmost car:', self.leftmost_car,  ' rightmost car :', self.rightmost_car)            
                print('   Final (cx,cy,sx,sy): ', cx,cy,sx,sy)
            self.cars.append([cx,cy,sx,sy,False])


        elif shape == "tree":
            if verbose:
                print(' Build Tree')
            color = random.choice(Image.treecolors)    
            group_range_y = 25
            group_range_x = 25

            ## 1 - Get SX , SY between limits. 
            # max_y_dim        = self.config.max_dim[shape]
            # min_y_dim        = self.config.min_dim[shape]

            ## 2 - get CX, CY between allowable limits
            dflt_min_range_y = self.horizon[0] - (min_y_dim //2)
            dflt_max_range_y = self.height     - (max_y_dim //2)
            dflt_min_range_x = 0               # self.config.Min_X[shape]
            dflt_max_range_x = self.width      #    self.config.Max_X[shape]
            if verbose:
                print('   First Tree       : ', self.first_tree)
                print('   Horizion         : ', self.horizon[0])
                print('   lowest building  : ', self.lowest_building  , ' highest building  : ', self.highest_building)
                print('   lowest car       : ', self.lowest_car       , ' highest car       : ', self.highest_car)
                print('   dflt min rage    : ', dflt_min_range_y      , ' dflt max range    : ', dflt_max_range_y)

            
            if self.built_counts[shape] == 0 :
                min_range_x = dflt_min_range_x
                max_range_x = dflt_max_range_x
                min_range_y = dflt_min_range_y
                max_range_y = min(self.highest_car + 10, dflt_max_range_y) 
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


        elif shape == "sun":
            color = random.choice(Image.suncolors)    
            if verbose:
                print(' Build Sun')
                print('  Sun Colors is :', color, type(color), color.dtype)
            cx = random.randint(min_range_x, max_range_x)
            cy = random.randint(min_range_y, max_range_y)

            sy = int(np.interp([cy],[min_range_y, max_range_y], [min_y_dim, max_y_dim]))
            sx = sy


        elif shape == "cloud":
            if verbose:
                print(' Build Cloud')
            color = random.choice(Image.cloudcolors)            

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

        elif shape == "old cloud":
            print(' Build Cloud')
            color = random.choice(Image.cloudcolors)            
            cx = random.randint(min_range_x, max_range_x)
            cy = random.randint(min_range_y, max_range_y)

            sx = int(np.interp([cy],[min_range_y, max_range_y], [min_y_dim, max_y_dim]))
        #     min_height ,max_height = 10, 20
        #     sy = random.randint(min_height, max_height)
            sx = sy *  random.randint(3, 5)

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
    def find_hidden_shapes(self, shapes):
        '''
        A variation of load_masks customized to find objects that
        are completely hidden by other shapes
        '''

        # print('\n Load Mask information (shape, (color rgb), (x_ctr, y_ctr, size) ): ')
        # p4.pprint(info['shapes'])
        hidden_shapes = []
        count  = len(shapes)
        mask   = np.zeros( [self.height, self.width, count], dtype=np.uint8)

        ## get masks for each shape
        for i, (shape, _, dims) in enumerate(shapes):
            mask[:, :, i:i + 1] = draw_object(mask[:, :, i:i + 1].copy(), shape, dims, 1)

        #----------------------------------------------------------------------------------
        #  Start with last shape as the occlusion mask
        #   Occlusion starts with the last object an list and in each iteration of the loop
        #   adds an additional  object. Pixes assigned to objects are 0. Non assigned pixels
        #   are 1
        #-----------------------------------------------------------------------------------
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)

        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

            ##-------------------------------------------------------------------------------------
            ## if the shape has been completely occluded by other shapes, it's mask is all zeros.
            ## in this case np.any(mask) will return FALSE.
            ## For these completely hidden objects, we record their id in hidden []
            ## and later remove them from the  list of shapes
            ##-------------------------------------------------------------------------------------
            if ( ~np.any(mask[:,:,i]) ) :
                # print(' !!!!!!  Zero Mask Found !!!!!!' )
                hidden_shapes.append(i)

        # if len(hidden_shapes) > 0 :
            # print(' ===> Find Hidden Shapes() found hidden objects ')
            # p8.pprint(shapes)
            # print(' ****** Objects completely hidden are : ', hidden_shapes)
            # for i in hidden_shapes:
                # p8.pprint(shapes[i])
        return hidden_shapes



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
            print(' **  ix : ', i, 'ctr (x,y)', cx,' ',cy,'\n box:', boxes[i], ' compare ',i, ' with ', ixs[1:])
            print('     area[i]: ', area[i], 'area[ixs[1:]] :',area[ixs[1:]] )   

        # Compute IoU of the picked box with the rest
        iou = debug_compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        
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
        iou, inter, union, occlusion = debug_compute_iou_2(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
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

  
def get_max_occlusion(new_shape, other_shapes, verbose = False ):
    '''
    Determined occlusion between new_shape and existing shapes  
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    '''
    if verbose:
        print('====> get_max_occlusion()  - len(other_shapes) :', len(other_shapes))
    if len(other_shapes) == 0:
        return np.array([0], dtype=np.float)

    x, y, sx, sy = new_shape[2]
    new_box = np.array([y - sy, x - sx, y + sy, x + sx])
    new_class = new_shape[0]
    new_area = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1]) 
    
    boxes = []
    classes = []
    for shp in other_shapes:
        # print('   class:', shp[0], '     box:', shp[2])
        x, y, sx, sy = shp[2]
        boxes.append([y - sy, x - sx, y + sy, x + sx])
        classes.append(shp[0])
    boxes   = np.array(boxes)
    # print('       Boxes shape', boxes.shape)
    classes = np.array(classes)
    scores  = np.arange(len(boxes))
    areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1]) 
    cy = boxes[:,0] + (boxes[:,2] - boxes[:,0]) //2
    cx = boxes[:,1] + (boxes[:,3] - boxes[:,1]) //2
    
    if verbose:
        print('     Test: ', new_box , '     area: ', new_area, '   class ', new_class)
        print('       against:')
        for box, cls, scr ,ar, y,x  in zip(boxes, classes, scores, areas, cy,cx):
            print('     scr:', scr, '  ', box, '   ', ar,  '    ', cls,'   CY/CX:   ',y,x)

    # Compute cliiped box areas
    clipped_boxes = np.zeros_like(boxes)
    clp_y1 = clipped_boxes[:,0] = np.maximum(boxes[:, 0], 0)    ## y1
    clp_x1 = clipped_boxes[:,1] = np.maximum(boxes[:, 1], 0)    ## x1
    clp_y2 = clipped_boxes[:,2] = np.minimum(boxes[:, 2], 128)  ## y2
    clp_x2 = clipped_boxes[:,3] = np.minimum(boxes[:, 3], 128)  ## x2
    clp_areas = (clp_y2 - clp_y1) * (clp_x2 - clp_x1) 

    if verbose:
        print('====> After Clipping ')
        for box,cls,scr, ar in zip(clipped_boxes, classes, scores, clp_areas):
            print('     scr:', scr, '  ', box, '   ', ar, '    ', cls)
    
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)
      
    # Get indicies of boxes sorted by scores (highest first)
    # ixs = scores.argsort()[::-1]
    # Get indicies of boxes sorted by Area (LOWEST first)
    # ixs = area.argsort()
    # Get indicies of boxes sorted by highest in image (Top objects  first)
    # ixs = boxes[:,0].argsort()

    # pick = []
    # if verbose:
        # print('====> After Sorting - sort indices: ', ixs)
        # for i in ixs : 
            # print( 'scr:', scores[i], '  ', clipped_boxes[i], '   ', area[i], '    ', classes[i])

    
    # while len(ixs) > 0:
    #     # Pick top box and add its index to the list
    #     i = ixs[0]
    #     cls = classes[i]
    #     pick.append(i)
        
        # Compute IoU of the picked box with the rest
    iou, inter, union, occlusion = debug_compute_iou_2(new_box, boxes, new_area, areas)
    if verbose:
        print()
        print(' **  new shape : ',new_class,'     box:', new_box, '     area: ', new_area)
        print('           clsses: ', ''.join( [i.rjust(11) for i in classes]))
        print('            areas: ', areas )   
        print('              iou: ', iou)
        print('     intersection: ', inter)
        print('            union: ', union)
        print('        occlusion: ', occlusion)

        # Identify boxes with IoU over the threshold. This returns indicies into ixs[1:], so add 1 to get
        # indicies into ixs.
        
        # tst =  np.where(iou>threshold)
        # remove_ixs = np.where(iou > threshold)[0] + 1
        
        # Remove indicies of the picked and overlapped boxes.
        # ixs = np.delete(ixs, remove_ixs)
        # ixs = np.delete(ixs, 0)

        # if verbose:
            # print('     np.where( iou > threshold) : ' ,tst, 'tst[0] (index into ixs[1:]: ', tst[0], 
                    # ' remove_ixs (index into ixs) : ',remove_ixs)
            # print('     ending ixs (after deleting ixs[0]): ', ixs, ' picked so far: ',pick)
    # if verbose:    
        # print('====> Final Picks: ', pick)

    return occlusion

     
def debug_compute_iou_2(box, boxes, box_area, boxes_area, verbose = False):
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