"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import glob
import random
import math
import datetime
import itertools
import time
import json
import re
import logging
from collections import OrderedDict
import numpy as np
import scipy.misc
import tensorflow    as tf
import keras
import keras.backend as KB
import keras.layers  as KL
import keras.initializers as KI
import keras.engine  as KE
import keras.models  as KM
import pprint 
pp = pprint.PrettyPrinter(indent=4, width=100)

# def get_layer_output(model, output_layer, model_input, training_flag = True):
    # get_output = KB.function([model.input]+[KB.learning_phase()],
                              # [model.layers[output_layer].output])
    # output = get_output([model_input,training_flag])[0]                                
    # return output
    
class MyCallback(keras.callbacks.Callback):

    def __init__(self): 

        return 
        
        # , pool_shape, image_shape, **kwargs):
        # super(PyramidROIAlign, self).__init__(**kwargs)
        # self.pool_shape = tuple(pool_shape)
        # self.image_shape = tuple(image_shape)

    def on_epoch_begin(self, epoch, logs = {}) :
        print('\n>>> Start epoch {}  \n'.format(epoch))
        return 

    def on_epoch_end  (self, epoch, logs = {}): 
        print('\n>>>End   epoch {}  \n'.format(epoch))
        return 

    def on_batch_begin(self, batch, logs = {}):
        print('\n... Start training of batch {} size {} '.format(batch,logs['size']))
        pp.pprint(self.model._feed_inputs)
        k_sess = KB.get_session()
        # self.model._feed_inputs[1].eval(session=k_sess)
        return  
        
    def on_batch_end  (self, batch, logs = {}): 
        print('\n... End   training of batch {} '.format(batch,logs['loss']))
        pp.pprint(logs)
        return                                          
        
    def on_train_begin(self,logs = {}):        
        pp = pprint.PrettyPrinter(indent=4)
        print('\n *****  Start of Training {} '.format(time.time()))
        return 
        
    def on_train_end  (self,logs = {}):        
        pp = pprint.PrettyPrinter(indent=4)  
        print('\n\n')
        print('***** End of Training   {} '.format(time.time()))    
        return 
