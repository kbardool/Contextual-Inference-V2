# -*- coding: utf-8 -*-
"""Training-related part of the Keras engine.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import copy
import numpy as np
from scipy.sparse import issparse

# from .topology import Container
# from .topology import Layer
from keras import backend as K
from keras import optimizers
from keras import losses
from keras import metrics as metrics_module
from keras.utils.data_utils import Sequence
from keras.utils.data_utils import GeneratorEnqueuer
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.generic_utils import Progbar
from keras import callbacks as cbks
# from ..legacy import interfaces
import pprint
pp = pprint.PrettyPrinter(indent=4, width=100)

def _standardize_sample_or_class_weights(x_weight, output_names, weight_type):
    """Maps `sample_weight` or `class_weight` to model outputs.

    # Arguments
        x_weight: User-provided `sample_weight` or `class_weight` argument.
        output_names: List of output names (strings) in the model.
        weight_type: A string used purely for exception printing.

    # Returns
        A list of `sample_weight` or `class_weight` where there are exactly
            one element per model output.

    # Raises
        ValueError: In case of invalid user-provided argument.
    """
    if x_weight is None or len(x_weight) == 0:
        return [None for _ in output_names]
    if len(output_names) == 1:
        if isinstance(x_weight, list) and len(x_weight) == 1:
            return x_weight
        if isinstance(x_weight, dict) and output_names[0] in x_weight:
            return [x_weight[output_names[0]]]
        else:
            return [x_weight]
    if isinstance(x_weight, list):
        if len(x_weight) != len(output_names):
            raise ValueError('Provided `' + weight_type + '` was a list of ' +
                             str(len(x_weight)) +
                             ' elements, but the model has ' +
                             str(len(output_names)) + ' outputs. '
                             'You should provide one `' + weight_type + '`'
                             'array per model output.')
        return x_weight
    if isinstance(x_weight, dict):
        x_weights = []
        for name in output_names:
            x_weights.append(x_weight.get(name))
        return x_weights
    else:
        raise TypeError('The model has multiple outputs, so `' +
                        weight_type + '` '
                        'should be either a list or a dict. '
                        'Provided `' + weight_type +
                        '` type not understood: ' +
                        str(x_weight))


def _standardize_class_weights(class_weight, output_names):
    return _standardize_sample_or_class_weights(class_weight,
                                                output_names,
                                                'class_weight')


def _standardize_sample_weights(sample_weight, output_names):
    return _standardize_sample_or_class_weights(sample_weight,
                                                output_names,
                                                'sample_weight')


def _check_array_lengths(inputs, targets, weights=None):
    """Checks if batch axes are the same for numpy arrays.

    # Arguments
        inputs: list of Numpy arrays of inputs.
        targets: list of Numpy arrays of targets.
        weights: list of Numpy arrays of sample weights.

    # Raises
        ValueError: in case of incorrectly formatted data.
    """
    def set_of_lengths(x):
        # return a set with the variation between
        # different shapes, with None => 0
        if x is None:
            return {0}
        else:
            return set([0 if y is None else y.shape[0] for y in x])

    set_x = set_of_lengths(inputs)
    set_y = set_of_lengths(targets)
    set_w = set_of_lengths(weights)
    if len(set_x) > 1:
        raise ValueError('All input arrays (x) should have '
                         'the same number of samples. Got array shapes: ' +
                         str([x.shape for x in inputs]))
    if len(set_y) > 1:
        raise ValueError('All target arrays (y) should have '
                         'the same number of samples. Got array shapes: ' +
                         str([y.shape for y in targets]))
    if set_x and set_y and list(set_x)[0] != list(set_y)[0]:
        raise ValueError('Input arrays should have '
                         'the same number of samples as target arrays. '
                         'Found ' + str(list(set_x)[0]) + ' input samples '
                         'and ' + str(list(set_y)[0]) + ' target samples.')
    if len(set_w) > 1:
        raise ValueError('All sample_weight arrays should have '
                         'the same number of samples. Got array shapes: ' +
                         str([w.shape for w in weights]))
    if set_y and set_w and list(set_y)[0] != list(set_w)[0]:
        raise ValueError('Sample_weight arrays should have '
                         'the same number of samples as target arrays. Got ' +
                         str(list(set_y)[0]) + ' input samples and ' +
                         str(list(set_w)[0]) + ' target samples.')


def _check_loss_and_target_compatibility(targets, loss_fns, output_shapes):
    """Does validation on the compatibility of targets and loss functions.

    This helps prevent users from using loss functions incorrectly.

    # Arguments
        targets: list of Numpy arrays of targets.
        loss_fns: list of loss functions.
        output_shapes: list of shapes of model outputs.

    # Raises
        ValueError: if a loss function or target array
            is incompatible with an output.
    """
    key_losses = {losses.mean_squared_error,
                  losses.binary_crossentropy,
                  losses.categorical_crossentropy}
    for y, loss, shape in zip(targets, loss_fns, output_shapes):
        if y is None or loss is None:
            continue
        if loss is losses.categorical_crossentropy:
            if y.shape[-1] == 1:
                raise ValueError(
                    'You are passing a target array of shape ' + str(y.shape) +
                    ' while using as loss `categorical_crossentropy`. '
                    '`categorical_crossentropy` expects '
                    'targets to be binary matrices (1s and 0s) '
                    'of shape (samples, classes). '
                    'If your targets are integer classes, '
                    'you can convert them to the expected format via:\n'
                    '```\n'
                    'from keras.utils import to_categorical\n'
                    'y_binary = to_categorical(y_int)\n'
                    '```\n'
                    '\n'
                    'Alternatively, you can use the loss function '
                    '`sparse_categorical_crossentropy` instead, '
                    'which does expect integer targets.')
        if loss in key_losses:
            for target_dim, out_dim in zip(y.shape[1:], shape[1:]):
                if out_dim is not None and target_dim != out_dim:
                    raise ValueError(
                        'A target array with shape ' + str(y.shape) +
                        ' was passed for an output of shape ' + str(shape) +
                        ' while using as loss `' + loss.__name__ + '`. '
                        'This loss expects '
                        'targets to have the same shape '
                        'as the output.')


def _collect_metrics(metrics, output_names):
    """Maps metric functions to model outputs.

    # Arguments
        metrics: a list or dict of metric functions.
        output_names: a list of the names (strings) of model outputs.

    # Returns
        A list (one entry per model output) of lists of metric functions.
        For instance, if the model has 2 outputs, and for the first output
        we want to compute "binary_accuracy" and "binary_crossentropy",
        and just "binary_accuracy" for the second output,
        the list would look like:
            `[[binary_accuracy, binary_crossentropy], [binary_accuracy]]`

    # Raises
        TypeError: if an incorrect type is passed for the `metrics` argument.
    """
    if not metrics:
        return [[] for _ in output_names]
    if isinstance(metrics, list):
        # we then apply all metrics to all outputs.
        return [copy.copy(metrics) for _ in output_names]
    elif isinstance(metrics, dict):
        nested_metrics = []
        for name in output_names:
            output_metrics = metrics.get(name, [])
            if not isinstance(output_metrics, list):
                output_metrics = [output_metrics]
            nested_metrics.append(output_metrics)
        return nested_metrics
    else:
        raise TypeError('Type of `metrics` argument not understood. '
                        'Expected a list or dictionary, found: ' +
                        str(metrics))


            
def _standardize_input_data(data, names, shapes=None,
                            check_batch_axis=True,
                            exception_prefix=''):
    """Normalizes inputs and targets provided by users.

    Users may pass data as a list of arrays, dictionary of arrays,
    or as a single array. We normalize this to an ordered list of
    arrays (same order as `names`), while checking that the provided
    arrays have shapes that match the network's expectations.

    # Arguments
        data: User-provided input data (polymorphic).
        names: List of expected array names.
        shapes: Optional list of expected array shapes.
        check_batch_axis: Boolean; whether to check that
            the batch axis of the arrays matches the expected
            value found in `shapes`.
        exception_prefix: String prefix used for exception formatting.

    # Returns
        List of standardized input arrays (one array per model input).

    # Raises
        ValueError: in case of improperly formatted user-provided data.

    """
    print(' Stand Inptu data :', type(data), len(data))
    print('             names:', names)
    print('            shapes:', shapes)
    if not names:
        if data is not None and hasattr(data, '__len__') and len(data):
            raise ValueError('Error when checking model ' +
                             exception_prefix + ': '
                             'expected no data, but got:', data)
        return []
    if data is None:
        return [None for _ in range(len(names))]

    if isinstance(data, dict):
        try:
            data = [data[x].values if data[x].__class__.__name__ == 'DataFrame' else data[x] for x in names]
        except KeyError as e:
            raise ValueError(
                'No data provided for "' + e.args[0] + '". Need data '
                'for each key in: ' + str(names))
    elif isinstance(data, list):
        if len(names) == 1 and data and isinstance(data[0], (float, int)):
            data = [np.asarray(data)]
        else:
            print(' -- data is list len <> 1')
            data = [x.values if x.__class__.__name__ == 'DataFrame' else x for x in data]
    else:
        data = data.values if data.__class__.__name__ == 'DataFrame' else data
        data = [data]
        
    data = [np.expand_dims(x, 1) if x is not None and x.ndim == 1 else x for x in data]

    if data is not None:
        for i in data:
            if i is not None :
                print(' Data shape:', i.shape)
            else :
                print(' None')
        
    if len(data) != len(names):
        if data and hasattr(data[0], 'shape'):
            raise ValueError(
                'Error when checking model ' + exception_prefix +
                ': the list of Numpy arrays that you are passing to '
                'your model is not the size the model expected. '
                'Expected to see ' + str(len(names)) + ' array(s), '
                'but instead got the following list of ' +
                str(len(data)) + ' arrays: ' + str(data)[:200] + '...')
        elif len(names) > 1:
            raise ValueError(
                'Error when checking model ' + exception_prefix +
                ': you are passing a list as input to your model, '
                'but the model expects a list of ' + str(len(names)) +
                ' Numpy arrays instead. The list you passed was: ' +
                str(data)[:200])
        elif len(data) == 1 and not hasattr(data[0], 'shape'):
            raise TypeError(
                'Error when checking model ' + exception_prefix +
                ': data should be a Numpy array, or list/dict of '
                'Numpy arrays. Found: ' + str(data)[:200] + '...')
        elif len(names) == 1:
            data = [np.asarray(data)]

    # Check shapes compatibility.
    if shapes:
        for i in range(len(names)):
            print(' Check shapes compatibility:', names[i], shapes[i])
            if shapes[i] is not None:
                data_shape = data[i].shape
                shape = shapes[i]
                if data[i].ndim != len(shape):
                    raise ValueError(
                        'Error when checking ' + exception_prefix +
                        ': expected ' + names[i] + ' to have ' +
                        str(len(shape)) + ' dimensions, but got array '
                        'with shape ' + str(data_shape))
                if not check_batch_axis:
                    data_shape = data_shape[1:]
                    shape = shape[1:]
                for dim, ref_dim in zip(data_shape, shape):
                    if ref_dim != dim and ref_dim:
                        raise ValueError(
                            'Error when checking ' + exception_prefix +
                            ': expected ' + names[i] + ' to have shape ' +
                            str(shape) + ' but got array with shape ' +
                            str(data_shape))
    return data


    
def my_standardize_user_data(self, x, y,
                           sample_weight=None, class_weight=None,
                           check_array_lengths=True, batch_size=None):
    if not hasattr(self, 'optimizer'):
        raise RuntimeError('You must compile a model before '
                           'training/testing. '
                           'Use `model.compile(optimizer, loss)`.')

    output_shapes = []
    for output_shape, loss_fn in zip(self._feed_output_shapes, self._feed_loss_fns):
        if loss_fn is losses.sparse_categorical_crossentropy:
            output_shapes.append(output_shape[:-1] + (1,))
        elif (not hasattr(loss_fn, '__name__') or
              getattr(losses, loss_fn.__name__, None) is None):
            # If `loss_fn` is not a function (e.g. callable class)
            # or if it not in the `losses` module, then
            # it is a user-defined loss and we make no assumptions
            # about it.
            print(' user defined loss -- make no assumptions!')
            output_shapes.append(None)
        else:
            print(' NOT user defined loss -- make assumptions!')
            output_shapes.append(output_shape)
    # `check_batch_axis` is set to False since `x` may contain multiple batches
    #  and in general `x[0].shape[0] != self._feed_input_shapes[0][0]`
    print(' feed input names:' , self._feed_input_names)
    print(' feed input shapes: ', self._feed_input_shapes)
    
    x = _standardize_input_data(x, self._feed_input_names,
                                self._feed_input_shapes,
                                check_batch_axis=False,
                                exception_prefix='input')
                                
    print('self._feed_output_names:' , self._feed_output_names)
    print('self._feed_output_shapes: ', self._feed_output_shapes)
    print('  output shapes    :', output_shapes)
    print('self.outputNames   :', self.output_names)
    print('self.outputShapes  :', self.output_shape)
    # for i in y:
        # print(i.shape)

    # y = _standardize_input_data(y, self.output_names,
                                # self.output_shape,
                                # check_batch_axis=False,
                                # exception_prefix='target')
    y = _standardize_input_data(y, self._feed_output_names,
                                output_shapes,
                                check_batch_axis=False,
                                exception_prefix='target')
                                
                                
    sample_weights= _standardize_sample_weights(sample_weight,
                                                 self._feed_output_names)
    class_weights = _standardize_class_weights(class_weight,
                                               self._feed_output_names)
    sample_weights = [_standardize_weights(ref, sw, cw, mode)
                      for (ref, sw, cw, mode)
                      in zip(y, sample_weights, class_weights, self._feed_sample_weight_modes)]

    if check_array_lengths:
        _check_array_lengths(x, y, sample_weights)
    _check_loss_and_target_compatibility(y,
                                         self._feed_loss_fns,
                                         self._feed_output_shapes)
    if self.stateful and batch_size:
        if x[0].shape[0] % batch_size != 0:
            raise ValueError('In a stateful network, '
                             'you should only pass inputs with '
                             'a number of samples that can be '
                             'divided by the batch size. Found: ' +
                             str(x[0].shape[0]) + ' samples')
    return x, y, sample_weights
            