import numpy as np
import tensorflow as tf
import functools
import copy
import os 
import collections 

# Extras
# ----------------------------------
def l2loss(params):
    if len(params) == 0:
        return tf.constant(0.)
    return tf.add_n([sum(tf.square(p)) for p in params])

def lrelu(x, leak=.2):
    f1 = .5 * (1 + leak)
    f2 = .5 * (1 - leak)
    return f1 * x + f2 * abs(x)

def categorical_sample_logits(X):
    U = tf.random_uniform(tf.shape(X))
    return tf.argmax(X - tf.log(-tf.log(U)), dimension=1)


class _Function(object):
    def __init__(self, inputs, outputs, updates, givens, check_nan=False):
        assert all(len(i.op.inputs) == 0 for i in inputs), "inputs should all be placeholders"
        self.inputs = inputs 
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens 
        self.check_nan = check_nan 
        
    def __call__(self, *inputvals):
        assert len(inputvals) == len(self.inputs)
        feed_dict = dict(zip(self.inputs, inputvals))
        feed_dict.update(self.givens)
        results = tf.get_default_session().run(self.outputs_update, feed_dict=feed_dict)[:-1]
        if self.check_nan:
            if any(np.isnan(r).any() for r in results):
                raise RuntimeError("Nan dectected")
        return results
    
def function(inputs, outputs, updates=None, givens=None):
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        f = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *inputs : type(outputs)(zip(outputs.keys(), f(*inputs)))
    else:
        f = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *inputs : f(*inputs)[0]
    
ALREADY_INITIALIZED = set()
def initialize():
    new_vars = set(tf.all_variables()) - ALREADY_INITIALIZED
    tf.get_default_session().run(tf.initialize_variables(new_vars))
    ALREADY_INITIALIZED.update(new_vars)
