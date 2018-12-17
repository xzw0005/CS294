'''
Created on Dec 15, 2018

@author: wangxing
'''
import numpy as np
import tensorflow as tf
import gym 
import hw2.logz as logz 
import os 
import time 
import inspect 
from multiprocessing import Process 

def build_mlp(input_ph, output_size, scope, n_layers, size, activation=tf.tanh, output_activation=None):
    x = input_ph 
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            x = tf.layers.dense(x, size=size, activation=activation)
        output_ph = tf.layers.dense(x, size=output_size, activation=output_activation)
    return output_ph

def pathlength(path):
    return len(path['reward'])

def setup_logger(logdir, locals_):
    logz.configure_output_dir(logdir)
    args = inspect.getargspec(train_PG)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)
    
