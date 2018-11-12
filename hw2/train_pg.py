'''
Created on Oct 26, 2018

@author: wangxing
'''
import numpy as np
import tensorflow as tf 
import gym 
import os
import time 
import inspect
from multiprocessing import Process 

import hw2.logz as logz

def build_mlp(input_placeholder, output_size, scope, n_layers, size, activation=tf.tanh, output_activation=None):
    x = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            x = tf.layers.dense(x, size, activation=activation)
        x = tf.layers.dense(x, output_size, activation=output_activation)
    return x

def pathlength(path):
    return len(path['reward'])

def setup_logger(logdir, locals_):
    logz.configure_output_dir(logdir)
    args = inspect.getargspec(train_PG)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)
    
def train_PG():
    pass

class Agent(object):
    def __init__(self, computation_graph_args, sample_trajectory_args, estimate_return_args):
        super(Agent, self).__init__()
        self.ob_dim = computation_graph_args['ob_dim']
        self.ac_dim = computation_graph_args['ac_cim']
        self.discrete = computation_graph_args['discrete']
        self.size = computation_graph_args['size']
        self.n_layers = computation_graph_args['n_layers']
        self.learning_rate = computation_graph_args['learning_rate']
        
        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']
        
        self.gamma = estimate_return_args['gamma']
        self.reward_to_go = estimate_return_args['reward_to_go']
        self.nn_baseline = estimate_return_args['nn_baseline']
        self.normalize_advantages = estimate_return_args['normalize_advantages']
        
    def init_tf_sess(self):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__()   # equivalent to 'with self.sess'
        tf.global_variables_initializer()
        
    def build_computaion_graph(self):
        # Define placeholders
        self.sy_ob_no, self.sy_ac_na, self.adv_n = self.define_placeholder()
        # The policy takes an observation as input and outputs a distribution over the action space
        self.policy_parameters = self.policy_forward_pass(self.sy_ob_no)
        # Sample actions
        self.sy_sampled_ac = self.sample_action(self.policy_parameters)
        # Compute logprob of actions actually taken by the policy
        self.sy_logprob_n = self.get_log_prob(self.policy_parameters, self.sy_ac_na)
        # Loss function & training operation
        loss = tf.reduce_mean(-self.sy_logprob_n * self.sy_adv_n)
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
    
    def get_log_prob(self, policy_parameters, sy_ac_na):
        if self.discrete:
            sy_logits_na = policy_parameters
            ##### TODO
            sy_logprob_n = None
        else:
            sy_mean, sy_logstd = policy_parameters
            ##### TODO
            sy_logprob_n = None
        return sy_logprob_n
    
    def sample_action(self, policy_parameters):
        if self.discrete:
            sy_logits_na = policy_parameters
            sy_sampled_ac = tf.multinomial(logits=sy_logits_na, num_samples=1)[0]
        else:
            sy_mean, sy_logstd = policy_parameters
            sy_sampled_ac = tf.random_normal(shape=sy_mean.shape, mean=sy_mean, stddev=tf.exp(sy_logstd), dtype=tf.float32)
        return sy_sampled_ac


    def policy_forward_pass(self, sy_ob_no):
        logits = build_mlp(sy_ob_no, self.ac_dim, 'policy_model', self.n_layers, self.size)
        if self.discrete:       # NN outputs the logits of a categorical distribution
            sy_logits_na = logits
            return sy_logits_na
        else:                   # returns (mean, log_std) tuple of a Gaussian distribution over actions
            sy_mean = logits
            sy_logstd = tf.get_variable('logstd', [self.ac_dim], dtype=tf.float32)
            return (sy_mean, sy_logstd)
        

    def define_placeholder(self):
        # placeholder for observations
        sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], dtype=tf.float32, name='ob')
        # placeholder for actions
        if self.discrete:
            sy_ac_na = tf.placeholder(shape=[None], dtype=tf.int32, name='ac')                  
        else:
            sy_ac_na = tf.placeholder(shape=[None, self.ac_dim], dtype=tf.float32, name='ac')
        # placeholder for advantages
        sy_adv_n = tf.placeholder(shape=[None], dtype=tf.float32, name='adv')
        return sy_ob_no, sy_ac_na, sy_adv_n
    
    