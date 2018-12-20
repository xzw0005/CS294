'''
Created on Dec 20, 2018

@author: wangxing
'''

import numpy as np 
import tensorflow as tf 
import time

class Agent(object):
    
    def __init__(self, computation_graph_args, sample_trajectory_args, estimate_return_args):
        '''
        Constructor
        '''
        super(Agent, self).__init__()
        self.obs_dim = computation_graph_args['obs_dim']
        self.act_dim = computation_graph_args['act_dim']
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