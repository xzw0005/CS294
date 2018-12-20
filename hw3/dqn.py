'''
Created on Dec 20, 2018

@author: wangxing
'''
import uuid
import time 
import pickle
import sys 
import gym.spaces
import itertools
import random 
import numpy as np
import tensorflow as tf 
import tensorflow.contrib.layers as layers
from collections import namedtuple
from dqn_utils import *

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

class QLearner(object):
    def __init__(self, env, q_func, optimizer_spec, session, exploration=LinearSchedule(1000000, 0.1), stopping_criterion=None,\
                 replay_buffer_size=1000000, batch_size=32, gamma=0.99, learning_starts=50000, learning_freq=4, frame_history_len=4,
                 target_update_freq=10000, grad_norm_clipping=10, rew_file=None, double_q=True, lander=False):
        assert type(env.observation_space) == gym.spaces.Box
        assert type(env.action_space) == gym.spaces.Discrete 
        self.target_update_freq = target_update_freq
        self.optimizer_spec = optimizer_spec
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.stopping_criterion = stopping_criterion
        self.env = env 
        self.session = session
        self.exploration = exploration
        self.rew_file = str(uuid.uuid4()) + '.pkl' if rew_file is None else rew_file
        ## Build Model ##
        if len(self.env.observation_space.shape) == 1:
            input_shape = self.env.observation_space.shape 
        else:
            img_h, img_w, img_c = self.env.observation_space.shape 
            input_shape = (img_h, img_w, frame_history_len * img_c)
            self.num_actions = self.env.action_space.n 
        # set up placeholders
        self.obs_t_ph = tf.placeholder(dtype=tf.float32 if lander else tf.uint8, shape=[None]+list(input_shape))
        self.act_t_ph = tf.placeholder(dtype=tf.int32, shape=[None])
        self.rew_t_ph = tf.placeholder(dtype=tf.float32, shape=[None])
        self.obs_tp1_ph = tf.placeholder(dtype=tf.float32 if lander else tf.uint8, shape=[None]+list(input_shape))
        self.done_mask_ph =tf.placeholder(dtype=tf.float32, shape=[None])
        if lander:
            obs_t_float = self.obs_t_ph
            obs_tp1_float = self.obs_tp1_ph
        else:
            obs_t_float = tf.cast(self.obs_t_ph, tf.float32) / 255.
            obs_tp1_float = tf.cast(self.obs_tp1_ph, tf.float32) / 255.
        
        
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=(), name="learning_rate")
        optimizer = self.optimizer_spec.constructor(learning_rate=self.learning_rate, **self.optimizer_spec.kwargs)
        self.train_fn = minimize_and_clip(optimizer, objective=self.total_error, var_list=q_func_vars, clip_val=grad_norm_clipping)
        
        self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, lander=lander)
        self.replay_buffer_idx = None 
        
        ## Run Env ##
        self.model_initialized = False
        self.num_param_updates = 0
        self.mean_episode_reward = -float('nan')
        self.best_mean_episode_reward = -float('inf')
        self.last_obs = self.env.reset()
        
        self.start_time = None
        self.t = 0    
    
    def stopping_criterion_met(self):
        return self.stopping_criterion is not None and self.stopping_criterion(self.env, self.t)
    
    def step_env(self):
        pass
    
    def update_model(self):
        pass
    
    def log_progress(self):
        episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)
        if self.t % self.log_every_n_steps == 0 and self.model_initialized:
            print("Timestep %d"%(self.t,))
            print("mean reward (100 episodes) %f" % self.mean_episode_reward)
            print("best mean reward %f" % self.best_mean_episode_reward)
            print("episode %d" % len(episode_rewards))
            print("exploration %f" % self.optimizer_spec.lr_schedule.value(self.t))
            print("learning_rate %f" % self.optimizer_spec.lr_schedule.value(self.t))
            if self.start_time is not None:
                print("running time %f" % ((time.time - self.start_time)/60.) )
            self.start_time = time.time()
            sys.stdout.flush()
            with open(self.rew_file, 'wb') as f:
                pickle.dump(episode_rewards, f, pickle.HIGHEST_PROTOCOL)
    
def learn(*args, **kwargs):
    agent = QLearner(*args, **kwargs)
    while not agent.stopping_criterion_met():
        agent.step_env()
        agent.update_model()
        agent.log_progress()