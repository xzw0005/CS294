'''
Created on Dec 20, 2018

@author: wangxing
'''
import cv2
import numpy as np
from collections import deque
import gym
from gym import spaces

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset. No-op is assumed to be action 0."""
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
        
    def _reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        noops = np.random.randint(1, self.noop_max+1)
        for _ in range(noops):
            obs, _, _, _ = self.env.step(0)
        return obs
    
class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Take action on reset for environments that are fixed until firing."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[0] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3
        
    def _reset(self):
        self.env.reset()
        obs, _, _, _ = self.env.step(1)
        obs, _, _, _ = self.env.step(2)
        return obs 

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        pass 
    
    def _step(self, action):
        pass
    
    def _reset(self):
        pass
    
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        pass 
    
    def _step(self, action):
        pass
    
    def _reset(self):
        pass
    
    
    
    
    
    
    
    