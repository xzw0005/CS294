import os
import pickle 
import tensorflow as tf
import numpy as np
import gym

import argparse

import tf_util as tf_util
from load_policy import load_policy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--max_timesteps', type=int)
    parser.add_argument('--num_rollouts', type=int, default=20, help='Number of expert roll outs')
    args = parser.parse_args()
    
    print('loading and building expert policy')
    policy_fn = load_policy(args.expert_policy_file)
    print('loaded and built')
    
    with tf.Session():
        tf_util.initialize()
        
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit
        
        returns = []
        observations = []
        actions = [] 
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            total_rewards = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None, :])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                total_rewards += r 
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0:
                    print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(total_rewards)
            
        print("Episodic returns", returns)
        print("Mean return", np.mean(returns))
        print("Std of return", np.std(returns))
        
        expert_data = {'observations': np.array(observations), 'actions': np.array(actions)}
        
        with open(os.path.join('expert_data', args.envname + '.pkl'), 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)
        
if __name__ == "__main__":
    main()