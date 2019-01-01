'''
Created on Oct 26, 2018

@author: wangxing
'''
import numpy as np
import tensorflow as tf
import gym
import argparse
import os
import time
import inspect
import multiprocessing as mp

import logz
from PgAgent import Agent

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(trainPG)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

def trainPG(exp_name, env_name, n_iter, gamma, min_timesteps_per_batch, max_path_length,\
            learning_rate, reward_to_go, animate, logdir, normalize_advantages, nn_baseline,\
            seed, n_layers, size):
    tic = time.time()
    setup_logger(logdir, locals())
    
    env = gym.make(env_name)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    
    max_path_length = max_path_length or env.spec.max_episode_steps 
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n if discrete else env.action_space.shape[0]
    
    ## Initialize Agent
    computation_graph_args = {
            'n_layers': n_layers,
            'obs_dim': obs_dim,
            'act_dim': act_dim,
            'discrete': discrete,
            'size': size,
            'learning_rate': learning_rate,
        }
    sample_trajectory_args = {
            'animate': animate,
            'max_path_length': max_path_length,
            'min_timesteps_per_batch': min_timesteps_per_batch,
        }
    estimate_return_args = {
            'gamma': gamma,
            'reward_to_go': reward_to_go,
            'nn_baseline': nn_baseline,
            'normalize_advantages': normalize_advantages,
        }
    agent = Agent(computation_graph_args, sample_trajectory_args, estimate_return_args)
    agent.build_computation_graph()
    agent.init_tf_sess()
    
    ## Training Loop
    total_time_steps = 0
    for itr in range(n_iter):
        print("********* Iteration %i *********"%itr)
        paths, timesteps_this_batch = agent.sample_trajectory(itr, env)
        total_time_steps += timesteps_this_batch
        
        obs_no = np.concatenate([path['observation'] for path in paths])
        act_na = np.concatenate([path['action'] for path in paths])
        rew_n = [path['reward'] for path in paths]
        
        q_n, adv_n = agent.estimate_return(obs_no, rew_n)
        agent.update_parameters(obs_no, act_na, q_n, adv_n)
        
        # Log diagnostics
        returns = [path['reward'].sum() for path in paths]
        ep_lengths = [len(path['reward']) for path in paths]
        logz.log_tabular('Time', time.time() - tic)
        logz.log_tabular('Iteration', itr)
        logz.log_tabular('AverageReturn', np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenSt", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_time_steps)
        
        logz.dump_tabular()
        logz.pickle_tf_vars()
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=10)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.0)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('-dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    args = parser.parse_args()
    
    max_path_length = args.ep_len if args.ep_len > 0 else None

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime('%d-%m-%Y_%H-%M-%S')
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    
    procs = []
    for i in range(args.n_experiments):
        seed = args.seed + 10*i
        print('Running experiment with seed %d'%seed)
        def train_func():
            trainPG(
                    exp_name=args.exp_name, 
                    env_name=args.env_name,
                    n_iter=args.n_iter,
                    gamma = args.discount,
                    min_timesteps_per_batch=args.batch_size,
                    max_path_length=max_path_length,
                    learning_rate=args.learning_rate,
                    reward_to_go=args.reward_to_go,
                    animate=args.render,
                    logdir=os.path.join(logdir, '%d'%seed),
                    normalize_advantages=not(args.dont_normalize_advantages),
                    nn_baseline=args.nn_baseline,
                    seed=seed,
                    n_layers=args.n_layers,
                    size=args.size
                )
        p = mp.Process(target=train_func, args=tuple())
        p.start()
#         procs.append(p)
#     for p in procs:
        p.join()
        
if __name__ == '__main__':
    main()