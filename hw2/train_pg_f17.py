'''
Created on Jan 1, 2019

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

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(trainPG)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

def build_mlp(input_ph, output_size, scope, n_layers=2, size=64, activation=tf.tanh, output_activation=None):
    x = input_ph 
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            x = tf.layers.dense(x, size, activation=activation)
        x = tf.layers.dense(x, output_size, activation=output_activation)
    return x

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
    
    ## Define Placeholders
    obs_ph = tf.placeholder(shape=[None, obs_dim], dtype=tf.float32, name='obs')
    if discrete:
        act_ph = tf.placeholder(shape=[None], dtype=tf.int32, name='act')
    else:
        act_ph = tf.placeholder(shape=[None, act_dim], dtype=tf.float32, name='act')
    adv_ph = tf.placeholder(shape=[None], dtype=tf.float32, name='adv')
    
    ## Build computation graph, define forward pass
    nn_out = build_mlp(input_ph=obs_ph, output_size=act_dim, scope='policy_model', n_layers=n_layers, size=size)
    if discrete:
        logits_ph = nn_out
        sampled_action_ph = tf.multinomial(logits=logits_ph, num_samples=1)[0]
        logprob_ph = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=act_ph, logits=logits_ph)
    else:
        mu_ph = nn_out
        logstd_ph = tf.get_variable('logstd', [act_dim], dtype=tf.float32)
        sampled_action_ph = tf.random_normal(shape=mu_ph.shape, mean=mu_ph, stddev=tf.exp(logstd_ph))
        logprob_ph = -tf.contrib.distributions.MultivariateNormalDiag(loc=mu_ph, scale_diag=tf.exp(logstd_ph))
    
    ## Define Loss Function and Training Operation
    loss = tf.reduce_mean(-logprob_ph * adv_ph)
    update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    if nn_baseline:
        baseline_pred_ph = tf.squeeze(
                build_mlp(input_ph=obs_ph, output_size=1, scope='nn_baseline', n_layers=n_layers, size=size)
            )
        baseline_target_ph = tf.placeholder(shape=[None], dtype=tf.float32, name='baseline')
        baseline_loss = tf.nn.l2_loss(baseline_pred_ph - baseline_target_ph)
        baseline_update_op = tf.train.AdamOptimizer(learning_rate).minimize(baseline_loss)
        
    ## Initialize Tensorflow Configs
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 
    sess = tf.Session(config=tf_config)
    sess.__enter__()           # equivalent to "with self.sess:"
    tf.global_variables_initializer().run()

    ## Training Loop
    total_time_steps = 0
    for itr in range(n_iter):
        print("********* Iteration %i *********"%itr)
        ### Sample_Trajectories
        timesteps_this_batch = 0
        paths = []
        while True:
            #### Sample a trajectory
            observations, actions, rewards = [], [], []
            animate_this_episode = (animate and len(paths)==0 and itr%10==0)
            s = env.reset()
            steps = 0
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.1)
                a = sess.run(sampled_action_ph, feed_dict={obs_ph: s[None]})
                a = a[0]
                sp, r, done, _ = env.step(a)
                observations.append(s)
                actions.append(a)
                rewards.append(r)
                steps += 1
                if done or steps > max_path_length:
                    break
                s = sp
            #### End of Sample a trajectory
            path = {'observation': np.array(observations, dtype=np.float32),
                    'action': np.array(actions, dtype=np.int32 if discrete else np.float32),
                    'reward': np.array(rewards, dtype=np.float32) }
            paths.append(path)
            timesteps_this_batch += steps
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_time_steps += timesteps_this_batch
        
        ## Build arrays for observation, action for the policy gradient update by concatenating across paths
        obs = np.concatenate([path['observation'] for path in paths])
        act = np.concatenate([path['action'] for path in paths])
        rew = [path['reward'] for path in paths]
        
        ## Estimate Return
        ### Compute Q-values
        qvals = []
        for path_rewards in rew:
            q_path = []
            q = 0
            for r in reversed(path_rewards):
                q = r + gamma * q
                q_path.append(q)
            if reward_to_go:
                q_path.reverse()
            else:
                q_path = [q for _ in range(len(path_rewards))]
            qvals.extend(q_path)
        ### Compute Advantages
        if nn_baseline:
            bl = sess.run(baseline_pred_ph, feed_dict={obs_ph: obs})
            bl = bl * np.std(qvals) + np.mean(qvals)
            adv = qvals - bl
            #### TODO: GAE implementation
        else:
            adv = qvals.copy()
        if normalize_advantages:
            adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
            
        ## Policy Network Parameters Update
        if nn_baseline:
            sess.run([baseline_update_op], feed_dict={baseline_target_ph: adv, obs_ph: obs})
        _, loss_policy = sess.run([update_op, loss], \
                    feed_dict={obs_ph: obs, act_ph: act, adv_ph: adv})
        
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
        logz.log_tabular("PolicyLoss", loss_policy)
        
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
                    min_steps_per_batch=args.batch_size,
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