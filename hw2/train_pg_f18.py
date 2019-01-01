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
# from PgAgent import Agent
import logz

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
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)
    
    
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
     
    def init_tf_sess(self):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__()   # equivalent to "with self.sess:"
        tf.global_variables_initializer().run()
     
    def build_computation_graph(self):
        """
            loss: a function of self.sy_logprob_n and self.sy_adv_n that we will differentiate to get the policy gradient.
        """ 
        self.sy_obs_no, self.sy_act_na, self.sy_adv_n = self.define_placeholders()
        self.policy_parameters = self.policy_forward_pass(self.sy_obs_no)
        self.sy_sampled_act = self.sample_action(self.policy_parameters)
        self.sy_logprob_n = self.get_log_prob(self.policy_parameters, self.sy_act_na)
        loss = tf.reduce_mean(-self.sy_logprob_n * self.sy_adv_n)
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        if self.nn_baseline:
            self.baseline_prediction = tf.squeeze(
                    build_mlp(input_placeholder=self.sy_obs_no, output_size=1, scope="nn_baseline", n_layers=self.n_layers, size=self.size)
                )
            self.sy_target_n = tf.placeholder(dtype=tf.float32, shape=[None], name="baseline_target")
            baseline_loss = tf.nn.l2_loss(self.sy_target_n - self.baseline_prediction)
            self.baseline_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(baseline_loss)
         
    def define_placeholders(self):
        """
        Notes on notation:
            Symbolic variables have the prefix sy_, to distinguish them from the numerical values
            that are computed later in the function
            Prefixes and suffixes:
                obs - observation 
                act - action
                _no - this tensor should have shape (batch self.size /n/, observation dim)
                _na - this tensor should have shape (batch self.size /n/, action dim)
                _n  - this tensor should have shape (batch self.size /n/)
        """
        sy_obs_no = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim], name='obs')
        if self.discrete:
            sy_act_na = tf.placeholder(dtype=tf.int32, shape=[None], name='act')
        else:
            sy_act_na = tf.placeholder(dtype=tf.float32, shape=[None, self.act_dim], name='act')
        sy_adv_n = tf.placeholder(dtype=tf.float32, shape=[None], name='adv')
        return sy_obs_no, sy_act_na, sy_adv_n
     
    def policy_forward_pass(self, sy_obs_no):
        logits = build_mlp(input_placeholder=sy_obs_no, output_size=self.act_dim, \
                           scope='policy_model', n_layers=self.n_layers, size=self.size)
        if self.discrete:       # NN outputs the logits of a categorical distribution
            sy_logits_na = logits
            return sy_logits_na
        else:                   # returns (mean, log_std) tuple of a Gaussian distribution over actions
            sy_mean = logits
            sy_logstd = tf.get_variable('logstd', [self.act_dim], dtype=tf.float32)
            return (sy_mean, sy_logstd)
     
    def sample_action(self, policy_parameters):
        if self.discrete:
            sy_logits_na = policy_parameters
            sy_sampled_act = tf.multinomial(logits=sy_logits_na, num_samples=1)[0]
        else:
            sy_mean, sy_logstd = policy_parameters
            sy_sampled_act = tf.random_normal(shape=sy_mean.shape, mean=sy_mean, stddev=tf.exp(sy_logstd), dtype=tf.float32)
        return sy_sampled_act
     
    def get_log_prob(self, policy_parameters, sy_act_na):
        if self.discrete:
            sy_logits_na = policy_parameters
            sy_logprob_n = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sy_act_na, logits=sy_logits_na)
        else:
            sy_mean, sy_logstd = policy_parameters
            sy_logprob_n = tf.contrib.distributions.MultivariateNormalDia(\
                                loc=sy_mean, scale_diag=tf.exp(sy_logstd)).log_prob(sy_act_na)
        return sy_logprob_n
     
    def sample_trajectory(self, env, animate_this_episode):
        s = env.reset()
        observations, actions, rewards = [], [], []
        steps = 0 
        while True:
            if animate_this_episode:
                env.render()
                time.sleep(0.1)
            a = self.sess.run(self.sy_sampled_act, feed_dict={self.sy_obs_no: s[None]})#s.reshape[1, self.obs_dim]}) 
            a = a[0]
            sp, r, done, _ = env.step(a)
            observations.append(s)
            actions.append(a)
            rewards.append(r)
            steps += 1
            if done or steps>self.max_path_length:
                break
            s = sp
        path = {"observation": np.array(observations, dtype=np.float32),
                "action": np.array(actions, dtype=np.float32),
                "reward": np.array(rewards, dtype=np.float32)}
        return path
     
    def sample_trajectories(self, itr, env):
        timesteps_this_batch = 0
        paths = []
        while True:
            animate_this_episode = (self.animate and len(paths)==0 and itr%10 == 0)
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
        return paths, timesteps_this_batch
     
    def estimate_return(self, obs_no, ret_n):
        Q_n = self.sum_of_rewards(ret_n)
        adv_n = self.compute_advantage(obs_no, Q_n)
        if self.normalize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / np.std(adv_n)
        return Q_n, adv_n
 
    def sum_of_rewards(self, ret_n):
        q_n = []
        for path_rewards in ret_n:
            q_path = []
            q = 0
            for r in reversed(path_rewards):
                q = r + self.gamma * q 
                q_path.append(q)
            if not self.reward_to_go:       # Case 1: Trajectory-based PG
            # Use the total discounted sum of rewards for entire trajectory, i.e. Q_t = Ret(tau)
            # where Ret(tau) = sum_{t'=0}^T gamma^t' r(t')
            # Estimated gradient is E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]
                q_path.reverse()
            else:                            # Case 2: Reward-to-go PG
            # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
                q_path = [q for _ in range(len(path_rewards))]
            q_n.extend(q_path)
        return q_n
     
    def compute_advantage(self, obs_no, q_n):
        if self.nn_baseline:
            b_n = self.sess.run(self.baseline_prediction, feed_dict={self.sy_obs_no:obs_no})
            b_n = b_n * np.std(q_n) + np.mean(q_n)
            adv_n = q_n - b_n 
        else:
            adv_n = q_n.copy()
        return adv_n
     
    def update_parameters(self, obs_no, act_na, q_n, adv_n):
        if self.nn_baseline:
            target_n = self.compute_advantage(obs_no, q_n)
            self.sess.run([self.baseline_update_op], feed_dict={self.sy_target_n:target_n, self.sy_obs_no:obs_no})
        feed_dict = {self.sy_obs_no: obs_no, self.sy_act_na: act_na, self.sy_adv_n: adv_n}
        self.sess.run([self.update_op], feed_dict=feed_dict)
     
    
def train_PG(exp_name, env_name, n_iter, \
             gamma, min_timesteps_per_batch, max_path_length, learning_rate, \
             reward_to_go, animate, logdir, normalize_advantages, nn_baseline, \
             seed, n_layers, size):
    start = time.time()
    setup_logger(logdir, locals())  ## Set up Logger
    
    env = gym.make(env_name)
    tf.set_random_seed(seed)
    env.seed(seed)
    
    max_path_length = max_path_length or env.spec.max_episode_steps
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n if discrete else env.action_space.shape[0]
    
    ## Initialize Agent
    computation_graph_args = {'n_layers': n_layers, 'obs_dim': obs_dim, 'act_dim': act_dim, \
                              'discrete': discrete, 'size': size, 'learning_rate': learning_rate}
    sample_trajectory_args = {'animate': animate, 'max_path_length': max_path_length, \
                              'min_timesteps_per_batch': min_timesteps_per_batch}
    estimate_return_args = {'gamma': gamma, 'reward_to_go': reward_to_go, \
                            'nn_baseline': nn_baseline, 'normalize_advantages': normalize_advantages}
    agent = Agent(computation_graph_args, sample_trajectory_args, estimate_return_args)
    agent.build_computation_graph()
    agent.init_tf_sess()
    
    ## Training Loop
    total_time_steps = 0 
    for itr in range(n_iter):
        print("********* Iteration %i *********"%itr)
        paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
        total_time_steps += timesteps_this_batch
        
        obs_no = np.concatenate([path['observation'] for path in paths])
        act_na = np.concatenate([path['action'] for path in paths])
        ret_n = [path['reward'] for path in paths]
        
        q_n, adv_n = agent.estimate_return(obs_no, ret_n)
        agent.update_parameters(obs_no, act_na, q_n, adv_n)
        
        # Log dianostics
        returns = [path['reward'].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time()-start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    args = parser.parse_args()
    
    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir) 
    max_path_length = args.ep_len if args.ep_len > 0 else None
    
    processes = []
    
    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)
        
        def train_func():
            train_PG(exp_name=args.exp_name, env_name=args.env_name, n_iter=args.n_iter, \
                     gamma=args.discount, min_timesteps_per_batch=args.batch_size, \
                     max_path_length=max_path_length, learning_rate=args.learning_rate, \
                     reward_to_go=args.reward_to_go, animate=args.render, \
                     logdir=os.path.join(logdir, '%d'%seed), \
                     normalize_advantages=not(args.dont_normalize_advantages), \
                     nn_baseline=args.nn_baseline, seed=seed, n_layers=args.n_layers, size=args.size)
        p = Process(target=train_func, args=tuple())
        p.start()
        processes.append(p)
        # if you comment in the line below, then the loop will block 
        # until this process finishes
        # p.join()

    for p in processes:
        p.join()

if __name__=='__main__':
    main()