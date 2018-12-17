'''
Created on Dec 16, 2018

@author: wangxing
'''
import numpy as np 
import tensorflow as tf 
# from train_pg_f18 import *

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
        tf.global_variables_initializer()
    
    def build_computation_graph(self):
        """
            loss: a function of self.sy_logprob_n and self.sy_adv_n that we will differentiate to get the policy gradient.
        """ 
        self.sy_obs_no, self.sy_act_na, self.sy_adv_n = self.define_placeholders()
        self.policy_parameters = self.policy_forward_pass(self.sy_obs_no)
        self.sy_sampled_act = self.sample_action(self.policy_parameters)
        self.sy_logprob_n = self.get_log_probs(self.policy_parameters, self.sy_act_na)
        loss = tf.reduce_mean(-self.sy_logprob_n * self.sy_adv_n)
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        
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
            a = self.sess.run(self.sy_sampled_act, feed_dict={self.sy_obs_no:s.reshape[1, self.obs_dim]})
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
            raise NotImplementedError
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
            raise NotImplementedError
            b_n = None
            adv_n = q_n - b_n 
        else:
            adv_n = q_n.copy()
        return adv_n
    
    def update_parameters(self, obs_no, act_na, q_n, adv_n):
        if self.nn_baseline:
            raise NotImplementedError
        feed_dict = {self.sy_obs_no: obs_no, self.sy_act_na: act_na, self.sy_adv_n: adv_n}
        self.sess.run([self.update_op], feed_dict=feed_dict)
    
    