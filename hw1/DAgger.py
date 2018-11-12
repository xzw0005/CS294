'''
Created on Oct 26, 2018

@author: wangxing
'''
import pickle
import tensorflow as tf
import numpy as np
import gym

import hw1.load_policy

envname = 'HalfCheetah-v2'
render = True
max_timesteps = 1000
num_episodes = 20
hidden_sizes = [128, 64]
train_epochs = 2
batch_size = 256
learning_rate = .001


def load_expert_data(envname):
    filename = 'expert_data/{}.pkl'.format(envname)
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())
    observations = data['observations']
    actions = data['actions']
    return observations, actions 

def supervised_model(obs, action_size):
    with tf.variable_scope('model'):
        for i in range(len(hidden_sizes)+1):
            layer = "layer{}".format(i)
            with tf.variable_scope(layer):
                if i == 0:
                    x = obs
                if i == len(hidden_sizes):
                    num_outputs = action_size
                    activation = None
                else:
                    num_outputs = hidden_sizes[i]
                    activation = tf.nn.relu
                x = tf.contrib.layers.fully_connected(x, num_outputs=num_outputs, activation_fn=activation)
    return x
    
def main():
    observations, actions = load_expert_data(envname)
    print(observations.shape, actions.shape)
    obs_size = observations.shape[-1]
    num_samples = actions.shape[0]
    action_size = actions.shape[-1]
    actions = actions.reshape([num_samples, action_size])
    print('expert data loaded')
    
    expert_policy_file = 'experts/{}.pkl'.format(envname)
    expert_policy_fn = hw1.load_policy.load_policy(expert_policy_file)
    print('expert policy loaded and built')
        
    in_ph = tf.placeholder(dtype=tf.float32, shape=[None, obs_size])
    out_ph = tf.placeholder(dtype=tf.float32, shape=[None, action_size])
    pred = supervised_model(in_ph, action_size)
    
    loss = tf.reduce_mean(tf.square(pred - out_ph))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('supervised model initialized')
        print(tf.global_variables())

        num_samples = observations.shape[0]
        indices = list(range(num_samples))
        for epoch in range(1, train_epochs+1):
            np.random.shuffle(indices)
            for i in range(num_samples / batch_size):
                batch_idx = indices[i*batch_size : (i+1)*batch_size]
#                 indices = np.random.randint(0, num_samples, size=batch_size)
                X_batch = observations[batch_idx]
                Y_batch = actions[batch_idx]
                _, loss_run = sess.run([optimizer, loss], feed_dict={in_ph : X_batch, out_ph : Y_batch})
            print('{0:04d} mse: {1:.3f}'.format(epoch, loss_run))
        print('model trained')
            
        env = gym.make(envname)
        for ep in range(20):
            obs = env.reset()
            done = False
            steps = 0
#             new_observations = []
            while not done:
                obs = np.array([obs])
                act = sess.run(pred, feed_dict={in_ph: obs})[0]
                human_label = expert_policy_fn(obs)
                np.vstack([observations, obs])
                np.vstack([actions, human_label])
#                 new_observations.append(obs)
                

                obs, r, done, _ = env.step(act)
                steps += 1
                if steps >= max_timesteps:
                    break 
            
#             human_labels = expert_policy_fn(new_observations)
#             observations.extend(new_observations)
#             actions.extend(human_labels)
            
            for epoch in range(1, train_epochs+1):
                num_samples = observations.shape[0]
                indices = np.random.randint(0, num_samples, size=batch_size)
                X_batch = observations[indices]
                Y_batch = actions[indices]
                _, loss_run = sess.run([optimizer, loss], feed_dict={in_ph : X_batch, out_ph : Y_batch})
                if epoch % 1000 == 0:
                    print('{0:04d} mse: {1:.3f}'.format(epoch, loss_run)) 
            print('model %d-th time re-trained'%(ep+1))
    
                
        returns = []
        for ep in range(num_episodes):
            print('iter', ep)
            obs = env.reset()
            done = False
            total_reward = 0.
            steps = 0
            while not done:
                obs =np.array([obs])
                act = sess.run(pred, feed_dict={in_ph: obs})[0]
                obs, r, done, _ = env.step(act)
                total_reward += r
                steps += 1 
                if render:
                    env.render()
                if steps >= max_timesteps:
                    break
            returns.append(total_reward)
             
        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))    

if __name__=='__main__':
    main()