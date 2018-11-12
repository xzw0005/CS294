'''
Created on Oct 25, 2018

@author: wangxing
'''
import pickle 
import tensorflow as tf
import numpy as np
import gym 

train_epochs = 10
batch_size = 256
learning_rate=.001

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())
#     print(data.keys()) 
#     'observations': np.array(observations), 'actions':
    observations = data['observations']
    actions = data['actions']
    assert len(observations) == len(actions), '{} data not coincided'
    return observations, actions 

# def fc_layer(x, name, layer_size=100, activation=None):
#     in_size = x.shape[1]
#     with tf.variable_scope(name):
#         W = tf.get_variable('W', shape=[in_size, layer_size], initializer=tf.contrib.layers.xavier_initializer())
#         b = tf.get_variable('b', shape=[layer_size], initializer=tf.contrib.layers.xavier_initializer())
#         layer = tf.matmul(x, W) + b 
#         if activation is not None:
#             layer = activation(layer)
#     return layer 
# 
# def policy_function(obs, num_actions):
#     with tf.variable_scope('model'):
#         x = fc_layer(obs, 'layer0', layer_size=128, activation=tf.nn.relu)
#         x = fc_layer(x, 'layer1', layer_size=64, activation=tf.nn.relu)
#         x = fc_layer(x, 'layer2', layer_size=num_actions)
#     return x

def policy_function(obs, num_actions):
    with tf.variable_scope('layer0'):
        x = tf.contrib.layers.fully_connected(obs, num_outputs=128, activation_fn=tf.nn.relu)
    with tf.variable_scope('layer1'):
        x = tf.contrib.layers.fully_connected(x, num_outputs=64, activation_fn=tf.nn.relu)
    with tf.variable_scope('layer2'):
        x = tf.contrib.layers.fully_connected(x, num_outputs=num_actions, activation_fn=None)
    return x
        
def main():
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--max_timesteps', type=int)
    parser.add_argument('--num_episodes', type=int, default=20, help='Number of test episodes')
    args = parser.parse_args()
    
    envname = args.envname
    render = args.render 
    max_timesteps = args.max_timesteps 
    num_episodes = args.num_episodes
    
    filename = 'expert_data/{}.pkl'.format(envname)
    observations, actions = load_data(filename)
    print('data loaded...')
    
    obs_size = observations.shape[-1]
    num_samples = actions.shape[0]
    action_size = actions.shape[-1]
    actions = actions.reshape([num_samples, action_size])
    
    input_ph = tf.placeholder(dtype=tf.float32, shape=[None, obs_size])
    output_ph = tf.placeholder(dtype=tf.float32, shape=[None, action_size])
    output_pred = policy_function(input_ph, action_size)
    
    loss = tf.reduce_mean(tf.square(output_pred - output_ph))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#     sess = tf.Session()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        num_samples = observations.shape[0]
        indices = list(range(num_samples))
        for epoch in range(1, train_epochs+1):
            np.random.shuffle(indices)
            for i in range(int(num_samples/batch_size)+1):
                batch_idx = indices[i*batch_size : (i+1)*batch_size]
#                 indices = np.random.randint(0, num_samples, size=batch_size)
                X_batch = observations[batch_idx]
                Y_batch = actions[batch_idx]
                _, loss_run = sess.run([optimizer, loss], feed_dict={input_ph : X_batch, output_ph : Y_batch})
            print('{0:04d} mse: {1:.3f}'.format(epoch, loss_run))
        print('model trained')
        
        print('behavior cloning model trained...')
    #     print(tf.global_variables())
    
        env = gym.make(envname)
        obs_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        max_steps = max_timesteps or env.spec.timestep_limit 
        
        returns = []
        for i in range(num_episodes):
            obs = env.reset()
            done = False
            total_reward = 0.
            steps = 0
            while not done:
                obs =np.array([obs])
                a = sess.run(output_pred, feed_dict={input_ph: obs})[0]
                obs, r, done, _ = env.step(a)
                total_reward += r
                steps += 1 
                if render:
                    env.render()
                if steps >= max_steps:
                    break
            print('iter {}, return = {}'.format(i, total_reward))
            returns.append(total_reward)
         
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))    

if __name__=='__main__':
    main()