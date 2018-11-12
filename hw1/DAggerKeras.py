'''
Created on Oct 26, 2018

@author: wangxing
'''
'''
Created on Oct 26, 2018

@author: wangxing
'''
import pickle
import tensorflow as tf
import numpy as np
import gym

import hw1.load_policy

envname = 'Ant-v2'
render = True
max_timesteps = 1000
num_episodes = 20
hidden_sizes = [128, 64]
batch_size = 256
learning_rate = .001


def load_expert_data(envname):
    filename = 'expert_data/{}.pkl'.format(envname)
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())
    observations = data['observations']
    actions = data['actions']
    return observations, actions 
    
def main():
    observations, actions = load_expert_data(envname)
    num_samples, obs_size = observations.shape
    action_size = actions.shape[-1]
#     print(actions.shape)
    actions = actions.reshape([num_samples, action_size])
    print('expert data loaded')
    
    expert_policy_file = 'experts/{}.pkl'.format(envname)
    expert_policy_fn = hw1.load_policy.load_policy(expert_policy_file)
    print('expert policy loaded and built')
        
#     from tf.keras.layers import Dense, Activation
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(100, input_shape=(obs_size, ), activation='relu'))
    model.add(tf.keras.layers.Dense(60, activation='relu'))
    model.add(tf.keras.layers.Dense(action_size))
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(observations, actions, epochs=1, batch_size=32, verbose=0)
    print('model trained')

    env = gym.make(envname)
    for ep in range(20):
        obs = env.reset()
        done = False
        steps = 0
        new_observations = []
        while not done:
            act = model.predict(np.array([obs]))[0]
            new_observations.append(obs)
            obs, r, done, _ = env.step(act)
            steps += 1
            if steps >= max_timesteps:
                break 
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            human_labels = expert_policy_fn(new_observations)
            observations = np.concatenate((observations, np.array(new_observations)), axis=0)
            actions = np.concatenate((actions, human_labels), axis=0)
        model.fit(observations, actions, epochs=1, batch_size=32, verbose=0)
        print('model %d-th time re-trained'%(ep+1))

    returns = []
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.
        steps = 0
        while not done:
            obs =np.array([obs])
            act = model.predict(obs)[0]
            obs, r, done, _ = env.step(act)
            total_reward += r
            steps += 1 
            if render:
                env.render()
            if steps >= max_timesteps:
                break
        print('iter {}, total reward = {}'.format(ep, total_reward))
        returns.append(total_reward)
         
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))    

if __name__=='__main__':
    main()