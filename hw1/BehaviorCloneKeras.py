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

envname = 'HalfCheetah-v2'
render = True
max_timesteps = 1000
num_episodes = 20
hidden_sizes = [128, 64]
train_epochs = 100
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
    
#     from tf.keras.layers import Dense, Activation
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, input_shape=(obs_size, ), activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(action_size))
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(observations, actions, epochs=10, batch_size=256)
        
    env = gym.make(envname)
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
        print('iter {}, return = {}'.format(ep, total_reward))
        returns.append(total_reward)
         
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))    

if __name__=='__main__':
    main()