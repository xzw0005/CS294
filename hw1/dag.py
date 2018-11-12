import gym
import pickle
import random
import hw1.load_policy as load_policy
import numpy as np
import tensorflow as tf


def fc(x, size, name):
    in_ = x.shape[1]
    with tf.variable_scope(name):
        w = tf.get_variable('w', shape=[in_, size])
        b = tf.get_variable('b', shape=[size])
        result = tf.add(tf.matmul(x, w), b)
        result = tf.nn.leaky_relu(result)
    return result


def load_expert_data(data_path):

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    return data


def gen_policy(num_actions):

    def policy(state):
        with tf.variable_scope('policy'):

            x = fc(state, 128, 'fc_1')
            x = fc(x, 64, 'fc_2')
            x = fc(x, num_actions, 'fc_3')

        return x

    return policy


def main():
    envname = 'Hopper-v2'
    train_epoches = 1000
    num_rollouts = 10
    render = True
    batch_size = 64
    memory = []

    filename = 'expert_data/{}.pkl'.format(envname)
    data = load_expert_data(filename)

    print ("Load Data From File ...")
    provided_state = data['observations']
    provided_action = data['actions']
    provided_data = zip(provided_state, provided_action)

    env = gym.make(envname)
    max_steps = env.spec.timestep_limit

    memory = memory.extend(provided_data)

    num_actions = 3 # env.action_space
    num_state = 11 # env.observation_space

    policy = gen_policy(num_actions)
    
    expert_policy_file = 'experts/{}.pkl'.format(envname)
    expert_policy = load_policy.load_policy(expert_policy_file)

    input_state = tf.placeholder(dtype=tf.float32, shape=[None, num_state])
    label = tf.placeholder(dtype=tf.float32, shape=[None, num_actions])

    output = policy(input_state)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(output-label))

    with tf.name_scope('train_step'):
        train_step = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(loss)

    # train
    def train(sess, dataset, train_epoches):
        num_samples = len(dataset)

        ratio = 0.2
        split_index = int(ratio * num_samples)
        valid_data = dataset[:split_index]
        train_data = dataset[split_index:]
        num_samples = len(train_data)

        for epoch in range(train_epoches):
            random.shuffle(train_data)
            for idx in range(num_samples / batch_size):
                batch_data = train_data[idx * batch_size: (idx + 1) * batch_size]
                batch_state = np.array([x[0] for x in batch_data])
                batch_action = np.array([x[1] for x in batch_data])
                batch_action = np.reshape(batch_action, (batch_size, num_actions))

                _loss, _ = sess.run([loss, train_step], feed_dict={input_state: batch_state, label: batch_action})

        test_state = np.array([x[0] for x in valid_data])
        test_action = np.array([x[1] for x in valid_data])
        test_action = np.reshape(test_action, (len(valid_data), num_actions))

        _loss = sess.run(loss, feed_dict={input_state: test_state, label: test_action})
        print("[Test][Loss: {}]".format(_loss))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in range(train_epoches):
            train(sess, memory, 30)
            returns = []
            observations = []
            actions = []

            for i in range(num_rollouts):
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    observations.append(obs)
                    obs = np.array([obs])
                    action = sess.run(output, feed_dict={input_state: obs})[0]
                    expert_action = expert_policy(obs)
                    actions.append(expert_action)
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if render:
                        env.render()
                    # if steps % 100 == 0:
                    #     print("iter:%i %i/%i" % (i, steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

            print('[Epoch: {}][Mean: {}]'.format(epoch, np.mean(returns)))
            print('[Epoch: {}][Std: {}]'.format(epoch, np.std(returns)))

            print(len(memory))
            memory += zip(observations, actions)
            print(len(memory))


if __name__ == '__main__':

    main()