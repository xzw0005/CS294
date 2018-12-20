import pickle, tensorflow as tf, numpy as np
import tf_util

def load_policy(filename):
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())
        # data should be a dict with 2 keys: 'GaussianPolicy' and 'nonlin_type'
#         print(data)
        
    nonlin_type = data['nonlin_type']
    policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]
    assert policy_type == 'GaussianPolicy', 'Policy type {} not supported'.format(policy_type)
    policy_params = data[policy_type]
#     print(policy_params.keys())
#     print(policy_params['obsnorm'])
    assert set(policy_params.keys()) == {'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'}
    
    obs_bo = tf.placeholder(tf.float32, [None, None])
    a_ba = build_policy(obs_bo, policy_params, nonlin_type)
    policy_fn = tf_util.function([obs_bo], a_ba) 
    return policy_fn
    
def build_policy(obs_bo, policy_params, nonlin_type):
    # first, observation normalization
    assert list(policy_params['obsnorm'].keys()) == ['Standardizer']
    obsnorm_mean = policy_params['obsnorm']['Standardizer']['mean_1_D']
    obsnorm_meansq = policy_params['obsnorm']['Standardizer']['meansq_1_D']
    obsnorm_stdev = np.sqrt(np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean)))
    print('obs', obsnorm_mean.shape, obsnorm_stdev.shape)
    normed_obs_bo = (obs_bo - obsnorm_mean) / (obsnorm_stdev + 1e-6)
    
    curr_activation_bd = normed_obs_bo
    
    # Hidden layer
    assert list(policy_params['hidden'].keys()) == ['FeedforwardNet']
    layer_params = policy_params['hidden']['FeedforwardNet']
#     print(layer_params.keys())
    for layer_name in sorted(layer_params.keys()):
        L = layer_params[layer_name]
        W, b = read_layer(L)
        curr_activation_bd = apply_nonlin(tf.matmul(curr_activation_bd, W) + b, nonlin_type)
        
    # Output layer
    W, b = read_layer(policy_params['out'])
    output_bo = tf.matmul(curr_activation_bd, W) + b
    return output_bo
    
def read_layer(L):
    assert list(L.keys()) == ['AffineLayer']
    assert sorted(L['AffineLayer'].keys()) == ['W', 'b']
    return L['AffineLayer']['W'].astype(np.float32), L['AffineLayer']['b'].astype(np.float32)

def apply_nonlin(x, nonlin_type):
    if nonlin_type == 'lrelu':
        return tf_util.lrelu(x, leak=.01)
    elif nonlin_type == 'tanh':
        return tf.tanh(x)
    else:
        return NotImplementedError(nonlin_type)

    
if __name__ == "__main__":                                                                                                               
    filename = 'experts/HalfCheetah-v2.pkl'
    load_policy(filename)