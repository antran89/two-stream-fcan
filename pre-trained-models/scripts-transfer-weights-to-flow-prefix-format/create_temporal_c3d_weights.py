#!/usr/bin/env python
'''
Transfering weights from (3*L channels) rgb C3D weights into (2*L channels) flow C3D weights.
By default, L is 16 frames. But following LTC paper, we can investigate longer
temporal length if possible.
'''

import numpy as np

caffe_root = '/home/tranlaman/Desktop/caffe-workspace/my-very-deep-caffe/'
import sys

sys.path.insert(0, caffe_root + 'cmake-build-c3d/install/python')
import caffe

caffe.set_mode_cpu()

# Motion-CNN models and weights
base_net_model = '../c3d_rgb_train_val_lmdb_nd_conv_deploy.prototxt'
base_net_weights = '../conv3d_deepnetA_sport1m_iter_1900000_newcaffe_format.caffemodel'
base_net = caffe.Net(base_net_model, base_net_weights, caffe.TEST)

# load foreground flow net
transfer_model = '../c3d_flow_sport1m_nd_conv_deploy.prototxt'
transfer_net = caffe.Net(transfer_model, caffe.TEST)

print 'parameters of transfer net'
for k, v in transfer_net.params.items():
    print (k, v[0].data.shape)

params_names = []
for k, v in base_net.params.items():
    params_names.append(k)
params_names.remove('conv1a')
assert len(params_names) == 10

# copying params from flow net to fg net
# Python just assigns pointers
base_params = base_net.params
transfer_params = transfer_net.params
for name in params_names:
    transfer_name = 'flow_' + name
    transfer_params[transfer_name][0].data.flat = base_params[name][0].data.flat
    transfer_params[transfer_name][1].data.flat = base_params[name][1].data.flat

# replicating params for first layer
# biases
layer_name = 'conv1a'
transfer_name = 'flow_' + layer_name
first_layer_bias = base_params[layer_name][1].data
transfer_params[transfer_name][1].data.flat = base_params[layer_name][1].data.flat
# weights
numChannels = 2
first_layer_weights = base_params[layer_name][0].data
single_channel_weights = np.mean(first_layer_weights, axis=1)
single_channel_weights = np.expand_dims(single_channel_weights, axis=1)
new_weights = np.tile(single_channel_weights, [1, numChannels, 1, 1, 1])
transfer_params[transfer_name][0].data.flat = new_weights.flat

# testing conv3a layer weights
layer_name = 'conv3a'
transfer_name = 'flow_' + layer_name
layer_bias = base_net.params[layer_name][1].data
transfer_layer_bias = transfer_net.params[transfer_name][1].data
diff = transfer_layer_bias - layer_bias
print 'bias norm of difference %f\n' % np.linalg.norm(diff)
layer_weights = base_net.params[layer_name][0].data
transfer_layer_weights = transfer_net.params[transfer_name][0].data
diff = transfer_layer_weights - layer_weights
print 'weights norm of difference %f\n' % np.linalg.norm(diff)

# print conv1a params of both networks
layer_name = 'conv1a'
conv1a_layer = base_net.params[layer_name]
print 'Parameters of base net'
print layer_name + ' layer weights:'
print conv1a_layer[0].data
print layer_name + ' layer bias'
print conv1a_layer[1].data

transfer_name = 'flow_' + layer_name
conv1a_layer = transfer_net.params[transfer_name]
print 'Parameters of transferred net'
print transfer_name + ' layer weights:'
print conv1a_layer[0].data
print transfer_name + ' layer bias'
print conv1a_layer[1].data

# save the modified net
print 'Saving the network!'
transfer_net.save('../c3d_flow_sport1m_newcaffe_format.caffemodel')
