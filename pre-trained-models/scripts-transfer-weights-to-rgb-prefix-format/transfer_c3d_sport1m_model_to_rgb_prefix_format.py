#!/usr/bin/python
'''
Create Caffenet weights for newer format with rgb_ name in each layer.
It is for easier to train with other two-stream CNN.
'''

import numpy as np
import sys
import os

caffe_root = '/home/tranlaman/Desktop/caffe-workspace/my-very-deep-caffe/'
sys.path.insert(0, caffe_root + 'cmake-build-c3d/install/python')
import caffe

caffe.set_mode_cpu()

# Spatial-CNN models and weights
base_net_model = '../c3d_rgb_sport1m_nd_conv_deploy.prototxt'
base_net_weights = '../conv3d_deepnetA_sport1m_iter_1900000_newcaffe_format.caffemodel'
base_net = caffe.Net(base_net_model, base_net_weights, caffe.TEST)

# load transfered net
transfer_model = '../c3d_rgb_sport1m_nd_conv_rgb_prefix_deploy.prototxt'
transfer_net = caffe.Net(transfer_model, caffe.TEST)
print 'Transfer net parameters:'
for k, v in transfer_net.params.items():
    print (k, v[0].data.shape)

base_params_names = []
for k, v in base_net.params.items():
    base_params_names.append(k)
assert len(base_params_names) == 11

# copying params from spatial net to cross-and-stitch net
# Python just assigns pointers
base_params = base_net.params
transfer_params = transfer_net.params
for name in base_params_names:
    if name == 'fc8-1':
        transfer_name = 'rgb_fc8_org'
    else:
        transfer_name = 'rgb_' + name
    assert(transfer_params[transfer_name][0].data.size == base_params[name][0].data.size)
    assert(transfer_params[transfer_name][1].data.size == base_params[name][1].data.size)    
    transfer_params[transfer_name][0].data.flat = base_params[name][0].data.flat
    transfer_params[transfer_name][1].data.flat = base_params[name][1].data.flat        

# testing conv1_2 layer weights
layer_name = 'conv2a'
transfer_name = 'rgb_' + layer_name
base_layer_bias = base_net.params[layer_name][1].data
transfer_layer_bias = transfer_net.params[transfer_name][1].data
bias_diff = base_layer_bias - transfer_layer_bias
print 'norm of rgb bias difference %f\n' % np.linalg.norm(bias_diff)

base_weight = base_net.params[layer_name][0].data
transfer_weight = transfer_net.params[transfer_name][0].data
weight_diff = transfer_weight - base_weight
print 'norm of rgb weight difference %f\n' % np.linalg.norm(weight_diff)

# save the modified net
print 'Saving the network!'
file_name = os.path.splitext(base_net_weights)
new_file_name = file_name[0] + '_rgb_prefix' + file_name[1]
transfer_net.save(new_file_name)
