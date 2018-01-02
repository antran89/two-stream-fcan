'''
Transfer weights from Du Tran's old format to new format. Python does not allow 
import two different packages with the same name.
http://stackoverflow.com/questions/5062793/is-it-possible-to-use-two-python-packages-with-the-same-name
Solution: save weights into pickle file (in DataFrames). Then load it back latter.
'''

import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd

caffe_root = '/home/tranlaman/Desktop/caffe-workspace/New-C3D-Caffe/'

sys.path.insert(0, caffe_root + 'cmake-build/install/python')
import caffe as caffe_old
print 'Caffe old version: ' + caffe_old.__version__

caffe_old.set_mode_cpu()

# Temporal-CNN models and weights
net_model = 'c3d_rgb_fb_old_format_deploy.prototxt'
net_weights = '../conv3d_deepnetA_sport1m_iter_1900000'
net = caffe_old.Net(net_model, net_weights, caffe_old.TEST)

# flow-net information
net_params_names = []
for k, v in net.params.items():
    print (k, v[0].data.shape)
    print (k, v[1].data.shape)
    net_params_names.append(k)

# copying params from temporal net to cross-and-stitch net
# Python just assigns pointers
layer_name = 'conv2a'
con1a_layer = net.params[layer_name]
print layer_name + ' layer weights:'
print con1a_layer[0].data
print layer_name + ' layer bias'
print con1a_layer[1].data

# save net parameters into DataFrame
df = pd.DataFrame(index=net_params_names, columns=['weight', 'bias'])
for layer_name in net_params_names:
    if layer_name in ['fc6-1', 'fc7-1']:
        transfer_name = layer_name[:-2]
    else:
        transfer_name = layer_name
    layer_weights = net.params[layer_name][0].data
    layer_bias = net.params[layer_name][1].data
    df.at[transfer_name, 'weight'] = layer_weights         # at -- fast scalar accessor
    df.at[transfer_name, 'bias'] = layer_bias

# saving df
df.to_pickle('c3d.pickle')
