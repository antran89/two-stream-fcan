'''
Load C3D weights old format from Du Tran into intermediate pickle files.
'''

import sys
import pandas as pd

# import new Caffe
caffe_root = '/home/tranlaman/Desktop/caffe-workspace/my-very-deep-caffe/'
sys.path.insert(0, caffe_root + 'cmake-build/install/python')
import caffe as caffe_new
print 'Caffe new version: ' + caffe_new.__version__

caffe_new.set_mode_cpu()

net_model = 'c3d_rgb_train_val_lmdb_nd_conv_deploy.prototxt'
new_net = caffe_new.Net(net_model, caffe_new.TEST)

# flow-net information
net_params_names = []
for k, v in new_net.params.items():
    print (k, v[0].data.shape)
    print (k, v[1].data.shape)
    net_params_names.append(k)
    
# load data frame back
df = pd.read_pickle('c3d.pickle')

# load network parameters into new Caffemodels
for layer_name in net_params_names:
    assert(new_net.params[layer_name][0].data.size == df.at[layer_name, 'weight'].size)
    assert(new_net.params[layer_name][1].data.size == df.at[layer_name, 'bias'].size)
    new_net.params[layer_name][0].data.flat = df.at[layer_name, 'weight'].flat
    new_net.params[layer_name][1].data.flat = df.at[layer_name, 'bias'].flat
    
# copying params from temporal net to cross-and-stitch net
# Python just assigns pointers
layer_name = 'conv2a'
con1a_layer = new_net.params[layer_name]
print layer_name + ' layer weights:'
print con1a_layer[0].data
print layer_name + ' layer bias'
print con1a_layer[1].data

# save the modified net
print 'Saving the network!'
new_net.save('../conv3d_deepnetA_sport1m_iter_1900000_newcaffe_format.caffemodel')