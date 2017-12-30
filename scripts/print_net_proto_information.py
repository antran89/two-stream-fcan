#!/usr/bin/env python
"""
Print net prototext information of a Caffemodel.
Usage: python print_net_proto_information.py --model_prototxt=filename
"""

caffe_root = '/home/tranlaman/Desktop/caffe-workspace/my-very-deep-caffe/'
import sys
import argparse

sys.path.insert(0, caffe_root + 'cmake-build/install/python')
import caffe

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Print net prototext information of a Caffemodel.')
    parser.add_argument('--model_prototxt', dest='model_prototxt', help='Caffe model prototxt', required=True,
                        type=str)

    args = parser.parse_args()

    return args

def main():
    caffe.set_mode_cpu()
    
    args = parse_args()
    model_prototxt = args.model_prototxt
    
    # initialize the net
    net = caffe.Net(model_prototxt, caffe.TEST)
    
    #net.blobs['data'].reshape(50, 20, 224, 224)
    print 'Net blobs shape:'
    for k, v in net.blobs.items():
        print (k, v.data.shape)
    
    print 'Net params shape:'
    num_params = 0
    for k, v in net.params.items():
        print (k, v[0].data.shape)
        print (k, v[1].data.shape)
        num_params += v[0].data.size
        num_params += v[1].data.size
        
    print 'Total number of params in the network %d' % num_params

if __name__ == '__main__':
    main()