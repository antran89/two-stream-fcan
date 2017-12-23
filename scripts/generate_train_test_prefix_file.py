# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 10:47:04 2016

@author: tranlaman

Create train/test prefix file for extracting features from a deep net.
"""

import os
import numpy as np
import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Create train/test prefix file for extracting features from a deep net.')
    parser.add_argument('--output_folder', dest='output_folder', help='Output feature folder', default='out_folder',
                        type=str)
    parser.add_argument('--train_key_file', dest='train_key_file', help='train keys file', default=None, type=str)
    parser.add_argument('--test_key_file', dest='test_key_file', help='test keys file', default=None, type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='batch size', default=50, type=int)
    
    parser.add_argument('--train_list_file', dest='train_list_file', help='train list prefix file', 
                        default='feature_extraction_train_list_1view_prefix.txt', type=str)
    parser.add_argument('--test_list_file', dest='test_list_file', help='test list prefix file', 
                        default='feature_extraction_test_list_1view_prefix.txt', type=str)

    args = parser.parse_args()

    return args
    
def main():
    args = parse_args()
    outFolder = args.output_folder
    trainKeyFile = args.train_key_file
    testKeyFile = args.test_key_file
    batch_size = args.batch_size
    train_list_file = args.train_list_file
    test_list_file = args.test_list_file
    
    if not os.path.isfile(trainKeyFile) or not os.path.isfile(testKeyFile):
        print 'Error: please provide both train and test keys file. '
        exit

    trainFid = open(train_list_file, 'w')
    trainKeys = np.loadtxt(trainKeyFile, dtype=str, delimiter=' ', comments=None)
    
    for key in trainKeys:
        outPath = os.path.join(outFolder, key[0])
        trainFid.write('%s\n' % os.path.join(outPath, key[1]))
        
    trainFid.close
    
    numSamples = len(trainKeys)
    print 'number of segment %d' % numSamples
    print 'with batch size: %d, we need to extract number of train batches: %d' \
    % (batch_size, numSamples/batch_size + 1)
    
    testFid = open(test_list_file, 'w')
    testKeys = np.loadtxt(testKeyFile, dtype=str, delimiter=' ', comments=None)
    
    for key in testKeys:
        outPath = os.path.join(outFolder, key[0])
        testFid.write('%s\n' % os.path.join(outPath, key[1]))
        
    testFid.close
    
    numSamples = len(testKeys)
    print 'number of segment %d' % numSamples
    print 'with batch size: %d, we need to extract number of test batches: %d' \
    % (batch_size, numSamples/batch_size + 1)
    
if __name__ == '__main__':
    main()