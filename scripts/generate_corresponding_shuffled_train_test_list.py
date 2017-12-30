'''
Generate shuffled sliding window (non-overlapping) of train/test lists for creating rgb lmdb database.
The key files of rgb train_test list are inheritted from the corresponding flow lmdb datase.
It is to ensure the matchings between a segment of flow and the corresponding segment of rgb frames.
Also copy key values file into current folder.
'''

import os
import numpy as np
import shutil
import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Creating shuffled train test list from keys file.')
    parser.add_argument('--dataset_folder', dest='dataset_folder', help='dataset folder.', required=True,
                        type=str)
    parser.add_argument('--split', dest='split', help='split of ucf101 dataset', default=1,
                        type=int)
    parser.add_argument('--train_key_file', dest='train_key_file', help='Train key file.', required=True,
                        type=str)
    parser.add_argument('--test_key_file', dest='test_key_file', help='Test key file.', required=True,
                        type=str)
    parser.add_argument('--shuffled_train_list_file', dest='shuffled_train_list_file', help='shuffled train list file.',
                        required=True, type=str)
    parser.add_argument('--shuffled_test_list_file', dest='shuffled_test_list_file', help='shuffled test list file.',
                        required=True, type=str)

    args = parser.parse_args()

    return args
    
def main():

    args = parse_args()
    dataset = args.dataset_folder
    split = args.split    
    trainKeyFile = args.train_key_file
    testKeyFile = args.test_key_file
    shuffled_train_list_file = args.shuffled_train_list_file
    shuffled_test_list_file = args.shuffled_test_list_file
    
    trainFid = open(shuffled_train_list_file, 'w')
    trainKeys = np.loadtxt(trainKeyFile, dtype=str, delimiter=' ', comments=None)
    
    for key in trainKeys:
        vidPath = os.path.join(dataset, key[0])
        trainFid.write('%s %d %s\n' % (vidPath, int(key[1]), key[2]))
        
    trainFid.close
    
    testFid = open(shuffled_test_list_file, 'w')
    testKeys = np.loadtxt(testKeyFile, dtype=str, delimiter=' ', comments=None)
    
    for key in testKeys:
        vidPath = os.path.join(dataset, key[0])
        testFid.write('%s %d %s\n' % (vidPath, int(key[1]), key[2]))
        
    testFid.close
    
if __name__ =='__main__':
    main()