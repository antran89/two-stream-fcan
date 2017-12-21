#!/usr/bin/python
'''
Compute clip accuracy of UCF101 dataset.
'''

import numpy as np
import os
import glob
import struct
import argparse
import sys
from datetime import datetime


def load_blob_from_binary(file_name):
    with open(file_name, mode='rb') as file: # b is important -> binary
        file_content = file.read()
    shape = struct.unpack('iiiii', file_content[:20])
    dim = np.prod(shape)
    blob = struct.unpack('f' * dim, file_content[20:])
    return blob

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Compute clip accuracy of UCF101 datasets.')
    parser.add_argument('--feature_folder', dest='feature_folder', help='Extracted feature folder', required=True,
                        type=str)
    parser.add_argument('--split', dest='split', help='split of UCF101', default=1,
                        type=int)
    parser.add_argument('--prob_extension', dest='prob_extension', help='File extension of predicted probability', default='prob',
                        type=str)

    args = parser.parse_args()

    return args

def main():
    
    args = parse_args()
    feature_dataset = args.feature_folder
    split = args.split
    prob_extension = args.prob_extension
        
    print 'computing clip accuracy of UCF101 split %d datasets ...' % split
    
    start_time = datetime.now()
    
    # getting train/test files information
    test_info_file = '/home/tranlaman/Public/data/video/UCF101/ucfTrainTestlist/testlist%02d.txt' % split
    test_files = np.loadtxt(test_info_file, dtype=str, delimiter='/')
    num_files = len(test_files)
    
    actions = os.listdir(feature_dataset)
    actions.sort()
    num_actions = len(actions)    
    video_labels = np.zeros((num_files, 1), dtype=int)
    label = 0;
    video_labels[0] = label 
    for ind in xrange(1, num_files):
        if test_files[ind][0] != test_files[ind-1][0]:
            label += 1
        video_labels[ind] = label
    assert(label == num_actions-1)
    
    numClassClip = np.zeros((num_actions))
    perClassAccuracy = np.zeros((num_actions))
    numCorrectPrediction = 0
    numClip = 0    
    for i in xrange(num_files):
        line = test_files[i]
        action = line[0]
        video = line[1][:-4]
        video_label = video_labels[i]
        
        feature_path = os.path.join(feature_dataset, action, video, '*.%s' % prob_extension)
        feature_files = glob.glob(feature_path)
        feature_files.sort()
        if len(feature_files) == 0:
            print 'There is no clips in video %s' % video
            continue
            #sys.exit(1)
        
        for ind in xrange(0, len(feature_files)):
            numClassClip[video_label] += 1    
            numClip += 1
            prob = load_blob_from_binary(feature_files[ind])
            prediction = np.argmax(prob)
            if prediction == video_label:
                numCorrectPrediction += 1
                perClassAccuracy[prediction] += 1
            
    accuracy = float(numCorrectPrediction)/numClip
    
    current_time = datetime.now()
    run_time = current_time - start_time
    print 'Run-time: ', run_time
    print "Clip accuracy on test set: {}".format(accuracy)
    
    # compute per class accuracy
    perClassAccuracy = perClassAccuracy / numClassClip
    
    # write the results into txt file
    expString = 'Compute clip accuracy of UCF101 %d dataset.\n' % split
    fid = open('results.txt', 'a')
    fid.write('\n--------------------------------------------------\n')
    fid.write('%s' %(expString))
    fid.write('Feature folder: %s\n' % feature_dataset)
    fid.write('Probability extention: %s\n' % prob_extension)
    fid.write('Clip accuracy on testset split %d of UCF101 is %f\n' %(split, accuracy))
    fid.write('Expriment finished at %s \n' %current_time)
    fid.close()
    
    # write per class accuracy
    # write the results into txt file
    fid = open('perClassAccuracy.txt', 'a')
    fid.write('\n--------------------------------------------------\n')
    fid.write('%s' %(expString))
    fid.write('Feature folder: %s\n' % feature_dataset)
    fid.write('Probability extention: %s\n' % prob_extension)
    fid.write('Clip accuracy on testset split %d of UCF101 is %f\n' %(split, accuracy))
    fid.write('Expriment finished at %s \n' % current_time)
    fid.write('Per class accuracy\n')
    for ind in xrange(0, num_actions):
        fid.write('Class %d: ... %0.2f\n' %(ind, perClassAccuracy[ind]))
    fid.close()
    
if __name__ == '__main__':
    main()