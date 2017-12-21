#!/usr/bin/python
'''
Compute video accuracy of HMDB51 dataset. NOTICE: some videos in HMDB51 have strange
name patterns that the current glob.glob() function does not work. So we instead
use os.listdir(), but make sure that all feature files in the folder are from only
one type. We choose fc8 features because it gains good performances.
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
    parser = argparse.ArgumentParser(description='Compute video accuracy of HMDB51 datasets.')
    parser.add_argument('--feature_folder', dest='feature_folder', help='Extracted feature folder', default=None,
                        type=str)
    parser.add_argument('--split', dest='split', help='split of HMDB51', default=1,
                        type=int)
    parser.add_argument('--prob_extension', dest='prob_extension', help='File extension of predicted probability', 
                        default='prob', type=str)

    args = parser.parse_args()
    
    return args    

def main():
    
    args = parse_args()
    feature_dataset = args.feature_folder
    split = args.split   
    prob_extension = args.prob_extension
    
    print 'computing video accuracy of UCF101 split %d datasets ...' % split
    start_time = datetime.now()
    
    # getting train/test files information
    info_folder = '/home/tranlaman/Public/data/video/HMDB/TrainTestSplit/TrainTestSplit'
    classFile = '%s/classes.txt' % info_folder
    actions = np.loadtxt(classFile, dtype=str, delimiter=' ')
    num_actions = len(actions)
    
    trainFiles = []
    testFiles = []
    for ind in xrange(len(actions)):
        action = actions[ind]
        action_split_file = '%s/%s_test_split%d.txt' % (info_folder, action, split)
        lines = np.loadtxt(action_split_file, dtype=str, delimiter=' ', comments=None)
        for line in lines:
           if int(line[1]) == 1:
               trainFiles.append(np.array([line[0][:-4], ind]))
           elif int(line[1]) == 2:
               testFiles.append(np.array([line[0][:-4], ind]))
        
    numTrainFile = len(trainFiles)
    numTestFiles = len(testFiles)
    assert(numTrainFile == 3570)
    assert(numTestFiles == 1530)
    
    numClassVideo = np.zeros((num_actions))
    perClassAccuracy = np.zeros((num_actions))    
    num_files = len(testFiles)
    numCorrectPrediction = 0
    numVideo = 0
    for i in xrange(num_files):
        video = testFiles[i][0]
        video_label = int(testFiles[i][1])
        
        action = actions[video_label]
        feature_path = os.path.join(feature_dataset, action, video)
        feature_files = os.listdir(feature_path)
        feature_files.sort()
        if len(feature_files) == 0:
            print 'There is no clips in video %s' % video
            #continue
            sys.exit(1)        
        
        numVideo += 1
        numClassVideo[video_label] += 1
        prob_mat = np.zeros((num_actions, len(feature_files)))
        #prob_mat = np.zeros((101, len(feature_files)))
        for ind in xrange(0, len(feature_files)):
            blob = load_blob_from_binary(os.path.join(feature_path, feature_files[ind]))
            prob_mat[:, ind] = blob
            
        mean_prob = np.mean(prob_mat, axis=1)
        prediction = np.argmax(mean_prob)
        conf = mean_prob[prediction]
        if prediction == video_label:
            numCorrectPrediction += 1
            perClassAccuracy[prediction] += 1
            
    accuracy = float(numCorrectPrediction)/numVideo
    
    print "Video accuracy on test set: {}".format(accuracy)
    
    # print to files some information
    # compute per class accuracy
    perClassAccuracy = perClassAccuracy / numClassVideo
    current_time = datetime.now()
    run_time = current_time - start_time
    print 'Run-time: ', run_time
    
    # write the results into txt file
    expString = 'Compute video accuracy of HMDB51 dataset.\n'
    fid = open('results.txt', 'a')
    fid.write('\n--------------------------------------------------\n')
    fid.write('%s' %(expString))
    fid.write('Feature folder: %s\n' % feature_dataset)
    fid.write('Probability extention: %s\n' % prob_extension)
    fid.write('Video accuracy on testset split %d of HMDB51 is %f\n' %(split, accuracy))
    fid.write('Expriment finished at %s \n' % current_time)
    fid.close()
    
    # write per class accuracy
    # write the results into txt file
    fid = open('perClassAccuracy.txt', 'a')
    fid.write('\n--------------------------------------------------\n')
    fid.write('%s' %(expString))
    fid.write('Feature folder: %s\n' % feature_dataset)
    fid.write('Probability extention: %s\n' % prob_extension)
    fid.write('Video accuracy on testset split %d of HMDB51 is %f\n' %(split, accuracy))
    fid.write('Expriment finished at %s \n' % current_time)
    fid.write('Per class accuracy\n')
    for ind in xrange(0, num_actions):
        fid.write('Class %d: ... %0.2f\n' %(ind, perClassAccuracy[ind]))
    fid.close()

if __name__ == '__main__':
    main()