#!/usr/bin/python
'''
Compute multi-features video accuracy of HMDB51 datasets.
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
    parser = argparse.ArgumentParser(description='Compute multi-features clip accuracy of HMDB51 datasets.')
    parser.add_argument('--feature_folders', dest='feature_folders', help='Extracted feature folder', required=True,
                        type=str, action='append')
    parser.add_argument('--split', dest='split', help='split of HMDB51', default=1,
                        type=int)
    parser.add_argument('--prob_extensions', dest='prob_extensions', help='File extension of predicted probability', required=True,
                        type=str, action='append')
    parser.add_argument('--fusion_weights', dest='fusion_weights', help='Fusion weights of spatial features', required=True,
                        type=float, action='append')

    args = parser.parse_args()

    return args

def main():
    
    args = parse_args()
    feature_folders = args.feature_folders
    prob_extensions = args.prob_extensions
    fusion_weights = args.fusion_weights
    split = args.split
    
    num_features = len(feature_folders)
    assert(num_features >= 1)
    assert(len(prob_extensions) == 1 or len(prob_extensions) == num_features)
    assert(len(fusion_weights) == 1 or len(fusion_weights) == num_features)
    if (len(prob_extensions) == 1):
        prob_extensions = np.repeat(prob_extensions, num_features)
    if (len(fusion_weights) == 1):
        fusion_weights = np.repeat(fusion_weights, num_features)
    
    print 'computing multi-features video accuracy of HMDB51 split %d datasets ...' % split
    
    start_time = datetime.now()

    # getting train/test files information
    info_folder = '/home/tranlaman/Public/data/video/HMDB/TrainTestSplit/TrainTestSplit'
    classFile = '%s/classes.txt' % info_folder
    actions = np.loadtxt(classFile, dtype=str, delimiter=' ')
    num_actions = len(actions)
    
    train_files = []
    test_files = []
    for ind in xrange(len(actions)):
        action = actions[ind]
        action_split_file = '%s/%s_test_split%d.txt' % (info_folder, action, split)
        lines = np.loadtxt(action_split_file, dtype=str, delimiter=' ', comments=None)
        for line in lines:
           if int(line[1]) == 1:
               train_files.append(np.array([line[0][:-4], ind]))
           elif int(line[1]) == 2:
               test_files.append(np.array([line[0][:-4], ind]))
        
    num_train_files = len(train_files)
    num_test_files = len(test_files)
    assert(num_train_files == 3570)
    assert(num_test_files == 1530)
    
    numClassVideo = np.zeros((num_actions))
    perClassAccuracy = np.zeros((num_actions))
    numCorrectPrediction = 0
    numVideo = 0
    for i in xrange(num_test_files):
        video = test_files[i][0]
        video_label = int(test_files[i][1])
        action = actions[video_label]
        
        mean_prob = np.zeros((num_actions))
        for k in xrange(0, num_features):
            # load spatial features
            feature_path = os.path.join(feature_folders[k], action, video)
            feature_files = os.listdir(feature_path)
            feature_files.sort()
            if len(feature_files) == 0:
                print 'There is no clips in video %s' % video
                sys.exit(1)

            prob_mat = np.zeros((num_actions, len(feature_files)))
            for ind in xrange(0, len(feature_files)):
                file_path = os.path.join(feature_folders[k], action, video, feature_files[ind])
                blob = load_blob_from_binary(file_path)
                prob_mat[:, ind] = blob

            feature_mean_prob = np.mean(prob_mat, axis=1)
            mean_prob = mean_prob + fusion_weights[k] * feature_mean_prob
        
        numVideo += 1
        numClassVideo[video_label] += 1
        prediction = np.argmax(mean_prob)
        conf = mean_prob[prediction]
        if prediction == video_label:
            numCorrectPrediction += 1
            perClassAccuracy[prediction] += 1

    accuracy = float(numCorrectPrediction)/numVideo
    
    current_time = datetime.now()
    run_time = current_time - start_time
    print 'Run-time: ', run_time
    print "Video accuracy of two-stream on test set: {}".format(accuracy)
        
    # compute per class accuracy
    perClassAccuracy = perClassAccuracy / numClassVideo
    
    # write the results into txt file
    expString = 'Video classification with multiple cnn features.\n'
    fid = open('results.txt', 'a')
    fid.write('\n--------------------------------------------------\n')
    fid.write('%s' %(expString))
    fid.write('Video accuracy on testset split %d of HMDB51 is %f\n' %(split, accuracy))
    fid.write('Test on following feature folders:\n')
    for k in xrange(num_features):
        fid.write('Feature folder: %s\n' % feature_folders[k])
        fid.write('Prob extension: %s\n' % prob_extensions[k])
        fid.write('Fusion weight: %s\n' % fusion_weights[k])
    fid.write('Expriment finished at %s \n' % current_time)
    fid.close()
    
    # write per class accuracy
    # write the results into txt file
    fid = open('perClassAccuracy.txt', 'a')
    fid.write('\n--------------------------------------------------\n')
    fid.write('%s' %(expString))
    fid.write('Video accuracy on testset split %d of HMDB51 is %f\n' %(split, accuracy))
    fid.write('Test on following feature folders:\n')
    for k in xrange(num_features):
        fid.write('Feature folder: %s\n' % feature_folders[k])
        fid.write('Prob extension: %s\n' % prob_extensions[k])
        fid.write('Fusion weight: %s\n' % fusion_weights[k])
    fid.write('Expriment finished at %s \n' % current_time)
    fid.write('Per class accuracy\n')
    for ind in xrange(0, num_actions):
        fid.write('Class %d: ... %0.2f\n' %(ind, perClassAccuracy[ind]))
    fid.close()
    
if __name__ == '__main__':
    main()