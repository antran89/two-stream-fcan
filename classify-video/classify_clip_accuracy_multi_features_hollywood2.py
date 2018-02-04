#!/usr/bin/python
'''
Compute multi-features clip accuracy of Hollywood2 datasets.
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
    parser = argparse.ArgumentParser(description='Compute multi-features clip accuracy of Hollywood2 datasets.')
    parser.add_argument('--feature_folders', dest='feature_folders', help='Extracted feature folder', required=True,
                        type=str, action='append')
    parser.add_argument('--split', dest='split', help='split of Hollywood2', default=1,
                        type=int)
    parser.add_argument('--prob_extensions', dest='prob_extensions', help='File extension of predicted probability', required=True,
                        type=str, action='append')
    parser.add_argument('--fusion_weights', dest='fusion_weights', help='Fusion weights of spatial features', required=True,
                        type=int, action='append')

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
    
    print 'computing multiple features clip accuracy of Hollywood2 split %d datasets ...' % split
    
    start_time = datetime.now()
    
    # getting train/test files information
    info_folder = '/home/tranlaman/Public/data/video/Hollywood2/ClipSets/'
    actions = ['AnswerPhone', 'DriveCar', 'Eat', 'FightPerson', 'GetOutCar', 
    'HandShake', 'HugPerson', 'Kiss', 'Run', 'SitDown', 'SitUp', 'StandUp']
    num_actions = len(actions)
    
    test_info_file = '/home/tranlaman/Public/data/video/Hollywood2/ClipSets/actions_test.txt'
    lines = np.loadtxt(test_info_file, dtype=str, delimiter='  ', comments=None)
    testFiles = lines[:, 0]
    test_labels = []
    numTestFiles = len(testFiles)
    
    list_videos_per_action = []
    for ind in xrange(len(actions)):
        action = actions[ind]
        action_split_file = '%s/%s_test.txt' % (info_folder, action)
        lines = np.loadtxt(action_split_file, dtype=str, delimiter='  ', comments=None)
        videos_per_action = []
        for line in lines:
           if int(line[1]) == 1:
               videos_per_action.append(line[0])
        list_videos_per_action.append(videos_per_action)
        
    for fi in testFiles:
        video_labels = []
        for ind in xrange(len(actions)):
            if fi in list_videos_per_action[ind]:
                video_labels.append(ind)
        test_labels.append(video_labels)    
    
    numClassClip = np.zeros((num_actions))
    perClassAccuracy = np.zeros((num_actions))    
    numCorrectPrediction = 0
    numClip = 0    
    for i in xrange(numTestFiles):
        video = testFiles[i]
        video_label = test_labels[i]
        
        # get features for the first feature type
        first_feature_path = os.path.join(feature_folders[0], video, '*.%s' % prob_extensions[0])
        first_feature_files = glob.glob(first_feature_path)
        first_feature_files.sort()
        if len(first_feature_files) == 0:
            print 'There is no clips in video %s' % video
            sys.exit(1)
        
        for ind in xrange(0, len(first_feature_files)):
            numClassClip[video_label] += 1    
            numClip += 1
            # get base file names
            first_file_name = os.path.basename(first_feature_files[ind])
            file_name_ = os.path.splitext(first_file_name)[0]
            prob = np.zeros((num_actions, 1))
            for k in xrange(0, num_features):
                file_path = os.path.join(feature_folders[k], video, 
                                                      '%s.%s' % (file_name_, prob_extensions[k]))
                if not os.path.isfile(file_path):
                    print 'File %s does not exist' % file_path
                    sys.exit(1)
                feature_prob = np.array(load_blob_from_binary(file_path))
                prob = prob + fusion_weights[k] * feature_prob

            # fusion prob
            prediction = np.argmax(prob)
            if prediction in video_label:
                numCorrectPrediction += 1
                perClassAccuracy[prediction] += 1
            
    accuracy = float(numCorrectPrediction)/numClip
    
    current_time = datetime.now()
    run_time = current_time - start_time
    print 'Run-time: ', run_time
    print "Clip accuracy of two-stream on test set: {}".format(accuracy)
    
    # compute per class accuracy
    perClassAccuracy = perClassAccuracy / numClassClip
    
    # write the results into txt file
    expString = 'Clips classification with multi-features on Hollywood2 dataset.\n'
    fid = open('results.txt', 'a')
    fid.write('\n--------------------------------------------------\n')
    fid.write('%s' %(expString))
    fid.write('Clip accuracy on testset split %d of Hollywood2 is %f\n' %(split, accuracy))
    fid.write('Test on following feature folders:\n')
    for k in xrange(num_features):
        fid.write('Feature folder: %s\n' % feature_folders[k])
        fid.write('Prob extension: %s\n' % prob_extensions[k])
        fid.write('Fusion weight: %s\n' % fusion_weights[k])
    fid.write('Expriment finished at %s \n' %current_time)
    fid.close()
    
    # write per class accuracy
    # write the results into txt file
    fid = open('perClassAccuracy.txt', 'a')
    fid.write('\n--------------------------------------------------\n')
    fid.write('%s' %(expString))
    fid.write('Clip accuracy on testset split %d of Hollywood2 is %f\n' %(split, accuracy))
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