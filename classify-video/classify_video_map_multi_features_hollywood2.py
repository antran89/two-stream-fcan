#!/usr/bin/python
'''
Compute multi-features video accuracy of Hollywood2 datasets.
'''

import numpy as np
import os
import glob
import struct
import argparse
import sys
from datetime import datetime
import ml_metrics as metrics

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

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
    
    print 'computing multi-features video accuracy of Hollywood2 split %d datasets ...' % split
    
    start_time = datetime.now()

    # getting train/test files information
    info_folder = '/home/tranlaman/Public/data/video/Hollywood2/ClipSets/'
    actions = ['AnswerPhone', 'DriveCar', 'Eat', 'FightPerson', 'GetOutCar', 
    'HandShake', 'HugPerson', 'Kiss', 'Run', 'SitDown', 'SitUp', 'StandUp']
    num_actions = len(actions)
    
    test_info_file = '/home/tranlaman/Public/data/video/Hollywood2/ClipSets/actions_test.txt'
    lines = np.loadtxt(test_info_file, dtype=str, delimiter='  ', comments=None)
    testFiles = lines[:, 0]
    numTestFiles = len(testFiles)
    assert(numTestFiles == 884)

    list_video_gt_per_action = []
    for ind in xrange(len(actions)):
        action = actions[ind]
        action_split_file = '%s/%s_test.txt' % (info_folder, action)
        lines = np.loadtxt(action_split_file, dtype=str, delimiter='  ', comments=None)
        video_gt_per_action = []
        line_index = 0
        for line in lines:
           if int(line[1]) == 1:
               video_gt_per_action.append(line_index)
           line_index += 1
        list_video_gt_per_action.append(video_gt_per_action)
    
    # get prediction of videos
    list_video_prediction = list()
    list_video_score = list()
    for i in xrange(numTestFiles):
        video = testFiles[i]

        mean_prob = np.zeros((num_actions))
        for k in xrange(0, num_features):
            # load spatial features
            feature_path = os.path.join(feature_folders[k], video, '*.%s' % prob_extensions[k])
            feature_files = glob.glob(feature_path)
            feature_files.sort()
            if len(feature_files) == 0:
                print 'There is no clips in video %s' % video
                sys.exit(1)

            prob_mat = np.zeros((num_actions, len(feature_files)))
            for ind in xrange(0, len(feature_files)):
                file_path = os.path.join(feature_folders[k], video, feature_files[ind])
                blob = load_blob_from_binary(file_path)
                prob_mat[:, ind] = blob

            feature_mean_prob = np.mean(prob_mat, axis=1)
            mean_prob = mean_prob + fusion_weights[k] * feature_mean_prob

        mean_prob = softmax(mean_prob)
        prediction = np.argmax(mean_prob)
        conf = mean_prob[prediction]
        list_video_prediction.append(prediction)
        list_video_score.append(conf)

    list_video_prediction = np.array(list_video_prediction)
    list_video_score = np.array(list_video_score)
        
    # compute average precision
    ap = np.zeros(len(actions))
    list_video_id = np.array(range(numTestFiles))
    for ind in xrange(len(actions)):
        pred_index = (list_video_prediction == ind)
        video_id_pred = list_video_id[pred_index]
        score_pred = list_video_score[pred_index]
        #sorted_idx = score_pred.argsort()[::-1]
        sorted_idx = sorted(range(len(score_pred)), key=lambda k: score_pred[k], reverse=True)
        sorted_video_id_pred = video_id_pred[sorted_idx]
        
        video_id_gt = list_video_gt_per_action[ind]
        ap[ind] = metrics.apk(video_id_gt, sorted_video_id_pred)
        
    meanAP = ap.mean()

    current_time = datetime.now()
    run_time = current_time - start_time
    print 'Run-time: ', run_time
    print "Video meanAP of two-stream on test set: {}".format(meanAP)

    # write the results into txt file
    expString = 'Video classification (meanAP) with multiple cnn features.\n'
    fid = open('results.txt', 'a')
    fid.write('\n--------------------------------------------------\n')
    fid.write('%s' %(expString))
    fid.write('Video meanAP on testset split %d of Hollywood2 is %f\n' % (split, meanAP))
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
    fid.write('Video meanAP on testset split %d of Hollywood2 is %f\n' % (split, meanAP))
    fid.write('Test on following feature folders:\n')
    for k in xrange(num_features):
        fid.write('Feature folder: %s\n' % feature_folders[k])
        fid.write('Prob extension: %s\n' % prob_extensions[k])
        fid.write('Fusion weight: %s\n' % fusion_weights[k])
    fid.write('Expriment finished at %s \n' % current_time)
    fid.write('Per class AP\n')
    for ind in xrange(0, num_actions):
        fid.write('Class %d: ... %0.2f\n' %(ind, ap[ind]))
    fid.close()
    
if __name__ == '__main__':
    main()