'''
Creating HMDB51 shuffled flow train test list.
Generate shuffled train/test lists for creating overlapping segments flow lmdb database.
Also store key values in lmdb database in text files with format:
video starting_frame_index video_label
'''

import os
import sys
import glob
import numpy as np
import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Creating HMDB51 shuffled flow train test list.')
    parser.add_argument('--dataset_folder', dest='dataset_folder', help='dataset folder.', required=True,
                        type=str)
    parser.add_argument('--split', dest='split', help='split of hmdb51 dataset', default=1,
                        type=int)
    parser.add_argument('--new_length', dest='new_length', help='length of a segment of frames', default=10,
                        type=int)
    parser.add_argument('--shuffled_train_list_file', dest='shuffled_train_list_file', help='shuffled train list file.',
                        required=True, type=str)
    parser.add_argument('--shuffled_test_list_file', dest='shuffled_test_list_file', help='shuffled test list file.',
                        required=True, type=str)
    parser.add_argument('--shuffle', dest='shuffle', help='wheather to shuffle the index files.',
                        default=True, type=bool)
    parser.add_argument('--num_segments', dest='num_segments', help='number of segment clips per video', default=25,
                        type=int)
    args = parser.parse_args()

    return args

def main():
    
    args = parse_args()
    dataset = args.dataset_folder
    split = args.split
    new_length = args.new_length
    shuffled_train_list_file = args.shuffled_train_list_file
    shuffled_test_list_file = args.shuffled_test_list_file
    shuffle = args.shuffle
    num_segments = args.num_segments
    
    # gathering dataset information
    info_folder = '/home/tranlaman/Public/data/video/HMDB/TrainTestSplit/TrainTestSplit/'
    classFile = '/home/tranlaman/Public/data/video/HMDB/TrainTestSplit/TrainTestSplit/classes.txt'
    actions = np.loadtxt(classFile, dtype=str, delimiter=' ')

    trainFiles = []
    testFiles = []
    for ind in xrange(len(actions)):
        action = actions[ind]
        action_split_file = '%s/%s_test_split%d.txt' % (info_folder, action, split)
        lines = np.loadtxt(action_split_file, dtype=str, delimiter=' ', comments=None)
        for line in lines:
           if int(line[1]) == 1:
               trainFiles.append(np.array([line[0][:-4], str(ind)]))
           elif int(line[1]) == 2:
               testFiles.append(np.array([line[0][:-4], str(ind)]))
        
    numTrainFiles = len(trainFiles)
    numTestFiles = len(testFiles)
    assert(numTrainFiles == 3570)
    assert(numTestFiles == 1530)

    # train files
    trainLists = []
    trainKeys = []
    miniBatchCounter = 0
    for ind in xrange(0, numTrainFiles):
        vid = trainFiles[ind][0]
        video_label = int(trainFiles[ind][1])
        action = actions[int(video_label)]
        framePath = os.path.join(dataset, action, vid)
        frames = os.listdir(framePath)
        frames.sort()
        num_frames = len(frames)/2
        if num_frames < new_length:
            print 'There is no clips in video %s' % vid
            trainLists.append('%s %d %d' % (framePath, 1, video_label))
            trainKeys.append('%s %04d %d' % (os.path.join(action, vid), 1, video_label))
            continue
        step = int((num_frames - new_length) / num_segments)
        if step == 0:
            print 'There are not enough frames in video %s' % vid
            step = 1
        num_iters = min(int((num_frames - new_length + 1) / step), num_segments)
        for seg in xrange(0, num_iters):
            i = seg * step
            miniBatchCounter += 1
            trainLists.append('%s %d %d' % (framePath, i + 1, video_label))
            trainKeys.append('%s %04d %d' % (os.path.join(action, vid), i + 1, video_label))
                
    print 'number of training segments %d' % miniBatchCounter
    
    # shuffle trainFiles
    if shuffle:
        shuffle_index = np.random.permutation(len(trainLists))
        trainLists = np.array(trainLists)
        trainLists = trainLists[shuffle_index]
        trainKeys = np.array(trainKeys)
        trainKeys = trainKeys[shuffle_index]
    
    trainFid = open(shuffled_train_list_file, 'w')
    trainKeyFid = open('train_lmdb_keys.txt', 'w')
    for ind in xrange(len(trainLists)):
        trainFid.write('%s\n' % trainLists[ind])
        trainKeyFid.write('%s\n' % trainKeys[ind])
    trainFid.close()
    trainKeyFid.close()

    # test files
    testLists = []
    testKeys = []
    miniBatchCounter = 0
    for ind in xrange(0, numTestFiles):
        vid = testFiles[ind][0]
        video_label = int(testFiles[ind][1])
        action = actions[int(video_label)]
        framePath = os.path.join(dataset, action, vid)
        frames = os.listdir(framePath)
        frames.sort()
        num_frames = len(frames)/2
        if num_frames < new_length:
            print 'There is no clips in video %s' % vid
            testLists.append('%s %d %d' % (framePath, 1, video_label))
            testKeys.append('%s %04d %d' % (os.path.join(action, vid), 1, video_label))
            continue
        step = int((num_frames - new_length) / num_segments)
        if step == 0:
            print 'There are not enough frames in video %s' % vid
            step = 1
        num_iters = min(int((num_frames - new_length + 1) / step), num_segments)
        for seg in xrange(0, num_iters):
            i = seg * step
            miniBatchCounter += 1
            testLists.append('%s %d %d' % (framePath, i + 1, video_label))
            testKeys.append('%s %04d %d' % (os.path.join(action, vid), i + 1, video_label))
                            
    print 'number of test segments %d' % miniBatchCounter
    
    # shuffle test files
    if shuffle:
        shuffle_index = np.random.permutation(len(testLists))
        testLists = np.array(testLists)
        testLists = testLists[shuffle_index]
        testKeys = np.array(testKeys)
        testKeys = testKeys[shuffle_index]
    
    testFid = open(shuffled_test_list_file, 'w')
    testKeyFid = open('test_lmdb_keys.txt', 'w')
    for ind in xrange(len(testLists)):
        testFid.write('%s\n' % testLists[ind])
        testKeyFid.write('%s\n' % testKeys[ind])
    testFid.close()
    testKeyFid.close()

if __name__ == '__main__':
    main()