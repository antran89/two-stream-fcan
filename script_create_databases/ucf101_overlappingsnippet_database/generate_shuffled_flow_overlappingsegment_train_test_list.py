'''
Creating UCF101 shuffled flow train test list.
Generate shuffled train/test lists for creating overlapping segments flow lmdb database.
Also store key values in lmdb database in text files with format:
video starting_frame_index video_label
EXCEPTIONS: For LTC experiments, some videos have number of frames smaller than
temporal length T, we only get 1 segment.
'''

import os
import sys
import glob
import numpy as np
import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Creating UCF101 shuffled flow train test list.')
    parser.add_argument('--dataset_folder', dest='dataset_folder', help='dataset folder.', required=True,
                        type=str)
    parser.add_argument('--split', dest='split', help='split of ucf101 dataset', default=1,
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
    classFile = '/home/tranlaman/Public/data/video/UCF101/ucfTrainTestlist/classInd.txt'
    actions = np.loadtxt(classFile, dtype=str, delimiter=' ')
    
    trainFile = '/home/tranlaman/Public/data/video/UCF101/ucfTrainTestlist/trainlist%02d.txt' % split
    trainFiles = np.loadtxt(trainFile, str, delimiter=' ')
    numTrainFile = len(trainFiles)

    # train files
    trainLists = []
    trainKeys = []
    miniBatchCounter = 0
    for ind in xrange(0, numTrainFile):
        video = trainFiles[ind][0]
        video_label = int(trainFiles[ind][1]) - 1
        vid = video[0:-4]
        framePath = os.path.join(dataset, vid)
        frames = glob.glob(os.path.join(framePath, '*.jpg'))
        frames.sort()
        num_frames = len(frames)/2
        if num_frames < new_length:
            print 'There is no clips in video %s' % vid
            miniBatchCounter += 1
            trainLists.append('%s %d %d' % (framePath, 1, video_label))
            trainKeys.append('%s %04d %d' % (vid, 1, video_label))
            continue
            # sys.exit(1)
        step = int((num_frames - new_length) / num_segments)
        if step == 0:
            print 'There are not enough frames in video %s' % vid
            step = 1
        num_iters = min(int((num_frames - new_length + 1) / step), num_segments)
        for seg in xrange(0, num_iters):
            i = seg * step
            miniBatchCounter += 1
            trainLists.append('%s %d %d' % (framePath, i + 1, video_label))
            trainKeys.append('%s %04d %d' % (vid, i + 1, video_label))
                
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
    
    testFile = '/home/tranlaman/Public/data/video/UCF101/ucfTrainTestlist/testlist%02d.txt' % split
    testFiles = np.loadtxt(testFile, str, delimiter='/')
    numTestFiles = len(testFiles)
    # making the label of test data
    test_labels = np.zeros((numTestFiles, 1), dtype=int)
    label = 0
    test_labels[0] = label
    for ind in xrange(1, numTestFiles):
        if testFiles[ind][0] != testFiles[ind-1][0]:
            label = label + 1
        test_labels[ind] = label
    assert(label == 100)
    # load test fiel again with different format    
    testFiles = np.loadtxt(testFile, str, delimiter=' ')    

    # test files
    testLists = []
    testKeys = []
    miniBatchCounter = 0
    for ind in xrange(0, numTestFiles):
        video = testFiles[ind]
        video_label = test_labels[ind]
        vid = video[0:-4]
        framePath = os.path.join(dataset, vid)
        frames = glob.glob(os.path.join(framePath, '*.jpg'))
        frames.sort()
        num_frames = len(frames)/2
        if num_frames < new_length:
            print 'There is no clips in video %s' % vid
            miniBatchCounter += 1
            testLists.append('%s %d %d' % (framePath, 1, video_label))
            testKeys.append('%s %04d %d' % (vid, 1, video_label))
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
            testKeys.append('%s %04d %d' % (vid, i + 1, video_label))
                            
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