# -*- coding: utf-8 -*-
'''
Generate shuffled sliding window (non-overlapping) of train/test lists for creating rgb lmdb database.
The key files of rgb train_test list are inheritted from the corresponding flow lmdb datase.
It is to ensure the matchings between a segment of flow and the corresponding segment of rgb frames.
Also copy key values file into current folder.
'''

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# gathering dataset information
flow_dataset = '/home/tranlaman/Public/data/video/ucf101_comp_tvl1_128_171/flow_folder/'
temporal_length = 60

actions = os.listdir(flow_dataset)
durations = []
for action in actions:
    videos = os.listdir(os.path.join(flow_dataset, action))
    for vid in videos:
        frame_path = os.path.join(flow_dataset, action, vid)
        frames = glob.glob(os.path.join(frame_path, '*.jpg'))
        num_frames = len(frames)/2
        durations.append(num_frames)

# some statistics of 
durations = np.array(durations)
print('total number of videos: %0.2f' % len(durations))
print('min, max duration: %0.2f, %0.2f' % (np.min(durations), np.max(durations)))
print('mean, median duration: %0.2f, %0.2f' % (np.mean(durations), np.median(durations))) 
print('5, 95 quantiles of duration: %0.2f, %0.2f' % (np.percentile(durations, 5), np.percentile(durations, 95)))
print('Number of videos has less than %d frames: %d' % (temporal_length, len(np.where(durations < 60)[0])))

# draw histogram
n, bins, patches = plt.hist(durations, 50, range=(0,500), normed=True, facecolor='green', alpha=0.75)
plt.xticks(range(0,500,50))
plt.title('Histogram of video durations in UCF101 dataset')
plt.ylabel('Frequency')
plt.xlabel('Duration in frames')
filename='ucf101_histogram_duration.pdf'
plt.savefig(filename, bbox_inches='tight')
plt.show()