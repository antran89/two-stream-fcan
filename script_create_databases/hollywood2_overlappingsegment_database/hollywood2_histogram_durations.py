# -*- coding: utf-8 -*-
'''
Histogram of number of frames.
'''

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# gathering dataset information
flow_dataset = '/home/tranlaman/Public/data/video/Hollywood2_comp_tvl1_128_171/flow_folder/'
temporal_length = 60

videos = os.listdir(flow_dataset)
durations = []
for vid in videos:
    frame_path = os.path.join(flow_dataset, vid)
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
n, bins, patches = plt.hist(durations, 50, normed=True, facecolor='green', alpha=0.75)
plt.title('Histogram of video durations in HMDB51 dataset')
plt.ylabel('Frequency')
plt.xlabel('Duration in frames')
plt.show()