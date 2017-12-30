#!/usr/bin/env python
__author__ = 'tranlaman'

import os
import argparse

dataset = '/home/tranlaman/Public/data/video/UCF101/UCF-101'
new_length = 10

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Create output folders to store features extracted from deep network.')
    parser.add_argument('--output_folder', dest='output_folder', help='Output feature folder', default=None,
                        type=str)

    args = parser.parse_args()

    return args

def main():
    
    args = parse_args()
    outFolder = args.output_folder
    
    actions = os.listdir(dataset)
    actions.sort()
    
    for action in actions:
        videos = os.listdir(os.path.join(dataset, action))
        videos.sort()
        actionFolder = os.path.join(outFolder, action)
        if not os.path.isdir(actionFolder):
            os.makedirs(actionFolder)
        for vid in videos:
            vidFolder = os.path.join(outFolder, action, vid[:-4])
            if not os.path.isdir(vidFolder):
                os.makedirs(vidFolder)
                
    print 'Done!'
    

if __name__ == '__main__':
    main()