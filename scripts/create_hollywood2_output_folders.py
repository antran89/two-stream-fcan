#!/usr/bin/env python
__author__ = 'tranlaman'

import os
import argparse
import numpy as np

test_file = '/home/tranlaman/Public/data/video/Hollywood2/ClipSets/actions_test.txt'

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
    
    # video
    videos = np.loadtxt(test_file, dtype=str, delimiter='  ', comments=None)
    for video in videos:
        vid = video[0]
        vidFolder = os.path.join(outFolder, vid)
        if not os.path.isdir(vidFolder):
            os.makedirs(vidFolder)
                
    print 'Done!'
    

if __name__ == '__main__':
    main()