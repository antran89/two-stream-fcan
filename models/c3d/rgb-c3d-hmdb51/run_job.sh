#!/bin/sh
# running job with at command

# change to the folder containing the script & execute the command
DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $DIR
./train_action_recognition_rgb_c3d.sh > c3d_rgb_overlapping_bs128_split2_fi1.log 2>&1