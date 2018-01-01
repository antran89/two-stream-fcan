#!/bin/sh
# running job with at command

# change to the folder containing the script & execute the command
DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $DIR
bash train_action_recognition_rgb_fcan.sh > hmdb51_c3d_rgb_fcan_pool1_sz112_len16_bs64_split2_fi01.log 2>&1