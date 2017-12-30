#!/bin/sh
# running job with at command

# change to the folder containing the script & execute the command
DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $DIR
bash train_action_recognition_flow_c3d.sh > ucf101_step2_caffenet_twostream_m_fusion_fc6.log 2>&1