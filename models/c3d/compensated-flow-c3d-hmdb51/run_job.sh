#!/bin/sh
# running job with at command

# change to the folder containing the script & execute the command
DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $DIR
bash train_action_recognition_flow_c3d.sh > hmdb51_c3d_flow_step2_compensated_bs120_split1_wi1.log 2>&1