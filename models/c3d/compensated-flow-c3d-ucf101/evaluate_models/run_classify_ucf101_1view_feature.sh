#!/usr/bin/env bash

FEATURE_FOLDER=/home/tranlaman/Public/data/ucf101_features/c3d-features/

bash classify_clip_and_video_accuracy_ucf101.sh $FEATURE_FOLDER/c3d_flow_nd_conv_bs150_model1_iter_10000 1 prob
