#!/usr/bin/env bash

FEATURE_FOLDER=../../../../internal-data/features/ucf101_features/c3d-features/
SPLIT=1

./ucf101_c3d_nd_conv_1view_testset_feature_extraction.sh 0 ../c3d_rgb_nd_conv_bs105_fi1_iter_20000.caffemodel $FEATURE_FOLDER &
sleep 2s
wait

# classification
cd ../evaluate_models
bash classify_clip_and_video_accuracy_ucf101.sh $FEATURE_FOLDER/c3d_rgb_nd_conv_bs105_fi1_iter_20000 $SPLIT rgb_fc8