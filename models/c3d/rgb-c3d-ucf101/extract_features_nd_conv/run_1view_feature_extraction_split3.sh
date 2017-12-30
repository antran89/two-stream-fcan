#!/usr/bin/env bash

FEATURE_FOLDER=/home/tranlaman/Public/data/ucf101_features/c3d-features/
SPLIT=3

./ucf101_c3d_nd_conv_1view_testset_split3_feature_extraction.sh 2 ../ucf101_c3d_rgb_split3_bs120_fi1_iter_20000.caffemodel $FEATURE_FOLDER
sleep 2s
wait

# classification
cd ../evaluate_models
bash classify_clip_and_video_accuracy_ucf101.sh $FEATURE_FOLDER/ucf101_c3d_rgb_split3_bs120_fi1_iter_20000 $SPLIT rgb_fc8
bash classify_clip_and_video_accuracy_ucf101.sh $FEATURE_FOLDER/ucf101_c3d_rgb_split3_bs120_fi1_iter_20000 $SPLIT rgb_prob