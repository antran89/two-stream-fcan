#!/usr/bin/env bash

FEATURE_FOLDER=/home/tranlaman/Public/data/hmdb51_features/c3d-features/
SPLIT=2

./hmdb51_c3d_nd_conv_1view_testset_split2_feature_extraction.sh 3 ../c3d_rgb_overlapping_bs128_split2_fi1_iter_10000.caffemodel $FEATURE_FOLDER &
sleep 2s
wait

# classification
cd ../evaluate_models
bash classify_clip_and_video_accuracy_hmdb51.sh $FEATURE_FOLDER/c3d_rgb_overlapping_bs128_split2_fi1_iter_10000 $SPLIT rgb_fc8