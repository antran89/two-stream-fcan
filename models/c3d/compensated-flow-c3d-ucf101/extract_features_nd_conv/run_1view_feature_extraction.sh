#!/usr/bin/env bash

FEATURE_FOLDER=../../../../internal-data/features/ucf101_features/c3d-features/
SPLIT=1

./ucf101_c3d_flow_nd_conv_1view_testset_feature_extraction.sh 0 ../ucf101_c3d_compensated_flow_bs120_wi1_iter_20000.caffemodel $FEATURE_FOLDER &
sleep 2s
wait

# classification
cd ../evaluate_models
bash classify_clip_and_video_accuracy_ucf101.sh $FEATURE_FOLDER/ucf101_c3d_compensated_flow_bs120_wi1_iter_20000 $SPLIT flow_fc8