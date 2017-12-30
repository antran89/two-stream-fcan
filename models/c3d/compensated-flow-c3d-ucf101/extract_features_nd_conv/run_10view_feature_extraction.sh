#!/usr/bin/env bash

FEATURE_FOLDER=/home/tranlaman/Public/data/extracted_features/new_test_data/c3d-features-10view/
SPLIT=1

./ucf101_c3d_flow_nd_conv_10view_testset_feature_extraction.sh 0 ../c3d_flow_nd_conv_bs150_model3_iter_20000.caffemodel $FEATURE_FOLDER &
sleep 2s
./ucf101_c3d_flow_nd_conv_10view_testset_feature_extraction.sh 1 ../c3d_flow_nd_conv_bs150_model2_iter_20000.caffemodel $FEATURE_FOLDER &
wait

# classification
cd ../evaluate_models
bash classify_clip_and_video_accuracy_ucf101.sh $FEATURE_FOLDER/c3d_flow_nd_conv_bs150_model3_iter_20000 $SPLIT flow_prob
bash classify_clip_and_video_accuracy_ucf101.sh $FEATURE_FOLDER/c3d_flow_nd_conv_bs150_model2_iter_20000 $SPLIT flow_prob