#!/usr/bin/env bash

FEATURE_FOLDER=/home/tranlaman/Public/data/hmdb51_features/c3d-features/
SPLIT=2

# ./hmdb51_c3d_flow_nd_conv_1view_testset_feature_extraction.sh 0 ../hmdb51_c3d_compensated_flow_bs120_split1_wi1_iter_20000.caffemodel $FEATURE_FOLDER &
./hmdb51_c3d_flow_nd_conv_1view_testset_split2_feature_extraction.sh 0 ../hmdb51_c3d_compensated_flow_bs120_split2_wi1_iter_20000.caffemodel $FEATURE_FOLDER
# ./hmdb51_c3d_flow_bn_1view_testset_feature_extraction.sh 0 ../hmdb51_c3d_flow_bs105_bn_split1_fi2_iter_20000.caffemodel $FEATURE_FOLDER &
sleep 2s
wait

# classification
cd ../evaluate_models
bash classify_clip_and_video_accuracy_hmdb51.sh $FEATURE_FOLDER/hmdb51_c3d_compensated_flow_bs120_split2_wi1_iter_20000 $SPLIT flow_fc8
bash classify_clip_and_video_accuracy_hmdb51.sh $FEATURE_FOLDER/hmdb51_c3d_compensated_flow_bs120_split2_wi1_iter_20000 $SPLIT flow_prob