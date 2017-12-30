#!/usr/bin/env bash

FEATURE_FOLDER=/home/tranlaman/Public/data/hmdb51_features/features-10view/
SPLIT=1

./hmdb51_c3d_flow_nd_conv_10view_testset_feature_extraction.sh 3 ../c3d_flow_nd_conv_split1_wi1_iter_10000.caffemodel $FEATURE_FOLDER &
sleep 2s
wait

# classification
cd ../evaluate_models
bash classify_clip_and_video_accuracy_hmdb51.sh $FEATURE_FOLDER/c3d_flow_nd_conv_split1_wi1_iter_10000 $SPLIT flow_prob