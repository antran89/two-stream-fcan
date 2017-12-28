#!/usr/bin/env bash

FEATURE_FOLDER=/home/tranlaman/Public/data/hollywood2_features/c3d-features/
SPLIT=1

./hollywood2_c3d_flow_nd_conv_1view_testset_feature_extraction.sh 0 ../hollywood2_c3d_comp_flow_bs120_wi1_iter_10000.caffemodel $FEATURE_FOLDER &
sleep 2s
wait

# classification
cd ../evaluate_models
bash classify_video_map_hollywood2.sh $FEATURE_FOLDER/hollywood2_c3d_comp_flow_bs120_wi1_iter_10000 $SPLIT flow_fc8