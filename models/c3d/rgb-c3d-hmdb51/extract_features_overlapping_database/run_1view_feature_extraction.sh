#!/usr/bin/env bash

FEATURE_FOLDER=../../../../internal-data/features/hmdb51_features/c3d-features/
SPLIT=1

./hmdb51_c3d_nd_conv_1view_testset_feature_extraction.sh 0 ../hmdb51_c3d_rgb_bs124_split1_of1_iter_10000.caffemodel $FEATURE_FOLDER &
sleep 2s
wait

# classification
cd ../evaluate_models
bash classify_clip_and_video_accuracy_hmdb51.sh $FEATURE_FOLDER/hmdb51_c3d_rgb_bs124_split1_of1_iter_10000 $SPLIT rgb_fc8