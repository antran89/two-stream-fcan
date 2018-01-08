#!/usr/bin/env bash

FEATURE_FOLDER=../../../../internal-data/features/hmdb51_features/c3d-ltc-fcan-comp/
SPLIT=1

./hmdb51_c3d_rgb_fcan_pool1_sz112_len16_feature_extraction.sh 0 ../hmdb51_c3d_rgb_fcan_pool1_sz112_len16_bs64_split1_fi01_iter_10000.caffemodel $FEATURE_FOLDER
wait

# classification
cd ../evaluate_models
bash classify_clip_and_video_accuracy_hmdb51.sh $FEATURE_FOLDER/hmdb51_c3d_rgb_fcan_pool1_sz112_len16_bs64_split1_fi01_iter_10000 $SPLIT rgb_fc8