#!/usr/bin/env bash

FEATURE_FOLDER=/home/tranlaman/Public/data/hollywood2_features/fcan-c3d-features/
SPLIT=1

# ./hollywood2_c3d_fcan_pool1_1view_testset_feature_extraction.sh 1 ../hollywood2_c3d_rgb_fcan_comp_pool1_sz112_len16_bs64_split1_fi1_iter_5000.caffemodel $FEATURE_FOLDER &
./hollywood2_c3d_fcan_pool2_1view_testset_feature_extraction.sh 1 ../hollywood2_c3d_rgb_fcan_comp_pool2_sz112_len16_bs64_split1_fi1_iter_5000.caffemodel $FEATURE_FOLDER
sleep 2s
wait

# classification
cd ../evaluate_models
bash classify_video_map_hollywood2.sh $FEATURE_FOLDER/hollywood2_c3d_rgb_fcan_comp_pool2_sz112_len16_bs64_split1_fi1_iter_5000 $SPLIT rgb_fc8