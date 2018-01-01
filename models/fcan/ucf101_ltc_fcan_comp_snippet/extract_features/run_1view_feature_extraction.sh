#!/usr/bin/env bash

FEATURE_FOLDER=/home/tranlaman/Public/data/ucf101_features/c3d-ltc-fcan-comp/
SPLIT=1

# ./ucf101_c3d_rgb_ltc_fcan_maxpool_sz58_len48_feature_extraction.sh 0 ../c3d_rgb_ltc_fcan_maxpool_sz58_len48_1_fc_split1_bs64_fi1_iter_20000.caffemodel $FEATURE_FOLDER &
# ./ucf101_c3d_rgb_fcan_pool1_sz112_len16_split2_feature_extraction.sh 2 ../c3d_rgb_fcan_pool1_sz112_len16_split2_bs64_fi2_iter_20000.caffemodel $FEATURE_FOLDER
# ./ucf101_c3d_rgb_fcan_pool1_sz112_len16_feature_extraction.sh 0 ../c3d_rgb_fcan_pool1_sz112_len16_split1_bs64_fi1_iter_20000.caffemodel $FEATURE_FOLDER
./ucf101_c3d_rgb_ltc_fcan_maxpool_sz58_len16_feature_extraction.sh 1 ../c3d_rgb_ltc_fcan_maxpool_sz58_len16_1_fc_split1_bs64_fi1_iter_20000.caffemodel $FEATURE_FOLDER
wait

# classification
cd ../evaluate_models
bash classify_clip_and_video_accuracy_ucf101.sh $FEATURE_FOLDER/c3d_rgb_ltc_fcan_maxpool_sz58_len16_1_fc_split1_bs64_fi1_iter_20000 $SPLIT rgb_fc8