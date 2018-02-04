#!/usr/bin/env bash

SPLIT=1

# feature 1
# FEATURE_FOLDER1=/home/tranlaman/Public/data/hmdb51_features/tsn_bn_inception/hmdb51_tsn_bn_inception_wholevideo_bs128_li1_iter_2500
FEATURE_FOLDER1=/home/tranlaman/Public/data/hmdb51_features/c3d-features/hmdb51_c3d_rgb_ltc_maxpool_sz58_len60_1_fc_bs64_split1_ti1_iter_10000
EXT1=rgb_fc8
WEIGHT1=1
# feature 2
# FEATURE_FOLDER2=/home/tranlaman/Public/data/hmdb51_features/tsn_bn_inception/hmdb51_tsn_bn_inception_flow_wholevideo_bs128_fi1_iter_7000
FEATURE_FOLDER2=/home/tranlaman/Public/data/hmdb51_features/c3d-features/hmdb51_c3d_flow_ltc_maxpool_sz58_len60_1_fc_bs64_split1_wi1_iter_20000
EXT2=flow_fc8
WEIGHT2=1

# video classification
python classify_video_accuracy_multi_features_hmdb51.py --split=$SPLIT --feature_folders=$FEATURE_FOLDER1 --prob_extensions=$EXT1 --fusion_weights=$WEIGHT1\
 --feature_folders=$FEATURE_FOLDER2 --prob_extensions=$EXT2 --fusion_weights=$WEIGHT2

# clip classification
python classify_clip_accuracy_multi_features_hmdb51.py --split=$SPLIT --feature_folders=$FEATURE_FOLDER1 --prob_extensions=$EXT1 --fusion_weights=$WEIGHT1\
 --feature_folders=$FEATURE_FOLDER2 --prob_extensions=$EXT2 --fusion_weights=$WEIGHT2
