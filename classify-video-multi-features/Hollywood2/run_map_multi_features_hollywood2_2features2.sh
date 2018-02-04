#!/usr/bin/env bash

SPLIT=1

# feature 1
FEATURE_FOLDER1=/home/tranlaman/Public/data/hollywood2_features/tsn_bn_inception/hollywood2_tsn_bn_inception_rgb_wholevideo_bs128_li1_iter_2500
EXT1=rgb_fc
WEIGHT1=1
# feature 2
FEATURE_FOLDER2=/home/tranlaman/Public/data/hollywood2_features/tsn_bn_inception/hollywood2_tsn_bn_inception_flow_wholevideo_bs128_fi1_iter_7000
EXT2=flow_fc
WEIGHT2=1

# video classification
python classify_video_map_multi_features_hollywood2.py --split=$SPLIT --feature_folders=$FEATURE_FOLDER1 --prob_extensions=$EXT1 --fusion_weights=$WEIGHT1\
 --feature_folders=$FEATURE_FOLDER2 --prob_extensions=$EXT2 --fusion_weights=$WEIGHT2
