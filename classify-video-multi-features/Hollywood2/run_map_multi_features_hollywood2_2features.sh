#!/usr/bin/env bash

SPLIT=1

# feature 1
FEATURE_FOLDER1=/home/tranlaman/Public/data/hollywood2_features/c3d-features//hollywood2_c3d_rgb_bs64_fi1_iter_10000
# FEATURE_FOLDER1=/home/tranlaman/Public/data/hollywood2_features/fcan-c3d-features/hollywood2_c3d_rgb_fcan_comp_pool1_sz112_len16_bs64_split1_fi1_iter_5000
EXT1=rgb_fc8
WEIGHT1=1
# feature 2
# FEATURE_FOLDER2=/home/tranlaman/Public/data/hollywood2_features/c3d-features/hollywood2_c3d_comp_flow_bs120_wi1_iter_10000
FEATURE_FOLDER2=/home/tranlaman/Public/data/hollywood2_features/c3d-features//hollywood2_c3d_flow_bs64_wi1_iter_10000
EXT2=flow_fc8
WEIGHT2=1

# video classification
python classify_video_map_multi_features_hollywood2.py --split=$SPLIT --feature_folders=$FEATURE_FOLDER1 --prob_extensions=$EXT1 --fusion_weights=$WEIGHT1\
 --feature_folders=$FEATURE_FOLDER2 --prob_extensions=$EXT2 --fusion_weights=$WEIGHT2
