#!/usr/bin/env bash

SPLIT=1

# feature 1
FEATURE_FOLDER1=/home/tranlaman/Public/data/olympic_features/fcan-c3d-features/c3d_fcan_pool2_learned_link_bs128_fi2_iter_10000/
EXT1=rgb_fc8
WEIGHT1=1
# feature 2
FEATURE_FOLDER2=/home/tranlaman/Public/data/olympic_features/c3d-features/olympic_c3d_comp_flow_bs128_wi1_iter_10000/
EXT2=flow_fc8
WEIGHT2=1
# feature 3
FEATURE_FOLDER3=/home/tranlaman/Public/data/olympic_features/c3d-features//olympic_c3d_rgb_bs128_fi1_iter_5000
EXT3=rgb_fc8
WEIGHT3=1

# video classification
python classify_video_map_multi_features_olympic.py --split=$SPLIT --feature_folders=$FEATURE_FOLDER1 --prob_extensions=$EXT1 --fusion_weights=$WEIGHT1\
 --feature_folders=$FEATURE_FOLDER2 --prob_extensions=$EXT2 --fusion_weights=$WEIGHT2 \
 --feature_folders=$FEATURE_FOLDER3 --prob_extensions=$EXT3 --fusion_weights=$WEIGHT3
