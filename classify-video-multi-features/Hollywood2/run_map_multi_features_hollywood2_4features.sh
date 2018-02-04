#!/usr/bin/env bash

SPLIT=1

# feature 1
FEATURE_FOLDER1=/home/tranlaman/Public/data/hollywood2_features/fcan-c3d-features/hollywood2_c3d_rgb_fcan_comp_pool1_sz112_len16_bs64_split1_fi1_iter_10000
EXT1=rgb_fc8
WEIGHT1=1
# feature 2
FEATURE_FOLDER2=/home/tranlaman/Public/data/hollywood2_features/c3d-features/hollywood2_c3d_comp_flow_bs120_wi1_iter_10000
# FEATURE_FOLDER2=/home/tranlaman/Public/data/hollywood2_features/c3d-features//hollywood2_c3d_flow_bs64_wi1_iter_10000
EXT2=flow_fc8
WEIGHT2=1

# feature 3
FEATURE_FOLDER3=/home/tranlaman/Public/data/hollywood2_features/tsn_bn_inception/hollywood2_tsn_bn_inception_rgb_wholevideo_bs128_li1_iter_2500
EXT3=rgb_fc
WEIGHT3=1
# feature 2
FEATURE_FOLDER4=/home/tranlaman/Public/data/hollywood2_features/tsn_bn_inception/hollywood2_tsn_bn_inception_flow_wholevideo_bs128_fi1_iter_7000
EXT4=flow_fc
WEIGHT4=1

# video classification
python classify_video_map_multi_features_hollywood2.py --split=$SPLIT --feature_folders=$FEATURE_FOLDER1 --prob_extensions=$EXT1 --fusion_weights=$WEIGHT1\
 --feature_folders=$FEATURE_FOLDER2 --prob_extensions=$EXT2 --fusion_weights=$WEIGHT2 --feature_folders=$FEATURE_FOLDER3 --prob_extensions=$EXT3 --fusion_weights=$WEIGHT3\
 --feature_folders=$FEATURE_FOLDER4 --prob_extensions=$EXT4 --fusion_weights=$WEIGHT4 
