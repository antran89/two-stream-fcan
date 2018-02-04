#!/usr/bin/env bash

SPLIT=3

# feature 1
FEATURE_FOLDER1=/home/tranlaman/Public/data/ucf101_features/c3d-ltc-fcan-comp/c3d_rgb_fcan_pool1_sz112_len16_split3_bs64_fi1_iter_20000
# FEATURE_FOLDER1=/home/tranlaman/Public/data/ucf101_features/c3d-fcan-features/c3d_fcan_pool1_learned_link_fi1_iter_20000
EXT1=rgb_fc8
WEIGHT1=1
# feature 2
FEATURE_FOLDER2=/home/tranlaman/Public/data/ucf101_features/c3d-features/ucf101_c3d_compensated_flow_split3_bs126_fi1_iter_20000
# FEATURE_FOLDER2=/home/tranlaman/Public/data/ucf101_features/c3d-features/c3d_flow_nd_conv_bs150_fi1_iter_20000
EXT2=flow_fc8
WEIGHT2=1

# video classification
python classify_video_accuracy_multi_features_ucf101.py --split=$SPLIT --feature_folders=$FEATURE_FOLDER1 --prob_extensions=$EXT1 --fusion_weights=$WEIGHT1\
 --feature_folders=$FEATURE_FOLDER2 --prob_extensions=$EXT2 --fusion_weights=$WEIGHT2

# clip classification
python classify_clip_accuracy_multi_features_ucf101.py --split=$SPLIT --feature_folders=$FEATURE_FOLDER1 --prob_extensions=$EXT1 --fusion_weights=$WEIGHT1\
 --feature_folders=$FEATURE_FOLDER2 --prob_extensions=$EXT2 --fusion_weights=$WEIGHT2
