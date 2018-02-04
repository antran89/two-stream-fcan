#!/usr/bin/env bash

SPLIT=1

# feature 1
# FEATURE_FOLDER1=/home/tranlaman/Public/data/ucf101_features/features/caffenet_rgb_bs300_split1_fi1_iter_20000/
FEATURE_FOLDER1=/home/tranlaman/Public/data/ucf101_features/fcan-features//ucf101_caffenet_fcan_comp_pool1_learned_link_bs256_fi1_iter_20000
EXT1=rgb_fc8
WEIGHT1=1
# feature 2
# FEATURE_FOLDER2=/home/tranlaman/Public/data/ucf101_features/features/caffenet_tvl1_1frame_bs300_split1_wi1_iter_20000
FEATURE_FOLDER2=/home/tranlaman/Public/data/ucf101_features/features/ucf101_caffenet_comp_flow_1frame_snippet_bs256_fi2_iter_20000/
EXT2=flow_fc8
WEIGHT2=1

# video classification
python classify_video_accuracy_multi_features_ucf101.py --split=$SPLIT --feature_folders=$FEATURE_FOLDER1 --prob_extensions=$EXT1 --fusion_weights=$WEIGHT1\
 --feature_folders=$FEATURE_FOLDER2 --prob_extensions=$EXT2 --fusion_weights=$WEIGHT2

# clip classification
python classify_clip_accuracy_multi_features_ucf101.py --split=$SPLIT --feature_folders=$FEATURE_FOLDER1 --prob_extensions=$EXT1 --fusion_weights=$WEIGHT1\
 --feature_folders=$FEATURE_FOLDER2 --prob_extensions=$EXT2 --fusion_weights=$WEIGHT2
