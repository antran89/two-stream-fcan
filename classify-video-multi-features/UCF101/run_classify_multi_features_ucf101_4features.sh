#!/usr/bin/env bash

SPLIT=2

# feature 1
FEATURE_FOLDER1=/home/tranlaman/Public/data/ucf101_features/c3d-ltc-fcan-comp/c3d_rgb_fcan_pool1_sz112_len16_split2_bs64_fi1_iter_20000
EXT1=rgb_fc8
WEIGHT1=1
# feature 2
FEATURE_FOLDER2=/home/tranlaman/Public/data/ucf101_features/c3d-features/ucf101_c3d_compensated_flow_split2_bs120_fi1_iter_20000
EXT2=flow_fc8
WEIGHT2=1
# feature 3
FEATURE_FOLDER3=/home/tranlaman/Public/data/ucf101_features/tsn_bn_inception/ucf101_tsn_bn_inception_rgb_wholevideo_split2_bs128_li1_iter_3500
EXT3=rgb_fc
WEIGHT3=1
# feature 4
FEATURE_FOLDER4=/home/tranlaman/Public/data/ucf101_features/tsn_bn_inception/ucf101_tsn_bn_inception_flow_wholevideo_split2_bs128_fi1_iter_18000
EXT4=flow_fc
WEIGHT4=1

# video classification
python classify_video_accuracy_multi_features_ucf101.py --split=$SPLIT --feature_folders=$FEATURE_FOLDER1 --prob_extensions=$EXT1 --fusion_weights=$WEIGHT1\
 --feature_folders=$FEATURE_FOLDER2 --prob_extensions=$EXT2 --fusion_weights=$WEIGHT2 --feature_folders=$FEATURE_FOLDER3 --fusion_weights=$WEIGHT3\
  --prob_extensions=$EXT3 --feature_folders=$FEATURE_FOLDER4 --prob_extensions=$EXT4 --fusion_weights=$WEIGHT4

# clip classification
python classify_clip_accuracy_multi_features_ucf101.py --split=$SPLIT --feature_folders=$FEATURE_FOLDER1 --prob_extensions=$EXT1 --fusion_weights=$WEIGHT1\
 --feature_folders=$FEATURE_FOLDER2 --prob_extensions=$EXT2 --fusion_weights=$WEIGHT2 --feature_folders=$FEATURE_FOLDER3 --fusion_weights=$WEIGHT3\
  --prob_extensions=$EXT3 --feature_folders=$FEATURE_FOLDER4 --prob_extensions=$EXT4 --fusion_weights=$WEIGHT4
