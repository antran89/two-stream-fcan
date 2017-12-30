#!/usr/bin/env bash

FEATURE_FOLDER=/home/tranlaman/Public/data/extracted_features/new_test_data/c3d-features/

bash classify_clip_and_video_accuracy_ucf101.sh $FEATURE_FOLDER/c3d_rgb_lmdb_fb_split1_model4_iter_10000 1 rgb_prob
bash classify_clip_and_video_accuracy_ucf101.sh $FEATURE_FOLDER/c3d_rgb_nd_conv_bs150_model1_iter_10000 1 rgb_prob
