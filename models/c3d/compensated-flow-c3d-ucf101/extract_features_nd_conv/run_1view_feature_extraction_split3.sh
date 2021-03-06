#!/usr/bin/env bash

FEATURE_FOLDER=/home/tranlaman/Public/data/ucf101_features/c3d-features/
SPLIT=3

./ucf101_c3d_flow_nd_conv_1view_testset_split3_feature_extraction.sh 2 ../c3d_flow_train_val_lmdb_split3_bs126_fi1_iter_20000.caffemodel $FEATURE_FOLDER
sleep 2s
wait

# classification
cd ../evaluate_models
bash classify_clip_and_video_accuracy_ucf101.sh $FEATURE_FOLDER/c3d_flow_train_val_lmdb_split3_bs126_fi1_iter_20000 $SPLIT flow_fc8
bash classify_clip_and_video_accuracy_ucf101.sh $FEATURE_FOLDER/c3d_flow_train_val_lmdb_split3_bs126_fi1_iter_20000 $SPLIT flow_prob