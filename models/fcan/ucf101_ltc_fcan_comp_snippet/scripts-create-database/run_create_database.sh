#!/usr/bin/env bash
# Create the rgb flow txt inputs

SPLIT=1
LENGTH=16
DATABASE_NAME=$(printf "ucf101_tvl1_overlapping_len%d_train_test_split%d" $LENGTH $SPLIT)
RESP_FLOW_DATABASE=../../../../internal-data/video-snippets-database/$DATABASE_NAME
DATABASE_FOLDER=..

RGB_DATASET=/home/tranlaman/Public/data/video/ucf101_tvl1_128_171/img_folder/
FLOW_DATASET=/home/tranlaman/Public/data/video/ucf101_comp_tvl1_128_171/flow_folder/
bash create_ucf101_corresponding_rgb_snippet.sh $RGB_DATASET $RESP_FLOW_DATABASE $LENGTH $SPLIT $DATABASE_FOLDER
bash create_ucf101_corresponding_flow_snippet.sh $FLOW_DATASET $RESP_FLOW_DATABASE $LENGTH $SPLIT $DATABASE_FOLDER
